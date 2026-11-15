"""
AIS-NIDS weeks 3-6: unknown detector, temperature scaling, stream rule, memory module,
incremental update (paper 4-step + KD/L2/WA extensions), validation gate, and structured experiment dumps.

Run (from the repo root, after skeleton.py has produced weights or let this script train them):
      python pipeline.py --smoke                    # tiny data, writes only to ./smoke_out
      python pipeline.py --model mlp                # loads weights/mlp_seed42.pt, trains it if missing
      python pipeline.py --model mlp --dedup --coverage 0.95 --gate-min-support 30 --who Khiem
      python pipeline.py --model mlp --dedup --sweep 0.80,0.90,0.95,0.99 --temp-scale
      python pipeline.py --model mlp --dedup --dump results/run_dedup_seed42.json --who Khiem

Episode per unknown class (order Hulk -> Bot -> slowloris, plan section 8):
    stream = BENIGN block + first 60% of the unknown class + BENIGN block
    detect_unknown -> StreamCollector (runs < 4 dropped) -> incremental_update -> gate -> commit / discard
    the last 40% of the unknown class is held out for the before/after check.
Then the benign negative test (paper Table 5): BENIGN rows posing as a new attack; the gate should reject.

Decisions taken here (also documented in results.md):
  - t is calibrated on known-class VALIDATION rows only (coverage 0.95 -> 95% of known val rows pass).
  - Optional temperature scaling (--temp-scale) fits T on known-class validation logits via NLL before calibrating t.
  - VD1 is drawn from the validation split (not the test split as in the paper), so test stays clean.
  - All runs >= min_run that come out of one episode are merged into one candidate class. Benign runs that
    were falsely flagged can therefore contaminate it; the purity number in the log measures that.
  - Gate: ACCEPT iff max per-class forgetting < 10 points AND average-accuracy drop <= 5 points
    (Algorithm 1 line 22 read as "drop of at most 5 points"). Both total drop (including the new class)
    and old-class-only drop are reported (--drop-mode chooses which one enforces the gate).
"""
import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import print_dedup_stats, resolve_payload_conflicts
import skeleton as sk

Split = Tuple[np.ndarray, np.ndarray]           # (X uint8 (n,1500), y int64 (n,))

UNKNOWN_ORDER = ["hulk", "bot", "slowloris"]    # episode order from plan section 8
BENIGN_NAME = "BENIGN"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def n_outputs(model: nn.Module) -> int:
    return model.head.out_features


def _as_uint8_batch(x) -> torch.Tensor:
    """Accept one packet (1500,) or a batch (n,1500), numpy or torch, always uint8 0..255.
    Floats are refused on purpose: sk.to_input divides by 255, so pre-scaled input would be scaled twice."""
    t = torch.as_tensor(x)
    if t.dtype != torch.uint8:
        raise TypeError(f"expected uint8 payload bytes, got {t.dtype}")
    return t.unsqueeze(0) if t.dim() == 1 else t


def fit_temperature(model: nn.Module, X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 2048) -> float:
    """Fit temperature scalar T > 0 on known validation logits by minimizing cross-entropy (NLL)."""
    model.eval()
    X_t = _as_uint8_batch(X_val)
    with torch.no_grad():
        logits_list = [model(sk.to_input(X_t[i:i + batch_size])) for i in range(0, len(X_t), batch_size)]
        logits = torch.cat(logits_list, dim=0)
    targets = torch.as_tensor(y_val, dtype=torch.long)
    T = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp_min(0.05), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp_min(0.05).item())


@torch.no_grad()
def predict_conf(model: nn.Module, X, batch_size: int = 2048, temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Max softmax probability (optionally temperature-scaled) and argmax class for every row."""
    model.eval()
    X = _as_uint8_batch(X)
    if len(X) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    confs, preds = [], []
    inv_t = 1.0 / max(float(temperature), 1e-6)
    for i in range(0, len(X), batch_size):
        logits = model(sk.to_input(X[i:i + batch_size])) * inv_t
        p = torch.softmax(logits, dim=1)
        c, k = p.max(dim=1)
        confs.append(c.numpy())
        preds.append(k.numpy())
    return np.concatenate(confs), np.concatenate(preds)


def train_epochs(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> List[float]:
    """Plain supervised training reusing skeleton's loader and epoch loop."""
    loader = sk.make_loader(X, y, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    return [sk.train_one_epoch(model, loader, opt, loss_fn) for _ in range(epochs)]


# ----------------------------------------------------------------------------
# 2. Week 3: unknown detector (softmax threshold)
# ----------------------------------------------------------------------------
def detect_unknown(model: nn.Module, x, t: float, temperature: float = 1.0):
    """Flag packets whose max softmax confidence is below t.
    Returns (conf, pred, is_unknown) as numpy arrays; `pred` is still the closest known class."""
    conf, pred = predict_conf(model, x, temperature=temperature)
    return conf, pred, conf < t


def calibrate_threshold(model: nn.Module, X_val_known: np.ndarray, coverage: float = 0.95,
                        temperature: float = 1.0) -> float:
    """t such that `coverage` of known validation packets have confidence >= t.
    Never tune t on unknown classes."""
    conf, _ = predict_conf(model, X_val_known, temperature=temperature)
    return float(np.quantile(conf, 1.0 - coverage))


def unknown_detection_report(model, t, X_known_test, y_known_test, unknown_sets: Dict[str, np.ndarray],
                             class_names: List[str], temperature: float = 1.0) -> Dict[str, float]:
    """Week 3 table: per-unknown-class flag rate, false-unknown rate (overall and BENIGN), confidence gap."""
    conf_k, _, flag_k = detect_unknown(model, X_known_test, t, temperature=temperature)
    rep = {
        "temperature": float(temperature),
        "t": float(t),
        "false_unknown_all": float(flag_k.mean()),
        "mean_conf_known": float(conf_k.mean()),
    }
    if BENIGN_NAME in class_names:
        b = y_known_test == class_names.index(BENIGN_NAME)
        rep["false_unknown_benign"] = float(flag_k[b].mean()) if b.any() else float("nan")
    unk_confs = []
    for name, Xu in unknown_sets.items():
        conf_u, _, flag_u = detect_unknown(model, Xu, t, temperature=temperature)
        rep[f"flag_rate[{name}]"] = float(flag_u.mean())
        rep[f"mean_conf[{name}]"] = float(conf_u.mean())
        unk_confs.append(conf_u)
    if unk_confs:
        all_u = np.concatenate(unk_confs)
        rep["mean_conf_unknown"] = float(all_u.mean())
        rep["conf_gap"] = float(conf_k.mean() - all_u.mean())
    return rep


# ----------------------------------------------------------------------------
# 3a. Week 4: continuous stream rule
# ----------------------------------------------------------------------------
class StreamCollector:
    """Buffers consecutive unknown packets. The counter resets whenever a packet is NOT unknown:
    a run shorter than `min_len` is discarded as bad data, a run of >= `min_len` is emitted whole.

    `item` can be the packet itself or its index in the stream (the pipeline uses indices so it can
    measure purity afterwards)."""

    def __init__(self, min_len: int = 4):
        self.min_len = min_len
        self._buf: list = []
        self.kept_runs: int = 0
        self.discarded_runs: int = 0
        self.discarded_packets: int = 0

    @property
    def current_len(self) -> int:
        return len(self._buf)

    def push(self, item, is_unknown: bool) -> Optional[list]:
        """Feed one packet. Returns a completed run (list of items) when a valid run just ended."""
        if is_unknown:
            self._buf.append(item)
            return None
        return self._close()

    def flush(self) -> Optional[list]:
        """End of stream: close whatever run is open."""
        return self._close()

    def _close(self) -> Optional[list]:
        if not self._buf:
            return None
        run, self._buf = self._buf, []
        if len(run) >= self.min_len:
            self.kept_runs += 1
            return run
        self.discarded_runs += 1
        self.discarded_packets += len(run)
        return None


def collect_unknown_runs(model, X_stream: np.ndarray, t: float, min_len: int = 4, temperature: float = 1.0):
    """Detect over an ordered stream and apply the stream rule. Returns (runs as index lists, flags, collector)."""
    _, _, flags = detect_unknown(model, X_stream, t, temperature=temperature)
    col = StreamCollector(min_len)
    runs = []
    for i, f in enumerate(flags):
        r = col.push(i, bool(f))
        if r is not None:
            runs.append(r)
    r = col.flush()
    if r is not None:
        runs.append(r)
    return runs, flags, col


# ----------------------------------------------------------------------------
# 3b. Week 4: memory module (ILD1 exemplars + VD1 validation set)
# ----------------------------------------------------------------------------
def per_class_sample(y: np.ndarray, frac: float, rng: np.random.Generator, min_per_class: int = 1) -> np.ndarray:
    """Indices of a random `frac` of every class (at least `min_per_class`, so rare classes are not lost)."""
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = min(len(idx), max(min_per_class, int(round(frac * len(idx)))))
        keep.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(keep))


@dataclass
class MemoryModule:
    ild1_X: np.ndarray
    ild1_y: np.ndarray
    vd1_X: np.ndarray
    vd1_y: np.ndarray
    ild_frac: float = 0.05
    vd_frac: float = 0.10
    ild_min_per_class: int = 30
    vd_min_per_class: int = 30
    seed: int = sk.SEED

    @classmethod
    def build(cls, train: Split, vd_source: Split, ild_frac: float = 0.05, vd_frac: float = 0.10,
              ild_min_per_class: int = 30, vd_min_per_class: int = 30, seed: int = sk.SEED) -> "MemoryModule":
        """ILD1 = max(ild_frac, ild_min_per_class) of every known class from the training split.
        VD1  = max(vd_frac, vd_min_per_class) of every known class from `vd_source` (validation split)."""
        rng = np.random.default_rng(seed)
        ti = per_class_sample(train[1], ild_frac, rng, min_per_class=ild_min_per_class)
        vi = per_class_sample(vd_source[1], vd_frac, rng, min_per_class=vd_min_per_class)
        return cls(train[0][ti].copy(), train[1][ti].copy(), vd_source[0][vi].copy(), vd_source[1][vi].copy(),
                   ild_frac, vd_frac, ild_min_per_class, vd_min_per_class, seed)

    @property
    def ild1(self) -> Split:
        return self.ild1_X, self.ild1_y

    @property
    def vd1(self) -> Split:
        return self.vd1_X, self.vd1_y

    def split_new_class(self, X_new: np.ndarray, new_label: int) -> Tuple[np.ndarray, Split]:
        """Hold out max(vd_frac, vd_min_per_class) of a collected stream (capped at half the stream) as VD1 rows."""
        rng = np.random.default_rng(self.seed + new_label)
        perm = rng.permutation(len(X_new))
        if len(X_new) >= 2:
            target = max(self.vd_min_per_class, int(round(self.vd_frac * len(X_new))))
            n_vd = max(1, min(len(X_new) // 2, target))
        else:
            n_vd = 0
        vd_idx, tr_idx = perm[:n_vd], perm[n_vd:]
        vd_new = (X_new[vd_idx], np.full(len(vd_idx), new_label, dtype=np.int64))
        return X_new[tr_idx], vd_new

    def commit_new_class(self, X_train_new: np.ndarray, vd_new: Split, new_label: int) -> None:
        """After ACCEPT: store max(ild_frac, ild_min_per_class) of the new class as exemplars and its held-out rows in VD1."""
        rng = np.random.default_rng(self.seed + 1000 + new_label)
        k = min(len(X_train_new), max(self.ild_min_per_class, int(round(self.ild_frac * len(X_train_new)))))
        pick = rng.choice(len(X_train_new), size=k, replace=False)
        self.ild1_X = np.concatenate([self.ild1_X, X_train_new[pick]])
        self.ild1_y = np.concatenate([self.ild1_y, np.full(k, new_label, dtype=np.int64)])
        self.vd1_X = np.concatenate([self.vd1_X, vd_new[0]])
        self.vd1_y = np.concatenate([self.vd1_y, vd_new[1]])

    def summary(self) -> str:
        return (f"ILD1 {len(self.ild1_y)} rows / {len(np.unique(self.ild1_y))} classes, "
                f"VD1 {len(self.vd1_y)} rows / {len(np.unique(self.vd1_y))} classes")


# ----------------------------------------------------------------------------
# 4. Week 4: incremental update (paper 3.2.4 + regularization extensions)
# ----------------------------------------------------------------------------
def extend_head(old_head: nn.Linear) -> nn.Linear:
    """Same layer with one extra output node. Old rows are copied; the new row keeps PyTorch's default init."""
    new = nn.Linear(old_head.in_features, old_head.out_features + 1, bias=old_head.bias is not None)
    new = new.to(device=old_head.weight.device, dtype=old_head.weight.dtype)
    with torch.no_grad():
        new.weight[:-1].copy_(old_head.weight)
        if old_head.bias is not None:
            new.bias[:-1].copy_(old_head.bias)
    return new


def restore_backbone(model_u: nn.Module, model_b: nn.Module) -> None:
    """Step 3: put every parameter/buffer back to model_b's value except those of `head`."""
    sd_b = model_b.state_dict()
    sd_u = model_u.state_dict()
    for k in sd_u:
        if not k.startswith("head."):
            sd_u[k] = sd_b[k].clone()
    model_u.load_state_dict(sd_u)


@dataclass
class UpdateConfig:
    perturb_epochs: int = 1          # step 2: new class only
    perturb_lr: float = 1e-3
    refine_epochs: int = 10          # step 4: new class + ILD1
    refine_lr: float = 5e-4          # joint refinement learning rate
    lambda_kd: float = 2.0           # old-logit MSE distillation weight (L_KD); 0.0 disables KD
    lambda_l2: float = 1e-3          # parameter L2 anchoring weight (L_L2); 0.0 disables L2
    wa_gamma: float = 0.83           # post-refinement Weight Aligning factor; None/0 disables WA
    new_class_cap: Optional[int] = 320   # step 4 under-sampling; None -> size of the largest ILD1 class
    balance_ild: bool = True         # oversample smaller ILD1 classes to match max_old
    use_pseudo_labels: bool = True   # use model_b argmax on ILD1 rows during refinement


def incremental_update(model_b: nn.Module, X_new: np.ndarray, ild1: Split, new_label: int,
                       cfg: UpdateConfig = UpdateConfig(), seed: int = sk.SEED) -> nn.Module:
    """theta_b -> theta_u in four steps (Algorithm 1, lines 13-17). model_b is never modified."""
    C = n_outputs(model_b)
    if new_label != C:
        raise ValueError(f"new_label must be the next free index {C}, got {new_label}")
    rng = np.random.default_rng(seed + new_label)

    # (1) Initialize: copy of theta_b with one extra output node.
    model_u = copy.deepcopy(model_b)
    model_u.head = extend_head(model_u.head)

    # (2) Learn the new class on its own samples only (this disturbs the backbone heavily).
    y_new = np.full(len(X_new), new_label, dtype=np.int64)
    train_epochs(model_u, X_new, y_new, cfg.perturb_epochs, cfg.perturb_lr)

    # (3) Restore all weights to theta_b except the new class output row, and align new-row norm (WA).
    restore_backbone(model_u, model_b)
    if cfg.wa_gamma is not None and cfg.wa_gamma > 0:
        with torch.no_grad():
            model_u.head.weight[:C].copy_(model_b.head.weight)
            if model_u.head.bias is not None and model_b.head.bias is not None:
                model_u.head.bias[:C].copy_(model_b.head.bias)
            old_norm = model_b.head.weight.norm(dim=1).mean()
            new_norm = model_u.head.weight[C].norm().clamp_min(1e-6)
            model_u.head.weight[C].mul_(old_norm / new_norm)
            if model_u.head.bias is not None and model_b.head.bias is not None:
                model_u.head.bias[C].copy_(model_b.head.bias.mean())

    # (4) Iterative refinement on (under-sampled) new class + ILD1 exemplars
    #     with L_total = L_CE + lambda_kd * L_KD + lambda_l2 * L_L2.
    max_old = int(np.bincount(ild1[1]).max())
    if cfg.balance_ild:
        X_old_bal, y_old_bal = [], []
        for c in np.unique(ild1[1]):
            idx_c = np.flatnonzero(ild1[1] == c)
            if len(idx_c) < max_old:
                pick_c = np.concatenate([idx_c, rng.choice(idx_c, size=max_old - len(idx_c), replace=True)])
            else:
                pick_c = idx_c
            X_old_bal.append(ild1[0][pick_c])
            y_old_bal.append(ild1[1][pick_c])
        X_old_bal = np.concatenate(X_old_bal)
        y_old_bal = np.concatenate(y_old_bal)
    else:
        X_old_bal, y_old_bal = ild1[0], ild1[1]

    cap = cfg.new_class_cap if cfg.new_class_cap is not None else len(X_new)
    sel = rng.choice(len(X_new), size=min(cap, len(X_new)), replace=False)
    X_ref = np.concatenate([X_new[sel], X_old_bal])
    y_ref = np.concatenate([y_new[sel], y_old_bal])

    loader = sk.make_loader(X_ref, y_ref, shuffle=True)
    opt = torch.optim.Adam(model_u.parameters(), lr=cfg.refine_lr)

    model_b.eval()
    model_u.train()
    for _ in range(cfg.refine_epochs):
        for xb, yb in loader:
            inp = sk.to_input(xb)
            logits_u = model_u(inp)
            with torch.no_grad():
                logits_b = model_b(inp)
                yb_eff = yb.clone()
                old_mask = yb < C
                if cfg.use_pseudo_labels:
                    yb_eff[old_mask] = logits_b[old_mask].argmax(dim=1)

            loss_ce = F.cross_entropy(logits_u, yb_eff)
            if cfg.lambda_kd > 0 and old_mask.any():
                loss_kd = F.mse_loss(logits_u[old_mask, :C], logits_b[old_mask])
            else:
                loss_kd = torch.tensor(0.0, device=logits_u.device)

            if cfg.lambda_l2 > 0:
                loss_l2_bb = sum(
                    (pu - pb.detach()).pow(2).sum()
                    for pu, pb in zip(model_u.backbone.parameters(), model_b.backbone.parameters())
                )
                loss_l2_head = (model_u.head.weight[:C] - model_b.head.weight.detach()).pow(2).sum()
                if model_u.head.bias is not None and model_b.head.bias is not None:
                    loss_l2_head = loss_l2_head + (model_u.head.bias[:C] - model_b.head.bias.detach()).pow(2).sum()
                loss_l2 = loss_l2_bb + loss_l2_head
            else:
                loss_l2 = torch.tensor(0.0, device=logits_u.device)

            loss = loss_ce + cfg.lambda_kd * loss_kd + cfg.lambda_l2 * loss_l2
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Post-refinement Weight Aligning (Zhao et al., CVPR 2020)
    if cfg.wa_gamma is not None and cfg.wa_gamma > 0:
        with torch.no_grad():
            old_norm = model_u.head.weight[:C].norm(dim=1).mean()
            new_norm = model_u.head.weight[C].norm().clamp_min(1e-6)
            model_u.head.weight[C].mul_(cfg.wa_gamma * (old_norm / new_norm))

    return model_u


# ----------------------------------------------------------------------------
# 5. Week 5: validation gate (paper 3.2.5)
# ----------------------------------------------------------------------------
def per_class_accuracy(model: nn.Module, vd: Split) -> Dict[int, float]:
    """Recall per class, for classes present in vd that the model can output."""
    X, y = vd
    _, pred = predict_conf(model, X)
    return {int(c): float((pred[y == c] == c).mean()) for c in np.unique(y) if c < n_outputs(model)}


def evaluate_forgetting(model_old: nn.Module, model_new: nn.Module, vd1: Split) -> Dict[int, float]:
    """Forgetting rate per OLD class = acc(current model) - acc(updated model), measured on VD1."""
    acc_old = per_class_accuracy(model_old, vd1)
    acc_new = per_class_accuracy(model_new, vd1)
    return {c: acc_old[c] - acc_new[c] for c in acc_old}


def evaluate_avg_accuracy(model: nn.Module, vd1: Split) -> float:
    """Average incremental accuracy: mean per-class accuracy over all classes the model has learned."""
    acc = per_class_accuracy(model, vd1)
    return float(np.mean(list(acc.values())))


@dataclass
class GateResult:
    accept: bool
    forgetting: Dict[int, float]
    max_forgetting: float
    worst_class: Optional[int]
    avg_old: float
    avg_new: float
    avg_drop: float
    old_only_drop: float = 0.0
    new_class_acc: Optional[float] = None
    acc_old_per_class: Dict[int, float] = field(default_factory=dict)
    acc_new_per_class: Dict[int, float] = field(default_factory=dict)
    support: Dict[int, int] = field(default_factory=dict)
    ignored: List[int] = field(default_factory=list)
    reason: str = ""


def gate(model_old: nn.Module, model_new: nn.Module, vd1: Split, vd_new: Optional[Split] = None,
         max_forgetting: float = 0.10, max_avg_drop: float = 0.05, min_support: int = 20,
         drop_mode: str = "all") -> GateResult:
    """ACCEPT iff max per-class forgetting (over classes with >= min_support VD1 rows) < max_forgetting
    AND avg-accuracy drop <= max_avg_drop.
    `drop_mode`: 'all' includes the new class in avg_new (paper definition); 'old' computes drop over old classes only."""
    fr = evaluate_forgetting(model_old, model_new, vd1)
    support = {int(c): int((vd1[1] == c).sum()) for c in np.unique(vd1[1])}
    eligible = {c: f for c, f in fr.items() if support.get(c, 0) >= min_support}
    ignored = sorted(set(fr) - set(eligible))

    acc_old_map = per_class_accuracy(model_old, vd1)
    acc_new_old_map = per_class_accuracy(model_new, vd1)
    if eligible:
        avg_old = float(np.mean([acc_old_map[c] for c in eligible]))
        old_only_new = float(np.mean([acc_new_old_map[c] for c in eligible]))
        elig_new_vals = [acc_new_old_map[c] for c in eligible]
    else:
        vd_all = vd1 if vd_new is None else (np.concatenate([vd1[0], vd_new[0]]), np.concatenate([vd1[1], vd_new[1]]))
        avg_old = evaluate_avg_accuracy(model_old, vd1)
        old_only_new = evaluate_avg_accuracy(model_new, vd1)
        elig_new_vals = [evaluate_avg_accuracy(model_new, vd_all)]

    new_acc = None
    acc_new_full = dict(acc_new_old_map)
    if vd_new is not None and len(vd_new[1]):
        new_c = int(vd_new[1][0])
        new_acc = per_class_accuracy(model_new, vd_new).get(new_c)
        if new_acc is not None:
            acc_new_full[new_c] = new_acc
        if eligible and new_acc is not None and len(vd_new[1]) >= min_support:
            elig_new_vals.append(new_acc)
    avg_new = float(np.mean(elig_new_vals)) if elig_new_vals else 0.0

    if not eligible:
        worst, max_fr = None, float("inf")
    else:
        worst = max(eligible, key=eligible.get)
        max_fr = eligible[worst]
    drop = avg_old - avg_new
    old_only_drop = avg_old - old_only_new
    eff_drop = old_only_drop if drop_mode == "old" else drop

    ok_fr = max_fr < max_forgetting
    ok_drop = eff_drop <= max_avg_drop
    accept = bool(ok_fr and ok_drop)
    reasons = []
    if not ok_fr:
        reasons.append(f"maxFR {max_fr:.3f} >= {max_forgetting:.2f}")
    if not ok_drop:
        reasons.append(f"drop({drop_mode}) {eff_drop:.3f} > {max_avg_drop:.2f}")
    reason = "pass (both maxFR and drop within thresholds)" if accept else "reject: " + ", ".join(reasons)

    return GateResult(
        accept=accept, forgetting=fr, max_forgetting=max_fr, worst_class=worst,
        avg_old=avg_old, avg_new=avg_new, avg_drop=drop, old_only_drop=old_only_drop,
        new_class_acc=new_acc, acc_old_per_class=acc_old_map, acc_new_per_class=acc_new_full,
        support=support, ignored=ignored, reason=reason
    )


def print_gate(res: GateResult, class_names: List[str]) -> None:
    worst_name = class_names[res.worst_class] if res.worst_class is not None else "none"
    print(f"  gate: {'ACCEPT' if res.accept else 'REJECT'} ({res.reason}) | max forgetting {res.max_forgetting:+.3f} "
          f"on {worst_name} | avg acc {res.avg_old:.3f} -> {res.avg_new:.3f} "
          f"(drop {res.avg_drop:+.3f}, old-only drop {res.old_only_drop:+.3f})"
          + (f" | new class acc {res.new_class_acc:.3f}" if res.new_class_acc is not None else ""))
    for c, f in sorted(res.forgetting.items()):
        n = res.support.get(c, 0)
        tag = f" (ignored, n={n})" if c in res.ignored else f" (n={n})"
        print(f"    {class_names[c]:<30s} acc {res.acc_old_per_class.get(c, 0):.3f} -> "
              f"{res.acc_new_per_class.get(c, 0):.3f} | forgetting {f:+.3f}{tag}")


# ----------------------------------------------------------------------------
# Orchestration: one episode = detect -> stream rule -> update -> gate -> commit/discard
# ----------------------------------------------------------------------------
@dataclass
class EpisodeLog:
    name: str
    true_attack: str
    n_stream: int
    n_flagged: int
    kept_runs: int
    discarded_runs: int
    discarded_packets: int
    n_collected: int
    purity: Optional[float] = None
    gate: Optional[GateResult] = None
    accepted: bool = False
    heldout_before_unknown_rate: float = float("nan")
    heldout_after_unknown_rate: float = float("nan")
    heldout_routed_rate: float = 0.0
    notes: List[str] = field(default_factory=list)


def learn_candidate(model, memory: MemoryModule, X_new: np.ndarray, class_names: List[str], cfg: UpdateConfig,
                    name: str, gate_min_support: int = 20, drop_mode: str = "all"):
    """Update + gate + commit for an already collected candidate class. Returns (model, gate result, accepted)."""
    new_label = n_outputs(model)
    X_tr, vd_new = memory.split_new_class(X_new, new_label)
    model_u = incremental_update(model, X_tr, memory.ild1, new_label, cfg, seed=memory.seed)
    res = gate(model, model_u, memory.vd1, vd_new, min_support=gate_min_support, drop_mode=drop_mode)
    print_gate(res, class_names + [name])
    if res.accept:
        memory.commit_new_class(X_tr, vd_new, new_label)
        class_names.append(name)
        return model_u, res, True
    return model, res, False         # discard theta_u and the collected samples


def run_episode(model, memory: MemoryModule, X_stream: np.ndarray, t: float, class_names: List[str],
                cfg: UpdateConfig, min_run: int = 4, true_names: Optional[np.ndarray] = None,
                gate_min_support: int = 20, temperature: float = 1.0, drop_mode: str = "all",
                true_attack: str = ""):
    """class_names is extended in place when an update is accepted."""
    name = f"Unknown-{sum(n.startswith('Unknown-') for n in class_names) + 1}"
    runs, flags, col = collect_unknown_runs(model, X_stream, t, min_run, temperature=temperature)
    idx = np.array([i for r in runs for i in r], dtype=np.int64)
    log = EpisodeLog(name, true_attack, len(X_stream), int(flags.sum()), col.kept_runs,
                     col.discarded_runs, col.discarded_packets, len(idx))
    print(f"[{name}] stream {len(X_stream)} | flagged {log.n_flagged} | runs kept {col.kept_runs}, "
          f"discarded {col.discarded_runs} ({col.discarded_packets} pkts) | collected {len(idx)}")
    if true_names is not None and len(idx):
        vals, counts = np.unique(true_names[idx], return_counts=True)
        log.purity = float(counts.max() / counts.sum())
        print(f"  purity {log.purity:.3f} (majority: {vals[counts.argmax()]})")
    if len(idx) < 2 * min_run:
        log.notes.append("too few collected packets, nothing learned")
        print("  too few collected packets, nothing learned")
        return model, log
    model, log.gate, log.accepted = learn_candidate(
        model, memory, X_stream[idx], class_names, cfg, name,
        gate_min_support=gate_min_support, drop_mode=drop_mode
    )
    return model, log


def benign_negative_test(model, memory: MemoryModule, X_benign: np.ndarray, class_names: List[str],
                         cfg: UpdateConfig, gate_min_support: int = 20, drop_mode: str = "all") -> GateResult:
    """Feed BENIGN rows as if they were a new attack. Expected: REJECT. Memory and model are never changed."""
    print("[benign negative test]")
    mem = copy.deepcopy(memory)
    names = list(class_names)
    _, res, _ = learn_candidate(model, mem, X_benign, names, cfg, "Fake-BENIGN",
                                gate_min_support=gate_min_support, drop_mode=drop_mode)
    return res


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--weights", default=None, help="checkpoint to load; default <out>/weights/<model>_seed<seed>[_dedup].pt")
    ap.add_argument("--seed", type=int, default=sk.SEED)
    ap.add_argument("--coverage", type=float, default=0.95, help="share of known val rows above t")
    ap.add_argument("--sweep", default=None, help="comma-separated coverages to sweep in week 3 (e.g. 0.80,0.90,0.95,0.99)")
    ap.add_argument("--temp-scale", action="store_true", help="fit temperature T on known validation logits via NLL")
    ap.add_argument("--min-run", type=int, default=4)
    ap.add_argument("--stream-frac", type=float, default=0.60, help="share of each unknown class used as stream")
    ap.add_argument("--benign-block", type=int, default=300, help="BENIGN rows before and after each attack block")
    ap.add_argument("--dedup", action="store_true", help="C1 offline strict-majority payload conflict cleaning")
    ap.add_argument("--vd-min", type=int, default=30, help="minimum VD1 rows per class")
    ap.add_argument("--gate-min-support", type=int, default=30, help="minimum VD1 support for a class to decide gate")
    ap.add_argument("--drop-mode", choices=["all", "old"], default="all",
                    help="'all' includes new class in avg_new; 'old' computes drop over old classes only")
    ap.add_argument("--wa-gamma", type=float, default=0.83)
    ap.add_argument("--new-class-cap", type=int, default=320)
    ap.add_argument("--dump", default=None, help="optional path to write full JSON experiment dump")
    ap.add_argument("--no-log", action="store_true", help="do not append to results.md")
    ap.add_argument("--who", default="unknown")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = sk.SMOKE_DIR if args.smoke else "."
    os.makedirs(out, exist_ok=True)
    sk.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    suffix = "_dedup" if args.dedup else ""

    # ---- data (same path as skeleton.py, plus optional C1 dedup) ----
    X, labels = sk.load_sample()
    if args.smoke:
        X, labels = sk.smoke_subsample(X, labels, seed=args.seed)
    dedup_stats = None
    if args.dedup:
        X, labels, dedup_stats = resolve_payload_conflicts(X, labels)
        print_dedup_stats(dedup_stats)
    (Xk, lk), (Xu, lu) = sk.split_known_unknown(X, labels)
    label_map = sk.build_label_map(lk, os.path.join(out, sk.LABEL_MAP_PATH))
    class_names = sk.class_names(label_map)
    yk = np.array([label_map[l] for l in lk], dtype=np.int64)
    sp_path = os.path.join(out, f"splits_seed{args.seed}{suffix}.npz")
    try:
        sp = sk.load_or_make_splits(yk, lk, args.seed, sp_path, reuse=not args.smoke)
    except RuntimeError:
        sp = sk.load_or_make_splits(yk, lk, args.seed, sp_path, reuse=False)
    tr, va, te = sp["train"], sp["val"], sp["test"]

    # ---- closed-set model ----
    tag = f"{args.model}_seed{args.seed}{suffix}"
    ckpt = args.weights or os.path.join(out, sk.WEIGHTS_DIR, f"{tag}.pt")
    model = sk.build_model(args.model, len(class_names))
    need_train = not os.path.exists(ckpt)
    if not need_train and args.dedup and os.path.exists(sp_path):
        if os.path.getmtime(sp_path) > os.path.getmtime(ckpt):
            need_train = True
    if not need_train:
        model.load_state_dict(torch.load(ckpt))
        print(f"loaded {ckpt}")
    else:
        print(f"{ckpt} not found or outdated -> training")
        epochs = sk.SMOKE_EPOCHS if args.smoke else sk.EPOCHS
        sk.fit(model, sk.make_loader(Xk[tr], yk[tr], True), sk.make_loader(Xk[va], yk[va], False), tag, epochs,
               weights_dir=os.path.join(out, sk.WEIGHTS_DIR))

    # ---- week 3: optional temperature scaling + threshold + detection report ----
    unknown_sets = {}
    for key in UNKNOWN_ORDER:
        name = next(n for n in sorted(set(lu)) if key in n.lower())
        unknown_sets[name] = Xu[lu == name]

    temperature = fit_temperature(model, Xk[va], yk[va]) if args.temp_scale else 1.0
    if args.temp_scale:
        print(f"fitted temperature on validation set: T = {temperature:.4f}")

    sweep_reports = {}
    if args.sweep:
        for cov_str in args.sweep.split(","):
            cov = float(cov_str.strip())
            t_cov = calibrate_threshold(model, Xk[va], cov, temperature=temperature)
            sweep_reports[str(cov)] = unknown_detection_report(
                model, t_cov, Xk[te], yk[te], unknown_sets, class_names, temperature=temperature
            )
        print("week 3 sweep:", json.dumps(sweep_reports, indent=2))

    t = calibrate_threshold(model, Xk[va], args.coverage, temperature=temperature)
    rep = unknown_detection_report(model, t, Xk[te], yk[te], unknown_sets, class_names, temperature=temperature)
    print("week 3:", json.dumps(rep, indent=2))

    # ---- week 4/5: memory, episodes, gate ----
    memory = MemoryModule.build((Xk[tr], yk[tr]), (Xk[va], yk[va]), vd_min_per_class=args.vd_min, seed=args.seed)
    print("memory:", memory.summary())
    benign_te = te[yk[te] == label_map[BENIGN_NAME]]
    cfg = UpdateConfig(wa_gamma=args.wa_gamma, new_class_cap=args.new_class_cap)
    logs: List[EpisodeLog] = []
    stage_accuracies: List[Dict[str, float]] = []

    # Stage 0: initial known classes on VD1
    init_acc = per_class_accuracy(model, memory.vd1)
    stage_accuracies.append({
        "stage": "Initial (11 classes)",
        "avg_acc": float(np.mean(list(init_acc.values()))),
        "per_class": {class_names[c]: v for c, v in init_acc.items()},
    })

    for name, Xc in unknown_sets.items():
        n_stream = int(round(args.stream_frac * len(Xc)))
        X_attack, X_hold = Xc[:n_stream], Xc[n_stream:]            # keep file order: attacks arrive contiguously
        b1, b2 = (rng.choice(benign_te, size=min(args.benign_block, len(benign_te)), replace=False) for _ in range(2))
        X_stream = np.concatenate([Xk[b1], X_attack, Xk[b2]])
        truth = np.array([BENIGN_NAME] * len(b1) + [name] * len(X_attack) + [BENIGN_NAME] * len(b2))

        _, _, before = detect_unknown(model, X_hold, t, temperature=temperature)
        model, log = run_episode(
            model, memory, X_stream, t, class_names, cfg, args.min_run, truth,
            gate_min_support=args.gate_min_support, temperature=temperature,
            drop_mode=args.drop_mode, true_attack=name
        )
        conf, pred, after_unknown = detect_unknown(model, X_hold, t, temperature=temperature)
        log.heldout_before_unknown_rate = float(before.mean()) if len(before) else float("nan")
        log.heldout_after_unknown_rate = float(after_unknown.mean()) if len(after_unknown) else float("nan")
        log.heldout_routed_rate = float((pred == n_outputs(model) - 1).mean()) if (log.accepted and len(pred)) else 0.0
        print(f"  held-out {name}: flagged unknown before {log.heldout_before_unknown_rate:.3f}, "
              f"after {log.heldout_after_unknown_rate:.3f}, "
              f"routed to {log.name if log.accepted else '-'} {log.heldout_routed_rate:.3f}")
        logs.append(log)

        cur_acc = per_class_accuracy(model, memory.vd1)
        stage_accuracies.append({
            "stage": f"After {name} ({log.name if log.accepted else 'rejected'})",
            "accepted": log.accepted,
            "avg_acc": float(np.mean(list(cur_acc.values()))),
            "per_class": {class_names[c]: v for c, v in cur_acc.items()},
        })

    # random BENIGN test rows (preferably ones not used in the last stream) posing as a new attack
    rest = np.setdiff1d(benign_te, np.concatenate([b1, b2]))
    pool = rest if len(rest) >= 2 * args.min_run else benign_te
    fake = rng.choice(pool, size=min(1000, len(pool)), replace=False)
    neg = benign_negative_test(model, memory, Xk[fake], class_names, cfg,
                               gate_min_support=args.gate_min_support, drop_mode=args.drop_mode)
    print(f"benign negative test -> {'ACCEPT (gate too loose!)' if neg.accept else 'REJECT (expected)'}")

    final_map = {n: i for i, n in enumerate(class_names)}
    with open(os.path.join(out, f"label_map_incremental{suffix}.json"), "w") as f:
        json.dump({"mapping": final_map, "true_source": {l.name: l.true_attack for l in logs if l.accepted}}, f, indent=2)

    if args.dump:
        os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
        dump_data = {
            "args": vars(args),
            "temperature": temperature,
            "threshold": t,
            "week3_report": rep,
            "week3_sweep": sweep_reports,
            "episodes": [asdict(l) for l in logs],
            "stage_accuracies": stage_accuracies,
            "benign_negative": asdict(neg),
            "final_class_names": class_names,
        }
        with open(args.dump, "w") as f:
            json.dump(dump_data, f, indent=2)
        print(f"saved experiment dump to {args.dump}")

    if not args.smoke and not args.no_log:
        ep_parts = []
        for l in logs:
            if l.gate:
                worst_n = class_names[l.gate.worst_class] if l.gate.worst_class is not None else "none"
                ep_parts.append(
                    f"{l.true_attack}->{l.name if l.accepted else 'rejected'} "
                    f"(purity {l.purity:.3f}, maxFR {l.gate.max_forgetting:.3f} on {worst_n}, "
                    f"drop {l.gate.avg_drop:.3f}, oldDrop {l.gate.old_only_drop:.3f}, "
                    f"holdout unk {l.heldout_before_unknown_rate:.3f}->{l.heldout_after_unknown_rate:.3f}, "
                    f"routed {l.heldout_routed_rate:.3f})"
                )
            else:
                ep_parts.append(f"{l.true_attack}: no update")
        neg_worst = class_names[neg.worst_class] if neg.worst_class is not None else "none"
        neg_str = (f"benign negative {'ACCEPT' if neg.accept else 'REJECT'} "
                   f"(maxFR {neg.max_forgetting:.3f} on {neg_worst}, drop {neg.avg_drop:.3f}, oldDrop {neg.old_only_drop:.3f})")
        sk.log_result(
            args.who,
            f"[W3-W6] pipeline {args.model}{suffix}, dedup={args.dedup}, seed={args.seed}, vd_min={args.vd_min}, "
            f"gate_min_support={args.gate_min_support}, drop_mode={args.drop_mode}, T={temperature:.3f}, "
            f"t={t:.4f} (cov {args.coverage}), min_run {args.min_run}, {cfg}",
            "; ".join(ep_parts + [neg_str])
        )


if __name__ == "__main__":
    main()
