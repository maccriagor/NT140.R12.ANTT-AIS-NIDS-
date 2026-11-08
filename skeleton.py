"""
Shared skeleton for the AIS-NIDS prototype (weeks 1-2).

Run:  python skeleton.py --model mlp
      python skeleton.py --model cnn1d
      python skeleton.py --model mlp --smoke      # ~1,100 rows, 2 epochs, writes only to ./smoke_out
      python skeleton.py --model mlp --seed 7     # other seeds only for robustness runs; say so in results.md
(--model takes any name registered under models/; run from the repo root.)

Contract every model must follow (see plan, section 2):
  - input  : float tensor (batch, 1500), values 0..1
  - output : logits (batch, n_classes)
  - last layer is a single nn.Linear stored as `model.head`
  - n_classes comes from the data, never typed by hand
  - models are created only through build_model(name, n_classes)

Models live in models/<name>.py, written and owned by their authors, never in this file. A model file looks like:
    from models.registry import register, N_BYTES
    @register("mlp")
    class MLP(nn.Module):
        def __init__(self, n_classes): ...      # must define self.head = nn.Linear(...)
models/__init__.py imports every file in models/, so adding a model needs no edit here.

Shared artifacts (commit the small ones to git so everyone uses the same files):
  label_map.json            name <-> int for the known classes
  splits_seed<seed>.npz     train/val/test indices (known rows only) + a fingerprint of the sample
"""
import argparse
import hashlib
import json
import os
import random
import time
from datetime import date
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Importing models.registry first runs models/__init__.py, which imports every model file so each one registers.
from models.registry import MODELS, N_BYTES

# ----------------------------------------------------------------------------
# 0. Config
# ----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
META_COLS = ["ttl", "total_len", "protocol", "t_delta", "label"]
UNKNOWN_KEYS = ["hulk", "slowloris", "bot"]      # held-out classes (matched case-insensitively)
DROP_KEYS = ["sql injection"]                    # only 12 rows -> dropped, list as a limitation
HOLDOUT_FRAC = 0.30                              # val+test share, split evenly -> 70/15/15
SAMPLE_PATH = "capped_sample.pkl"
LABEL_MAP_PATH = "label_map.json"
WEIGHTS_DIR = "weights"
RESULTS_PATH = "results.md"
SMOKE_PER_CLASS = 80                             # rows per class in --smoke mode
SMOKE_EPOCHS = 2
SMOKE_DIR = "smoke_out"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------
# 1. Data (stream A)
# ----------------------------------------------------------------------------
def _matches(label: str, keys) -> bool:
    low = label.lower()
    return any(k in low for k in keys)


def load_sample(path: str = SAMPLE_PATH):
    """Return X (N,1500) uint8 and labels (N,) as strings. Extra columns are ignored for now."""
    df = pd.read_pickle(path)
    payload_cols = [c for c in df.columns if c not in META_COLS]
    assert len(payload_cols) == N_BYTES, f"expected {N_BYTES} payload columns, got {len(payload_cols)}"
    X = np.ascontiguousarray(df[payload_cols].to_numpy(dtype=np.uint8))
    labels = df["label"].astype(str).str.strip().to_numpy()
    return X, labels


def smoke_subsample(X, labels, per_class: int = SMOKE_PER_CLASS, seed: int = SEED):
    """Tiny random sample (up to `per_class` rows of every label) for fast code-change tests."""
    rng = np.random.default_rng(seed)
    keep = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        keep.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], labels[keep]


def split_known_unknown(X, labels):
    """Drop tiny classes, then separate held-out unknown classes from known ones."""
    drop = np.array([_matches(l, DROP_KEYS) for l in labels], dtype=bool)
    if drop.any():
        print("dropped:", {n: int((labels[drop] == n).sum()) for n in sorted(set(labels[drop]))})
    X, labels = X[~drop], labels[~drop]

    is_unknown = np.array([_matches(l, UNKNOWN_KEYS) for l in labels], dtype=bool)
    Xk, lk = X[~is_unknown], labels[~is_unknown]
    Xu, lu = X[is_unknown], labels[is_unknown]

    assert len(lu) > 0, "no unknown rows found - check UNKNOWN_KEYS against the real label spelling"
    for k in UNKNOWN_KEYS:
        assert any(k in l.lower() for l in set(lu)), f"UNKNOWN_KEYS entry '{k}' matched no label"
    print("unknown labels:", {n: int((lu == n).sum()) for n in sorted(set(lu))})
    return (Xk, lk), (Xu, lu)


def build_label_map(known_labels, path: str = LABEL_MAP_PATH) -> dict:
    """name -> int, built from the data and saved so every script reuses it."""
    names = sorted(set(known_labels))
    mapping = {name: i for i, name in enumerate(names)}
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    return mapping


def load_label_map(path: str = LABEL_MAP_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def class_names(label_map: dict) -> list:
    return [n for n, _ in sorted(label_map.items(), key=lambda kv: kv[1])]


def make_splits(y: np.ndarray, seed: int = SEED):
    """Stratified 70/15/15 -> index arrays (train, val, test) into the known rows."""
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=HOLDOUT_FRAC, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    return tr, va, te


def fingerprint(lk, seed: int) -> str:
    """Hash of the known-row label sequence, the split fraction and the seed. A teammate with a different
    capped_sample.pkl or row order would get indices pointing at different rows, and a changed fraction or
    seed would leave an old file silently in use; in every case the load fails loudly instead."""
    h = hashlib.sha1()
    h.update("\n".join(lk).encode())
    h.update(f"|holdout={HOLDOUT_FRAC}|seed={seed}".encode())
    return h.hexdigest()[:16]


def load_or_make_splits(yk, lk, seed: int, path: str, reuse: bool = True) -> dict:
    """Reuse the saved split file if present (and compatible); otherwise create and save it."""
    fp = fingerprint(lk, seed)
    if reuse and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if str(z["fingerprint"]) != fp:
            raise RuntimeError(
                f"{path} does not match this sample/seed/split fraction (fingerprint {str(z['fingerprint'])} vs {fp}). "
                "Make sure everyone has the same capped_sample.pkl, or delete the split file to regenerate."
            )
        print(f"loaded splits from {path}")
        return {k: z[k] for k in ("train", "val", "test")}
    tr, va, te = make_splits(yk, seed)
    np.savez(path, train=tr, val=va, test=te, fingerprint=np.array(fp))
    print(f"saved splits to {path}")
    return {"train": tr, "val": va, "test": te}


def make_loader(X_uint8: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    """Keep X as uint8 in memory (about 4x smaller); scaling to 0..1 happens per batch in to_input()."""
    ds = TensorDataset(torch.from_numpy(X_uint8), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def to_input(xb: torch.Tensor) -> torch.Tensor:
    """uint8 (batch,1500) -> float32 in [0,1]. Every model sees exactly this."""
    return xb.float() / 255.0


# ----------------------------------------------------------------------------
# 2. Model contract (stream B writes the models themselves, in models/)
# ----------------------------------------------------------------------------
def build_model(name: str, n_classes: int) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'; registered: {sorted(MODELS)}")
    model = MODELS[name](n_classes)
    check_interface(model, n_classes)
    return model


def check_interface(model: nn.Module, n_classes: int) -> None:
    """Fails loudly if a model breaks the shared contract."""
    assert isinstance(getattr(model, "head", None), nn.Linear), "model.head must be an nn.Linear"
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(2, N_BYTES))
    assert out.shape == (2, n_classes), f"expected (2,{n_classes}), got {tuple(out.shape)}"


# ----------------------------------------------------------------------------
# 3. Training and evaluation (shared)
# ----------------------------------------------------------------------------
class Pred(NamedTuple):
    y_true: np.ndarray
    y_pred: np.ndarray
    conf: np.ndarray   # max softmax probability; reused by the unknown detector in week 3
    loss: float        # mean cross-entropy, or nan when with_loss=False


def train_one_epoch(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(to_input(xb)), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(yb)
        n += len(yb)
    return total / n


@torch.no_grad()
def predict(model, loader, with_loss: bool = True) -> Pred:
    """Run the model over a loader. Pass with_loss=False for unknown-class rows: their labels are not
    valid class indices, so a loss would be meaningless (and cross_entropy would raise)."""
    model.eval()
    ys, preds, confs = [], [], []
    loss_sum, n = 0.0, 0
    for xb, yb in loader:
        logits = model(to_input(xb))
        if with_loss:
            loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        conf, pred = torch.softmax(logits, dim=1).max(dim=1)
        ys.append(yb.numpy()); preds.append(pred.numpy()); confs.append(conf.numpy())
        n += len(yb)
    loss = loss_sum / n if with_loss else float("nan")
    return Pred(np.concatenate(ys), np.concatenate(preds), np.concatenate(confs), loss)


def fit(model, train_loader, val_loader, tag: str, epochs: int = EPOCHS, weights_dir: str = WEIGHTS_DIR):
    os.makedirs(weights_dir, exist_ok=True)
    ckpt = os.path.join(weights_dir, f"{tag}.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, epoch_times = -1.0, []
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        dt = time.perf_counter() - t0
        vp = predict(model, val_loader)
        acc = float((vp.y_true == vp.y_pred).mean())
        epoch_times.append(dt)
        print(f"[{tag}] epoch {ep:02d}  train_loss {train_loss:.4f}  val_loss {vp.loss:.4f}  "
              f"val_acc {acc:.4f}  time {dt:.1f}s")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt))  # back to the best epoch
    return best_acc, float(np.mean(epoch_times))


def evaluate(model, loader, label_map: dict) -> dict:
    """Per-class report + macro-F1 + confusion matrix + loss."""
    names = class_names(label_map)
    labels = list(range(len(names)))
    p = predict(model, loader)
    return {
        "report": classification_report(p.y_true, p.y_pred, labels=labels,
                                        target_names=names, digits=3, zero_division=0),
        "macro_f1": f1_score(p.y_true, p.y_pred, average="macro", zero_division=0),
        "accuracy": float((p.y_true == p.y_pred).mean()),
        "loss": p.loss,
        "confusion": confusion_matrix(p.y_true, p.y_pred, labels=labels),
    }


def save_confusion(conf: np.ndarray, label_map: dict, path: str) -> None:
    """CSV with true classes as rows, predicted classes as columns."""
    names = class_names(label_map)
    pd.DataFrame(conf, index=names, columns=names).to_csv(path, index_label="true\\pred")


def log_result(who: str, setting: str, numbers: str, path: str = RESULTS_PATH) -> None:
    """Every experiment appends one entry here: date, who, setting, numbers."""
    with open(path, "a") as f:
        f.write(f"\n- {date.today()} | {who} | {setting} | {numbers}\n")


# ----------------------------------------------------------------------------
# 4. Entry point
# ----------------------------------------------------------------------------
def main():
    if not MODELS:
        raise SystemExit("no models registered - add a file under models/ that uses @register(<name>)")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--who", default="unknown")
    ap.add_argument("--smoke", action="store_true",
                    help=f"~{SMOKE_PER_CLASS} rows per class, {SMOKE_EPOCHS} epochs, outputs go to ./{SMOKE_DIR}")
    args = ap.parse_args()

    out = SMOKE_DIR if args.smoke else "."
    os.makedirs(out, exist_ok=True)
    epochs = min(args.epochs, SMOKE_EPOCHS) if args.smoke else args.epochs
    tag = f"{args.model}_seed{args.seed}"

    set_seed(args.seed)
    X, labels = load_sample()
    if args.smoke:
        X, labels = smoke_subsample(X, labels, seed=args.seed)
        print(f"SMOKE MODE: {len(labels)} rows")
    (Xk, lk), (Xu, lu) = split_known_unknown(X, labels)

    label_map = build_label_map(lk, os.path.join(out, LABEL_MAP_PATH))
    n_classes = len(label_map)
    yk = np.array([label_map[l] for l in lk])
    print(f"known rows {len(yk)}, unknown rows {len(lu)}, n_classes {n_classes}")
    for name, i in label_map.items():
        print(f"  {i:2d} {name}: {(yk == i).sum()}")

    sp = load_or_make_splits(yk, lk, args.seed,
                             os.path.join(out, f"splits_seed{args.seed}.npz"), reuse=not args.smoke)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"known split: train {len(tr)}, val {len(va)}, test {len(te)}")

    train_loader = make_loader(Xk[tr], yk[tr], shuffle=True)
    val_loader = make_loader(Xk[va], yk[va], shuffle=False)
    test_loader = make_loader(Xk[te], yk[te], shuffle=False)

    model = build_model(args.model, n_classes)
    best_val, mean_epoch = fit(model, train_loader, val_loader, tag, epochs,
                               weights_dir=os.path.join(out, WEIGHTS_DIR))

    res = evaluate(model, test_loader, label_map)
    print(res["report"])
    conf_path = os.path.join(out, f"confusion_{tag}.csv")
    save_confusion(res["confusion"], label_map, conf_path)
    print(f"saved confusion matrix to {conf_path}")

    if not args.smoke:  # smoke runs never touch the shared results log
        log_result(args.who, f"{args.model}, {epochs} ep, bs {BATCH_SIZE}, lr {LR}, seed {args.seed}",
                   f"test acc {res['accuracy']:.4f}, macro-F1 {res['macro_f1']:.4f}, "
                   f"test loss {res['loss']:.4f}, mean epoch {mean_epoch:.1f}s")


if __name__ == "__main__":
    main()