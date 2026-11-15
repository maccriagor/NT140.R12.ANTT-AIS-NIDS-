"""
AIS-NIDS Week 7-8 End-to-End Autonomous Demo Script (MLP Prototype).

Demonstrates all 5 core objectives from plan.md Section 1 in a single automated run:
  1. Classify KNOWN attacks on test traffic (Closed-Set MLP).
  2. Flag unseen packets as UNKNOWN via calibrated softmax threshold + stream rule (discard runs < 4).
  3. Collect continuous unknown streams (DoS Hulk -> Bot -> DoS slowloris), assign temporary labels
     (`Unknown-1`, `Unknown-2`, `Unknown-3`), and learn each via 4-step incremental update + KD/L2/WA.
  4. Accept or reject each update via the autonomous Validation Gate (maxFR < 10%, drop <= 5%).
  5. Run the Benign Negative Test (fake attack stream of BENIGN traffic) and show the gate REJECTS it,
      plus a short-burst noise test (runs of length < 4) showing outlier rejection before learning.

Usage:
      python demo.py                 # Full end-to-end demo (~30 seconds)
      python demo.py --fast          # Fast mode for quick video recording / preview
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch

import pipeline as pl
from preprocess import resolve_payload_conflicts
import skeleton as sk

SEP = "=" * 88
SUB = "-" * 88


def banner(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def stream_profile(df_sub: pd.DataFrame) -> str:
    """Optional stream summary from plan.md Section 8: mean packet len, protocol ratio, t_delta."""
    if df_sub is None or len(df_sub) == 0:
        return "n/a"
    parts = []
    if "total_len" in df_sub.columns:
        parts.append(f"mean_pkt_len={df_sub['total_len'].mean():.1f}B")
    if "ttl" in df_sub.columns:
        parts.append(f"mean_ttl={df_sub['ttl'].mean():.1f}")
    if "protocol" in df_sub.columns:
        vc = df_sub["protocol"].value_counts(normalize=True)
        top_p = ", ".join(f"{k}:{v*100:.0f}%" for k, v in vc.head(2).items())
        parts.append(f"proto=[{top_p}]")
    if "t_delta" in df_sub.columns:
        parts.append(f"median_dt={df_sub['t_delta'].median():.5f}s")
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=sk.SEED)
    ap.add_argument("--coverage", type=float, default=0.95)
    args = ap.parse_args()

    t_start = time.perf_counter()
    sk.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    banner("STAGE 1: DATA LOADING, PREPROCESSING & CLOSED-SET MLP BASELINE")
    X0, L0 = sk.load_sample()
    X, labels, dedup_stats = resolve_payload_conflicts(X0, L0)
    print(f"Loaded capped_sample.pkl: {len(L0):,} packets -> {len(labels):,} after C1 payload conflict cleaning")
    (Xk, lk), (Xu, lu) = sk.split_known_unknown(X, labels)
    label_map = sk.build_label_map(lk)
    class_names = sk.class_names(label_map)
    yk = np.array([label_map[l] for l in lk], dtype=np.int64)

    sp = sk.load_or_make_splits(yk, lk, args.seed, f"splits_seed{args.seed}_dedup.npz", reuse=True)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"Known classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"Stratified 70/15/15 split: train={len(tr):,}, val={len(va):,}, test={len(te):,}")

    ckpt = os.path.join(sk.WEIGHTS_DIR, f"mlp_seed{args.seed}_dedup.pt")
    model = sk.build_model("mlp", len(class_names))
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        print(f"Loaded trained closed-set MLP checkpoint: {ckpt}")
    else:
        print(f"Training closed-set MLP ({sk.EPOCHS} epochs)...")
        sk.fit(model, sk.make_loader(Xk[tr], yk[tr], True), sk.make_loader(Xk[va], yk[va], False),
               f"mlp_seed{args.seed}_dedup", sk.EPOCHS)

    with torch.random.fork_rng():
        ev = sk.evaluate(model, sk.make_loader(Xk[te], yk[te], False), label_map)
    print(f"\nClosed-Set Test Accuracy: {ev['accuracy']*100:.2f}% | Macro-F1: {ev['macro_f1']*100:.2f}%")

    banner("STAGE 2: UNKNOWN ATTACK DETECTOR (SOFTMAX THRESHOLD CALIBRATION)")
    unknown_sets = {}
    for key in pl.UNKNOWN_ORDER:
        name = next(n for n in sorted(set(lu)) if key in n.lower())
        unknown_sets[name] = Xu[lu == name]

    t = pl.calibrate_threshold(model, Xk[va], args.coverage)
    rep = pl.unknown_detection_report(model, t, Xk[te], yk[te], unknown_sets, class_names)
    print(f"Calibrated threshold on known validation split (coverage={args.coverage:.2f}): t = {t:.4f}")
    print(f"  Known Test Mean Confidence : {rep['mean_conf_known']:.4f} | False-Unknown Rate (All): {rep['false_unknown_all']*100:.2f}% | (BENIGN): {rep['false_unknown_benign']*100:.2f}%")
    for name in unknown_sets:
        print(f"  Unseen Attack [{name:<13s}]: Flagged Unknown = {rep[f'flag_rate[{name}]']*100:6.2f}% | Mean Confidence = {rep[f'mean_conf[{name}]']:.4f}")

    banner("STAGE 3: STREAM RULE OUTLIER REJECTION TEST (RUNS < 4 PACKETS)")
    col_test = pl.StreamCollector(min_len=4)
    sample_flags = [False, True, True, True, False, True, True, False, True, True, True, True, False]
    kept = []
    for idx_pkt, fl in enumerate(sample_flags):
        r = col_test.push(idx_pkt, fl)
        if r is not None:
            kept.append(r)
    print(f"Simulated flag sequence: {[int(b) for b in sample_flags]}")
    print(f"  Runs < 4 discarded as corrupted/bad data : {col_test.discarded_runs} runs ({col_test.discarded_packets} packets)")
    print(f"  Runs >= 4 buffered as valid unknown stream: {col_test.kept_runs} run -> indices {kept[0]}")

    banner("STAGE 4: AUTONOMOUS INCREMENTAL LEARNING EPISODES (HULK -> BOT -> SLOWLORIS)")
    memory = pl.MemoryModule.build((Xk[tr], yk[tr]), (Xk[va], yk[va]), vd_min_per_class=30, seed=args.seed)
    print(f"Memory initialized: {memory.summary()}")
    benign_te = te[yk[te] == label_map[pl.BENIGN_NAME]]
    cfg = pl.UpdateConfig(refine_epochs=10, wa_gamma=0.83, new_class_cap=320)

    for ep_idx, (name, Xc) in enumerate(unknown_sets.items(), 1):
        print(f"\n{SUB}\n  EPISODE {ep_idx}: Incoming Stream with Unseen Attack '{name}'\n{SUB}")
        n_stream = int(round(0.60 * len(Xc)))
        X_attack, X_hold = Xc[:n_stream], Xc[n_stream:]
        b1, b2 = (rng.choice(benign_te, size=300, replace=False) for _ in range(2))
        X_stream = np.concatenate([Xk[b1], X_attack, Xk[b2]])
        truth = np.array([pl.BENIGN_NAME] * len(b1) + [name] * len(X_attack) + [pl.BENIGN_NAME] * len(b2))
        print(f"  Stream layout: 300 BENIGN -> {len(X_attack)} {name} -> 300 BENIGN (total {len(X_stream)} pkts)")

        _, _, before = pl.detect_unknown(model, X_hold, t)
        model, log = pl.run_episode(
            model, memory, X_stream, t, class_names, cfg, min_run=4, true_names=truth,
            gate_min_support=30, true_attack=name
        )
        _, pred_after, after_unk = pl.detect_unknown(model, X_hold, t)
        routed = float((pred_after == pl.n_outputs(model) - 1).mean()) if log.accepted else 0.0
        print(f"  [Before/After on {len(X_hold)} Held-Out '{name}' Packets]")
        print(f"    Before update: {before.mean()*100:6.2f}% flagged unknown")
        print(f"    After update : {after_unk.mean()*100:6.2f}% flagged unknown | {routed*100:6.2f}% classified as '{log.name if log.accepted else '-'}'")
        print(f"    Memory state : {memory.summary()}")

    banner("STAGE 5: BENIGN NEGATIVE TEST (SELF-SUSTAINING GATE REJECTS FAKE ATTACK)")
    rest = np.setdiff1d(benign_te, np.concatenate([b1, b2]))
    fake = rng.choice(rest, size=min(1000, len(rest)), replace=False)
    print(f"Feeding {len(fake)} held-out BENIGN packets as a candidate unknown attack ('Fake-BENIGN')...")
    neg = pl.benign_negative_test(model, memory, Xk[fake], class_names, cfg, gate_min_support=30)
    print(f"  Result: {'ACCEPT (FAILURE)' if neg.accept else 'REJECT (SUCCESS - Fake update discarded, production model preserved)'}")

    banner("FINAL SUMMARY")
    print(f"  Total classes in production model : {pl.n_outputs(model)} ({', '.join(class_names)})")
    print(f"  Total demo runtime                : {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
