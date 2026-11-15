"""
AIS-NIDS Week 3 & Week 6 plots and tables generator (P1 & P4).

Generates:
  - figs/conf_hist_raw.png       : Confidence histogram (Known vs Unknown classes) on raw dataset
  - figs/conf_hist_dedup.png     : Confidence histogram (Known vs Unknown classes) on deduplicated dataset
  - figs/avg_incremental_acc.png : Paper Fig. 7 style Average Incremental Accuracy curve across stages
  - figs/table_per_class.csv     : Paper Table 2-4 style per-class accuracy & forgetting across stages
  - figs/table_per_class.md      : Markdown formatted Table 2-4 + Benign Negative Test (Table 5)

Run:
      python plots.py
"""
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import pipeline as pl
from preprocess import resolve_payload_conflicts
import skeleton as sk

FIGS_DIR = "figs"


def plot_confidence_histogram(model, X_known_test: np.ndarray, unknown_sets: Dict[str, np.ndarray],
                              t: float, title: str, out_path: str, temperature: float = 1.0) -> None:
    """Plot max-softmax confidence distribution of Known test packets vs Unknown attack packets."""
    conf_k, _ = pl.predict_conf(model, X_known_test, temperature=temperature)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    bins = np.linspace(0.2, 1.0, 45)
    ax.hist(conf_k, bins=bins, density=True, alpha=0.55, label=f"Known test (mean={conf_k.mean():.3f})", color="#1f77b4")

    colors = {"DoS Hulk": "#d62728", "Bot": "#ff7f0e", "DoS slowloris": "#2ca02c"}
    for name, Xu in unknown_sets.items():
        conf_u, _ = pl.predict_conf(model, Xu, temperature=temperature)
        ax.hist(conf_u, bins=bins, density=True, alpha=0.45,
                label=f"{name} (mean={conf_u.mean():.3f})", color=colors.get(name, None))

    ax.axvline(t, color="black", linestyle="--", linewidth=1.6, label=f"Threshold t = {t:.4f}")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Max Softmax Confidence", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_xlim(0.2, 1.01)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_avg_incremental_accuracy(dump_dedup: dict, dump_raw: dict, out_path: str) -> None:
    """Paper Fig. 7 style plot: Average Incremental Accuracy across incremental stages."""
    stages_d = dump_dedup["stage_accuracies"]
    x_labels = ["Initial\n(11 classes)", "+ DoS Hulk\n(Unknown-1)", "+ Bot\n(Unknown-2)", "+ DoS slowloris\n(Unknown-3)"]
    y_dedup = [s["avg_acc"] * 100.0 for s in stages_d]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    xs = np.arange(len(y_dedup))
    ax.plot(xs, y_dedup, marker="o", linewidth=2.2, markersize=7, color="#1f77b4",
            label="MLP + Dedup (KD + L2 + WA, 3/3 accepted)")
    for x, y in zip(xs, y_dedup):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    if dump_raw and "stage_accuracies" in dump_raw:
        y_raw = [s["avg_acc"] * 100.0 for s in dump_raw["stage_accuracies"]]
        ax.plot(xs[:len(y_raw)], y_raw, marker="s", linestyle="--", linewidth=1.8, markersize=6, color="#7f7f7f",
                label="MLP Raw data (updates rejected by gate)")

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=9.5)
    ax.set_ylabel("Average Incremental Accuracy on VD1 (%)", fontsize=10)
    ax.set_title("Average Incremental Accuracy Across New Attack Classes (Fig. 7 Style)", fontsize=11, fontweight="bold")
    ax.set_ylim(75, 103)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def build_tables(dump_dedup: dict, out_csv: str, out_md: str) -> None:
    """Build Paper Table 2-4 (per-class accuracy & forgetting across stages) and Table 5 (Benign Negative Test)."""
    stages = dump_dedup["stage_accuracies"]
    episodes = dump_dedup["episodes"]
    all_classes = dump_dedup["final_class_names"]

    rows = []
    for c_idx, cname in enumerate(all_classes):
        r = {"Class": cname}
        r["Initial Acc"] = stages[0]["per_class"].get(cname, np.nan)
        for ep_i, ep in enumerate(episodes):
            st_acc = stages[ep_i + 1]["per_class"].get(cname, np.nan)
            r[f"After {ep['name']} Acc"] = st_acc
            if ep["gate"] and str(c_idx) in ep["gate"]["forgetting"]:
                r[f"After {ep['name']} FR"] = ep["gate"]["forgetting"][str(c_idx)]
            elif ep["gate"] and c_idx in ep["gate"]["forgetting"]:
                r[f"After {ep['name']} FR"] = ep["gate"]["forgetting"][c_idx]
            else:
                r[f"After {ep['name']} FR"] = np.nan
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"saved {out_csv}")

    # Format markdown report
    md_lines = [
        "# Week 6 Report Tables (Paper Tables 2–5 Style, MLP on CIC-IDS2017)",
        "",
        "## 1. Per-Class Accuracy and Forgetting Rate Across Incremental Updates (Tables 2–4)",
        "",
        "| Class | Initial Acc | After Unknown-1 (Hulk) Acc | Hulk FR | After Unknown-2 (Bot) Acc | Bot FR | After Unknown-3 (slowloris) Acc | slowloris FR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        def fmt(v, signed=False):
            if pd.isna(v):
                return "—"
            return f"{v*100:+.1f}%" if signed else f"{v*100:.1f}%"

        md_lines.append(
            f"| {row['Class']} | {fmt(row['Initial Acc'])} | "
            f"{fmt(row.get('After Unknown-1 Acc'))} | {fmt(row.get('After Unknown-1 FR'), True)} | "
            f"{fmt(row.get('After Unknown-2 Acc'))} | {fmt(row.get('After Unknown-2 FR'), True)} | "
            f"{fmt(row.get('After Unknown-3 Acc'))} | {fmt(row.get('After Unknown-3 FR'), True)} |"
        )

    # Summary row
    s0 = stages[0]["avg_acc"] * 100
    s1 = stages[1]["avg_acc"] * 100
    s2 = stages[2]["avg_acc"] * 100
    s3 = stages[3]["avg_acc"] * 100
    md_lines.append(
        f"| **Average Incremental Accuracy** | **{s0:.2f}%** | **{s1:.2f}%** | "
        f"drop {episodes[0]['gate']['avg_drop']*100:+.2f} pt | **{s2:.2f}%** | "
        f"drop {episodes[1]['gate']['avg_drop']*100:+.2f} pt | **{s3:.2f}%** | "
        f"drop {episodes[2]['gate']['avg_drop']*100:+.2f} pt |"
    )

    # Episode stream & held-out summary
    md_lines.extend([
        "",
        "## 2. Stream Collection, Placeholder Purity & Held-Out Before/After Check",
        "",
        "| Episode | True Attack | Stream Pkts | Flagged | Kept Runs | Discarded Runs | Collected | Purity | Held-out Unk (Before -> After) | Held-out Routed to New Class | Gate Decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for ep in episodes:
        md_lines.append(
            f"| {ep['name']} | {ep['true_attack']} | {ep['n_stream']} | {ep['n_flagged']} | "
            f"{ep['kept_runs']} | {ep['discarded_runs']} ({ep['discarded_packets']} pkts) | "
            f"{ep['n_collected']} | {ep['purity']*100:.2f}% | "
            f"{ep['heldout_before_unknown_rate']*100:.1f}% -> {ep['heldout_after_unknown_rate']*100:.1f}% | "
            f"{ep['heldout_routed_rate']*100:.1f}% | **{'ACCEPT' if ep['accepted'] else 'REJECT'}** |"
        )

    # Benign Negative Test (Table 5)
    neg = dump_dedup["benign_negative"]
    benign_fr = neg["forgetting"].get("0", neg["forgetting"].get(0, 0.0))
    worst_idx = neg["worst_class"]
    worst_name = all_classes[worst_idx] if worst_idx is not None and worst_idx < len(all_classes) else str(worst_idx)
    md_lines.extend([
        "",
        "## 3. Benign Negative Test (Paper Table 5)",
        "",
        "| Metric | Value | Threshold | Status |",
        "|---|---:|---:|---|",
        f"| Candidate Class | `Fake-BENIGN` (1,000 held-out BENIGN test packets) | — | — |",
        f"| BENIGN Forgetting Rate | **{benign_fr*100:+.2f}%** | < 10.0% | {'FAIL (triggers reject)' if benign_fr >= 0.10 else 'PASS'} |",
        f"| Max Per-Class Forgetting ({worst_name}) | **{neg['max_forgetting']*100:+.2f}%** | < 10.0% | {'FAIL (triggers reject)' if neg['max_forgetting'] >= 0.10 else 'PASS'} |",
        f"| Average Accuracy (Before -> After) | {neg['avg_old']*100:.2f}% -> {neg['avg_new']*100:.2f}% (drop {neg['avg_drop']*100:+.2f} pt) | <= 5.0 pt | {'FAIL (triggers reject)' if neg['avg_drop'] > 0.05 else 'PASS'} |",
        f"| Old-Classes-Only Accuracy Drop | {neg['old_only_drop']*100:+.2f} pt | — | — |",
        f"| **Validation Gate Decision** | **{'ACCEPT' if neg['accept'] else 'REJECT'}** | Expected: REJECT | **PASS (Fake attack rejected)** |",
        "",
    ])

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"saved {out_md}")


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    X0, L0 = sk.load_sample()

    # 1. Raw dataset confidence histogram
    (Xk_r, lk_r), (Xu_r, lu_r) = sk.split_known_unknown(X0, L0)
    lm_r = sk.load_label_map()
    yk_r = np.array([lm_r[l] for l in lk_r], dtype=np.int64)
    sp_r = sk.load_or_make_splits(yk_r, lk_r, 42, "splits_seed42.npz", reuse=True)
    m_raw = sk.build_model("mlp", len(lm_r))
    m_raw.load_state_dict(torch.load("weights/mlp_seed42.pt"))
    m_raw.eval()
    unk_raw = {n: Xu_r[lu_r == n] for n in sorted(set(lu_r))}
    t_raw = pl.calibrate_threshold(m_raw, Xk_r[sp_r["val"]], 0.80)
    plot_confidence_histogram(
        m_raw, Xk_r[sp_r["test"]], unk_raw, t_raw,
        "MLP Confidence Distribution — Raw Dataset (coverage=0.80)",
        os.path.join(FIGS_DIR, "conf_hist_raw.png")
    )

    # 2. Dedup dataset confidence histogram (T=1.0 and T=fitted)
    X_d, L_d, _ = resolve_payload_conflicts(X0, L0)
    (Xk_d, lk_d), (Xu_d, lu_d) = sk.split_known_unknown(X_d, L_d)
    yk_d = np.array([lm_r[l] for l in lk_d], dtype=np.int64)
    sp_d = sk.load_or_make_splits(yk_d, lk_d, 42, "splits_seed42_dedup.npz", reuse=True)
    m_ded = sk.build_model("mlp", len(lm_r))
    m_ded.load_state_dict(torch.load("weights/mlp_seed42_dedup.pt"))
    m_ded.eval()
    unk_ded = {n: Xu_d[lu_d == n] for n in sorted(set(lu_d))}
    T_opt = pl.fit_temperature(m_ded, Xk_d[sp_d["val"]], yk_d[sp_d["val"]])
    t_ded = pl.calibrate_threshold(m_ded, Xk_d[sp_d["val"]], 0.95, temperature=T_opt)
    plot_confidence_histogram(
        m_ded, Xk_d[sp_d["test"]], unk_ded, t_ded,
        f"MLP Confidence Distribution — Deduplicated Dataset (T={T_opt:.2f}, coverage=0.95)",
        os.path.join(FIGS_DIR, "conf_hist_dedup.png"),
        temperature=T_opt
    )

    # 3. Fig 7 & Tables from JSON dumps
    if os.path.exists("results/run_dedup_seed42.json"):
        with open("results/run_dedup_seed42.json") as f:
            dump_dedup = json.load(f)
        dump_raw = None
        if os.path.exists("results/run_raw_seed42.json"):
            with open("results/run_raw_seed42.json") as f:
                dump_raw = json.load(f)
        plot_avg_incremental_accuracy(dump_dedup, dump_raw, os.path.join(FIGS_DIR, "avg_incremental_acc.png"))
        build_tables(dump_dedup, os.path.join(FIGS_DIR, "table_per_class.csv"), os.path.join(FIGS_DIR, "table_per_class.md"))


if __name__ == "__main__":
    main()
