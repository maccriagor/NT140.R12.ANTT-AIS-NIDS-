# AIS-NIDS Prototype — Results Log & Methodological Decisions (`results.md`)

## 1. Methodological Decisions & Project Limitations (Documented per `plan.md`)

1. **Gate 1 — Model Selection (`[Gate-1]`):**
   - **Decision:** Keep **MLP** (`1500 -> 512 -> 256 -> n_classes`) as the sole closed-set classifier for the prototype; **do not implement 1D-CNN or 2D-CNN** (moved to future work).
   - **Evidence:**
     - Training speed is **~1.6s/epoch** on CPU (10 epochs in ~16s), allowing rapid iteration across incremental episodes and seeds.
     - Closed-set test accuracy is **88.47%** (macro-F1 **0.8125**) on raw `capped_sample.pkl` and **99.48%** (macro-F1 **0.9851**) after C1 payload-conflict deduplication.
     - Most importantly (`plan.md` Section 4 milestone 2), MLP successfully supports the 4-step incremental update + validation gate pipeline (`DoS Hulk -> Unknown-1`, `Bot -> Unknown-2`, `DoS slowloris -> Unknown-3`, and `Fake-BENIGN -> REJECT`).
2. **Validation Subset `VD1` (`plan.md` Section 4.2):**
   - Drawn from the **validation split** (`max(10%, vd_min=30)` per class) rather than the test split (`CS-D3` in the paper) so the held-out test set remains completely untouched. Note that the softmax threshold `t` is also calibrated on the known validation split.
3. **Validation Gate Thresholds & Drop Definition (`plan.md` Section 5.2):**
   - Algorithm 1 line 22 ("Average Incremental Accuracy < 5%") is interpreted as **average accuracy drop $\le 5$ percentage points** alongside **per-class forgetting rate $< 10\%$** on classes with support $\ge 30$ in `VD1`.
   - Under `gate_min_support=30`, `PortScan` (only 23 validation rows after deduplication) is reported in gate tables with an `(ignored, n=23)` tag so noisy 1-sample fluctuations ($\pm 4.3\%$) do not falsely trigger rejection.
   - We log both **total drop** (`avg_old - avg_new` where `avg_new` includes the newly learned class) and **old-classes-only drop** (`oldDrop`).
4. **Offline Payload-Conflict Cleaning (`C1 Dedup`, `preprocess.py`):**
   - In packet-level classification using only the 1,500 payload bytes, `Payload-Byte`'s 5-tuple matching without timestamps caused ephemeral port reuse across the Wednesday HTTP DoS attacks, creating identical or template-identical payloads with contradictory labels (`DoS GoldenEye`, `DoS Slowhttptest`, `DoS Hulk`, `DoS slowloris`).
   - **Limitation:** Because `resolve_payload_conflicts` normalizes HTTP templates (including a regex for Hulk GET queries) and runs across the full sample before known/unknown partitioning (mirroring the paper's offline curation in Section 3.1.1), it uses unknown-class label frequencies during cleaning. Also, macro-F1 on deduplicated data (0.9851) is **not directly comparable** to raw data (0.8125) because conflicting rows in `DoS GoldenEye` (1,500 $\to$ 131 test rows), `DoS Slowhttptest` (1,500 $\to$ 181), and `PortScan` (125 $\to$ 24) were removed. Both raw and deduplicated results are reported below.
5. **Dataset Sampling & Dropped Classes (`plan.md` Section 3):**
   - `capped_sample.pkl` (116,726 rows, up to 10,000 per class) takes the **first rows in file order** rather than a uniform random sample.
   - `Web Attack – Sql Injection` has only 12 rows (11 after dedup) and is dropped, leaving 11 known classes and 3 unknown classes (`DoS Hulk`, `Bot`, `DoS slowloris`).
6. **Single-Threshold Unknown Detector Limitation on `Bot` (`plan.md` Section 6):**
   - Unlike the paper's two-stage detector (which sends packets predicted as `BENIGN` to a separate K-means + stacking open-set classifier), our prototype uses a single softmax confidence threshold `t`. Because MLP routes `Bot` packets primarily to `BENIGN` and `SSH-Patator` with high confidence (mean confidence 0.796 on raw, 0.966 on dedup), `Bot` has the lowest flag rate (22.2% on raw at coverage 0.95; 52.8% on dedup). Temperature scaling (`--temp-scale`, optimal $T = 1.3826$ on validation NLL) resolves confidence saturation near 1.0 (`t` shifts from `0.9964` to `0.9798`).
7. **Incremental Update Regularization Extensions (`plan.md` Section 4.3 & P3):**
   - The paper's plain 4-step update (fine-tuning on the full new-class stream + 5% `ILD1` with plain Cross-Entropy) suffers catastrophic forgetting on `DoS GoldenEye` (`maxFR = 73.3%`, rejected by gate) because `DoS Hulk` shares HTTP GET structure with `GoldenEye`.
   - Our update adds four standard class-incremental regularizers: **(a)** class-balanced `ILD1` replay + new-class under-sampling (`new_class_cap=320`), **(b)** old-logit MSE Knowledge Distillation ($\lambda_{\text{KD}}=2.0$), **(c)** L2 parameter anchoring ($\lambda_{L2}=10^{-3}$), and **(d)** post-refinement Weight Aligning ($\gamma=0.83$).
   - **Limitation on boundary sensitivity:** On `DoS Hulk`, `DoS GoldenEye` forgetting sits right at the boundary (`maxFR = 0.100`, 3/30 validation rows). Across 3 random seeds (42, 7, 123), `DoS slowloris` is accepted in 3/3 seeds, `Bot` in 2/3 seeds, `DoS Hulk` in 1/3 seeds (in seed 7 `maxFR = 0.033` passes easily, but total drop is `0.056 > 0.05` due to lower initial `Unknown-1` recall on `VD1`; under `oldDrop = 0.002` it passes), and `Fake-BENIGN` is rejected in **3/3 seeds**.

---

## 2. Week 2 — Closed-Set MLP Baseline (`[W2]`)

| Setting | Known Rows (Train / Val / Test) | Test Acc | Macro-F1 | Test Loss | Mean Epoch Time |
|---|---|---:|---:|---:|---:|
| MLP (`1500->512->256->11`), Raw data, seed 42 | 65,919 / 14,126 / 14,126 | **88.47%** | **0.8125** | 0.2046 | 1.6s |
| MLP (`1500->512->256->11`), C1 Dedup, seed 42 | 52,649 / 11,282 / 11,282 | **99.48%** | **0.9851** | 0.0198 | 1.4s |

Weak classes on raw test set (`confusion_mlp_seed42.csv`): `DoS GoldenEye` (recall 17.8%, F1 0.274 — 1,223/1,500 predicted as `DoS Slowhttptest`), `PortScan` (recall 8.0%, F1 0.147 — 115/125 predicted as `DDoS`).

---

## 3. Week 3 — Unknown Attack Detector Calibration & Sweep (`[W3]`)

Calibrated on **known-class validation rows only** (`calibrate_threshold`). Paper reference flag rates: Hulk **89%**, Bot **90%**, slowloris **84%**. Histograms saved to `figs/conf_hist_raw.png` and `figs/conf_hist_dedup.png`.

| Dataset | Temp $T$ | Val Coverage | Threshold $t$ | False-Unknown (All Known) | False-Unknown (`BENIGN`) | Flag Rate `DoS Hulk` | Flag Rate `Bot` | Flag Rate `DoS slowloris` | Mean Conf (Known vs Unk / Gap) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 1.000 | 0.80 | 0.6509 | 19.76% | 3.13% | **89.07%** | 23.24% | **82.35%** | 0.8913 vs 0.6143 (+0.2770) |
| Raw | 1.000 | 0.90 | 0.5230 | 10.12% | 1.27% | 45.81% | 22.18% | 34.61% | 0.8913 vs 0.6143 (+0.2770) |
| Raw | 1.000 | 0.95 | 0.5118 | 5.04% | 1.13% | 20.96% | 22.18% | 15.59% | 0.8913 vs 0.6143 (+0.2770) |
| Raw | 1.000 | 0.99 | 0.5007 | 0.99% | 1.07% | 2.82% | 22.14% | 1.48% | 0.8913 vs 0.6143 (+0.2770) |
| Dedup | 1.000 | 0.80 / 0.90 | 0.9977 | 8.04% | 8.47% | 93.93% | 57.24% | 100.00% | 0.9950 vs 0.8935 (+0.1016) |
| **Dedup (chosen)** | **1.000** | **0.95** | **0.9964** | **5.20%** | **7.27%** | **89.07%** | **52.83%** | **100.00%** | **0.9950 vs 0.8935 (+0.1016)** |
| Dedup | 1.000 | 0.99 | 0.8986 | 1.19% | 2.67% | 20.50% | 28.58% | 100.00% | 0.9950 vs 0.8935 (+0.1016) |
| Dedup + TempScale | 1.383 | 0.95 | 0.9798 | 5.20% | 7.27% | 89.07% | 52.83% | 100.00% | 0.9901 vs 0.8442 (+0.1459) |

---

## 4. P3 — Incremental Update Ablation & Multi-Seed Robustness (`[P3]`)

### 4.1 Pseudo-Episode Hyperparameter Selection (Holding out known class `DoS Slowhttptest` as pseudo-unknown)
Before touching the 3 true unknown classes, `DoS Slowhttptest` was held out from the 11 known classes (10 known + 1 pseudo-unknown) to verify the regularization pipeline on `VD1`:
- Plain CE (`lambda_kd=0, lambda_l2=0`): `maxFR = 6.67%` on `DoS GoldenEye`, `old_drop = 0.74%`, `new_acc = 95.83%` $\to$ **ACCEPT**.
- Adding KD + L2 + WA (`lambda_kd=2.0, lambda_l2=1e-3, wa_gamma in [0.80, 0.85]`): reduces `maxFR` to **0.00%**, `old_drop = 0.00%`, `new_acc = 91.67%` $\to$ **ACCEPT**.

### 4.2 Component Ablation on Episode 1 (`DoS Hulk`, Seed 42, Dedup)

| Variant | Gate Decision | Max Forgetting (`DoS GoldenEye`) | Avg Acc Drop (All) | Avg Acc Drop (Old Only) | New Class (`Unknown-1`) Val Acc |
|---|---|---:|---:|---:|---:|
| 1. Paper 4-step baseline (CE only, 5 ep, lr 1e-4) | REJECT | **73.33%** | +7.07 pt | +7.87 pt | 100.00% |
| 2. + Balanced `ILD1` & Under-sampling (`cap=320`) | REJECT | 16.67% | +1.60 pt | +1.80 pt | 99.52% |
| 3. + Logit Distillation ($\lambda_{\text{KD}}=2.0$) | REJECT | 20.00% | +3.76 pt | +2.00 pt | 77.80% |
| 4. + L2 Weight Anchoring ($\lambda_{L2}=10^{-3}$) | REJECT | 20.00% | +4.00 pt | +2.00 pt | 75.18% |
| 5. + Weight Aligning ($\gamma=0.85$) | **ACCEPT** | **10.00%** | +3.56 pt | +1.00 pt | 69.93% |
| 6. + Weight Aligning ($\gamma=0.83$, chosen) | **ACCEPT** | **10.00%** | +4.02 pt | +1.00 pt | 64.92% |

### 4.3 Multi-Seed Robustness Across All 3 Episodes + Benign Negative Test (Dedup, `cov=0.95`)

| Seed | Episode 1 (`DoS Hulk`) | Episode 2 (`Bot`) | Episode 3 (`DoS slowloris`) | Benign Negative Test (`Fake-BENIGN`) |
|---:|---|---|---|---|
| **42** | **ACCEPT** (`maxFR=0.100`, `drop=+0.040`, `oldDrop=+0.010`, routed `59.4%`) | **ACCEPT** (`maxFR=0.007`, `drop=-0.002`, `oldDrop=+0.004`, routed `54.7%`) | **ACCEPT** (`maxFR=0.000`, `drop=-0.005`, `oldDrop=-0.001`, routed `100.0%`) | **REJECT** (`drop=+0.069 > 0.05`, `newAcc=0.000`) |
| **7** | REJECT (`maxFR=0.033`, `drop=+0.056 > 0.05`, `oldDrop=+0.002`) | **ACCEPT** (`maxFR=0.000`, `drop=+0.004`, `oldDrop=+0.000`, routed `54.7%`) | **ACCEPT** (`maxFR=0.033`, `drop=+0.002`, `oldDrop=+0.005`, routed `100.0%`) | **REJECT** (`drop=+0.078 > 0.05`, `newAcc=0.000`) |
| **123** | REJECT (`maxFR=0.100` $\ge 0.10$, `drop=+0.054`, `oldDrop=+0.011`) | REJECT (`maxFR=0.000`, `drop=+0.085`, `oldDrop=-0.003`) | **ACCEPT** (`maxFR=0.000`, `drop=-0.010`, `oldDrop=-0.007`, routed `100.0%`) | **REJECT** (`drop=+0.083 > 0.05`, `newAcc=0.000`) |

*(Full per-class tables and plots are generated in `figs/table_per_class.md`, `figs/table_per_class.csv`, and `figs/avg_incremental_acc.png`).*

---

## 5. Chronological Experiment Log

- 2026-11-09 | Khiem | `[W2]` mlp, 10 ep, bs 256, lr 0.001, seed 42 (raw) | test acc 0.8847, macro-F1 0.8125, test loss 0.2046, mean epoch 1.6s
- 2026-11-09 | Khiem | `[W4-W5]` pipeline mlp (raw), t=0.5118 (cov 0.95), min_run 4, UpdateConfig(perturb_epochs=1, perturb_lr=0.001, refine_epochs=5, refine_lr=0.0001, new_class_cap=None) | DoS Hulk->rejected purity 1.0 maxFR 0.607 drop 0.076; Bot: no update; DoS slowloris: no update; benign negative REJECT
- 2026-11-09 | Khiem | `[W4-W5]` pipeline mlp (raw), t=0.6509 (cov 0.8), min_run 4, UpdateConfig(perturb_epochs=1, perturb_lr=0.001, refine_epochs=5, refine_lr=0.0001, new_class_cap=None) | DoS Hulk->rejected purity 1.0 maxFR 0.520 drop 0.072; Bot: no update; DoS slowloris->rejected purity 1.0 maxFR 0.460 drop 0.062; benign negative REJECT
- 2026-11-09 | Khiem | `[W4-W5]` pipeline mlp_dedup, dedup=True, vd_min=30, gate_min_support=30, t=0.9937 (cov 0.8), min_run 4, UpdateConfig(perturb_epochs=1, perturb_lr=0.001, refine_epochs=5, refine_lr=0.0001, new_class_cap=None) | DoS Hulk->rejected purity 1.0 maxFR 0.573 drop 0.110; Bot->rejected purity 1.0 maxFR 0.194 drop 0.043; DoS slowloris->rejected purity 1.0 maxFR 0.274 drop 0.097; benign negative REJECT
- 2026-11-10 | Khiem | `[W4-W5]` pipeline mlp_dedup, dedup=True, vd_min=30, gate_min_support=30, t=0.9977 (cov 0.8), min_run 4, UpdateConfig(perturb_epochs=1, perturb_lr=0.001, refine_epochs=10, refine_lr=0.0005, lambda_kd=2.0, lambda_l2=0.001, wa_gamma=0.85, new_class_cap=320) | DoS Hulk->rejected purity 1.0 maxFR 0.133 drop 0.041; Bot->Unknown-1 purity 1.0 maxFR 0.007 drop 0.002; DoS slowloris->Unknown-2 purity 0.9995 maxFR 0.000 drop -0.001; benign negative REJECT
- 2026-11-10 | Khiem | `[W4-W5]` pipeline mlp_dedup, dedup=True, vd_min=30, gate_min_support=30, t=0.9977 (cov 0.8), min_run 4, UpdateConfig(perturb_epochs=1, perturb_lr=0.001, refine_epochs=10, refine_lr=0.0005, lambda_kd=2.0, lambda_l2=0.001, wa_gamma=0.83, new_class_cap=320) | DoS Hulk->Unknown-1 purity 1.0 maxFR 0.100 drop 0.044; Bot->Unknown-2 purity 1.0 maxFR 0.009 drop 0.001; DoS slowloris->Unknown-3 purity 0.9995 maxFR 0.033 drop -0.002; benign negative REJECT
- 2026-11-10 | Khiem | `[W4-W6]` pipeline mlp_dedup, dedup=True, vd_min=30, gate_min_support=30, t=0.9964 (cov 0.95), min_run 4, UpdateConfig(perturb_epochs=1, perturb_lr=0.001, refine_epochs=10, refine_lr=0.0005, lambda_kd=2.0, lambda_l2=0.001, wa_gamma=0.83, new_class_cap=320) | DoS Hulk->Unknown-1 purity 1.0 maxFR 0.100 drop 0.040; Bot->Unknown-2 purity 1.0 maxFR 0.007 drop -0.002; DoS slowloris->Unknown-3 purity 0.9995 maxFR 0.000 drop -0.005; benign negative REJECT
- 2026-11-15 | Khiem | `[W3-W6]` pipeline mlp_dedup, dedup=True, seed=42, vd_min=30, gate_min_support=30, drop_mode=all, T=1.000, t=0.9964 (cov 0.95), min_run 4, UpdateConfig(perturb_epochs=1, perturb_lr=0.001, refine_epochs=10, refine_lr=0.0005, lambda_kd=2.0, lambda_l2=0.001, wa_gamma=0.83, new_class_cap=320) | DoS Hulk->Unknown-1 (purity 1.000, maxFR 0.100 on DoS GoldenEye, drop 0.040, oldDrop 0.010, holdout unk 0.893->1.000, routed 0.594); Bot->Unknown-2 (purity 1.000, maxFR 0.007 on BENIGN, drop -0.002, oldDrop 0.004, holdout unk 0.591->0.592, routed 0.547); DoS slowloris->Unknown-3 (purity 0.999, maxFR 0.000 on BENIGN, drop -0.005, oldDrop -0.001, holdout unk 1.000->1.000, routed 1.000); benign negative REJECT (maxFR 0.010 on Unknown-1, drop 0.069, oldDrop 0.001)
- 2026-11-15 | Khiem | `[W3-W6]` pipeline mlp_dedup, dedup=True, seed=42, vd_min=30, gate_min_support=30, drop_mode=all, T=1.383 (temp-scale), t=0.9798 (cov 0.95), min_run 4, UpdateConfig(perturb_epochs=1, perturb_lr=0.001, refine_epochs=10, refine_lr=0.0005, lambda_kd=2.0, lambda_l2=0.001, wa_gamma=0.83, new_class_cap=320) | DoS Hulk->Unknown-1 (purity 1.000, maxFR 0.100 on DoS GoldenEye, drop 0.039, oldDrop 0.010, holdout unk 0.893->0.997, routed 0.591); Bot->Unknown-2 (purity 1.000, maxFR 0.007 on BENIGN, drop 0.000, oldDrop 0.007, holdout unk 0.591->0.590, routed 0.547); DoS slowloris->Unknown-3 (purity 0.999, maxFR 0.000 on BENIGN, drop -0.004, oldDrop 0.000, holdout unk 1.000->1.000, routed 1.000); benign negative REJECT (maxFR 0.014 on Unknown-1, drop 0.069, oldDrop 0.001)
- 2026-11-15 | Khiem | `[W3-W6]` pipeline mlp (raw), dedup=False, seed=42, vd_min=30, gate_min_support=30, drop_mode=all, T=1.000, t=0.6509 (cov 0.80), min_run 4, UpdateConfig(perturb_epochs=1, perturb_lr=0.001, refine_epochs=10, refine_lr=0.0005, lambda_kd=2.0, lambda_l2=0.001, wa_gamma=0.83, new_class_cap=320) | DoS Hulk->rejected (purity 1.000, maxFR 0.220 on DoS Slowhttptest, drop 0.044, oldDrop 0.004, holdout unk 0.888->0.888, routed 0.000); Bot: no update (collected 0 pkts); DoS slowloris->rejected (purity 1.000, maxFR 0.240 on DoS Slowhttptest, drop 0.097, oldDrop 0.025, holdout unk 0.839->0.839, routed 0.000); benign negative REJECT (maxFR 0.020 on DoS Slowhttptest, drop 0.063, oldDrop -0.011)
