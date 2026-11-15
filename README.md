# AIS-NIDS Prototype (MLP) — `NT140.R12.ANTT`

Packet-level autonomous network intrusion detection prototype based on **Farrukh et al., *AIS-NIDS: An intelligent and self-sustaining network intrusion detection system* (Computers & Security, 2024)**, implemented with an **MLP** backbone on **CIC-IDS2017** payload bytes (`capped_sample.pkl`).

## 1. Core Capabilities (Weeks 1–8 Plan)

1. **Closed-Set Classification (`skeleton.py`, `models/mlp.py`):**
   - MLP (`1500 -> 512 -> 256 -> n_classes`) operating on normalized packet payload bytes in $[0, 1]$.
   - Stratified `70/15/15` split (`seed=42`) over 11 known classes (after dropping `Web Attack – Sql Injection`).
2. **Unknown Attack Detector (`pipeline.py`):**
   - Calibrates softmax confidence threshold $t$ on known validation packets only (`--coverage 0.95`), with optional NLL temperature scaling (`--temp-scale`).
3. **Continuous Stream Rule & Memory Module (`pipeline.py`):**
   - `StreamCollector(min_len=4)` discards short unknown bursts ($< 4$ consecutive packets) as bad data and buffers continuous streams ($\ge 4$ packets).
   - `MemoryModule` maintains `ILD1` (5% exemplars per class, min 30) for replay and `VD1` (10% validation subset per class, min 30) for gating.
4. **4-Step Class-Incremental Update + Regularization (`pipeline.py`):**
   - Expands `model.head` by 1 neuron (`Unknown-1`, `Unknown-2`, `Unknown-3`) and runs: (1) weight copy, (2) new-class perturbation, (3) backbone restoration + weight norm alignment, and (4) joint refinement on under-sampled new stream + balanced `ILD1` with Knowledge Distillation ($\lambda_{\text{KD}}=2.0$), L2 anchoring ($\lambda_{L2}=10^{-3}$), and Weight Aligning ($\gamma=0.83$).
5. **Autonomous Validation Gate & Benign Negative Test (`pipeline.py`):**
   - Accepts an update iff max per-class forgetting $< 10\%$ and average accuracy drop $\le 5$ percentage points on `VD1`.
   - Automatically rejects fake attacks (`Fake-BENIGN` negative test).

---

## 2. Exact Reproduction Commands

Make sure `capped_sample.pkl` is placed in the repository root.

```powershell
# 1. Run unit tests (StreamCollector, extend_head, restore_backbone, short-run pipeline test)
python -m pytest -v

# 2. Train/evaluate Week 2 closed-set MLP baseline (raw data)
python skeleton.py --model mlp --who Khiem

# 3. Run full Weeks 3-6 incremental pipeline (C1 deduplicated data, seed 42)
python pipeline.py --model mlp --dedup --coverage 0.95 --sweep 0.80,0.90,0.95,0.99 --gate-min-support 30 --dump results/run_dedup_seed42.json --who Khiem

# 4. Generate Week 3 histograms & Week 6 report tables/plots (Paper Tables 2-5 & Fig. 7)
python plots.py

# 5. Run Week 7-8 end-to-end automated demo
python demo.py
```

---

## 3. Repository Structure

| File / Folder | Purpose |
|---|---|
| `skeleton.py` | Shared data loader, stratified split, model contract check, training & evaluation loop |
| `models/mlp.py` | Registered MLP (`1500 -> 512 -> 256 -> head`) |
| `preprocess.py` | Offline strict-majority payload conflict cleaning (`C1 dedup`) |
| `pipeline.py` | Unknown detector, temperature scaling, stream collector, memory module, 4-step update, validation gate |
| `plots.py` | Generates confidence histograms (`figs/conf_hist_*.png`), Fig. 7 curve (`figs/avg_incremental_acc.png`), and Tables 2–5 (`figs/table_per_class.md`) |
| `demo.py` | End-to-end automated console demo for video recording |
| `tests/` | Pytest unit and integration tests (`test_stream.py`, `test_update.py`, `test_pipeline_short.py`) |
| `results.md` | Complete experiment log, ablation tables, multi-seed robustness, and documented limitations |
| `figs/` | Generated plots and markdown/CSV tables for the final report and slides |
