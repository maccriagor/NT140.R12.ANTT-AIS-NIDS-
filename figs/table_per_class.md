# Week 6 Report Tables (Paper Tables 2–5 Style, MLP on CIC-IDS2017)

## 1. Per-Class Accuracy and Forgetting Rate Across Incremental Updates (Tables 2–4)

| Class | Initial Acc | After Unknown-1 (Hulk) Acc | Hulk FR | After Unknown-2 (Bot) Acc | Bot FR | After Unknown-3 (slowloris) Acc | slowloris FR |
|---|---:|---:|---:|---:|---:|---:|---:|
| BENIGN | 95.3% | 95.3% | +0.0% | 94.7% | +0.7% | 94.7% | +0.0% |
| DDoS | 100.0% | 100.0% | +0.0% | 100.0% | +0.0% | 100.0% | +0.0% |
| DoS GoldenEye | 100.0% | 90.0% | +10.0% | 93.3% | -3.3% | 93.3% | +0.0% |
| DoS Slowhttptest | 96.7% | 96.7% | +0.0% | 96.7% | +0.0% | 96.7% | +0.0% |
| FTP-Patator | 100.0% | 100.0% | +0.0% | 100.0% | +0.0% | 100.0% | +0.0% |
| Heartbleed | 100.0% | 100.0% | +0.0% | 100.0% | +0.0% | 100.0% | +0.0% |
| Infiltration | 99.3% | 99.3% | +0.0% | 99.3% | +0.0% | 99.3% | +0.0% |
| PortScan | 87.0% | 87.0% | +0.0% | 87.0% | +0.0% | 87.0% | +0.0% |
| SSH-Patator | 100.0% | 100.0% | +0.0% | 100.0% | +0.0% | 100.0% | +0.0% |
| Web Attack – Brute Force | 100.0% | 100.0% | +0.0% | 100.0% | +0.0% | 100.0% | +0.0% |
| Web Attack – XSS | 100.0% | 100.0% | +0.0% | 100.0% | +0.0% | 100.0% | +0.0% |
| Unknown-1 | — | 64.9% | — | 64.9% | +0.0% | 66.3% | -1.4% |
| Unknown-2 | — | — | — | 94.4% | — | 94.4% | +0.0% |
| Unknown-3 | — | — | — | — | — | 100.0% | — |
| **Average Incremental Accuracy** | **98.03%** | **94.43%** | drop +4.02 pt | **94.64%** | drop -0.17 pt | **95.12%** | drop -0.47 pt |

## 2. Stream Collection, Placeholder Purity & Held-Out Before/After Check

| Episode | True Attack | Stream Pkts | Flagged | Kept Runs | Discarded Runs | Collected | Purity | Held-out Unk (Before -> After) | Held-out Routed to New Class | Gate Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Unknown-1 | DoS Hulk | 5581 | 4476 | 365 | 165 (283 pkts) | 4193 | 100.00% | 89.3% -> 100.0% | 59.4% | **ACCEPT** |
| Unknown-2 | Bot | 2124 | 902 | 62 | 347 (543 pkts) | 359 | 100.00% | 59.1% -> 59.2% | 54.7% | **ACCEPT** |
| Unknown-3 | DoS slowloris | 2588 | 2053 | 1 | 59 (64 pkts) | 1989 | 99.95% | 100.0% -> 100.0% | 100.0% | **ACCEPT** |

## 3. Benign Negative Test (Paper Table 5)

| Metric | Value | Threshold | Status |
|---|---:|---:|---|
| Candidate Class | `Fake-BENIGN` (1,000 held-out BENIGN test packets) | — | — |
| BENIGN Forgetting Rate | **+0.00%** | < 10.0% | PASS |
| Max Per-Class Forgetting (Unknown-1) | **+0.95%** | < 10.0% | PASS |
| Average Accuracy (Before -> After) | 95.75% -> 88.85% (drop +6.91 pt) | <= 5.0 pt | FAIL (triggers reject) |
| Old-Classes-Only Accuracy Drop | +0.07 pt | — | — |
| **Validation Gate Decision** | **REJECT** | Expected: REJECT | **PASS (Fake attack rejected)** |
