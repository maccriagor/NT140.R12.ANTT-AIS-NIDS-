# AIS-NIDS: Technical Methodology & Architecture Specification

> [!IMPORTANT]
> **Critical Acronym & Architecture Clarification Before Implementation:**
> In the provided 16-page research paper (`AIS-NIDS.pdf`, *Farrukh et al., Computers & Security, 2024*), **AIS-NIDS** does **not** stand for *"Artificial Immune System-based Network Intrusion Detection System"*. Instead, it stands for **"An Intelligent and Self-Sustaining Network Intrusion Detection System"**.
>
> The paper does **not** use classical biological immune algorithms (such as Negative Selection Algorithm [NSA], Clonal Selection [CLONALG], aiNet, or Euclidean/Hamming affinity balls). Instead, it implements an **autonomous cyber-defense pipeline** combining **Packet-Level Feature Serialization**, **Closed-Set Deep Concatenated 2D-CNNs**, **Open-Set Recognition (K-Means Benign Sub-Clustering + Stacking Meta-Learners)**, **Exemplar Memory Replay**, **Class-Incremental Learning with Weight Restoration**, and an **Autonomous Validation Gate**.
>
> Below is the 100% faithful technical extraction of the paper's exact methodology and its alignment with the provided dataset (`capped_sample.pkl`), structured into the exact 6 sections so that a downstream coding LLM (e.g., Claude 3 Opus) can generate a step-by-step Python implementation plan without hallucinating biological AIS heuristics.

---

## 1. System Architecture & Data Flow

AIS-NIDS consists of **five interconnected modules** operating sequentially on packet-level network traffic to classify known attacks, isolate unknown (zero-day) attacks, and autonomously learn new attack classes without full retraining.

### Step-by-Step Pipeline (Raw Packet to Final Classification & Self-Update)

1. **Packet Capture & Labeling (Offline Curation via `Payload-Byte`):**
   * Raw `.pcap` files are matched against labeled flow metadata using 5-tuple attributes (`Source IP`, `Destination IP`, `Source Port`, `Destination Port`, `Protocol`).
   * Each packet is converted into a 1D vector of **1,500 payload bytes** (`payload_byte_1` to `payload_byte_1500`) plus **header/packet attributes** (`ttl`, `total_len`, `protocol`, and `t_delta` in `capped_sample.pkl`, totaling 1,504 features plus `label`).
2. **Feature Space Transformation (Serialized RGB Image Construction):**
   * In the paper's full 2D-CNN formulation, each 1,504-feature packet vector is zero-padded at both ends to **2,048 features**.
   * **5 consecutive packets** are grouped together to capture temporal correlation, and **array reversal (flipping)** is applied to populate the remaining RGB channels, producing a serialized 3-channel image tensor of shape:
     $$\mathbf{I} \in \mathbb{R}^{128 \times 64 \times 3}$$
   * *(Note: In the workspace prototype `skeleton.py` and `plan.md`, models can also operate directly on the normalized 1D payload tensor of shape $(\text{batch}, 1500)$ in $[0, 1]$ before optionally scaling up to the 2D-CNN $128 \times 64 \times 3$ image representation).*
3. **Component #1 — Closed-Set Classifier (Misuse Detection):**
   * Incoming packet representations pass first through the **Closed-Set Classifier** ($\Theta_b$), a 4-block Deep Concatenated 2D-CNN trained on $N$ known attack classes plus `BENIGN` ($N + 1$ total classes).
   * **Branch A (Known Attack Detected):** If $\Theta_b$ classifies the sample as one of the specific known attack classes, the system immediately triggers an intrusion alarm and terminates inference for that sample.
   * **Branch B (Classified as Benign):** Because $\Theta_b$ operates under a closed-world assumption, both truly `BENIGN` traffic and previously unseen `Unknown Attacks` are predicted as `BENIGN` (or low-confidence). All samples predicted as `BENIGN` by $\Theta_b$ are forwarded to **Component #2**.
4. **Component #2 — Open-Set Classifier (Anomaly / Zero-Day Detection):**
   * Receives only samples labeled as `BENIGN` by the Closed-Set Classifier.
   * Uses a two-level **Benign Sub-Clustering + Stacking Ensemble** architecture:
     * **Base Learners:** Output cluster membership probabilities $P(\text{Benign\_Cluster\_}k)$ across $K$ unsupervised K-Means sub-clusters of normal traffic.
     * **Probability Aggregator:** Concatenates the cluster probability outputs into a meta-feature vector.
     * **Meta Learners:** An ensemble of `XGBoost`, `LightGBM`, and a Multi-Layer Perceptron (MLP) votes on the aggregated probabilities.
   * **Branch B1 (Confirmed Benign):** If the Open-Set Classifier predicts `Benign`, the system logs normal operation (`No threat detected`).
   * **Branch B2 (Unknown Attack Detected):** If the Open-Set Classifier flags the sample as an `Unknown Attack`, the sample is routed to **Component #3 (Memory Module)**.
5. **Component #3 — Memory Module (Stream Filtering & Exemplar Replay Buffer):**
   * **Continuous Stream Rule (Outlier Rejection):** Tracks consecutive samples flagged as `Unknown Attack`.
     * If the continuous stream length is $< 4$ samples, the module classifies them as corrupted/bad data instances, raises a bad-data alert, and discards them.
     * If the continuous stream length is $\ge 4$ consecutive samples, the module raises an unknown-attack alert and buffers the entire stream until the stream terminates.
   * Combines the newly collected unknown attack stream (under-sampled if excessive to match class ratios) with stored **Exemplars (`ILD1`)** (5% of training samples from each previously known class) to construct the incremental training set $X_{\text{train}}$, and reserves a subset of the new class for validation (`VD1`).
6. **Component #4 — Incremental Learning Module (Dynamic Architecture Expansion):**
   * Clones the current Closed-Set Classifier $\Theta_b \to \Theta_u$ and appends **1 new output neuron** to the final classification layer (`head`) for the new attack class $X_{N+1}$.
   * Executes a 4-stage update: **(1) Weight Copy Initialization**, **(2) Initial New-Class Perturbation Step**, **(3) Backbone Weight Restoration** (restoring all hidden layers to $\Theta_b$ while keeping the new output layer weights), and **(4) Iterative Refinement** on $X_{\text{train}}$ (`Unknown Attacks` $\cup$ `ILD1`).
7. **Component #5 — Autonomous Validation Module (Self-Sustaining Gate):**
   * Evaluates both $\Theta_b$ (current model) and $\Theta_u$ (updated model) on the held-out validation memory subset **`VD1`**.
   * Computes two gating metrics: **Per-Class Forgetting Rate ($FR_c$)** and **Average Incremental Accuracy Drop ($\Delta \text{AvgAcc}$)**.
   * **Accept Update:** If $FR_c < 10\%$ for all known classes and $\Delta \text{AvgAcc} < 5\%$, $\Theta_u$ replaces $\Theta_b$ in production, the new class is registered (e.g., `X1` / `Unknown-1`), and proportional exemplars are added to `ILD1` and `VD1`.
   * **Reject Update:** Otherwise, $\Theta_u$ and the buffered stream are discarded, preserving $\Theta_b$.

---

## 2. Data Preprocessing

### 2.1 Datasets Used
The paper evaluates the methodology across three packet-level datasets (with full architectural walkthroughs and tables reported on **CIC-IDS2017**):
1. **CIC-IDS2017** (Primary dataset, used in `capped_sample.pkl`): Simulated enterprise network traffic (PCAP + CICFlowMeter CSVs) containing `BENIGN` and 14 attack classes.
2. **UNSW-NB15**: 99 GB of PCAPs across 79 files + Argus/Bro-IDS flow metadata covering 9 attack categories.
3. **ACI IoT Network Traffic Dataset 2023**: Real/simulated home IoT wired and wireless packet captures collected over 1 week.

### 2.2 Packet Extraction, Cleaning & Balancing (`Payload-Byte`)
* **5-Tuple Flow-to-Packet Labeling:** Raw `.pcap` packets are matched to flow labels via `(Source IP, Destination IP, Source Port, Destination Port, Protocol)`.
* **Payload Extraction & Zero-Padding:**
  * Maximum payload length is fixed at **1,500 bytes**.
  * Each byte's hexadecimal value (`0x00`–`0xFF`) is converted to an unsigned integer in $[0, 255]$.
  * Payloads shorter than 1,500 bytes are zero-padded (`0`) up to index 1,500.
* **Header Attributes Extracted:**
  * Paper text specifies `Time to Live (TTL)`, `Total Length`, and `Protocol` (plus inter-packet time delta `t_delta` in the released `Payload-Byte` dataframe, yielding $1500 + 4 = 1504$ raw features).
* **Cleaning & Filtering:**
  * Packets with **empty payloads** (0 payload bytes) are removed.
  * **Duplicate packet vectors** are removed.
  * **Class Imbalance Handling:** The majority `BENIGN` class is randomly under-sampled to mitigate skew against minority attack classes.

### 2.3 Workspace Dataset Schema (`capped_sample.pkl`)
The provided `capped_sample.pkl` is a `pandas.DataFrame` of shape `(116726, 1505)` capped at a maximum of 10,000 rows per class:

| Column Group | Columns | Dtype / Range | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| **Payload Bytes** | `payload_byte_1` ... `payload_byte_1500` | `uint8` ($0 \dots 255$) | Primary packet payload feature vector (1,500 features) |
| **Header / Packet Metadata** | `ttl` ($37 \dots 255$), `total_len` ($42 \dots 23400$), `protocol` (`tcp`, `udp`), `t_delta` ($-0.000012 \dots 5.202\text{ s}$) | `int` / `str` / `float` | 4 supplementary header features (combined with 1,500 payload bytes = 1,504 features in paper) |
| **Target Class** | `label` | `str` (15 raw labels) | Ground-truth class name |

#### Class Distribution in `capped_sample.pkl` & Partition Role:

| Class Label | Row Count in `capped_sample.pkl` | Role in Paper (Section 3.1.3 & Table 1) | Role in Repo (`skeleton.py`) |
| :--- | :--- | :--- | :--- |
| `BENIGN` | 10,000 | Known Class (Normal) | Known Class |
| `DDoS` | 10,000 | Known Attack Class #1 | Known Attack Class |
| `DoS GoldenEye` | 10,000 | Known Attack Class #2 | Known Attack Class |
| `DoS Slowhttptest` | 10,000 | Known Attack Class #3 (Also in `OS-D3`) | Known Attack Class |
| `FTP-Patator` | 10,000 | Known Attack Class #4 (Also in `OS-D3`) | Known Attack Class |
| `Heartbleed` | 10,000 | Known Attack Class #5 | Known Attack Class |
| `Infiltration` | 10,000 | Known Attack Class #6 | Known Attack Class |
| `PortScan` | 830 | Known Attack Class #7 | Known Attack Class |
| `SSH-Patator` | 10,000 | Known Attack Class #8 | Known Attack Class |
| `Web Attack – Brute Force` | 10,000 | Known Attack Class #9 | Known Attack Class |
| `Web Attack – XSS` | 3,341 | Known Attack Class #10 | Known Attack Class |
| `Web Attack – Sql Injection` | 12 | Known Attack Class #11 (Also in `OS-D3`) | Dropped in `skeleton.py` (only 12 rows) |
| **`DoS Hulk`** | 10,000 | **Held-Out Unknown Attack ($X_1$)** | **Held-Out Unknown Attack ($X_1$)** |
| **`Bot`** | 2,543 | **Held-Out Unknown Attack ($X_2$)** | **Held-Out Unknown Attack ($X_2$)** |
| **`DoS slowloris`** | 10,000 | **Held-Out Unknown Attack ($X_3$)** | **Held-Out Unknown Attack ($X_3$)** |

### 2.4 Categorical Encoding, Normalization & Feature Transformation
* **Categorical Encoding:**
  * Target `label`: Encoded via integer mapping $y \in \{0, 1, \dots, N_{\text{known}}-1\}$ for known classes; dynamically extended to $N_{\text{known}}, N_{\text{known}}+1, \dots$ as unknown classes are incrementally learned.
  * `protocol` (`tcp`, `udp`): Binary/one-hot encoded (`0`/`1`) or integer-mapped when header features are included.
* **Numerical Normalization:**
  * Payload bytes $\mathbf{x} \in \{0, \dots, 255\}^{1500}$ are scaled to floating-point $[0.0, 1.0]$ via division by $255.0$:
    $$\tilde{x}_j = \frac{x_j}{255.0}, \quad j \in \{1, \dots, 1500\}$$
  * Optional metadata scaling (if included alongside payload bytes): $\text{ttl} / 255.0$, $\text{total\_len} / 1500.0$, and $\log(1 + \max(0, \text{t\_delta}))$.
* **Dimensionality Expansion (Paper Section 3.1.2 — Serialized RGB Image):**
  * **Zero-Padding Expansion:** The 1,504-element 1D packet vector is padded symmetrically with zeros at both ends to length $2,048$.
  * **Multi-Packet Temporal Grouping:** $5$ consecutive packets are grouped to encapsulate temporal context.
  * **3-Channel RGB Construction:** Channel 1 holds the forward feature sequence; Channels 2 and 3 are generated via **array reversal (flipping)**, reshaped into a 3D tensor of size $128 \times 64 \times 3$.
  * **No PCA / Feature Selection:** The paper deliberately avoids dropping payload bytes so that fine-grained byte-level exploit signatures are preserved.

### 2.5 Exact Data Partitioning Ratios (Section 3.1.3 & Figure 1)
After isolating the **3 Unknown Attack Classes** (`DoS Hulk`, `DoS slowloris`, `Bot`), the remaining data (`BENIGN` + **11 Known Attack Classes**, i.e., 12 total known classes) is partitioned into two parallel tracks plus two memory subsets:

| Partition Subset | Source Pool | Split Ratio / Composition | Purpose |
| :--- | :--- | :--- | :--- |
| **`CS-D1`** | Combined `BENIGN` + 11 Known Attack Classes | **50%** of Known Pool | Training set for Closed-Set Classifier ($\Theta_b$) |
| **`CS-D2`** | Combined `BENIGN` + 11 Known Attack Classes | **20%** of Known Pool | Validation set for Closed-Set Classifier ($\Theta_b$) |
| **`CS-D3`** | Combined `BENIGN` + 11 Known Attack Classes | **30%** of Known Pool | Test set for Closed-Set Classifier ($\Theta_b$) |
| **`OS-D1`** (Base Learner Dataset) | Separated `BENIGN` only | **50%** of `BENIGN` instances | Unsupervised K-Means clustering & Base Learner training in Open-Set Classifier |
| **`OS-D2`** (Meta Learner Dataset) | `BENIGN` + 8 randomly chosen Known Attack Classes | **30%** of `BENIGN` + **8 Known Attack Classes** | Training the Stacking Meta-Learners (`XGBoost`, `LightGBM`, `MLP`) to distinguish Benign vs. Anomaly |
| **`OS-D3`** (Evaluation Dataset) | `BENIGN` + 3 remaining Known Attack Classes | **20%** of `BENIGN` + **3 Attack Classes** (`DoS Slowhttptest`, `FTP-Patator`, `Web Attack–Sql Injection`) | Evaluating Open-Set Classifier anomaly detection before full deployment |
| **`ILD1`** (Incremental Learning Data 1) | Duplicated from Known Pool (`BENIGN` + 11 Known Attacks) | **5%** randomly duplicated per known class | Stored in **Memory Module** as exemplars to prevent catastrophic forgetting |
| **`VD1`** (Validation Data 1) | Duplicated from `CS-D3` | **10%** of `CS-D3` samples | Stored in **Memory Module** for the **Validation Module** gate to measure Forgetting Rate & Avg Incremental Accuracy |

---

## 3. AIS Conceptual Mapping (Crucial)

As noted at the top of this specification, **AIS-NIDS** in `AIS-NIDS.pdf` stands for *"An Intelligent and Self-Sustaining Network Intrusion Detection System"* rather than a biological Artificial Immune System. However, its architectural lifecycle is a direct deep-learning analogue of an adaptive immune response. Below is the exact mapping between the biological AIS concepts and the concrete mathematical structures implemented in the paper:

| Biological AIS Concept | Exact Equivalent in `AIS-NIDS.pdf` | Mathematical Representation & Tensor Shape |
| :--- | :--- | :--- |
| **Antigen (Ag)** | **Network Packet / Serialized Packet Image** | **1D Raw Vector:** $\mathbf{x} \in [0, 1]^{1500}$ (payload bytes) or $\mathbb{R}^{1504}$ (payload + 4 header attributes).<br>**2D Serialized RGB Image:** $\mathbf{I} \in \mathbb{R}^{128 \times 64 \times 3}$ constructed from zero-padded ($2048$-length), array-reversed consecutive packets. |
| **Innate / Memory Antibodies (Known Detectors)** | **Component #1: Closed-Set Classifier ($\Theta_b$) Output Neurons** | A Deep Concatenated 2D-CNN (or 1D-CNN/MLP in prototype) mapping input $\mathbf{x}$ to logits $\mathbf{z} \in \mathbb{R}^{N+1}$ and softmax probabilities $\mathbf{p} = \text{Softmax}(\mathbf{z}) \in [0,1]^{N+1}$, where class $0$ is `BENIGN` and classes $1 \dots N$ are known attacks. |
| **Self / Non-Self Discrimination (Negative Selection Analogue)** | **Component #2: Open-Set Classifier (K-Means Benign Sub-Clustering + Stacking Meta-Learners)** | **Self-Space Modeling:** `BENIGN` training data (`OS-D1`) is partitioned into $K$ sub-clusters via K-Means.<br>**Base Learners** output cluster probability vector $\mathbf{c}(\mathbf{x}) = [P(\text{Cluster}_1 \mid \mathbf{x}), \dots, P(\text{Cluster}_K \mid \mathbf{x})]^T \in [0,1]^K$.<br>**Meta-Learners** (`XGBoost`, `LightGBM`, `MLP`) take $\mathbf{c}(\mathbf{x})$ and vote `Benign` (Self) vs. `Unknown Attack` (Non-Self). *(In simplified prototype: $\max_i p_i(\mathbf{x}) < \tau \implies \text{Unknown Attack}$).* |
| **Danger Signal / Anomaly Persistence Filter** | **Component #3: Continuous Stream Rule ($L_{\text{stream}} \ge 4$)** | A sequential counter $L_{\text{stream}}$ tracking consecutive `Unknown Attack` detections. Isolated anomalies ($L_{\text{stream}} < 4$) are rejected as noise/corrupted packets; sustained streams ($L_{\text{stream}} \ge 4$) trigger immune learning. |
| **Immune Memory Cells (B-Cell Memory Pool)** | **Component #3: Memory Module (`ILD1` & `VD1`)** | **`ILD1`:** Exemplar buffer storing $5\%$ of samples from every known class $c \in \{0, \dots, N\}$.<br>**`VD1`:** Validation buffer storing $10\%$ of test samples (`CS-D3`) across all known classes. |
| **Clonal Expansion & Somatic Hypermutation** | **Component #4: Incremental Learning Module ($\Theta_b \to \Theta_u$)** | Clones model parameters $\Theta_b \to \Theta_u$, expands output layer `head` from $\mathbb{R}^{d_{\text{feat}} \times (N+1)}$ to $\mathbb{R}^{d_{\text{feat}} \times (N+2)}$, perturbs weights on the new attack class, restores backbone weights $\Theta_b^{\text{backbone}}$, and fine-tunes on $X_{\text{train}} = X_{\text{unknown}} \cup \text{ILD1}$. |
| **Autoimmune Tolerance Check (Censoring Immature Detectors)** | **Component #5: Validation Module Gate** | Evaluates candidate model $\Theta_u$ against $\Theta_b$ on `VD1`. Rejects $\Theta_u$ if any previously learned class suffers a Forgetting Rate $\ge 10\%$ or if overall Average Incremental Accuracy drops by $\ge 5\%$. |

---

## 4. Core Algorithms & Mathematical Formulations

### 4.1 Overall Operational Algorithm (Algorithm 1 in Paper)

```text
ALGORITHM 1: AIS-NIDS Operational & Autonomous Learning Pipeline
--------------------------------------------------------------------------------
Input:  Stream of raw network packets P = {p_1, p_2, ...}
Output: Prediction -> Benign or Specific Attack Class (with autonomous model updates)

Initialization:
1. ILD1 <- 5% of Known Class samples from training dataset (stored in Memory Module)
2. VD1  <- 10% of Known Class samples from testing dataset CS-D3 (stored in Memory Module)
3. Feature Space Transformation: Raw packets -> Serialized RGB Images (128x64x3) [or normalized 1500-byte vectors]
4. Load pre-trained Closed-Set Classifier (\Theta_b) and Open-Set Classifier

Runtime Loop:
5. WHILE System is Operational DO:
6.     Pass transformed packet x to Closed-Set Classifier (\Theta_b)
7.     IF \Theta_b labels x as a Specific Known Attack (Class 1..N) THEN:
8.         Initiate an alarm to prevent the identified threat
9.         Reset unknown stream counter/buffer
10.    ELSE (\Theta_b labels x as Benign):
11.        Pass x to Open-Set Classifier
12.        IF Open-Set Classifier labels x as Benign THEN:
13.            No threat detected; system is under normal conditions
14.            Reset unknown stream counter/buffer
15.        ELSE (Open-Set Classifier flags x as Unknown Attack):
16.            Append x to continuous Unknown Attack Stream Buffer S_unk
17.            IF |S_unk| < 4 continuous samples when stream ends THEN:
18.                Discard S_unk and raise alert for bad/corrupted data instances
19.            ELSE (|S_unk| >= 4 continuous samples):
20.                Generate alerts and continue buffering until stream of unknown attacks ends
21.                Construct X_train <- S_unk U ILD1 (under-sampling S_unk if needed to match class ratio)
22.                Pass X_train and \Theta_b to Incremental Learning Module
23.                Produce candidate model \Theta_u via 4-Step Incremental Learning Approach
24.                Pass \Theta_u to Validation Module
25.                Evaluate \Theta_b and \Theta_u on VD1 to compute Forgetting Rate (FR_c) and Avg Incremental Accuracy
26.                IF (max_c FR_c < 10%) AND (Delta Avg Incremental Accuracy < 5%) THEN:
27.                    Commit update: \Theta_b <- \Theta_u with new attack class X_{N+1}
28.                    Include proportional samples of X_{N+1} into ILD1 and VD1
29.                ELSE:
30.                    Discard candidate model \Theta_u and buffered samples S_unk
```

---

### 4.2 Component #1: Deep Concatenated 2D-CNN (Closed-Set Classifier $\Theta_b$)
As depicted in **Figure 3** of the paper, the Closed-Set Classifier consists of **3 DenseNet-style Convolutional Blocks** with input-output concatenation followed by **Block #4 (Fully Connected Classification Head)**:

* **Blocks #1, #2, and #3 (Feature Extraction with Skip Concatenation):**
  For each block $b \in \{1, 2, 3\}$ receiving input tensor $\mathbf{H}_{b-1}$ (where $\mathbf{H}_0 = \mathbf{I} \in \mathbb{R}^{128 \times 64 \times 3}$):
  $$\mathbf{U}_b = \text{Conv2D}\Big(\sigma\big(\text{BatchNorm}(\mathbf{H}_{b-1})\big)\Big)$$
  $$\mathbf{V}_b = \text{Conv2D}\Big(\sigma\big(\text{BatchNorm}(\mathbf{U}_b)\big)\Big)$$
  $$\mathbf{H}_b = \text{Concat}\big(\mathbf{H}_{b-1}, \mathbf{V}_b\big)$$
  where $\sigma(\cdot)$ is the non-linear activation function (e.g., ReLU) and $\text{Concat}(\cdot)$ concatenates feature maps along the channel axis.
* **Block #4 (Classification Head):**
  $$\mathbf{f}_0 = \text{Flatten}(\mathbf{H}_3)$$
  $$\mathbf{f}_1 = \text{Dropout}\big(\sigma(\mathbf{W}_1 \mathbf{f}_0 + \mathbf{b}_1)\big)$$
  $$\mathbf{f}_2 = \text{Dropout}\big(\sigma(\mathbf{W}_2 \mathbf{f}_1 + \mathbf{b}_2)\big)$$
  $$\mathbf{z} = \mathbf{W}_{\text{head}} \mathbf{f}_2 + \mathbf{b}_{\text{head}} \in \mathbb{R}^{C}, \quad \hat{\mathbf{p}} = \text{Softmax}(\mathbf{z})$$
  where $C = N + 1$ is the total number of known classes ($N$ attack classes + $1$ `BENIGN` class), and $\hat{y} = \arg\max_{c \in \{0, \dots, N\}} \hat{p}_c$.

*(Note for CPU / 1D-Vector Implementation in `skeleton.py` & `plan.md`: When running on the 1D 1,500-byte vector $\tilde{\mathbf{x}} \in [0,1]^{1500}$, the backbone is modularized as either an MLP `1500 -> 512 -> 256 -> head(C)` with ReLU activations or a 2-block `1D-CNN` with Conv1D + MaxPool1D + Flatten + `head(C)`, trained with `CrossEntropyLoss` and `Adam`)*:
$$\mathcal{L}_{\text{CE}}(\Theta_b) = -\frac{1}{B} \sum_{m=1}^{B} \sum_{c=0}^{N} y_{m,c} \log \hat{p}_{m,c}$$

---

### 4.3 Component #2: Open-Set Classifier (K-Means Sub-Clustering + Stacking)
As depicted in **Figure 4** of the paper, the Open-Set Classifier detects unknown attacks hidden among samples predicted as `BENIGN` by $\Theta_b$:

1. **Unsupervised Benign Sub-Clustering (on `OS-D1`):**
   * Normal traffic does not follow a single uniform distribution. K-Means partitions the `OS-D1` benign dataset into $K$ distinct clusters $\{\mathcal{C}_1, \mathcal{C}_2, \dots, \mathcal{C}_K\}$ by minimizing within-cluster Euclidean sum-of-squares:
     $$\min_{\{\boldsymbol{\mu}_k\}_{k=1}^K} \sum_{k=1}^{K} \sum_{\mathbf{x} \in \mathcal{C}_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|_2^2$$
2. **Base Learners & Probability Aggregator:**
   * Base learners trained on the cluster assignments output the posterior probability $P(\text{Benign\_Cluster\_}k \mid \mathbf{x})$ for each cluster $k \in \{1, \dots, K\}$.
   * The **Probability Aggregator** forms the $K$-dimensional meta-feature vector:
     $$\mathbf{m}(\mathbf{x}) = \big[P(\text{Benign\_Cluster\_}1 \mid \mathbf{x}), P(\text{Benign\_Cluster\_}2 \mid \mathbf{x}), \dots, P(\text{Benign\_Cluster\_}K \mid \mathbf{x})\big]^T \in [0, 1]^K$$
3. **Meta-Learners & Ensemble Voting (trained on `OS-D2`):**
   * Using `OS-D2` (which contains 30% `BENIGN` labeled as $0$ and 8 known attack classes labeled as $1$ / Anomaly), three meta-learners—**`XGBoost`** ($M_1$), **`LightGBM`** ($M_2$), and an **MLP / Neural Network** ($M_3$)—are trained on $\mathbf{m}(\mathbf{x})$.
   * At runtime, the final binary open-set decision $d_{\text{OS}}(\mathbf{x}) \in \{\text{Benign}, \text{Unknown Attack}\}$ is produced by majority voting across $\{M_1, M_2, M_3\}$:
     $$d_{\text{OS}}(\mathbf{x}) = \text{Vote}\big(M_1(\mathbf{m}(\mathbf{x})), M_2(\mathbf{m}(\mathbf{x})), M_3(\mathbf{m}(\mathbf{x}))\big)$$
   *(Alternative Single-Threshold Open-Set Detector in `plan.md`): Flag $\mathbf{x}$ as `Unknown Attack` if maximum softmax confidence $\max_{c} \hat{p}_c(\mathbf{x}) < \tau$, where threshold $\tau$ is calibrated at the 90th/95th/99th percentile on known-class validation data).*

---

### 4.4 Component #4: 4-Step Class-Incremental Learning Algorithm
Let $\Theta_b = \{\Theta_b^{\text{backbone}}, \mathbf{W}_{\text{head}}^{(b)}, \mathbf{b}_{\text{head}}^{(b)}\}$ denote the current Closed-Set Classifier trained on $C$ classes (where $\mathbf{W}_{\text{head}}^{(b)} \in \mathbb{R}^{C \times d_{\text{feat}}}$, $\mathbf{b}_{\text{head}}^{(b)} \in \mathbb{R}^{C}$). When a validated stream of a new unknown attack class $X_{\text{new}}$ arrives:

1. **Step 1 — Initialization (Neuron Expansion $\Theta_b \to \Theta_u$):**
   Create replica model $\Theta_u$ with $C + 1$ output nodes:
   $$\Theta_u^{\text{backbone}} \leftarrow \text{copy}\big(\Theta_b^{\text{backbone}}\big)$$
   $$\mathbf{W}_{\text{head}}^{(u)}[0:C, :] \leftarrow \mathbf{W}_{\text{head}}^{(b)}, \quad \mathbf{b}_{\text{head}}^{(u)}[0:C] \leftarrow \mathbf{b}_{\text{head}}^{(b)}$$
   $$\mathbf{W}_{\text{head}}^{(u)}[C, :] \sim \mathcal{U}\left(-\frac{1}{\sqrt{d_{\text{feat}}}}, \frac{1}{\sqrt{d_{\text{feat}}}}\right), \quad \mathbf{b}_{\text{head}}^{(u)}[C] \sim \mathcal{U}\left(-\frac{1}{\sqrt{d_{\text{feat}}}}, \frac{1}{\sqrt{d_{\text{feat}}}}\right)$$
2. **Step 2 — Learning New Class (Initial Perturbation Step):**
   Train $\Theta_u$ exclusively on the newly collected unknown attack instances $S_{\text{unk}}$ (with target label $y = C$):
   $$\tilde{\Theta}_u \leftarrow \text{Train}\big(\Theta_u, \text{Data} = S_{\text{unk}}, \text{Labels} = C\big)$$
   *(Note: Training solely on class $C$ forces strong gradient updates into the newly added output neuron $\mathbf{W}_{\text{head}}^{(u)}[C, :]$, but severely perturbs $\tilde{\Theta}_u^{\text{backbone}}$ and degrades old classes).*
3. **Step 3 — Weight Restoration (Backbone Reset):**
   Restore all hidden/backbone weights of $\tilde{\Theta}_u$ back to the original weights of $\Theta_b$, while **retaining** the updated last-layer (`head`) weights from Step 2:
   $$\Theta_u^{\text{backbone}} \leftarrow \text{copy}\big(\Theta_b^{\text{backbone}}\big)$$
   $$\mathbf{W}_{\text{head}}^{(u)} \leftarrow \tilde{\mathbf{W}}_{\text{head}}^{(u)}, \quad \mathbf{b}_{\text{head}}^{(u)} \leftarrow \tilde{\mathbf{b}}_{\text{head}}^{(u)}$$
4. **Step 4 — Iterative Refinement (Exemplar Replay Fine-Tuning):**
   Combine the new attack samples $S_{\text{unk}}$ (under-sampled if necessary to match exemplar proportions) with the memory module exemplars `ILD1`:
   $$X_{\text{train}} = S_{\text{unk}} \cup \text{ILD1}$$
   Fine-tune $\Theta_u$ over several iterations on $X_{\text{train}}$ across all $C + 1$ classes to align the decision boundaries between the $C$ old classes and the $(C+1)$-th new class.

---

### 4.5 Component #5: Validation Module Mathematical Formulas
The Validation Module evaluates both the current model $\Theta_b$ and the candidate updated model $\Theta_u$ on the validation memory set `VD1`. Let $\text{Acc}(\Theta, c)$ be the classification accuracy (recall) of model $\Theta$ on samples belonging to class $c$:
$$\text{Acc}(\Theta, c) = \frac{\text{TP}_c(\Theta)}{\text{TP}_c(\Theta) + \text{FN}_c(\Theta)} \times 100\%$$

1. **Per-Class Forgetting Rate ($FR_c$):**
   For each previously learned class $c \in \{0, 1, \dots, C-1\}$:
   $$FR_c = \text{Acc}(\Theta_b, c) - \text{Acc}(\Theta_u, c)$$
   *(In Tables 2–5 of the paper, if accuracy remains unchanged, $FR_c = 0\%$; it quantifies the accuracy drop on pre-trained classes).*
2. **Average Incremental Accuracy ($\text{AvgAcc}$):**
   After learning $k$ new classes on top of $N$ base attack classes (plus `BENIGN`, so $C_{\text{total}} = N + 1 + k$ total classes):
   $$\text{AvgAcc}(\Theta_u) = \frac{1}{C_{\text{total}}} \sum_{c=0}^{C_{\text{total}}-1} \text{Acc}(\Theta_u, c)$$
   And the **Average Incremental Accuracy Drop ($\Delta \text{AvgAcc}$)** between the current model $\Theta_b$ and the updated model $\Theta_u$ is:
   $$\Delta \text{AvgAcc} = \text{AvgAcc}(\Theta_b) - \text{AvgAcc}(\Theta_u)$$
3. **Autonomous Acceptance Criterion:**
   $$\text{Decision}(\Theta_u) = \begin{cases} \text{ACCEPT}, & \text{if } \left(\max_{c \in \{0, \dots, C-1\}} FR_c < 10\%\right) \land \left(\Delta \text{AvgAcc} < 5\%\right) \\ \text{REJECT}, & \text{otherwise} \end{cases}$$
   *(Note on Algorithm 1 Line 22: The pseudocode shorthand `Average Incremental Accuracy < 5%` is explicitly defined in Section 3.2.5, page 9 as: "we set a threshold value for Avg Acc, ensuring it does not exceed 5% between the updated and current models", i.e., $\Delta \text{AvgAcc} \le 5\%$).*

---

## 5. Hyperparameters, Variables & Thresholds

The table below consolidates every explicit hyperparameter, dimension, split ratio, and threshold from `AIS-NIDS.pdf` alongside the concrete implementation defaults in `skeleton.py` and `plan.md`:

| Category | Variable / Hyperparameter | Exact Value in `AIS-NIDS.pdf` | Value in Repo Prototype (`skeleton.py` / `plan.md`) | Description / Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Packet Feature Space** | Max Payload Length ($N_{\text{bytes}}$) | `1500` bytes | `1500` bytes (`uint8`) | Zero-padded if packet payload $< 1500$ bytes |
| **Packet Feature Space** | Byte Integer Range | $[0, 255]$ | $[0, 255] \to [0.0, 1.0]$ | Normalized by dividing by `255.0` |
| **Packet Feature Space** | Total Raw Features per Packet | `1504` | `1500` payload + `4` meta (`ttl`, `total_len`, `protocol`, `t_delta`) | Payload bytes + packet header metadata |
| **Image Transformation** | Zero-Padded Feature Length | `2048` | Optional (2D-CNN extension) | Padded at both ends from `1504` to `2048` |
| **Image Transformation** | Consecutive Packets per Image | `5` packets | `1` packet per row in `capped_sample.pkl` | Encapsulates temporal information |
| **Image Transformation** | Serialized RGB Image Shape | $128 \times 64 \times 3$ | $(B, 1500)$ for MLP/1D-CNN | 3 channels via array reversal (flipping) |
| **Dataset Partitioning** | Total Classes (CIC-IDS2017) | `15` (`1` Benign + `14` Attacks) | `15` in `capped_sample.pkl` (`116,726` rows total) | `Web Attack – Sql Injection` has 12 rows (dropped in `skeleton.py`) |
| **Dataset Partitioning** | Base Known Classes ($N_{\text{known}}$) | `12` (`BENIGN` + `11` Attacks) | `11` (`BENIGN` + `10` Attacks) | Classes known to initial Closed-Set Classifier $\Theta_b$ |
| **Dataset Partitioning** | Held-Out Unknown Classes | `3` (`DoS Hulk`, `Bot`, `DoS slowloris`) | `3` (`DoS Hulk`: 10,000; `Bot`: 2,543; `DoS slowloris`: 10,000) | Excluded from initial training to simulate zero-day attacks |
| **Dataset Partitioning** | Closed-Set Split (`CS-D1:CS-D2:CS-D3`) | `50% : 20% : 30%` | `70% : 15% : 15%` (stratified, `SEED = 42`) | Train : Validation : Test split for Closed-Set Classifier |
| **Dataset Partitioning** | Open-Set Benign Split (`OS-D1:OS-D2:OS-D3`) | `50% : 30% : 20%` of `BENIGN` | — | `OS-D2` adds 8 known attacks; `OS-D3` adds 3 known attacks |
| **Memory Module** | Exemplar Buffer Ratio (`ILD1`) | `5%` of each known class | `5%` of each known class | Replayed during Incremental Learning Step 4 |
| **Memory Module** | Validation Buffer Ratio (`VD1`) | `10%` of `CS-D3` | `10%` of test/val split | Used by Validation Module to compute $FR_c$ and $\Delta \text{AvgAcc}$ |
| **Stream Rule** | Minimum Continuous Stream Length ($L_{\min}$) | `4` continuous samples | `4` continuous samples | Streams $< 4$ unknown samples are discarded as bad data/outliers |
| **Unknown Evaluation** | Injected Unknown Sample Size | `5000` samples (`DoS Hulk`) | ~60% stream / 40% holdout test | Streamed into system during unknown-class evaluation |
| **Validation Gate** | Max Per-Class Forgetting Rate ($FR_{\max}$) | `< 10%` (`0.10`) per class | `< 10%` (`0.10`) per class | Maximum allowed accuracy drop on any single known class |
| **Validation Gate** | Max Average Incremental Accuracy Drop ($\Delta \text{AvgAcc}_{\max}$) | `< 5%` (`0.05`) | `< 5%` (`0.05`) | Maximum allowed drop in mean accuracy across all learned classes |
| **Training Config** | Batch Size / Optimizer / LR / Epochs | Deep 2D-CNN (per `Farrukh et al., 2023b`) | `BATCH_SIZE = 256`, `Adam`, `LR = 1e-3`, `EPOCHS = 10` | Standard training hyperparameters in `skeleton.py` |

---

## 6. Detector Generation (Training) vs. Detection (Testing) Phases

### 6.1 Phase 1: Initial Offline Training (Pre-Deployment)
1. **Data Segregation:**
   * Hold out the 3 unknown attack classes (`DoS Hulk`, `Bot`, `DoS slowloris`) completely.
2. **Closed-Set Classifier ($\Theta_b$) Training:**
   * Train the multi-class Closed-Set Classifier on `CS-D1` (50% of known data across `BENIGN` + known attack classes) and validate on `CS-D2` (20%).
   * Evaluate baseline closed-set accuracy on `CS-D3` (30%). In the paper (Table 1 & Fig. 6), $\Theta_b$ achieves **83% overall accuracy** and **85% F1-score** on CIC-IDS2017 known classes (100% on `BENIGN`, `FTP-Patator`, `Heartbleed`, `Infiltration`, `SSH-Patator`, `Brute Force`, `XSS`; 99% on `DDoS`; 66–69% on `DoS GoldenEye`, `DoS Slowhttptest`, `PortScan`, `Sql Injection`).
3. **Open-Set Classifier Training:**
   * Fit unsupervised **K-Means** on `OS-D1` (50% of `BENIGN` data) to partition normal traffic into sub-clusters, and train the **Base Learners** to predict cluster membership probabilities.
   * Generate probability vectors on `OS-D2` (30% `BENIGN` + 8 known attack classes) via the **Probability Aggregator** and train the **Meta-Learners** (`XGBoost`, `LightGBM`, `MLP`) to classify `Benign` vs. `Unknown Attack`.
   * Verify open-set anomaly detection on `OS-D3` (20% `BENIGN` + 3 held-out known attacks).
4. **Memory Module Initialization:**
   * Populate `ILD1` with 5% randomly sampled exemplars per known class.
   * Populate `VD1` with 10% randomly sampled instances per known class from `CS-D3`.

---

### 6.2 Phase 2: Runtime Detection & Autonomous Incremental Adaptation (Testing / Deployment)
During live or streaming evaluation, packets arrive sequentially:

1. **Two-Stage Hierarchical Filtering:**
   * **Stage 1 (Closed-Set $\Theta_b$):** Every packet is evaluated by $\Theta_b$. If predicted as a known attack ($c \in \{1, \dots, N\}$), an immediate alarm is raised.
   * **Stage 2 (Open-Set Classifier):** Only packets predicted as `BENIGN` ($c = 0$) by $\Theta_b$ are passed to the Open-Set Classifier. If the Open-Set Classifier confirms `Benign`, traffic passes normally.
2. **Stream Buffering & Noise Rejection:**
   * If the Open-Set Classifier predicts `Unknown Attack`, the sample is appended to the stream buffer $S_{\text{unk}}$.
   * By the **Continuous Stream Rule**, if the contiguous sequence ends with $|S_{\text{unk}}| < 4$, the buffer is cleared and logged as a bad data instance.
   * If $|S_{\text{unk}}| \ge 4$, the system continues collecting until the unknown attack stream ends (in the paper's experiments, 89% of `DoS Hulk`, 90% of `Bot`, and 84% of `DoS slowloris` packets were flagged as `Unknown` and routed to the Memory Module).
3. **Autonomous Incremental Learning & Validation Gate Execution:**
   * Once the stream concludes, the Memory Module pairs the newly collected unknown samples (assigned temporary label $X_1$, $X_2$, $X_3$, etc.) with `ILD1` exemplars to form $X_{\text{train}}$.
   * The **Incremental Learning Module** executes the 4-step update ($\Theta_b \to \Theta_u$: Neuron Expansion $\to$ Initial Perturbation on $S_{\text{unk}}$ $\to$ Backbone Weight Restoration $\to$ Iterative Refinement on $X_{\text{train}}$).
   * The **Validation Module** tests $\Theta_u$ vs. $\Theta_b$ on `VD1`:
     * **True New Attacks Accepted (Tables 2–4):**
       * *Iteration 1 ($N+1$, `X1 - DoS Hulk`):* Learns `DoS Hulk` at 88% accuracy; max $FR_c = 9\% < 10\%$ (on `DoS GoldenEye` and `DoS Slowhttptest`), $\text{AvgAcc}$ drops from $83\% \to 80\%$ ($\Delta \text{AvgAcc} = 3\% < 5\%$) $\implies$ **ACCEPTED**.
       * *Iteration 2 ($N+2$, `X2 - Bot`):* Learns `Bot` at 87% accuracy; max $FR_c = 5\% < 10\%$, $\text{AvgAcc}$ drops from $80\% \to 78\%$ ($\Delta \text{AvgAcc} = 2\% < 5\%$) $\implies$ **ACCEPTED**.
       * *Iteration 3 ($N+3$, `X3 - DoS slowloris`):* Learns `DoS slowloris` at 78% accuracy; max $FR_c = 7\% < 10\%$, $\text{AvgAcc}$ drops from $78\% \to 74\%$ ($\Delta \text{AvgAcc} = 4\% < 5\%$) $\implies$ **ACCEPTED**.
     * **False Alarm / Benign Negative Test Rejected (Table 5):**
       * When random `BENIGN` samples are falsely injected as a pseudo-unknown attack stream ($N+4$), `BENIGN` accuracy collapses from $100\% \to 70\%$ ($FR_{\text{BENIGN}} = 30\% \ge 10\%$) and $\text{AvgAcc}$ drops from $74\% \to 60\%$ ($\Delta \text{AvgAcc} = 14\% \ge 5\%$) $\implies$ **REJECTED** (model $\Theta_u$ and false stream are automatically discarded).
