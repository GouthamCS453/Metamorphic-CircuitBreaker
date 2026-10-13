# Protocol Deviations & Design Specifications Report (v4 Protocol)

## Domain: Autonomous Vehicles — GTSRB Traffic Sign Perception
- **Backbone:** MobileNetV3-Small (`mobilenetv3_small_100`)
- **Checkpoint:** `checkpoints/best_model.pth` (18.11 MB, 43 classes)
- **Held-Out Pool:** `checkpoints/val_split.csv` ($N = 11,763$)

---

### 1. Checkpoint Incident & Resolution
- **Initial Status:** Checkpoint was absent from working tree due to `.gitignore` exclusion of `checkpoints/*.pth`.
- **Resolution:** Recovered identical original checkpoint from `C:\Users\Aparna\Metamorphic-CircuitBreaker\checkpoints\best_model.pth`.
- **Integrity Verification:** Matches training logs (validation accuracy 95.94%, loss 0.8546) and companion split SHA-256 hashes. No retraining or weight alteration was performed.

---

### 2. Domain Metamorphic Invariance Design
#### Horizontal Flip Replacement (Section 3.2 & 3.3)
- **Rule:** Directional domains must not use horizontal reflection because reflection alters semantic labels (e.g., *Turn Left* $\leftrightarrow$ *Turn Right*, *Keep Left* $\leftrightarrow$ *Keep Right*).
- **Implementation:** Horizontal flip is disabled across GTSRB and substituted with horizontal affine shear (`shear_x=0.05`, `single` tier, weight 1.0).

---

### 3. Dataset Split & Group Identifier Logic (Section 4.1)
#### Group Identifiers
- **Dataset Property:** GTSRB split metadata in `checkpoints/val_split.csv` contains columns `['image', 'image_path', 'label']` and does not provide track identifiers.
- **Protocol Action:** In accordance with Section 4.1 fallback rule ("otherwise `group_id = sample_id` and log 'no group ids'"), `group_id = sample_id` is applied. Stratified splitting on `y_true_correct` is performed.

#### Checkpoint Selection
- **Status:** The trained checkpoint was selected on this validation split. Per Section 4.1, this is explicitly documented: "checkpoint selected on evaluation split".

---

### 4. Safety-Critical Error Definition (Section 4.2)
- **Definition:** For autonomous vehicle traffic sign recognition, safety-critical errors represent ground-truth life-safety or priority signs that the model misclassifies:
  - Class 13: Yield
  - Class 14: Stop
  - Class 17: No entry
  - Class 26: Traffic signals
  - Class 27: Pedestrians
  - Class 28: Children crossing
- **Operational Rule:**
  `critical_error = 1` if `y_true in {13, 14, 17, 26, 27, 28}` and `correct == 0`, else `0`.

---

### 5. Baselines Specification (Section 5)
- **MC-Dropout:** MobileNetV3-Small does not include dropout in its inverted bottleneck blocks. In accordance with Section 5, MC-Dropout is implemented **head-only** ($p=0.2, T=10$) on pooled penultimate features on the same device (CPU) as MCB.
- **Temperature Scaling:** Fitted exclusively on Partition A validation logits using bounded NLL optimization.

---

### 6. Hardware & Execution Device (Section 9.1 & 12)
- **Device:** Intel CPU (all baselines, MCB, MC-Dropout, and latency benchmarks evaluated on the exact same CPU device; no mixed CPU/GPU comparisons).

---

### 7. Calibration Grid Boundary Extension (Section 6.2)
- **Status:** During Step 6 calibration on Partition A, the optimum for `op_j` initially selected `tau_fam = 0.10`, which was on the lower boundary of `TAU = [0.10, 0.15, ...]`.
- **Protocol Action:** In strict accordance with Section 6.2 ("Boundary check: if the optimum is at the edge of the grid, extend that edge once by one step and note it in `deviations.md`"), the grid was extended once by one step to include `tau_fam = 0.05`.
- **Outcome:** The final calibrated threshold for `op_j` is `tau_fam = 0.05, theta_trip = 0.30` (Youden J = 0.7314, Recall = 0.8560, FAR = 0.1246); `op_far10` selected `tau_fam = 0.10, theta_trip = 0.25` (Recall = 0.8054, FAR = 0.0882).
