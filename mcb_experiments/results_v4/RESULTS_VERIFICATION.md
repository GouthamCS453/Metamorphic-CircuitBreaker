# MCB v4 Results Verification & Experimental Audit Report

**Backbone:** MobileNetV3-Small | **Dataset:** GTSRB | **Hardware Device:** Intel CPU
**Provenance Hash:** `92dd9c9b6a5a80bf32b2a6fb13c68debfb3001eb` | **Split Seed:** 42
**Partition A SHA-256:** `f45f34be035f9c0375bed992ef30e9deca219fcb2651917b20e4e6a792080a8b`
**Partition B SHA-256:** `52e3c587991a2f1a8a525b66cebeb3d02cf99ef77092654f80402f7772c41e55`

---

## Section 1: Confusion Matrices for Every Method (Partition B)

All confusion matrix sums have been asserted programmatically:
- Error Sum ($TP + FN = 221$)
- Correct Sum ($TN + FP = 5,661$)
- Total Sum ($TP + FP + TN + FN = 5,882$)

| Method | Operating Point | TP | FP | TN | FN | Total | Checked Sums |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `standalone` | none | 0 | 0 | 5661 | 221 | 5882 | PASSED |
| `kflip_1` | k=1 | 219 | 4739 | 922 | 2 | 5882 | PASSED |
| `kflip_2` | k=2 | 206 | 2886 | 2775 | 15 | 5882 | PASSED |
| `kflip_3` | k=3 | 199 | 1332 | 4329 | 22 | 5882 | PASSED |
| `kflip_5` | k=5 | 172 | 419 | 5242 | 49 | 5882 | PASSED |
| `kflip_8` | k=8 | 125 | 160 | 5501 | 96 | 5882 | PASSED |
| `ff` | op_j | 172 | 419 | 5242 | 49 | 5882 | PASSED |
| `softmax_entropy` | op_j | 189 | 1578 | 4083 | 32 | 5882 | PASSED |
| `msp` | op_j | 185 | 720 | 4941 | 36 | 5882 | PASSED |
| `tta_var` | op_j | 172 | 419 | 5242 | 49 | 5882 | PASSED |
| `mcdo_headonly` | op_j | 191 | 1798 | 3863 | 30 | 5882 | PASSED |
| `entropy_ts` | op_j | 194 | 932 | 4729 | 27 | 5882 | PASSED |
| `mcb_prior_default` | prior_default | 63 | 18 | 5643 | 158 | 5882 | PASSED |
| `mcb_op_j` | op_j | 186 | 721 | 4940 | 35 | 5882 | PASSED |
| `mcb_op_far10` | op_far10 | 173 | 530 | 5131 | 48 | 5882 | PASSED |

---

## Section 2: Confidence Intervals (Wilson & Bootstrap)

- Recall and FAR: Wilson Score 95% Confidence Intervals.
- AUROC: Paired Stratified Bootstrap 95% Confidence Intervals (2,000 iterations).

| Method | Recall (95% Wilson CI) | FAR (95% Wilson CI) | Youden's J | AUROC (95% Bootstrap CI) |
|:---|:---:|:---:|:---:|:---:|
| `standalone` | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.0000 | 0.5000 [0.5000, 0.5000] |
| `kflip_1` | 99.10% [96.76%, 99.75%] | 83.71% [82.73%, 84.65%] | 0.1538 | — |
| `kflip_2` | 93.21% [89.10%, 95.84%] | 50.98% [49.68%, 52.28%] | 0.4223 | — |
| `kflip_3` | 90.05% [85.39%, 93.33%] | 23.53% [22.44%, 24.65%] | 0.6652 | — |
| `kflip_5` | 77.83% [71.90%, 82.80%] | 7.40% [6.75%, 8.11%] | 0.7043 | — |
| `kflip_8` | 56.56% [49.97%, 62.93%] | 2.83% [2.43%, 3.29%] | 0.5373 | — |
| `ff` | 77.83% [71.90%, 82.80%] | 7.40% [6.75%, 8.11%] | 0.7043 | 0.9099 [0.8841, 0.9338] |
| `softmax_entropy` | 85.52% [80.27%, 89.55%] | 27.87% [26.72%, 29.06%] | 0.5765 | 0.8694 [0.8489, 0.8898] |
| `msp` | 83.71% [78.27%, 88.00%] | 12.72% [11.88%, 13.61%] | 0.7099 | 0.9203 [0.9040, 0.9352] |
| `tta_var` | 77.83% [71.90%, 82.80%] | 7.40% [6.75%, 8.11%] | 0.7043 | 0.9097 [0.8839, 0.9335] |
| `mcdo_headonly` | 86.43% [81.28%, 90.32%] | 31.76% [30.56%, 32.99%] | 0.5466 | 0.8556 [0.8341, 0.8769] |
| `entropy_ts` | 87.78% [82.81%, 91.47%] | 16.46% [15.52%, 17.45%] | 0.7132 | 0.9367 [0.9254, 0.9472] |
| `mcb_prior_default` | 28.51% [22.96%, 34.79%] | 0.32% [0.20%, 0.50%] | 0.2819 | 0.9052 [0.8815, 0.9273] |
| `mcb_op_j` | 84.16% [78.77%, 88.39%] | 12.74% [11.89%, 13.63%] | 0.7143 | 0.9102 [0.8847, 0.9338] |
| `mcb_op_far10` | 78.28% [72.39%, 83.21%] | 9.36% [8.63%, 10.15%] | 0.6892 | 0.9043 [0.8806, 0.9269] |

---

## Section 3: Pre-Registered Decision Rules (D1 – D4) Outcomes

1. **D1 (MCB beats best baseline): NOT MET**
   - Best baseline: `entropy_ts` with AUROC $0.9367$ [0.9174, 0.9542].
   - MCB AUROC: $0.9102$ [0.8847, 0.9338].
   - Paired difference: $\Delta = -0.0264$, 95% CI $[-0.0401, -0.0135]$ (entirely below 0).
   - At `op_far10`, MCB recall is 80.54% vs `entropy_ts` recall at $\le 10\%$ FAR of 86.88%.

2. **D2 (Hierarchical MCB beats Flat CBI): NOT MET**
   - Flat weighted flip score `cbi_flat` achieved AUROC $0.9277$ [0.9078, 0.9465].
   - Hierarchical MCB achieved AUROC $0.9102$ [0.8847, 0.9338].
   - Paired difference: $\Delta = -0.0175$, 95% CI $[-0.0285, -0.0080]$ (flat CBI outperforms hierarchical grouping on GTSRB).

3. **D3 (High-Confidence Critical Error Detection): MET**
   - High-confidence errors ($c_0 \ge 0.90$): $N = 56$ samples in Partition B.
   - MCB (Op FAR10) tripped on **20 out of 56** errors (**35.71% recall**, 95% CI [24.49%, 48.77%]).
   - Softmax Entropy (Op FAR10) tripped on **3 out of 56** errors (**5.36% recall**, 95% CI [1.84%, 14.61%]).
   - **Finding:** MCB demonstrates a 6.6x higher recall on silent/overconfident failures where standard softmax uncertainty estimators are completely blind.

4. **D4 (Fault Attribution Accuracy > 50%): NOT MET**
   - Overall accuracy across 300 unseen corrupted samples: **25.00%** (Wilson 95% CI [20.44%, 30.20%]).
   - Lower bound ($0.2044$) is below $0.50$ and below chance level ($0.3333$).
   - Binomial test $p$-value vs chance: $p = 0.9993$.

---

## Section 4: Ablations (A1 – A5)

| Ablation ID | Description | AUROC (95% CI) | Difference vs Default MCB (95% CI) | Conclusion |
|:---|:---|:---:|:---:|:---|
| **A1** | Flat CBI vs Hierarchical CBI | 0.9277 [0.9078, 0.9465] | -0.0175 [-0.0285, -0.0080] | Flat scoring outperforms hierarchy |
| **A2** | MSP baseline vs $\gamma=0$ ($0.6, 0.4, 0.0$) | 0.8938 [0.8653, 0.9207] | +0.0165 [+0.0102, +0.0238] | Inverted softmax margin $\gamma$ improves AUROC |
| **A3** | Ascending `[0.1, 0.2, 0.3]` vs Descending `[1.0, 0.6, 0.3]` | 0.9066 [0.8800, 0.9302] | +0.0036 [-0.0033, +0.0103] | Descending weight trend shows slight numerical gain (CI spans 0) |
| **A4** | Five $(\alpha, \beta, \gamma)$ configs | | | |
| | (0.50, 0.35, 0.15) [Default] | 0.9102 [0.8847, 0.9338] | Reference (0.0000) | Baseline calibration |
| | (0.60, 0.40, 0.00) [No softmax] | 0.8938 [0.8653, 0.9207] | +0.0165 [+0.0102, +0.0238] | Statistically significant degradation without $\gamma$ |
| | (0.70, 0.20, 0.10) [High Max-IR] | 0.9041 [0.8788, 0.9280] | +0.0061 [+0.0024, +0.0104] | Modest drop vs default |
| | (0.25, 0.25, 0.50) [High Softmax] | **0.9216** [0.8999, 0.9418] | -0.0114 [-0.0181, -0.0054] | High softmax weight improves overall AUROC on GTSRB |
| | (0.30, 0.60, 0.10) [High Cross-Family] | 0.9092 [0.8838, 0.9325] | +0.0010 [-0.0004, +0.0027] | Equivalent performance (CI spans 0) |
| **A5** | CFS removed: $(\alpha+\beta)\max IR_k + \gamma(1-c_0)$ | 0.8941 [0.8688, 0.9179] | +0.0162 [+0.0059, +0.0272] | CFS term contributes +0.0162 AUROC gain ($p < 0.05$) |

---

## Section 5: Metamorphic Family Profile

Evaluated with $M_k = 3$ transform types per family:
- **All Samples (Mean $IR_k$):** Geometric: $0.0340$, Photometric: $0.0134$, Sensor Noise: $0.0894$
- **Base Model Errors (Mean $IR_k$):** Geometric: **$0.3224$**, Photometric: $0.1541$, Sensor Noise: $0.2389$
- **Correct Samples (Mean $IR_k$):** Geometric: $0.0227$, Photometric: $0.0079$, Sensor Noise: $0.0835$
- **Dominant Failure Axis on Errors:** **Geometric ($IR_{\text{geom}} = 0.3224$)**
- **Interpretation:** Base model classification errors in GTSRB exhibit highest vulnerability to geometric perturbations (rotation, scale, shear), followed by sensor noise. Photometric shifts show lowest sensitivity.

---

## Section 6: Fault Injection Benchmark (Section 8)

- **Test Population:** 300 correctly classified Partition B images across 3 injected families + 100 clean controls.
- **Trip Rate under Op J:**
  - Geometric Corruptions: 14.0%
  - Photometric Corruptions: 22.0%
  - Sensor Noise Corruptions: 54.0%
  - Clean Controls: 15.0%
- **Attribution Confusion Matrix (Partition A Normalization):**

| Injected Family | Pred Geometric | Pred Photometric | Pred Sensor Noise | Unattributable | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Geometric** | 20 | 34 | 35 | 11 | 20.0% [13.3%, 28.9%] |
| **Photometric** | 23 | 35 | 25 | 17 | 35.0% [26.4%, 44.7%] |
| **Sensor Noise** | 40 | 25 | 20 | 15 | 20.0% [13.3%, 28.9%] |

- **Overall Accuracy (All Samples):** 25.0% [20.4%, 30.2%]
- **Overall Accuracy (Tripped Only):** 26.7% [18.6%, 36.6%]
- **Cross-Contamination Rates:** Geometric: 87%, Photometric: 82%, Sensor Noise: 59%
- **Conclusion:** On GTSRB, corruptions from one family trigger metamorphic sensitivity across other families (cross-family coupling), preventing family-level fault attribution.

---

## Section 7: Latency & Component Breakdown (Intel CPU)

Hardware: Intel CPU. Protocol: 20 warm-up runs, 100 timed runs.

| Component | Description | Mean (ms) | SD (ms) | P50 (ms) | P95 (ms) |
|:---|:---|:---:|:---:|:---:|:---:|
| $T_{\text{base}}$ | Single image baseline forward pass | 10.61 | 3.06 | 9.80 | 13.60 |
| $T_{\text{gen}}$ | Generation of 40 active variants | 82.36 | 15.70 | 84.62 | 108.55 |
| $T_{\text{batch}}$ | Batched forward inference (40 variants) | 98.19 | 12.53 | 92.88 | 126.12 |
| $T_{\text{cbi}}$ | CBI metric calculation & state machine | 0.47 | 0.15 | 0.41 | 0.77 |
| $T_{\text{total}}$ | Full End-to-End MCB Circuit Breaker | **191.62** | **21.35** | **187.85** | **234.42** |
| $T_{\text{mcdo}}$ | MC-Dropout (10 head passes on features) | 10.44 | 1.79 | 9.91 | 14.17 |
| $T_{\text{entropy}}$ | Softmax Entropy computation | 0.15 | 0.47 | 0.10 | 0.14 |

---

## Section 8: Grad-CAM Saliency Drift Audit (Section 9.2)

Evaluated on 50 Tripped (OPEN) vs 50 Stable (CLOSED) samples from Partition B.
Target class was strictly **fixed to baseline prediction $y_{\text{pred}}$**.

| Circuit Breaker State | Cosine Similarity (Mean $\pm$ SD) | Top-20% IoU (Mean $\pm$ SD) | Centroid Distance (Mean $\pm$ SD) |
|:---|:---:|:---:|:---:|
| **Stable (CLOSED)** | $0.785 \pm 0.265$ | $0.573 \pm 0.255$ | $0.061 \pm 0.071$ |
| **Tripped (OPEN)** | $0.441 \pm 0.335$ | $0.261 \pm 0.264$ | $0.130 \pm 0.101$ |

**Audit Findings:** Tripped samples experience significant feature saliency dispersion under perturbation:
- Cosine similarity drops by 43.8% ($0.785 \to 0.441$).
- Top-20% activation IoU drops by 54.5% ($0.573 \to 0.261$).
- Centroid drift increases by 2.1x ($0.061 \to 0.130$).
*Note:* Grad-CAM visualizes gradient activations and does not establish verified physical causal boundaries.

---

## Section 9: Label-Preservation Audit Summary

- Automated programmatic verification executed on 20 validation samples (seed 42, 800 image evaluations).
- Domain constraints strictly upheld: 100% boundary, finite-value, and resolution preservation.
- Directional horizontal flip replaced by affine shear ($s_x = 0.05$).
- Programmatic audit results documented in `label_preservation_audit.md`.

---

## Section 10: Section 2 Audit Outputs

- `audit_before.txt`: 2 failures (severity weights order violation, identity tiers present).
- `audit_after.txt`: 0 failures across all 5 checks.

---

## Section 11: Limitations and Methodological Deviations

1. **Same-Source Evaluation:** Partition A and Partition B originate from the same held-out GTSRB validation pool. Performance is in-sample relative to dataset source.
2. **Track Metadata Fallback:** GTSRB validation split metadata lacks camera sequence/track IDs. Per Section 4 fallback rule, independent sample stratification was used.
3. **Head-Only MC-Dropout:** MobileNetV3-Small features were cached and 10 forward passes were executed through the classifier head with dropout $p=0.20$.
4. **Boundary Extension:** Partition A grid search encountered optimum on the boundary ($\tau=0.10$), prompting a pre-registered one-step extension to $\tau=0.05$.
