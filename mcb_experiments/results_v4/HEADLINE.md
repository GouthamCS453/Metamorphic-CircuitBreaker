# Metamorphic Circuit Breaker (MCB) v4: Executive Summary (GTSRB MobileNetV3-Small)

**Evaluation Date:** October 2026 | **Protocol:** `MCB_STANDARDIZED_TESTING_PROTOCOL_v4.md`
**Dataset:** German Traffic Sign Recognition Benchmark (GTSRB, 43 classes)
**Model Backbone:** MobileNetV3-Small (Pretrained, frozen checkpoint `checkpoints/best_model.pth`)
**Device:** Intel CPU | **Evaluation Policy:** Partition B evaluated exactly once with frozen thresholds

---

## 1. Split & Operational Configuration

- **Held-Out Validation Pool:** $N = 11,763$, Base Model Errors = 478 (Overall Error Rate: 4.06%)
- **Partition A (Calibration):** $N = 5,881$, Errors = 257
- **Partition B (Evaluation):** $N = 5,882$, Errors = 221
- **Frozen Calibration Thresholds (from Partition A):**
  - **`mcb_op_j` (Youden's J):** $\tau_{\text{fam}}^* = 0.05$, $\theta_{\text{trip}}^* = 0.30$, $\theta_{\text{warn}}^* = 0.150$
  - **`mcb_op_far10` (FAR $\le 10\%$):** $\tau_{\text{fam}}^* = 0.10$, $\theta_{\text{trip}}^* = 0.25$, $\theta_{\text{warn}}^* = 0.125$
  - **Weight Configuration:** $\alpha = 0.50$ (Max Instability), $\beta = 0.35$ (Cross-Family Spread), $\gamma = 0.15$ (Inverted Softmax Margin)

---

## 2. Partition B Evaluation: Primary Comparison Table

All metrics reported with 95% confidence intervals (Wilson score for Recall & FAR; Paired Bootstrap 2,000 resamples for AUROC).

| Method | Operating Point | Recall (95% CI) | FAR (95% CI) | Youden's J | AUROC (95% CI) | Critical-Error Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MCB (Op J)** | op_j | 84.16% [78.8%, 88.4%] | 12.74% [11.9%, 13.6%] | 0.7143 | 0.9102 [0.8847, 0.9338] | 100.0% |
| **MCB (Op FAR10)** | op_far10 | 78.28% [72.4%, 83.2%] | 9.36% [8.6%, 10.1%] | 0.6892 | 0.9043 [0.8806, 0.9269] | 100.0% |
| **MCB (Prior Default)** | prior_default | 28.51% [23.0%, 34.8%] | 0.32% [0.2%, 0.5%] | 0.2819 | 0.9052 [0.8815, 0.9273] | 50.0% |
| **Softmax Entropy** | op_j | 85.52% [80.3%, 89.6%] | 27.87% [26.7%, 29.1%] | 0.5765 | 0.8694 [0.8489, 0.8898] | 100.0% |
| **Entropy (Temp-Scaled)** | op_j | 87.78% [82.8%, 91.5%] | 16.46% [15.5%, 17.5%] | 0.7132 | 0.9367 [0.9254, 0.9472] | 100.0% |
| **MSP (1 - max p)** | op_j | 83.71% [78.3%, 88.0%] | 12.72% [11.9%, 13.6%] | 0.7099 | 0.9203 [0.9040, 0.9352] | 100.0% |
| **MC-Dropout (Head)** | op_j | 86.43% [81.3%, 90.3%] | 31.76% [30.6%, 33.0%] | 0.5466 | 0.8556 [0.8341, 0.8769] | 100.0% |
| **TTA Variance** | op_j | 77.83% [71.9%, 82.8%] | 7.40% [6.7%, 8.1%] | 0.7043 | 0.9097 [0.8839, 0.9335] | 100.0% |
| **Naive 1-Flip** | k=1 | 99.10% [96.8%, 99.8%] | 83.71% [82.7%, 84.7%] | 0.1538 | — | 100.0% |
| **5-Flip Majority** | k=5 | 77.83% [71.9%, 82.8%] | 7.40% [6.7%, 8.1%] | 0.7043 | — | 100.0% |
| **8-Flip Majority** | k=8 | 56.56% [50.0%, 62.9%] | 2.83% [2.4%, 3.3%] | 0.5373 | — | 100.0% |

---

## 3. Pre-Registered Decision Rules (D1 – D4)

- **D1 (MCB beats best baseline): NOT MET**
  - Best baseline: `entropy_ts` (AUROC = $0.9367$ [0.9174, 0.9542] vs MCB $0.9102$ [0.8847, 0.9338], $\Delta = -0.0264$, CI $[-0.0401, -0.0135]$ entirely below 0).
- **D2 (Hierarchical MCB beats Flat CBI): NOT MET**
  - Flat weighted flips `cbi_flat` achieved AUROC $0.9277$ [0.9078, 0.9465] vs MCB $0.9102$, $\Delta = -0.0175$, CI $[-0.0285, -0.0080]$ entirely below 0.
- **D3 (High-Confidence Silent Failure Detection): MET**
  - Among high-confidence errors ($c_0 \ge 0.90$ / top-quartile fallback, $N=56$), MCB (Op FAR10) achieved **35.7% recall** (20/56 caught, 95% CI [24.5%, 48.8%]) vs Softmax Entropy **5.4% recall** (3/56 caught, 95% CI [1.8%, 14.6%]) at matched $\le 10\%$ FAR.
  - On domain-specific critical sign confusions ($N=6$ in Partition B), MCB achieved 100.0% recall (6/6 caught).
- **D4 (Fault Attribution Accuracy > 50%): NOT MET**
  - Overall attribution accuracy across 300 corrupted images: **25.0%** [20.4%, 30.2%] (Wilson lower bound 0.204 < 0.50 threshold; chance = 33.3%).

---

## 4. Latency & Operational Overhead (Intel CPU)

- **Baseline Forward Pass ($T_{\text{base}}$):** 10.61 ms $\pm$ 3.06 ms
- **Transform Generation ($T_{\text{gen}}$):** 82.36 ms (40 variants)
- **Batched Model Inference ($T_{\text{batch}}$):** 98.19 ms
- **CBI Scoring Computation ($T_{\text{cbi}}$):** 0.47 ms
- **Total MCB Overhead ($T_{\text{total}}$):** **191.62 ms** (P50: 187.85 ms, P95: 234.42 ms)
- **MC-Dropout Overhead ($T_{\text{mcdo}}$):** 10.44 ms | **Softmax Entropy ($T_{\text{entropy}}$):** 0.15 ms
