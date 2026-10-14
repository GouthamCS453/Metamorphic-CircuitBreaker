# Metamorphic Circuit Breaker (MCB) Framework
## Standardized Cross-Domain Testing & Rectification Protocol (Member-Agnostic)

**Document Version:** 3.0  
**Target Audience:** All Framework Implementations & Research Team Members  
**Scope:** Universal across all model backbones (`ConvNeXt-Base`, `MobileNetV3-Small`, `ResNet-18`, `EfficientNet-B0`) and domains (`ISIC 2019`, `GTSRB`, `Oxford Pet`, `Plant Village`).

---

### Executive Summary & Purpose

A comprehensive mathematical and statistical audit of our preliminary benchmark runs revealed critical methodological contradictions across our evaluation pipeline:
1. **Severity Weight Inversion:** The codebase used increasing weights `[0.1, 0.2, 0.3]`, penalizing severe transformations $3\times$ higher than mild ones. This contradicts our fundamental premise that **mild-tier flips represent severe brittleness** and should carry the highest penalty.
2. **Family Cardinality Division Bias ($M_k$ Skew):** Families with 3 transform types require more total flips to exceed $\tau_{\text{fam}}=0.35$ than families with only 2 transform types, creating an artificial mathematical bias that made Sensor/Noise appear as the dominant failure axis.
3. **State Machine Collapse ($\theta_{\text{trip}} < \tau_{\text{warn}}$):** Lowering trip cutoffs during optimization violated the three-state ordering ($\text{CLOSED} \to \text{HALF-OPEN} \to \text{OPEN}$).
4. **Invalid Domain Perturbations:** Directional classes (e.g., Turn Left vs. Turn Right in GTSRB) were subjected to horizontal reflection, creating invalid semantic flips.
5. **In-Sample Evaluation Bias:** Thresholds were calibrated and evaluated on the exact same images without an independent held-out partition.
6. **Missing Control Baselines:** Comparisons only used a naive 1-flip strawman instead of standard $k$-of-$n$ flip fractions and Test-Time Augmentation (TTA) variance.

**Every member must execute the exact fixes and protocol below to regenerate verified, statistically publication-ready results.**

---

## 1. Problem Verification Checklist (Run Before Changing Code)

Before applying fixes, run this quick audit script inside your environment to confirm the presence of the mathematical bugs:

```python
# verify_audit_issues.py
from src.metamorphic_families import SEVERITY_WEIGHTS, HIERARCHICAL_TEST_MATRIX
from src.circuit_breaker import CircuitBreakerConfig

print("--- AUDIT VERIFICATION ---")

# 1. Check severity weight ordering
mild_w = SEVERITY_WEIGHTS.get("mild", None)
sev_w = SEVERITY_WEIGHTS.get("severe", None)
print(f"Severity Weights: mild={mild_w}, severe={sev_w}")
if mild_w < sev_w:
    print("[CRITICAL BUG CONFIRMED]: Weights are inverted! Severe is weighted higher than Mild.")
else:
    print("[OK]: Weights are properly descending.")

# 2. Check family sizes (Mk)
families = {}
for t in HIERARCHICAL_TEST_MATRIX:
    families.setdefault(t.family, set()).add(t.transform_type)
for fam, types in families.items():
    print(f"Family '{fam}' has {len(types)} transform types: {types}")
if len(set(len(v) for v in families.values())) > 1:
    print("[CRITICAL BUG CONFIRMED]: Unequal Mk across families! Integer division skews IR_k.")

# 3. Check State Machine ordering under lower trip thresholds
cfg = CircuitBreakerConfig(theta_warn=0.25, theta_trip=0.20)
if cfg.theta_trip < cfg.theta_warn:
    print("[CRITICAL BUG CONFIRMED]: theta_trip (0.20) < theta_warn (0.25). State machine collapses.")
```

---

## 2. Core Framework File Modifications (All Members)

Apply these precise changes in your implementation repository:

### Fix 2.1: Invert Severity Ladder Weights
**Target File:** `src/metamorphic_families.py` (or your local transform definitions file)  
**Location:** `SEVERITY_WEIGHTS` dictionary

#### 💡 Rationale & Why This Fix is Critical:
In metamorphic safety testing, perturbation magnitude has an inverse relationship with model vulnerability:
- A flip under a **mild perturbation** (e.g., 5° rotation, 2% noise) proves extreme model brittleness right at the decision boundary. It indicates that tiny, imperceptible real-world jitter triggers misclassification. Therefore, **mild flips must receive the highest penalty ($1.0$)**.
- Conversely, a flip under a **severe perturbation** (e.g., 30° rotation, heavy optical blur) is expected even for robust human observers. Penalizing severe flips higher (e.g., `0.3` vs `0.1`) caused the system to over-react to extreme distortions while remaining blind to subtle, catastrophic boundary failures.

Replace:
```python
# INCORRECT (OLD)
SEVERITY_WEIGHTS: Dict[str, float] = {
    "mild": 0.1,      # or 1.0 depending on local branch
    "moderate": 0.2,  # or 0.6
    "severe": 0.3,    # or 0.3
}
```
With the **standardized descending weights**:
```python
# CORRECT (NEW) - Subtle brittleness receives highest penalty
SEVERITY_WEIGHTS: Dict[str, float] = {
    "mild": 1.0,      # Maximum penalty: flipping on a 5-deg tilt or 2% noise indicates extreme brittleness
    "moderate": 0.6,  # Medium penalty: boundary sensitivity
    "severe": 0.3,    # Lowest penalty: aggressive distortion where flips are more excusable
}
```

---

### Fix 2.2: Equalize Intra-Family Cardinality ($M_k$ Normalization)
**Target File:** `src/circuit_breaker.py` (and test scripts computing $IR_k$)  
**Location:** Inside `evaluate()` at Level 2 Normalization:

#### 💡 Rationale & Why This Fix is Critical:
The original Level 2 normalization used unweighted averaging across transform types: $\text{IR}_k = \frac{1}{M_k} \sum \text{Score}_m$.
- **Geometric** and **Photometric** families have $M_k = 3$ transform types. A complete collapse of one transform yields $\text{IR}_k = 1/3 = 0.333 < \tau_{\text{fam}} = 0.35$. Consequently, the family **could never activate** from a single collapsed transform type.
- **Sensor/Noise** has only $M_k = 2$ transform types (blur and noise). A collapse of blur alone yields $\text{IR}_k = 1/2 = 0.50 \ge 0.35$, easily activating the family.
- **Consequence:** Sensor/Noise appeared to be the dominant vulnerability axis purely due to an artifact of integer division ($1/2 > 1/3$).
- **The Solution:** The hybrid formula ($0.70 \cdot \text{mean} + 0.30 \cdot \text{peak}$) ensures that a single completely broken transform type produces $\text{IR}_k \approx 0.53 - 0.65$ across all families, guaranteeing equal triggering power and eliminating family size bias.

Replace:
```python
# INCORRECT (OLD) - Confounded by unequal number of transform types
family_instability[fam] = sum(type_scores[t] for t in t_types) / len(t_types)
```

With **Max-Transform Anchored Scoring**:
```python
# CORRECT (NEW) - Normalized Intra-Family Invalidation Ratio
if t_types:
    # 1. Mean instability across transform types
    mean_ir = sum(type_scores[t] for t in t_types) / len(t_types)
    # 2. Maximum transform type failure
    max_type_ir = max(type_scores[t] for t in t_types)
    # Hybrid formulation: 70% mean family instability + 30% peak type vulnerability
    ir_k = 0.70 * mean_ir + 0.30 * max_type_ir
else:
    ir_k = 0.0

family_instability[fam] = round(float(ir_k), 4)
```

---

### Fix 2.3: Dynamically Couple Three-State Thresholds
**Target File:** `src/circuit_breaker.py`  
**Location:** Inside `CircuitBreakerConfig` and state decision block:

#### 💡 Rationale & Why This Fix is Critical:
In previous calibration sweeps, optimizing for safety recall lowered $\theta_{\text{trip}}$ to values like $0.10$ or $0.20$, while $\tau_{\text{warn}}$ remained fixed at $0.25$.
- This caused $\theta_{\text{trip}} < \tau_{\text{warn}}$, completely breaking the state machine: a score of $0.15$ triggered `OPEN` (tripped) before even reaching `HALF-OPEN` (warning).
- The three-state progression ($\text{CLOSED} \to \text{HALF-OPEN} \to \text{OPEN}$) must always be mathematically monotonic. Dynamically anchoring $\tau_{\text{warn}} = 0.50 \cdot \theta_{\text{trip}}$ guarantees structural integrity across any threshold sweep.

Ensure `theta_warn` is automatically scaled to prevent state-ordering violations:
```python
@dataclass
class CircuitBreakerConfig:
    alpha: float = 0.50
    beta: float = 0.35
    gamma: float = 0.15
    tau_fam: float = 0.25       # Calibrated default family cutoff
    theta_trip: float = 0.30     # Trip threshold
    theta_warn: Optional[float] = None  # Warning threshold

    def __post_init__(self) -> None:
        # Dynamically set theta_warn to strictly half of theta_trip if unassigned or invalid
        if self.theta_warn is None or self.theta_warn >= self.theta_trip:
            self.theta_warn = round(0.50 * self.theta_trip, 4)
            
        assert self.theta_warn < self.theta_trip, (
            f"State ordering violated: theta_warn ({self.theta_warn}) must be < theta_trip ({self.theta_trip})"
        )
```

State assignment logic:
```python
if cbi >= self.config.theta_trip:
    state = CircuitBreakerState.OPEN
    action = "TRIP_TO_EXPERT"
elif cbi >= self.config.theta_warn:
    state = CircuitBreakerState.HALF_OPEN
    action = "WARN_AND_PASS"
else:
    state = CircuitBreakerState.CLOSED
    action = "AUTO_APPROVE"
```

---

### Fix 2.4: Domain-Specific Perturbation Filtering (GTSRB & Directional Tasks)
**Target File:** Dataset loader / transform generator in member repository.

#### 💡 Rationale & Why This Fix is Critical:
A fundamental axiom of metamorphic testing is that the transformation must be **label-preserving (invariant)**.
- For traffic signs, horizontal reflection fundamentally changes semantic meaning:
  - Sign 33 (Turn Right) mirrored becomes Turn Left (Sign 34).
  - Sign 38 (Keep Right) mirrored becomes Keep Left (Sign 39).
- When a model flips its prediction under horizontal flip on a directional sign, the model is actually behaving **correctly**, not failing. Including horizontal flip inflated the naive metamorphic false alarm rate to an artificial $84.1\%$.
- Directional signs must have horizontal flip disabled or replaced with an invariant distortion (e.g., small camera perspective tilt).

**Rule:** Any dataset containing classes with semantic chirality (directionality) **must exclude or mask horizontal reflection**.
* For **GTSRB (Autonomous Driving)**:
  * Directional sign IDs (e.g., Turn Left `[33]`, Turn Right `[34]`, Keep Left `[39]`, Keep Right `[38]`, Roundabout Mandatory `[40]`) **must not undergo horizontal flip**.
  * Replace horizontal flip on directional signs with a **Vertical Perspective Tilt** (`shear_x=0.05` or pitch simulation) to preserve semantic invariance.
* For **ISIC 2019**, **Plant Village**, and **Oxford Pet**:
  * Horizontal flip is strictly invariant (organic lesions, leaves, and animals possess isotropic or mirror invariance) and remains active.

---

## 3. Standardized Testing Mechanisms & Baselines

#### 💡 Rationale for Expanding Baseline Comparisons:
In peer-reviewed AI safety literature, evaluating against only an unmonitored model or a naive 1-flip detector creates two fatal vulnerabilities:
1. **The "Strawman" Criticism:** A naive 1-flip breaker trips on *any single prediction change*. On 22 transformations, this virtually guarantees high false alarm rates (e.g. $80\%+$), making it an artificially weak baseline that reviewers reject. Continuous baselines ($k$-of-$n$ flip fraction and TTA prediction variance) provide the necessary, rigorous benchmark.
2. **The Softmax Entropy Blind Spot:** Reviewers frequently ask: *"Why build a complex metamorphic pipeline when simple softmax entropy detects misclassifications for free?"* Benchmarking against Softmax Entropy and MC-Dropout across identical sample sets allows us to explicitly demonstrate where metamorphic testing succeeds (catching overconfident, high-probability errors where entropy fails).

All members must execute **four comparative baselines** on the exact same evaluation data to provide a fair, rigorous comparison:

### Baseline 1: Standalone Model (Unmonitored)
* The raw prediction $f(x)$. Intercepts $0\%$ of errors; $0\%$ false trips.

### Baseline 2: Continuous $k$-of-$n$ Flip Fraction (Replaces 1-Flip Strawman)
* Let $n$ be the total number of metamorphic transforms ($n=22$).
* Calculate the flip fraction:
  $$\text{FF}(x) = \frac{1}{n} \sum_{i=1}^n \mathbb{I}[f(t_i(x)) \neq f(x)]$$
* Evaluate across thresholds $k \in \{1, 2, 3, 5, 8\}$ flips. (1-flip is reported as the special case $k=1$).

### Baseline 3: Softmax Entropy Thresholding
* Compute predictive Shannon entropy:
  $$H(x) = -\sum_{c=1}^C P_c \log P_c$$
* Flag/trip if $H(x) \ge \tau_{\text{entropy}}$, where $\tau_{\text{entropy}}$ is set to the 85th percentile of the calibration split. Report AUROC across all cutoffs.

### Baseline 4: Test-Time Augmentation (TTA) Prediction Variance
* Run the same transformed variants through the model. Compute the categorical variance of predictions:
  $$\text{Var}_{\text{TTA}}(x) = 1.0 - \max_c \left(\frac{1}{n+1} \sum_{i=0}^n \mathbb{I}[f(t_i(x)) == c]\right)$$
* This directly benchmarks MCB against an unweighted, non-hierarchical variance monitor.

### Baseline 5: Monte Carlo Dropout (MCDO)
* Execute 10 stochastic forward passes with dropout active ($p=0.2$).
* **Hardware Standardization:** If using a CNN/ViT without convolutional dropout, perform head-only dropout passes. **Run on the exact same compute device (GPU or CPU) as the base model** to ensure latency measurements are directly comparable.

---

## 4. Required Data Splitting Protocol (Eliminating In-Sample Bias)

#### 💡 Rationale for 50/50 Calibration vs. Evaluation Splitting:
Evaluating threshold performance on the same dataset used to find optimal cutoffs creates **in-sample optimistic bias (data leakage)**:
- In our preliminary tests, thresholds were swept across all $N=250$ samples, and peak Youden $J$ was reported on the same $N=250$. Reviewers immediately spot that this overstates real-world generalization.
- Dividing the evaluation pool into **Partition A (Calibration)** and **Partition B (Held-Out Test)** guarantees that reported metrics reflect genuine, out-of-sample performance:
  - $\tau_{\text{fam}}^*$ and $\theta_{\text{trip}}^*$ are tuned **only on Partition A**.
  - All headline metrics ($\text{Recall}$, $\text{FAR}$, $J$, $\text{AUROC}$) are evaluated **strictly on Partition B** with the thresholds frozen.

1. **Total Evaluation Samples ($N \ge 200$):**
   * **Partition A: Calibration Split (50% of samples, e.g. $N=100$ or $125$):**
     * Used exclusively to run Test 5 (2D grid search over $\tau_{\text{fam}}$ and $\theta_{\text{trip}}$) to identify the operating point maximizing Youden’s index:
       $$J = \text{Recall} - \text{FAR} = \frac{\text{TP}}{\text{TP} + \text{FN}} - \frac{\text{FP}}{\text{FP} + \text{TN}}$$
   * **Partition B: Held-Out Test Split (50% of samples, e.g. $N=100$ or $125$):**
     * The optimal thresholds $(\tau_{\text{fam}}^*, \theta_{\text{trip}}^*)$ derived from Partition A are applied frozen to Partition B.
     * **All reported Table metrics must be calculated strictly on Partition B.**

---

## 5. Direct Semantic Attribution Validation (Controlled Fault Injection)

To provide experimental proof that *"MCB's value lies in structured semantic fault attribution"*, every member must execute a controlled synthetic injection test:

1. Select 30 correctly classified test images.
2. Synthetically inject isolated, domain-calibrated corruptions:
   * **Batch 1 (Sensor Noise Only):** Apply severe Gaussian Blur ($\sigma=1.5$) or heavy sensor noise.
   * **Batch 2 (Photometric Drift Only):** Apply severe contrast scaling ($0.6\times$) or illumination drop.
   * **Batch 3 (Geometric Distort Only):** Apply severe scale/zoom or rotation.
3. Pass each batch through the updated MCB engine.
4. Record whether the **tripped family matches the injected corruption family**:
   $$\text{Attribution Accuracy} = \frac{\text{Count}(\text{Peak Family} == \text{Injected Family})}{\text{Total Tripped Injected Samples}}$$
*Target: Attribution Accuracy must exceed $80\%$ across all batches.*

---

## 6. Output Delivery Specification (Format Requirements)

Once your tests complete, generate your final output in the following standardized format so it can be automatically ingested into our cross-domain paper generator:

### File 1: `results_summary.json`
Save inside your `results/` folder with this exact schema:

```json
{
  "member_info": {
    "domain_name": "Autonomous Vehicles (GTSRB Traffic Signs)",
    "model_backbone": "MobileNetV3-Small",
    "hardware_device": "cuda:0 (NVIDIA RTX 3050)"
  },
  "dataset_split": {
    "total_samples": 250,
    "calibration_samples": 125,
    "held_out_test_samples": 125,
    "held_out_base_model_errors": 18,
    "held_out_base_model_correct": 107
  },
  "calibrated_thresholds": {
    "tau_fam": 0.25,
    "theta_trip": 0.30,
    "theta_warn": 0.15,
    "calibration_youden_j": 0.582
  },
  "held_out_test_metrics": {
    "standalone": { "tp": 0, "fp": 0, "tn": 107, "fn": 18, "recall": 0.0, "far": 0.0, "auroc": null, "latency_ms": 10.5 },
    "softmax_entropy": { "tp": 11, "fp": 19, "tn": 88, "fn": 7, "recall": 0.611, "far": 0.178, "auroc": 0.835, "latency_ms": 10.9 },
    "naive_1_flip": { "tp": 16, "fp": 45, "tn": 62, "fn": 2, "recall": 0.889, "far": 0.421, "auroc": 0.734, "latency_ms": 42.1 },
    "k_of_n_flips": { "k_threshold": 3, "tp": 13, "fp": 22, "tn": 85, "fn": 5, "recall": 0.722, "far": 0.206, "auroc": 0.812, "latency_ms": 42.1 },
    "tta_variance": { "tp": 12, "fp": 20, "tn": 87, "fn": 6, "recall": 0.667, "far": 0.187, "auroc": 0.805, "latency_ms": 42.1 },
    "mc_dropout": { "tp": 14, "fp": 21, "tn": 86, "fn": 4, "recall": 0.778, "far": 0.196, "auroc": 0.841, "latency_ms": 15.2 },
    "mcb_prior_default": { "tau_fam": 0.35, "theta_trip": 0.55, "tp": 4, "fp": 3, "tn": 104, "fn": 14, "recall": 0.222, "far": 0.028, "auroc": 0.848, "latency_ms": 43.2 },
    "mcb_calibrated": { "tau_fam": 0.25, "theta_trip": 0.30, "tp": 13, "fp": 14, "tn": 93, "fn": 5, "recall": 0.722, "far": 0.131, "auroc": 0.848, "latency_ms": 43.2 }
  },
  "family_vulnerability_profile": {
    "geometric_mean_ir": 0.038,
    "photometric_mean_ir": 0.041,
    "sensor_noise_mean_ir": 0.185,
    "dominant_failure_axis": "sensor_noise"
  },
  "fault_attribution_experiment": {
    "total_injected": 30,
    "sensor_noise_accuracy": 0.90,
    "photometric_accuracy": 0.80,
    "geometric_accuracy": 0.80,
    "overall_attribution_accuracy": 0.833
  }
}
```

### File 2: `RESULTS_VERIFICATION.md`
A Markdown document providing:
1. Exact confusion matrices ($\text{TP}, \text{FP}, \text{TN}, \text{FN}$) confirming that $\text{TP}+\text{FN}=\text{Total Errors}$ and $\text{TN}+\text{FP}=\text{Total Correct}$.
2. Wilson $95\%$ Confidence Intervals for Safety Recall and False Alarm Rate.
3. Component-wise latency breakdown:
   * Base forward pass ($T_{\text{base}}$)
   * Transform batch generation ($T_{\text{gen}}$)
   * Batched tensor inference ($T_{\text{batch}}$)
   * CBI scoring ($T_{\text{cbi}}$)
   * Total latency ($T_{\text{total}}$)
4. Saliency drift metrics (Cosine Similarity, IoU, Centroid Euclidean Drift) comparing untripped vs. tripped samples.

---

## 7. Submission Checklist for Members

- [ ] Ran `verify_audit_issues.py` and confirmed bug identification.
- [ ] Updated `SEVERITY_WEIGHTS` to descending order (`mild: 1.0, moderate: 0.6, severe: 0.3`).
- [ ] Implemented max-transform intra-family normalization ($M_k$ bias elimination).
- [ ] Enforced $\theta_{\text{warn}} = 0.5 \cdot \theta_{\text{trip}}$ to safeguard state transitions.
- [ ] Filtered horizontal flips on directional/asymmetric classes (if applicable).
- [ ] Split dataset into Calibration (50%) and Held-Out Evaluation (50%) splits.
- [ ] Implemented Continuous $k$-of-$n$, Softmax Entropy, and TTA Variance baselines.
- [ ] Executed the Controlled Fault Injection experiment (attributing injected corruptions).
- [ ] Outputted verified `results_summary.json` and `RESULTS_VERIFICATION.md`.
