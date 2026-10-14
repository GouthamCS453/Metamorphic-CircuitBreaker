# Metamorphic Circuit Breaker (MCB) Framework: Technical Architecture & Operation Manual

**Document Version:** 4.1 (Post-Audit Rectified Edition)  
**Target Audience:** Core Engineering Team, Research Collaborators, Scientific Reviewers  
**Codebase References:** [`src/metamorphic_families.py`](file:///c:/Users/Goutham%20C%20S/Desktop/main_implementation/src/metamorphic_families.py), [`src/circuit_breaker.py`](file:///c:/Users/Goutham%20C%20S/Desktop/main_implementation/src/circuit_breaker.py), [`mcb_experiments/v4/`](file:///c:/Users/Goutham%20C%20S/Desktop/main_implementation/mcb_experiments/v4/)

---

## 1. High-Level Concept & Problem Statement

Modern Deep Neural Networks (DNNs) deployed in high-consequence domains (e.g., dermatoscopic cancer triage, autonomous traffic perception, precision agriculture) suffer from **silent, overconfident failures**. A vision model can classify a life-threatening melanoma as a benign nevus with $98\%$ softmax confidence because it evaluated the query on an unmonitored point in its feature space.

Standard runtime monitors attempt to catch these failures via:
1. **Uncertainty Estimators (Softmax Entropy, MSP):** Blind to overconfident misclassifications where the network produces peak probabilities on the wrong class.
2. **Stochastic Sampling (Monte Carlo Dropout):** Expensive (requires multiple forward passes through modified weights), alters the underlying model architecture, and depends on internal dropout layers.
3. **Naive 1-Flip Metamorphic Testing:** A strawman approach where *any single label change* across metamorphic transforms trips an alarm. When checking 20+ transforms, natural boundary variance triggers catastrophic false alarm rates ($70\% - 85\%$), inducing crippling alert fatigue.

### The MCB Solution
The **Metamorphic Circuit Breaker (MCB)** operates as a non-invasive, model-agnostic runtime safety wrapper. It subjects the incoming image to structured, semantic perturbation families organized in a **hierarchical severity ladder**. Rather than tripping on isolated flips, MCB uses a **two-level normalization engine** to compute a bounded continuous index: the **Composite Brittleness Index ($\text{CBI}$)**, which drives a fail-safe **Three-State Machine** (`CLOSED`, `HALF_OPEN`, `OPEN`).

```
                    ┌────────────────────────┐
                    │ Query Input Image x    │
                    └───────────┬────────────┘
                                │
               ┌────────────────┴────────────────┐
               │                                 │
     ┌─────────▼────────┐              ┌─────────▼────────┐
     │ Baseline Model   │              │ Metamorphic Bank │
     │ Prediction f(x)  │              │ 25 Perturbations │
     │ c0 = max P(x)    │              │ 3 Orthogonal Fam │
     └─────────┬────────┘              └─────────┬────────┘
               │                                 │
               │               ┌─────────────────▼────────┐
               │               │ Forward Pass & Flips     │
               │               │ flip_i = (f(t_i) != f(x))│
               │               └─────────────────┬────────┘
               │                                 │
               │               ┌─────────────────▼────────┐
               │               │ Level 1: Score_m         │
               │               │ (Severity-Weighted Type) │
               │               └─────────────────┬────────┘
               │                                 │
               │               ┌─────────────────▼────────┐
               │               │ Level 2: IR_k (M_k = 3)  │
               │               │ Intra-Family Invalidation│
               │               └─────────────────┬────────┘
               │                                 │
               └────────────────┬────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │ CBI(x) Aggregator   │
                     │ Peak + Spread + Conf│
                     └──────────┬──────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
   CBI < theta_warn   theta_warn <= CBI < theta   CBI >= theta_trip
         │                      │                      │
   ┌─────▼──────┐        ┌──────▼──────┐        ┌──────▼──────┐
   │   CLOSED   │        │  HALF_OPEN  │        │    OPEN     │
   │Auto-Approve│        │Doctor Warning│       │Trip to Human│
   └────────────┘        └─────────────┘        └─────────────┘
```

---

## 2. Metamorphic Transformation Taxonomy & Severity Ladders

All test perturbations are organized strictly into **three semantic families** ($K=3$), each containing **exactly three transform types** ($M_k = 3$) to guarantee equal mathematical weighting.

### Severity Weighting Logic ($W_{\text{tier}}$)
Unlike naive robustness benchmarking where heavy corruptions receive the highest penalty, **metamorphic safety testing uses strictly descending severity weights**:
* **Mild ($\mathbf{1.0}$):** Minute perturbations (e.g., $5^\circ$ rotation, $2\%$ noise). A label flip under an imperceptible shift demonstrates **extreme decision boundary fragility**. This carries the highest penalty ($W_{\text{mild}} = 1.0$).
* **Moderate ($\mathbf{0.6}$):** Standard operational variance (e.g., $10^\circ$ tilt, contrast boost). Carries medium penalty ($W_{\text{mod}} = 0.6$).
* **Severe ($\mathbf{0.3}$):** Substantial distortion (e.g., $15^\circ$ tilt, heavy defocus, low JPEG quality). Label flips are more excusable here, so it receives the lowest penalty ($W_{\text{sev}} = 0.3$).
* **Single Tier ($\mathbf{1.0}$):** Applied to unitary invariant operations like Horizontal Flip ($W_{\text{single}} = 1.0$).

```
Sensitivity to Perturbation:
┌────────────────────────────────────────────────────────┐
│ MILD Flip (5 deg tilt)    --> Penalty = 1.0 (Critical) │
│ MODERATE Flip (10 deg)    --> Penalty = 0.6 (Moderate) │
│ SEVERE Flip (15 deg)      --> Penalty = 0.3 (Minor)    │
└────────────────────────────────────────────────────────┘
```

### Full 25-Variant Transformation Registry

| Family ($k$) | Transform Type ($m$) | Tiers Evaluated | Parameter / Magnitude | Weight ($W$) | Safety Rationale |
|---|---|---|---|:---:|---|
| **Geometric** | Rotation | Mild<br>Moderate<br>Severe | Clockwise $+5^\circ$<br>Clockwise $+10^\circ$<br>Clockwise $+15^\circ$ | 1.0<br>0.6<br>0.3 | Patient positioning tilt, dermatoscopic handheld rotation |
| | Scale (Zoom-Crop) | Mild<br>Moderate<br>Severe | Center crop 0.95x & resize<br>Center crop 0.90x & resize<br>Center crop 0.80x & resize | 1.0<br>0.6<br>0.3 | Lens distance variation, lesion cropping zoom |
| | Horizontal Reflection | Single (Moderate) | Left-to-right mirror flip | 1.0 | Organic bilateral invariance (lesions/leaves) |
| **Photometric** | Brightness | Mild<br>Moderate<br>Severe | Scale luminance 1.15x<br>Scale luminance 1.35x<br>Scale luminance 1.55x | 1.0<br>0.6<br>0.3 | Dermatoscope LED power drift, ambient lighting |
| | Contrast | Mild<br>Moderate<br>Severe | Scale contrast 1.15x<br>Scale contrast 1.35x<br>Scale contrast 1.45x | 1.0<br>0.6<br>0.3 | Skin melanin boundary variance, pigment dynamics |
| | Saturation | Mild<br>Moderate<br>Severe | Color scale 0.80x<br>Color scale 0.55x<br>Color scale 0.25x | 1.0<br>0.6<br>0.3 | Sensor white balance fault, blood perfusion drift |
| **Sensor/Noise** | Gaussian Blur | Mild<br>Moderate<br>Severe | Radius $\sigma=1.0\text{ px}$<br>Radius $\sigma=2.0\text{ px}$<br>Radius $\sigma=3.5\text{ px}$ | 1.0<br>0.6<br>0.3 | Optical lens defocus, gel layer refraction |
| | Gaussian Noise | Mild<br>Moderate<br>Severe | Zero-mean additive $\sigma=0.02$<br>Zero-mean additive $\sigma=0.05$<br>Zero-mean additive $\sigma=0.09$ | 1.0<br>0.6<br>0.3 | High ISO thermal sensor floor noise |
| | JPEG Compression | Mild<br>Moderate<br>Severe | Quality factor $q=90$<br>Quality factor $q=70$<br>Quality factor $q=50$ | 1.0<br>0.6<br>0.3 | Tele-dermatology PACS ingestion compression artifacts |

---

## 3. Mathematical Evaluation Pipeline

Given query image $x$, original model prediction $f(x) \in \{1, \dots, C\}$ with confidence $c_0 = \max_c P(y=c \mid x)$, the framework computes decision boundaries through four distinct stages:

### Stage 1: Level 1 Normalization ($\text{Score}_m$)
For each transform type $m$ within family $k$ (e.g., Rotation in Geometric), multiple severity variants are executed. $\text{Score}_m(x)$ measures the severity-weighted fraction of flips:

$$\text{Score}_m(x) = \frac{\sum_{t \in T_m} W_t \cdot \mathbb{I}[f(t(x)) \neq f(x)]}{\sum_{t \in T_m} W_t} \in [0.0, 1.0]$$

* **Key property:** If a model flips on a *Mild* perturbation, $\text{Score}_m$ spikes dramatically (numerator gains $1.0$). If it only flips on *Severe*, the numerator gains only $0.3$, dampening false panic.

---

### Stage 2: Level 2 Normalization ($\text{IR}_k$) & Cardinality Equalization
To aggregate transform types into an overall **Family Invalidation Ratio ($\text{IR}_k$)**, earlier iterations suffered from an integer division bias because families had different transform counts ($M_k$). 

In the rectified v4 protocol, all families possess exactly $M_k = 3$ transform types. The family invalidation ratio is computed via unweighted averaging across types:

$$\text{IR}_k(x) = \frac{1}{M_k} \sum_{m=1}^{M_k} \text{Score}_m(x) = \frac{1}{3} \sum_{m=1}^3 \text{Score}_m(x) \in [0.0, 1.0]$$

* **Why unweighted averaging across types?** If rotation, scale, and flip all flip, $\text{Score}_m = 1.0$ for each, giving $\text{IR}_{\text{geom}} = 1.0$. If only rotation collapses while scale and flip remain stable, $\text{IR}_{\text{geom}} = 0.333$.

---

### Stage 3: Cross-Family Spread ($\text{CFS}$)
A crucial insight of the circuit breaker is that localized sensitivity along a single axis (e.g., mild rotation sensitivity) is often benign. In contrast, **multi-axis instability** (collapsing across geometry, lighting, and sensor noise simultaneously) indicates total epistemic breakdown.

Let $\tau_{\text{fam}}$ be the family compromise cutoff (calibrated on Partition A):

$$\text{CFS}(x) = \frac{1}{K} \sum_{k=1}^K \mathbb{I}[\text{IR}_k(x) \ge \tau_{\text{fam}}] \in \{0.0, \frac{1}{3}, \frac{2}{3}, 1.0\}$$

---

### Stage 4: Composite Brittleness Index ($\text{CBI}$)
The complete vulnerability index combines the worst-family collapse, multi-family spread, and baseline model uncertainty:

$$\text{CBI}(x) = \alpha \cdot \max_{k} \text{IR}_k(x) + \beta \cdot \text{CFS}(x) + \gamma \cdot (1.0 - c_0)$$

$$\text{Subject to: } \alpha + \beta + \gamma = 1.0, \quad \alpha, \beta, \gamma \ge 0$$

* **Standard Default Allocation:** $\alpha = 0.50$ (Peak Family Collapse), $\beta = 0.35$ (Cross-Family Spread), $\gamma = 0.15$ (Model Confidence Discount).
* **Bounds:** $\text{CBI}(x)$ is bounded strictly in $[0.0, 1.0]$.

---

## 4. The Three-State Machine & Decision Logic

Unlike binary detectors that can only flag or pass, MCB operates a **tri-state finite state machine** that balances doctor safety with clinical workflow efficiency:

```
                    ┌──────────────────┐
                    │      CLOSED      │
                    │   Auto-Approve   │
                    └────────┬─────────┘
                             │ CBI >= theta_warn
                             ▼
                    ┌──────────────────┐
                    │    HALF_OPEN     │
                    │ Doctor Warning   │
                    └────────┬─────────┘
                             │ CBI >= theta_trip
                             ▼
                    ┌──────────────────┐
                    │       OPEN       │
                    │ Tripped to Human │
                    └──────────────────┘
```

### State Definitions
1. **`CLOSED` ($\text{CBI}(x) < \theta_{\text{warn}}$):**
   * **Clinical Action:** `AUTO_APPROVE`
   * The query demonstrates invariant consistency across all semantic perturbations and high baseline confidence. Automated reporting is authorized.
2. **`HALF_OPEN` ($\theta_{\text{warn}} \le \text{CBI}(x) < \theta_{\text{trip}}$):**
   * **Clinical Action:** `WARN_AND_PASS` (Advisory Warning)
   * The model prediction is displayed to the clinician, but accompanied by an attention advisory (e.g., *"Model exhibits moderate illumination sensitivity; review pigment borders"*).
3. **`OPEN` ($\text{CBI}(x) \ge \theta_{\text{trip}}$):**
   * **Clinical Action:** `TRIP_TO_EXPERT` (Hard Interception)
   * Automated prediction is blocked. The case is routed directly to senior pathologist/doctor review as unclassifiable by AI.

### Dynamic Coupling Guard
To prevent state machine inversion during threshold sweeps (e.g., where a low $\theta_{\text{trip}}$ drops below $\theta_{\text{warn}}$, causing cases to jump from CLOSED to OPEN without warning), the engine enforces:

$$\theta_{\text{warn}} = \min(\theta_{\text{warn}}, \text{round}(0.50 \cdot \theta_{\text{trip}}, 4))$$

---

## 5. Implementation Architecture & Data Flow

The production code is organized across modular components in `src/`:

### 1. [`src/metamorphic_families.py`](file:///c:/Users/Goutham%20C%20S/Desktop/main_implementation/src/metamorphic_families.py)
* **`MetamorphicTest` Dataclass:** Holds transform metadata (`id`, `family`, `transform_type`, `severity`, `weight`, `apply_fn`).
* **`HIERARCHICAL_TEST_MATRIX`:** Registered list of 25 transforms.
* **Deterministic Transformation Callables:** Uses OpenCV and PIL with seeded random noise (`RandomState(42)`) to ensure identical per-image reproducibility.

### 2. [`src/circuit_breaker.py`](file:///c:/Users/Goutham%20C%20S/Desktop/main_implementation/src/circuit_breaker.py)
* **`CircuitBreakerConfig`:** Dataclass holding $\alpha, \beta, \gamma, \tau_{\text{fam}}, \theta_{\text{warn}}, \theta_{\text{trip}}$. Auto-validates weight summation to $1.0$ and enforces state ordering in `__post_init__`.
* **`MetamorphicCircuitBreaker.evaluate(img)`:** Main inference method:
  1. Executes baseline prediction $f(x)$ and saves confidence $c_0$.
  2. Runs 25 transformed images through the model (batched on GPU or sequential on CPU).
  3. Groups results by `(family, transform_type)` and computes Level 1 $\text{Score}_m$.
  4. Computes Level 2 $\text{IR}_k$ across the 3 types.
  5. Computes $\text{CFS}$ and composite $\text{CBI}$.
  6. Evaluates threshold transitions and emits a `DiagnosticReport`.
* **`evaluate_with_gradcam(img)`:** Extends diagnostic evaluation by computing paired baseline and flipped Grad-CAM saliency heatmaps to inspect attention shifts.

---

## 6. How the Framework Operates in Practice: Concrete Walkthrough

### Example A: Clean, Robust Nevus (Safe Pass)
* **Input:** Correctly diagnosed benign nevus ($y_{\text{true}} = 1, y_{\text{pred}} = 1, c_0 = 0.94$).
* **Metamorphic Behavior:**
  * Geometric: 0 flips $\to \text{Score} = 0.0$
  * Photometric: 1 flip on *Severe Brightness* ($W = 0.3$) $\to \text{Score}_{\text{bright}} = 0.3 / 1.9 = 0.158 \to \text{IR}_{\text{photo}} = 0.053$
  * Sensor/Noise: 0 flips $\to \text{Score} = 0.0$
* **Metrics:**
  * $\max \text{IR} = 0.053 < \tau_{\text{fam}} (0.10) \implies \text{CFS} = 0.0$
  * Uncertainty: $1.0 - 0.94 = 0.06$
  * $\text{CBI} = 0.50(0.053) + 0.35(0.0) + 0.15(0.06) = \mathbf{0.0355}$
* **Decision:** $\text{CBI} = 0.0355 < \theta_{\text{warn}} (0.125) \implies$ **`CLOSED` (Auto-Approved)**.

---

### Example B: Silent Melanoma Misclassification (Safety Interception)
* **Input:** True Melanoma misclassified as Benign Nevus with high model confidence ($y_{\text{true}} = 0, y_{\text{pred}} = 1, c_0 = 0.88$).
* **Metamorphic Behavior:**
  * Boundary instability causes model to flip to Melanoma or Basal Cell on:
    * *Mild Rotation* ($5^\circ$, $W=1.0$)
    * *Mild Contrast* ($1.15\times$, $W=1.0$)
    * *Mild Blur* ($\sigma=1.0$, $W=1.0$)
  * Level 1 scores spike: $\text{Score}_{\text{rot}} = 0.526$, $\text{Score}_{\text{contrast}} = 0.476$, $\text{Score}_{\text{blur}} = 0.526$.
  * Level 2 family ratios: $\text{IR}_{\text{geom}} = 0.220$, $\text{IR}_{\text{photo}} = 0.185$, $\text{IR}_{\text{sensor}} = 0.210$.
* **Metrics:**
  * All three families exceed $\tau_{\text{fam}} = 0.10 \implies \text{CFS} = 3 / 3 = \mathbf{1.0}$.
  * Worst-Family Collapse: $\max \text{IR} = 0.220$.
  * Uncertainty: $1.0 - 0.88 = 0.12$.
  * $\text{CBI} = 0.50(0.220) + 0.35(1.0) + 0.15(0.12) = 0.110 + 0.350 + 0.018 = \mathbf{0.478}$.
* **Decision:** $\text{CBI} = 0.478 \ge \theta_{\text{trip}} (0.25) \implies$ **`OPEN` (Tripped to Expert)**.
* **Outcome:** The silent, high-confidence clinical error was intercepted and prevented from reaching the patient.

---

## 7. Known Empirical Limitations & Operating Guidance

Following rigorous validation on Protocol v4, the engineering team and researchers must operate under these verified scientific realities:

1. **Test-Time Augmentation (TTA) Domination:**
   * On in-distribution dermoscopy, unweighted categorical TTA prediction variance achieves **$52.4\%$ Recall at $5.8\%$ FAR**, outperforming MCB's **$52.4\%$ Recall at $8.7\%$ FAR**. If computational resources are strictly limited and multi-axis introspection is not required, continuous TTA variance is the superior detector.
2. **Attribution Collapse in Medical Imaging:**
   * Controlled fault injection achieved **$28.3\%$ accuracy** (indistinguishable from $33.3\%$ chance), collapsing to Geometric in $70\%$ of cases. In organic lesions, lighting and noise perturbations induce feature shifts that trigger boundary sensitivity. MCB cannot be claimed to provide causal root-cause attribution in dermatology.
3. **Execution Latency:**
   * On CPU devices, evaluating 25 metamorphic variants requires **$\sim 2.6\text{ seconds}$ per image**. In production environments, MCB must be deployed on GPU hardware, where batched tensor evaluation runs in **$\sim 48.6\text{ ms}$**.

---

## Summary Checklist for Deployment

| Component | Setting / Parameter | Verified Value | Function |
|---|---|:---:|---|
| **Severity Weights** | `mild / mod / sev / single` | `1.0 / 0.6 / 0.3 / 1.0` | Inverts penalty to penalize subtle boundary fragility |
| **Transform Bank** | $M_k$ per family | $3$ types each ($25$ variants) | Eliminates family cardinality division skew |
| **CBI Formulation** | $\alpha / \beta / \gamma$ | $0.50 / 0.35 / 0.15$ | Bounded composite vulnerability index |
| **Calibrated Thresholds** | $\tau_{\text{fam}}^* / \theta_{\text{trip}}^*$ | $0.10 / 0.25$ | Operating point for balanced clinical safety ($52.4\%$ recall, $8.7\%$ FAR) |
| **State Machine Guard** | $\theta_{\text{warn}}$ | $0.50 \cdot \theta_{\text{trip}} = 0.125$ | Guarantees monotonic three-state progression |
