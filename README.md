# Metamorphic Circuit Breaker Framework

> **Runtime Transform-Brittleness Detection and Explainability for Vision Models**
> Applied to GTSRB Traffic Sign Classification with MobileNetV3-Small

---

## 1. Project Overview

The **Metamorphic Circuit Breaker (MCB)** is a runtime AI safety and diagnostic integrity framework designed to detect prediction brittleness in vision models. Traditional evaluation primarily measures model performance on static, i.i.d. test data. However, vision models deployed in real-world environments can produce unstable predictions when the same visual input is subjected to transformations such as changes in viewpoint, scale, brightness, contrast, blur, or sensor noise.

This framework implements a real-time, two-level hierarchical circuit breaker that evaluates model prediction invariance under metamorphic transformations, calculates a mathematical **Composite Brittleness Index (CBI)**, and regulates predictions across three operational states:

1. **CLOSED (Auto-Approved):** Predictions remain sufficiently invariant across metamorphic perturbations.
2. **HALF-OPEN (Monitor / Warning):** Localized sensitivity is detected and the prediction is flagged for additional review.
3. **OPEN (Safety Intercept / Tripped):** Significant multi-family instability is detected and the prediction is intercepted rather than treated as reliable.

The framework is demonstrated on the **German Traffic Sign Recognition Benchmark (GTSRB)** using **MobileNetV3-Small**. The traffic-sign classification scenario provides a compact real-world safety-oriented example in which an unreliable visual prediction could potentially affect a downstream decision.

The underlying model architecture is not modified by the circuit breaker. The framework operates as an external safety layer around the vision model, demonstrating its **model-agnostic design**.

---

## 2. Architecture and Methodology

```text
                   Input Image (x)
                         |
        +----------------+----------------+
        |                                 |
Baseline Inference (f(x))       22 Metamorphic Tests (t_i(x))
        |                         |  3 Semantic Families
        |                         |  Mild / Moderate / Severe tiers
        |                         v
        |              Live PyTorch Inference on all 22 variants
        |                         |
        +----------------+--------+
                         |
              Prediction Flip Analysis
                         |
        Level 1: Score_m per Transform Type (w(s) weighted)
                         |
        Level 2: Intra-Family Instability Ratio (IR_k)
                         |
        Cross-Family Spread (CFS) and Uncertainty (1 - c_0)
                         |
        Master Composite Brittleness Index (CBI)
                         |
      +------------------+------------------+
      |                  |                  |
CBI < theta_warn   theta_warn <= CBI    CBI >= theta_trip
   [CLOSED]         < theta_trip             [OPEN]
Auto-Approved         [HALF-OPEN]       Safety Intercept
                   Monitor / Warning     Prediction Block
                                                |
                                                v
                                     Grad-CAM Shift Analysis
```

### 2.1 Metamorphic Family Hierarchy (22 Tests)

Tests are organized into 3 semantic families across 8 transform types, each with a calibrated three-tier severity ladder:

| Family                           | Transform Type  | Severity Tiers and Parameters                                 | Severity Weights w(s)            |
| -------------------------------- | --------------- | ------------------------------------------------------------- | -------------------------------- |
| **Geometric** (M_geom = 3)       | Rotation        | Mild (5 deg), Moderate (15 deg), Severe (30 deg)              | Mild = 1.0, Mod = 0.6, Sev = 0.3 |
|                                  | Scale / Zoom    | Mild (0.95x), Moderate (0.90x), Severe (0.80x)                | Mild = 1.0, Mod = 0.6, Sev = 0.3 |
|                                  | Horizontal Flip | Moderate (Reflection)                                         | Mod = 0.6                        |
| **Photometric** (M_photo = 3)    | Brightness      | Mild (1.15x), Moderate (1.30x), Severe (1.55x)                | Mild = 1.0, Mod = 0.6, Sev = 0.3 |
|                                  | Contrast        | Mild (1.20x), Moderate (1.35x), Severe (1.45x)                | Mild = 1.0, Mod = 0.6, Sev = 0.3 |
|                                  | Saturation      | Mild (0.75x), Moderate (0.50x), Severe (0.25x)                | Mild = 1.0, Mod = 0.6, Sev = 0.3 |
| **Sensor / Noise** (M_noise = 2) | Gaussian Blur   | Mild (r=1.5), Moderate (r=2.5), Severe (r=3.5)                | Mild = 1.0, Mod = 0.6, Sev = 0.3 |
|                                  | Gaussian Noise  | Mild (sigma=0.02), Moderate (sigma=0.05), Severe (sigma=0.09) | Mild = 1.0, Mod = 0.6, Sev = 0.3 |

### 2.2 Mathematical Engine

1. **Level 1 Normalization (Transform Type Score):**

   \(\text{Score}_m = \frac{\sum_{s \in S_m} w(s) \cdot \mathbb{I}[f(t_{m,s}(x)) \neq y_0]}{\sum_{s \in S_m} w(s)}\)

   Penalizes mild failures heavily ($w=1.0$) while down-weighting severe perturbation failures ($w=0.3$).

2. **Level 2 Normalization (Intra-Family Instability Ratio):**

   \(\text{IR}_k = \frac{1}{M_k} \sum_{m=1}^{M_k} \text{Score}_m\)

   Unweighted average across transform types within family $k$. This prevents families with more tests from artificially skewing the risk score.

3. **Cross-Family Spread Score (CFS):**

   \(\text{CFS} = \frac{1}{|F|} \sum_{k \in F} \mathbb{I}[\text{IR}_k \ge \tau_{\text{fam}}]\)

   Measures what fraction of families have instability beyond the activation threshold $\tau_{\text{fam}} = 0.35$.

4. **Master Composite Brittleness Index (CBI):**

   \(\text{CBI}(x) = \alpha \cdot \max_k \text{IR}_k(x) + \beta \cdot \text{CFS}(x) + \gamma \cdot (1 - c_0)\)

   Default weights:

   * $\alpha = 0.50$ — peak instability
   * $\beta = 0.35$ — systemic spread
   * $\gamma = 0.15$ — baseline uncertainty

5. **Operational State Thresholds:**

   * **CLOSED:** $\text{CBI} < \theta_{\text{warn}} = 0.25$
   * **HALF-OPEN:** $\theta_{\text{warn}} \le \text{CBI} < \theta_{\text{trip}} = 0.55$
   * **OPEN:** $\text{CBI} \ge \theta_{\text{trip}} = 0.55$

---

## 3. Why Predictions with Multiple Flips Can Still Be `CLOSED`

A naive circuit breaker rule ("trip if any flip occurs") can produce excessive false alarms. Some strong or severe transformations may cause isolated prediction changes even when the underlying visual content remains recognizable.

Under the Phase 2 Hierarchical Engine:

* Flips on severe tiers are downweighted to $w=0.3$ compared with $w=1.0$ for mild transformations.
* A family becomes compromised when its average instability $\text{IR}*k \ge \tau*{\text{fam}} = 0.35$.
* If prediction flips are scattered across different families and occur primarily on severe transformations, no individual family may reach the activation threshold.
* Consequently, the overall CBI may remain below the warning threshold.
* Conversely, concentrated prediction instability on lower-severity transformations can increase the corresponding family instability and trigger **HALF-OPEN** or **OPEN**.

This hierarchical design allows the circuit breaker to distinguish isolated transformation sensitivity from broader prediction brittleness.

---

## 4. Model and Dataset

### 4.1 Model

The current demonstration uses:

**MobileNetV3-Small (`mobilenetv3_small_100`)**

The model is initialized with ImageNet-pretrained weights and fine-tuned for the GTSRB classification task.

The final classification layer is configured for **43 traffic-sign classes**.

### 4.2 Dataset

The **German Traffic Sign Recognition Benchmark (GTSRB)** is used as the demonstration dataset.

GTSRB contains **43 traffic-sign classes**, represented by class IDs from `0` to `42`.

The training pipeline uses the labeled `Train.csv` data and performs a stratified split:

```text
GTSRB training data
        |
        +-- 70% --> Training set
        |
        +-- 30% --> Validation set
```

For the current implementation:

* Total labeled images: **39,209**
* Training images: **27,446**
* Validation images: **11,763**
* Number of classes: **43**
* Input resolution used by MobileNetV3-Small: **224 x 224**

The model achieved approximately **95.94% validation accuracy after the first training epoch** in the current experiment.

### 4.3 Safety-Oriented Scenario

Traffic-sign recognition is a useful demonstration scenario for the framework because a vision model may be used to identify signs such as speed limits, stop signs, yield signs, and other road signs.

For example:

```text
Traffic-sign image
       |
       v
MobileNetV3-Small
       |
       v
Predicted traffic-sign class
       |
       v
Metamorphic Circuit Breaker
       |
       +-- Stable prediction --> CLOSED
       |
       +-- Local instability --> HALF-OPEN
       |
       +-- Significant instability --> OPEN
```

The circuit breaker does not replace the classifier. Instead, it provides an additional runtime reliability check before the prediction is treated as trustworthy.

---

## 5. Repository Structure

```text
Metamorphic-CircuitBreaker/
|-- checkpoints/
|   |-- best_model.pth           # Local trained MobileNetV3-Small checkpoint
|   |-- val_split.csv            # Stratified validation split
|   |-- train_split.csv          # Stratified training split
|   `-- training_log.json        # Epoch metrics and loss curves
|-- data/
|   `-- gtsrb/                   # GTSRB dataset
|       |-- Train/
|       |   |-- 0/
|       |   |-- 1/
|       |   |-- ...
|       |   `-- 42/
|       |-- Test/
|       |-- Train.csv
|       |-- Test.csv
|       `-- Meta.csv
|-- outputs/                     # Phase 1 batch pipeline run archives
|-- scripts/
|   |-- update_presets.py        # Curate verified live test images
|   |-- verify_all.py            # Complete backend verification suite
|   |-- verify_states.py         # Circuit breaker state validation
|   `-- demo_presets.py          # Automatic preset discoverer
|-- src/
|   |-- circuit_breaker.py       # MetamorphicCircuitBreaker and DiagnosticReport
|   |-- gradcam.py               # Grad-CAM utilities and comparison figures
|   |-- metamorphic.py           # Phase 1: 8 flat metamorphic relations
|   |-- metamorphic_families.py  # Phase 2: 22 hierarchical tests
|   |-- predict.py               # Phase 1 batch inference runner
|   |-- train.py                 # GTSRB MobileNetV3-Small training script
|   |-- utils.py                 # Model loaders, transforms, helpers, class names
|   |-- visualize.py             # Phase 1 batch charting utilities
|   `-- models/
|       |-- base.py              # BaseVisionModel abstract base class
|       |-- mobilenet_adapter.py # MobileNetV3-Small adapter
|       |-- convnext_adapter.py  # Original ConvNeXt adapter
|       `-- dummy_model.py       # Mock model adapter
|-- run_phase1.py                # Phase 1 batch pipeline entrypoint
|-- streamlit_app.py             # Interactive live dashboard
|-- requirements.txt
`-- README.md
```

---

## 6. Setup and Installation

### 6.1 Prerequisites

* Python 3.10 to 3.12
* PyTorch 2.0+
* CPU or CUDA-supported GPU

### 6.2 Installation

```bash
# Clone the repository and enter the directory
cd Metamorphic-CircuitBreaker

# Install dependencies
pip install -r requirements.txt
```

### 6.3 Dataset Setup

Download and extract the GTSRB dataset into:

```text
data/gtsrb/
```

The expected structure is:

```text
data/gtsrb/
├── Train/
├── Test/
├── Train.csv
├── Test.csv
└── Meta.csv
```

### 6.4 Train MobileNetV3-Small

Run:

```bash
python src/train.py --data_dir data/gtsrb --epochs 1 --batch_size 32
```

The best validation checkpoint is saved to:

```text
checkpoints/best_model.pth
```

The checkpoint contains the trained MobileNetV3-Small model state, validation metrics, optimizer state, and training configuration.

---

## 7. How to Run the System

### 7.1 Interactive Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

#### Live Circuit Breaker Evaluation

The dashboard accepts a traffic-sign image and evaluates it using:

1. Baseline MobileNetV3-Small inference.
2. 22 metamorphic transformations.
3. Prediction-flip analysis.
4. Hierarchical CBI calculation.
5. Circuit breaker state determination.
6. Grad-CAM analysis when applicable.

The dashboard reports the resulting:

* Baseline predicted class
* Prediction confidence
* Metamorphic test results
* Family instability ratios
* CFS
* CBI
* Circuit breaker state
* Grad-CAM visualizations

### 7.2 Command-Line Verification

Run the complete verification suite:

```bash
python scripts/verify_all.py
```

The verification suite validates:

* Metamorphic family matrix completeness.
* 22-test evaluation.
* Model-agnostic adapter capability.
* MobileNetV3-Small forward evaluation.
* Circuit breaker CBI calculation.
* Grad-CAM integration.
* Comparison figure generation.

Run the circuit breaker state verification:

```bash
python scripts/verify_states.py
```

This validates the circuit breaker state logic and model-independent behavior.

---

## 8. Current Verification Results

The MobileNetV3-Small + GTSRB implementation has been verified through the project test suite.

The verification includes:

```text
22 metamorphic tests
        ↓
CBI calculation
        ↓
MobileNetV3-Small evaluation
        ↓
Grad-CAM integration
        ↓
Comparison figure generation
```

The current verification run successfully reports:

```text
STEP 1: 22 tests complete
STEP 2: CBI formula verified; DummyVisionModel model-agnostic
STEP 3: MobileNetAdapter evaluation pipeline
STEP 4: evaluate_with_gradcam (GradCAM integration)
STEP 5: make_comparison_figure

ALL CHECKS PASSED
```

---

## 9. Model-Agnostic Extensibility

The framework is designed to be independent of a specific vision architecture.

A model can be integrated through the `BaseVisionModel` interface in `src/models/base.py`.

A compatible adapter should implement:

```python
from src.models.base import BaseVisionModel
from PIL import Image
from typing import Tuple, List, Optional
import numpy as np

class CustomModelAdapter(BaseVisionModel):

    def __init__(self, weights_path: str):
        # Load custom model weights / API client
        ...

    def predict(
        self,
        image: Image.Image
    ) -> Tuple[int, str, float, List[float]]:
        # Return:
        # (class_idx, label_name, confidence, softmax_probabilities)
        ...

    def explain(
        self,
        image: Image.Image,
        target_class: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # Return a normalized 2D attention heatmap
        # or None if explainability is unavailable.
        ...
```

The adapter can then be passed directly to the circuit breaker:

```python
from src.circuit_breaker import MetamorphicCircuitBreaker

adapter = CustomModelAdapter("path/to/weights.pth")

cb = MetamorphicCircuitBreaker(adapter)

report = cb.evaluate(image)
```

The current repository provides:

* `MobileNetAdapter` — GTSRB + MobileNetV3-Small implementation
* `ConvNeXtAdapter` — original model implementation
* `DummyVisionModel` — model-agnostic testing adapter

This demonstrates that the circuit breaker operates at the **model-adapter interface** rather than depending on a particular neural network architecture.

---

## 10. Traffic Sign Classes

GTSRB contains 43 traffic-sign classes represented by class IDs `0` through `42`.

The current implementation preserves the official numeric class IDs:

```text
Class 0
Class 1
Class 2
...
Class 42
```

The classifier therefore performs a **43-class multiclass classification task**, using a softmax probability distribution over the 43 output classes.

---

## 11. Project Demonstration

The current implementation demonstrates the following pipeline:

```text
                GTSRB Traffic Sign
                       |
                       v
             MobileNetV3-Small
                       |
                       v
              Baseline Prediction
                       |
             +---------+---------+
             |                   |
             v                   v
       Original Input       22 Metamorphic
                            Transformations
                                   |
                                   v
                         Repeated Model Inference
                                   |
                                   v
                         Prediction Flip Analysis
                                   |
                                   v
                         Hierarchical CBI Engine
                                   |
              +--------------------+--------------------+
              |                    |                    |
              v                    v                    v
           CLOSED              HALF-OPEN              OPEN
        Stable result        Warning state       Safety intercept
                                                        |
                                                        v
                                                  Grad-CAM
```

The purpose is not to retrain or modify the underlying vision model. Instead, the **Metamorphic Circuit Breaker acts as an external runtime safety layer** that checks whether the model's prediction remains sufficiently stable under controlled transformations.

---

## 12. Team and Project Details

* **Abel Louis Fernandez** (B23CS2103)
* **Akshay P Benjamin** (B23CS2112)
* **Aparna S** (B23CS2116)
* **Goutham C S** (B23CS2130)

**Project Guide:** Ms. Neethi Narayanan, Assistant Professor, Department of Computer Science and Engineering
