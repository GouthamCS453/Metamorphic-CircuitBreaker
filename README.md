# Metamorphic Circuit Breaker Framework

> **Runtime Transform-Brittleness Detection and Explainability for Medical Vision Models**  
> Applied to ISIC 2019 Skin Lesion Classification with ConvNeXt-Base

---

## 1. Project Overview

The **Metamorphic Circuit Breaker (MCB)** is a runtime AI safety and diagnostic integrity framework. Traditional evaluation evaluates model performance exclusively on static, i.i.d. test splits. However, clinical computer vision models deployed in practice suffer from **transform-brittleness**: clinically invariant perturbations (such as slight camera probe angle, lighting temperature, defocus blur, or digital sensor noise) can trigger unexpected prediction flips.

This framework implements a real-time, two-level hierarchical circuit breaker that evaluates model prediction invariance at runtime, calculates a mathematical **Composite Brittleness Index (CBI)**, and regulates predictions across three operational states:
1. **CLOSED (Auto-Approved):** Predictions remain invariant across metamorphic perturbations.
2. **HALF-OPEN (Monitor / Warning):** Localized family sensitivity detected; prediction is delivered with an amber warning flag for secondary clinical review.
3. **OPEN (Safety Intercept / Tripped):** Multi-family collapse detected; automated diagnosis is blocked and intercepted for doctor review alongside Grad-CAM difference heatmaps.

---

## 2. Architecture and Methodology

```
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
                   Monitor / Warning     Doctor Handover
                                                |
                                                v
                                     Grad-CAM Shift Analysis
```

### 2.1 Metamorphic Family Hierarchy (22 Tests)

Tests are organized into 3 semantic families across 8 transform types, each with a calibrated three-tier severity ladder:

| Family | Transform Type | Severity Tiers and Parameters | Severity Weights w(s) |
|---|---|---|---|
| **Geometric** (M_geom = 3) | Rotation | Mild (5 deg), Moderate (15 deg), Severe (30 deg) | Mild = 1.0, Mod = 0.6, Sev = 0.3 |
| | Scale / Zoom | Mild (0.95x), Moderate (0.90x), Severe (0.80x) | Mild = 1.0, Mod = 0.6, Sev = 0.3 |
| | Horizontal Flip | Moderate (Reflection) | Mod = 0.6 |
| **Photometric** (M_photo = 3) | Brightness | Mild (1.15x), Moderate (1.30x), Severe (1.55x) | Mild = 1.0, Mod = 0.6, Sev = 0.3 |
| | Contrast | Mild (1.20x), Moderate (1.35x), Severe (1.45x) | Mild = 1.0, Mod = 0.6, Sev = 0.3 |
| | Saturation | Mild (0.75x), Moderate (0.50x), Severe (0.25x) | Mild = 1.0, Mod = 0.6, Sev = 0.3 |
| **Sensor / Noise** (M_noise = 2) | Gaussian Blur | Mild (r=1.5), Moderate (r=2.5), Severe (r=3.5) | Mild = 1.0, Mod = 0.6, Sev = 0.3 |
| | Gaussian Noise | Mild (sigma=0.02), Moderate (sigma=0.05), Severe (sigma=0.09) | Mild = 1.0, Mod = 0.6, Sev = 0.3 |

### 2.2 Mathematical Engine

1. **Level 1 Normalization (Transform Type Score):**
   $$\text{Score}_m = \frac{\sum_{s \in S_m} w(s) \cdot \mathbb{I}[f(t_{m,s}(x)) \neq y_0]}{\sum_{s \in S_m} w(s)}$$
   Penalizes mild failures heavily ($w=1.0$) while down-weighting artificial boundary noise ($w=0.3$).

2. **Level 2 Normalization (Intra-Family Instability Ratio):**
   $$\text{IR}_k = \frac{1}{M_k} \sum_{m=1}^{M_k} \text{Score}_m$$
   Unweighted average across transform types within family $k$. Prevents families with more tests from artificially skewing the risk score.

3. **Cross-Family Spread Score (CFS):**
   $$\text{CFS} = \frac{1}{|F|} \sum_{k \in F} \mathbb{I}[\text{IR}_k \ge \tau_{\text{fam}}]$$
   Measures what fraction of families have collapsed beyond the activation threshold $\tau_{\text{fam}} = 0.35$.

4. **Master Composite Brittleness Index (CBI):**
   $$\text{CBI}(x) = \alpha \cdot \max_k \text{IR}_k(x) + \beta \cdot \text{CFS}(x) + \gamma \cdot (1 - c_0)$$
   Default weights: $\alpha = 0.50$ (peak instability), $\beta = 0.35$ (systemic spread), $\gamma = 0.15$ (baseline uncertainty).

5. **Operational State Thresholds:**
   - **CLOSED:** $\text{CBI} < \theta_{\text{warn}} = 0.25$
   - **HALF-OPEN:** $\theta_{\text{warn}} \le \text{CBI} < \theta_{\text{trip}} = 0.55$
   - **OPEN:** $\text{CBI} \ge \theta_{\text{trip}} = 0.55$

---

## 3. Why Predictions with Multiple Flips Can Still Be `CLOSED`

A naive circuit breaker rule ("trip if any flip occurs") creates severe false-alarm fatigue in clinical workflows. When benign skin lesion images are subjected to extreme boundary noise (e.g. severe $0.80\times$ zoom, $0.25\times$ desaturation, or high Gaussian noise), peripheral pixels can trigger isolated edge flips while the primary lesion pathology remains clear.

Under the Phase 2 Hierarchical Engine:
- Flips on severe tiers are downweighted to $w=0.3$ (compared to $w=1.0$ for mild).
- A family only becomes compromised when its average instability $\text{IR}_k \ge \tau_{\text{fam}} = 0.35$.
- If an image has 4 flips scattered across different families exclusively on severe/boundary tiers, no single family reaches 0.35, the spread score $\text{CFS}$ remains 0.0, and $\text{CBI}$ stays below 0.25 ($\theta_{\text{warn}}$).
- Conversely, an image with **only 3 flips** on mild rotation tests will push $\text{IR}_{\text{geom}} \ge 0.35$ and immediately trigger **HALF-OPEN** or **OPEN**.

---

## 4. Repository Structure

```
main_implementation/
|-- checkpoints/
|   |-- best_model.pth           # Trained ConvNeXt-Base model (1.05 GB)
|   |-- val_split.csv            # Stratified validation split
|   |-- train_split.csv          # Stratified training split
|   `-- training_log.json        # Epoch metrics and loss curves
|-- data/
|   |-- isic2019/                # Dataset directory
|   |   |-- ISIC_2019_Training_Input/
|   |   `-- ISIC_2019_Training_GroundTruth.csv
|   `-- presets/                 # Live-verified demo preset cases
|       |-- preset_stable_nevus/      # CLOSED state (0 flips)
|       |-- preset_boundary_flips/    # CLOSED state with 4 boundary flips (anti-false-alarm demo)
|       |-- preset_half_open_warning/ # HALF-OPEN state (3 mild geometric flips)
|       `-- preset_open_tripped/      # OPEN state (12 multi-family flips)
|-- outputs/                     # Phase 1 batch pipeline run archives
|-- scripts/
|   |-- update_presets.py        # Curate verified live test images into data/presets/
|   |-- verify_all.py            # Complete backend verification suite
|   |-- verify_states.py         # Circuit breaker state validation script
|   `-- demo_presets.py          # Automatic preset discoverer
|-- src/
|   |-- circuit_breaker.py       # MetamorphicCircuitBreaker and DiagnosticReport
|   |-- gradcam.py               # ConvNeXtGradCAM and make_comparison_figure()
|   |-- metamorphic.py           # Phase 1: 8 flat metamorphic relations
|   |-- metamorphic_families.py  # Phase 2: 22 hierarchical tests (3 families)
|   |-- predict.py               # Phase 1 batch inference runner
|   |-- train.py                 # PyTorch ConvNeXt-Base fine-tuning script
|   |-- utils.py                 # Model loaders, transforms, helpers, class names
|   |-- visualize.py             # Phase 1 batch charting utilities
|   `-- models/
|       |-- base.py              # BaseVisionModel abstract base class
|       |-- convnext_adapter.py  # ConvNeXt implementation of BaseVisionModel
|       `-- dummy_model.py       # Mock model adapter for model-agnostic demonstrations
|-- run_phase1.py                # Phase 1 batch pipeline entrypoint
|-- streamlit_app.py             # Phase 2 interactive live dashboard
|-- requirements.txt
`-- README.md
```

---

## 5. Setup and Installation

### 5.1 Prerequisites
- Python 3.10 to 3.12 (or 3.14 on Windows)
- PyTorch 2.0+ (CPU or CUDA)

### 5.2 Installation
```bash
# Clone the repository and enter the directory
cd main_implementation

# Install dependencies
pip install -r requirements.txt
```

### 5.3 Initialize Verified Demo Presets
Populate `data/presets/` with verified live validation images covering all three states:
```bash
python scripts/update_presets.py
```

---

## 6. How to Run the System

### 6.1 Interactive Dashboard (Primary Interface)

Launch the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```
Open **http://localhost:8501** in your browser.

#### Tab 1: Live Circuit Breaker Studio
- **Input Options:** Upload your own JPG/PNG dermoscopy image or select from pre-verified demo presets.
- **Real-Time PyTorch Inference:** Every run executes 23 actual forward passes on CPU (~3.5 seconds) in memory.
- **State Decision:** Clear banner display (`CLOSED`, `HALF-OPEN`, or `OPEN`) with corresponding clinical callouts.
- **Grad-CAM Attention Analysis:** Generates 4-panel side-by-side attention comparisons (`Original Image`, `Transformed Image`, `Baseline GradCAM`, `Flipped GradCAM`) for any prediction flip.
- **Mathematical Attribution:** Exploded formula breakdown with live substitution of Level 1 scores, Level 2 family IR values, and CFS.

#### Tab 2: Parameter Calibration
- Interactively tune weights ($\alpha, \beta, \gamma$) with live sum-to-1.0 constraints.
- Adjust decision thresholds ($\theta_{\text{warn}}, \theta_{\text{trip}}, \tau_{\text{fam}}$).
- Changes made here immediately update the active engine in Tab 1.
- Test hypothetical clinical risk scenarios using the simulator widget.

#### Tab 3: Batch Run Archive
- Access historical Phase 1 batch evaluations from `outputs/run_*`.
- Dynamically switch between runs to inspect sample counts, flip rates, and prediction tables.
- Option to launch a batch evaluation run on a specified sample slice.

---

## 7. Command-Line Verification Scripts

### Complete Verification Suite
```bash
python -m scripts.verify_all
```
Validates:
- Metamorphic family matrix completeness (22 tests across 3 families).
- Dummy model plug-in capability.
- ConvNeXt forward evaluation pipeline.
- Grad-CAM heatmaps and in-memory figure synthesis.

### Circuit Breaker State Verification
```bash
python -m scripts.verify_states
```
Validates:
- `retain_grad` runtime hook fix under `torch.no_grad()`.
- State `CLOSED` with real ConvNeXt model.
- State `HALF-OPEN` / `OPEN` with high-flip mock model.
- End-to-end Grad-CAM extraction on real dataset images.

---

## 8. Model-Agnostic Extensibility

The framework is decoupled from the ConvNeXt model. Another team member can plug in their own architecture and dataset:

1. **Subclass `BaseVisionModel`** (`src/models/base.py`):
   ```python
   from src.models.base import BaseVisionModel
   from PIL import Image
   from typing import Tuple, List, Optional
   import numpy as np

   class CustomModelAdapter(BaseVisionModel):
       def __init__(self, weights_path: str):
           # Load custom model weights / API client
           ...

       def predict(self, image: Image.Image) -> Tuple[int, str, float, List[float]]:
           # Return (class_idx, label_name, confidence, softmax_probabilities)
           ...

       def explain(self, image: Image.Image, target_class: Optional[int] = None) -> Optional[np.ndarray]:
           # Return 2D float32 normalized attention heatmap [0, 1] (GradCAM, attention map, etc.)
           ...
   ```

2. **Initialize Circuit Breaker with Your Adapter:**
   ```python
   from src.circuit_breaker import MetamorphicCircuitBreaker

   adapter = CustomModelAdapter("path/to/weights.pth")
   cb = MetamorphicCircuitBreaker(adapter)
   report = cb.evaluate(image)
   ```

A working reference mock adapter is available in `src/models/dummy_model.py`.

---

## 9. ISIC 2019 Diagnostic Classes

| Code | Diagnostic Class | Clinical Category |
|---|---|---|
| **MEL** | Melanoma | Malignant |
| **NV** | Melanocytic Nevus | Benign |
| **BCC** | Basal Cell Carcinoma | Malignant |
| **AK** | Actinic Keratosis | Pre-malignant |
| **BKL** | Benign Keratosis | Benign |
| **DF** | Dermatofibroma | Benign |
| **VASC** | Vascular Lesion | Benign |
| **SCC** | Squamous Cell Carcinoma | Malignant |

---

## 10. Team and Project Details

- **Abel Louis Fernandez** (B23CS2103)
- **Akshay P Benjamin** (B23CS2112)
- **Aparna S** (B23CS2116)
- **Goutham C S** (B23CS2130)

**Project Guide:** Ms. Neethi Narayanan, Assistant Professor, Department of Computer Science and Engineering
