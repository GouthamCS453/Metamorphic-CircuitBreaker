# Metamorphic Circuit Breaker — Initial Phase

> **Detecting Transform-Brittleness in Deployed Vision Models** 
> Applied to ISIC 2019 Skin Lesion Classification with ConvNeXt-Base

---

## Project Overview

This project implements the **initial phase** of the Metamorphic Circuit Breaker Framework — a runtime safety layer that detects when a deployed vision model gives inconsistent predictions under semantically-irrelevant image transformations (*transform-brittleness*).

**Core idea:** If a skin-lesion classifier changes its diagnosis just because an image is rotated 15° or slightly blurred, that is a dangerous unreliability indicator. This framework systematically identifies and explains such failures.

### Framework Components (Initial Phase)

| Component | Purpose |
|---|---|
| `src/train.py` | Train ConvNeXt-Base on 70% of ISIC 2019 |
| `src/metamorphic.py` | 8 hardcoded Metamorphic Relations (MRs) |
| `src/predict.py` | Run inference on 100 images × 8 MRs |
| `src/gradcam.py` | GradCAM heatmaps for wrong predictions |
| `src/visualize.py` | 4 graphical outputs |
| `run_phase1.py` | Main orchestration (all steps) |
| `streamlit_app.py` | Interactive dashboard |
| `random_gradcam.py` | Random image GradCAM showcase |

---

## Project Structure

```
main_implementation/

 Abstract/
 main_abstract_01.pdf # Project abstract

 data/
 isic2019/ # ← Place ISIC 2019 dataset here
 ISIC_2019_Training_Input/ # JPEG images
 ISIC_2019_Training_GroundTruth.csv

 checkpoints/ # Created by training
 best_model.pth # Best validation checkpoint
 training_log.json # Epoch-wise metrics
 train_split.csv # 70% train image paths
 val_split.csv # 30% val image paths

 outputs/ # One folder per run_phase1.py call
 run_YYYYMMDD_HHMMSS/
 metadata.json # Run config + sampled image IDs
 predictions_table.csv # Full results (100 × 9 rows)
 sample_images/ # 100 original JPEGs (224×224)
 transformed/
 MR1/ # Rotated variants
 MR2/ # Flipped variants
 ... # MR3-MR8
 gradcam/
 wrong_predictions/ # 4-panel comparison PNGs
 random_showcase/ # Output of random_gradcam.py
 plots/
 mr_consistency.png
 wrong_counts.png
 overall_pie.png
 class_brittleness.png

 src/
 __init__.py
 utils.py # Constants, model loader, transforms
 metamorphic.py # 8 MR definitions
 train.py # Training script
 predict.py # Inference runner
 gradcam.py # GradCAM engine
 visualize.py # Plot generation

 run_phase1.py # Main orchestration script
 streamlit_app.py # Interactive dashboard
 random_gradcam.py # Random GradCAM showcase
 requirements.txt
 README.md # This file
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Windows Note:** If you encounter DataLoader issues, use `--num_workers 0` in the training command.

### 2. Prepare the ISIC 2019 Dataset

Download from [Kaggle — ISIC 2019 Challenge](https://www.kaggle.com/datasets/andrewmvd/isic-2019):

```bash
# Using Kaggle CLI (optional)
kaggle datasets download andrewmvd/isic-2019
```

Place the files at:
```
data/isic2019/
 ISIC_2019_Training_Input/ ← All JPEG images
 ISIC_2019_Training_GroundTruth.csv
```

---

## Commands

### Step 1 — Train the Model

```bash
python src/train.py --data_dir data/isic2019 --epochs 30 --batch_size 32
```

| Parameter | Default | Description |
|---|---|---|
| `--data_dir` | `data/isic2019` | Path to ISIC 2019 dataset root |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `1e-4` | AdamW initial learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--num_workers` | `0` | DataLoader workers (use 0 on Windows) |
| `--seed` | `42` | Random seed |
| `--no_amp` | flag | Disable automatic mixed precision |
| `--resume` | `None` | Resume from a checkpoint path |

**Outputs:**
- `checkpoints/best_model.pth` — Best validation checkpoint (saved after each epoch if accuracy improves)
- `checkpoints/training_log.json` — Epoch-level metrics (loss, accuracy, LR)
- `checkpoints/train_split.csv`, `checkpoints/val_split.csv` — Dataset splits

---

### Step 2 — Run Phase 1 Pipeline

> Requires `checkpoints/best_model.pth` to exist. The script **exits with a clear message** if the checkpoint is missing.

```bash
python run_phase1.py --data_dir data/isic2019 --n_samples 100
```

| Parameter | Default | Description |
|---|---|---|
| `--data_dir` | `data/isic2019` | ISIC 2019 dataset root |
| `--n_samples` | `100` | Number of validation images to test |
| `--seed` | `42` | Random seed for image sampling |
| `--checkpoint` | `checkpoints/best_model.pth` | Override checkpoint path |
| `--skip_gradcam` | flag | Skip GradCAM generation (much faster) |

**Pipeline steps:**
1. Verify checkpoint exists
2. Create timestamped `outputs/run_YYYYMMDD_HHMMSS/`
3. Load ConvNeXt-Base model
4. Sample 100 validation images (stratified)
5. Apply all 8 MRs + run inference → `predictions_table.csv`
6. Generate GradCAM comparison figures for every wrong prediction
7. Generate 4 graphical plots
8. Print summary report

**Outputs:**
- `outputs/run_*/predictions_table.csv` — 900 rows (100 images × 9 variants)
- `outputs/run_*/gradcam/wrong_predictions/*.png` — 4-panel GradCAM comparisons
- `outputs/run_*/plots/*.png` — 4 charts

---

### Step 3 — Streamlit Dashboard

The dashboard has **two modes** controlled by the `--mode` flag:

#### Stored Mode (default) — Browse Existing Results
```bash
streamlit run streamlit_app.py -- --mode stored
```
Select any previous run folder from the sidebar dropdown.

#### Live Mode — Run Pipeline + Display
```bash
streamlit run streamlit_app.py -- --mode live --data_dir data/isic2019 --n_samples 100
```

| Parameter | Default | Description |
|---|---|---|
| `--mode` | `stored` | `stored` = load existing results, `live` = run pipeline then display |
| `--data_dir` | `data/isic2019` | Dataset path (live mode only) |
| `--n_samples` | `100` | Sample count (live mode only) |
| `--checkpoint` | `checkpoints/best_model.pth` | Model checkpoint |

**Dashboard Tabs:**
| Tab | Content |
|---|---|
| Overview | Key metrics, per-MR stability summary, run metadata |
| Prediction Table | Filterable full results table with CSV download |
| GradCAM Viewer | Browse wrong-prediction comparison figures |
| Charts | All 4 generated plots with descriptions |
| Image Explorer | Select any of the 100 images, see all 8 transforms + predictions |

---

### Step 4 — Random GradCAM Showcase

```bash
python random_gradcam.py
```

| Parameter | Default | Description |
|---|---|---|
| `--run_dir` | Latest run | Path to a specific `run_*` directory |
| `--image_id` | Random | Specific ISIC image ID to showcase |
| `--seed` | None | Random seed for image selection |
| `--checkpoint` | `checkpoints/best_model.pth` | Override model checkpoint |
| `--save_only` | flag | Save PNG without opening display window |

**Examples:**
```bash
# Random image from latest run, show in window
python random_gradcam.py

# Specific image, save only
python random_gradcam.py --image_id ISIC_0024306 --save_only

# Specific run directory
python random_gradcam.py --run_dir outputs/run_20240810_143000

# Reproducible random selection
python random_gradcam.py --seed 42
```

**Output:** `outputs/run_*/gradcam/random_showcase/<image_id>_gradcam_showcase.png` 
A 3×3 grid of panels showing Original + 8 MRs. Each panel: image thumbnail (left) | GradCAM overlay (right).

---

## Metamorphic Relations Reference

| ID | Name | Transformation | Parameters | Rationale |
|---|---|---|---|---|
| MR1 | Rotation | Rotate 15° clockwise | `angle=15°` | Lesion orientation is diagnostically irrelevant |
| MR2 | HorizontalFlip | Mirror left-to-right | — | Symmetric lesions should be robust to reflection |
| MR3 | Zoom | Centre-crop 80% + resize | `factor=0.8` | Slight zoom should not change diagnosis |
| MR4 | Brightness | Increase brightness × contrast | `brightness=1.5, contrast=1.2` | Lighting variation in imaging |
| MR5 | GaussianNoise | Additive pixel noise | `σ=0.05` | Sensor noise robustness |
| MR6 | GaussianBlur | Low-pass Gaussian filter | `radius=2.0` | Defocus / motion blur |
| MR7 | Sharpening | Unsharp masking | `factor=2.0` | Post-processing artifact |
| MR8 | Saturation | Colour saturation reduction | `factor=0.5` | Camera colour profile variation |

---

## Output Files Reference

### `predictions_table.csv` — Column Definitions

| Column | Type | Description |
|---|---|---|
| `image_id` | str | ISIC image identifier |
| `ground_truth` | int | True class index (0–8) |
| `ground_truth_name` | str | Class abbreviation (e.g. MEL) |
| `mr_id` | str | Transform ID: "Original", "MR1"…"MR8" |
| `mr_name` | str | Human-readable transform name |
| `prediction` | int | Model's predicted class index |
| `prediction_name` | str | Predicted class name |
| `confidence` | float | Softmax probability for predicted class |
| `original_pred` | int | Model's prediction on original (unmodified) image |
| `original_pred_name` | str | Original prediction class name |
| `original_conf` | float | Confidence on original image |
| `prediction_stable` | bool | **True** when `prediction == original_pred` |
| `original_correct` | bool | True when original prediction matches ground truth |
| `transformed_correct` | bool | True when transformed prediction matches ground truth |

### GradCAM Figure Layout

Each wrong-prediction figure (`<image_id>_<MR_id>_gradcam.png`) contains:

```

 Original Image GradCAM (Original) 
 Pred: NV (92.3%) Attention region 

 Transformed Image GradCAM (Transform)
 MR3: Zoom Shifted attention 
 Pred: MEL (78.1%) explains flip 

 PREDICTION FLIP: NV → MEL
```

---

## Architecture Notes

### ConvNeXt-Base
- Pretrained on ImageNet-1K via `timm.create_model("convnext_base")`
- Fine-tuned with: AdamW + CosineAnnealingLR + AMP + label smoothing (0.1)
- GradCAM target layer: `model.stages[-1].blocks[-1]` (last ConvNeXt block)

### ISIC 2019 Classes
| Abbr. | Full Name |
|---|---|
| MEL | Melanoma |
| NV | Melanocytic Nevus |
| BCC | Basal Cell Carcinoma |
| AK | Actinic Keratosis |
| BKL | Benign Keratosis |
| DF | Dermatofibroma |
| VASC | Vascular Lesion |
| SCC | Squamous Cell Carcinoma |
| UNK | Unknown |

---

## Full Workflow (Quick Reference)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train (one-time, ~hours on GPU)
python src/train.py --data_dir data/isic2019 --epochs 30

# 3. Run metamorphic testing
python run_phase1.py --data_dir data/isic2019 --n_samples 100

# 4a. View results (stored mode)
streamlit run streamlit_app.py -- --mode stored

# 4b. Run pipeline + view (live mode)
streamlit run streamlit_app.py -- --mode live --data_dir data/isic2019

# 5. Random GradCAM showcase
python random_gradcam.py
```

---

## Team

| Name | Roll No |
|---|---|
| Abel Louis Fernandez | B23CS2103 |
| Akshay P Benjamin | B23CS2112 |
| Aparna S | B23CS2116 |
| Goutham C S | B23CS2130 |

**Guide:** Ms. Neethi Narayanan, Assistant Professor, Department of CSE
