"""
mcb_experiments/v4/test7_gradcam_audit.py — Step 10B: Grad-CAM Saliency Drift Audit.

Protocol Requirements (Section 9.2):
- OPEN vs CLOSED samples from Partition B.
- Target class FIXED to baseline prediction y_pred of original image:
  cam_orig = gradcam(model, x_orig, target=y_pred_orig)
  cam_trans = gradcam(model, x_trans, target=y_pred_orig)
- Metrics:
  - Cosine similarity
  - Top-20% IoU
  - Centroid distance (normalized in [0, 1] relative to image diagonal)
- Reports mean and SD for Tripped (OPEN) vs Stable (CLOSED).
- Saves comparison figures in mcb_experiments/results_v4/gradcam_panels/.
- Saves mcb_experiments/results_v4/gradcam_results.json.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "mcb_experiments" / "results_v4"
CSV_PATH = RESULTS_DIR / "per_sample_scores.csv"
CALIB_PATH = RESULTS_DIR / "calibrated_thresholds.json"
PANELS_DIR = RESULTS_DIR / "gradcam_panels"
PANELS_DIR.mkdir(parents=True, exist_ok=True)

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mobilenet_adapter import MobileNetAdapter
from src.utils import BEST_MODEL_PATH
from src.metamorphic_families import HIERARCHICAL_TEST_MATRIX

def compute_cosine_similarity(c1: np.ndarray, c2: np.ndarray) -> float:
    v1 = c1.flatten()
    v2 = c2.flatten()
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-10: return 1.0
    return float(np.dot(v1, v2) / denom)

def compute_top20_iou(c1: np.ndarray, c2: np.ndarray) -> float:
    t1 = np.percentile(c1, 80)
    t2 = np.percentile(c2, 80)
    b1 = (c1 >= t1)
    b2 = (c2 >= t2)
    union = (b1 | b2).sum()
    if union == 0: return 1.0
    return float((b1 & b2).sum() / union)

def compute_centroid_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    h, w = c1.shape
    diag = math.sqrt(h * h + w * w)
    w1 = c1 / (c1.sum() + 1e-10)
    w2 = c2 / (c2.sum() + 1e-10)
    y_idx, x_idx = np.indices((h, w))
    cx1 = (x_idx * w1).sum()
    cy1 = (y_idx * w1).sum()
    cx2 = (x_idx * w2).sum()
    cy2 = (y_idx * w2).sum()
    dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return float(dist / diag)

def run_gradcam_audit(n_samples: int = 50):
    print("=" * 70)
    print("STEP 10B: GRAD-CAM SALIENCY DRIFT AUDIT (Section 9.2)")
    print("=" * 70)

    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib = json.load(f)
    tau_star = calib["mcb_op_j"]["tau_fam"]
    theta_star = calib["mcb_op_j"]["theta_trip"]

    df = pd.read_csv(CSV_PATH)
    val_csv = PROJECT_ROOT / "checkpoints" / "val_split.csv"
    val_df = pd.read_csv(val_csv)
    id_to_path = dict(zip(val_df["image"].astype(str), val_df["image_path"]))

    df_B = df[df.partition == "B"].copy().reset_index(drop=True)
    c0_B = df_B.c0.to_numpy()
    M_B = np.stack([df_B[f"ir_{f}"].to_numpy() for f in ["geometric", "photometric", "sensor_noise"]], axis=1)
    cfs_B = (M_B >= tau_star).mean(axis=1)
    cbi_B = 0.50 * M_B.max(axis=1) + 0.35 * cfs_B + 0.15 * (1.0 - c0_B)
    df_B["cbi"] = cbi_B

    open_samples = df_B[df_B.cbi >= theta_star].head(n_samples)
    closed_samples = df_B[df_B.cbi < (0.5 * theta_star)].head(n_samples)

    print(f"Sampling {len(open_samples)} Tripped (OPEN) and {len(closed_samples)} Stable (CLOSED) samples...")

    adapter = MobileNetAdapter(checkpoint_path=str(BEST_MODEL_PATH))

    def load_img(sample_id):
        p = id_to_path[str(sample_id)]
        pth = Path(p)
        if not pth.is_absolute():
            cand = PROJECT_ROOT / p
            if cand.exists(): pth = cand
            else: pth = PROJECT_ROOT / "data" / "gtsrb" / p
        return Image.open(pth).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)

    # Use severe sensor noise (e.g. index 36: Noise severe) as audit transform
    sev_noise_test = HIERARCHICAL_TEST_MATRIX[36]

    metrics_tripped = {"cosine": [], "iou": [], "centroid": []}
    metrics_stable = {"cosine": [], "iou": [], "centroid": []}

    # 1. Evaluate Tripped (OPEN)
    for idx, (_, row) in enumerate(open_samples.iterrows()):
        sid = str(row["sample_id"])
        y_pred = int(row["y_pred"])
        img_orig = load_img(sid)
        img_trans = sev_noise_test(img_orig, seed=42)

        cam_orig = adapter.explain(img_orig, target_class=y_pred)
        cam_trans = adapter.explain(img_trans, target_class=y_pred)  # Target FIXED to baseline prediction

        if cam_orig is None or cam_trans is None: continue

        cos_sim = compute_cosine_similarity(cam_orig, cam_trans)
        iou_val = compute_top20_iou(cam_orig, cam_trans)
        cent_dist = compute_centroid_distance(cam_orig, cam_trans)

        metrics_tripped["cosine"].append(cos_sim)
        metrics_tripped["iou"].append(iou_val)
        metrics_tripped["centroid"].append(cent_dist)

        # Save first 3 panel visualizations
        if idx < 3:
            fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
            axes[0].imshow(img_orig)
            axes[0].set_title("Original Image")
            axes[1].imshow(cam_orig, cmap="jet")
            axes[1].set_title(f"Grad-CAM Orig (Class {y_pred})")
            axes[2].imshow(img_trans)
            axes[2].set_title("Transformed (Severe Noise)")
            axes[3].imshow(cam_trans, cmap="jet")
            axes[3].set_title(f"Grad-CAM Trans (Target {y_pred})")
            for ax in axes: ax.axis("off")
            plt.suptitle(f"Tripped OPEN Sample {sid} | IoU: {iou_val:.3f} | Cosine: {cos_sim:.3f}", fontsize=11)
            plt.tight_layout()
            plt.savefig(PANELS_DIR / f"panel_tripped_{idx+1}.png", dpi=200)
            plt.close()

    # 2. Evaluate Stable (CLOSED)
    for idx, (_, row) in enumerate(closed_samples.iterrows()):
        sid = str(row["sample_id"])
        y_pred = int(row["y_pred"])
        img_orig = load_img(sid)
        img_trans = sev_noise_test(img_orig, seed=42)

        cam_orig = adapter.explain(img_orig, target_class=y_pred)
        cam_trans = adapter.explain(img_trans, target_class=y_pred)

        if cam_orig is None or cam_trans is None: continue

        cos_sim = compute_cosine_similarity(cam_orig, cam_trans)
        iou_val = compute_top20_iou(cam_orig, cam_trans)
        cent_dist = compute_centroid_distance(cam_orig, cam_trans)

        metrics_stable["cosine"].append(cos_sim)
        metrics_stable["iou"].append(iou_val)
        metrics_stable["centroid"].append(cent_dist)

        if idx < 3:
            fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
            axes[0].imshow(img_orig)
            axes[0].set_title("Original Image")
            axes[1].imshow(cam_orig, cmap="jet")
            axes[1].set_title(f"Grad-CAM Orig (Class {y_pred})")
            axes[2].imshow(img_trans)
            axes[2].set_title("Transformed (Severe Noise)")
            axes[3].imshow(cam_trans, cmap="jet")
            axes[3].set_title(f"Grad-CAM Trans (Target {y_pred})")
            for ax in axes: ax.axis("off")
            plt.suptitle(f"Stable CLOSED Sample {sid} | IoU: {iou_val:.3f} | Cosine: {cos_sim:.3f}", fontsize=11)
            plt.tight_layout()
            plt.savefig(PANELS_DIR / f"panel_stable_{idx+1}.png", dpi=200)
            plt.close()

    summary_gradcam = {
        "stable": {
            "cosine_mean": float(np.mean(metrics_stable["cosine"])),
            "cosine_sd": float(np.std(metrics_stable["cosine"])),
            "iou_mean": float(np.mean(metrics_stable["iou"])),
            "iou_sd": float(np.std(metrics_stable["iou"])),
            "centroid_mean": float(np.mean(metrics_stable["centroid"])),
            "centroid_sd": float(np.std(metrics_stable["centroid"])),
        },
        "tripped": {
            "cosine_mean": float(np.mean(metrics_tripped["cosine"])),
            "cosine_sd": float(np.std(metrics_tripped["cosine"])),
            "iou_mean": float(np.mean(metrics_tripped["iou"])),
            "iou_sd": float(np.std(metrics_tripped["iou"])),
            "centroid_mean": float(np.mean(metrics_tripped["centroid"])),
            "centroid_sd": float(np.std(metrics_tripped["centroid"])),
        },
        "target_fixed_to_baseline_class": True,
        "n_samples_evaluated": {"tripped": len(metrics_tripped["cosine"]), "stable": len(metrics_stable["cosine"])},
    }

    print("\nGrad-CAM Saliency Drift Audit Results:")
    print(f"{'State':<10} | {'Cosine Sim (Mean ± SD)':<25} | {'Top-20% IoU (Mean ± SD)':<25} | {'Centroid Dist (Mean ± SD)':<25}")
    print("-" * 90)
    for st in ["stable", "tripped"]:
        d = summary_gradcam[st]
        print(f"{st.upper():<10} | {d['cosine_mean']:.3f} ± {d['cosine_sd']:.3f}             | {d['iou_mean']:.3f} ± {d['iou_sd']:.3f}             | {d['centroid_mean']:.3f} ± {d['centroid_sd']:.3f}")

    json_path = RESULTS_DIR / "gradcam_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_gradcam, f, indent=2)
    print(f"\n[Saved] {json_path}")
    print(f"[Saved] Inspection panels saved in {PANELS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    run_gradcam_audit(n_samples=50)
