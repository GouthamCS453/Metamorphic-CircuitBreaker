"""
random_gradcam.py — Randomly select one image from a Phase 1 run and generate
GradCAM heatmaps for all 8 metamorphic transformations.

The script produces a grid of 9 panels (Original + 8 MRs). Each panel shows
the image on the left half and its GradCAM overlay on the right half.
The figure is saved to ``outputs/run_*/gradcam/random_showcase/`` and,
unless ``--save_only`` is set, also displayed interactively.

Usage
-----
    # Use latest run, pick random image
    python random_gradcam.py

    # Specify a run directory and/or a specific image
    python random_gradcam.py --run_dir outputs/run_20240810_123456
    python random_gradcam.py --image_id ISIC_0024306

    # Save without displaying
    python random_gradcam.py --save_only

    # Set random seed for reproducible selection
    python random_gradcam.py --seed 7
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import (
    CLASS_NAMES, BEST_MODEL_PATH,
    get_device, load_model, load_json, get_latest_run_dir,
)
from src.gradcam import ConvNeXtGradCAM, overlay_heatmap_on_image
from src.metamorphic import MR_REGISTRY

DARK_BG  = "#0f0e17"
PANEL_BG = "#1a1a2e"
TEXT_CLR = "#e0e0e0"
RED_CLR  = "#ff6b6b"


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GradCAM showcase for a randomly selected image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run_dir",   type=str, default=None,
        help="Path to a run_* directory. Defaults to the most recent run.",
    )
    p.add_argument(
        "--image_id",  type=str, default=None,
        help="Specific ISIC image ID to visualise. Random if omitted.",
    )
    p.add_argument(
        "--seed",      type=int, default=None,
        help="Random seed for image selection.",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Override path to model checkpoint (.pth).",
    )
    p.add_argument(
        "--save_only", action="store_true",
        help="Save the figure without opening an interactive window.",
    )
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Resolve run directory ─────────────────────────────────────────────────
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = get_latest_run_dir()
        if run_dir is None:
            print("[ERROR] No run directories found.")
            print("        Run `python run_phase1.py` first.")
            sys.exit(1)

    print(f"[GradCAM Showcase] Run directory : {run_dir}")

    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        print(f"[ERROR] metadata.json not found in {run_dir}")
        sys.exit(1)

    meta       = load_json(meta_path)
    sample_ids = meta["sample_ids"]

    # ── Pick image ────────────────────────────────────────────────────────────
    if args.image_id:
        if args.image_id in sample_ids:
            image_id = args.image_id
        else:
            print(f"[WARNING] {args.image_id!r} not in sample list. Picking randomly.")
            image_id = random.Random(args.seed).choice(sample_ids)
    else:
        image_id = random.Random(args.seed).choice(sample_ids)

    print(f"[GradCAM Showcase] Selected image: {image_id}")

    orig_path = run_dir / "sample_images" / f"{image_id}.jpg"
    if not orig_path.exists():
        print(f"[ERROR] Image not found: {orig_path}")
        sys.exit(1)

    orig_image = Image.open(orig_path).convert("RGB")

    # ── Load model ────────────────────────────────────────────────────────────
    checkpoint = Path(args.checkpoint) if args.checkpoint else BEST_MODEL_PATH
    device     = get_device()
    print(f"[GradCAM Showcase] Loading model on {device}…")
    model      = load_model(checkpoint, device)
    gradcam    = ConvNeXtGradCAM(model, device)

    # ── Build panel list: Original + 8 MRs ───────────────────────────────────
    entries = [("Original", "Original", orig_image)]
    for mr in MR_REGISTRY:
        try:
            t_img = mr(orig_image)
        except Exception as exc:
            print(f"  [WARNING] {mr.id} transform failed: {exc}")
            t_img = orig_image
        entries.append((mr.id, mr.name, t_img))

    # ── Layout: 3 rows × 3 columns = 9 panels ────────────────────────────────
    ncols = 3
    nrows = 3   # ceil(9 / 3) == 3

    fig = plt.figure(figsize=(ncols * 5.5, nrows * 4.8))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.18)

    print("[GradCAM Showcase] Generating heatmaps for all 9 panels…")

    for idx, (mr_id, mr_name, img) in enumerate(entries):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        try:
            cam, pred_cls, conf = gradcam.generate(img)
            overlay             = overlay_heatmap_on_image(img, cam, out_size=112)

            # Side-by-side: original thumbnail | GradCAM overlay
            thumb    = np.array(img.resize((112, 112)))
            combined = np.concatenate([thumb, overlay], axis=1)  # (112, 224, 3)
            ax.imshow(combined)

        except Exception as exc:
            ax.set_facecolor(PANEL_BG)
            ax.text(
                0.5, 0.5, f"Error:\n{exc}",
                transform=ax.transAxes, ha="center", va="center",
                color=RED_CLR, fontsize=7, wrap=True,
            )
            pred_cls, conf = -1, 0.0

        pred_name = (
            CLASS_NAMES[pred_cls]
            if 0 <= pred_cls < len(CLASS_NAMES)
            else "n/a"
        )
        border_color = "#06d6a0" if mr_id == "Original" else TEXT_CLR
        ax.set_title(
            f"{mr_id}: {mr_name}\nPred: {pred_name}  ({conf:.1%})",
            color=border_color, fontsize=8.5, pad=5,
            fontfamily="monospace",
        )
        ax.axis("off")

        # Thin border to separate panels
        for spine in ax.spines.values():
            spine.set_edgecolor(PANEL_BG)

    # Hide any unused panels
    for idx in range(len(entries), nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    fig.suptitle(
        f"GradCAM Showcase  —  {image_id}\n"
        f"[ Left half: image  |  Right half: GradCAM attention overlay ]",
        color=TEXT_CLR, fontsize=12, fontweight="bold", y=1.01,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    save_dir = run_dir / "gradcam" / "random_showcase"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{image_id}_gradcam_showcase.png"
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    print(f"[GradCAM Showcase] Saved → {save_path}")

    if not args.save_only:
        plt.show()

    plt.close(fig)
    gradcam.remove_hooks()
    print("[GradCAM Showcase] Done.")


if __name__ == "__main__":
    main()
