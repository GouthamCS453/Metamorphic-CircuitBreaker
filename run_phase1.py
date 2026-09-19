"""
run_phase1.py — Main orchestration script for Metamorphic Circuit Breaker Phase 1.

Pipeline
--------
1.  Verify ``checkpoints/best_model.pth`` exists  (exits gracefully if missing).
2.  Create a timestamped output directory under ``outputs/``.
3.  Load the trained MobileNetV3-Small model.
4.  Sample *n_samples* images from the validation split.
5.  Apply all 8 MRs + run inference  → ``predictions_table.csv``.
6.  Generate GradCAM comparison figures for every wrong prediction.
7.  Generate 4 graphical plots (consistency bar, wrong counts, pie, heatmap).
8.  Print a summary report to stdout.

Usage
-----
    python run_phase1.py
    python run_phase1.py --data_dir data/gtsrb --n_samples 100
    python run_phase1.py --skip_gradcam          # faster, skips GradCAM step
    python run_phase1.py --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations
from PIL import Image
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import (
    BEST_MODEL_PATH, setup_output_dir, save_json,
    get_device,
)
from src.models.mobilenet_adapter import MobileNetAdapter
from src.predict import sample_images_from_val_split, run_predictions
from src.visualize import generate_all_plots


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Metamorphic Circuit Breaker Phase 1 pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="data/gtsrb",
        help="Path to the GTSRB dataset root directory",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of validation images to sample for metamorphic testing",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible image sampling",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None, help="Override path to trained model checkpoint (.pth)",
    )
    p.add_argument(
        "--skip_gradcam",
        action="store_true",
        help="Skip GradCAM generation step (much faster, no heatmap figures)",
    )
    return p.parse_args()


# ─── Checkpoint Gate ──────────────────────────────────────────────────────────

def verify_checkpoint(checkpoint_path: Path) -> None:
    """
    Abort with a clear, actionable message if the checkpoint is missing.

    This is the mandatory guard that prevents Phase 1 from running without
    a trained model.
    """
    if not checkpoint_path.exists():
        print()
        print("=" * 66)
        print("  ❌  MODEL CHECKPOINT NOT FOUND")
        print("=" * 66)
        print(f"\n  Expected location : {checkpoint_path}")
        print()
        print("  The Phase 1 pipeline requires a trained MobileNetV3-Small model.")
        print("  Please run training first:")
        print()
        print("    python src/train.py --data_dir data/gtsrb --epochs 30")
        print()
        print("  If you already have a checkpoint elsewhere, specify it:")
        print()
        print("    python run_phase1.py --checkpoint path/to/model.pth")
        print("=" * 66)
        print()
        sys.exit(1)


# ─── Summary Report ───────────────────────────────────────────────────────────

def print_summary(df, run_dir: Path, elapsed: float) -> None:
    """Print a formatted Phase 1 run summary to stdout."""
    mr_only  = df[df["mr_id"] != "Original"]
    total    = len(mr_only)
    stable   = int(mr_only["prediction_stable"].sum())
    unstable = total - stable

    per_mr = (
        mr_only.groupby("mr_name")["prediction_stable"]
        .agg(["sum", "count"])
        .assign(pct=lambda x: 100.0 * x["sum"] / x["count"])
    )

    print()
    print("=" * 66)
    print("  📊  PHASE 1 — RUN SUMMARY")
    print("=" * 66)
    print(f"\n  Run directory  : {run_dir}")
    print(f"  Images tested  : {df['image_id'].nunique()}")
    print(f"  Transforms     : {mr_only['mr_id'].nunique()}")
    print(f"  Total tests    : {total}")
    print(f"  Stable preds   : {stable:>5}  ({100 * stable / total:.1f}%)")
    print(f"  Flipped preds  : {unstable:>5}  ({100 * unstable / total:.1f}%)")

    print(f"\n  {'Transform':<22} {'Stable':>7} {'Total':>7} {'%':>7}  Bar")
    print(f"  {'-' * 58}")
    for name, row in per_mr.iterrows():
        bar_len  = int(row["pct"] / 5)
        bar      = "█" * bar_len + "░" * (20 - bar_len)
        print(
            f"  {name:<22} {int(row['sum']):>7} {int(row['count']):>7} "
            f"{row['pct']:>6.1f}%  {bar}"
        )

    print(f"\n  ⏱  Elapsed : {elapsed:.1f} s")
    print("=" * 66)
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t0   = time.time()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else BEST_MODEL_PATH
    data_dir        = Path(args.data_dir)

    # ── 1. Checkpoint gate ────────────────────────────────────────────────────
    print("\n[Phase 1] Verifying model checkpoint…")
    verify_checkpoint(checkpoint_path)
    print(f"[Phase 1] ✓ Checkpoint found: {checkpoint_path}")

    # ── 2. Create output directory ────────────────────────────────────────────
    run_dir = setup_output_dir()
    print(f"[Phase 1] Output directory  : {run_dir}")

    # ── 3. Load model ─────────────────────────────────────────────────────────
    device = get_device()
    print(f"[Phase 1] Loading MobileNetV3-Small adapter on : {device}")
    model = MobileNetAdapter(checkpoint_path, device)

    # ── 4. Sample images ──────────────────────────────────────────────────────
    print(f"[Phase 1] Sampling {args.n_samples} validation images (seed={args.seed})…")
    sample_df = sample_images_from_val_split(
        n_samples=args.n_samples,
        seed=args.seed,
        data_dir=data_dir,
    )

    # Save run metadata
    save_json(
        {
            "run_dir":    str(run_dir),
            "n_samples":  args.n_samples,
            "seed":       args.seed,
            "checkpoint": str(checkpoint_path),
            "sample_ids": sample_df["image"].tolist(),
        },
        run_dir / "metadata.json",
    )

    # ── 5. Run inference ──────────────────────────────────────────────────────
    df = run_predictions(model, sample_df, run_dir)

    # ── 6. GradCAM ────────────────────────────────────────────────────────────
    if not args.skip_gradcam:
        print("\n[Phase 1] Generating Grad-CAM explanations…")

        gradcam_dir = run_dir / "gradcam"
        gradcam_dir.mkdir(parents=True, exist_ok=True)

        generated = 0

        for _, row in df[df["prediction_stable"] == False].iterrows():
            image_id = str(row["image_id"])
            mr_id = str(row["mr_id"])

            if mr_id == "Original":
                continue

            image_path = run_dir / "transformed" / mr_id / f"{image_id}.jpg"

            if not image_path.exists():
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                cam = model.explain(
                    image,
                    target_class=int(row["prediction"]),
                )

                if cam is not None:
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(6, 5))
                    plt.imshow(image)
                    plt.imshow(cam, cmap="jet", alpha=0.45)
                    plt.axis("off")
                    plt.title(
                        f"{image_id} | {mr_id} | "
                        f"Pred: {row['prediction_name']}"
                    )
                    plt.tight_layout()

                    output_path = gradcam_dir / f"{image_id}_{mr_id}.png"
                    plt.savefig(output_path, dpi=150, bbox_inches="tight")
                    plt.close()

                    generated += 1

            except Exception as exc:
                print(
                    f"  [WARNING] Grad-CAM failed for "
                    f"{image_id}/{mr_id}: {exc}"
                )

        print(f"[Phase 1] Generated {generated} Grad-CAM figures.")
    else:
        print("[Phase 1] GradCAM skipped (--skip_gradcam flag set).")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print("\n[Phase 1] Generating visualisation plots…")
    generate_all_plots(df, run_dir)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    print_summary(df, run_dir, time.time() - t0)


if __name__ == "__main__":
    main()
