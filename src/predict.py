"""
src/predict.py — Run inference on 100 sampled images with all 8 metamorphic
transformations and record prediction stability.

For each of the 100 images the runner:
  1. Records the model's prediction on the original image (baseline).
  2. Applies each of the 8 MRs and records the prediction on the
     transformed variant.
  3. Flags the prediction as *unstable* when the transformed prediction
     differs from the original.

The full result is saved as ``predictions_table.csv`` inside the run directory.

Columns
-------
image_id            : ISIC image identifier
ground_truth        : Integer class index (0-8)
ground_truth_name   : Class name string
mr_id               : "Original", "MR1" … "MR8"
mr_name             : Human-readable transform name
prediction          : Predicted class index
prediction_name     : Predicted class name
confidence          : Softmax probability for predicted class
original_pred       : Model's prediction on the unmodified image
original_pred_name  : Class name of original prediction
original_conf       : Confidence on original image
prediction_stable   : True when prediction == original_pred
original_correct    : True when original_pred == ground_truth
transformed_correct : True when prediction == ground_truth
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.utils import (
    CLASS_NAMES, CHECKPOINT_DIR,
    pil_to_tensor, get_device, load_model,
)
from src.metamorphic import MR_REGISTRY


# ─── Inference Runner ──────────────────────────────────────────────────────────

class PredictionRunner:
    """
    Wraps the model and runs batched inference for metamorphic testing.

    Args:
        model:  Loaded ConvNeXt model in ``eval()`` mode.
        device: Compute device.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model  = model
        self.device = device

    @torch.no_grad()
    def predict_pil(self, pil_image: Image.Image):
        """
        Run single-image inference.

        Args:
            pil_image: Input PIL Image.

        Returns:
            pred_class : argmax class index.
            confidence : softmax probability for *pred_class*.
            all_probs  : full softmax probability vector as a list.
        """
        tensor = pil_to_tensor(pil_image).to(self.device)
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze()
        pred   = int(probs.argmax().item())
        return pred, float(probs[pred].item()), probs.cpu().tolist()

    def run_on_sample(
        self,
        image_id: str,
        orig_image: Image.Image,
        ground_truth: int,
    ) -> list:
        """
        Run inference on the original image and all 8 MR variants.

        Args:
            image_id:     ISIC identifier string.
            orig_image:   Original PIL Image.
            ground_truth: Integer class index.

        Returns:
            List of record dicts (9 rows: Original + 8 MRs).
        """
        orig_pred, orig_conf, _ = self.predict_pil(orig_image)
        orig_correct = (orig_pred == ground_truth)

        orig_name = CLASS_NAMES[orig_pred] if orig_pred < len(CLASS_NAMES) else str(orig_pred)

        def _make_row(mr_id, mr_name, pred, conf, stable):
            pred_name = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else str(pred)
            return {
                "image_id":           image_id,
                "ground_truth":       ground_truth,
                "ground_truth_name":  CLASS_NAMES[ground_truth] if ground_truth < len(CLASS_NAMES) else str(ground_truth),
                "mr_id":              mr_id,
                "mr_name":            mr_name,
                "prediction":         pred,
                "prediction_name":    pred_name,
                "confidence":         round(conf, 6),
                "original_pred":      orig_pred,
                "original_pred_name": orig_name,
                "original_conf":      round(orig_conf, 6),
                "prediction_stable":  stable,
                "original_correct":   orig_correct,
                "transformed_correct": (pred == ground_truth),
            }

        records = [_make_row("Original", "Original", orig_pred, orig_conf, True)]

        for mr in MR_REGISTRY:
            try:
                t_img = mr(orig_image)
            except Exception as exc:
                print(f"  [WARNING] {mr.id} failed on {image_id}: {exc}")
                continue
            t_pred, t_conf, _ = self.predict_pil(t_img)
            records.append(_make_row(mr.id, mr.name, t_pred, t_conf, t_pred == orig_pred))

        return records


# ─── Sampling ────────────────────────────────────────────────────────────────────

def sample_images_from_val_split(
    n_samples: int = 100,
    seed: int = 42,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Draw *n_samples* images from the validation split saved by ``train.py``.

    If ``checkpoints/val_split.csv`` exists it is used directly; otherwise
    the split is recomputed from *data_dir* (requires training CSV to be
    present).

    Args:
        n_samples: Number of images to sample.
        seed:      Random seed for reproducible sampling.
        data_dir:  ISIC 2019 root directory (fallback only).

    Returns:
        DataFrame with columns ``image``, ``image_path``, ``label``.
    """
    val_csv = CHECKPOINT_DIR / "val_split.csv"

    if val_csv.exists():
        df = pd.read_csv(val_csv)
        print(f"[Predict] Loaded val split: {len(df)} images")
    elif data_dir is not None:
        # Lazy import to avoid circular dependency with train.py
        from src.train import load_isic_dataframe
        from sklearn.model_selection import train_test_split

        df_full = load_isic_dataframe(data_dir)
        _, df   = train_test_split(
            df_full, test_size=0.30,
            stratify=df_full["label"], random_state=seed,
        )
        print(f"[Predict] Recomputed val split: {len(df)} images")
    else:
        raise FileNotFoundError(
            "checkpoints/val_split.csv not found.\n"
            "Run training first or pass --data_dir."
        )

    n_samples = min(n_samples, len(df))
    return df.sample(n=n_samples, random_state=seed).reset_index(drop=True)


# ─── Main runner ────────────────────────────────────────────────────────────────

def run_predictions(
    model: torch.nn.Module,
    device: torch.device,
    sample_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Apply all 8 MRs to each image in *sample_df*, run inference, and save
    ``predictions_table.csv`` inside *output_dir*.

    Side-effects
    ------------
    * Saves original images to ``output_dir/sample_images/<image_id>.jpg``.
    * Saves transformed images to
      ``output_dir/transformed/<MR_id>/<image_id>.jpg``.
    * Writes ``predictions_table.csv``.

    Args:
        model:      Trained ConvNeXt model.
        device:     Compute device.
        sample_df:  DataFrame with columns ``image``, ``image_path``, ``label``.
        output_dir: Run-level output directory.

    Returns:
        DataFrame with one row per (image, transform) pair.
    """
    runner = PredictionRunner(model, device)

    sample_img_dir  = output_dir / "sample_images"
    transformed_dir = output_dir / "transformed"
    sample_img_dir.mkdir(parents=True, exist_ok=True)

    for mr in MR_REGISTRY:
        (transformed_dir / mr.id).mkdir(parents=True, exist_ok=True)

    all_records = []
    print(f"\n[Predict] Running inference on {len(sample_df)} images × 8 MRs…")

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), unit="img"):
        image_id  = str(row["image"])
        img_path  = str(row["image_path"])
        label     = int(row["label"])

        try:
            orig_image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"  [WARNING] Cannot open {img_path}: {exc}")
            continue

        # Persist original (224 × 224 JPEG) for GradCAM and Streamlit
        dest = sample_img_dir / f"{image_id}.jpg"
        if not dest.exists():
            orig_image.resize((224, 224)).save(dest, quality=95)

        # Persist each transformed variant
        for mr in MR_REGISTRY:
            try:
                t_img = mr(orig_image)
                t_dest = transformed_dir / mr.id / f"{image_id}.jpg"
                if not t_dest.exists():
                    t_img.resize((224, 224)).save(t_dest, quality=95)
            except Exception:
                pass

        records = runner.run_on_sample(image_id, orig_image, label)
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    csv_path = output_dir / "predictions_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Predict] Saved {len(df)} rows → {csv_path}")
    return df
