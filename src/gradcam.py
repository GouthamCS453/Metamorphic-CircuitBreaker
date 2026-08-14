"""
src/gradcam.py — Gradient-weighted Class Activation Mapping (GradCAM) for
ConvNeXt models.

Implementation notes
--------------------
* Target layer : ``model.stages[-1].blocks[-1]`` (last block of last stage).
* Gradient source: ``torch.autograd.grad`` with ``retain_graph=False``,
  preceded by ``out.retain_grad()`` inside the forward hook so that the
  gradient accumulates on the non-leaf activation tensor.
* NCHW vs NHWC  : timm's ConvNeXt may output either format depending on
  the version; both are handled by inspecting the spatial dimensions.

Public API
----------
* :class:`ConvNeXtGradCAM`              — hook-based GradCAM extractor.
* :func:`overlay_heatmap_on_image`      — blend heatmap onto PIL image.
* :func:`save_gradcam_comparison`       — 4-panel side-by-side figure.
* :func:`generate_all_wrong_prediction_gradcams` — batch driver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.utils import CLASS_NAMES, pil_to_tensor, get_device


# ─── GradCAM Core ─────────────────────────────────────────────────────────────

class ConvNeXtGradCAM:
    """
    GradCAM extractor for ConvNeXt-Base (timm implementation).

    Usage::

        gradcam = ConvNeXtGradCAM(model, device)
        cam, pred_cls, conf = gradcam.generate(pil_image)
        gradcam.remove_hooks()
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None) -> None:
        self.model  = model
        self.device = device or get_device()
        self._activations: Optional[torch.Tensor] = None
        self._handles: list = []
        self._register_hooks()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_target_layer(self) -> torch.nn.Module:
        """
        Return the last block of the last ConvNeXt stage.

        Falls back to the entire last stage if ``blocks`` is not accessible.
        """
        try:
            return self.model.stages[-1].blocks[-1]
        except AttributeError:
            return self.model.stages[-1]

    def _to_nchw(self, t: torch.Tensor) -> torch.Tensor:
        """
        Ensure a 4-D tensor is in NCHW format.

        timm ConvNeXt may output (B, H, W, C) in NHWC mode; this helper
        detects the layout by comparing the channel dim to the spatial dims
        and permutes if necessary.
        """
        if t.ndim != 4:
            return t
        # If dim-1 < dim-1 compared to the last dim, it looks like NHWC
        # (height and width are larger than channels in most feature maps)
        if t.shape[1] < t.shape[2] and t.shape[1] < t.shape[3]:
            # (B, H, W, C) → (B, C, H, W)
            return t.permute(0, 3, 1, 2).contiguous()
        return t

    def _register_hooks(self) -> None:
        target = self._get_target_layer()

        def fwd_hook(module, inp, out):
            # Retain gradient on the non-leaf activation tensor so that
            # torch.autograd.grad can compute it without retain_graph=True.
            act = self._to_nchw(out)
            act.retain_grad()
            self._activations = act

        self._handles.append(target.register_forward_hook(fwd_hook))

    def remove_hooks(self) -> None:
        """Deregister all hooks. Call this when done to free resources."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.enable_grad()
    def generate(
        self,
        pil_image: Image.Image,
        target_class: Optional[int] = None,
    ):
        """
        Compute a GradCAM heatmap for the given PIL image.

        Args:
            pil_image:    Input PIL Image (any size; will be resized
                          to 224 × 224 internally).
            target_class: Class index to explain. If None, the model's
                          predicted class is used.

        Returns:
            cam          : float32 ndarray of shape (H, W) in [0, 1],
                           where H × W matches the feature-map spatial size.
            pred_class   : Predicted (or specified) class index.
            confidence   : Softmax probability for ``pred_class``.
        """
        self.model.eval()
        self.model.zero_grad()

        tensor = pil_to_tensor(pil_image).to(self.device)

        # Forward pass — activations captured by hook
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(logits.argmax(1).item())
        confidence = float(probs[0, target_class].item())

        if self._activations is None:
            raise RuntimeError(
                "Forward hook did not fire. Verify the target layer exists."
            )

        # Compute gradient of class score w.r.t. activations
        score = logits[0, target_class]
        grads = torch.autograd.grad(
            score, self._activations, retain_graph=False, create_graph=False
        )[0]                                              # (1, C, H, W)

        # Global-average-pool the gradients → channel weights
        weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self._activations).sum(dim=1)   # (1, H, W)
        cam = F.relu(cam).squeeze()                      # (H, W)
        cam = cam.detach().cpu().numpy()

        # Normalise to [0, 1]
        mn, mx = cam.min(), cam.max()
        cam = (cam - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(cam)

        return cam, target_class, confidence


# ─── Overlay Helper ───────────────────────────────────────────────────────────

def overlay_heatmap_on_image(
    pil_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.50,
    colormap: int = cv2.COLORMAP_JET,
    out_size: int = 224,
) -> np.ndarray:
    """
    Blend a GradCAM heatmap onto a PIL image.

    Args:
        pil_image: Source PIL Image (resized to *out_size* × *out_size*).
        cam:       Float32 heatmap array in [0, 1].
        alpha:     Heatmap opacity (0 = invisible, 1 = full heatmap).
        colormap:  OpenCV colourmap constant (default ``COLORMAP_JET``).
        out_size:  Output square size in pixels.

    Returns:
        RGB uint8 numpy array of shape (*out_size*, *out_size*, 3).
    """
    img_np = np.array(pil_image.resize((out_size, out_size))).astype(np.float32) / 255.0

    cam_resized = cv2.resize(cam, (out_size, out_size))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = alpha * heatmap + (1.0 - alpha) * img_np
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)


# ─── Comparison Figure ────────────────────────────────────────────────────────

def save_gradcam_comparison(
    original_image: Image.Image,
    transformed_image: Image.Image,
    gradcam: ConvNeXtGradCAM,
    mr_id: str,
    mr_name: str,
    image_id: str,
    output_dir: Path,
    ground_truth_cls: Optional[int] = None,
) -> Path:
    """
    Build and save a 4-panel GradCAM comparison figure.

    Panel layout::

        [Original Image]     [Original GradCAM]
        [Transformed Image]  [Transformed GradCAM]

    Each panel is annotated with the predicted class and confidence.
    A banner at the top highlights the prediction flip.

    Args:
        original_image:    Unmodified PIL Image.
        transformed_image: MR-transformed PIL Image.
        gradcam:           Initialised :class:`ConvNeXtGradCAM` instance.
        mr_id:             MR identifier string (e.g. ``"MR3"``).
        mr_name:           Human-readable MR name (e.g. ``"Zoom"``).
        image_id:          ISIC image identifier (used in filename).
        output_dir:        Directory in which to save the figure.
        ground_truth_cls:  Ground-truth class index, or None if unknown.

    Returns:
        Path to the saved PNG file.
    """
    DARK_BG = "#0f0e17"

    # Generate heatmaps
    orig_cam,  orig_pred,  orig_conf  = gradcam.generate(original_image)
    trans_cam, trans_pred, trans_conf = gradcam.generate(transformed_image)

    orig_overlay  = overlay_heatmap_on_image(original_image,    orig_cam)
    trans_overlay = overlay_heatmap_on_image(transformed_image, trans_cam)

    orig_lbl  = CLASS_NAMES[orig_pred]  if orig_pred  < len(CLASS_NAMES) else str(orig_pred)
    trans_lbl = CLASS_NAMES[trans_pred] if trans_pred < len(CLASS_NAMES) else str(trans_pred)
    gt_lbl    = CLASS_NAMES[ground_truth_cls] if (
        ground_truth_cls is not None and ground_truth_cls < len(CLASS_NAMES)
    ) else "?"

    # Build figure
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.12)

    panels = [
        (gs[0, 0], np.array(original_image.resize((224, 224))),
         f"Original\nPred: {orig_lbl}  ({orig_conf:.1%})"),
        (gs[0, 1], orig_overlay,
         f"GradCAM — Original\nModel attention region"),
        (gs[1, 0], np.array(transformed_image.resize((224, 224))),
         f"{mr_id}: {mr_name}\nPred: {trans_lbl}  ({trans_conf:.1%})"),
        (gs[1, 1], trans_overlay,
         f"GradCAM — {mr_name}\nAttention shift → explains flip"),
    ]

    for spec, arr, title in panels:
        ax = fig.add_subplot(spec)
        ax.imshow(arr)
        ax.set_title(title, color="#e0e0e0", fontsize=10, pad=6,
                     fontfamily="monospace")
        ax.axis("off")

    fig.suptitle(
        f"⚠  PREDICTION FLIP DETECTED\n"
        f"Image: {image_id}  |  Ground Truth: {gt_lbl}  |  "
        f"Transform: {mr_id} ({mr_name})\n"
        f"Original Prediction: {orig_lbl}  →  Transformed Prediction: {trans_lbl}",
        color="#ff6b6b", fontsize=11, fontweight="bold", y=1.02,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{image_id}_{mr_id}_gradcam.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=120,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path


# ─── Batch Driver ─────────────────────────────────────────────────────────────

def generate_all_wrong_prediction_gradcams(
    predictions_df,
    gradcam: ConvNeXtGradCAM,
    sample_images_dir: Path,
    transformed_dir: Path,
    output_dir: Path,
) -> list:
    """
    Generate GradCAM comparison figures for every wrong prediction in
    *predictions_df*.

    A "wrong prediction" is any row where ``prediction_stable == False``
    (i.e. the transformed image received a different predicted class than
    the original image).

    Args:
        predictions_df:    DataFrame returned by :func:`run_predictions`.
        gradcam:           Initialised :class:`ConvNeXtGradCAM` instance.
        sample_images_dir: ``run_dir/sample_images/`` directory.
        transformed_dir:   ``run_dir/transformed/`` directory.
        output_dir:        ``run_dir/gradcam/`` directory.

    Returns:
        List of paths to saved PNG figures.
    """
    wrong_df = predictions_df[
        (predictions_df["mr_id"] != "Original") &
        (~predictions_df["prediction_stable"])
    ]
    saved = []
    print(f"[GradCAM] Generating comparisons for {len(wrong_df)} wrong predictions…")

    for _, row in wrong_df.iterrows():
        image_id = str(row["image_id"])
        mr_id    = str(row["mr_id"])
        mr_name  = str(row["mr_name"])
        gt       = int(row["ground_truth"]) if "ground_truth" in row else None

        orig_path  = sample_images_dir / f"{image_id}.jpg"
        trans_path = transformed_dir   / mr_id / f"{image_id}.jpg"

        if not orig_path.exists() or not trans_path.exists():
            continue

        try:
            orig_img  = Image.open(orig_path).convert("RGB")
            trans_img = Image.open(trans_path).convert("RGB")
            path = save_gradcam_comparison(
                orig_img, trans_img, gradcam,
                mr_id, mr_name, image_id,
                output_dir / "wrong_predictions",
                ground_truth_cls=gt,
            )
            saved.append(path)
        except Exception as exc:
            print(f"  [WARNING] GradCAM failed for {image_id}/{mr_id}: {exc}")

    print(f"[GradCAM] Saved {len(saved)} comparison figure(s) → {output_dir}")
    return saved
