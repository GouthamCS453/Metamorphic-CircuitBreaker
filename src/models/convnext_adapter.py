"""
src/models/convnext_adapter.py — ConvNeXt implementation of BaseVisionModel.

Wraps the PyTorch ConvNeXt-Base checkpoint with:
- .predict(): Runs forward pass, applies softmax, returns (class_idx, label, conf, probs).
- .explain(): Hooks last convolutional stage to generate GradCAM attention heatmaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.base import BaseVisionModel
from src.utils import (
    CLASS_NAMES,
    BEST_MODEL_PATH,
    get_device,
    load_model,
    pil_to_tensor,
)
from src.gradcam import ConvNeXtGradCAM


class ConvNeXtAdapter(BaseVisionModel):
    """
    Adapter bridging ConvNeXt-Base to the model-agnostic BaseVisionModel interface.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or get_device()
        ckpt = Path(checkpoint_path) if checkpoint_path else BEST_MODEL_PATH
        self.model = load_model(ckpt, self.device)
        self.model.eval()
        self._gradcam: Optional[ConvNeXtGradCAM] = None

    def predict(self, image: Image.Image) -> Tuple[int, str, float, List[float]]:
        tensor = pil_to_tensor(image).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())
        label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
        all_probs = probs.cpu().tolist()

        return pred_idx, label, confidence, all_probs

    def explain(
        self,
        image: Image.Image,
        target_class: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if self._gradcam is None:
            self._gradcam = ConvNeXtGradCAM(self.model, self.device)

        cam, _, _ = self._gradcam.generate(image, target_class=target_class)
        return cam

    def close(self) -> None:
        if self._gradcam is not None:
            self._gradcam.remove_hooks()
            self._gradcam = None
