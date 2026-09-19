"""
src/models/mobilenet_adapter.py — MobileNetV3-Small implementation
of BaseVisionModel.

Wraps the MobileNetV3-Small checkpoint with:
- .predict(): Runs forward pass, applies softmax, and returns
  (class_idx, label, confidence, probabilities).
- .explain(): Generates a Grad-CAM heatmap for MobileNetV3-Small.
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


class MobileNetAdapter(BaseVisionModel):
    """
    Adapter bridging MobileNetV3-Small to the model-agnostic
    BaseVisionModel interface.
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

    def predict(
        self,
        image: Image.Image,
    ) -> Tuple[int, str, float, List[float]]:
        tensor = pil_to_tensor(image).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())

        label = (
            CLASS_NAMES[pred_idx]
            if pred_idx < len(CLASS_NAMES)
            else str(pred_idx)
        )

        all_probs = probs.cpu().tolist()

        return pred_idx, label, confidence, all_probs

    def explain(
        self,
        image: Image.Image,
        target_class: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Generate a Grad-CAM heatmap for MobileNetV3-Small.
        """
        tensor = pil_to_tensor(image).to(self.device)

        # Last convolutional feature layer before global pooling.
        target_layer = self.model.blocks[5][0].conv

        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(
            backward_hook
        )

        try:
            self.model.zero_grad()

            logits = self.model(tensor)

            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            score = logits[0, target_class]
            score.backward()

            activation = activations[0]
            gradient = gradients[0]

            # Grad-CAM channel weights.
            weights = gradient.mean(dim=(2, 3), keepdim=True)

            # Weighted combination of feature maps.
            cam = (weights * activation).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            # Resize CAM to original image dimensions.
            cam = F.interpolate(
                cam,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )

            cam = cam.squeeze().detach().cpu().numpy()

            # Normalize to [0, 1].
            cam_min = cam.min()
            cam_max = cam.max()

            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            return cam

        finally:
            forward_handle.remove()
            backward_handle.remove()

    def close(self) -> None:
        pass