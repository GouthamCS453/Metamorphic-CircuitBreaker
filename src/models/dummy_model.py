from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from src.models.base import BaseVisionModel


class DummyVisionModel(BaseVisionModel):
    """
    Random-prediction mock adapter for BaseVisionModel.

    Useful for:
    - Demonstrating model-agnostic plug-in to another team member.
    - Unit-testing the circuit breaker without ConvNeXt / GPU.

    Set flip_rate > 0 to simulate an unstable model that triggers
    HALF_OPEN or OPEN circuit-breaker states.
    """

    def __init__(
        self,
        num_classes: int = 8,
        class_names: Optional[List[str]] = None,
        seed: int = 42,
        flip_rate: float = 0.3,
    ) -> None:
        self.num_classes = num_classes
        self.class_names = class_names or [f"CLASS_{i}" for i in range(num_classes)]
        self.seed = seed
        self.flip_rate = flip_rate
        rng = np.random.RandomState(seed)
        self._base_probs = rng.dirichlet(np.ones(num_classes)).astype(np.float32)
        self._base_idx = int(np.argmax(self._base_probs))

    def predict(self, image: Image.Image) -> Tuple[int, str, float, List[float]]:
        arr = np.array(image, dtype=np.float32)
        img_hash = int(arr.mean() * 1000) % (2 ** 16)
        rng = np.random.RandomState(self.seed ^ img_hash)
        if rng.random() < self.flip_rate:
            probs = rng.dirichlet(np.ones(self.num_classes)).astype(np.float32)
        else:
            probs = self._base_probs.copy()
        pred_idx = int(np.argmax(probs))
        return pred_idx, self.class_names[pred_idx], float(probs[pred_idx]), probs.tolist()

    def explain(
        self,
        image: Image.Image,
        target_class: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        rng = np.random.RandomState(self.seed + (target_class or 0))
        cam = rng.rand(7, 7).astype(np.float32)
        for i in range(7):
            for j in range(7):
                cam[i, j] *= np.exp(-((i - 3.0) ** 2 + (j - 3.0) ** 2) / 5.0)
        mn, mx = cam.min(), cam.max()
        return (cam - mn) / (mx - mn + 1e-8)
