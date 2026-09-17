"""
src/models/base.py — Model-Agnostic Abstract Vision Interface.

This abstract base class decouples the Metamorphic Circuit Breaker from any
underlying neural network architecture (ConvNeXt, ResNet, ViT, or remote cloud API).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np


class BaseVisionModel(ABC):
    """
    Standardized abstract interface that allows the Metamorphic Circuit Breaker
    to connect to ANY vision model architecture or external API.
    """

    @abstractmethod
    def predict(self, image: Image.Image) -> Tuple[int, str, float, List[float]]:
        """
        Runs forward inference on a single image.

        Args:
            image: PIL RGB Image.

        Returns:
            pred_idx   : Integer class index (e.g. 0-7).
            pred_label : Class name string (e.g. "MEL", "NV").
            confidence : Top-1 softmax probability (0.0 to 1.0).
            all_probs  : Full probability distribution over all classes.
        """
        pass

    @abstractmethod
    def explain(
        self,
        image: Image.Image,
        target_class: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Generates a 2D visual explanation heatmap [0.0, 1.0].

        Args:
            image: Input PIL Image.
            target_class: Class index to explain. If None, uses top-1 predicted class.

        Returns:
            2D numpy array of shape (H, W) with values normalized to [0.0, 1.0],
            or None if explainability is not supported for this model type.
        """
        pass
