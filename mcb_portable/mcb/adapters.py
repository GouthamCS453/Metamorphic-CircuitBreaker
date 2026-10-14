"""
mcb/adapters.py - Model adapters.

The framework only needs ONE thing from a model:

    model.predict_proba(images: List[PIL.Image]) -> np.ndarray of shape (len(images), num_classes)

(softmax probabilities, rows sum to 1).  Optional attributes: `device_name`, `sync()`.

Use `TorchClassifier` for any torch nn.Module that returns logits, or `CallableClassifier`
to wrap any function (Keras, sklearn, ONNX, API call, ...).
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from PIL import Image


class TorchClassifier:
    def __init__(self, model, preprocess: Callable, device: Optional[str] = None,
                 batch_size: int = 32, amp: bool = False, apply_softmax: bool = True):
        import torch
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.amp = amp and self.device.type == "cuda"
        self.apply_softmax = apply_softmax
        self.device_name = (torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu")

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()

    def predict_proba(self, images: List[Image.Image]) -> np.ndarray:
        torch = self.torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                x = torch.stack([self.preprocess(im) for im in images[i:i + self.batch_size]]).to(self.device)
                if self.amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = self.model(x)
                else:
                    logits = self.model(x)
                logits = logits.float()
                p = torch.softmax(logits, dim=1) if self.apply_softmax else logits
                outs.append(p.cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float64)


class CallableClassifier:
    """Wrap `fn(list_of_PIL) -> probs (n, C)`."""

    def __init__(self, fn: Callable, device_name: str = "cpu"):
        self.fn = fn
        self.device_name = device_name

    def sync(self):
        pass

    def predict_proba(self, images: List[Image.Image]) -> np.ndarray:
        return np.asarray(self.fn(images), dtype=np.float64)
