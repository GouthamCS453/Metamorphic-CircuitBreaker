"""
mcb/shifts.py - Realistic input degradations used by the graded-shift test (C) and the
fault-attribution test (G).  These are DISTRIBUTION SHIFTS applied to the input (not part
of the metamorphic bank).  Each shift has 3 severity levels (1 = mild ... 3 = strong).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .transforms import op_jpeg, op_noise, op_rotate


def _color_cast(img: Image.Image, s: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr[..., 0] *= (1.0 + s)
    arr[..., 2] *= (1.0 - s)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


SHIFTS: Dict[str, dict] = {
    "blur": dict(family="sensor_noise", levels=[1.0, 2.0, 3.5],
                 fn=lambda im, p: im.filter(ImageFilter.GaussianBlur(radius=p))),
    "noise": dict(family="sensor_noise", levels=[0.03, 0.07, 0.12],
                  fn=lambda im, p: op_noise(im, p, seed=7)),
    "jpeg": dict(family="sensor_noise", levels=[60, 35, 15], fn=lambda im, p: op_jpeg(im, p)),
    "darken": dict(family="photometric", levels=[0.75, 0.55, 0.35],
                   fn=lambda im, p: ImageEnhance.Brightness(im).enhance(p)),
    "color_cast": dict(family="photometric", levels=[0.15, 0.30, 0.50], fn=_color_cast),
    "rotate": dict(family="geometric", levels=[10.0, 20.0, 35.0], fn=lambda im, p: op_rotate(im, p)),
}


def make_shift(name: str, level: int) -> Callable[[Image.Image], Image.Image]:
    """level in {1,2,3}."""
    spec = SHIFTS[name]
    p = spec["levels"][level - 1]
    return lambda im, f=spec["fn"], v=p: f(im, v)
