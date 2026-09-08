"""
src/metamorphic.py — Eight Metamorphic Relations (MRs) for the Metamorphic
Circuit Breaker framework.

Each MR is a callable object that accepts a PIL Image and returns a transformed
PIL Image that is semantically equivalent for skin-lesion classification.

Transformation table
--------------------
ID   Name             Transformation           Key Parameter
MR1  Rotation         Rotate image             15° clockwise
MR2  HorizontalFlip   Mirror left-right        —
MR3  Zoom             Center-crop + resize     factor = 0.8
MR4  Brightness       Brightness × Contrast    1.5×, 1.2×
MR5  GaussianNoise    Additive noise           σ = 0.05
MR6  GaussianBlur     Low-pass filter          radius = 2.0
MR7  Sharpening       Unsharp masking          factor = 2.0×
MR8  Saturation       Colour saturation        0.5×
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ─── Base Data Class ──────────────────────────────────────────────────────────

@dataclass
class MetamorphicRelation:
    """
    Encapsulates one metamorphic transformation.

    Attributes:
        id          : Short identifier, e.g. ``"MR1"``.
        name        : Human-readable name, e.g. ``"Rotation"``.
        description : One-sentence explanation of what the transform does.
        parameters  : Dict of named parameters used by the transform.
    """

    id: str
    name: str
    description: str
    parameters: Dict
    _fn: Callable[[Image.Image], Image.Image] = field(repr=False)

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply this MR to *image* and return the transformed PIL Image."""
        return self._fn(image)


# ─── Transform Implementations ────────────────────────────────────────────────

def _rotate(image: Image.Image, angle: float = 15.0) -> Image.Image:
    """
    Rotate the image by *angle* degrees clockwise.

    Uses bilinear resampling; canvas size is preserved (no expand),
    so corner pixels are filled with black.

    Args:
        image: Input PIL Image.
        angle: Rotation angle in degrees (positive = clockwise).
    """
    return image.rotate(-angle, resample=Image.BILINEAR, expand=False)


def _horizontal_flip(image: Image.Image) -> Image.Image:
    """
    Mirror the image left-to-right (horizontal reflection).

    This is a lossless, information-preserving transform and is the
    canonical rotation-invariance test for symmetric lesions.
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def _zoom(image: Image.Image, factor: float = 0.8) -> Image.Image:
    """
    Centre-crop to *factor* of the original size, then resize back.

    A factor of 0.8 crops to 80 % of the original width/height (zoom-in
    effect) and resamples back to the original resolution.

    Args:
        image:  Input PIL Image.
        factor: Crop fraction in (0, 1]. Values closer to 1 are subtler.
    """
    w, h   = image.size
    new_w  = int(w * factor)
    new_h  = int(h * factor)
    left   = (w - new_w) // 2
    top    = (h - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)


def _brightness(
    image: Image.Image,
    brightness: float = 1.5,
    contrast: float = 1.2,
) -> Image.Image:
    """
    Multiply brightness by *brightness* and contrast by *contrast*.

    Apply a Gaussian low-pass filter.
    
    This transformation evaluates robustness to image quality
    degradation and loss of high-frequency information.

    Args:
        image:      Input PIL Image.
        brightness: Multiplicative brightness factor (1.0 = original).
        contrast:   Multiplicative contrast factor (1.0 = original).
    """
    img = ImageEnhance.Brightness(image).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def _gaussian_noise(image: Image.Image, sigma: float = 0.05) -> Image.Image:
    """
    Add zero-mean Gaussian noise with standard deviation *sigma*.

    *sigma* is expressed in the normalised [0, 1] pixel range, so
    ``sigma=0.05`` corresponds to noise with std = 12.75 on [0, 255].

    Args:
        image: Input PIL Image.
        sigma: Noise standard deviation in [0, 1].
    """
    arr   = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    arr   = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _gaussian_blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
    """
    Apply a Gaussian low-pass filter with the given *radius*.

    Simulates defocus blur or motion blur that can appear in handheld
    dermatoscope captures.

    Args:
        image:  Input PIL Image.
        radius: Gaussian blur radius (higher = blurrier).
    """
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _sharpening(image: Image.Image, factor: float = 2.0) -> Image.Image:
    """
    Sharpen the image using PIL's unsharp masking.

    A factor of 1.0 returns the original; values > 1 increase sharpness.

    Args:
        image:  Input PIL Image.
        factor: Sharpness enhancement factor (1.0 = original).
    """
    return ImageEnhance.Sharpness(image).enhance(factor)


def _saturation(image: Image.Image, factor: float = 0.5) -> Image.Image:
    """
    Adjust colour saturation by *factor*.

    A factor of 0 produces a grayscale image; 1.0 leaves colours unchanged;
    values > 1 increase saturation. 0.5 produces a desaturated (faded) image,
    simulating ageing or different imaging conditions.

    Args:
        image:  Input PIL Image.
        factor: Colour saturation multiplier.
    """
    return ImageEnhance.Color(image).enhance(factor)


# ─── Registry ─────────────────────────────────────────────────────────────────

MR_REGISTRY: List[MetamorphicRelation] = [
    MetamorphicRelation(
        id="MR1",
        name="Rotation",
        description="Rotate the image 15° clockwise. Preserves lesion semantics.",
        parameters={"angle_deg": 15},
        _fn=lambda img: _rotate(img, angle=15.0),
    ),
    MetamorphicRelation(
        id="MR2",
        name="HorizontalFlip",
        description="Mirror the image left-to-right (horizontal reflection).",
        parameters={},
        _fn=_horizontal_flip,
    ),
    MetamorphicRelation(
        id="MR3",
        name="Zoom",
        description="Centre-crop to 80 % then resize back (zoom-in by 1.25×).",
        parameters={"factor": 0.8},
        _fn=lambda img: _zoom(img, factor=0.8),
    ),
    MetamorphicRelation(
        id="MR4",
        name="Brightness",
        description="Increase brightness by 1.5× and contrast by 1.2×.",
        parameters={"brightness": 1.5, "contrast": 1.2},
        _fn=lambda img: _brightness(img, brightness=1.5, contrast=1.2),
    ),
    MetamorphicRelation(
        id="MR5",
        name="GaussianNoise",
        description="Add zero-mean Gaussian noise with σ=0.05 (pixel-level).",
        parameters={"sigma": 0.05},
        _fn=lambda img: _gaussian_noise(img, sigma=0.05),
    ),
    MetamorphicRelation(
        id="MR6",
        name="GaussianBlur",
        description="Apply a Gaussian low-pass filter with radius=2.0.",
        parameters={"radius": 2.0},
        _fn=lambda img: _gaussian_blur(img, radius=2.0),
    ),
    MetamorphicRelation(
        id="MR7",
        name="Sharpening",
        description="Apply unsharp masking with sharpness factor 2.0×.",
        parameters={"factor": 2.0},
        _fn=lambda img: _sharpening(img, factor=2.0),
    ),
    MetamorphicRelation(
        id="MR8",
        name="Saturation",
        description="Reduce colour saturation to 0.5× (semi-desaturated).",
        parameters={"factor": 0.5},
        _fn=lambda img: _saturation(img, factor=0.5),
    ),
]

# Convenience lists
MR_IDS   = [mr.id   for mr in MR_REGISTRY]
MR_NAMES = [mr.name for mr in MR_REGISTRY]


# ─── Public API ───────────────────────────────────────────────────────────────

def apply_all_transforms(image: Image.Image) -> Dict[str, Image.Image]:
    """
    Apply all 8 MRs to a PIL image.

    Args:
        image: Source PIL Image.

    Returns:
        Ordered dict mapping ``"Original"`` and ``"MR1"`` … ``"MR8"``
        to their respective PIL Images.
    """
    result: Dict[str, Image.Image] = {"Original": image}
    for mr in MR_REGISTRY:
        result[mr.id] = mr(image)
    return result


def get_mr_by_id(mr_id: str) -> MetamorphicRelation:
    """
    Look up a :class:`MetamorphicRelation` by its ``id`` string.

    Args:
        mr_id: e.g. ``"MR1"``

    Raises:
        KeyError: If *mr_id* is not found in the registry.
    """
    for mr in MR_REGISTRY:
        if mr.id == mr_id:
            return mr
    raise KeyError(f"Unknown MR id: {mr_id!r}. Valid ids: {MR_IDS}")
