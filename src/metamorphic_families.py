"""
src/metamorphic_families.py — Hierarchical Metamorphic Transformation Families with Severity Ladders.

Organizes metamorphic relations into 3 semantic families:
- Geometric: Rotation (mild/mod/sev), Scale (mild/mod/sev), Horizontal Flip (mod)
- Photometric: Brightness (mild/mod), Contrast (mild/mod), Saturation (mild/mod)
- Sensor / Noise: Gaussian Blur (mild/mod/sev), Gaussian Noise (mild/mod/sev)

Each relation specifies:
- id: e.g. "GEOM_ROT_MILD"
- family: "geometric" | "photometric" | "sensor_noise"
- transform_type: "rotation" | "scale" | "flip" | "brightness" | "contrast" | "saturation" | "blur" | "noise"
- severity: "mild" | "moderate" | "severe"
- weight: 1.0 (mild), 0.6 (moderate), 0.3 (severe)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ─── Severity Weights ──────────────────────────────────────────────────────────
SEVERITY_WEIGHTS: Dict[str, float] = {
    "mild": 1.0,
    "moderate": 0.6,
    "severe": 0.3,
}


@dataclass(frozen=True)
class MetamorphicTest:
    """Represents an individual metamorphic test instance."""
    id: str
    name: str
    family: str
    transform_type: str
    severity: str
    weight: float
    description: str
    apply_fn: Callable[[Image.Image], Image.Image]

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.apply_fn(img)


# ─── Transform Functions ──────────────────────────────────────────────────────

def _rotate(img: Image.Image, angle: float) -> Image.Image:
    """Rotate image clockwise by angle degrees with bilinear interpolation."""
    return img.rotate(-angle, resample=Image.Resampling.BILINEAR, expand=False)


def _hflip(img: Image.Image) -> Image.Image:
    """Mirror image horizontally."""
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def _zoom_crop(img: Image.Image, factor: float) -> Image.Image:
    """Center-crop factor of original size and resize back to original dimensions."""
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.Resampling.BILINEAR)


def _brightness(img: Image.Image, factor: float) -> Image.Image:
    """Scale brightness."""
    return ImageEnhance.Brightness(img).enhance(factor)


def _contrast(img: Image.Image, factor: float) -> Image.Image:
    """Scale contrast."""
    return ImageEnhance.Contrast(img).enhance(factor)


def _saturation(img: Image.Image, factor: float) -> Image.Image:
    """Scale color saturation."""
    return ImageEnhance.Color(img).enhance(factor)


def _blur(img: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur with given pixel radius."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Add zero-mean Gaussian noise scaled by sigma in [0, 1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    rng = np.random.RandomState(42)
    noise = rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).astype(np.uint8))


# ─── Full Test Matrix Registry ────────────────────────────────────────────────

HIERARCHICAL_TEST_MATRIX: List[MetamorphicTest] = [
    # ── Geometric Family ──────────────────────────────────────────────────────
    MetamorphicTest(
        id="GEOM_ROT_MILD",
        name="Rotation (5 deg)",
        family="geometric",
        transform_type="rotation",
        severity="mild",
        weight=SEVERITY_WEIGHTS["mild"],
        description="Rotate image 5 degrees clockwise",
        apply_fn=lambda img: _rotate(img, 5.0),
    ),
    MetamorphicTest(
        id="GEOM_ROT_MOD",
        name="Rotation (10 deg)",
        family="geometric",
        transform_type="rotation",
        severity="moderate",
        weight=SEVERITY_WEIGHTS["moderate"],
        description="Rotate image 10 degrees clockwise",
        apply_fn=lambda img: _rotate(img, 10.0),
    ),
    MetamorphicTest(
        id="GEOM_ROT_SEV",
        name="Rotation (15 deg)",
        family="geometric",
        transform_type="rotation",
        severity="severe",
        weight=SEVERITY_WEIGHTS["severe"],
        description="Rotate image 15 degrees clockwise",
        apply_fn=lambda img: _rotate(img, 15.0),
    ),
    MetamorphicTest(
        id="GEOM_SCALE_MILD",
        name="Zoom (0.95x)",
        family="geometric",
        transform_type="scale",
        severity="mild",
        weight=SEVERITY_WEIGHTS["mild"],
        description="Center-crop 95% and resize back",
        apply_fn=lambda img: _zoom_crop(img, 0.95),
    ),
    MetamorphicTest(
        id="GEOM_SCALE_MOD",
        name="Zoom (0.90x)",
        family="geometric",
        transform_type="scale",
        severity="moderate",
        weight=SEVERITY_WEIGHTS["moderate"],
        description="Center-crop 90% and resize back",
        apply_fn=lambda img: _zoom_crop(img, 0.90),
    ),
    MetamorphicTest(
        id="GEOM_SCALE_SEV",
        name="Zoom (0.80x)",
        family="geometric",
        transform_type="scale",
        severity="severe",
        weight=SEVERITY_WEIGHTS["severe"],
        description="Center-crop 80% and resize back",
        apply_fn=lambda img: _zoom_crop(img, 0.80),
    ),
    MetamorphicTest(
        id="GEOM_FLIP_MOD",
        name="Horizontal Flip",
        family="geometric",
        transform_type="flip",
        severity="moderate",
        weight=SEVERITY_WEIGHTS["moderate"],
        description="Mirror image horizontally",
        apply_fn=_hflip,
    ),

    # ── Photometric Family ────────────────────────────────────────────────────
    MetamorphicTest(
        id="PHOTO_BRIGHT_MILD",
        name="Brightness (1.15x)",
        family="photometric",
        transform_type="brightness",
        severity="mild",
        weight=SEVERITY_WEIGHTS["mild"],
        description="Increase brightness by 15%",
        apply_fn=lambda img: _brightness(img, 1.15),
    ),
    MetamorphicTest(
        id="PHOTO_BRIGHT_MOD",
        name="Brightness (1.35x)",
        family="photometric",
        transform_type="brightness",
        severity="moderate",
        weight=SEVERITY_WEIGHTS["moderate"],
        description="Increase brightness by 35%",
        apply_fn=lambda img: _brightness(img, 1.35),
    ),
    MetamorphicTest(
        id="PHOTO_CONTRAST_MILD",
        name="Contrast (1.15x)",
        family="photometric",
        transform_type="contrast",
        severity="mild",
        weight=SEVERITY_WEIGHTS["mild"],
        description="Increase contrast by 15%",
        apply_fn=lambda img: _contrast(img, 1.15),
    ),
    MetamorphicTest(
        id="PHOTO_CONTRAST_MOD",
        name="Contrast (1.35x)",
        family="photometric",
        transform_type="contrast",
        severity="moderate",
        weight=SEVERITY_WEIGHTS["moderate"],
        description="Increase contrast by 35%",
        apply_fn=lambda img: _contrast(img, 1.35),
    ),
    MetamorphicTest(
        id="PHOTO_SAT_MILD",
        name="Saturation (0.80x)",
        family="photometric",
        transform_type="saturation",
        severity="mild",
        weight=SEVERITY_WEIGHTS["mild"],
        description="Reduce color saturation to 80%",
        apply_fn=lambda img: _saturation(img, 0.80),
    ),
    MetamorphicTest(
        id="PHOTO_SAT_MOD",
        name="Saturation (0.55x)",
        family="photometric",
        transform_type="saturation",
        severity="moderate",
        weight=SEVERITY_WEIGHTS["moderate"],
        description="Reduce color saturation to 55%",
        apply_fn=lambda img: _saturation(img, 0.55),
    ),
    MetamorphicTest(
        id="PHOTO_BRIGHT_SEV",
        name="Brightness (1.55x)",
        family="photometric",
        transform_type="brightness",
        severity="severe",
        weight=SEVERITY_WEIGHTS["severe"],
        description="Increase brightness by 55% (flash overexposure)",
        apply_fn=lambda img: _brightness(img, 1.55),
    ),
    MetamorphicTest(
        id="PHOTO_CONTRAST_SEV",
        name="Contrast (1.45x)",
        family="photometric",
        transform_type="contrast",
        severity="severe",
        weight=SEVERITY_WEIGHTS["severe"],
        description="Increase contrast by 45% (high-skin-pigmentation boundary)",
        apply_fn=lambda img: _contrast(img, 1.45),
    ),
    MetamorphicTest(
        id="PHOTO_SAT_SEV",
        name="Saturation (0.25x)",
        family="photometric",
        transform_type="saturation",
        severity="severe",
        weight=SEVERITY_WEIGHTS["severe"],
        description="Reduce color saturation to 25% (near-greyscale, white balance fault)",
        apply_fn=lambda img: _saturation(img, 0.25),
    ),

    # ── Sensor / Noise Family ─────────────────────────────────────────────────
    MetamorphicTest(
        id="NOISE_BLUR_MILD",
        name="Gaussian Blur (r=1.0)",
        family="sensor_noise",
        transform_type="blur",
        severity="mild",
        weight=SEVERITY_WEIGHTS["mild"],
        description="Gaussian filter with radius 1.0",
        apply_fn=lambda img: _blur(img, 1.0),
    ),
    MetamorphicTest(
        id="NOISE_BLUR_MOD",
        name="Gaussian Blur (r=2.0)",
        family="sensor_noise",
        transform_type="blur",
        severity="moderate",
        weight=SEVERITY_WEIGHTS["moderate"],
        description="Gaussian filter with radius 2.0",
        apply_fn=lambda img: _blur(img, 2.0),
    ),
    MetamorphicTest(
        id="NOISE_BLUR_SEV",
        name="Gaussian Blur (r=3.5)",
        family="sensor_noise",
        transform_type="blur",
        severity="severe",
        weight=SEVERITY_WEIGHTS["severe"],
        description="Gaussian filter with radius 3.5 (significant optical defocus)",
        apply_fn=lambda img: _blur(img, 3.5),
    ),
    MetamorphicTest(
        id="NOISE_GAUSS_MILD",
        name="Gaussian Noise (sigma=0.02)",
        family="sensor_noise",
        transform_type="noise",
        severity="mild",
        weight=SEVERITY_WEIGHTS["mild"],
        description="Additive Gaussian noise sigma=0.02",
        apply_fn=lambda img: _gaussian_noise(img, 0.02),
    ),
    MetamorphicTest(
        id="NOISE_GAUSS_MOD",
        name="Gaussian Noise (sigma=0.05)",
        family="sensor_noise",
        transform_type="noise",
        severity="moderate",
        weight=SEVERITY_WEIGHTS["moderate"],
        description="Additive Gaussian noise sigma=0.05",
        apply_fn=lambda img: _gaussian_noise(img, 0.05),
    ),
    MetamorphicTest(
        id="NOISE_GAUSS_SEV",
        name="Gaussian Noise (sigma=0.09)",
        family="sensor_noise",
        transform_type="noise",
        severity="severe",
        weight=SEVERITY_WEIGHTS["severe"],
        description="Additive Gaussian noise sigma=0.09 (high ISO sensor noise floor)",
        apply_fn=lambda img: _gaussian_noise(img, 0.09),
    ),
]

FAMILIES: List[str] = ["geometric", "photometric", "sensor_noise"]
