"""
src/metamorphic_families.py â€” Hierarchical Metamorphic Transformation Families with Severity Ladders.

Organizes metamorphic relations into 3 semantic families dynamically loaded from transform_bank.yaml:
- Geometric: Rotation (mild/mod/sev), Scale (mild/mod/sev), Shear (single)
- Photometric: Brightness (mild/mod/sev), Contrast (mild/mod/sev), Saturation (mild/mod/sev)
- Sensor / Noise: Gaussian Blur (mild/mod/sev), Gaussian Noise (mild/mod/sev), JPEG Compression (mild/mod/sev)
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import yaml

# â”€â”€â”€ Severity Weights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEVERITY_WEIGHTS: Dict[str, float] = {
    "mild": 1.0,
    "moderate": 0.6,
    "severe": 0.3,
    "single": 1.0,
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
    apply_fn: Callable[..., Image.Image]

    def __call__(self, img: Image.Image, seed: Optional[int] = None) -> Image.Image:
        try:
            return self.apply_fn(img, seed=seed)
        except TypeError:
            return self.apply_fn(img)


# â”€â”€â”€ Transform Implementation Functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _rotate(img: Image.Image, angle: float) -> Image.Image:
    """Rotate image clockwise by angle degrees with bilinear interpolation."""
    return img.rotate(-angle, resample=Image.Resampling.BILINEAR, expand=False)


def _zoom_crop(img: Image.Image, factor: float) -> Image.Image:
    """Center-crop factor of original size and resize back to original dimensions."""
    w, h = img.size
    if factor <= 1.0:
        new_w, new_h = int(w * factor), int(h * factor)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.Resampling.BILINEAR)
    else:
        # Zoom out / expand with pad
        scaled_w, scaled_h = int(w / factor), int(h / factor)
        resized = img.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
        padded = Image.new(img.mode, (w, h), (0, 0, 0))
        left = (w - scaled_w) // 2
        top = (h - scaled_h) // 2
        padded.paste(resized, (left, top))
        return padded


def _shear_x(img: Image.Image, factor: float = 0.05) -> Image.Image:
    """Apply horizontal affine shear simulating perspective distortion."""
    w, h = img.size
    return img.transform(
        (w, h),
        Image.Transform.AFFINE,
        (1.0, factor, -factor * h / 2.0, 0.0, 1.0, 0.0),
        resample=Image.Resampling.BILINEAR,
    )


def _brightness(img: Image.Image, value: float) -> Image.Image:
    """Scale or shift brightness."""
    # If delta given as percentage offset e.g. 10 -> factor 1.10
    factor = 1.0 + value / 100.0 if abs(value) > 2 else value
    return ImageEnhance.Brightness(img).enhance(factor)


def _contrast(img: Image.Image, factor: float) -> Image.Image:
    """Scale contrast."""
    return ImageEnhance.Contrast(img).enhance(factor)


def _saturation(img: Image.Image, factor: float) -> Image.Image:
    """Scale color saturation."""
    return ImageEnhance.Color(img).enhance(factor)


def _blur(img: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur with given radius/sigma."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _gaussian_noise(img: Image.Image, sigma: float, seed: Optional[int] = None) -> Image.Image:
    """Add zero-mean Gaussian noise scaled by sigma in [0, 1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    rng = np.random.RandomState(seed if seed is not None else 42)
    noise = rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).astype(np.uint8))


def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Simulate lossy JPEG transmission artifacts at given quality factor."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# â”€â”€â”€ YAML Loader to build HIERARCHICAL_TEST_MATRIX â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_test_matrix_from_yaml() -> List[MetamorphicTest]:
    """Load transform_bank.yaml and instantiate MetamorphicTest registry."""
    candidates = [
        Path(__file__).resolve().parent.parent / "transform_bank.yaml",
        Path(__file__).resolve().parent / "transform_bank.yaml",
        Path("transform_bank.yaml"),
    ]
    yaml_path = None
    for p in candidates:
        if p.exists():
            yaml_path = p
            break

    if yaml_path is None:
        raise FileNotFoundError("Could not locate transform_bank.yaml")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tests: List[MetamorphicTest] = []
    for fam_name, fam_data in cfg.get("families", {}).items():
        for t_type, t_data in fam_data.get("transforms", {}).items():
            for sev, sev_data in t_data.get("tiers", {}).items():
                weight = SEVERITY_WEIGHTS[sev]
                vals = sev_data.get("values", [])
                for idx, val in enumerate(vals):
                    suffix = "" if len(vals) == 1 else f"_{'POS' if val > 0 else 'NEG'}"
                    test_id = f"{fam_name[:4].upper()}_{t_type[:4].upper()}_{sev[:3].upper()}_{idx+1}{suffix}"

                    if t_type == "rotation":
                        fn = (lambda v: lambda img: _rotate(img, v))(val)
                    elif t_type == "scale":
                        fn = (lambda v: lambda img: _zoom_crop(img, v))(val)
                    elif t_type == "shear":
                        fn = (lambda v: lambda img: _shear_x(img, v))(val)
                    elif t_type == "brightness":
                        fn = (lambda v: lambda img: _brightness(img, v))(val)
                    elif t_type == "contrast":
                        fn = (lambda v: lambda img: _contrast(img, v))(val)
                    elif t_type == "saturation":
                        fn = (lambda v: lambda img: _saturation(img, v))(val)
                    elif t_type == "blur":
                        fn = (lambda v: lambda img: _blur(img, v))(val)
                    elif t_type == "noise":
                        fn = (lambda v: lambda img, seed=None: _gaussian_noise(img, v, seed=seed))(val)
                    elif t_type == "compression":
                        fn = (lambda v: lambda img: _jpeg_compress(img, int(v)))(val)
                    else:
                        continue

                    tests.append(
                        MetamorphicTest(
                            id=test_id,
                            name=f"{t_type.capitalize()} ({sev} {val})",
                            family=fam_name,
                            transform_type=t_type,
                            severity=sev,
                            weight=weight,
                            description=f"{t_type} perturbation at {sev} tier ({val})",
                            apply_fn=fn,
                        )
                    )
    return tests


HIERARCHICAL_TEST_MATRIX: List[MetamorphicTest] = _build_test_matrix_from_yaml()
FAMILIES: List[str] = ["geometric", "photometric", "sensor_noise"]

