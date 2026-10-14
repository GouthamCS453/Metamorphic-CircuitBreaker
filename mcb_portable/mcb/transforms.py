"""
mcb/transforms.py - The metamorphic transform bank (25 label-preserving relations).

Three families x several transform types x severity ladder (mild / moderate / severe).
`build_bank(scale)` multiplies every perturbation magnitude by `scale` so the
"range sensitivity" of the bank can be studied (scale=1.0 is the nominal bank
that every project member must use for results to be combinable).
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

FAMILIES = ["geometric", "photometric", "sensor_noise"]
SEVERITY_WEIGHTS = {"mild": 1.0, "moderate": 0.6, "severe": 0.3}

# (id, family, transform_type, severity, nominal magnitude)
BANK_SPEC = [
    ("GEOM_ROT_MILD", "geometric", "rotation", "mild", 5.0),
    ("GEOM_ROT_MOD", "geometric", "rotation", "moderate", 10.0),
    ("GEOM_ROT_SEV", "geometric", "rotation", "severe", 15.0),
    ("GEOM_SCALE_MILD", "geometric", "scale", "mild", 0.95),
    ("GEOM_SCALE_MOD", "geometric", "scale", "moderate", 0.90),
    ("GEOM_SCALE_SEV", "geometric", "scale", "severe", 0.80),
    ("GEOM_FLIP_MOD", "geometric", "flip", "moderate", 1.0),
    ("PHOTO_BRIGHT_MILD", "photometric", "brightness", "mild", 1.15),
    ("PHOTO_BRIGHT_MOD", "photometric", "brightness", "moderate", 1.35),
    ("PHOTO_BRIGHT_SEV", "photometric", "brightness", "severe", 1.55),
    ("PHOTO_CONTRAST_MILD", "photometric", "contrast", "mild", 1.15),
    ("PHOTO_CONTRAST_MOD", "photometric", "contrast", "moderate", 1.35),
    ("PHOTO_CONTRAST_SEV", "photometric", "contrast", "severe", 1.45),
    ("PHOTO_SAT_MILD", "photometric", "saturation", "mild", 0.80),
    ("PHOTO_SAT_MOD", "photometric", "saturation", "moderate", 0.55),
    ("PHOTO_SAT_SEV", "photometric", "saturation", "severe", 0.25),
    ("NOISE_BLUR_MILD", "sensor_noise", "blur", "mild", 1.0),
    ("NOISE_BLUR_MOD", "sensor_noise", "blur", "moderate", 2.0),
    ("NOISE_BLUR_SEV", "sensor_noise", "blur", "severe", 3.5),
    ("NOISE_GAUSS_MILD", "sensor_noise", "noise", "mild", 0.02),
    ("NOISE_GAUSS_MOD", "sensor_noise", "noise", "moderate", 0.05),
    ("NOISE_GAUSS_SEV", "sensor_noise", "noise", "severe", 0.09),
    ("NOISE_JPEG_MILD", "sensor_noise", "jpeg", "mild", 90),
    ("NOISE_JPEG_MOD", "sensor_noise", "jpeg", "moderate", 70),
    ("NOISE_JPEG_SEV", "sensor_noise", "jpeg", "severe", 50),
]


# --- primitive operations ----------------------------------------------------
def op_rotate(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(-angle, resample=Image.Resampling.BILINEAR, expand=False)


def op_zoom(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.Resampling.BILINEAR)


def op_hflip(img: Image.Image, _unused: float = 1.0) -> Image.Image:
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def op_brightness(img, f):
    return ImageEnhance.Brightness(img).enhance(f)


def op_contrast(img, f):
    return ImageEnhance.Contrast(img).enhance(f)


def op_saturation(img, f):
    return ImageEnhance.Color(img).enhance(f)


def op_blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def op_noise(img, sigma, seed: int = 42):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    rng = np.random.RandomState(seed)
    noisy = np.clip(arr + rng.normal(0.0, sigma, arr.shape).astype(np.float32), 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).astype(np.uint8))


def op_jpeg(img, quality):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.load()
    return out


_OPS: Dict[str, Callable] = {
    "rotation": op_rotate, "scale": op_zoom, "flip": op_hflip,
    "brightness": op_brightness, "contrast": op_contrast, "saturation": op_saturation,
    "blur": op_blur, "noise": op_noise, "jpeg": op_jpeg,
}


def _scaled(kind: str, p: float, s: float):
    """Scale a nominal magnitude by `s` (s=0 -> identity, s=1 -> nominal)."""
    if kind == "rotation":
        return p * s
    if kind == "scale":
        return max(0.3, 1.0 - (1.0 - p) * s)
    if kind in ("brightness", "contrast"):
        return max(0.0, 1.0 + (p - 1.0) * s)
    if kind == "saturation":
        return max(0.0, 1.0 - (1.0 - p) * s)
    if kind in ("blur", "noise"):
        return p * s
    if kind == "jpeg":
        return int(min(100, max(5, round(100 - (100 - p) * s))))
    return p


@dataclass(frozen=True)
class Test:
    id: str
    family: str
    type: str
    severity: str
    weight: float
    param: float
    fn: Callable[[Image.Image], Image.Image]


def build_bank(scale: float = 1.0) -> List[Test]:
    bank = []
    for tid, fam, kind, sev, nominal in BANK_SPEC:
        val = _scaled(kind, nominal, scale)
        bank.append(Test(tid, fam, kind, sev, SEVERITY_WEIGHTS[sev], val,
                         (lambda img, k=kind, v=val: _OPS[k](img, v))))
    return bank


def bank_meta(bank: List[Test]) -> List[dict]:
    return [dict(id=t.id, family=t.family, type=t.type, severity=t.severity,
                 weight=t.weight, param=t.param) for t in bank]


def bank_hash(scale: float = 1.0) -> str:
    """Fingerprint of the bank definition; combine_results warns if members differ."""
    payload = json.dumps({"spec": BANK_SPEC, "weights": SEVERITY_WEIGHTS, "scale": scale}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
