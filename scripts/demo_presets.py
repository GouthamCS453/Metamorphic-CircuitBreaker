"""
scripts/demo_presets.py -- Curate demo preset images from data/isic2019.

Selects up to 3 representative images from the ISIC 2019 dataset that
exhibit diverse circuit breaker behaviors and copies them to data/presets/.

Usage:
    python scripts/demo_presets.py

Each preset gets a metadata JSON file alongside it describing its expected
circuit breaker behavior (CLOSED / HALF_OPEN / OPEN) for UI display.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "isic2019"
PRESET_DIR = ROOT / "data" / "presets"

# Manually curated preset definitions.
# Each entry picks the first available image matching the class prefix.
PRESETS = [
    {
        "name": "Stable Nevus (Benign Mole)",
        "preset_id": "preset_stable",
        "class_hint": "NV",      # Melanocytic Nevus -- usually very stable
        "expected_state": "CLOSED",
        "description": (
            "A clearly defined benign mole. The model prediction is expected to remain "
            "consistent across all metamorphic perturbations -- demonstrating CLOSED "
            "(healthy) circuit breaker state."
        ),
    },
    {
        "name": "Ambiguous Lesion (Borderline)",
        "preset_id": "preset_borderline",
        "class_hint": "BKL",     # Benign Keratosis-like -- often borderline
        "expected_state": "HALF_OPEN",
        "description": (
            "A lesion with ambiguous morphology. The model may show mild sensitivity "
            "to geometric or photometric perturbations, potentially triggering HALF_OPEN "
            "(monitor/warning) state."
        ),
    },
    {
        "name": "High-Risk Melanoma Candidate",
        "preset_id": "preset_brittle",
        "class_hint": "MEL",     # Melanoma -- often the most borderline cases
        "expected_state": "OPEN",
        "description": (
            "A potentially malignant melanoma candidate. Expect strong prediction "
            "inconsistencies across rotation, lighting, and noise perturbations, "
            "likely tripping the circuit breaker to OPEN (safety intercept) state."
        ),
    },
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_images_for_class(data_dir: Path, class_hint: str, n: int = 5) -> list[Path]:
    """Find image files matching a class prefix hint from any subdirectory."""
    candidates = []
    for ext in IMAGE_EXTENSIONS:
        candidates.extend(data_dir.rglob(f"{class_hint}_*{ext}"))
        candidates.extend(data_dir.rglob(f"*_{class_hint}_*{ext}"))
        candidates.extend((data_dir / class_hint).glob(f"*{ext}") if (data_dir / class_hint).exists() else [])
    seen = set()
    unique = []
    for p in candidates:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)
    random.shuffle(unique)
    return unique[:n]


def build_presets():
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Presets] Scanning {DATA_DIR} ...")
    created = []
    for preset in PRESETS:
        pid = preset["preset_id"]
        dest_dir = PRESET_DIR / pid
        dest_dir.mkdir(parents=True, exist_ok=True)

        images = find_images_for_class(DATA_DIR, preset["class_hint"], n=3)
        if not images:
            # Fallback: pick any image from the dataset
            all_imgs = []
            for ext in IMAGE_EXTENSIONS:
                all_imgs.extend(DATA_DIR.rglob(f"*{ext}"))
            images = all_imgs[:1] if all_imgs else []

        if not images:
            print(f"  [WARN] No images found for preset '{pid}' -- skipping.")
            continue

        # Use the first resolved image
        src = images[0]
        dest = dest_dir / src.name
        shutil.copy2(src, dest)

        # Write metadata
        meta = {**preset, "image_filename": src.name}
        (dest_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"  [OK] Preset '{pid}': {src.name} -> {dest}")
        created.append(pid)

    print(f"[Presets] Done. Created {len(created)} presets in {PRESET_DIR}")
    return created


if __name__ == "__main__":
    build_presets()
