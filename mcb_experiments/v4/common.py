"""
test_suite/common.py — Shared Utilities, Configuration, and GTSRB Helpers
for the Metamorphic Circuit Breaker Testing Suite.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import BEST_MODEL_PATH, get_device
from src.models.mobilenet_adapter import MobileNetAdapter
from src.circuit_breaker import (
    MetamorphicCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    DiagnosticReport,
)
from src.metamorphic_families import HIERARCHICAL_TEST_MATRIX

TEST_SUITE_DIR = PROJECT_ROOT / "test_suite"
RESULTS_DIR = TEST_SUITE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Standard GTSRB 43 Traffic Sign Class Names ──────────────────────────────
GTSRB_CLASSES: Dict[int, str] = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing restrictions",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}


def get_class_name(class_id: int | str) -> str:
    """Return friendly sign name for GTSRB class id."""
    try:
        cid = int(class_id)
        return GTSRB_CLASSES.get(cid, f"Sign Class {cid}")
    except (ValueError, TypeError):
        return str(class_id)


def load_gtsrb_val_dataset(limit: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """
    Load held-out validation samples from checkpoints/val_split.csv or data/gtsrb/Test.csv.
    Ensures absolute or valid relative paths are resolved.
    """
    val_csv = PROJECT_ROOT / "checkpoints" / "val_split.csv"
    if val_csv.exists():
        df = pd.read_csv(val_csv)
    else:
        test_csv = PROJECT_ROOT / "data" / "gtsrb" / "Test.csv"
        if test_csv.exists():
            df = pd.read_csv(test_csv)
            if "Path" in df.columns:
                df["image_path"] = df["Path"].apply(lambda p: str(PROJECT_ROOT / "data" / "gtsrb" / p))
            if "ClassId" in df.columns:
                df["label"] = df["ClassId"]
        else:
            raise FileNotFoundError("Could not find validation or test dataset CSV.")

    # Fix relative image paths if necessary
    def _resolve_img_path(p: str) -> str:
        pth = Path(p)
        if pth.is_absolute() and pth.exists():
            return str(pth)
        cand = PROJECT_ROOT / p
        if cand.exists():
            return str(cand)
        cand2 = PROJECT_ROOT / "data" / "gtsrb" / p
        if cand2.exists():
            return str(cand2)
        return str(pth)

    df["resolved_path"] = df["image_path"].apply(_resolve_img_path)
    df = df[df["resolved_path"].apply(lambda p: Path(p).exists())].copy()

    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
    return df


def get_model_and_cb(
    config: Optional[CircuitBreakerConfig] = None,
) -> Tuple[MobileNetAdapter, MetamorphicCircuitBreaker]:
    """Initialize MobileNetAdapter and MetamorphicCircuitBreaker."""
    model = MobileNetAdapter(checkpoint_path=BEST_MODEL_PATH)
    cb = MetamorphicCircuitBreaker(model=model, config=config)
    return model, cb


def save_test_output(test_key: str, data: Dict[str, Any], markdown_content: str) -> Tuple[Path, Path]:
    """
    Save test output both as a JSON file and a Markdown report in test_suite/results/.
    """
    json_path = RESULTS_DIR / f"{test_key}.json"
    md_path = RESULTS_DIR / f"{test_key}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"\n[Artifact Saved] JSON: {json_path}")
    print(f"[Artifact Saved] Markdown: {md_path}")
    return json_path, md_path
