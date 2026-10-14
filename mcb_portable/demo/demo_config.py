"""
demo/demo_config.py - Config for the synthetic smoke test (shows exactly what a member config contains).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_demo import COLORS, make_image  # noqa: E402

HERE = Path(__file__).resolve().parent

MODEL_NAME = "DemoCentroid"
DATASET_NAME = "SyntheticBlobs"
DATASET_VERSION = "v1"
CLASS_NAMES = ["red", "green", "blue", "yellow", "cyan"]
CRITICAL_CLASSES = ["red"]
SEED = 42
N_SAMPLES = 0
EXTERNAL_DIR = HERE / "data" / "external"
RUN_ID = "demo__synthetic"


def _feat(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB").resize((12, 12), Image.Resampling.BILINEAR), dtype=np.float32).ravel() / 255.0


class _Model:
    device_name = "cpu"

    def __init__(self, temp=0.9):
        rng = np.random.default_rng(123)
        self.cent = np.stack([np.mean([_feat(make_image(rng, c, 0.0)) for _ in range(15)], axis=0) for c in range(5)])
        self.temp = temp

    def sync(self):
        pass

    def predict_proba(self, images):
        f = np.stack([_feat(i) for i in images])
        d = ((f[:, None, :] - self.cent[None]) ** 2).sum(-1)
        z = -d / self.temp
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


def load_model():
    return _Model()


def get_dataset():
    p = HERE / "data" / "dataset.csv"
    if not p.exists():
        raise SystemExit("Run `python demo/make_demo.py` first.")
    return pd.read_csv(p)
