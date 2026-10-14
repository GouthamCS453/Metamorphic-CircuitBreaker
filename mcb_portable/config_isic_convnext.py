"""
config_isic_convnext.py - member config for THIS project: ConvNeXt-Base trained on ISIC 2019.

Uses the project's own loader/preprocessing (src/utils.py), the held-out val split
(checkpoints/val_split.csv, 7600 images, never trained on) and the 40 external images in
mcb_experiments/inputs/web_images. Nothing is copied; paths point at the main project.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent                      # main_implementation/
sys.path.insert(0, str(PROJECT))

MODEL_NAME = "ConvNeXt-Base"
DATASET_NAME = "ISIC2019"
DATASET_VERSION = "valsplit"
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]   # = src.utils.CLASS_NAMES
CRITICAL_CLASSES = ["MEL", "BCC", "SCC"]
SEED = 42
N_SAMPLES = 1500                       # stratified subset of the 7600 val images (0 = all)
EXTERNAL_DIR = PROJECT / "mcb_experiments" / "inputs" / "web_images"   # labels.csv + jpgs (flat)


def load_model():
    from src.utils import load_model as _load, get_val_transform, get_device
    from mcb.adapters import TorchClassifier
    device = get_device()
    model = _load(PROJECT / "checkpoints" / "best_model.pth", device)
    model.eval()
    return TorchClassifier(model, get_val_transform(), batch_size=32, amp=True)


def get_dataset() -> pd.DataFrame:
    df = pd.read_csv(PROJECT / "checkpoints" / "val_split.csv")
    paths = [str(PROJECT / str(p).replace("\\", "/")) for p in df["image_path"]]
    return pd.DataFrame({"image_path": paths, "label": df["label"].astype(int)})
