"""
src/utils.py — Shared constants, helpers, model loader, and transform pipelines
for the Metamorphic Circuit Breaker framework.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
import timm
from torchvision import transforms

# ─── Constants ────────────────────────────────────────────────────────────────

# GTSRB class names — class IDs 0 to 42
CLASS_NAMES = [str(i) for i in range(43)]

CLASS_FULL_NAMES = {
    str(i): f"Traffic Sign Class {i}"
    for i in range(43)
}

NUM_CLASSES = len(CLASS_NAMES)  # 43

# Root paths (resolved relative to this file's grandparent = project root)
ROOT_DIR       = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
OUTPUTS_DIR    = ROOT_DIR / "outputs"

BEST_MODEL_PATH   = CHECKPOINT_DIR / "best_model.pth"
TRAINING_LOG_PATH = CHECKPOINT_DIR / "training_log.json"

# ImageNet stats (ConvNeXt pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> torch.nn.Module:
    model = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def load_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Load a trained ConvNeXt model from a saved checkpoint.

    Args:
        checkpoint_path: Path to ``.pth`` file. Defaults to
                         ``checkpoints/best_model.pth``.
        device:          Target device. Auto-detected if None.

    Returns:
        Model in ``eval()`` mode on the target device.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if checkpoint_path is None:
        checkpoint_path = BEST_MODEL_PATH
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"\n  Checkpoint not found: {checkpoint_path}\n"
            "  Please train the model first:\n"
            "    python src/train.py --data_dir data/isic2019\n"
        )

    if device is None:
        device = get_device()

    model = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Support both raw state_dict and wrapped checkpoint dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# ─── Transform Pipelines ──────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """
    Augmented preprocessing pipeline for training.

    Applies random crops, flips, rotations and colour jitter followed
    by ImageNet normalisation.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    """
    Clean preprocessing pipeline for validation and inference.

    Resizes to the model's expected input size and normalises
    using ImageNet statistics.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def pil_to_tensor(pil_image) -> torch.Tensor:
    """
    Convert a PIL image to a normalised 4-D tensor ready for inference.

    Returns:
        Tensor of shape (1, 3, H, W).
    """
    return get_val_transform()(pil_image).unsqueeze(0)


# ─── Output Directory Management ──────────────────────────────────────────────

def setup_output_dir() -> Path:
    """
    Create a timestamped output directory for one pipeline run.

    Directory structure::

        outputs/run_YYYYMMDD_HHMMSS/
        ├── sample_images/
        ├── transformed/
        ├── gradcam/
        │   ├── wrong_predictions/
        │   └── random_showcase/
        └── plots/

    Returns:
        Path to the newly created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = OUTPUTS_DIR / f"run_{timestamp}"
    for sub in [
        "sample_images",
        "transformed",
        "gradcam/wrong_predictions",
        "gradcam/random_showcase",
        "plots",
    ]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def get_latest_run_dir() -> Optional[Path]:
    """Return the most recently created run directory, or None if none exist."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = sorted(OUTPUTS_DIR.glob("run_*"), reverse=True)
    return runs[0] if runs else None


def list_run_dirs() -> list:
    """Return all run directories sorted newest-first."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(OUTPUTS_DIR.glob("run_*"), reverse=True)


# ─── JSON I/O ─────────────────────────────────────────────────────────────────

def save_json(data, path: Path) -> None:
    """Serialise *data* to a JSON file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def load_json(path: Union[str, Path]):
    """Load and return the contents of a JSON file."""
    with open(path) as fh:
        return json.load(fh)
