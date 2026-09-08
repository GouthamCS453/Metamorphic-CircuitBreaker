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
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18


# ─── Constants ────────────────────────────────────────────────────────────────

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

NUM_CLASSES = 10

# CIFAR-10 images are 32 × 32 pixels
IMAGE_SIZE = 32

# Standard CIFAR-10 normalization statistics
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


# ─── Root Paths ───────────────────────────────────────────────────────────────

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Model checkpoints
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# Experiment outputs
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Best trained model
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# Training history
TRAINING_LOG_PATH = CHECKPOINT_DIR / "training_log.json"


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Return the best available compute device.

    Priority:
        1. CUDA GPU
        2. Apple MPS
        3. CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
) -> torch.nn.Module:
    """
    Build a CIFAR-10 adapted ResNet-18 model.

    The model is trained from scratch by default.

    Standard ImageNet ResNet-18 uses:
        - 7 × 7 first convolution
        - stride 2
        - max pooling

    These operations are too aggressive for CIFAR-10's
    32 × 32 images.

    Therefore, this implementation uses:
        - 3 × 3 first convolution
        - stride 1
        - no max pooling
        - 10-class output layer

    Args:
        num_classes:
            Number of output classes.

        pretrained:
            Kept as an argument for compatibility with the rest
            of the project. Defaults to False because this project
            trains ResNet-18 from scratch.

    Returns:
        CIFAR-10 adapted ResNet-18 model.
    """

    # Train completely from scratch.
    # weights=None means no ImageNet pretrained weights.
    model = resnet18(weights=None)

    # ── CIFAR-10 input stem ───────────────────────────────────────────────

    # Original ImageNet ResNet:
    #   7 × 7 convolution, stride 2
    #
    # CIFAR-10:
    #   32 × 32 images
    #
    # Therefore use a smaller 3 × 3 convolution with stride 1.
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # Remove ImageNet-style max pooling.
    model.maxpool = nn.Identity()

    # ── CIFAR-10 classification head ─────────────────────────────────────

    # ResNet-18 normally outputs 1000 ImageNet classes.
    # Replace it with 10 CIFAR-10 classes.
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Load a trained CIFAR-10 ResNet-18 model from a checkpoint.

    Args:
        checkpoint_path:
            Path to the .pth checkpoint.
            Defaults to checkpoints/best_model.pth.

        device:
            Target device.
            Automatically detected if None.

    Returns:
        Trained ResNet-18 model in evaluation mode.

    Raises:
        FileNotFoundError:
            If the checkpoint does not exist.
    """

    if checkpoint_path is None:
        checkpoint_path = BEST_MODEL_PATH

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"\n  Checkpoint not found: {checkpoint_path}\n"
            "  Please train the model first:\n"
            "    python src/train.py --data_dir data/cifar10\n"
        )

    if device is None:
        device = get_device()

    # Create the same architecture used during training.
    # No pretrained weights are loaded.
    model = build_model(
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    # Load trained weights.
    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # Support wrapped checkpoint dictionaries:
    #
    # {
    #     "model_state_dict": ...,
    #     ...
    # }
    #
    # as well as raw state_dict files.
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
    Return the CIFAR-10 training augmentation pipeline.

    Augmentations:
        - Random crop with padding
        - Random horizontal flip
        - Tensor conversion
        - CIFAR-10 normalization
    """

    return transforms.Compose([
        transforms.RandomCrop(
            IMAGE_SIZE,
            padding=4,
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])


def get_val_transform() -> transforms.Compose:
    """
    Return the validation/test preprocessing pipeline.

    No random augmentation is applied.
    """

    return transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])


# ─── PIL Image Conversion ────────────────────────────────────────────────────

def pil_to_tensor(pil_image) -> torch.Tensor:
    """
    Convert a PIL image into a normalized 4-D tensor
    ready for ResNet-18 inference.

    Returns:
        Tensor of shape (1, 3, 32, 32).
    """

    return get_val_transform()(pil_image).unsqueeze(0)


# ─── Output Directory Management ─────────────────────────────────────────────

def setup_output_dir() -> Path:
    """
    Create a timestamped output directory for one pipeline run.

    Directory structure:

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

    run_dir = OUTPUTS_DIR / f"run_{timestamp}"

    for sub in [
        "sample_images",
        "transformed",
        "gradcam/wrong_predictions",
        "gradcam/random_showcase",
        "plots",
    ]:
        (run_dir / sub).mkdir(
            parents=True,
            exist_ok=True,
        )

    return run_dir


def get_latest_run_dir() -> Optional[Path]:
    """
    Return the most recently created run directory.

    Returns:
        Path to latest run directory, or None if no runs exist.
    """

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = sorted(
        OUTPUTS_DIR.glob("run_*"),
        reverse=True,
    )

    return runs[0] if runs else None


def list_run_dirs() -> list:
    """
    Return all run directories sorted newest-first.
    """

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sorted(
        OUTPUTS_DIR.glob("run_*"),
        reverse=True,
    )


# ─── JSON I/O ─────────────────────────────────────────────────────────────────

def save_json(
    data,
    path: Path,
) -> None:
    """
    Serialize data to a JSON file.

    Parent directories are created automatically.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w") as fh:
        json.dump(
            data,
            fh,
            indent=2,
        )


def load_json(
    path: Union[str, Path],
):
    """
    Load and return the contents of a JSON file.
    """

    with open(path) as fh:
        return json.load(fh)
