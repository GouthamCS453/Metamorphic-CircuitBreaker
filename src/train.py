"""
src/train.py — Train ResNet-18 from scratch on the CIFAR-10 dataset.

Steps
-----
1. Download/load the CIFAR-10 dataset.
2. Split the official 50,000 training images into:
       - 45,000 training images
       - 5,000 validation images
3. Train a CIFAR-10 adapted ResNet-18 from scratch.
4. Use SGD + momentum with CosineAnnealingLR.
5. Save the best model to ``checkpoints/best_model.pth``.
6. Write epoch-level metrics to ``checkpoints/training_log.json``.

Usage
-----
    python src/train.py --data_dir data/cifar10 --epochs 100 --batch_size 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    CLASS_NAMES,
    NUM_CLASSES,
    build_model,
    get_device,
    CHECKPOINT_DIR,
    TRAINING_LOG_PATH,
    get_train_transform,
    get_val_transform,
    save_json,
)


# ─── Training Helpers ─────────────────────────────────────────────────────────


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
):
    """
    Run one complete training epoch.

    Returns:
        Tuple of:
            average_loss,
            accuracy
    """

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        loader,
        desc="  Train",
        leave=False,
        unit="batch",
    )

    for images, labels in pbar:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # AMP is enabled only when using CUDA.
        with autocast(enabled=(device.type == "cuda" and scaler.is_enabled())):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item() * images.size(0)

        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total += images.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ─── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    """
    Evaluate the model on the validation set.

    Returns:
        Tuple of:
            average_loss,
            accuracy
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        loader,
        desc="  Val  ",
        leave=False,
        unit="batch",
    )

    for images, labels in pbar:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

        total_loss += loss.item() * images.size(0)

        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total += images.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ─── Test Evaluation ──────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_test(
    model,
    loader,
    criterion,
    device,
):
    """
    Evaluate the final/best model on the official CIFAR-10 test set.

    Returns:
        Tuple of:
            average_loss,
            accuracy
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        loader,
        desc="  Test ",
        leave=False,
        unit="batch",
    )

    for images, labels in pbar:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

        total_loss += loss.item() * images.size(0)

        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total += images.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train ResNet-18 from scratch on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/cifar10",
        help="CIFAR-10 dataset root directory",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Mini-batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Initial SGD learning rate",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="SGD weight decay",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is safest on Windows)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from an existing checkpoint",
    )

    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():

    args = parse_args()

    # ── Device ─────────────────────────────────────────────────────────────

    device = get_device()

    print(f"[INFO] Device       : {device}")
    print(f"[INFO] Epochs       : {args.epochs}")
    print(f"[INFO] Batch size   : {args.batch_size}")
    print(f"[INFO] Learning rate: {args.lr}")
    print(f"[INFO] Weight decay : {args.weight_decay}")
    print(f"[INFO] Momentum     : {args.momentum}")

    # ── Reproducibility ───────────────────────────────────────────────────

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Data ───────────────────────────────────────────────────────────────

    data_dir = Path(args.data_dir)

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\n[INFO] Loading CIFAR-10 dataset from: "
        f"{data_dir}"
    )

    # ----------------------------------------------------------------------
    # Important:
    #
    # We create TWO versions of the CIFAR-10 training dataset:
    #
    # 1. train_full:
    #       uses training augmentation
    #
    # 2. val_full:
    #       uses clean validation preprocessing
    #
    # They contain exactly the same 50,000 CIFAR-10 training images.
    #
    # The same indices will then be used for the 45k/5k split.
    # ----------------------------------------------------------------------

    train_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_train_transform(),
    )

    val_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=get_val_transform(),
    )

    # Official CIFAR-10 training set contains 50,000 images.
    total_train = len(train_full)

    if total_train != 50000:
        raise RuntimeError(
            f"Expected 50,000 CIFAR-10 training images, "
            f"but found {total_train}."
        )

    # ── Create deterministic 45k / 5k split ──────────────────────────────

    generator = torch.Generator()

    generator.manual_seed(args.seed)

    indices = torch.randperm(
        total_train,
        generator=generator,
    ).tolist()

    train_indices = indices[:45000]

    val_indices = indices[45000:]

    print(
        f"\n[INFO] Dataset split:"
    )

    print(
        f"       Training   : {len(train_indices)}"
    )

    print(
        f"       Validation : {len(val_indices)}"
    )

    # Subset keeps the exact same image indices,
    # but allows different transforms for train/validation.

    train_ds = Subset(
        train_full,
        train_indices,
    )

    val_ds = Subset(
        val_full,
        val_indices,
    )

    # ── Official CIFAR-10 test set ────────────────────────────────────────

    test_ds = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_val_transform(),
    )

    print(
        f"       Test       : {len(test_ds)}"
    )

    # ── DataLoaders ───────────────────────────────────────────────────────

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # ── Class information ─────────────────────────────────────────────────

    print(
        "\n[INFO] CIFAR-10 classes:"
    )

    for idx, name in enumerate(CLASS_NAMES):

        print(
            f"       {idx}: {name}"
        )

    # ── Model ─────────────────────────────────────────────────────────────

    model = build_model(
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    model = model.to(device)

    print(
        "\n[INFO] ResNet-18 built."
    )

    print(
        "[INFO] Training from scratch: YES"
    )

    print(
        "[INFO] ImageNet pretrained weights: NO"
    )

    # ── Training configuration ────────────────────────────────────────────

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=0.0,
    )

    # AMP is used only on CUDA.
    amp_enabled = (
        not args.no_amp
        and device.type == "cuda"
    )

    scaler = GradScaler(
        enabled=amp_enabled
    )

    # ── Resume support ────────────────────────────────────────────────────

    start_epoch = 1

    best_val_acc = 0.0

    if args.resume:

        resume_path = Path(args.resume)

        if resume_path.exists():

            print(
                f"\n[INFO] Resuming from: "
                f"{resume_path}"
            )

            checkpoint = torch.load(
                resume_path,
                map_location=device,
                weights_only=False,
            )

            if isinstance(checkpoint, dict):

                if "model_state_dict" in checkpoint:
                    model.load_state_dict(
                        checkpoint["model_state_dict"]
                    )

                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(
                        checkpoint["optimizer_state_dict"]
                    )

                start_epoch = (
                    checkpoint.get("epoch", 0) + 1
                )

                best_val_acc = checkpoint.get(
                    "val_acc",
                    0.0,
                )

                # Restore scheduler position if possible.
                for _ in range(
                    start_epoch - 1
                ):
                    scheduler.step()

                print(
                    f"[INFO] Resumed at epoch "
                    f"{start_epoch}"
                )

                print(
                    f"[INFO] Best validation accuracy: "
                    f"{best_val_acc:.4f}"
                )

            else:

                model.load_state_dict(
                    checkpoint
                )

        else:

            print(
                f"[WARNING] Resume checkpoint not found: "
                f"{resume_path}"
            )

    # ── Checkpoint directory ──────────────────────────────────────────────

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ── Training log ──────────────────────────────────────────────────────

    log = []

    # ── Training loop ─────────────────────────────────────────────────────

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"  Training CIFAR-10 ResNet-18 from scratch"
    )

    print(
        f"  {args.epochs} epochs | "
        f"45,000 train | "
        f"5,000 validation"
    )

    print(
        f"{'=' * 70}\n"
    )

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        print(
            f"Epoch {epoch:>3}/{args.epochs}"
        )

        # ── Train ────────────────────────────────────────────────────────

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
        )

        # ── Validation ───────────────────────────────────────────────────

        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Update learning rate.
        scheduler.step()

        # ── Best model ───────────────────────────────────────────────────

        is_best = val_acc > best_val_acc

        marker = (
            "  ★ NEW BEST"
            if is_best
            else ""
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"  train_loss={train_loss:.4f}  "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
            f"{marker}"
        )

        if is_best:

            best_val_acc = val_acc

            checkpoint = {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "val_acc": val_acc,

                "val_loss": val_loss,

                "args": vars(args),

                "class_names": CLASS_NAMES,

                "num_classes": NUM_CLASSES,
            }

            torch.save(
                checkpoint,
                CHECKPOINT_DIR / "best_model.pth",
            )

            print(
                f"  → Best checkpoint saved "
                f"(val_acc={val_acc:.4f})"
            )

        # ── Logging ──────────────────────────────────────────────────────

        log.append(
            {
                "epoch": epoch,

                "train_loss":
                    round(train_loss, 6),

                "train_acc":
                    round(train_acc, 6),

                "val_loss":
                    round(val_loss, 6),

                "val_acc":
                    round(val_acc, 6),

                "lr":
                    round(current_lr, 8),
            }
        )

        save_json(
            log,
            TRAINING_LOG_PATH,
        )

    # ── Load best model ───────────────────────────────────────────────────

    best_model_path = (
        CHECKPOINT_DIR / "best_model.pth"
    )

    if best_model_path.exists():

        print(
            "\n[INFO] Loading best model "
            "for final test evaluation..."
        )

        checkpoint = torch.load(
            best_model_path,
            map_location=device,
            weights_only=False,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

    # ── Final test evaluation ─────────────────────────────────────────────

    test_loss, test_acc = evaluate_test(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"  TRAINING COMPLETE"
    )

    print(
        f"{'=' * 70}"
    )

    print(
        f"  Best validation accuracy : "
        f"{best_val_acc:.4f}"
    )

    print(
        f"  Test loss                : "
        f"{test_loss:.4f}"
    )

    print(
        f"  Test accuracy            : "
        f"{test_acc:.4f}"
    )

    print(
        f"  Test accuracy (%)        : "
        f"{test_acc * 100:.2f}%"
    )

    print(
        f"  Checkpoint               : "
        f"{best_model_path}"
    )

    print(
        f"  Training log             : "
        f"{TRAINING_LOG_PATH}"
    )

    print(
        f"{'=' * 70}\n"
    )


# ─── Entry Point ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
