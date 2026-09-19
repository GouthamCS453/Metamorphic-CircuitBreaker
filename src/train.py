"""
src/train.py — Train MobileNetV3-Small on the GTSRB traffic-sign dataset.
Steps
-----
1. Load ISIC_2019_Training_GroundTruth.csv and resolve image paths.
2. Perform a stratified 70/30 train/val split.
3. Train with AdamW + CosineAnnealingLR and automatic mixed precision (AMP).
4. After every epoch, save the model as ``checkpoints/best_model.pth`` only
   when validation accuracy improves (best-model checkpoint mechanism).
5. Write epoch-level metrics to ``checkpoints/training_log.json``.

Usage
-----
    python src/train.py --data_dir data/gtsrb --epochs 30 --batch_size 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    CLASS_NAMES, IMAGE_SIZE,
    build_model, get_device,
    CHECKPOINT_DIR, TRAINING_LOG_PATH,
    get_train_transform, get_val_transform,
    save_json,
)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class GTSRBDataset(Dataset):
    """
    PyTorch Dataset for ISIC 2019.

    Args:
        image_paths: List of absolute paths to JPEG images.
        labels:      Corresponding integer class indices (0-8).
        transform:   torchvision transform pipeline.
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path  = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"[WARNING] Cannot open {path}: {exc}. Using blank image.")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image, label


def load_gtsrb_dataframe(data_dir: Path) -> pd.DataFrame:
    """
    Parse GTSRB Train.csv and resolve image file paths.

    Expected layout:

        data_dir/
          Train.csv
          Train/
            0/
            1/
            ...
            42/

    Returns:
        DataFrame with columns ``image``, ``image_path``, ``label``.
    """
    csv_path = data_dir / "Train.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"GTSRB Train.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Resolve image paths from the Path column
    df["image_path"] = df["Path"].apply(
        lambda p: str(data_dir / p)
    )

    # GTSRB ClassId is already the integer label
    df["label"] = df["ClassId"].astype(int)

    # Keep an image identifier for compatibility with the existing code
    df["image"] = df["Path"].apply(
        lambda p: Path(p).stem
    )

    # Verify that referenced images exist
    missing = df[~df["image_path"].map(lambda p: Path(p).exists())]

    if not missing.empty:
        raise FileNotFoundError(
            f"{len(missing)} GTSRB images referenced by Train.csv were not found."
        )

    return df[["image", "image_path", "label"]]


# --- Training Helpers ---------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """
    Run one forward + backward pass over the training DataLoader.

    Returns:
        Tuple of (average_loss, accuracy) over the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Evaluate the model on the validation DataLoader.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False, unit="batch"):
        images, labels = images.to(device), labels.to(device)
        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# --- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train MobileNetV3-Small on GTSRB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
    "--data_dir",
    type=str,
    default="data/gtsrb",
    help="GTSRB dataset root",
)
    p.add_argument("--epochs",       type=int,   default=30,              help="Number of training epochs")
    p.add_argument("--batch_size",   type=int,   default=32,              help="Mini-batch size")
    p.add_argument("--lr",           type=float, default=1e-4,            help="AdamW initial learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4,            help="AdamW weight decay")
    p.add_argument("--num_workers",  type=int,   default=0,               help="DataLoader workers (use 0 on Windows)")
    p.add_argument("--seed",         type=int,   default=42,              help="Random seed")
    p.add_argument("--no_amp",       action="store_true",                  help="Disable automatic mixed precision")
    p.add_argument("--resume",       type=str,   default=None,            help="Resume from checkpoint path")
    return p.parse_args()


# --- Main ---------------------------------------------------------------------

def main():
    args   = parse_args()
    device = get_device()
    print(f"[INFO] Device  : {device}")
    print(f"[INFO] Epochs  : {args.epochs}")
    print(f"[INFO] Batch   : {args.batch_size}")
    print(f"[INFO] LR      : {args.lr}")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- Data ------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    print(f"\n[INFO] Loading dataset from: {data_dir}")
    df = load_gtsrb_dataframe(data_dir)
    print(f"[INFO] Total images: {len(df)}")

    class_counts = df["label"].value_counts().sort_index()
    print("[INFO] Class distribution:")
    for idx, cnt in class_counts.items():
        print(f"       {CLASS_NAMES[idx]:>4}  {cnt}")

    # Stratified 70/30 split
    train_df, val_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=args.seed,
    )
    print(f"\n[INFO] Train : {len(train_df)} images")
    print(f"[INFO] Val   : {len(val_df)} images")

    # Persist split CSVs so predict.py can load them later
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    train_df[["image", "image_path", "label"]].to_csv(
        CHECKPOINT_DIR / "train_split.csv", index=False
    )
    val_df[["image", "image_path", "label"]].to_csv(
        CHECKPOINT_DIR / "val_split.csv", index=False
    )
    print(f"[INFO] Split CSVs saved -> {CHECKPOINT_DIR}")

    train_ds = GTSRBDataset(train_df["image_path"].tolist(),
                           train_df["label"].tolist(),
                           transform=get_train_transform())
    val_ds   = GTSRBDataset(val_df["image_path"].tolist(),
                           val_df["label"].tolist(),
                           transform=get_val_transform())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))

    # -- Model -----------------------------------------------------------------
    model = build_model(pretrained=True).to(device)
    print("\n[INFO] MobileNetV3-Small built (ImageNet pretrained)")

    start_epoch   = 1
    best_val_acc  = 0.0

    # Optional resume
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch  = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("val_acc", 0.0)
            print(f"[INFO] Resumed from epoch {start_epoch - 1} "
                  f"(best val acc = {best_val_acc:.4f})")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler(enabled=not args.no_amp and device.type == "cuda")

    # -- Training Loop ---------------------------------------------------------
    log = []
    print(f"\n{'='*62}")
    print(f"  Training MobileNetV3-Small for {args.epochs} epochs")
    print(f"{'='*62}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch:>3}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        is_best = val_acc > best_val_acc
        marker  = "  ★ NEW BEST" if is_best else ""
        print(
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{marker}"
        )

        if is_best:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc":          val_acc,
                    "val_loss":         val_loss,
                    "args":             vars(args),
                },
                CHECKPOINT_DIR / "best_model.pth",
            )
            print(f"  → Checkpoint saved  (val_acc={val_acc:.4f})")

        log.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "lr":         scheduler.get_last_lr()[0],
        })
        save_json(log, TRAINING_LOG_PATH)

    print(f"\n{'='*62}")
    print(f"  DONE — best val_acc = {best_val_acc:.4f}")
    print(f"  Checkpoint → {CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
