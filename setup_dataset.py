"""
setup_dataset.py — Extract isic.zip and generate GroundTruth CSV.

The zip is structured as:
    <CLASS_NAME>/ISIC_XXXXXXX.jpg

This script:
1. Extracts all images into data/isic2019/ISIC_2019_Training_Input/
2. Generates data/isic2019/ISIC_2019_Training_GroundTruth.csv with one-hot columns
"""

import csv
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import sys

# ─── Config ───────────────────────────────────────────────────────────────────

ZIP_PATH   = Path("data/isic.zip")
OUT_DIR    = Path("data/isic2019")
IMG_DIR    = OUT_DIR / "ISIC_2019_Training_Input"
CSV_PATH   = OUT_DIR / "ISIC_2019_Training_GroundTruth.csv"

# ISIC 2019 class list — keep consistent column order with training code
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]


def main():
    if not ZIP_PATH.exists():
        print(f"[ERROR] Zip not found: {ZIP_PATH}")
        sys.exit(1)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Setup] Extracting {ZIP_PATH} → {IMG_DIR}")
    print("[Setup] This may take a few minutes for a 9 GB archive…\n")

    records = []  # list of (image_id, class_name)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".jpg") or m.endswith(".jpeg") or m.endswith(".png")]
        total = len(members)
        print(f"[Setup] {total} image files to extract")

        for i, member in enumerate(members, 1):
            parts = member.strip("/").split("/")
            if len(parts) < 2:
                continue
            class_name = parts[0].upper()   # e.g. "AK"
            filename   = parts[-1]           # e.g. "ISIC_0024468.jpg"
            image_id   = Path(filename).stem # e.g. "ISIC_0024468"

            dest = IMG_DIR / filename
            if dest.exists():
                # Already extracted — just record
                records.append((image_id, class_name))
                if i % 1000 == 0:
                    print(f"  Skipping (already exists) {i}/{total}  {filename}", end="\r")
                continue

            # Extract to flat directory
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())

            records.append((image_id, class_name))

            if i % 500 == 0 or i == total:
                pct = 100 * i / total
                print(f"  {i:>6}/{total}  ({pct:5.1f}%)  last: {filename}    ", end="\r")

    print(f"\n\n[Setup] Extracted {len(records)} images → {IMG_DIR}")

    # ── Generate CSV ──────────────────────────────────────────────────────────
    print(f"[Setup] Writing ground-truth CSV → {CSV_PATH}")

    # Determine which classes actually appear in the zip
    present_classes = sorted({r[1] for r in records})
    print(f"[Setup] Classes found: {present_classes}")

    # Build ordered class list: keep CLASS_NAMES order, skip missing ones
    ordered_classes = [c for c in CLASS_NAMES if c in present_classes]
    # Append any extra classes not in CLASS_NAMES
    for c in present_classes:
        if c not in ordered_classes:
            ordered_classes.append(c)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["image"] + ordered_classes
        writer.writerow(header)

        for image_id, class_name in sorted(records):
            row = [image_id] + [
                "1.0" if c == class_name else "0.0"
                for c in ordered_classes
            ]
            writer.writerow(row)

    print(f"[Setup] CSV written: {len(records)} rows, columns: image + {ordered_classes}")

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    dist = Counter(r[1] for r in records)
    print("\n[Setup] Class distribution:")
    for cls in ordered_classes:
        print(f"  {cls:<8} {dist.get(cls, 0):>6} images")
    print(f"  {'TOTAL':<8} {len(records):>6} images")
    print("\n[Setup] ✅ Dataset ready!")
    print(f"         Images : {IMG_DIR}")
    print(f"         CSV    : {CSV_PATH}")


if __name__ == "__main__":
    main()
