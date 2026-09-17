import pandas as pd, shutil, json
from pathlib import Path
from src.utils import CLASS_NAMES

df = pd.read_csv("checkpoints/val_split.csv")
print("Class distribution:")
print(df["label"].value_counts().sort_index())
print("CLASS_NAMES:", CLASS_NAMES)

PRESET_DIR = Path("data/presets")
PRESET_TARGETS = {
    "NV": ("preset_stable", "Stable Nevus (Benign Mole)", "CLOSED",
           "A clearly defined benign mole expected to remain stable across all perturbations."),
    "BKL": ("preset_borderline", "Ambiguous Lesion (Borderline)", "HALF_OPEN",
            "A lesion with ambiguous morphology -- may show mild sensitivity to perturbations."),
    "MEL": ("preset_brittle", "High-Risk Melanoma Candidate", "OPEN",
            "Potentially malignant candidate -- expect prediction inconsistencies."),
}

nv_idx = CLASS_NAMES.index("NV") if "NV" in CLASS_NAMES else None
bkl_idx = CLASS_NAMES.index("BKL") if "BKL" in CLASS_NAMES else None
mel_idx = CLASS_NAMES.index("MEL") if "MEL" in CLASS_NAMES else None

label_map = {}
for class_name, label_idx in zip(CLASS_NAMES, range(len(CLASS_NAMES))):
    label_map[class_name] = label_idx

for cn, (pid, name, state, desc) in PRESET_TARGETS.items():
    if cn not in label_map:
        print(f"Class {cn} not found in CLASS_NAMES, skipping")
        continue
    label_id = label_map[cn]
    sub = df[df["label"] == label_id]
    if sub.empty:
        print(f"No images found for {cn} (label={label_id})")
        continue
    row = sub.iloc[0]
    src = Path(row["image_path"])
    if not src.exists():
        print(f"Image not found: {src}")
        continue
    dest_dir = PRESET_DIR / pid
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    meta = {
        "preset_id": pid,
        "name": name,
        "expected_state": state,
        "description": desc,
        "class_name": cn,
        "label_id": label_id,
        "image_filename": src.name,
    }
    (dest_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  OK: {pid} -> {src.name} (class={cn}, label={label_id})")

print("Done.")
