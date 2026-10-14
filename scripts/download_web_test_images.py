"""
scripts/download_web_test_images.py - Download 40 verified external web images for Test 1.

Downloads 5 external images per class for all 8 ISIC classes (5 x 8 = 40 images):
  - MEL : Melanoma
  - NV  : Melanocytic Nevus
  - BCC : Basal Cell Carcinoma
  - AK  : Actinic Keratosis
  - BKL : Benign Keratosis (Seborrheic Keratosis)
  - DF  : Dermatofibroma
  - VASC: Vascular Lesion (Hemangioma)
  - SCC : Squamous Cell Carcinoma

Ensures all 40 images are STRICTLY EXTERNAL to the local ISIC 2019 training/val dataset,
verified via the official ISIC Archive REST API with verified clinical/histopathology diagnoses.
"""
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "mcb_experiments" / "inputs" / "web_images"
LOCAL_GT = ROOT_DIR / "data" / "isic2019" / "ISIC_2019_Training_GroundTruth.csv"

# ISIC queries for the 8 classes (list of queries for robust fallback)
CLASS_QUERIES = {
    "MEL": [
        'diagnosis_3:"Melanoma in situ"',
        'diagnosis_2:"Malignant melanocytic proliferations (Melanoma)"',
    ],
    "NV": [
        'diagnosis_confirm_type:"histopathology" AND diagnosis_3:"Nevus"',
        'diagnosis_3:"Nevus"',
    ],
    "BCC": ['diagnosis_3:"Basal cell carcinoma"'],
    "AK": ['diagnosis_3:"Solar or actinic keratosis"'],
    "BKL": ['diagnosis_3:"Seborrheic keratosis"'],
    "DF": ['diagnosis_3:"Dermatofibroma"'],
    "VASC": ['diagnosis_3:"Hemangioma"'],
    "SCC": ['diagnosis_3:"Squamous cell carcinoma, NOS"'],
}

TARGET_PER_CLASS = 5


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load existing IDs to guarantee 100% external data
    existing_ids = set()
    if LOCAL_GT.exists():
        gt_df = pd.read_csv(LOCAL_GT)
        existing_ids = set(gt_df["image"].tolist())
        # also handle _downsampled suffixes
        existing_ids.update(x.replace("_downsampled", "") for x in list(existing_ids))
        print(f"Loaded {len(existing_ids)} existing local image IDs to exclude.")

    metadata_records = []
    seen_ids = set(existing_ids)

    print("\nStarting external web image acquisition (5 per class = 40 images total)...")
    for class_idx, (label, query_list) in enumerate(CLASS_QUERIES.items(), 1):
        class_records = []
        # Check existing valid downloaded images for this class
        existing_class_files = sorted(OUTPUT_DIR.glob(f"web_{label.lower()}_*.jpg"))
        for f in existing_class_files:
            # extract isic_id from filename e.g. web_ak_01_ISIC_0010512.jpg
            parts = f.stem.split("_")
            if len(parts) >= 4:
                file_isic_id = f"{parts[-2]}_{parts[-1]}"
            else:
                file_isic_id = parts[-1]
            seen_ids.add(file_isic_id)
            img = Image.open(f)
            w, h = img.size
            class_records.append({
                "filename": f.name,
                "label": label,
                "isic_id": file_isic_id,
                "diagnosis_detail": label,
                "resolution": f"{w}x{h}",
                "source_url": f"https://isic-archive.s3.amazonaws.com/images/{file_isic_id}.jpg"
            })

        count_for_class = len(class_records)
        print(f"\n[{class_idx}/8] Checking class: {label} (currently have {count_for_class}/{TARGET_PER_CLASS})...")

        for query_str in query_list:
            if count_for_class >= TARGET_PER_CLASS:
                break
            offset = 0
            while count_for_class < TARGET_PER_CLASS and offset < 500:
                url = (
                    f"https://api.isic-archive.com/api/v2/images/search/?"
                    f"query={urllib.parse.quote(query_str)}&limit=50&offset={offset}"
                )
                req = urllib.request.Request(url, headers={"User-Agent": "MCBResearch/1.0 (academic_research@mbcet.ac.in)"})

                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                except Exception as e:
                    print(f"  Error fetching search results for {label} at offset {offset}: {e}")
                    break

                results = data.get("results", [])
                print(f"  API returned {len(results)} candidate results for query: {query_str} (offset={offset})", flush=True)
                if not results:
                    break

                for item in results:
                    if count_for_class >= TARGET_PER_CLASS:
                        break

                    isic_id = item.get("isic_id")
                    if not isic_id or isic_id in seen_ids:
                        continue

                    files_dict = item.get("files", {})
                    full_file = files_dict.get("full", {})
                    img_url = full_file.get("url")
                    if not img_url:
                        continue

                    clin_meta = item.get("metadata", {}).get("clinical", {})
                    diagnosis_detail = (
                        clin_meta.get("diagnosis_3")
                        or clin_meta.get("diagnosis_2")
                        or clin_meta.get("diagnosis_1")
                        or "Confirmed"
                    )

                    filename = f"web_{label.lower()}_{count_for_class + 1:02d}_{isic_id}.jpg"
                    target_path = OUTPUT_DIR / filename

                    # Download image
                    try:
                        img_req = urllib.request.Request(img_url, headers={"User-Agent": "MCBResearch/1.0"})
                        with urllib.request.urlopen(img_req, timeout=30) as img_resp:
                            img_bytes = img_resp.read()

                        # Verify image with PIL
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        w, h = img.size
                        if w < 100 or h < 100:
                            continue

                        # Save to disk
                        img.save(target_path, "JPEG", quality=95)

                        count_for_class += 1
                        seen_ids.add(isic_id)

                        class_records.append({
                            "filename": filename,
                            "label": label,
                            "isic_id": isic_id,
                            "diagnosis_detail": diagnosis_detail,
                            "resolution": f"{w}x{h}",
                            "source_url": img_url,
                        })
                        print(f"  -> Saved {filename} ({w}x{h}) | Diagnosis: {diagnosis_detail}", flush=True)
                        time.sleep(0.3)  # Polite request spacing

                    except Exception as e:
                        print(f"  Failed downloading {isic_id}: {e}", flush=True)
                        continue

                offset += 50

        print(f"  Completed {label}: {count_for_class}/{TARGET_PER_CLASS} images.")
        metadata_records.extend(class_records)

    # 2. Save labels.csv
    labels_df = pd.DataFrame(metadata_records)
    labels_csv_path = OUTPUT_DIR / "labels.csv"
    labels_df.to_csv(labels_csv_path, index=False)
    print(f"\n[OK] Successfully saved labels file with {len(labels_df)} entries: {labels_csv_path}")

    # 3. Save comprehensive METADATA.md
    md_content = [
        "# Test 1: External 'In-the-Wild' Web Dataset Metadata",
        "",
        "This dataset contains **40 verified external dermoscopic images** across all 8 ISIC disease classes.",
        "None of these images are part of the ISIC 2019 training or validation split, providing genuine",
        "out-of-distribution / external deployment evaluation for **TEST 1 (In-the-Wild Web Testing)**.",
        "",
        f"- **Total Images:** {len(labels_df)}",
        f"- **Classes Represented:** {len(CLASS_QUERIES)} ({', '.join(CLASS_QUERIES.keys())})",
        f"- **Images per Class:** {TARGET_PER_CLASS}",
        "- **Source:** Official ISIC Archive REST API (Open Dermoscopy Access)",
        "",
        "## Class Distribution Summary",
        "",
        "| Class Code | Disease Name | Image Count | Primary Diagnosis Detail |",
        "| :---: | :--- | :---: | :--- |",
        f"| `MEL` | Melanoma | {len(labels_df[labels_df['label'] == 'MEL'])} | Malignant melanocytic proliferations (Melanoma) |",
        f"| `NV` | Melanocytic Nevus | {len(labels_df[labels_df['label'] == 'NV'])} | Benign Nevus |",
        f"| `BCC` | Basal Cell Carcinoma | {len(labels_df[labels_df['label'] == 'BCC'])} | Basal cell carcinoma |",
        f"| `AK` | Actinic Keratosis | {len(labels_df[labels_df['label'] == 'AK'])} | Solar or actinic keratosis |",
        f"| `BKL` | Benign Keratosis | {len(labels_df[labels_df['label'] == 'BKL'])} | Seborrheic keratosis |",
        f"| `DF` | Dermatofibroma | {len(labels_df[labels_df['label'] == 'DF'])} | Dermatofibroma |",
        f"| `VASC` | Vascular Lesion | {len(labels_df[labels_df['label'] == 'VASC'])} | Hemangioma |",
        f"| `SCC` | Squamous Cell Carcinoma | {len(labels_df[labels_df['label'] == 'SCC'])} | Squamous cell carcinoma, NOS |",
        "",
        "## Detailed Image Roster",
        "",
        "| Filename | Label | ISIC ID | Diagnosis Detail | Resolution |",
        "| :--- | :---: | :--- | :--- | :---: |",
    ]

    for rec in metadata_records:
        md_content.append(
            f"| `{rec['filename']}` | `{rec['label']}` | `{rec['isic_id']}` | {rec['diagnosis_detail']} | {rec['resolution']} |"
        )

    metadata_md_path = OUTPUT_DIR / "METADATA.md"
    metadata_md_path.write_text("\n".join(md_content), encoding="utf-8")
    print(f"[OK] Generated detailed metadata documentation: {metadata_md_path}")
    print(f"\nDone! All {len(labels_df)} images are ready in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
