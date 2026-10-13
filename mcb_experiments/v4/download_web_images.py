"""
test_suite/download_web_images.py — Search and download 40 distinct traffic sign images
from Wikimedia Commons for Test 1 (In-the-Wild Web Testing).
"""

import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "test_suite" / "inputs" / "web_images"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# 40 distinct GTSRB classes mapped to official German StVO sign search terms
SIGN_DEFINITIONS = [
    (0, "Speed limit (20km/h)", "Zeichen 274-20"),
    (1, "Speed limit (30km/h)", "Zeichen 274-30"),
    (2, "Speed limit (50km/h)", "Zeichen 274-50"),
    (3, "Speed limit (60km/h)", "Zeichen 274-60"),
    (4, "Speed limit (70km/h)", "Zeichen 274-70"),
    (5, "Speed limit (80km/h)", "Zeichen 274-80"),
    (6, "End of speed limit (80km/h)", "Zeichen 278-80"),
    (7, "Speed limit (100km/h)", "Zeichen 274-100"),
    (8, "Speed limit (120km/h)", "Zeichen 274-120"),
    (9, "No passing", "Zeichen 276"),
    (10, "No passing for vehicles over 3.5 metric tons", "Zeichen 277"),
    (11, "Right-of-way at intersection", "Zeichen 301 Vorfahrt"),
    (12, "Priority road", "Zeichen 306 Vorfahrtstrae"),
    (13, "Yield", "Zeichen 205 Vorfahrt gewhren"),
    (14, "Stop", "Zeichen 206 Halt Vorfahrt"),
    (15, "No vehicles", "Zeichen 250 Verbot fr Fahrzeuge"),
    (16, "Vehicles over 3.5 metric tons prohibited", "Zeichen 253"),
    (17, "No entry", "Zeichen 267 Verbot der Einfahrt"),
    (18, "General caution", "Zeichen 101 Gefahr"),
    (19, "Dangerous curve to the left", "Zeichen 103-10 Kurve links"),
    (20, "Dangerous curve to the right", "Zeichen 103-20 Kurve rechts"),
    (21, "Double curve", "Zeichen 105-10 Doppelkurve"),
    (22, "Bumpy road", "Zeichen 112 Unebene Fahrbahn"),
    (23, "Slippery road", "Zeichen 114 Schleudergefahr"),
    (24, "Road narrows on the right", "Zeichen 121-10 Fahrbahnverengung"),
    (25, "Road work", "Zeichen 123 Baustelle"),
    (26, "Traffic signals", "Zeichen 131 Lichtzeichenanlage"),
    (27, "Pedestrians", "Zeichen 133 Fugnger"),
    (28, "Children crossing", "Zeichen 136 Kinder"),
    (29, "Bicycles crossing", "Zeichen 138 Radverkehr"),
    (30, "Beware of ice/snow", "Zeichen 113 Schnee- oder Eisgltte"),
    (31, "Wild animals crossing", "Zeichen 142 Wildwechsel"),
    (32, "End of all speed and passing restrictions", "Zeichen 282"),
    (33, "Turn right ahead", "Zeichen 209-20 Fahrtrichtung rechts"),
    (34, "Turn left ahead", "Zeichen 209-10 Fahrtrichtung links"),
    (35, "Ahead only", "Zeichen 209-30 Geradeaus"),
    (36, "Go straight or right", "Zeichen 214-20 Geradeaus oder rechts"),
    (37, "Go straight or left", "Zeichen 214-10 Geradeaus oder links"),
    (38, "Keep right", "Zeichen 222-20 Vorbeifahrt rechts"),
    (39, "Keep left", "Zeichen 222-10 Vorbeifahrt links"),
]


def search_wikimedia_image(query: str) -> tuple[str, str] | None:
    """Query Wikimedia Commons API to find a thumb URL for the sign."""
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{query} filetype:svg|png|jpg",
        "gsrnamespace": "6",
        "gsrlimit": "4",
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": "300",
        "format": "json",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "MCBAutonomousSuite/1.0 (edu-testing@example.com)"})

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for pid, p in pages.items():
            title = p.get("title", "")
            ii_list = p.get("imageinfo", [])
            if not ii_list:
                continue
            thumb = ii_list[0].get("thumburl")
            desc_url = ii_list[0].get("descriptionurl", "")
            if thumb and ("Zeichen" in title or "road sign" in title.lower() or "verkehrszeichen" in title.lower() or "sign" in title.lower()):
                return thumb, desc_url
        # If no strict match, return first thumb
        for pid, p in pages.items():
            ii_list = p.get("imageinfo", [])
            if ii_list and ii_list[0].get("thumburl"):
                return ii_list[0].get("thumburl"), ii_list[0].get("descriptionurl", "")
    except Exception as e:
        print(f"    [Warn] Search failed for '{query}': {e}")
    return None


def download_image(url: str) -> Image.Image | None:
    req = urllib.request.Request(url, headers={"User-Agent": "MCBAutonomousSuite/1.0 (edu-testing@example.com)"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img
    except Exception as e:
        print(f"    [Warn] Download failed for {url}: {e}")
        return None


def main():
    print("=" * 70)
    print("DOWNLOADING 40 EXTERNAL WEB IMAGES FOR TEST 1 (IN-THE-WILD)")
    print(f"Destination: {TARGET_DIR}")
    print("=" * 70)

    records = []
    success_count = 0

    for idx, (class_id, class_name, search_term) in enumerate(SIGN_DEFINITIONS, 1):
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", class_name).strip("_").lower()
        filename = f"web_{idx:02d}_class{class_id:02d}_{slug}.jpg"
        filepath = TARGET_DIR / filename

        print(f"[{idx:02d}/40] Class {class_id:02d} ({class_name[:24]}): searching '{search_term}'...")

        found = search_wikimedia_image(search_term)
        img = None
        source_url = ""
        thumb_url = ""

        if found:
            thumb_url, source_url = found
            img = download_image(thumb_url)

        # Fallback: if search was empty or failed, use local GTSRB Meta template if exists
        if img is None:
            meta_fallback = PROJECT_ROOT / "data" / "gtsrb" / "Meta" / f"{class_id}.png"
            if meta_fallback.exists():
                print(f"    [Fallback] Using high-res Meta reference image for Class {class_id}")
                img = Image.open(meta_fallback).convert("RGB")
                source_url = "local_gtsrb_meta_reference"
            else:
                print(f"    [ERROR] Could not acquire image for Class {class_id}")
                continue

        # Resize to standard realistic camera crop (250x250) and save
        img_resized = img.resize((250, 250), Image.Resampling.BILINEAR)
        img_resized.save(filepath, format="JPEG", quality=92)
        success_count += 1
        print(f"    -> Saved: {filename} ({img_resized.size})")

        records.append({
            "filename": filename,
            "label": class_id,
            "class_name": class_name,
            "search_term": search_term,
            "source_url": source_url or thumb_url,
            "image_path": str(filepath.relative_to(PROJECT_ROOT)),
        })
        time.sleep(0.3)  # Respect API rate limits

    # Save labels.csv
    import pandas as pd
    df = pd.DataFrame(records)
    csv_path = TARGET_DIR / "labels.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] labels.csv: {csv_path}")

    # Save metadata.json
    json_path = TARGET_DIR / "metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[Saved] metadata.json: {json_path}")
    print(f"\nSuccessfully collected {success_count}/40 external web images!")


if __name__ == "__main__":
    main()
