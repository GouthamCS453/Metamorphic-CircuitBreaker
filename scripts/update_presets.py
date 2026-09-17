import shutil, json
from pathlib import Path

presets_data = [
    {
        "dir": "preset_stable_nevus",
        "src": "data/isic2019/ISIC_2019_Training_Input/ISIC_0067135.jpg",
        "meta": {
            "preset_id": "preset_stable_nevus",
            "name": "🟢 Rock-Solid Lesion (0 Flips)",
            "expected_state": "CLOSED",
            "cbi_ref": "0.0156",
            "flips_ref": "0 / 22",
            "description": "Demonstrates normal, safe operation. Predictions remain 100% stable across all geometric, photometric, and sensor noise perturbations. Circuit Breaker stays CLOSED (Auto-Approved)."
        }
    },
    {
        "dir": "preset_boundary_flips",
        "src": "data/isic2019/ISIC_2019_Training_Input/ISIC_0013736_downsampled.jpg",
        "meta": {
            "preset_id": "preset_boundary_flips",
            "name": "🟢 Boundary Sensitivity (4 Flips, but Safe)",
            "expected_state": "CLOSED",
            "cbi_ref": "0.1792",
            "flips_ref": "4 / 22",
            "description": "Illustrates Phase 2 resilience to false alarms. 4 flips occur, but exclusively at extreme severe boundary tiers (0.80x zoom, 0.25x saturation, high noise). No single family reaches tau_fam=0.35, keeping CBI < 0.25."
        }
    },
    {
        "dir": "preset_half_open_warning",
        "src": "data/isic2019/ISIC_2019_Training_Input/ISIC_0059024.jpg",
        "meta": {
            "preset_id": "preset_half_open_warning",
            "name": "🟡 Systematic Geometric Fragility (3 Mild Flips)",
            "expected_state": "HALF_OPEN",
            "cbi_ref": "0.3738",
            "flips_ref": "3 / 22",
            "description": "Demonstrates how a small number of MILD flips in a single family compromises that family (IR_geom >= 0.35), pushing CBI into HALF-OPEN [0.25 - 0.55]. Approved with clinical warning notice."
        }
    },
    {
        "dir": "preset_open_tripped",
        "src": "data/isic2019/ISIC_2019_Training_Input/ISIC_0055454.jpg",
        "meta": {
            "preset_id": "preset_open_tripped",
            "name": "🔴 Multi-Family Collapse (12 Flips - Tripped!)",
            "expected_state": "OPEN",
            "cbi_ref": "0.7183",
            "flips_ref": "12 / 22",
            "description": "Demonstrates full safety interception. The model collapses across rotation, lighting, and noise simultaneously. Both IR_geom and IR_photo cross 0.35 (CFS=0.667). CBI = 0.7183 >= 0.55 -> Breaker TRIPS to OPEN."
        }
    }
]

base_preset_dir = Path("data/presets")
base_preset_dir.mkdir(parents=True, exist_ok=True)
for child in base_preset_dir.iterdir():
    if child.is_dir():
        shutil.rmtree(child)

for p in presets_data:
    p_dir = base_preset_dir / p["dir"]
    p_dir.mkdir(parents=True, exist_ok=True)
    src_file = Path(p["src"])
    if src_file.exists():
        shutil.copy2(src_file, p_dir / src_file.name)
        p["meta"]["image_filename"] = src_file.name
        (p_dir / "metadata.json").write_text(json.dumps(p["meta"], indent=2))
        print("Setup", p["dir"], "with", src_file.name)
    else:
        print("Missing", src_file)
print("Preset setup complete.")
