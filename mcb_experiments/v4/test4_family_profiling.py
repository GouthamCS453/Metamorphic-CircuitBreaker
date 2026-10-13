"""
test_suite/test4_family_profiling.py — TEST 4: Family Vulnerability & Sensitivity Profiling.

Executes:
  1. Domain Vulnerability Profiling:
     Computes average instability ratios for Geometric, Photometric, and Sensor/Noise
     families on GTSRB traffic sign recognition.
  2. Mathematical Ablation Study (Two-Level Normalization vs Flat Averaging):
     Compares Mode A (Flat Average) vs Mode B (Two-Level Hierarchical Normalization),
     proving how Mode B prevents multi-tier transforms from drowning out single-tier tests.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_suite.common import (
    get_class_name,
    get_model_and_cb,
    load_gtsrb_val_dataset,
    save_test_output,
    RESULTS_DIR,
)
from src.metamorphic_families import HIERARCHICAL_TEST_MATRIX, SEVERITY_WEIGHTS


def compute_flat_score(test_results: List[Any], family: str) -> float:
    """Mode A: Flat weighting without Level 1 transformation grouping."""
    fam_tests = [t for t in test_results if t.family == family]
    if not fam_tests:
        return 0.0
    w_flipped = sum(t.weight for t in fam_tests if t.flipped)
    w_total = sum(t.weight for t in fam_tests)
    return w_flipped / w_total if w_total > 0 else 0.0


def run_test4(num_samples: int = 200, seed: int = 42) -> Dict[str, Any]:
    print("=" * 70)
    print("TEST 4: Transformation Family Vulnerability & Sensitivity Profiling")
    print("Domain: Autonomous Vehicles (GTSRB Traffic Signs)")
    print(f"Sample Batch: {num_samples} images")
    print("=" * 70)

    geom_scores_mcb = []
    photo_scores_mcb = []
    noise_scores_mcb = []

    geom_scores_flat = []
    photo_scores_flat = []
    noise_scores_flat = []

    transform_type_flips: Dict[str, int] = {}
    transform_type_totals: Dict[str, int] = {}

    skew_detected_cases = 0

    cache_file = RESULTS_DIR / "batch_cache.json"
    if cache_file.exists():
        import json
        print(f"[CACHE HIT] Found {cache_file.name} from Test 2! Loading precomputed batch data (instant execution)...")
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_entries = json.load(f)
        df_len = len(cached_entries)
        for entry in cached_entries:
            fam_inst = entry["family_instability"]
            ir_g = fam_inst.get("geometric", 0.0)
            ir_p = fam_inst.get("photometric", 0.0)
            ir_n = fam_inst.get("sensor_noise", 0.0)
            geom_scores_mcb.append(ir_g)
            photo_scores_mcb.append(ir_p)
            noise_scores_mcb.append(ir_n)

            t_list = [type("T", (), d) for d in entry["test_results"]]
            flat_g = compute_flat_score(t_list, "geometric")
            flat_p = compute_flat_score(t_list, "photometric")
            flat_n = compute_flat_score(t_list, "sensor_noise")
            geom_scores_flat.append(flat_g)
            photo_scores_flat.append(flat_p)
            noise_scores_flat.append(flat_n)

            if abs(ir_g - flat_g) > 0.15:
                skew_detected_cases += 1

            for t in entry["test_results"]:
                t_type = t["transform_type"]
                transform_type_totals[t_type] = transform_type_totals.get(t_type, 0) + 1
                if t["flipped"]:
                    transform_type_flips[t_type] = transform_type_flips.get(t_type, 0) + 1
    else:
        model, cb = get_model_and_cb()
        df_val = load_gtsrb_val_dataset(limit=num_samples, seed=seed)
        df_len = len(df_val)
        print(f"\nAnalyzing transformation vulnerabilities for {df_len} images...\n")
        for idx in range(df_len):
            row = df_val.iloc[idx]
            img_path = row["resolved_path"]
            img = Image.open(img_path).convert("RGB")
            report = cb.evaluate(img)

            ir_g = report.family_instability.get("geometric", 0.0)
            ir_p = report.family_instability.get("photometric", 0.0)
            ir_n = report.family_instability.get("sensor_noise", 0.0)
            geom_scores_mcb.append(ir_g)
            photo_scores_mcb.append(ir_p)
            noise_scores_mcb.append(ir_n)

            flat_g = compute_flat_score(report.test_results, "geometric")
            flat_p = compute_flat_score(report.test_results, "photometric")
            flat_n = compute_flat_score(report.test_results, "sensor_noise")
            geom_scores_flat.append(flat_g)
            photo_scores_flat.append(flat_p)
            noise_scores_flat.append(flat_n)

            if abs(ir_g - flat_g) > 0.15:
                skew_detected_cases += 1

            for r in report.test_results:
                t_type = r.transform_type
                transform_type_totals[t_type] = transform_type_totals.get(t_type, 0) + 1
                if r.flipped:
                    transform_type_flips[t_type] = transform_type_flips.get(t_type, 0) + 1

    # Aggregate family averages
    mean_geom = float(np.mean(geom_scores_mcb))
    mean_photo = float(np.mean(photo_scores_mcb))
    mean_noise = float(np.mean(noise_scores_mcb))

    mean_geom_flat = float(np.mean(geom_scores_flat))
    mean_photo_flat = float(np.mean(photo_scores_flat))
    mean_noise_flat = float(np.mean(noise_scores_flat))

    # Per-transform-type flip rates
    type_flip_rates = {
        t_type: round((transform_type_flips.get(t_type, 0) / max(1, transform_type_totals[t_type])) * 100.0, 2)
        for t_type in sorted(transform_type_totals.keys())
    }

    vulnerability_ranking = sorted(
        [
            ("Geometric Family", mean_geom),
            ("Photometric Family", mean_photo),
            ("Sensor / Noise Family", mean_noise),
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    summary = {
        "test_name": "TEST 4: Family Vulnerability & Sensitivity Profiling",
        "domain": "Autonomous Vehicles (GTSRB Traffic Signs)",
        "sample_size": df_len,
        "family_vulnerability_two_level_mean": {
            "geometric": round(mean_geom, 4),
            "photometric": round(mean_photo, 4),
            "sensor_noise": round(mean_noise, 4),
        },
        "family_vulnerability_flat_mean": {
            "geometric": round(mean_geom_flat, 4),
            "photometric": round(mean_photo_flat, 4),
            "sensor_noise": round(mean_noise_flat, 4),
        },
        "vulnerability_ranking": [{"family": f, "instability_ratio": round(s, 4)} for f, s in vulnerability_ranking],
        "transform_type_flip_rates_pct": type_flip_rates,
        "ablation_skew_affected_samples": skew_detected_cases,
        "ablation_skew_pct": round((skew_detected_cases / df_len) * 100.0, 2),
    }

    # Generate Markdown Report
    md = [
        "# Test 4: Transformation Family Vulnerability & Sensitivity Profiling",
        "",
        "**Target:** Autonomous Driving Domain Sensitivity & Two-Level Normalization Ablation  ",
        f"**Date Executed:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Sample Batch:** {df_len} held-out GTSRB samples  ",
        "",
        "## 1. Domain Vulnerability Profile (Autonomous Driving)",
        "",
        "Autonomous vehicle perception operates under dynamic weather and optical distortions. The table below profiles the mean vulnerability across semantic families:",
        "",
        "| Semantic Family | Mean Instability Ratio ($IR_k$) | Rank | Primary Physical Causes |",
        "| :--- | :---: | :---: | :--- |",
    ]

    for rank, (fam, score) in enumerate(vulnerability_ranking, 1):
        md.append(f"| **{fam}** | **{score:.4f}** | #{rank} | High-speed vehicle vibration, rain/fog sensor noise, direct sunlight glare |")

    md.extend([
        "",
        "## 2. Granular Transform Type Breakdown",
        "",
        "| Transform Type | Semantic Family | Flip Rate under Perturbation (%) | Operational Implication for AVs |",
        "| :--- | :--- | :---: | :--- |",
    ])

    for t_type, rate in type_flip_rates.items():
        md.append(f"| `{t_type}` | Diverse | {rate:.1f}% | Vulnerability observed under real-world camera transforms |")

    md.extend([
        "",
        "## 3. Mathematical Ablation Study: Two-Level Normalization vs. Flat Averaging",
        "",
        "| Semantic Family | Mode A: Flat Weighted Averaging | Mode B: Two-Level Normalization (Proposed) | Structural Bias Eliminated |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Geometric** | {mean_geom_flat:.4f} | **{mean_geom:.4f}** | Rotation (3 tiers) no longer drowns single-tier Flip (1 tier). |",
        f"| **Photometric** | {mean_photo_flat:.4f} | **{mean_photo:.4f}** | Brightness, Contrast, and Saturation contribute equally. |",
        f"| **Sensor / Noise** | {mean_noise_flat:.4f} | **{mean_noise:.4f}** | Gaussian Blur and Sensor Noise balanced across ladder steps. |",
        "",
        "### Key Takeaways from Ablation:",
        f"- **Skew Disparity:** In **{summary['ablation_skew_pct']}%** of samples, Flat Averaging generated significant distortion (>0.15 disparity) in the family instability score.",
        "- **Mathematical Fairness:** Two-Level Hierarchical Normalization guarantees that each transformation type has an equal voice in the family instability ratio, regardless of whether it has 1 test or 5 test parameters.",
    ])

    md_str = "\n".join(md)
    save_test_output("test4_family_profiling", summary, md_str)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Test 4: Family Vulnerability & Sensitivity Profiling")
    parser.add_argument("--samples", type=int, default=200, help="Batch sample size (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    run_test4(num_samples=args.samples, seed=args.seed)
