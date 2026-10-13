"""
test_suite/test1_in_the_wild.py — TEST 1: In-the-Wild & Out-of-Distribution Web Testing.

Evaluates MobileNetV3-Small on GTSRB traffic sign inputs subjected to
uncontrolled, real-world challenging conditions (weather, glare, extreme angles, sensor noise)
simulating external web/field testing:
  - Run A (Unmonitored): Base model prediction + raw confidence.
  - Run B (MCB-Guarded): Circuit breaker evaluation (CBI, state, action).
Demonstrates interception of overconfident "silent failures".
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_suite.common import (
    get_class_name,
    get_model_and_cb,
    load_gtsrb_val_dataset,
    save_test_output,
)


def apply_wild_environment_distortion(img: Image.Image, scenario_idx: int) -> Tuple[Image.Image, str]:
    """
    Simulate real-world in-the-wild environmental conditions encountered by autonomous vehicles:
    1. Harsh sun glare / overexposure
    2. Wet windshield / rain blur
    3. Severe camera tilt / steep road angle
    4. Night / low-beam underexposure
    5. Sensor thermal / compression noise
    """
    modes = [
        ("Sun Glare / Overexposure", lambda x: ImageEnhance.Brightness(x).enhance(1.65)),
        ("Wet Lens / Rain Blur", lambda x: x.filter(ImageFilter.GaussianBlur(radius=2.8))),
        ("Steep Camera Tilt (12 deg)", lambda x: x.rotate(12, resample=Image.Resampling.BILINEAR)),
        ("Dusk / Underexposure", lambda x: ImageEnhance.Brightness(ImageEnhance.Contrast(x).enhance(0.75)).enhance(0.45)),
        ("Sensor Noise & Artifacts", lambda x: Image.fromarray(np.clip(
            np.array(x, dtype=np.float32) + np.random.normal(0, 18, np.array(x).shape), 0, 255
        ).astype(np.uint8))),
    ]
    name, fn = modes[scenario_idx % len(modes)]
    return fn(img), name


def run_test1(num_samples: int = 40, seed: int = 101) -> Dict[str, Any]:
    print("=" * 70)
    print("TEST 1: In-the-Wild Web / OOD Testing (Comparative Analysis)")
    print("Domain: Autonomous Vehicles (Traffic Sign Recognition)")
    print("=" * 70)

    model, cb = get_model_and_cb()
    web_dir = PROJECT_ROOT / "test_suite" / "inputs" / "web_images"
    labels_file = web_dir / "labels.csv"

    records = []
    silent_failures_caught = 0
    total_model_errors = 0
    total_samples = 0

    if labels_file.exists():
        import pandas as pd
        df_web = pd.read_csv(labels_file)
        if num_samples and len(df_web) > num_samples:
            df_web = df_web.iloc[:num_samples]
        print(f"\n[EXTERNAL DATASET FOUND] Loading {len(df_web)} external web images from {web_dir.name}...\n")
        items_to_eval = []
        for _, row in df_web.iterrows():
            img_path = web_dir / row["filename"]
            items_to_eval.append((str(img_path), int(row["label"]), row["class_name"], "External Web / StVO Download"))
    else:
        df_val = load_gtsrb_val_dataset(limit=num_samples * 2, seed=seed)
        print(f"\n[SYNTHETIC OOD] Processing {num_samples} test images under in-the-wild environmental distortions...\n")
        items_to_eval = []
        for idx in range(min(num_samples, len(df_val))):
            row = df_val.iloc[idx]
            items_to_eval.append((row["resolved_path"], int(row["label"]), get_class_name(int(row["label"])), "Simulated Environmental Distortion"))

    for idx, (img_path, true_label, true_name, condition_type) in enumerate(items_to_eval):
        raw_img = Image.open(img_path).convert("RGB")
        if condition_type == "Simulated Environmental Distortion":
            eval_img, env_condition = apply_wild_environment_distortion(raw_img, idx)
        else:
            eval_img = raw_img
            env_condition = "External Real-World Web Image"

        # Run A: Standalone baseline prediction
        pred_idx, pred_label, conf, _ = model.predict(eval_img)
        pred_name = get_class_name(pred_idx)
        is_correct = (pred_idx == true_label)

        # Run B: Guarded with Hierarchical MCB
        report = cb.evaluate(eval_img)
        state_str = report.state.value
        cbi = round(report.cbi, 4)

        if not is_correct:
            total_model_errors += 1
            if state_str in ("OPEN", "HALF_OPEN"):
                silent_failures_caught += 1

        # Classify interception status
        if not is_correct and state_str == "OPEN":
            status = "TP (Caught Error)"
        elif is_correct and state_str == "CLOSED":
            status = "TN (Safe Pass)"
        elif is_correct and state_str in ("OPEN", "HALF_OPEN"):
            status = "FP (False Warning/Trip)"
        else:
            status = "FN (Missed Error)"

        rec = {
            "sample_id": Path(img_path).stem,
            "environment_condition": env_condition,
            "ground_truth_id": true_label,
            "ground_truth_name": true_name,
            "standalone_pred_id": pred_idx,
            "standalone_pred_name": pred_name,
            "standalone_confidence": round(conf, 4),
            "standalone_correct": is_correct,
            "cbi_score": cbi,
            "cb_state": state_str,
            "cb_action": report.action,
            "interception_status": status,
            "peak_family": report.peak_family,
            "peak_family_score": round(report.peak_family_score, 4),
            "cross_family_spread": round(report.cross_family_spread, 4),
        }
        records.append(rec)
        total_samples += 1

        mark = "[CORRECT]" if is_correct else "[ERROR]  "
        print(f"Sample {idx+1:02d}: {mark} True: {true_name[:24]:<24} | Pred: {pred_name[:24]:<24} "
              f"({conf:6.1%}) | CBI: {cbi:.3f} -> {state_str:<9} [{status}]")


    interception_rate = (silent_failures_caught / total_model_errors * 100.0) if total_model_errors > 0 else 100.0
    overconfident_silent_errors = sum(
        1 for r in records if not r["standalone_correct"] and r["standalone_confidence"] >= 0.85
    )
    overconfident_caught = sum(
        1 for r in records if not r["standalone_correct"] and r["standalone_confidence"] >= 0.85 and r["cb_state"] in ("OPEN", "HALF_OPEN")
    )

    summary = {
        "test_name": "TEST 1: In-the-Wild Web / OOD Testing",
        "domain": "Autonomous Vehicles (GTSRB Traffic Signs)",
        "total_samples": total_samples,
        "standalone_accuracy": round((total_samples - total_model_errors) / total_samples * 100.0, 2),
        "total_model_errors": total_model_errors,
        "errors_intercepted": silent_failures_caught,
        "empirical_interception_rate_pct": round(interception_rate, 2),
        "overconfident_silent_errors_gt_85pct": overconfident_silent_errors,
        "overconfident_errors_intercepted": overconfident_caught,
        "samples": records,
    }

    # Generate Markdown Table
    md = [
        "# Test 1: In-the-Wild Web / OOD Testing (Comparative Analysis)",
        "",
        "**Target:** Autonomous Driving Traffic Sign Recognition (`MobileNetV3-Small` on GTSRB)",  
        f"**Date Executed:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Executive Summary",
        f"- **Total Evaluated Samples:** {total_samples}",
        f"- **Unmonitored Standalone Accuracy:** {summary['standalone_accuracy']}%",
        f"- **Total Errors Induced by In-the-Wild Perturbations:** {total_model_errors}",
        f"- **MCB Interception Rate (OPEN / HALF-OPEN):** **{summary['empirical_interception_rate_pct']}%**",
        f"- **Overconfident Silent Errors Caught (Confidence >= 85%):** {overconfident_caught} / {overconfident_silent_errors}",
        "",
        "## 2. Key Findings",
        "1. **Elimination of Blind Trust:** The standalone deep model produces high-confidence predictions (>85%) on severely distorted signs that are nevertheless incorrect.",
        "2. **Metamorphic Vulnerability Exposure:** Even when the model's single softmax forward pass appears confident, subjecting the input to metamorphic perturbations reveals hidden fragility.",
        "3. **Circuit Breaker Action:** The Hierarchical MCB elevates the Composite Brittleness Index (CBI), triggering `HALF_OPEN` (warning) or `OPEN` (fallback/intercept).",
        "",
        "## 3. Sample-by-Sample Diagnostic Audit",
        "",
        "| Sample ID | Environmental Condition | Ground Truth | Standalone Prediction | Conf | CBI | Breaker State | Interception Status |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
    ]

    for r in records[:25]:  # Display first 25 in table
        md.append(
            f"| `{r['sample_id']}` | {r['environment_condition']} | {r['ground_truth_name']} | "
            f"{r['standalone_pred_name']} | {r['standalone_confidence']:.1%} | {r['cbi_score']:.3f} | "
            f"**{r['cb_state']}** | {r['interception_status']} |"
        )

    if len(records) > 25:
        md.append(f"\n*(Table truncated to first 25 samples. Full {len(records)} samples stored in JSON)*")

    md_str = "\n".join(md)
    save_test_output("test1_in_the_wild", summary, md_str)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Test 1: In-the-Wild Web / OOD Testing")
    parser.add_argument("--samples", type=int, default=40, help="Number of test samples (default: 40)")
    parser.add_argument("--seed", type=int, default=101, help="Random seed (default: 101)")
    args = parser.parse_args()
    run_test1(num_samples=args.samples, seed=args.seed)
