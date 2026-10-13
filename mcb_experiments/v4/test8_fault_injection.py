"""
mcb_experiments/v4/test8_fault_injection.py — Step 9: Fault Injection & Attribution Benchmark.

Protocol Requirements (Section 8):
- Sources: 400 disjoint correctly classified images from Partition B (100 per family + 100 clean control).
- Unseen corruptions not in transform bank.
- Saves fault_injection_config.yaml.
- Attribution rule (Section 8.2):
  Normalized by clean correct samples of Partition A:
  z_f = (IR_f - mean_f) / std_f
  Predicted family = argmax_f z_f (or 'unattributable' if all IR_f == 0).
- Computes raw argmax IR_f as well.
- Computes:
  - Trip rate per injected family & clean control
  - 3x3 confusion matrix + unattributable column
  - Attribution accuracy overall and per family (Wilson 95% CIs, binomial test vs chance 1/3)
  - Family-specific lift vs clean control
  - Cross-contamination fraction
  - Decision rule D4 (Wilson lower bound > 0.50)
- Saves fault_attribution_confusion.png and fault_injection_config.yaml.
"""

import json
import math
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import binomtest
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "mcb_experiments" / "results_v4"
CSV_PATH = RESULTS_DIR / "per_sample_scores.csv"
CALIB_PATH = RESULTS_DIR / "calibrated_thresholds.json"
VARIANTS_PATH = PROJECT_ROOT / "variants.json"

from src.models.mobilenet_adapter import MobileNetAdapter
from src.utils import BEST_MODEL_PATH, IMAGENET_MEAN, IMAGENET_STD
from src.metamorphic_families import HIERARCHICAL_TEST_MATRIX, SEVERITY_WEIGHTS, FAMILIES

# Unseen Corruptions (Section 8.1)
def motion_blur(img: np.ndarray, k: int = 7, ang: float = 0.0) -> np.ndarray:
    ker = np.zeros((k, k))
    ker[k // 2, :] = 1
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1)
    ker = cv2.warpAffine(ker, M, (k, k))
    ker /= (ker.sum() + 1e-8)
    return cv2.filter2D(img, -1, ker)

def defocus_blur(img: np.ndarray, r: int = 2) -> np.ndarray:
    ker = np.zeros((2 * r + 1, 2 * r + 1))
    cv2.circle(ker, (r, r), r, 1, -1)
    ker /= (ker.sum() + 1e-8)
    return cv2.filter2D(img, -1, ker)

def shot_noise(img: np.ndarray, scale: float = 30.0, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    return np.clip(rng.poisson(img / 255.0 * scale) / scale * 255.0, 0, 255).astype(np.uint8)

def salt_pepper(img: np.ndarray, p: float = 0.03, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    m = rng.random(img.shape[:2])
    o = img.copy()
    o[m < p / 2.0] = 0
    o[m > 1.0 - p / 2.0] = 255
    return o

def gamma(img: np.ndarray, g: float = 1.6) -> np.ndarray:
    return np.clip(255.0 * (img / 255.0) ** g, 0, 255).astype(np.uint8)

def vignette(img: np.ndarray, s: float = 0.5) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    d = np.sqrt(((x - w / 2.0) / (w / 2.0)) ** 2 + ((y - h / 2.0) / (h / 2.0)) ** 2)
    return np.clip(img * (1.0 - s * np.clip(d, 0, 1) ** 2)[..., None], 0, 255).astype(np.uint8)

def white_balance(img: np.ndarray, r: float = 1.15, b: float = 0.85) -> np.ndarray:
    o = img.astype(float)
    o[..., 0] *= r
    o[..., 2] *= b
    return np.clip(o, 0, 255).astype(np.uint8)

def translate(img: np.ndarray, f: float = 0.08) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, f * w], [0, 1, f * h]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def perspective(img: np.ndarray, f: float = 0.05) -> np.ndarray:
    h, w = img.shape[:2]
    s = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = s + np.float32([[f * w, f * h], [-f * w, f * h], [-f * w, -f * h], [f * w, -f * h]]) * 0.5
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(s, d), (w, h), borderMode=cv2.BORDER_REFLECT)

def crop_resize(img: np.ndarray, f: float = 0.08) -> np.ndarray:
    h, w = img.shape[:2]
    c = img[int(f * h):h - int(f * h), int(f * w):w - int(f * w)]
    return cv2.resize(c, (w, h))

def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))

def evaluate_mcb_on_image(img_corrupted: Image.Image, model, to_tensor, sample_id: str, variants_list):
    """Runs model on corrupted image + 40 bank transforms, returns ir dict, c0, cbi, flipped."""
    seed = zlib.crc32(sample_id.encode())
    t_imgs = [test(img_corrupted, seed=seed) for test in HIERARCHICAL_TEST_MATRIX]
    batch = torch.stack([to_tensor(img_corrupted)] + [to_tensor(t) for t in t_imgs])
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1).cpu().numpy()
        probs_np = probs.cpu().numpy()

    base_pred = int(preds[0])
    c0 = float(probs_np[0, base_pred])
    v_preds = preds[1:]
    flips = (v_preds != base_pred).astype(int)

    # Level 1 and Level 2
    groups = {}
    for j, v in enumerate(variants_list):
        groups.setdefault((v["family"], v["type"]), {}).setdefault(v["tier"], []).append(j)
    scores = {}
    for key, tiers in groups.items():
        num = sum(SEVERITY_WEIGHTS[t] * (flips[idx].mean()) for t, idx in tiers.items())
        denom = sum(SEVERITY_WEIGHTS[t] for t in tiers)
        scores[key] = num / denom

    ir = {}
    for f in FAMILIES:
        ms = [scores[key] for key in scores if key[0] == f]
        ir[f] = float(np.mean(ms))

    return ir, c0, base_pred

def run():
    print("=" * 70)
    print("STEP 9: FAULT INJECTION & ATTRIBUTION BENCHMARK (Section 8)")
    print("=" * 70)

    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib = json.load(f)
    tau_fam_star = calib["mcb_op_j"]["tau_fam"]
    theta_trip_star = calib["mcb_op_j"]["theta_trip"]

    with open(VARIANTS_PATH, "r", encoding="utf-8") as f:
        variants = json.load(f)

    df_all = pd.read_csv(CSV_PATH)
    adapter = MobileNetAdapter(checkpoint_path=str(BEST_MODEL_PATH))
    model = adapter.model
    model.eval()
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    # 1. Normalization Baseline from Clean Correct Samples of Partition A
    df_A_corr = df_all[(df_all.partition == "A") & (df_all.correct == 1)]
    mean_A = {f: float(df_A_corr[f"ir_{f}"].mean()) for f in FAMILIES}
    std_A = {f: float(df_A_corr[f"ir_{f}"].std()) + 1e-6 for f in FAMILIES}
    print(f"Partition A Clean Correct Normalizers: mean = {mean_A}, std = {std_A}")

    # 2. Sample 400 Disjoint Correctly Classified Partition B Images
    df_B_corr = df_all[(df_all.partition == "B") & (df_all.correct == 1)].sample(n=400, random_state=42).reset_index(drop=True)
    control_samples = df_B_corr.iloc[0:100]
    sensor_samples = df_B_corr.iloc[100:200]
    photo_samples = df_B_corr.iloc[200:300]
    geom_samples = df_B_corr.iloc[300:400]

    # Save fault_injection_config.yaml
    config_yaml = f"""# Fault Injection Configuration (Section 8.1)
version: 4.0
target_domain: GTSRB
sample_count_per_family: 100
control_sample_count: 100
families:
  sensor_noise:
    corruptions:
      motion_blur: {{k: 7, ang: 0.0}}
      defocus_blur: {{r: 2}}
      shot_noise: {{scale: 30.0}}
      salt_pepper: {{p: 0.03}}
  photometric:
    corruptions:
      gamma_dark: {{g: 0.6}}
      gamma_bright: {{g: 1.6}}
      vignette: {{s: 0.5}}
      white_balance: {{r: 1.15, b: 0.85}}
  geometric:
    corruptions:
      translate: {{f: 0.08}}
      perspective: {{f: 0.05}}
      crop_resize: {{f: 0.08}}
frozen_thresholds:
  tau_fam: {tau_fam_star}
  theta_trip: {theta_trip_star}
"""
    config_path = RESULTS_DIR / "fault_injection_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_yaml)
    print(f"[Saved] {config_path}")

    # Load image path mapping from checkpoints/val_split.csv
    val_df = pd.read_csv(PROJECT_ROOT / "checkpoints" / "val_split.csv")
    id_to_path = dict(zip(val_df["image"].astype(str), val_df["image_path"]))

    # Helper to resolve image paths
    def load_img_224(sample_id):
        p = id_to_path[str(sample_id)]
        pth = Path(p)
        if not pth.is_absolute():
            cand = PROJECT_ROOT / p
            if cand.exists(): pth = cand
            else: pth = PROJECT_ROOT / "data" / "gtsrb" / p
        return Image.open(pth).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)

    # 3. Evaluate Clean Control (100 samples)
    print("\nEvaluating Clean Control Pool (100 samples)...")
    control_irs = {f: [] for f in FAMILIES}
    control_tripped = 0
    for _, row in control_samples.iterrows():
        sid = str(row["sample_id"])
        img_224 = load_img_224(sid)
        ir, c0, pred = evaluate_mcb_on_image(img_224, model, to_tensor, sid, variants)
        for f in FAMILIES: control_irs[f].append(ir[f])
        cfs = sum(1 for f in FAMILIES if ir[f] >= tau_fam_star) / len(FAMILIES)
        cbi = 0.50 * max(ir.values()) + 0.35 * cfs + 0.15 * (1.0 - c0)
        if cbi >= theta_trip_star: control_tripped += 1

    control_mean_ir = {f: float(np.mean(control_irs[f])) for f in FAMILIES}
    control_trip_rate = control_tripped / 100.0
    print(f"Clean Control Trip Rate: {control_trip_rate:.4f} | Mean IR: {control_mean_ir}")

    # 4. Evaluate Injected Samples (300 total)
    injected_tasks = [
        ("sensor_noise", sensor_samples, [
            ("motion_blur", lambda a: motion_blur(a, k=7, ang=0)),
            ("defocus_blur", lambda a: defocus_blur(a, r=2)),
            ("shot_noise", lambda a: shot_noise(a, scale=30)),
            ("salt_pepper", lambda a: salt_pepper(a, p=0.03)),
        ]),
        ("photometric", photo_samples, [
            ("gamma_dark", lambda a: gamma(a, g=0.6)),
            ("gamma_bright", lambda a: gamma(a, g=1.6)),
            ("vignette", lambda a: vignette(a, s=0.5)),
            ("white_balance", lambda a: white_balance(a, r=1.15, b=0.85)),
        ]),
        ("geometric", geom_samples, [
            ("translate", lambda a: translate(a, f=0.08)),
            ("perspective", lambda a: perspective(a, f=0.05)),
            ("crop_resize", lambda a: crop_resize(a, f=0.08)),
            ("crop_resize_2", lambda a: crop_resize(a, f=0.06)),
        ]),
    ]

    all_injected_records = []
    # Confusion matrix: true_family -> {predicted_family: count} + unattributable
    conf_matrix = {
        true_f: {pred_f: 0 for pred_f in FAMILIES + ["unattributable"]}
        for true_f in FAMILIES
    }
    trip_counts = {f: 0 for f in FAMILIES}
    base_flip_counts = {f: 0 for f in FAMILIES}
    injected_irs = {true_f: {f: [] for f in FAMILIES} for true_f in FAMILIES}
    cross_contam_counts = {f: 0 for f in FAMILIES}

    for fam_name, df_fam, corr_fns in injected_tasks:
        print(f"Injecting family '{fam_name}' across 100 samples...")
        for idx, (_, row) in enumerate(df_fam.iterrows()):
            c_name, c_fn = corr_fns[idx % len(corr_fns)]
            sid = str(row["sample_id"])
            img_clean = load_img_224(sid)
            arr_clean = np.array(img_clean)
            arr_corrupted = c_fn(arr_clean)
            img_corrupted = Image.fromarray(arr_corrupted)

            # Evaluate with model + bank
            ir, c0, pred_corr = evaluate_mcb_on_image(img_corrupted, model, to_tensor, sid, variants)
            orig_label = int(row["y_true"])
            base_flipped = (pred_corr != orig_label)
            if base_flipped: base_flip_counts[fam_name] += 1

            for f in FAMILIES: injected_irs[fam_name][f].append(ir[f])

            # Check trip state
            cfs = sum(1 for f in FAMILIES if ir[f] >= tau_fam_star) / len(FAMILIES)
            cbi = 0.50 * max(ir.values()) + 0.35 * cfs + 0.15 * (1.0 - c0)
            is_tripped = (cbi >= theta_trip_star)
            if is_tripped: trip_counts[fam_name] += 1

            # Check cross-contamination (wrong family IR >= tau_fam_star)
            wrong_fams = [f for f in FAMILIES if f != fam_name]
            if any(ir[wf] >= tau_fam_star for wf in wrong_fams):
                cross_contam_counts[fam_name] += 1

            # Attribution Rule: z_f = (IR_f - mean_f) / std_f
            z_scores = {f: (ir[f] - mean_A[f]) / std_A[f] for f in FAMILIES}
            if all(ir[f] == 0.0 for f in FAMILIES):
                pred_fam = "unattributable"
            else:
                pred_fam = max(z_scores, key=lambda f: z_scores[f])

            conf_matrix[fam_name][pred_fam] += 1

            all_injected_records.append({
                "true_family": fam_name,
                "corruption": c_name,
                "base_flipped": base_flipped,
                "tripped": is_tripped,
                "predicted_family": pred_fam,
                "ir": ir,
                "z_scores": z_scores,
            })

    # 5. Metrics Compilation
    print("\n" + "=" * 70)
    print("FAULT INJECTION ATTRIBUTION RESULTS (300 Injected + 100 Control)")
    print("=" * 70)

    # Trip rates and Base model flip rates
    print("Base Model Vulnerability & Circuit Breaker Trip Rates:")
    for f in FAMILIES:
        print(f"  - {f:<13}: Base Flip Rate = {base_flip_counts[f]}/100 ({base_flip_counts[f]}%) | Trip Rate = {trip_counts[f]}/100 ({trip_counts[f]}%)")
    print(f"  - Clean Control: Trip Rate = {control_tripped}/100 ({control_trip_rate*100:.1f}%)")

    # Confusion matrix
    print("\nFault Attribution Confusion Matrix (Rows: Injected Ground Truth, Cols: Attributed):")
    header = f"{'True Family':<14} | " + " | ".join([f"{col:<13}" for col in FAMILIES + ['unattributable']])
    print(header)
    print("-" * len(header))
    for true_f in FAMILIES:
        row_str = f"{true_f:<14} | " + " | ".join([f"{conf_matrix[true_f][p]:<13}" for p in FAMILIES + ['unattributable']])
        print(row_str)

    # Accuracy overall (All 300 injected)
    total_injected = 300
    total_correct_all = sum(conf_matrix[f][f] for f in FAMILIES)
    acc_all = total_correct_all / total_injected
    ci_acc_all = wilson(total_correct_all, total_injected)

    # Binomial test vs chance (1/3 = 0.3333)
    binom_res = binomtest(total_correct_all, total_injected, p=1.0 / 3.0, alternative="greater")
    binom_p = float(binom_res.pvalue)

    # Accuracy on Tripped samples only
    tripped_records = [r for r in all_injected_records if r["tripped"]]
    n_tripped = len(tripped_records)
    n_correct_tripped = sum(1 for r in tripped_records if r["predicted_family"] == r["true_family"])
    acc_tripped = (n_correct_tripped / n_tripped) if n_tripped else 0.0
    ci_acc_tripped = wilson(n_correct_tripped, n_tripped) if n_tripped else (0.0, 0.0)

    # Family-specific accuracy
    fam_accs = {}
    for f in FAMILIES:
        cor_f = conf_matrix[f][f]
        fam_accs[f] = {
            "accuracy": cor_f / 100.0,
            "ci95": list(wilson(cor_f, 100)),
        }

    # Family-specific lift & cross-contamination
    lift_stats = {}
    for f in FAMILIES:
        mean_injected_f = float(np.mean(injected_irs[f][f]))
        mean_control_f = control_mean_ir[f]
        lift = mean_injected_f - mean_control_f
        cross_contam_rate = cross_contam_counts[f] / 100.0
        lift_stats[f] = {
            "mean_ir_injected": mean_injected_f,
            "mean_ir_control": mean_control_f,
            "lift": lift,
            "cross_contamination_rate": cross_contam_rate,
        }
        print(f"\n{f} Injected:")
        print(f"  - Attribution Accuracy: {fam_accs[f]['accuracy']*100:.1f}% (95% CI: {fam_accs[f]['ci95']})")
        print(f"  - Mean IR (Injected vs Control): {mean_injected_f:.4f} vs {mean_control_f:.4f} (Lift = {lift:+.4f})")
        print(f"  - Cross-Contamination Rate: {cross_contam_rate*100:.1f}%")

    print("\nOverall Attribution Performance:")
    print(f"  - All Injected Samples (N=300): Accuracy = {acc_all*100:.1f}% (95% CI: [{ci_acc_all[0]*100:.1f}%, {ci_acc_all[1]*100:.1f}%]) | Binomial p-value vs Chance (33.3%): {binom_p:.2e}")
    print(f"  - Tripped Samples Only (N={n_tripped}): Accuracy = {acc_tripped*100:.1f}% (95% CI: [{ci_acc_tripped[0]*100:.1f}%, {ci_acc_tripped[1]*100:.1f}%])")

    # DECISION RULE D4
    d4_met = ci_acc_all[0] > 0.50
    print(f"\nDecision Rule D4 (Wilson Lower Bound > 0.50): {ci_acc_all[0]:.4f} > 0.50 --> {'MET' if d4_met else 'NOT MET'}")

    # Plot Confusion Matrix Heatmap
    matrix_data = np.array([[conf_matrix[true_f][pred_f] for pred_f in FAMILIES + ['unattributable']] for true_f in FAMILIES])
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix_data, cmap="Blues", aspect="auto")
    plt.colorbar(label="Sample Count")
    for r in range(3):
        for c in range(4):
            plt.text(c, r, str(matrix_data[r, c]), ha="center", va="center", color="black" if matrix_data[r, c] < 50 else "white", fontsize=12)
    plt.xticks(ticks=range(4), labels=["Geometric", "Photometric", "Sensor Noise", "Unattributable"], rotation=15)
    plt.yticks(ticks=range(3), labels=["Geometric", "Photometric", "Sensor Noise"])
    plt.xlabel("Predicted Failure Family")
    plt.ylabel("Injected Ground Truth Family")
    plt.title("Fault Attribution Confusion Matrix (Unseen Perturbations)")
    plt.tight_layout()
    cm_png = RESULTS_DIR / "fault_attribution_confusion.png"
    plt.savefig(cm_png, dpi=300)
    plt.close()
    print(f"\n[Saved] {cm_png}")

    # Save results json
    fault_summary = {
        "n_per_family": 100,
        "n_control": 100,
        "trip_rates": {f: trip_counts[f] / 100.0 for f in FAMILIES},
        "control_trip_rate": control_trip_rate,
        "confusion_matrix": conf_matrix,
        "accuracy_all": acc_all,
        "accuracy_ci95": list(ci_acc_all),
        "accuracy_tripped_only": acc_tripped,
        "accuracy_tripped_ci95": list(ci_acc_tripped),
        "binomial_p_vs_chance": binom_p,
        "decision_rule_D4": "MET" if d4_met else "NOT MET",
        "family_accuracies": fam_accs,
        "family_lift": lift_stats,
    }
    with open(RESULTS_DIR / "fault_attribution_results.json", "w", encoding="utf-8") as f:
        json.dump(fault_summary, f, indent=2)

    print(f"[Saved] {RESULTS_DIR / 'fault_attribution_results.json'}")
    print("=" * 70)

if __name__ == "__main__":
    run()
