"""
mcb_experiments/v4/step6_calibrate.py — Step 6: Calibration on Partition A Only.

Protocol Requirements (Section 6.1 & 6.2):
- Calibrate strictly on Partition A only. Never touch Partition B.
- Grid:
  TAU = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
  THETA = [round(x, 2) for x in np.arange(0.10, 0.601, 0.05)]
- Operating points:
  op_j: max Youden J (recall - FAR), tie-break: lower FAR
  op_far10: highest recall subject to FAR <= 0.10, tie-break: lower FAR
- Prior default: tau_fam=0.35, theta_trip=0.55.
- Baselines thresholds on A: softmax_entropy, msp, tta_var, ff, mcdo, entropy_ts.
- Boundary check: if optimum on boundary, extend edge once by one step and log in deviations.md.
- Outputs:
  - mcb_experiments/results_v4/grid_search_A.csv
  - mcb_experiments/results_v4/calibration_heatmap.png
  - mcb_experiments/results_v4/calibrated_thresholds.json
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "mcb_experiments" / "results_v4"
CSV_PATH = RESULTS_DIR / "per_sample_scores.csv"

TAU = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]  # Extended once by one step at lower boundary per Section 6.2
THETA = [round(x, 2) for x in np.arange(0.10, 0.601, 0.05)]
FAMS = ["geometric", "photometric", "sensor_noise"]

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))

def confusion(err, trip):
    tp = int((err & trip).sum())
    fn = int((err & ~trip).sum())
    fp = int((~err & trip).sum())
    tn = int((~err & ~trip).sum())
    return tp, fp, tn, fn

def rates(err, trip):
    tp, fp, tn, fn = confusion(err, trip)
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    return dict(
        tp=tp, fp=fp, tn=tn, fn=fn,
        recall=rec, far=far, youden=rec - far,
        recall_ci=wilson(tp, tp + fn), far_ci=wilson(fp, fp + tn),
        precision=tp / (tp + fp) if (tp + fp) else 0.0
    )

def pick_threshold(score, err, rule, grid=None):
    cands = np.unique(score) if grid is None else grid
    best, best_key = None, None
    for t in cands:
        r = rates(err, score >= t)
        if rule == "op_j":
            key = (r["youden"], -r["far"])
        else:
            if r["far"] > 0.10: continue
            key = (r["recall"], -r["far"])
        if best_key is None or key > best_key:
            best, best_key = t, key
    return float(best) if best is not None else float(cands[-1])

def run():
    print("=" * 70)
    print("STEP 6: CALIBRATION ON PARTITION A ONLY (Section 6.2)")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)
    df_A = df[df.partition == "A"].copy()
    err_A = (df_A.correct == 0).to_numpy()
    c0_A = df_A.c0.to_numpy()

    ir_A = {f: df_A[f"ir_{f}"].to_numpy() for f in FAMS}
    M_A = np.stack([ir_A[f] for f in FAMS], axis=1)

    # 1. MCB Grid Sweep on Partition A
    grid_records = []
    youden_grid = np.zeros((len(TAU), len(THETA)))

    for i, tau in enumerate(TAU):
        cfs = (M_A >= tau).mean(axis=1)
        cbi_vals = 0.50 * M_A.max(axis=1) + 0.35 * cfs + 0.15 * (1.0 - c0_A)
        for j, theta in enumerate(THETA):
            # Trip state is OPEN: cbi >= theta
            trip = (cbi_vals >= theta)
            r = rates(err_A, trip)
            grid_records.append({
                "tau_fam": tau,
                "theta_trip": theta,
                "theta_warn": round(0.5 * theta, 4),
                "tp": r["tp"], "fp": r["fp"], "tn": r["tn"], "fn": r["fn"],
                "recall": r["recall"], "far": r["far"], "youden_j": r["youden"],
                "precision": r["precision"],
            })
            youden_grid[i, j] = r["youden"]

    df_grid = pd.DataFrame(grid_records)
    grid_csv = RESULTS_DIR / "grid_search_A.csv"
    df_grid.to_csv(grid_csv, index=False)
    print(f"[Saved] {grid_csv} ({len(df_grid)} configurations)")

    # Plot calibration heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(youden_grid, cmap="viridis", aspect="auto", origin="lower")
    plt.colorbar(label="Youden J Index")
    plt.xticks(ticks=np.arange(len(THETA)), labels=[f"{t:.2f}" for t in THETA], rotation=45)
    plt.yticks(ticks=np.arange(len(TAU)), labels=[f"{t:.2f}" for t in TAU])
    plt.xlabel("Trip Threshold (theta_trip)")
    plt.ylabel("Family Compromise Threshold (tau_fam)")
    plt.title("Partition A MCB Calibration Heatmap (Youden J)")
    plt.tight_layout()
    heatmap_png = RESULTS_DIR / "calibration_heatmap.png"
    plt.savefig(heatmap_png, dpi=300)
    plt.close()
    print(f"[Saved] {heatmap_png}")

    # Pick MCB op_j: max Youden J
    best_j_row = df_grid.sort_values(by=["youden_j", "far"], ascending=[False, True]).iloc[0]
    mcb_op_j_tau = float(best_j_row["tau_fam"])
    mcb_op_j_theta = float(best_j_row["theta_trip"])

    # Pick MCB op_far10: max recall with far <= 0.10
    far10_cands = df_grid[df_grid.far <= 0.10]
    if len(far10_cands) > 0:
        best_far10_row = far10_cands.sort_values(by=["recall", "far"], ascending=[False, True]).iloc[0]
        mcb_op_far10_tau = float(best_far10_row["tau_fam"])
        mcb_op_far10_theta = float(best_far10_row["theta_trip"])
    else:
        # Lowest FAR overall if none <= 0.10
        best_far10_row = df_grid.sort_values(by="far").iloc[0]
        mcb_op_far10_tau = float(best_far10_row["tau_fam"])
        mcb_op_far10_theta = float(best_far10_row["theta_trip"])

    print(f"\nMCB Operating Points on Partition A:")
    print(f"  - op_j     : tau_fam = {mcb_op_j_tau:.2f}, theta_trip = {mcb_op_j_theta:.2f} (J = {best_j_row['youden_j']:.4f}, Recall = {best_j_row['recall']:.4f}, FAR = {best_j_row['far']:.4f})")
    print(f"  - op_far10 : tau_fam = {mcb_op_far10_tau:.2f}, theta_trip = {mcb_op_far10_theta:.2f} (Recall = {best_far10_row['recall']:.4f}, FAR = {best_far10_row['far']:.4f})")

    # Boundary check
    on_boundary = (
        mcb_op_j_tau in (TAU[0], TAU[-1]) or mcb_op_j_theta in (THETA[0], THETA[-1]) or
        mcb_op_far10_tau in (TAU[0], TAU[-1]) or mcb_op_far10_theta in (THETA[0], THETA[-1])
    )
    print(f"  - Boundary Check: Grid optimum on boundary = {on_boundary}")

    # 2. Baseline Thresholds Calibration on Partition A
    baseline_thrs = {}
    baseline_cols = {
        "softmax_entropy": "entropy",
        "msp": "msp_score",
        "tta_var": "tta_var",
        "ff": "ff",
        "mcdo": "mcdo_score",
        "entropy_ts": "entropy_ts",
    }

    print("\nCalibrating Baseline Continuous Scores on Partition A:")
    for b_name, col in baseline_cols.items():
        score_A = df_A[col].to_numpy()
        grid_vals = np.linspace(score_A.min(), score_A.max(), 500)
        thr_j = pick_threshold(score_A, err_A, "op_j", grid_vals)
        thr_far10 = pick_threshold(score_A, err_A, "op_far10", grid_vals)
        baseline_thrs[b_name] = {"op_j": thr_j, "op_far10": thr_far10}
        r_j = rates(err_A, score_A >= thr_j)
        r_f10 = rates(err_A, score_A >= thr_far10)
        print(f"  - {b_name:<16}: op_j thr = {thr_j:.4f} (J = {r_j['youden']:.4f}) | op_far10 thr = {thr_far10:.4f} (Rec = {r_f10['recall']:.4f}, FAR = {r_f10['far']:.4f})")

    # Read fitted temperature T
    from scipy.optimize import minimize_scalar
    from scipy.special import log_softmax
    logits = np.load(RESULTS_DIR / "logits.npy")
    A_mask = (df.partition == "A").to_numpy()
    y_all = df.y_true.to_numpy()
    nll_fn = lambda T: -log_softmax(logits[A_mask] / T, axis=1)[np.arange(A_mask.sum()), y_all[A_mask]].mean()
    T_val = float(minimize_scalar(nll_fn, bounds=(0.05, 20.0), method="bounded").x)

    calib_json = {
        "mcb_op_j": {
            "tau_fam": mcb_op_j_tau,
            "theta_trip": mcb_op_j_theta,
            "theta_warn": round(0.5 * mcb_op_j_theta, 4),
        },
        "mcb_op_far10": {
            "tau_fam": mcb_op_far10_tau,
            "theta_trip": mcb_op_far10_theta,
            "theta_warn": round(0.5 * mcb_op_far10_theta, 4),
        },
        "mcb_prior_default": {
            "tau_fam": 0.35,
            "theta_trip": 0.55,
            "theta_warn": round(0.5 * 0.55, 4),
        },
        "baselines": baseline_thrs,
        "temperature_T": round(T_val, 4),
        "grid_optimum_on_boundary": bool(on_boundary),
    }

    out_json_path = RESULTS_DIR / "calibrated_thresholds.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(calib_json, f, indent=2)
    print(f"\n[Saved] {out_json_path}")
    print("=" * 70)

if __name__ == "__main__":
    run()
