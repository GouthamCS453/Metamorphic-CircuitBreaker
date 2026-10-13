"""
mcb_experiments/v4/step8_evaluate_B.py — Step 8: Evaluate Partition B Exactly Once & Ablations A1-A5.

Protocol Requirements (Sections 6.3, 6.4, 7):
- Frozen thresholds from calibrated_thresholds.json.
- Evaluate Partition B ONLY.
- Methods: standalone, kflip_1, kflip_2, kflip_3, kflip_5, kflip_8, ff, softmax_entropy, msp, tta_var, mcdo_headonly, entropy_ts, mcb_prior_default, mcb_op_j, mcb_op_far10.
- Metrics: confusion matrix (tp, fp, tn, fn), recall, far (Wilson 95% CIs), precision, youden_j, AUROC (bootstrap 95% CI), AUPRC, critical_error_recall, 3-state counts (CLOSED/HALF_OPEN/OPEN).
- Paired group bootstrap CIs (B=2000) for comparisons vs best baseline (D1), cbi_flat (D2).
- Pre-registered decision rules D1-D3.
- Ablations A1-A5:
  A1: cbi_flat vs hierarchical CBI
  A2: msp alone vs gamma=0 variant (0.6, 0.4, 0.0) vs default
  A3: ascending weights [0.1, 0.2, 0.3] vs new [1.0, 0.6, 0.3]
  A4: 5 alpha/beta/gamma configs
  A5: CFS removed
- Family profile: mean ir_* all, errors, correct, dominant family.
- Generates:
  - roc_curves.png
  - risk_coverage.png
  - family_radar.png
  - b_evaluation_raw.json
"""

import json
import math
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "mcb_experiments" / "results_v4"
CSV_PATH = RESULTS_DIR / "per_sample_scores.csv"
CALIB_PATH = RESULTS_DIR / "calibrated_thresholds.json"
VARIANTS_PATH = PROJECT_ROOT / "variants.json"

FAMS = ["geometric", "photometric", "sensor_noise"]
W_NEW = {"mild": 1.0, "moderate": 0.6, "severe": 0.3, "single": 1.0}
W_OLD = {"mild": 0.1, "moderate": 0.2, "severe": 0.3, "single": 0.1}

with open(VARIANTS_PATH, "r", encoding="utf-8") as f:
    VARIANTS = json.load(f)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
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
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return dict(
        tp=tp, fp=fp, tn=tn, fn=fn,
        recall=rec, far=far, youden=rec - far,
        recall_ci=wilson(tp, tp + fn), far_ci=wilson(fp, fp + tn),
        precision=prec,
    )

def bootstrap_auc_ci(err, score, groups, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    aucs = []
    for _ in range(B):
        sample_g = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in sample_g])
        if err[idx].min() == err[idx].max(): continue
        aucs.append(roc_auc_score(err[idx], score[idx]))
    return float(np.mean(aucs)), (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))

def paired_auc_diff(err, s1, s2, groups, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    diffs = []
    for _ in range(B):
        sample_g = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in sample_g])
        if err[idx].min() == err[idx].max(): continue
        diffs.append(roc_auc_score(err[idx], s1[idx]) - roc_auc_score(err[idx], s2[idx]))
    return float(np.mean(diffs)), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))

def paired_recall_diff_at_far10(err, trip1, trip2, groups, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    diffs = []
    for _ in range(B):
        sample_g = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in sample_g])
        e = err[idx]
        if e.sum() == 0: continue
        r1 = (e & trip1[idx]).sum() / e.sum()
        r2 = (e & trip2[idx]).sum() / e.sum()
        diffs.append(r1 - r2)
    return float(np.mean(diffs)), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))

def compute_cbi(ir, c0, tau_fam, a=0.50, b=0.35, g=0.15):
    M = np.stack([ir[f] for f in FAMS], axis=1)
    cfs = (M >= tau_fam).mean(axis=1)
    return a * M.max(axis=1) + b * cfs + g * (1.0 - c0)

def compute_cbi_flat(flips, c0, a=0.50, b=0.35, g=0.15):
    w = np.array([W_NEW[v["tier"]] for v in VARIANTS])
    flat = (flips * w).sum(axis=1) / w.sum()
    return (a + b) * flat + g * (1.0 - c0)

def compute_ir_custom_weights(flips, weights_dict):
    groups = {}
    for j, v in enumerate(VARIANTS):
        groups.setdefault((v["family"], v["type"]), {}).setdefault(v["tier"], []).append(j)
    scores = {}
    for key, tiers in groups.items():
        num = sum(weights_dict[t] * flips[:, idx].mean(axis=1) for t, idx in tiers.items())
        denom = sum(weights_dict[t] for t in tiers)
        scores[key] = num / denom
    ir = {}
    for f in FAMS:
        ms = [scores[key] for key in scores if key[0] == f]
        ir[f] = np.mean(ms, axis=0)
    return ir

def run():
    print("=" * 70)
    print("STEP 8: EVALUATE PARTITION B ONCE & RUN ABLATIONS A1-A5")
    print("=" * 70)

    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib = json.load(f)

    df_all = pd.read_csv(CSV_PATH)
    df_B = df_all[df_all.partition == "B"].copy().reset_index(drop=True)
    N_B = len(df_B)
    err_B = (df_B.correct == 0).to_numpy()
    crit_B = (df_B.critical_error == 1).to_numpy()
    n_crit = crit_B.sum()
    groups_B = df_B.group_id.to_numpy()
    c0_B = df_B.c0.to_numpy()

    flips_B = df_B[[f"flip_{i:03d}" for i in range(len(VARIANTS))]].to_numpy()
    ir_B = {f: df_B[f"ir_{f}"].to_numpy() for f in FAMS}

    print(f"Partition B Samples: N = {N_B} | Base Errors: {err_B.sum()} | Critical Errors: {n_crit}")

    # Compute CBI on Partition B for each rule
    cbi_op_j = compute_cbi(ir_B, c0_B, tau_fam=calib["mcb_op_j"]["tau_fam"])
    cbi_op_far10 = compute_cbi(ir_B, c0_B, tau_fam=calib["mcb_op_far10"]["tau_fam"])
    cbi_prior = compute_cbi(ir_B, c0_B, tau_fam=0.35)

    # Dictionary to hold results
    b_results = {}

    # 1. Standalone Base Model (No Breaker)
    b_results["standalone"] = {
        "operating_point": "none",
        "tp": 0, "fp": 0, "tn": int((~err_B).sum()), "fn": int(err_B.sum()),
        "recall": 0.0, "recall_ci95": [0.0, 0.0],
        "far": 0.0, "far_ci95": [0.0, 0.0],
        "youden_j": 0.0, "precision": 0.0,
        "auroc": 0.50, "auroc_ci95": [0.50, 0.50],
        "critical_error_recall": 0.0,
        "state_closed": N_B, "state_half_open": 0, "state_open": 0,
    }

    # 2. k-of-n Flips (Binary rules)
    for k in (1, 2, 3, 5, 8):
        trip_k = (df_B[f"kflip_{k}"] == 1).to_numpy()
        r = rates(err_B, trip_k)
        crit_rec = float((crit_B & trip_k).sum() / n_crit) if n_crit else 0.0
        b_results[f"kflip_{k}"] = {
            "operating_point": f"k={k}",
            "tp": r["tp"], "fp": r["fp"], "tn": r["tn"], "fn": r["fn"],
            "recall": r["recall"], "recall_ci95": list(r["recall_ci"]),
            "far": r["far"], "far_ci95": list(r["far_ci"]),
            "youden_j": r["youden"], "precision": r["precision"],
            "auroc": None, "auroc_ci95": [None, None],
            "critical_error_recall": crit_rec,
            "state_closed": int((~trip_k).sum()), "state_half_open": 0, "state_open": int(trip_k.sum()),
        }

    # 3. Continuous Baselines
    continuous_baselines = {
        "softmax_entropy": df_B["entropy"].to_numpy(),
        "msp": df_B["msp_score"].to_numpy(),
        "tta_var": df_B["tta_var"].to_numpy(),
        "ff": df_B["ff"].to_numpy(),
        "mcdo_headonly": df_B["mcdo_score"].to_numpy(),
        "entropy_ts": df_B["entropy_ts"].to_numpy(),
    }

    baseline_trips = {}

    for b_name, score in continuous_baselines.items():
        key_calib = "mcdo" if b_name == "mcdo_headonly" else b_name
        thr_j = calib["baselines"][key_calib]["op_j"]
        thr_far10 = calib["baselines"][key_calib]["op_far10"]

        m_auc, ci_auc = bootstrap_auc_ci(err_B, score, groups_B)
        baseline_trips[b_name] = {"op_j": score >= thr_j, "op_far10": score >= thr_far10}

        # Save primary result at op_j
        trip_j = score >= thr_j
        r_j = rates(err_B, trip_j)
        crit_rec = float((crit_B & trip_j).sum() / n_crit) if n_crit else 0.0

        b_results[b_name] = {
            "operating_point": "op_j",
            "threshold_op_j": float(thr_j),
            "threshold_op_far10": float(thr_far10),
            "tp": r_j["tp"], "fp": r_j["fp"], "tn": r_j["tn"], "fn": r_j["fn"],
            "recall": r_j["recall"], "recall_ci95": list(r_j["recall_ci"]),
            "far": r_j["far"], "far_ci95": list(r_j["far_ci"]),
            "youden_j": r_j["youden"], "precision": r_j["precision"],
            "auroc": m_auc, "auroc_ci95": list(ci_auc),
            "critical_error_recall": crit_rec,
            "state_closed": int((~trip_j).sum()), "state_half_open": 0, "state_open": int(trip_j.sum()),
            "op_far10_metrics": rates(err_B, score >= thr_far10),
        }

    # 4. MCB Evaluated at op_j, op_far10, and prior_default
    mcb_configs = {
        "mcb_op_j": (cbi_op_j, calib["mcb_op_j"]["theta_trip"], calib["mcb_op_j"]["theta_warn"], "op_j"),
        "mcb_op_far10": (cbi_op_far10, calib["mcb_op_far10"]["theta_trip"], calib["mcb_op_far10"]["theta_warn"], "op_far10"),
        "mcb_prior_default": (cbi_prior, 0.55, 0.275, "prior_default"),
    }

    mcb_trips = {}

    for m_name, (cbi_val, th_trip, th_warn, op_name) in mcb_configs.items():
        is_open = (cbi_val >= th_trip)
        is_half = (cbi_val >= th_warn) & (cbi_val < th_trip)
        is_closed = (cbi_val < th_warn)
        trip = is_open  # Primary breaker trip is OPEN

        mcb_trips[m_name] = trip
        r = rates(err_B, trip)
        m_auc, ci_auc = bootstrap_auc_ci(err_B, cbi_val, groups_B)
        crit_rec = float((crit_B & trip).sum() / n_crit) if n_crit else 0.0

        b_results[m_name] = {
            "operating_point": op_name,
            "theta_trip": float(th_trip),
            "theta_warn": float(th_warn),
            "tp": r["tp"], "fp": r["fp"], "tn": r["tn"], "fn": r["fn"],
            "recall": r["recall"], "recall_ci95": list(r["recall_ci"]),
            "far": r["far"], "far_ci95": list(r["far_ci"]),
            "youden_j": r["youden"], "precision": r["precision"],
            "auroc": m_auc, "auroc_ci95": list(ci_auc),
            "critical_error_recall": crit_rec,
            "state_closed": int(is_closed.sum()),
            "state_half_open": int(is_half.sum()),
            "state_open": int(is_open.sum()),
        }

    # ─── DECISION RULES D1 - D3 EVALUATION ──────────────────────────────────
    print("\nEvaluating Pre-Registered Decision Rules D1 - D3...")

    # Find best baseline among entropy, msp, tta_var, mcdo, entropy_ts
    base_names = ["softmax_entropy", "msp", "tta_var", "mcdo_headonly", "entropy_ts"]
    best_base_name = max(base_names, key=lambda b: b_results[b]["auroc"])
    best_base_auc = b_results[best_base_name]["auroc"]
    print(f"Best continuous baseline on Partition B: {best_base_name} (AUROC = {best_base_auc:.4f})")

    # D1 Paired AUC difference
    diff_d1_mean, diff_d1_ci = paired_auc_diff(err_B, cbi_op_j, continuous_baselines[best_base_name], groups_B)
    d1_rec_diff_mean, d1_rec_diff_ci = paired_recall_diff_at_far10(
        err_B, mcb_trips["mcb_op_far10"], baseline_trips[best_base_name]["op_far10"], groups_B
    )
    d1_met = (diff_d1_ci[0] > 0) or (d1_rec_diff_ci[0] > 0)
    print(f"  - D1: MCB vs {best_base_name}: Paired AUC Diff = {diff_d1_mean:+.4f} {diff_d1_ci} | Rec@FAR10 Diff = {d1_rec_diff_mean:+.4f} {d1_rec_diff_ci} --> {'MET' if d1_met else 'NOT MET'}")

    # D2 Paired test vs cbi_flat (Ablation A1)
    cbi_flat_vals = compute_cbi_flat(flips_B, c0_B)
    diff_d2_mean, diff_d2_ci = paired_auc_diff(err_B, cbi_op_j, cbi_flat_vals, groups_B)
    # Calibrate flat on A for far10 threshold
    df_A = df_all[df_all.partition == "A"]
    flips_A = df_A[[f"flip_{i:03d}" for i in range(len(VARIANTS))]].to_numpy()
    cbi_flat_A = compute_cbi_flat(flips_A, df_A.c0.to_numpy())
    flat_far10_thr = 0.25
    for t_c in np.linspace(cbi_flat_A.min(), cbi_flat_A.max(), 500):
        if rates((df_A.correct == 0).to_numpy(), cbi_flat_A >= t_c)["far"] <= 0.10:
            flat_far10_thr = t_c
            break
    d2_rec_diff_mean, d2_rec_diff_ci = paired_recall_diff_at_far10(
        err_B, mcb_trips["mcb_op_far10"], cbi_flat_vals >= flat_far10_thr, groups_B
    )
    d2_met = (diff_d2_ci[0] > 0) or (d2_rec_diff_ci[0] > 0)
    print(f"  - D2: MCB vs cbi_flat: Paired AUC Diff = {diff_d2_mean:+.4f} {diff_d2_ci} | Rec@FAR10 Diff = {d2_rec_diff_mean:+.4f} {d2_rec_diff_ci} --> {'MET' if d2_met else 'NOT MET'}")

    # D3 High-confidence errors
    high_conf_mask = (err_B & (c0_B >= 0.90))
    if high_conf_mask.sum() < 15:
        # Top-quartile confidence errors
        err_confs = c0_B[err_B]
        q75 = np.percentile(err_confs, 75)
        high_conf_mask = (err_B & (c0_B >= q75))
    n_hc = high_conf_mask.sum()
    mcb_hc_rec = float(mcb_trips["mcb_op_far10"][high_conf_mask].mean()) if n_hc else 0.0
    ent_hc_rec = float(baseline_trips["softmax_entropy"]["op_far10"][high_conf_mask].mean()) if n_hc else 0.0
    mcb_hc_ci = wilson(int((mcb_trips["mcb_op_far10"] & high_conf_mask).sum()), n_hc)
    ent_hc_ci = wilson(int((baseline_trips["softmax_entropy"]["op_far10"] & high_conf_mask).sum()), n_hc)
    d3_met = (mcb_hc_rec > ent_hc_rec) and (mcb_hc_ci[0] > ent_hc_ci[1])
    print(f"  - D3: High-Confidence Errors (N = {n_hc}): MCB Recall = {mcb_hc_rec:.4f} {mcb_hc_ci} vs Entropy = {ent_hc_rec:.4f} {ent_hc_ci} --> {'MET' if d3_met else 'NOT MET'}")

    # ─── ABLATIONS A1 - A5 ──────────────────────────────────────────────────
    print("\nRunning Ablations A1 - A5 (Partition B)...")
    ablations = {}

    # A1: cbi_flat vs default
    auc_flat, ci_flat = bootstrap_auc_ci(err_B, cbi_flat_vals, groups_B)
    ablations["A1"] = {
        "description": "Flat weighted flip score vs hierarchical CBI",
        "auroc_flat": auc_flat, "ci_flat": list(ci_flat),
        "auroc_diff_vs_mcb": diff_d2_mean, "ci_diff": list(diff_d2_ci),
    }

    # A2: msp alone vs gamma=0 vs default
    cbi_gamma0 = compute_cbi(ir_B, c0_B, tau_fam=calib["mcb_op_j"]["tau_fam"], a=0.60, b=0.40, g=0.0)
    auc_gamma0, ci_gamma0 = bootstrap_auc_ci(err_B, cbi_gamma0, groups_B)
    diff_g0, ci_diff_g0 = paired_auc_diff(err_B, cbi_op_j, cbi_gamma0, groups_B)
    ablations["A2"] = {
        "description": "Baseline msp vs gamma=0 (0.6, 0.4, 0.0) vs default",
        "auroc_msp": b_results["msp"]["auroc"],
        "auroc_gamma0": auc_gamma0, "ci_gamma0": list(ci_gamma0),
        "auroc_diff_mcb_vs_gamma0": diff_g0, "ci_diff_gamma0": list(ci_diff_g0),
    }

    # A3: Ascending weights [0.1, 0.2, 0.3] vs descending [1.0, 0.6, 0.3]
    ir_B_old = compute_ir_custom_weights(flips_B, W_OLD)
    cbi_asc = compute_cbi(ir_B_old, c0_B, tau_fam=calib["mcb_op_j"]["tau_fam"])
    auc_asc, ci_asc = bootstrap_auc_ci(err_B, cbi_asc, groups_B)
    diff_asc, ci_diff_asc = paired_auc_diff(err_B, cbi_op_j, cbi_asc, groups_B)
    ablations["A3"] = {
        "description": "Ascending weights [0.1, 0.2, 0.3] vs descending [1.0, 0.6, 0.3]",
        "auroc_ascending": auc_asc, "ci_ascending": list(ci_asc),
        "auroc_diff_mcb_vs_ascending": diff_asc, "ci_diff_ascending": list(ci_diff_asc),
    }

    # A4: Five (alpha, beta, gamma) configs
    configs_a4 = [
        (0.50, 0.35, 0.15),
        (0.60, 0.40, 0.00),
        (0.70, 0.20, 0.10),
        (0.25, 0.25, 0.50),
        (0.30, 0.60, 0.10),
    ]
    a4_results = []
    for a, b, g in configs_a4:
        cbi_cfg = compute_cbi(ir_B, c0_B, tau_fam=calib["mcb_op_j"]["tau_fam"], a=a, b=b, g=g)
        auc_cfg, ci_cfg = bootstrap_auc_ci(err_B, cbi_cfg, groups_B)
        d_cfg, ci_d_cfg = paired_auc_diff(err_B, cbi_op_j, cbi_cfg, groups_B)
        a4_results.append({
            "config": f"alpha={a:.2f}, beta={b:.2f}, gamma={g:.2f}",
            "auroc": auc_cfg, "ci": list(ci_cfg),
            "diff_vs_default": d_cfg, "ci_diff": list(ci_d_cfg),
        })
    ablations["A4"] = {"description": "Five (alpha, beta, gamma) weight configs", "configs": a4_results}

    # A5: CFS removed
    M_B = np.stack([ir_B[f] for f in FAMS], axis=1)
    cbi_no_cfs = (0.50 + 0.35) * M_B.max(axis=1) + 0.15 * (1.0 - c0_B)
    auc_no_cfs, ci_no_cfs = bootstrap_auc_ci(err_B, cbi_no_cfs, groups_B)
    diff_cfs, ci_diff_cfs = paired_auc_diff(err_B, cbi_op_j, cbi_no_cfs, groups_B)
    ablations["A5"] = {
        "description": "CFS removed: (alpha+beta)*max(IR_k) + gamma*(1-c0)",
        "auroc_no_cfs": auc_no_cfs, "ci_no_cfs": list(ci_no_cfs),
        "auroc_diff_mcb_vs_no_cfs": diff_cfs, "ci_diff_no_cfs": list(ci_diff_cfs),
    }

    # ─── FAMILY PROFILE ──────────────────────────────────────────────────────
    mean_ir_all = {f: float(ir_B[f].mean()) for f in FAMS}
    mean_ir_err = {f: float(ir_B[f][err_B].mean()) for f in FAMS}
    mean_ir_corr = {f: float(ir_B[f][~err_B].mean()) for f in FAMS}
    dominant_family = max(mean_ir_err, key=lambda f: mean_ir_err[f])

    family_profile = {
        "mean_ir_all": mean_ir_all,
        "mean_ir_errors": mean_ir_err,
        "mean_ir_correct": mean_ir_corr,
        "dominant_axis": dominant_family,
    }

    # ─── PLOTS: ROC CURVES, RISK-COVERAGE, FAMILY RADAR ──────────────────────
    print("\nGenerating experimental figures...")

    # 1. ROC Curves
    plt.figure(figsize=(8, 8))
    for name, s in [
        ("MCB (op_j)", cbi_op_j),
        (f"Best Baseline ({best_base_name})", continuous_baselines[best_base_name]),
        ("Softmax Entropy", continuous_baselines["softmax_entropy"]),
        ("TTA Variance", continuous_baselines["tta_var"]),
        ("MC-Dropout", continuous_baselines["mcdo_headonly"]),
    ]:
        fpr, tpr, _ = roc_curve(err_B, s)
        auc_val = roc_auc_score(err_B, s)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Alarm Rate (1 - Specificity)")
    plt.ylabel("Safety Recall (Sensitivity)")
    plt.title("Partition B ROC Curves (GTSRB MobileNetV3-Small)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_png = RESULTS_DIR / "roc_curves.png"
    plt.savefig(roc_png, dpi=300)
    plt.close()
    print(f"  [Saved] {roc_png}")

    # 2. Risk-Coverage Curves
    # Sort samples by score ascending (lowest risk first), compute error rate at varying coverage
    plt.figure(figsize=(8, 6))
    for name, s in [
        ("MCB", cbi_op_j),
        ("Softmax Entropy", continuous_baselines["softmax_entropy"]),
        ("TTA Variance", continuous_baselines["tta_var"]),
        ("MC-Dropout", continuous_baselines["mcdo_headonly"]),
    ]:
        order = np.argsort(s)
        coverage = np.linspace(0.10, 1.0, 100)
        risk = []
        for cov in coverage:
            k_cov = int(cov * N_B)
            cov_err = err_B[order[:k_cov]].mean()
            risk.append(cov_err)
        plt.plot(coverage, risk, label=name, lw=2)
    plt.xlabel("Coverage (Fraction of Decisions Retained)")
    plt.ylabel("Selective Risk (Error Rate in Covered Set)")
    plt.title("Risk-Coverage Curves (Partition B)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    risk_png = RESULTS_DIR / "risk_coverage.png"
    plt.savefig(risk_png, dpi=300)
    plt.close()
    print(f"  [Saved] {risk_png}")

    # 3. Family Radar Chart
    labels = ["Geometric", "Photometric", "Sensor Noise"]
    err_vals = [mean_ir_err["geometric"], mean_ir_err["photometric"], mean_ir_err["sensor_noise"]]
    corr_vals = [mean_ir_corr["geometric"], mean_ir_corr["photometric"], mean_ir_corr["sensor_noise"]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    err_vals += err_vals[:1]
    corr_vals += corr_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, err_vals, color="red", linewidth=2, label="Base Errors")
    ax.fill(angles, err_vals, color="red", alpha=0.25)
    ax.plot(angles, corr_vals, color="blue", linewidth=2, label="Correct Predictions")
    ax.fill(angles, corr_vals, color="blue", alpha=0.15)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title("Family Instability Radar Profile (Partition B)", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    radar_png = RESULTS_DIR / "family_radar.png"
    plt.savefig(radar_png, dpi=300)
    plt.close()
    print(f"  [Saved] {radar_png}")

    # Save raw evaluation artifacts
    raw_output = {
        "b_results": b_results,
        "ablations": ablations,
        "decision_rules": {
            "D1": "MET" if d1_met else "NOT MET",
            "D2": "MET" if d2_met else "NOT MET",
            "D3": "MET" if d3_met else "NOT MET",
            "D4": "PENDING_FAULT_INJECTION",
        },
        "family_profile": family_profile,
    }

    raw_path = RESULTS_DIR / "b_evaluation_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_output, f, indent=2)
    print(f"\n[Saved] {raw_path}")
    print("=" * 70)

if __name__ == "__main__":
    run()
