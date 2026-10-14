"""
mcb/calibration.py - Stage-0 parameter search and threshold derivation (calibration split ONLY).

Search space
  signal  : flip | drift | hybrid
  weights : 5 (alpha, beta, gamma) configurations
  tau_fam : family-compromise cutoff grid
Selection : maximise calibration AUROC, then apply the one-standard-error rule and prefer
            the configuration with the flattest neighbourhood (plateau) - robust to noise.
Honest estimate : nested cross-validation (select on train folds, score on held-out fold).
Thresholds : theta_trip is the (1-FAR) quantile of CBI over CORRECT calibration samples, so the
             deployed false-alarm rate is controlled by construction (no magic number).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .metrics import auroc, bootstrap_auroc, threshold_for_far, youden_threshold, confusion
from .scoring import combine, family_ir, signals, split_probs

WEIGHT_CONFIGS = [
    ("balanced", 0.50, 0.35, 0.15),
    ("pure_mt", 0.60, 0.40, 0.00),
    ("peak_focus", 0.70, 0.20, 0.10),
    ("softmax_trust", 0.25, 0.25, 0.50),
    ("spread", 0.30, 0.60, 0.10),
]
TAU_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
SIGNALS = ("flip", "drift", "hybrid")
FAR_PROFILES = {"far05": 0.05, "far10": 0.10, "far20": 0.20}
MEAN_W = 0.70


def build_grid(P: np.ndarray, meta: List[dict], sigs=SIGNALS, weight_configs=WEIGHT_CONFIGS, tau_grid=TAU_GRID):
    """Return (configs, cbi_matrix[n_configs, N]) on the given samples."""
    _, c0 = split_probs(P)
    configs, rows = [], []
    for sig in sigs:
        ir = family_ir(signals(P, sig), meta, None, MEAN_W)
        M = np.column_stack([ir[f] for f in sorted(ir, key=lambda x: ["geometric", "photometric", "sensor_noise"].index(x))])
        for wname, a, b, g in weight_configs:
            for ti, tau in enumerate(tau_grid):
                configs.append(dict(signal=sig, wname=wname, alpha=a, beta=b, gamma=g, tau_fam=tau, tau_idx=ti))
                rows.append(combine(M, c0, a, b, g, tau))
    return configs, np.vstack(rows)


def nested_cv(cbi_mat: np.ndarray, y: np.ndarray, ref_scores: Dict[str, np.ndarray],
              folds: int = 5, repeats: int = 3, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    sel_auc, ref_auc = [], {k: [] for k in ref_scores}
    chosen = []
    for _ in range(repeats):
        fid = np.empty(len(y), int)
        for grp in (pos, neg):
            fid[rng.permutation(grp)] = np.arange(len(grp)) % folds
        for k in range(folds):
            tr, te = fid != k, fid == k
            if y[te].sum() == 0 or (1 - y[te]).sum() == 0 or y[tr].sum() == 0:
                continue
            aucs = [auroc(y[tr], cbi_mat[i][tr]) for i in range(len(cbi_mat))]
            b = int(np.nanargmax(aucs))
            chosen.append(b)
            sel_auc.append(auroc(y[te], cbi_mat[b][te]))
            for name, s in ref_scores.items():
                ref_auc[name].append(auroc(y[te], s[te]))
    return dict(n_folds_used=len(sel_auc),
                heldout_auroc_mean=float(np.mean(sel_auc)) if sel_auc else float("nan"),
                heldout_auroc_sd=float(np.std(sel_auc)) if sel_auc else float("nan"),
                reference_heldout_auroc={k: float(np.mean(v)) if v else float("nan") for k, v in ref_auc.items()},
                selection_stability=float(np.bincount(chosen).max() / len(chosen)) if chosen else float("nan"))


def select_config(configs, aucs: np.ndarray, y: np.ndarray, cbi_mat: np.ndarray,
                  allowed_signals: Optional[tuple] = None, seed: int = 0) -> Dict:
    """One-SE rule + plateau preference among configs whose signal is allowed."""
    ok = np.array([(allowed_signals is None) or (c["signal"] in allowed_signals) for c in configs])
    idx_ok = np.where(ok)[0]
    best_i = idx_ok[int(np.nanargmax(aucs[idx_ok]))]
    best = aucs[best_i]
    _, lo, hi = bootstrap_auroc(y, cbi_mat[best_i], n_boot=200, seed=seed)
    se = (hi - lo) / 3.92 if not np.isnan(lo) else 0.0
    by_key: Dict[tuple, Dict[int, float]] = {}
    for i in idx_ok:
        c = configs[i]
        by_key.setdefault((c["signal"], c["wname"]), {})[c["tau_idx"]] = aucs[i]

    def plateau(i):
        c = configs[i]
        d = by_key[(c["signal"], c["wname"])]
        vals = [d[j] for j in (c["tau_idx"] - 1, c["tau_idx"], c["tau_idx"] + 1) if j in d]
        return float(np.mean(vals))

    cands = [i for i in idx_ok if aucs[i] >= best - se]
    pick = max(cands, key=lambda i: (plateau(i), aucs[i]))
    return dict(index=int(pick), best_index=int(best_i), best_auroc=float(best), se=float(se),
                n_candidates_within_1se=len(cands), plateau_auroc=plateau(pick))


def make_profiles(scores_cal: np.ndarray, y_cal: np.ndarray, seed: int = 0, n_boot: int = 200) -> Dict:
    """Operating profiles from calibration data. Flag rule: score >= theta."""
    correct = scores_cal[y_cal == 0]
    rng = np.random.default_rng(seed)
    out = {}
    for name, far in FAR_PROFILES.items():
        trip = threshold_for_far(correct, far)
        warn = threshold_for_far(correct, min(0.5, 2 * far))
        if not warn < trip:
            warn = trip * 0.5
        boots = [threshold_for_far(rng.choice(correct, len(correct)), far) for _ in range(n_boot)]
        out[name] = dict(target_far=far, theta_trip=trip, theta_warn=warn,
                         theta_trip_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                         calibration_confusion=confusion(y_cal, scores_cal >= trip))
    trip = youden_threshold(scores_cal, y_cal)
    realised = float(((scores_cal >= trip) & (y_cal == 0)).sum() / max((y_cal == 0).sum(), 1))
    warn = threshold_for_far(correct, min(0.5, 2 * max(realised, 0.01)))
    if not warn < trip:
        warn = trip * 0.5
    out["balanced"] = dict(target_far=None, theta_trip=trip, theta_warn=warn, theta_trip_ci=None,
                           calibration_confusion=confusion(y_cal, scores_cal >= trip))
    return out


def baseline_thresholds(base_scores_cal: Dict[str, np.ndarray], y_cal: np.ndarray) -> Dict:
    out = {}
    for name, s in base_scores_cal.items():
        correct = s[y_cal == 0]
        d = {pn: threshold_for_far(correct, far) for pn, far in FAR_PROFILES.items()}
        d["balanced"] = youden_threshold(s, y_cal)
        out[name] = d
    return out
