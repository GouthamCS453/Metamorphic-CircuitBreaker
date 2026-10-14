"""
mcb/metrics.py - Detection metrics, confidence intervals and threshold helpers.

Convention everywhere: POSITIVE = the base model made an ERROR (what we want to flag);
a "flag" is a trip (OPEN) or warning (HALF_OPEN) depending on the table.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def confusion(y_err, flag) -> Dict:
    y = np.asarray(y_err).astype(bool)
    f = np.asarray(flag).astype(bool)
    tp, fp = int((y & f).sum()), int((~y & f).sum())
    tn, fn = int((~y & ~f).sum()), int((y & ~f).sum())
    rec, rl, rh = wilson(tp, tp + fn)
    far, fl, fh = wilson(fp, fp + tn)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * (0 if math.isnan(rec) else rec) / max(prec + (0 if math.isnan(rec) else rec), 1e-12)
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, n=int(len(y)),
                recall=rec, recall_ci=[rl, rh], far=far, far_ci=[fl, fh],
                precision=prec, f1=f1,
                youden_j=(0 if math.isnan(rec) else rec) - (0 if math.isnan(far) else far),
                flag_rate=(tp + fp) / max(len(y), 1),
                silent_errors=fn,
                accepted_accuracy=tn / max(tn + fn, 1))


def auroc(y, s) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def bootstrap_auroc(y, a, b: Optional[np.ndarray] = None, n_boot: int = 1000, seed: int = 0):
    """Stratified bootstrap. b=None -> (auc, lo, hi); else paired difference (auc_a - auc_b, lo, hi)."""
    y = np.asarray(y).astype(int)
    a = np.asarray(a, dtype=float)
    b = None if b is None else np.asarray(b, dtype=float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    point = auroc(y, a) - (auroc(y, b) if b is not None else 0.0)
    if len(pos) < 2 or len(neg) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy = y[idx]
        vals[i] = auroc(yy, a[idx]) - (auroc(yy, b[idx]) if b is not None else 0.0)
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def threshold_for_far(correct_scores, far: float) -> float:
    """Threshold t such that flagging `score >= t` flags at most floor(far*n) of the given
    CORRECT samples (conservative: ties at the boundary are not flagged)."""
    s = np.sort(np.asarray(correct_scores, dtype=float))[::-1]
    n = len(s)
    if n == 0:
        return float("nan")
    k = int(math.floor(far * n))
    if k >= n:
        return float(s[-1])
    return float(np.nextafter(s[k], np.inf))


def youden_threshold(scores, y_err) -> float:
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y_err).astype(bool)
    best, best_t = -np.inf, float(scores.max())
    npos, nneg = max(y.sum(), 1), max((~y).sum(), 1)
    for t in np.unique(scores):
        f = scores >= t
        j = (y & f).sum() / npos - (~y & f).sum() / nneg
        if j >= best:  # ties -> higher threshold (fewer false alarms)
            best, best_t = j, float(t)
    return best_t


def roc_recall_at_far(y_err, score, far: float) -> float:
    """Best achievable recall with FAR <= far, thresholding on the evaluated data itself
    (an oracle / upper bound, not a deployable operating point)."""
    y = np.asarray(y_err).astype(int)
    s = np.asarray(score, dtype=float)
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tps, fps = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[ss[1:] != ss[:-1], True]
    tpr = tps[last] / max(y.sum(), 1)
    fpr = fps[last] / max((1 - y).sum(), 1)
    ok = fpr <= far
    return float(tpr[ok].max()) if ok.any() else 0.0


def risk_coverage(y_err, score) -> Dict:
    """Selective classification: accept lowest-score samples first."""
    y = np.asarray(y_err).astype(int)
    s = np.asarray(score, dtype=float)
    order = np.argsort(s, kind="stable")
    ys = y[order]
    n = len(ys)
    risk = np.cumsum(ys) / np.arange(1, n + 1)
    out = dict(aurc=float(risk.mean()))
    for c in (0.95, 0.9, 0.8, 0.7):
        k = max(1, int(round(c * n)))
        out[f"acc_at_cov_{int(c * 100)}"] = float(1 - risk[k - 1])
    return out


def binom_sf(k: int, n: int, p: float) -> float:
    """P[X >= k], X ~ Binomial(n, p)."""
    return float(sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1)))


def spearman(x, y) -> float:
    x, y = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y, method="spearman"))


def mean_ci(values) -> Tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
