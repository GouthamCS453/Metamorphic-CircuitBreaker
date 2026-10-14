"""
mcb/scoring.py - Hierarchical Composite Brittleness Index (CBI) and baseline scores.

Everything operates on a probability tensor P of shape (N, V+1, C):
  P[:, 0]   -> softmax of the original image
  P[:, 1:]  -> softmax of each of the V transformed variants
so all scoring / calibration / ablation is offline numpy (no model needed).

Per-variant "signal" S in [0,1]:
  flip  : 1 if the variant's argmax differs from the base prediction (original MCB)
  drift : max(0, c0 - p_variant(y_base))  continuous confidence drop of the base class
  hybrid: max(flip, drift)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .transforms import FAMILIES

DEFAULT_PARAMS = dict(alpha=0.50, beta=0.35, gamma=0.15, tau_fam=0.20, mean_w=0.70,
                      sev_weights=None, flat=False, families=None, signal="flip")
STATE_NAMES = {0: "CLOSED", 1: "HALF_OPEN", 2: "OPEN"}


def split_probs(P: np.ndarray):
    """Return (pred, c0) from the original-image slice."""
    idx = np.arange(P.shape[0])
    pred = P[:, 0, :].argmax(1)
    return pred, P[idx, 0, pred]


def signals(P: np.ndarray, kind: str = "flip") -> np.ndarray:
    n, v1, _ = P.shape
    v = v1 - 1
    pred, c0 = split_probs(P)
    var = P[:, 1:, :]
    flips = (var.argmax(2) != pred[:, None]).astype(float)
    pv = np.take_along_axis(var, np.broadcast_to(pred[:, None, None], (n, v, 1)), axis=2)[:, :, 0]
    drift = np.clip(c0[:, None] - pv, 0.0, 1.0)
    if kind == "flip":
        return flips
    if kind == "drift":
        return drift
    if kind == "hybrid":
        return np.maximum(flips, drift)
    raise ValueError(f"unknown signal kind: {kind}")


def _weights(meta, idxs, sev_weights):
    return np.array([(sev_weights[meta[i]["severity"]] if sev_weights else meta[i]["weight"]) for i in idxs],
                    dtype=float)


def family_ir(S: np.ndarray, meta: List[dict], sev_weights: Optional[dict] = None,
              mean_w: float = 0.70, families: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Level-1 (per transform type, severity-weighted) then Level-2 (mean_w*mean + (1-mean_w)*peak)."""
    families = families or FAMILIES
    out = {}
    for fam in families:
        types: Dict[str, List[int]] = {}
        for i, m in enumerate(meta):
            if m["family"] == fam:
                types.setdefault(m["type"], []).append(i)
        cols = []
        for idxs in types.values():
            w = _weights(meta, idxs, sev_weights)
            cols.append((S[:, idxs] * w).sum(1) / max(w.sum(), 1e-9))
        if cols:
            M = np.column_stack(cols)
            out[fam] = mean_w * M.mean(1) + (1.0 - mean_w) * M.max(1)
        else:
            out[fam] = np.zeros(S.shape[0])
    return out


def flat_signal(S, meta, sev_weights=None, families=None):
    families = families or FAMILIES
    idxs = [i for i, m in enumerate(meta) if m["family"] in families]
    w = _weights(meta, idxs, sev_weights)
    return (S[:, idxs] * w).sum(1) / max(w.sum(), 1e-9)


def combine(ir_matrix: np.ndarray, c0: np.ndarray, alpha, beta, gamma, tau) -> np.ndarray:
    """CBI = alpha*max_k(IR_k) + beta*CFS + gamma*(1-c0), CFS = fraction of families with IR_k >= tau."""
    peak = ir_matrix.max(1)
    cfs = (ir_matrix >= tau).sum(1) / ir_matrix.shape[1]
    return alpha * peak + beta * cfs + gamma * (1.0 - c0)


def compute_cbi(S: np.ndarray, c0: np.ndarray, meta: List[dict], params: Optional[dict] = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    fams = p["families"] or FAMILIES
    if p["flat"]:
        f = flat_signal(S, meta, p["sev_weights"], fams)
        cbi = (p["alpha"] + p["beta"]) * f + p["gamma"] * (1.0 - c0)
        return dict(cbi=cbi, peak=f, cfs=np.zeros_like(f), ir={})
    ir = family_ir(S, meta, p["sev_weights"], p["mean_w"], fams)
    M = np.column_stack([ir[f] for f in fams])
    return dict(cbi=combine(M, c0, p["alpha"], p["beta"], p["gamma"], p["tau_fam"]),
                peak=M.max(1), cfs=(M >= p["tau_fam"]).mean(1), ir=ir)


def variant_cbi(vcfg: dict, P: np.ndarray, meta: List[dict]) -> np.ndarray:
    """CBI of a calibrated variant (entry of calibrated_config.json['variants'])."""
    params = vcfg["params"]
    _, c0 = split_probs(P)
    return compute_cbi(signals(P, params["signal"]), c0, meta, params)["cbi"]


def states(score: np.ndarray, theta_warn: float, theta_trip: float) -> np.ndarray:
    """0 = CLOSED, 1 = HALF_OPEN, 2 = OPEN."""
    return np.where(score >= theta_trip, 2, np.where(score >= theta_warn, 1, 0))


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(-1)


BASELINE_NAMES = ["msp", "entropy", "margin", "tta_entropy", "tta_var", "kflip"]


def baseline_scores(P: np.ndarray) -> Dict[str, np.ndarray]:
    """Plain-model / test-time-augmentation uncertainty baselines (higher = more suspicious)."""
    n, v1, _ = P.shape
    pred, c0 = split_probs(P)
    p0 = P[:, 0, :]
    srt = np.sort(p0, axis=1)
    margin = 1.0 - (srt[:, -1] - srt[:, -2])
    flips = (P[:, 1:, :].argmax(2) != pred[:, None])
    pv_all = np.take_along_axis(P, np.broadcast_to(pred[:, None, None], (n, v1, 1)), axis=2)[:, :, 0]
    return dict(msp=1.0 - c0, entropy=entropy(p0), margin=margin,
                tta_entropy=entropy(P.mean(1)), tta_var=pv_all.var(1),
                kflip=flips.sum(1).astype(float))
