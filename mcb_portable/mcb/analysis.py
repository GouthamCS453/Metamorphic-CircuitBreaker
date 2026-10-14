"""
mcb/analysis.py - Shared helpers to load a run's cached arrays and the frozen calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .io_utils import ROOT, RunContext, load_member_config, read_json
from .scoring import split_probs
from .transforms import bank_meta, build_bank


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=str(ROOT / "member_config.py"),
                   help="member config file (default: member_config.py)")
    p.add_argument("--run-dir", default=None, help="override results folder")
    return p


def make_ctx(args) -> RunContext:
    return RunContext(load_member_config(args.config), args.run_dir)


@dataclass
class RunData:
    ids: List[str]
    df: pd.DataFrame
    P: np.ndarray
    labels: np.ndarray
    pred: np.ndarray
    c0: np.ndarray
    y_err: np.ndarray
    split: np.ndarray
    meta: List[dict]
    class_names: List[str]


def load_run(ctx: RunContext, arrays_path: Optional[Path] = None, samples: Optional[pd.DataFrame] = None) -> RunData:
    arrays_path = Path(arrays_path or ctx.arrays_npz)
    if not arrays_path.exists():
        raise FileNotFoundError(f"{arrays_path.name} not found. Run step0_extract.py first.")
    df = samples if samples is not None else pd.read_csv(ctx.samples_csv)
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    z = np.load(arrays_path, allow_pickle=False)
    ids = [str(x) for x in z["ids"]]
    P = z["probs"].astype(np.float64)
    d = df.set_index("sample_id").loc[ids].reset_index()
    labels = d["label"].astype(int).to_numpy()
    pred, c0 = split_probs(P)
    meta = bank_meta(build_bank(1.0))
    if P.shape[1] - 1 != len(meta):
        raise ValueError(f"arrays hold {P.shape[1] - 1} variants but the bank has {len(meta)}")
    split = d["split"].to_numpy() if "split" in d.columns else np.array(["eval"] * len(d))
    return RunData(ids, d, P, labels, pred, c0, (pred != labels).astype(int), split, meta, list(ctx.class_names))


def load_calibration(ctx: RunContext) -> dict:
    if not ctx.calibration_json.exists():
        raise FileNotFoundError("calibrated_config.json not found. Run step1_calibrate.py first "
                                "(no test may run on un-calibrated thresholds).")
    return read_json(ctx.calibration_json)


def critical_mask(ctx: RunContext, labels: np.ndarray) -> Optional[np.ndarray]:
    crit = getattr(ctx.cfg, "CRITICAL_CLASSES", None)
    if not crit:
        return None
    idx = [c if isinstance(c, int) else ctx.class_names.index(c) for c in crit]
    return np.isin(labels, idx)
