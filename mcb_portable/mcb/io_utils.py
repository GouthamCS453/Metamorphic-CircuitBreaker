"""
mcb/io_utils.py - Run context, result writing (results.json + report.md), manifest, markdown helpers.

Every test writes the SAME envelope so results from different members/models/datasets can be merged:
  <run_dir>/<subdir>/results.json   {schema_version, test_id, run_id, model, dataset, status, warnings, results}
  <run_dir>/<subdir>/report.md      human-readable report generated from the same numbers
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import SCHEMA_VERSION, __version__

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- json helpers
def jsonable(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if v != v else v
    if isinstance(o, np.ndarray):
        return [jsonable(x) for x in o.tolist()]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and o != o:
        return None
    raise TypeError(f"not serialisable: {type(o)}")


def clean(o: Any):
    """Recursively replace NaN floats by None so the JSON is strictly valid."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if v != v else v
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, default=jsonable), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(clean(obj), sort_keys=True, default=jsonable).encode()).hexdigest()[:16]


# ------------------------------------------------------------ markdown helpers
def pct(x: Optional[float], d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{100 * x:.{d}f}%"


def f3(x: Optional[float], d: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def ci_str(point: float, lo: float, hi: float, d: int = 3) -> str:
    return f"{f3(point, d)} [{f3(lo, d)}, {f3(hi, d)}]"


def pct_ci(point: float, ci) -> str:
    return f"{pct(point)} [{pct(ci[0])}, {pct(ci[1])}]"


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------- environment
def environment() -> Dict:
    env = dict(python=sys.version.split()[0], platform=platform.platform(), mcb_version=__version__)
    for mod in ("numpy", "pandas", "PIL", "torch"):
        try:
            m = __import__(mod)
            env[mod.lower()] = getattr(m, "__version__", "unknown")
        except Exception:
            env[mod.lower()] = None
    try:
        import torch
        env["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        env["cuda"] = None
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        env["git_commit"] = None
    return env


# ------------------------------------------------------------- member config
def load_member_config(path: str):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    spec = importlib.util.spec_from_file_location("member_config", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    missing = [a for a in ("MODEL_NAME", "DATASET_NAME", "CLASS_NAMES", "load_model", "get_dataset")
               if not hasattr(mod, a)]
    if missing:
        raise AttributeError(f"{path.name} is missing required entries: {missing}")
    return mod


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-").lower()


class RunContext:
    """Paths and metadata for one (model, dataset) run."""

    def __init__(self, cfg, run_dir: Optional[str] = None):
        self.cfg = cfg
        self.run_id = getattr(cfg, "RUN_ID", None) or f"{_slug(cfg.MODEL_NAME)}__{_slug(cfg.DATASET_NAME)}"
        self.run_dir = Path(run_dir) if run_dir else ROOT / "results" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = cfg.MODEL_NAME
        self.dataset_name = cfg.DATASET_NAME
        self.class_names = list(cfg.CLASS_NAMES)
        self.seed = int(getattr(cfg, "SEED", 42))

    # paths
    samples_csv = property(lambda s: s.run_dir / "samples.csv")
    arrays_npz = property(lambda s: s.run_dir / "arrays.npz")
    arrays_external_npz = property(lambda s: s.run_dir / "arrays_external.npz")
    arrays_shift_npz = property(lambda s: s.run_dir / "arrays_shift.npz")
    calibration_json = property(lambda s: s.run_dir / "calibrated_config.json")
    manifest_json = property(lambda s: s.run_dir / "run_manifest.json")

    def manifest(self) -> Dict:
        return read_json(self.manifest_json) if self.manifest_json.exists() else {}

    def update_manifest(self, **updates) -> Dict:
        m = self.manifest()
        m.update(updates)
        m.update(schema_version=SCHEMA_VERSION, run_id=self.run_id, model=self.model_name,
                 dataset=self.dataset_name, dataset_version=getattr(self.cfg, "DATASET_VERSION", None),
                 class_names=self.class_names, seed=self.seed, updated=datetime.now().isoformat(timespec="seconds"),
                 environment=environment())
        write_json(self.manifest_json, m)
        return m


def save_result(ctx: RunContext, test_id: str, subdir: str, title: str, results: Dict, md: str,
                status: str = "ok", warnings: Optional[List[str]] = None) -> Path:
    d = ctx.run_dir / subdir
    d.mkdir(parents=True, exist_ok=True)
    warnings = warnings or []
    doc = dict(schema_version=SCHEMA_VERSION, test_id=test_id, title=title, run_id=ctx.run_id,
               model=ctx.model_name, dataset=ctx.dataset_name, status=status,
               generated=datetime.now().isoformat(timespec="seconds"), warnings=warnings, results=results)
    write_json(d / "results.json", doc)
    head = [f"# {title}", "",
            f"_Run `{ctx.run_id}` | model: **{ctx.model_name}** | dataset: **{ctx.dataset_name}** | "
            f"generated {doc['generated']}_", ""]
    if warnings:
        head += ["> **Warnings**", *[f"> - {w}" for w in warnings], ""]
    (d / "report.md").write_text("\n".join(head) + md + "\n", encoding="utf-8")
    print(f"[OK] {test_id}: wrote {d / 'results.json'} and report.md")
    return d
