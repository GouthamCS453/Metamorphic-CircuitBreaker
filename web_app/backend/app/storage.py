"""
web_app/backend/app/storage.py
File-based JSON case repository and image/gradcam asset manager.
"""
from __future__ import annotations

import json
import uuid
import base64
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────

_BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _BACKEND_DIR / "data"
UPLOADS_DIR = _BACKEND_DIR / "uploads"
CASES_FILE = DATA_DIR / "cases.json"

DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

if not CASES_FILE.exists():
    CASES_FILE.write_text(json.dumps({}))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pil_to_b64(pil_img: Image.Image, size: int = 300, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.resize((size, size), Image.Resampling.LANCZOS).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _overlay_to_b64(pil_img: Image.Image, cam: np.ndarray, size: int = 300) -> str:
    import cv2
    img_np = np.array(pil_img.resize((size, size))).astype(np.float32) / 255.0
    cam_r = cv2.resize(cam, (size, size))
    heat = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = 0.50 * heat + 0.50 * img_np
    result = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    pil = Image.fromarray(result)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _load() -> Dict[str, Any]:
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


def _save(data: Dict[str, Any]) -> None:
    CASES_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ── Case ID ────────────────────────────────────────────────────────────────────

def _new_case_id() -> str:
    year = datetime.now(timezone.utc).year
    short = uuid.uuid4().hex[:6].upper()
    return f"AMH-DERM-{year}-{short}"


# ── Serialise ML objects into JSON-safe dicts ──────────────────────────────────

def serialise_report(report, config_dict: Dict) -> Dict:
    """Convert DiagnosticReport to a JSON-safe dict."""
    return {
        "baseline_label": report.baseline_label,
        "baseline_confidence": report.baseline_confidence,
        "baseline_idx": report.baseline_idx,
        "n_tests": len(report.test_results),
        "n_flips": sum(1 for r in report.test_results if r.flipped),
        "type_scores": report.type_scores,
        "family_instability": report.family_instability,
        "peak_family": report.peak_family,
        "peak_family_score": report.peak_family_score,
        "cross_family_spread": report.cross_family_spread,
        "compromised_families": report.compromised_families,
        "cbi": report.cbi,
        "state": report.state.value,
        "action": report.action,
        "config_used": config_dict,
        "test_results": [
            {
                "test_id": r.test_id,
                "test_name": r.test_name,
                "family": r.family,
                "transform_type": r.transform_type,
                "severity": r.severity,
                "weight": r.weight,
                "pred_label": r.pred_label,
                "confidence": r.confidence,
                "flipped": r.flipped,
            }
            for r in report.test_results
        ],
    }


def serialise_gradcam(orig_img: Image.Image, gradcam_data: Dict) -> Dict:
    """Convert gradcam_data (PIL + numpy arrays) to base64-encoded dict."""
    bc = gradcam_data.get("baseline_cam")
    baseline_b64 = None
    if bc is not None:
        baseline_b64 = _overlay_to_b64(orig_img, bc)
    else:
        baseline_b64 = _pil_to_b64(orig_img)

    flip_list = []
    for fc in gradcam_data.get("flip_cams", []):
        t_img = fc.get("transformed_image")
        cam = fc.get("cam")
        t_b64 = _pil_to_b64(t_img) if t_img else None
        ov_b64 = _overlay_to_b64(t_img, cam) if (t_img is not None and cam is not None) else t_b64
        flip_list.append({
            "test_id": fc["test_id"],
            "test_name": fc["test_name"],
            "family": fc["family"],
            "severity": fc["severity"],
            "weight": fc["weight"],
            "flipped_label": fc["flipped_label"],
            "flipped_confidence": fc["flipped_confidence"],
            "transformed_image_b64": t_b64,
            "cam_overlay_b64": ov_b64,
        })

    return {
        "original_image_b64": _pil_to_b64(orig_img),
        "baseline_cam_b64": baseline_b64,
        "flip_cams": flip_list,
    }


# ── Public CRUD ────────────────────────────────────────────────────────────────

def create_case(
    orig_img: Image.Image,
    report,
    gradcam_data: Dict,
    config_dict: Dict,
    patient_context: Optional[Dict] = None,
) -> str:
    case_id = _new_case_id()
    created_at = datetime.now(timezone.utc).isoformat()

    report_dict = serialise_report(report, config_dict)
    gradcam_dict = serialise_gradcam(orig_img, gradcam_data)

    state = report.state.value
    action = report.action
    if state == "CLOSED":
        status = "auto_approved"
    else:
        status = "pending_review"

    record = {
        "case_id": case_id,
        "created_at": created_at,
        "state": state,
        "action": action,
        "status": status,
        "baseline_label": report.baseline_label,
        "baseline_confidence": report.baseline_confidence,
        "cbi": report.cbi,
        "n_flips": report_dict["n_flips"],
        "patient_context": patient_context or {},
        "report": report_dict,
        "gradcam": gradcam_dict,
        "doctor_review": None,
        "doctor_name": None,
        "override_diagnosis": None,
    }

    data = _load()
    data[case_id] = record
    _save(data)
    return case_id


def get_case(case_id: str) -> Optional[Dict]:
    return _load().get(case_id)


def list_cases(status_filter: Optional[str] = None) -> List[Dict]:
    data = _load()
    cases = list(data.values())
    cases.sort(key=lambda c: c["created_at"], reverse=True)
    if status_filter:
        cases = [c for c in cases if c["status"] == status_filter]
    return cases


def apply_doctor_review(case_id: str, review: Dict) -> bool:
    data = _load()
    if case_id not in data:
        return False
    record = data[case_id]
    record["doctor_review"] = review
    record["doctor_name"] = review.get("doctor_name")
    if review.get("action") == "OVERRIDE":
        record["status"] = "overridden"
        record["override_diagnosis"] = review.get("override_diagnosis")
    else:
        record["status"] = "approved"
    _save(data)
    return True


def get_stats() -> Dict:
    data = _load()
    cases = list(data.values())
    total = len(cases)
    auto_approved = sum(1 for c in cases if c["status"] == "auto_approved")
    pending = sum(1 for c in cases if c["status"] == "pending_review")
    tripped = sum(1 for c in cases if c["state"] == "OPEN")
    half_open = sum(1 for c in cases if c["state"] == "HALF_OPEN")
    overridden = sum(1 for c in cases if c["status"] == "overridden")
    override_rate = (overridden / total) if total else 0.0
    recent = [c["state"] for c in cases[:10]]
    return {
        "total_cases": total,
        "auto_approved": auto_approved,
        "pending_review": pending,
        "tripped_open": tripped,
        "half_open": half_open,
        "overridden": overridden,
        "override_rate": round(override_rate, 4),
        "recent_states": recent,
    }
