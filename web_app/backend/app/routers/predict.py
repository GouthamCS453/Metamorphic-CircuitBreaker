"""
web_app/backend/app/routers/predict.py
POST /api/predict  — accepts image upload or preset ID, runs CB, creates case.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.config import get_active_config, PRESETS_DIR, PROJECT_ROOT
from app.schemas import PredictResponse
from app.storage import create_case, serialise_report, serialise_gradcam
from app.services.circuit_service import run_circuit_breaker

router = APIRouter(prefix="/api", tags=["predict"])

_PRESET_DIR = PRESETS_DIR


def _load_preset_image(preset_id: str) -> Image.Image:
    preset_dir = _PRESET_DIR / preset_id
    if not preset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")
    imgs = [p for p in preset_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not imgs:
        raise HTTPException(status_code=404, detail="No image found in preset")
    return Image.open(imgs[0]).convert("RGB")


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: Optional[UploadFile] = File(None),
    preset_id: Optional[str] = Form(None),
    patient_name: Optional[str] = Form(None),
    patient_age: Optional[str] = Form(None),
    patient_sex: Optional[str] = Form(None),
    lesion_site: Optional[str] = Form(None),
    symptoms: Optional[str] = Form(None),
):
    # Resolve image
    if file is not None:
        content = await file.read()
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
    elif preset_id:
        image = _load_preset_image(preset_id)
    else:
        raise HTTPException(status_code=400, detail="Provide an image file or preset_id")

    # Build patient context
    patient_context = {
        "patient_name": patient_name or "Anonymous",
        "patient_age": patient_age,
        "patient_sex": patient_sex,
        "lesion_site": lesion_site,
        "symptoms": symptoms,
    }

    # Run the actual circuit breaker pipeline
    try:
        report, gradcam_data = run_circuit_breaker(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    cfg = get_active_config()
    config_dict = cfg.to_dict()

    # Persist case
    case_id = create_case(image, report, gradcam_data, config_dict, patient_context)

    # Build response
    report_dict = serialise_report(report, config_dict)
    gradcam_dict = serialise_gradcam(image, gradcam_data)

    return PredictResponse(
        case_id=case_id,
        report=report_dict,
        gradcam=gradcam_dict,
    )
