"""
web_app/backend/app/routers/presets.py
GET /api/presets  — lists available ISIC demo presets
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List
from fastapi import APIRouter
from app.config import PRESETS_DIR
from app.schemas import PresetSchema

from fastapi.responses import FileResponse
from fastapi import HTTPException

router = APIRouter(prefix="/api/presets", tags=["presets"])
_PRESET_DIR = PRESETS_DIR


@router.get("", response_model=List[PresetSchema])
def list_presets():
    result = []
    if not _PRESET_DIR.exists():
        return result
    for d in sorted(_PRESET_DIR.iterdir()):
        mp = d / "metadata.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text(encoding="utf-8"))
        pid = meta.get("preset_id", d.name)
        result.append(PresetSchema(
            preset_id=pid,
            name=meta.get("name", d.name),
            description=meta.get("description", ""),
            expected_state=meta.get("expected_state", "CLOSED"),
            cbi_ref=str(meta.get("cbi_ref", "--")),
            flips_ref=str(meta.get("flips_ref", "--")),
            image_filename=meta.get("image_filename", ""),
            image_url=f"/api/presets/{pid}/image",
        ))
    return result


@router.get("/{preset_id}/image")
def get_preset_image(preset_id: str):
    preset_dir = _PRESET_DIR / preset_id
    if not preset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")
    imgs = [p for p in preset_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not imgs:
        raise HTTPException(status_code=404, detail="Image not found in preset")
    return FileResponse(imgs[0])
