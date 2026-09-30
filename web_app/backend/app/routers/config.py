"""
web_app/backend/app/routers/config.py
GET /api/config, POST /api/config, GET /api/config/profiles
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.config import (
    CBConfig, get_active_config, set_active_config,
    CLINICAL_PROFILES,
)
from app.schemas import ConfigSchema, ConfigUpdateRequest

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("", response_model=ConfigSchema)
def get_config():
    return get_active_config().to_dict()


@router.post("", response_model=ConfigSchema)
def update_config(req: ConfigUpdateRequest):
    cfg = CBConfig(
        alpha=req.alpha,
        beta=req.beta,
        gamma=req.gamma,
        tau_fam=req.tau_fam,
        theta_warn=req.theta_warn,
        theta_trip=req.theta_trip,
    )
    try:
        cfg.validate()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    set_active_config(cfg)
    return cfg.to_dict()


@router.get("/profiles")
def get_profiles():
    return [
        {
            "id": pid,
            "label": p["label"],
            "description": p["description"],
            "config": p["config"].to_dict(),
        }
        for pid, p in CLINICAL_PROFILES.items()
    ]


@router.post("/profiles/{profile_id}", response_model=ConfigSchema)
def apply_profile(profile_id: str):
    if profile_id not in CLINICAL_PROFILES:
        raise HTTPException(status_code=404, detail="Profile not found")
    cfg = CLINICAL_PROFILES[profile_id]["config"]
    set_active_config(cfg)
    return cfg.to_dict()
