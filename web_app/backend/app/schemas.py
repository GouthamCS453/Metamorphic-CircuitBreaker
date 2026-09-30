"""
web_app/backend/app/schemas.py
Pydantic v2 request/response models for all FastAPI endpoints.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Config ────────────────────────────────────────────────────────────────────

class ConfigSchema(BaseModel):
    alpha: float
    beta: float
    gamma: float
    tau_fam: float
    theta_warn: float
    theta_trip: float


class ConfigUpdateRequest(BaseModel):
    alpha: float = Field(ge=0, le=1)
    beta: float = Field(ge=0, le=1)
    gamma: float = Field(ge=0, le=1)
    tau_fam: float = Field(ge=0.05, le=0.80)
    theta_warn: float = Field(ge=0.05, le=0.90)
    theta_trip: float = Field(ge=0.05, le=0.99)


# ── Single metamorphic test result ────────────────────────────────────────────

class SingleTestResultSchema(BaseModel):
    test_id: str
    test_name: str
    family: str
    transform_type: str
    severity: str
    weight: float
    pred_label: str
    confidence: float
    flipped: bool


# ── GradCAM data sent to frontend ─────────────────────────────────────────────

class FlipCamSchema(BaseModel):
    test_id: str
    test_name: str
    family: str
    severity: str
    weight: float
    flipped_label: str
    flipped_confidence: float
    # base64 encoded PNGs
    transformed_image_b64: Optional[str] = None
    cam_overlay_b64: Optional[str] = None


class GradcamDataSchema(BaseModel):
    original_image_b64: Optional[str] = None  # raw original (no overlay)
    baseline_cam_b64: Optional[str] = None    # GradCAM overlay on original
    flip_cams: List[FlipCamSchema] = []


# ── Diagnostic report ─────────────────────────────────────────────────────────

class DiagnosticReportSchema(BaseModel):
    baseline_label: str
    baseline_confidence: float
    baseline_idx: int
    n_tests: int
    n_flips: int
    type_scores: Dict[str, float]
    family_instability: Dict[str, float]
    peak_family: str
    peak_family_score: float
    cross_family_spread: float
    compromised_families: List[str]
    cbi: float
    state: str            # "CLOSED" | "HALF_OPEN" | "OPEN"
    action: str
    config_used: ConfigSchema
    test_results: List[SingleTestResultSchema]


# ── Case ──────────────────────────────────────────────────────────────────────

class DoctorReviewRequest(BaseModel):
    action: str                   # "APPROVE" | "OVERRIDE"
    override_diagnosis: Optional[str] = None
    override_label: Optional[str] = None
    clinical_notes: str = ""
    doctor_name: str = "Dr. Sarah Jenkins, MD"
    biopsy_recommended: bool = False
    follow_up_urgency: str = "routine"   # "routine" | "urgent" | "emergency"


class CaseSummarySchema(BaseModel):
    case_id: str
    created_at: str
    state: str
    action: str
    baseline_label: str
    baseline_confidence: float
    cbi: float
    n_flips: int
    status: str           # "pending_review" | "approved" | "overridden" | "auto_approved"
    doctor_name: Optional[str] = None
    override_diagnosis: Optional[str] = None
    patient_context: Optional[Dict[str, Any]] = None


class CaseDetailSchema(CaseSummarySchema):
    report: DiagnosticReportSchema
    gradcam: GradcamDataSchema
    doctor_review: Optional[Dict[str, Any]] = None


# ── Predict ───────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    case_id: str
    report: DiagnosticReportSchema
    gradcam: GradcamDataSchema


# ── Simulate ──────────────────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    alpha: float
    beta: float
    gamma: float
    tau_fam: float
    theta_warn: float
    theta_trip: float
    hypothetical_ir_k: float = Field(ge=0, le=1)
    hypothetical_cfs: float = Field(ge=0, le=1)
    hypothetical_uncertainty: float = Field(ge=0, le=1)


class SimulationResponse(BaseModel):
    cbi: float
    state: str
    components: Dict[str, float]


# ── Presets ───────────────────────────────────────────────────────────────────

class PresetSchema(BaseModel):
    preset_id: str
    name: str
    description: str
    expected_state: str
    cbi_ref: str
    flips_ref: str
    image_filename: str
    image_url: Optional[str] = None


# ── Circuit Stats ─────────────────────────────────────────────────────────────

class CircuitStatsSchema(BaseModel):
    total_cases: int
    auto_approved: int
    pending_review: int
    tripped_open: int
    half_open: int
    overridden: int
    override_rate: float
    recent_states: List[str]
