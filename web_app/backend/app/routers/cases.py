"""
web_app/backend/app/routers/cases.py
GET /api/cases, GET /api/cases/{id}, POST /api/cases/{id}/review, GET /api/stats
"""
from __future__ import annotations
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from app.schemas import CaseSummarySchema, CaseDetailSchema, DoctorReviewRequest, CircuitStatsSchema
from app import storage

router = APIRouter(prefix="/api/cases", tags=["cases"])


@router.get("", response_model=List[CaseSummarySchema])
def list_cases(status: Optional[str] = None):
    return storage.list_cases(status_filter=status)


@router.get("/stats", response_model=CircuitStatsSchema)
def get_stats():
    return storage.get_stats()


@router.get("/{case_id}", response_model=CaseDetailSchema)
def get_case(case_id: str):
    case = storage.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.post("/{case_id}/review")
def doctor_review(case_id: str, review: DoctorReviewRequest):
    case = storage.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    ok = storage.apply_doctor_review(case_id, review.model_dump())
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save review")
    return {"message": "Review saved", "case_id": case_id, "status": review.action}
