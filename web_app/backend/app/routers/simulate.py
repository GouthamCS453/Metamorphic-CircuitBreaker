"""
web_app/backend/app/routers/simulate.py
POST /api/simulate  — What-if CBI calculator
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.schemas import SimulationRequest, SimulationResponse

router = APIRouter(prefix="/api", tags=["simulate"])


@router.post("/simulate", response_model=SimulationResponse)
def simulate(req: SimulationRequest):
    total = req.alpha + req.beta + req.gamma
    if abs(total - 1.0) > 1e-4:
        raise HTTPException(status_code=422, detail=f"alpha+beta+gamma must equal 1.0 (got {total:.4f})")
    if req.theta_warn >= req.theta_trip:
        raise HTTPException(status_code=422, detail="theta_warn must be < theta_trip")

    a_part = req.alpha * req.hypothetical_ir_k
    b_part = req.beta * req.hypothetical_cfs
    g_part = req.gamma * req.hypothetical_uncertainty
    cbi = round(max(0.0, min(1.0, a_part + b_part + g_part)), 4)

    if cbi < req.theta_warn:
        state = "CLOSED"
    elif cbi < req.theta_trip:
        state = "HALF_OPEN"
    else:
        state = "OPEN"

    return SimulationResponse(
        cbi=cbi,
        state=state,
        components={"alpha_part": round(a_part, 4), "beta_part": round(b_part, 4), "gamma_part": round(g_part, 4)},
    )
