"""FastAPI API for the additive vehicle fallback demonstration."""
from __future__ import annotations
import base64, io, json
from pathlib import Path
from typing import Set, Optional, List, Dict, Any
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
from src.circuit_breaker import MetamorphicCircuitBreaker, CircuitBreakerConfig
from src.models.mobilenet_adapter import MobileNetAdapter
from src.gradcam import overlay_heatmap_on_image
from fallback.controller import decision_to_dict
from fallback.integration import CircuitBreakerWithFallback

app = FastAPI(title="Metamorphic Circuit Breaker Fallback API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"] )
model = MobileNetAdapter()
existing_cb = MetamorphicCircuitBreaker(model)
system = CircuitBreakerWithFallback(existing_cb)
_last_state = "CLOSED"
_last_driver_present = True
_connections: Set[WebSocket] = set()

def _image_data_url(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def _gradcam_card(image: Image.Image, cam, label: str, confidence: float) -> dict:
    overlay = overlay_heatmap_on_image(image, cam, alpha=0.50, out_size=320)
    return {
        "label": label,
        "confidence": confidence,
        "image": _image_data_url(image.resize((320,320))),
        "heatmap": _image_data_url(Image.fromarray(overlay)),
    }

@app.get("/api/health")
def health() -> dict:
    return {"status":"ok","service":"fallback-api"}

@app.post("/api/evaluate")
async def evaluate(image: UploadFile = File(...), driver_present: bool = True) -> dict:
    try:
        raw = await image.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    report, gradcam_data = existing_cb.evaluate_with_gradcam(pil_image, max_gradcam_flips=3)
    global _last_state, _last_driver_present
    _last_state = report.state.value
    _last_driver_present = driver_present
    decision = system.fallback.evaluate(report.state, driver_present=driver_present)

    test_results = [{
        "id": r.test_id, "transform": r.test_name, "family": r.family,
        "severity": r.severity, "weight": r.weight, "prediction": r.pred_label,
        "confidence": r.confidence, "flipped": r.flipped
    } for r in report.test_results]

    gradcam = None
    if gradcam_data.get("baseline_cam") is not None:
        gradcam = {
            "baseline": _gradcam_card(pil_image, gradcam_data["baseline_cam"],
                                      report.baseline_label, report.baseline_confidence),
            "comparisons": [
                _gradcam_card(x["transformed_image"], x["cam"], x["flipped_label"], x["flipped_confidence"])
                | {"test_id":x["test_id"], "test_name":x["test_name"], "family":x["family"],
                   "severity":x["severity"], "weight":x["weight"],
                   "prediction_changed": x["flipped_label"] != report.baseline_label}
                for x in gradcam_data["flip_cams"]
            ],
        }

    payload = {
        "prediction":{"index":report.baseline_idx,"label":report.baseline_label,"confidence":report.baseline_confidence},
        "circuit_breaker":{
            "state":report.state.value,"action":report.action,"cbi":report.cbi,
            "peak_family":report.peak_family,"peak_family_score":report.peak_family_score,
            "cross_family_spread":report.cross_family_spread,
            "compromised_families":report.compromised_families,
            "family_instability":report.family_instability,"type_scores":report.type_scores,
            "test_results":test_results,"details":report.details,
        },
        "fallback":decision_to_dict(decision),
        "gradcam":gradcam,
    }
    await _broadcast(payload)
    return payload

@app.get("/api/takeover/status")
async def takeover_status() -> dict:
    """Return the current fallback decision so the UI can complete the timeout."""
    decision = system.fallback.evaluate(
        _last_state,
        driver_present=_last_driver_present,
    )
    payload = decision_to_dict(decision)
    await _broadcast({"fallback": payload})
    return payload

@app.post("/api/takeover/ack")
async def acknowledge_takeover() -> dict:
    system.acknowledge_takeover()
    return {"status":"acknowledged"}

RESULTS_DIR = Path(__file__).resolve().parent.parent / "mcb_experiments" / "results_v4"

class CalibrationUpdateRequest(BaseModel):
    alpha: float = 0.50
    beta: float = 0.35
    gamma: float = 0.15
    tau_fam: float = 0.05
    theta_trip: float = 0.30
    theta_warn: Optional[float] = None

@app.get("/api/calibration")
def get_calibration() -> dict:
    active = {
        "alpha": round(float(existing_cb.config.alpha), 3),
        "beta": round(float(existing_cb.config.beta), 3),
        "gamma": round(float(existing_cb.config.gamma), 3),
        "tau_fam": round(float(existing_cb.config.tau_fam), 3),
        "theta_trip": round(float(existing_cb.config.theta_trip), 4),
        "theta_warn": round(float(existing_cb.config.theta_warn), 4),
    }

    presets = {
        "op_j": {
            "id": "op_j",
            "name": "Optimal Youden's J (mcb_op_j)",
            "badge": "Maximum Balanced Detection",
            "tau_fam": 0.05,
            "theta_trip": 0.30,
            "theta_warn": 0.150,
            "alpha": 0.50,
            "beta": 0.35,
            "gamma": 0.15,
            "recall": 0.8416,
            "far": 0.1274,
            "youden_j": 0.7143,
            "description": "Selected on Partition A grid sweep. Maximizes J = Recall - FAR (84.2% recall at 12.7% FAR)."
        },
        "op_far10": {
            "id": "op_far10",
            "name": "High-Precision Safety (mcb_op_far10)",
            "badge": "Strict Alert Fatigue Limit (FAR <= 10%)",
            "tau_fam": 0.10,
            "theta_trip": 0.25,
            "theta_warn": 0.125,
            "alpha": 0.50,
            "beta": 0.35,
            "gamma": 0.15,
            "recall": 0.7828,
            "far": 0.0936,
            "youden_j": 0.6892,
            "description": "Enforces strict false alarm bound <= 10% (78.3% recall at 9.4% FAR on Partition B)."
        },
        "prior_default": {
            "id": "prior_default",
            "name": "Legacy Heuristic (v3 Default)",
            "badge": "Prior Heuristic",
            "tau_fam": 0.35,
            "theta_trip": 0.55,
            "theta_warn": 0.275,
            "alpha": 0.50,
            "beta": 0.35,
            "gamma": 0.15,
            "recall": 0.2851,
            "far": 0.0032,
            "youden_j": 0.2819,
            "description": "Pre-v4 heuristic settings. Ultra-conservative FAR (0.32%) but misses 71.5% of model errors."
        }
    }

    grid_points = []
    grid_csv = RESULTS_DIR / "grid_search_A.csv"
    if grid_csv.exists():
        try:
            import pandas as pd
            gdf = pd.read_csv(grid_csv)
            gdf = gdf.sort_values(by="youden_j", ascending=False).head(25)
            for _, r in gdf.iterrows():
                grid_points.append({
                    "tau_fam": round(float(r["tau_fam"]), 2),
                    "theta_trip": round(float(r["theta_trip"]), 2),
                    "theta_warn": round(float(r["theta_warn"]), 3),
                    "recall": round(float(r["recall"]), 4),
                    "far": round(float(r["far"]), 4),
                    "youden_j": round(float(r["youden_j"]), 4),
                    "precision": round(float(r["precision"]), 4),
                })
        except Exception:
            pass

    charts = [
        {"id": "calibration_heatmap", "title": "Grid Search Heatmap", "filename": "calibration_heatmap.png", "desc": "Youden's J across (tau, theta) grid on Partition A"},
        {"id": "roc_curves", "title": "ROC Curves", "filename": "roc_curves.png", "desc": "Partition B ROC curves comparing MCB vs all 10 baselines"},
        {"id": "risk_coverage", "title": "Risk-Coverage Tradeoff", "filename": "risk_coverage.png", "desc": "Residual risk as coverage increases on Partition B"},
        {"id": "family_radar", "title": "Family Vulnerability Radar", "filename": "family_radar.png", "desc": "Metamorphic instability profile across geometric, photometric, and noise"},
        {"id": "latency_breakdown", "title": "Latency Breakdown", "filename": "latency_breakdown.png", "desc": "Component runtime breakdown on Intel CPU"},
        {"id": "fault_attribution", "title": "Fault Attribution Matrix", "filename": "fault_attribution_confusion.png", "desc": "Attribution confusion matrix under unseen fault injections"},
    ]

    return {
        "active_config": active,
        "presets": presets,
        "grid_points": grid_points,
        "charts": charts,
    }

@app.post("/api/calibration/update")
async def update_calibration(req: CalibrationUpdateRequest) -> dict:
    total_weights = req.alpha + req.beta + req.gamma
    if abs(total_weights - 1.0) > 0.015:
        raise HTTPException(
            status_code=400,
            detail=f"Weights must sum to 1.0 (got alpha={req.alpha}, beta={req.beta}, gamma={req.gamma}, sum={total_weights:.3f})"
        )

    t_trip = req.theta_trip
    t_warn = req.theta_warn if req.theta_warn is not None else round(0.5 * t_trip, 4)
    if t_warn >= t_trip:
        raise HTTPException(
            status_code=400,
            detail=f"theta_warn ({t_warn}) must be strictly less than theta_trip ({t_trip})"
        )

    try:
        new_config = CircuitBreakerConfig(
            alpha=round(req.alpha, 4),
            beta=round(req.beta, 4),
            gamma=round(req.gamma, 4),
            tau_fam=round(req.tau_fam, 4),
            theta_trip=round(t_trip, 4),
            theta_warn=round(t_warn, 4),
        )
        existing_cb.config = new_config
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = {
        "type": "config_updated",
        "active_config": {
            "alpha": new_config.alpha,
            "beta": new_config.beta,
            "gamma": new_config.gamma,
            "tau_fam": new_config.tau_fam,
            "theta_trip": new_config.theta_trip,
            "theta_warn": new_config.theta_warn,
        }
    }
    await _broadcast(payload)
    return {"status": "ok", "active_config": payload["active_config"]}

@app.get("/api/calibration/chart/{chart_name}")
def get_calibration_chart(chart_name: str):
    allowed = {
        "calibration_heatmap.png",
        "roc_curves.png",
        "risk_coverage.png",
        "family_radar.png",
        "latency_breakdown.png",
        "fault_attribution_confusion.png"
    }
    fname = chart_name if chart_name.endswith(".png") else f"{chart_name}.png"
    if fname not in allowed:
        raise HTTPException(status_code=404, detail="Chart not found")

    chart_file = RESULTS_DIR / fname
    if not chart_file.exists():
        raise HTTPException(status_code=404, detail=f"File {fname} not found in results directory")

    return FileResponse(chart_file, media_type="image/png")

@app.websocket("/ws/fallback")
async def fallback_websocket(websocket: WebSocket) -> None:
    await websocket.accept(); _connections.add(websocket)
    try:
        while True:
            if await websocket.receive_text() == "ping":
                await websocket.send_text(json.dumps({"type":"pong"}))
    except WebSocketDisconnect: _connections.discard(websocket)
    except Exception: _connections.discard(websocket)

async def _broadcast(payload: dict) -> None:
    message=json.dumps(payload); dead=[]
    for connection in _connections:
        try: await connection.send_text(message)
        except Exception: dead.append(connection)
    for connection in dead: _connections.discard(connection)
