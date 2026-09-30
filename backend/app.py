"""FastAPI API for the additive vehicle fallback demonstration."""
from __future__ import annotations
import base64, io, json
from typing import Set
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from src.circuit_breaker import MetamorphicCircuitBreaker
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

@app.post("/api/takeover/ack")
async def acknowledge_takeover() -> dict:
    system.acknowledge_takeover()
    return {"status":"acknowledged"}

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
