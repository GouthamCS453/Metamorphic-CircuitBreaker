"""FastAPI API for the additive vehicle fallback demonstration.

This layer composes the existing Metamorphic Circuit Breaker with the
fallback controller. It does not modify the framework itself.
"""

from __future__ import annotations

import io
import json
from typing import Set

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.circuit_breaker import MetamorphicCircuitBreaker
from src.models.mobilenet_adapter import MobileNetAdapter
from fallback.controller import decision_to_dict
from fallback.integration import CircuitBreakerWithFallback

app = FastAPI(
    title="Metamorphic Circuit Breaker Fallback API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = MobileNetAdapter()
existing_cb = MetamorphicCircuitBreaker(model)
system = CircuitBreakerWithFallback(existing_cb)

_connections: Set[WebSocket] = set()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "fallback-api"}


@app.post("/api/evaluate")
async def evaluate(
    image: UploadFile = File(...),
    driver_present: bool = True,
) -> dict:
    """Run the unchanged circuit breaker, then apply the independent fallback layer."""
    try:
        raw = await image.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    result = system.evaluate(
        pil_image,
        driver_present=driver_present,
    )

    payload = {
        "prediction": {
            "index": result.report.baseline_idx,
            "label": result.report.baseline_label,
            "confidence": result.report.baseline_confidence,
        },
        "circuit_breaker": {
            "state": result.report.state.value,
            "action": result.report.action,
            "cbi": result.report.cbi,
            "peak_family": result.report.peak_family,
            "peak_family_score": result.report.peak_family_score,
            "cross_family_spread": result.report.cross_family_spread,
            "compromised_families": result.report.compromised_families,
        },
        "fallback": decision_to_dict(result.fallback),
    }

    await _broadcast(payload)
    return payload


@app.post("/api/takeover/ack")
async def acknowledge_takeover() -> dict:
    system.acknowledge_takeover()
    return {"status": "acknowledged"}


@app.websocket("/ws/fallback")
async def fallback_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    _connections.add(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        _connections.discard(websocket)
    except Exception:
        _connections.discard(websocket)


async def _broadcast(payload: dict) -> None:
    message = json.dumps(payload)
    dead = []
    for connection in _connections:
        try:
            await connection.send_text(message)
        except Exception:
            dead.append(connection)
    for connection in dead:
        _connections.discard(connection)
