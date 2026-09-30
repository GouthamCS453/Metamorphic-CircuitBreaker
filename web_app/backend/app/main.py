"""
web_app/backend/app/main.py
FastAPI application entry point.
- Loads ConvNeXt model once on startup (singleton in circuit_service)
- Mounts all routers
- Enables CORS for React dev server (localhost:5173)
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parents[4]  # main_implementation/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.routers import predict, cases, presets, config, simulate


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the ConvNeXt model once so first request is fast
    print("[*] Loading ConvNeXt-Base model checkpoint (~1 GB)...")
    from app.services.circuit_service import load_model
    load_model()
    print("[OK] Model loaded. Server ready.")
    yield
    print("[*] Server shutting down.")


app = FastAPI(
    title="Aegis MetroHealth — Metamorphic Circuit Breaker API",
    description="Clinical AI safety layer for ISIC skin lesion classification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(cases.router)
app.include_router(presets.router)
app.include_router(config.router)
app.include_router(simulate.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "hospital": "Aegis MetroHealth", "service": "Metamorphic CB API v1.0"}
