#!/bin/bash
# start_backend.sh — Linux/Mac launcher
cd "$(dirname "$0")/.."
source .venv/bin/activate
cd web_app
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
