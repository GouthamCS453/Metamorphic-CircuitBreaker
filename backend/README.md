# FastAPI Fallback API

This is an additive interface layer for the vehicle fallback demonstration.

It does **not** replace or modify the existing:

- Metamorphic Relations
- Circuit Breaker
- MobileNet adapter
- Model
- Streamlit application

## Start

From the repository root:

```bat
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /api/health`
- `POST /api/evaluate`
- `POST /api/takeover/ack`
- `WS /ws/fallback`

The evaluation flow is:

```text
Image
  -> Existing MobileNet adapter
  -> Existing Metamorphic Circuit Breaker
  -> Fallback controller
  -> JSON response / WebSocket event
```

The `SAFE_PULL_OVER` result is currently a simulated safety decision/event. It does not directly control a physical vehicle.
