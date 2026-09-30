# Aegis MetroHealth — Clinical AI Web Application

**React + FastAPI web application** demonstrating the Metamorphic Circuit Breaker framework on ISIC skin lesion classification with ConvNeXt-Base, presented as a realistic hospital clinical system.

This is a **separate** application from the Streamlit prototype (`streamlit_app.py`) and runs independently alongside it.

---

## Overview

The application simulates the **Aegis MetroHealth Centre for Advanced Dermatology & Cutaneous Oncology** — a hospital AI screening platform with:

- **Patient Portal** — Upload a lesion image, run 22 metamorphic safety tests, receive a verified AI diagnosis or specialist escalation notice.
- **Doctor/Physician Portal** — Triage queue of cases requiring review, side-by-side Grad-CAM attention evidence, and formal clinical override/approval capability.
- **Calibration Studio** — Real-time parameter sliders for circuit breaker weights and thresholds, what-if scenario simulator, and one-click clinical risk profiles.
- **Case Registry** — Full audit trail of all screenings with circuit breaker outcomes.

---

## Project Structure

```
main_implementation/
|-- web_app/
|   |-- backend/
|   |   |-- app/
|   |   |   |-- __init__.py
|   |   |   |-- main.py              # FastAPI app, CORS, lifespan model loader
|   |   |   |-- config.py            # CB parameter store + clinical profiles
|   |   |   |-- schemas.py           # Pydantic request/response models
|   |   |   |-- storage.py           # JSON case repository
|   |   |   |-- routers/
|   |   |   |   |-- predict.py       # POST /api/predict
|   |   |   |   |-- cases.py         # GET/POST /api/cases
|   |   |   |   |-- presets.py       # GET /api/presets
|   |   |   |   |-- config.py        # GET/POST /api/config
|   |   |   |   |-- simulate.py      # POST /api/simulate
|   |   |   |-- services/
|   |   |       |-- circuit_service.py   # Bridge to src/ ML pipeline
|   |   |-- data/
|   |   |   |-- cases.json           # Persisted case records (auto-created)
|   |   |-- uploads/                 # Reserved for future uploads (auto-created)
|   |
|   |-- frontend/                    # React + Vite + Tailwind CSS application
|   |   |-- src/
|   |   |   |-- api/client.js        # Axios API client
|   |   |   |-- context/RoleContext.jsx
|   |   |   |-- components/
|   |   |   |   |-- Navbar.jsx
|   |   |   |   |-- Footer.jsx
|   |   |   |   |-- StateBadge.jsx
|   |   |   |   |-- CbiGauge.jsx
|   |   |   |-- pages/
|   |   |       |-- HomePage.jsx
|   |   |       |-- PatientPortalPage.jsx
|   |   |       |-- DoctorPortalPage.jsx
|   |   |       |-- CalibrationPage.jsx
|   |   |       |-- CaseRegistryPage.jsx
|   |   |-- App.jsx
|   |   |-- main.jsx
|   |   |-- index.css
|   |
|   |-- run_server.py                # Python launcher for FastAPI
|   |-- start_backend.bat            # Windows quick-start for backend
|   |-- start_frontend.bat           # Windows quick-start for frontend
|
|-- src/                             # Existing ML pipeline (shared)
|   |-- circuit_breaker.py
|   |-- gradcam.py
|   |-- models/convnext_adapter.py
|   |-- metamorphic_families.py
|   |-- utils.py
|
|-- checkpoints/best_model.pth       # Trained ConvNeXt-Base (~1 GB)
|-- data/presets/                    # ISIC demo presets
|-- streamlit_app.py                 # Original Streamlit showcase (separate)
```

---

## Prerequisites

- Python 3.10+ (the project uses `.venv` in `main_implementation/`)
- Node.js 18+ and npm
- The trained checkpoint at `checkpoints/best_model.pth`

---

## Setup — First Time Only

### 1. Install Backend Dependencies

From the `main_implementation/` root directory:

```bat
.venv\Scripts\python.exe -m pip install fastapi uvicorn python-multipart
```

All other Python packages (torch, timm, Pillow, opencv-python, etc.) should already be installed in the virtual environment.

### 2. Install Frontend Dependencies

```bat
cd web_app\frontend
npm install
cd ..\..
```

---

## Running the Application

The backend and frontend are two separate processes. Open two terminal windows, both starting from `main_implementation/`.

### Terminal 1 — FastAPI Backend (port 8000)

**Option A — Using the batch file (Windows):**
```bat
web_app\start_backend.bat
```

**Option B — Directly:**
```bat
.venv\Scripts\python.exe web_app\run_server.py
```

The server will print:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

> **Note:** The ConvNeXt-Base model (~1 GB) is loaded once on startup. This may take 15-45 seconds on the first start depending on your hardware. Subsequent requests are fast because the model stays resident in memory.

### Terminal 2 — React Frontend (port 5173)

**Option A — Using the batch file (Windows):**
```bat
web_app\start_frontend.bat
```

**Option B — Directly:**
```bat
cd web_app\frontend
npm run dev
```

The Vite dev server will print:
```
VITE ready in ~300 ms
Local: http://localhost:5173/
```

### Access the Application

Open your browser and go to:

```
http://localhost:5173
```

---

## API Endpoints (Backend)

The FastAPI backend exposes the following REST API. Interactive docs are available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Server health check |
| POST | `/api/predict` | Run CB on uploaded image or preset |
| GET | `/api/presets` | List available ISIC demo presets |
| GET | `/api/cases` | List all cases (filterable by status) |
| GET | `/api/cases/stats` | Dashboard statistics |
| GET | `/api/cases/{id}` | Get full case detail with GradCAM |
| POST | `/api/cases/{id}/review` | Submit doctor approval or override |
| GET | `/api/config` | Get active CB configuration |
| POST | `/api/config` | Update CB parameters |
| GET | `/api/config/profiles` | List clinical risk profiles |
| POST | `/api/config/profiles/{id}` | Apply a clinical risk profile |
| POST | `/api/simulate` | Run hypothetical CBI calculation |

---

## Using the Application

### Patient Screening Flow

1. Navigate to **Patient Screening** in the navbar.
2. Choose **Upload Image** (drag-and-drop or click) or **Demo Preset** to pick a verified ISIC case.
3. Optionally fill in patient clinical context (name, age, lesion site, symptoms).
4. Click **Run AI Screening**.
5. The system runs 22 metamorphic perturbation tests and returns one of three outcomes:
   - **CLOSED (Auto-Approved):** Diagnosis is stable. Result shown immediately.
   - **HALF-OPEN (Under Review):** Case forwarded to physician portal for secondary verification.
   - **OPEN (Safety Intercept):** AI prediction suppressed. Specialist escalation required.

### Doctor Triage Flow

1. Switch role to **Doctor** using the Role Switcher in the navbar (top right).
2. Navigate to **Physician Portal**.
3. Select a pending case from the triage queue.
4. Review:
   - Baseline Grad-CAM attention heatmap.
   - Attention drift heatmaps for each prediction flip.
   - CBI gauge and family instability breakdown.
5. Choose an action:
   - **Pass and Verify AI** (for HALF-OPEN cases) — Signs off on the AI result.
   - **Clinical Override** — Selects the correct ISIC class from the dropdown, adds clinical notes, sets biopsy recommendation and follow-up urgency.
6. Click **Sign and Verify** or **Submit Clinical Override**.
7. The patient portal case status updates to reflect the doctor's decision.

### Parameter Calibration

1. Switch role to **Admin** or navigate directly to **Calibration Studio**.
2. Adjust the CBI formula weights (alpha, beta, gamma) — must sum to 1.000.
3. Adjust decision thresholds (tau_fam, theta_warn, theta_trip).
4. Click **Apply Configuration** — changes are immediately live for all subsequent screenings.
5. Use the **What-If Scenario Simulator** to test hypothetical CBI values without running a real image.
6. Click **Apply** next to any **Clinical Risk Profile** to load a preset configuration.

### Case Registry

Navigate to **Case Registry** to view the full audit trail. Use the search box to filter by Case ID, diagnosis label, or circuit state.

---

## Demo Presets

Four verified ISIC images are pre-loaded to demonstrate each circuit breaker state:

| Preset | Expected State | Description |
|--------|---------------|-------------|
| Rock-Solid Lesion (0 Flips) | CLOSED | Zero flips across all tests. Instant auto-approval. |
| Boundary Sensitivity (4 Flips, but Safe) | CLOSED | Flips at severe boundary only; CBI stays below theta_warn. |
| Systematic Geometric Fragility (3 Mild Flips) | HALF-OPEN | Single-family fragility; amber warning raised. |
| Multi-Family Collapse (12 Flips - Tripped!) | OPEN | Multi-family collapse; CBI exceeds theta_trip; AI blocked. |

---

## Circuit Breaker Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| alpha | 0.50 | Weight for peak family instability (max IR_k) |
| beta | 0.35 | Weight for cross-family spread (CFS) |
| gamma | 0.15 | Weight for baseline model uncertainty (1 - c0) |
| tau_fam | 0.35 | Threshold at which a family is considered compromised |
| theta_warn | 0.25 | CBI threshold to enter HALF-OPEN state |
| theta_trip | 0.55 | CBI threshold to enter OPEN (tripped) state |

**CBI Formula:**

```
CBI(x) = alpha * max_k(IR_k) + beta * CFS(x) + gamma * (1 - c0)

CLOSED    if CBI < theta_warn
HALF-OPEN if theta_warn <= CBI < theta_trip
OPEN      if CBI >= theta_trip
```

---

## Notes

- **This web application is completely separate from `streamlit_app.py`.** Both can run simultaneously; they do not share state or ports.
- **Case data** is persisted to `web_app/backend/data/cases.json` and survives server restarts.
- **No GPU required.** The ConvNeXt model runs on CPU (inference is ~15-40 seconds per image on CPU for the full 22-test suite).
- **This is a research prototype** and is not intended for actual clinical use.
