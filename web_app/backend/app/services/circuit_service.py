"""
web_app/backend/app/services/circuit_service.py
Bridge between FastAPI routes and the src/ ML pipeline.
Loads ConvNeXtAdapter once and reuses it across all requests.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image

from app.config import get_active_config, PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.convnext_adapter import ConvNeXtAdapter
from src.circuit_breaker import MetamorphicCircuitBreaker, DiagnosticReport


# ── Singleton model ────────────────────────────────────────────────────────────

_model: Optional[ConvNeXtAdapter] = None


def load_model() -> ConvNeXtAdapter:
    global _model
    if _model is None:
        _model = ConvNeXtAdapter()
    return _model


def get_model() -> ConvNeXtAdapter:
    return load_model()


# ── Inference ──────────────────────────────────────────────────────────────────

def run_circuit_breaker(
    image: Image.Image,
) -> Tuple[DiagnosticReport, Dict]:
    """
    Run the full metamorphic circuit breaker pipeline on an image.
    Uses the currently-active CBConfig (which can be changed via /api/config).

    Returns:
        (report, gradcam_data) — raw Python ML objects.
    """
    model = get_model()
    cfg = get_active_config().to_src_config()
    cb = MetamorphicCircuitBreaker(model, config=cfg)

    # Quick evaluation first to decide whether GradCAM is needed
    report_quick = cb.evaluate(image)
    n_flips = sum(1 for r in report_quick.test_results if r.flipped)

    if n_flips > 0 or report_quick.state.value in ("HALF_OPEN", "OPEN"):
        # Full GradCAM run (up to 3 flip explanations)
        report, gradcam_data = cb.evaluate_with_gradcam(image, max_gradcam_flips=3)
    else:
        # Just baseline GradCAM
        baseline_cam = model.explain(image, target_class=report_quick.baseline_idx)
        report = report_quick
        gradcam_data = {
            "baseline_cam": baseline_cam,
            "baseline_image": image,
            "flip_cams": [],
        }

    return report, gradcam_data
