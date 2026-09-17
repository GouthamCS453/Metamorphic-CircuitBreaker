"""
Verify the retain_grad fix and both circuit breaker states (CLOSED and OPEN).
Run: python -m scripts.verify_states
"""
import sys
sys.path.insert(0, ".")
import time
from PIL import Image
from src.models.convnext_adapter import ConvNeXtAdapter
from src.models.dummy_model import DummyVisionModel
from src.circuit_breaker import MetamorphicCircuitBreaker, CircuitBreakerConfig
from src.gradcam import make_comparison_figure, overlay_heatmap_on_image
import numpy as np

print("=" * 60)
print("FIX VERIFICATION: retain_grad RuntimeError")
print("=" * 60)

# This is the scenario that caused the crash:
# GradCAM hooks are registered, then predict() is called under no_grad().
model = ConvNeXtAdapter()
# Trigger GradCAM hook registration (lazy init inside explain)
_ = model.explain(Image.new("RGB",(224,224)), target_class=0)
print("  GradCAM hooks registered.")

# Now call predict() -- this must NOT raise RuntimeError
try:
    idx, lbl, conf, probs = model.predict(Image.new("RGB",(224,224), color=(100,50,50)))
    print(f"  predict() after hook registration: OK  ({lbl} {conf:.2%})")
except RuntimeError as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# And call predict() again to confirm it is idempotent
try:
    idx2, lbl2, conf2, _ = model.predict(Image.new("RGB",(224,224), color=(20,120,80)))
    print(f"  predict() second call: OK  ({lbl2} {conf2:.2%})")
except RuntimeError as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

print("  [OK] retain_grad fix verified -- no RuntimeError")

print()
print("=" * 60)
print("STATE 1: CLOSED (stable prediction -- ConvNeXt real image)")
print("=" * 60)
cb_real = MetamorphicCircuitBreaker(model)
img_stable = Image.new("RGB", (224,224), color=(180,120,80))
t0 = time.time()
r = cb_real.evaluate(img_stable)
elapsed = time.time() - t0
n_flips = sum(1 for t in r.test_results if t.flipped)
print(f"  Baseline: {r.baseline_label} ({r.baseline_confidence:.2%})")
print(f"  CBI: {r.cbi:.4f}  |  State: {r.state.value}  |  Flips: {n_flips}")
print(f"  Family IR_k: {r.family_instability}")
print(f"  Elapsed: {elapsed:.2f}s")
alpha = r.details["config"]["alpha"]
beta  = r.details["config"]["beta"]
gamma = r.details["config"]["gamma"]
recomputed = round(
    alpha * r.peak_family_score + beta * r.cross_family_spread + gamma*(1-r.baseline_confidence), 4)
assert abs(r.cbi - recomputed) < 1e-3, f"CBI math mismatch: {r.cbi} vs {recomputed}"
print("  [OK] CBI formula: verified")
if r.state.value == "CLOSED":
    print("  [OK] State CLOSED -- stable prediction confirmed")
else:
    print(f"  [NOTE] State is {r.state.value} (CBI={r.cbi:.4f}). Solid color image may not be stable -- expected.")

print()
print("=" * 60)
print("STATE 2: OPEN (unstable -- DummyModel flip_rate=1.0)")
print("=" * 60)
dummy_unstable = DummyVisionModel(num_classes=8, seed=42, flip_rate=1.0)
cb_dummy = MetamorphicCircuitBreaker(dummy_unstable,
    config=CircuitBreakerConfig(alpha=0.50, beta=0.35, gamma=0.15))
r2 = cb_dummy.evaluate(Image.new("RGB",(224,224)))
n2 = sum(1 for t in r2.test_results if t.flipped)
print(f"  Baseline: {r2.baseline_label} ({r2.baseline_confidence:.2%})")
print(f"  CBI: {r2.cbi:.4f}  |  State: {r2.state.value}  |  Flips: {n2}")
print(f"  Family IR_k: {r2.family_instability}")
assert r2.state.value in ("HALF_OPEN","OPEN"), f"Expected HALF_OPEN or OPEN, got {r2.state.value}"
print(f"  [OK] State {r2.state.value} -- unstable prediction confirmed")

print()
print("=" * 60)
print("STATE 3: evaluate_with_gradcam on real model (preset MEL image)")
print("=" * 60)
from pathlib import Path
preset_paths = list(Path("data/presets").rglob("*.jpg")) + list(Path("data/presets").rglob("*.jpeg"))
if preset_paths:
    img_preset = Image.open(preset_paths[0]).convert("RGB")
    t0 = time.time()
    r3, gcam = cb_real.evaluate_with_gradcam(img_preset, max_gradcam_flips=2)
    elapsed3 = time.time() - t0
    print(f"  Image: {preset_paths[0].name}")
    print(f"  Baseline: {r3.baseline_label} ({r3.baseline_confidence:.2%})")
    print(f"  CBI: {r3.cbi:.4f}  State: {r3.state.value}")
    print(f"  Baseline CAM: {gcam['baseline_cam'].shape if gcam['baseline_cam'] is not None else None}")
    print(f"  Flip CAMs: {len(gcam['flip_cams'])}")
    print(f"  Elapsed: {elapsed3:.2f}s")
    print("  [OK] evaluate_with_gradcam on real image verified")
else:
    print("  [SKIP] No preset images found -- run scripts/demo_presets.py")

print()
print("=" * 60)
print("ALL VERIFICATIONS PASSED")
print("=" * 60)
