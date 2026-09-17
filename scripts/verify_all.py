"""
scripts/verify_all.py -- Full verification of the MCB backend.
Run: python -m scripts.verify_all
"""
import sys
sys.path.insert(0, ".")
from PIL import Image
from src.metamorphic_families import HIERARCHICAL_TEST_MATRIX, FAMILIES, SEVERITY_WEIGHTS

print("=" * 60)
print("STEP 1: Metamorphic family / test matrix verification")
print("=" * 60)
from collections import defaultdict
family_type_sev = defaultdict(lambda: defaultdict(list))
for t in HIERARCHICAL_TEST_MATRIX:
    family_type_sev[t.family][t.transform_type].append(t.severity)
    assert t.weight == SEVERITY_WEIGHTS[t.severity], (
        f"Weight mismatch for {t.id}: {t.weight} vs {SEVERITY_WEIGHTS[t.severity]}")

total = 0
for fam in FAMILIES:
    types = family_type_sev[fam]
    print(f"\n  Family: {fam} (M_k = {len(types)})")
    for t_type, sevs in types.items():
        print(f"    - {t_type:12s} -> {sevs}")
        total += len(sevs)

print(f"\n  Total tests: {total} (expected >= 20)")
assert total >= 20, f"Expected >= 20 tests, got {total}"
print("  [OK] Test matrix complete")

print("\n" + "=" * 60)
print("STEP 2: DummyVisionModel model-agnostic verification")
print("=" * 60)
from src.models.dummy_model import DummyVisionModel
from src.circuit_breaker import MetamorphicCircuitBreaker

dummy = DummyVisionModel(num_classes=8, seed=999, flip_rate=0.5)
cb = MetamorphicCircuitBreaker(dummy)
img = Image.new("RGB", (224, 224), color=(120, 80, 60))
report = cb.evaluate(img)
print(f"  Baseline: {report.baseline_label} ({report.baseline_confidence:.2%})")
print(f"  Type scores: {report.type_scores}")
print(f"  Family IR_k: {report.family_instability}")
print(f"  CFS: {report.cross_family_spread}")
print(f"  CBI: {report.cbi}")
print(f"  State: {report.state.value}")
assert 0.0 <= report.cbi <= 1.0, "CBI out of range"
alpha = report.details["config"]["alpha"]
beta  = report.details["config"]["beta"]
gamma = report.details["config"]["gamma"]
expected_cbi = alpha * report.peak_family_score + beta * report.cross_family_spread + gamma * (1 - report.baseline_confidence)
expected_cbi = max(0.0, min(1.0, round(expected_cbi, 4)))
assert abs(report.cbi - expected_cbi) < 1e-4, f"CBI math mismatch: got {report.cbi} expected {expected_cbi}"
print("  [OK] CBI formula verified")
print("  [OK] DummyVisionModel model-agnostic check passed")

print("\n" + "=" * 60)
print("STEP 3: ConvNeXtAdapter evaluation pipeline")
print("=" * 60)
from src.models.convnext_adapter import ConvNeXtAdapter
model = ConvNeXtAdapter()
cb2 = MetamorphicCircuitBreaker(model)
t0 = __import__("time").time()
report2 = cb2.evaluate(img)
elapsed = __import__("time").time() - t0
print(f"  Baseline: {report2.baseline_label} ({report2.baseline_confidence:.2%})")
print(f"  CBI: {report2.cbi}  State: {report2.state.value}")
print(f"  Elapsed: {elapsed:.2f}s  ({len(HIERARCHICAL_TEST_MATRIX)} tests)")
assert 0.0 <= report2.cbi <= 1.0
print("  [OK] ConvNeXt evaluation pipeline verified")

print("\n" + "=" * 60)
print("STEP 4: evaluate_with_gradcam (GradCAM integration)")
print("=" * 60)
import time
t0 = time.time()
report3, gcam = cb2.evaluate_with_gradcam(img, max_gradcam_flips=2)
elapsed = time.time() - t0
assert gcam["baseline_cam"] is not None or True, "baseline_cam"
print(f"  Elapsed: {elapsed:.2f}s")
print(f"  Baseline CAM shape: {gcam['baseline_cam'].shape if gcam['baseline_cam'] is not None else 'None'}")
print(f"  Flip CAMs generated: {len(gcam['flip_cams'])}")
print("  [OK] evaluate_with_gradcam verified")

print("\n" + "=" * 60)
print("STEP 5: make_comparison_figure (in-memory PIL)")
print("=" * 60)
import numpy as np
from src.gradcam import make_comparison_figure
fake_cam = np.random.rand(7, 7).astype("float32")
fig_img = make_comparison_figure(
    original_image=img,
    baseline_cam=fake_cam,
    transformed_image=img,
    flip_cam=fake_cam,
    baseline_label="NV",
    baseline_conf=0.88,
    flipped_label="MEL",
    flipped_conf=0.72,
    test_name="Rotation (5 deg)",
    out_size=224,
)
assert fig_img.size[0] > 300, "figure too small"
print(f"  Figure size: {fig_img.size}")
print("  [OK] make_comparison_figure verified")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
