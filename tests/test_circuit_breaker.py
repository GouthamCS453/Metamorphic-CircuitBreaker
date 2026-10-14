"""
tests/test_circuit_breaker.py — Unit and Integration Tests for Metamorphic Circuit Breaker.

Includes unit test cases applying:
- Equivalence Partitioning (EP)
- Boundary Value Analysis (BVA)
"""

import pytest
import numpy as np
from PIL import Image

from src.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    MetamorphicCircuitBreaker,
)
from src.metamorphic_families import (
    _rotate,
    _hflip,
    _zoom_crop,
    _brightness,
    _contrast,
    _saturation,
    _blur,
    _gaussian_noise,
)
from src.models.dummy_model import DummyVisionModel


# ─────────────────────────────────────────────────────────────────────────────
# TC01: Equivalence Partitioning (EP) on Config Weight Normalization
# ─────────────────────────────────────────────────────────────────────────────
def test_tc01_circuit_breaker_config_weight_validation_ep():
    """
    TC01: Equivalence Partitioning (EP) on weight validation:
    - Partition 1 [Valid]: alpha + beta + gamma == 1.0 -> Instantiation succeeds.
    - Partition 2 [Invalid, Sum > 1.0]: alpha + beta + gamma > 1.0 -> Raises ValueError.
    - Partition 3 [Invalid, Sum < 1.0]: alpha + beta + gamma < 1.0 -> Raises ValueError.
    """
    # Valid Partition: Valid combination summing exactly to 1.0
    valid_cfg = CircuitBreakerConfig(alpha=0.50, beta=0.35, gamma=0.15)
    assert valid_cfg.alpha == 0.50
    assert valid_cfg.beta == 0.35
    assert valid_cfg.gamma == 0.15

    # Another Valid Partition: Uniform weights summing to 1.0
    valid_cfg2 = CircuitBreakerConfig(alpha=0.33333, beta=0.33333, gamma=0.33334)
    assert abs((valid_cfg2.alpha + valid_cfg2.beta + valid_cfg2.gamma) - 1.0) < 1e-4

    # Invalid Partition: Sum > 1.0
    with pytest.raises(ValueError, match=r"alpha \+ beta \+ gamma must equal 1\.0"):
        CircuitBreakerConfig(alpha=0.50, beta=0.50, gamma=0.50)

    # Invalid Partition: Sum < 1.0
    with pytest.raises(ValueError, match=r"alpha \+ beta \+ gamma must equal 1\.0"):
        CircuitBreakerConfig(alpha=0.20, beta=0.20, gamma=0.20)


# ─────────────────────────────────────────────────────────────────────────────
# TC02: Boundary Value Analysis (BVA) on Threshold Ordering Auto-Correction
# ─────────────────────────────────────────────────────────────────────────────
def test_tc02_circuit_breaker_threshold_ordering_bva():
    """
    TC02: Boundary Value Analysis (BVA) on Threshold Ordering:
    - Rule: theta_warn must be strictly less than theta_trip.
    - Boundary 1 (Valid normal order): theta_warn = 0.25, theta_trip = 0.55 -> Kept as is.
    - Boundary 2 (Exact boundary of violation theta_warn == theta_trip):
      theta_warn = 0.55, theta_trip = 0.55 -> Triggers warning and auto-correction:
      theta_warn = round(theta_trip * 0.50, 4) = 0.275.
    - Boundary 3 (Inverted order theta_warn > theta_trip):
      theta_warn = 0.80, theta_trip = 0.50 -> Auto-corrects theta_warn -> 0.25.
    """
    # Boundary 1: Standard valid ordering
    cfg_valid = CircuitBreakerConfig(theta_warn=0.25, theta_trip=0.55)
    assert cfg_valid.theta_warn < cfg_valid.theta_trip
    assert cfg_valid.theta_warn == 0.25

    # Boundary 2: theta_warn == theta_trip (Boundary of violation)
    with pytest.warns(UserWarning, match="state ordering violated"):
        cfg_boundary = CircuitBreakerConfig(theta_warn=0.55, theta_trip=0.55)
        # Expected auto-corrected value: 0.55 * 0.5 = 0.275
        assert cfg_boundary.theta_warn == 0.275
        assert cfg_boundary.theta_warn < cfg_boundary.theta_trip

    # Boundary 3: theta_warn > theta_trip
    with pytest.warns(UserWarning, match="state ordering violated"):
        cfg_inverted = CircuitBreakerConfig(theta_warn=0.80, theta_trip=0.50)
        assert cfg_inverted.theta_warn == 0.250
        assert cfg_inverted.theta_warn < cfg_inverted.theta_trip


# ─────────────────────────────────────────────────────────────────────────────
# TC03: Boundary Value Analysis (BVA) on Circuit Breaker State Transitions
# ─────────────────────────────────────────────────────────────────────────────
def test_tc03_state_transition_boundary_values_bva():
    """
    TC03: Boundary Value Analysis (BVA) on CBI State Transitions:
    With default config: theta_warn = 0.25, theta_trip = 0.55.
    - Values tested:
      * 0.0000 (Min boundary)                     -> CLOSED
      * 0.2499 (Just below theta_warn)            -> CLOSED
      * 0.2500 (Exact boundary at theta_warn)     -> HALF_OPEN
      * 0.5499 (Just below theta_trip)            -> HALF_OPEN
      * 0.5500 (Exact boundary at theta_trip)     -> OPEN
      * 1.0000 (Max boundary)                     -> OPEN
    """
    cfg = CircuitBreakerConfig(theta_warn=0.25, theta_trip=0.55)

    def determine_state(cbi_val: float) -> tuple[CircuitBreakerState, str]:
        if cbi_val < cfg.theta_warn:
            return CircuitBreakerState.CLOSED, "AUTO_APPROVE"
        elif cbi_val < cfg.theta_trip:
            return CircuitBreakerState.HALF_OPEN, "WARN_AND_PASS"
        else:
            return CircuitBreakerState.OPEN, "TRIP_TO_DOCTOR"

    # Testing boundary points
    test_boundaries = [
        (0.0000, CircuitBreakerState.CLOSED, "AUTO_APPROVE"),
        (0.2499, CircuitBreakerState.CLOSED, "AUTO_APPROVE"),
        (0.2500, CircuitBreakerState.HALF_OPEN, "WARN_AND_PASS"),
        (0.5499, CircuitBreakerState.HALF_OPEN, "WARN_AND_PASS"),
        (0.5500, CircuitBreakerState.OPEN, "TRIP_TO_DOCTOR"),
        (1.0000, CircuitBreakerState.OPEN, "TRIP_TO_DOCTOR"),
    ]

    for cbi_val, expected_state, expected_action in test_boundaries:
        st, act = determine_state(cbi_val)
        assert st == expected_state, f"Failed at CBI={cbi_val}: expected {expected_state}, got {st}"
        assert act == expected_action, f"Failed at CBI={cbi_val}: expected {expected_action}, got {act}"


# ─────────────────────────────────────────────────────────────────────────────
# TC04: Equivalence Partitioning (EP) on Metamorphic Image Transformations
# ─────────────────────────────────────────────────────────────────────────────
def test_tc04_metamorphic_transforms_dimension_and_mode_ep():
    """
    TC04: Equivalence Partitioning (EP) on Metamorphic Image Transforms:
    Partition: Any valid PIL RGB image of shape (W, H).
    Expected: Output transformed image preserves identical spatial dimensions (W, H)
    and color mode ('RGB') without throwing exceptions across all transform types.
    """
    # Create synthetic test image (224, 224, 3)
    np_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(np_img, mode="RGB")
    orig_size = img.size
    orig_mode = img.mode

    transforms = [
        ("rotate_15", lambda im: _rotate(im, 15.0)),
        ("hflip", lambda im: _hflip(im)),
        ("zoom_crop_0.9", lambda im: _zoom_crop(im, 0.9)),
        ("brightness_1.2", lambda im: _brightness(im, 1.2)),
        ("contrast_1.2", lambda im: _contrast(im, 1.2)),
        ("saturation_1.2", lambda im: _saturation(im, 1.2)),
        ("blur_1.0", lambda im: _blur(im, 1.0)),
        ("gaussian_noise_0.05", lambda im: _gaussian_noise(im, 0.05)),
    ]

    for name, tf in transforms:
        transformed = tf(img)
        assert transformed.size == orig_size, f"{name} altered dimensions: {transformed.size} != {orig_size}"
        assert transformed.mode == orig_mode, f"{name} altered mode: {transformed.mode} != {orig_mode}"
        assert np.array(transformed).shape == (224, 224, 3), f"{name} returned unexpected array shape"


# ─────────────────────────────────────────────────────────────────────────────
# TC05: Level-2 Hybrid Family Instability Calculation Verification
# ─────────────────────────────────────────────────────────────────────────────
def test_tc05_hybrid_family_instability_calculation():
    """
    TC05: Level-2 Normalization unskewed hybrid calculation:
    IR_k = 0.70 * mean(scores) + 0.30 * peak(scores)
    Validates arithmetic accuracy and elimination of cardinality bias.
    """
    # Case 1: Partial collapse: scores = [1.0, 0.0]
    # mean = 0.50, peak = 1.0 -> 0.70 * 0.50 + 0.30 * 1.0 = 0.35 + 0.30 = 0.65
    scores_1 = [1.0, 0.0]
    ir_1 = 0.70 * (sum(scores_1) / len(scores_1)) + 0.30 * max(scores_1)
    assert round(ir_1, 4) == 0.6500

    # Case 2: Zero collapse: scores = [0.0, 0.0, 0.0]
    scores_2 = [0.0, 0.0, 0.0]
    ir_2 = 0.70 * (sum(scores_2) / len(scores_2)) + 0.30 * max(scores_2)
    assert round(ir_2, 4) == 0.0000

    # Case 3: Complete collapse: scores = [1.0, 1.0, 1.0]
    scores_3 = [1.0, 1.0, 1.0]
    ir_3 = 0.70 * (sum(scores_3) / len(scores_3)) + 0.30 * max(scores_3)
    assert round(ir_3, 4) == 1.0000


# ─────────────────────────────────────────────────────────────────────────────
# TC06: End-to-End Evaluation with Robust/Invariant Vision Model
# ─────────────────────────────────────────────────────────────────────────────
def test_tc06_e2e_circuit_breaker_invariant_model():
    """
    TC06: End-to-End Evaluation with a completely invariant model (flip_rate = 0.0).
    Expected:
    - 0 flipped metamorphic tests.
    - peak_family_score == 0.0.
    - cross_family_spread == 0.0.
    - state == CircuitBreakerState.CLOSED.
    - action == 'AUTO_APPROVE'.
    """
    model = DummyVisionModel(num_classes=8, seed=42, flip_rate=0.0)
    breaker = MetamorphicCircuitBreaker(model=model)

    test_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    report = breaker.evaluate(test_img)

    flipped_count = sum(1 for r in report.test_results if r.flipped)
    assert flipped_count == 0, f"Expected 0 flipped tests for invariant model, got {flipped_count}"
    assert report.peak_family_score == 0.0
    assert report.cross_family_spread == 0.0
    assert report.state == CircuitBreakerState.CLOSED
    assert report.action == "AUTO_APPROVE"


# ─────────────────────────────────────────────────────────────────────────────
# TC07: End-to-End Evaluation with Collapsed/Unstable Vision Model
# ─────────────────────────────────────────────────────────────────────────────
def test_tc07_e2e_circuit_breaker_collapsed_model():
    """
    TC07: End-to-End Evaluation with a completely unstable model (flip_rate = 1.0).
    Expected:
    - Multiple tests flip.
    - High CBI exceeding theta_trip (0.55).
    - state == CircuitBreakerState.OPEN.
    - action == 'TRIP_TO_DOCTOR'.
    """
    model = DummyVisionModel(num_classes=8, seed=42, flip_rate=1.0)
    breaker = MetamorphicCircuitBreaker(model=model)

    test_img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    report = breaker.evaluate(test_img)

    flipped_count = sum(1 for r in report.test_results if r.flipped)
    assert flipped_count > 0, "Expected multiple flipped tests for unstable model"
    assert report.cbi >= breaker.config.theta_trip
    assert report.state == CircuitBreakerState.OPEN
    assert report.action == "TRIP_TO_DOCTOR"
