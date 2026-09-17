"""
src/circuit_breaker.py — Hierarchical Metamorphic Circuit Breaker Engine.

Implements the complete Phase 2 decision pipeline:
1. Level 1 Normalization: Computes Score_m for each transformation type.
2. Level 2 Normalization: Computes unweighted family average IR_k to eliminate skew.
3. Cross-Family Spread Score: CFS(x) based on tau_fam = 0.35 threshold.
4. Composite Brittleness Index: CBI(x) = alpha * max(IR_k) + beta * CFS + gamma * (1 - c0).
5. Three-State Machine: CLOSED (Normal), HALF_OPEN (Monitor), OPEN (Tripped).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from PIL import Image

from src.models.base import BaseVisionModel
from src.metamorphic_families import (
    HIERARCHICAL_TEST_MATRIX,
    FAMILIES,
    MetamorphicTest,
)


class CircuitBreakerState(str, Enum):
    CLOSED = "CLOSED"         # Normal operation: automated output approved
    HALF_OPEN = "HALF_OPEN"   # Monitor state: prediction approved with warning flag
    OPEN = "OPEN"             # Tripped: automated prediction blocked, routed to doctor


@dataclass
class CircuitBreakerConfig:
    """Configurable weights and thresholds for the circuit breaker."""
    alpha: float = 0.50        # Weight for peak family instability
    beta: float = 0.35         # Weight for cross-family spread score
    gamma: float = 0.15        # Weight for baseline model uncertainty (1 - c0)
    tau_fam: float = 0.35      # Family compromise threshold (35%)
    theta_warn: float = 0.25   # Warning threshold (enters HALF_OPEN)
    theta_trip: float = 0.55   # Trip threshold (enters OPEN)

    def __post_init__(self) -> None:
        total_weights = self.alpha + self.beta + self.gamma
        if abs(total_weights - 1.0) > 1e-5:
            raise ValueError(f"alpha + beta + gamma must equal 1.0, got {total_weights:.4f}")


@dataclass
class SingleTestResult:
    test_id: str
    test_name: str
    family: str
    transform_type: str
    severity: str
    weight: float
    pred_idx: int
    pred_label: str
    confidence: float
    flipped: bool


@dataclass
class DiagnosticReport:
    """Complete, transparent diagnostic report emitted for each query image."""
    baseline_idx: int
    baseline_label: str
    baseline_confidence: float
    test_results: List[SingleTestResult]
    type_scores: Dict[str, float]           # Level 1: Score_m per transformation type
    family_instability: Dict[str, float]    # Level 2: IR_k per family (unskewed)
    peak_family: str
    peak_family_score: float
    cross_family_spread: float             # CFS score
    compromised_families: List[str]
    cbi: float                              # Composite Brittleness Index [0.0, 1.0]
    state: CircuitBreakerState
    action: str                             # "AUTO_APPROVE" | "WARN_AND_PASS" | "TRIP_TO_DOCTOR"
    details: Dict[str, Any] = field(default_factory=dict)


class MetamorphicCircuitBreaker:
    """
    Model-agnostic Hierarchical Metamorphic Circuit Breaker Engine.
    """

    def __init__(
        self,
        model: BaseVisionModel,
        config: Optional[CircuitBreakerConfig] = None,
        test_matrix: Optional[List[MetamorphicTest]] = None,
    ) -> None:
        self.model = model
        self.config = config or CircuitBreakerConfig()
        self.test_matrix = test_matrix or HIERARCHICAL_TEST_MATRIX

    def evaluate(self, image: Image.Image) -> DiagnosticReport:
        """
        Executes the hierarchical metamorphic evaluation pipeline on an image.

        Args:
            image: PIL RGB Image.

        Returns:
            DiagnosticReport containing all mathematical scores and breaker decision.
        """
        # ── Step 1: Baseline inference on original image ──────────────────────
        base_idx, base_label, base_conf, all_probs = self.model.predict(image)

        # ── Step 2: Execute test matrix ───────────────────────────────────────
        test_results: List[SingleTestResult] = []
        for test in self.test_matrix:
            try:
                t_img = test(image)
                t_idx, t_label, t_conf, _ = self.model.predict(t_img)
            except Exception as exc:
                # If a transform fails computationally, mark as non-flipped to avoid false panic
                t_idx, t_label, t_conf = base_idx, base_label, base_conf

            flipped = (t_idx != base_idx)
            test_results.append(
                SingleTestResult(
                    test_id=test.id,
                    test_name=test.name,
                    family=test.family,
                    transform_type=test.transform_type,
                    severity=test.severity,
                    weight=test.weight,
                    pred_idx=t_idx,
                    pred_label=t_label,
                    confidence=t_conf,
                    flipped=flipped,
                )
            )

        # ── Step 3: Level 1 Normalization (Score_m per transform type) ────────
        # Group tests by (family, transform_type)
        type_groups: Dict[tuple, List[SingleTestResult]] = {}
        for r in test_results:
            key = (r.family, r.transform_type)
            type_groups.setdefault(key, []).append(r)

        type_scores: Dict[str, float] = {}
        family_to_types: Dict[str, List[str]] = {f: [] for f in FAMILIES}

        for (fam, t_type), tests in type_groups.items():
            weighted_failures = sum(t.weight for t in tests if t.flipped)
            total_possible_weights = sum(t.weight for t in tests)
            score_m = (weighted_failures / total_possible_weights) if total_possible_weights > 0 else 0.0

            type_scores[t_type] = round(score_m, 4)
            if t_type not in family_to_types[fam]:
                family_to_types[fam].append(t_type)

        # ── Step 4: Level 2 Normalization (Unweighted family average IR_k) ────
        family_instability: Dict[str, float] = {}
        for fam in FAMILIES:
            t_types = family_to_types[fam]
            if t_types:
                ir_k = sum(type_scores[t] for t in t_types) / len(t_types)
            else:
                ir_k = 0.0
            family_instability[fam] = round(ir_k, 4)

        # Peak family calculation
        peak_fam = max(family_instability, key=lambda k: family_instability[k])
        peak_fam_score = family_instability[peak_fam]

        # ── Step 5: Cross-Family Spread Score (CFS) ───────────────────────────
        compromised = [
            fam for fam, ir in family_instability.items()
            if ir >= self.config.tau_fam
        ]
        cfs = len(compromised) / len(FAMILIES) if FAMILIES else 0.0

        # ── Step 6: Composite Brittleness Index (CBI) ─────────────────────────
        baseline_uncertainty = 1.0 - base_conf
        cbi = (
            self.config.alpha * peak_fam_score
            + self.config.beta * cfs
            + self.config.gamma * baseline_uncertainty
        )
        cbi = max(0.0, min(1.0, round(cbi, 4)))

        # ── Step 7: Circuit Breaker State Transition ──────────────────────────
        if cbi < self.config.theta_warn:
            state = CircuitBreakerState.CLOSED
            action = "AUTO_APPROVE"
        elif cbi < self.config.theta_trip:
            state = CircuitBreakerState.HALF_OPEN
            action = "WARN_AND_PASS"
        else:
            state = CircuitBreakerState.OPEN
            action = "TRIP_TO_DOCTOR"

        return DiagnosticReport(
            baseline_idx=base_idx,
            baseline_label=base_label,
            baseline_confidence=round(base_conf, 4),
            test_results=test_results,
            type_scores=type_scores,
            family_instability=family_instability,
            peak_family=peak_fam,
            peak_family_score=peak_fam_score,
            cross_family_spread=round(cfs, 4),
            compromised_families=compromised,
            cbi=cbi,
            state=state,
            action=action,
            details={
                "config": {
                    "alpha": self.config.alpha,
                    "beta": self.config.beta,
                    "gamma": self.config.gamma,
                    "tau_fam": self.config.tau_fam,
                    "theta_warn": self.config.theta_warn,
                    "theta_trip": self.config.theta_trip,
                },
                "baseline_probs": all_probs,
            }
        )

    def evaluate_with_gradcam(
        self,
        image: Image.Image,
        max_gradcam_flips: int = 3,
    ) -> "tuple[DiagnosticReport, dict]":
        """
        Full evaluation pipeline extended with Grad-CAM attention maps.

        Runs :meth:`evaluate`, then generates:
        - A Grad-CAM heatmap on the **baseline** image.
        - A Grad-CAM heatmap for each **flipped** test (up to *max_gradcam_flips*),
          enabling visual attention-shift comparison in the UI.

        Args:
            image:              PIL RGB Image to evaluate.
            max_gradcam_flips:  Maximum number of flipped tests to explain
                                (most impactful by severity weight, then alphabetically).

        Returns:
            (report, gradcam_data) where ``gradcam_data`` is a dict with keys:
            - ``"baseline_cam"``: float32 ndarray (H, W) or None.
            - ``"baseline_image"``: original PIL image.
            - ``"flip_cams"``: List[dict] with keys:
              ``test_id``, ``test_name``, ``family``, ``severity``,
              ``transformed_image``, ``cam``, ``flipped_label``, ``flipped_confidence``.
        """
        report = self.evaluate(image)

        gradcam_data: dict = {
            "baseline_cam": None,
            "baseline_image": image,
            "flip_cams": [],
        }

        # ── Baseline Grad-CAM ─────────────────────────────────────────────────
        try:
            baseline_cam = self.model.explain(image, target_class=report.baseline_idx)
            gradcam_data["baseline_cam"] = baseline_cam
        except Exception:
            pass

        # ── Per-flip Grad-CAM for most impactful flipped tests ────────────────
        # Sort flipped results: prioritise mild (highest weight) then test_id
        flipped_results = sorted(
            [r for r in report.test_results if r.flipped],
            key=lambda r: (-r.weight, r.test_id),
        )

        for test_result in flipped_results[:max_gradcam_flips]:
            # Re-run the transform to get the transformed image
            matched = [t for t in self.test_matrix if t.id == test_result.test_id]
            if not matched:
                continue
            test_fn = matched[0]
            try:
                t_img = test_fn(image)
                flip_cam = self.model.explain(t_img, target_class=test_result.pred_idx)
                gradcam_data["flip_cams"].append({
                    "test_id": test_result.test_id,
                    "test_name": test_result.test_name,
                    "family": test_result.family,
                    "severity": test_result.severity,
                    "weight": test_result.weight,
                    "transformed_image": t_img,
                    "cam": flip_cam,
                    "flipped_label": test_result.pred_label,
                    "flipped_confidence": test_result.confidence,
                })
            except Exception:
                continue

        return report, gradcam_data

