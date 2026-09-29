"""Safe integration boundary for the existing Metamorphic Circuit Breaker.

This module is intentionally additive. It does not alter MetamorphicCircuitBreaker,
its thresholds, metamorphic relations, model, or state-transition logic.

Use evaluate_with_fallback() from a new application/UI layer when fallback behavior
is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image

from src.circuit_breaker import DiagnosticReport, MetamorphicCircuitBreaker
from .controller import FallbackController, FallbackDecision


@dataclass
class FallbackEvaluation:
    """Original framework report plus the independent fallback decision."""

    report: DiagnosticReport
    fallback: FallbackDecision


class CircuitBreakerWithFallback:
    """Composition wrapper: existing CB first, fallback second."""

    def __init__(
        self,
        circuit_breaker: MetamorphicCircuitBreaker,
        fallback: Optional[FallbackController] = None,
    ) -> None:
        self.circuit_breaker = circuit_breaker
        self.fallback = fallback or FallbackController()

    def evaluate(
        self,
        image: Image.Image,
        *,
        driver_present: bool = True,
    ) -> FallbackEvaluation:
        # The existing framework performs the complete evaluation unchanged.
        report = self.circuit_breaker.evaluate(image)

        # Fallback only consumes the state emitted by that framework.
        decision = self.fallback.evaluate(
            report.state,
            driver_present=driver_present,
        )

        return FallbackEvaluation(report=report, fallback=decision)

    def acknowledge_takeover(self) -> None:
        self.fallback.acknowledge_takeover()
