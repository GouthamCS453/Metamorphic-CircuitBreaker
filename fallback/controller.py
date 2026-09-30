"""Autonomous-vehicle fallback decision and escalation controller.

The existing MetamorphicCircuitBreaker remains the source of truth for
CLOSED/HALF_OPEN/OPEN. This module only consumes the detected state.

Policy:
- CLOSED: normal operation; no fallback action.
- HALF_OPEN: request driver takeover when a driver is present and start a 10-second window.
- OPEN: request driver takeover when a driver is present; otherwise, or after
  the 10-second takeover window expires, enter the minimum-risk maneuver.
- Driver occupancy is supplied by the caller; it is never inferred from ML.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import monotonic
from typing import Optional, Any


class FallbackAction(str, Enum):
    NORMAL = "NORMAL"
    TAKEOVER_REQUESTED = "TAKEOVER_REQUESTED"
    SAFE_PULL_OVER = "SAFE_PULL_OVER"


@dataclass(frozen=True)
class FallbackConfig:
    takeover_timeout_s: float = 10.0


@dataclass
class FallbackDecision:
    action: FallbackAction
    state: str
    driver_present: bool
    acknowledged: bool
    countdown_s: float
    reason: str
    critical: bool


class FallbackController:
    """State adapter for the fallback mechanism; it does not change the CB."""

    def __init__(self, config: Optional[FallbackConfig] = None) -> None:
        self.config = config or FallbackConfig()
        if self.config.takeover_timeout_s <= 0:
            raise ValueError("takeover_timeout_s must be > 0")
        self._takeover_started_at: Optional[float] = None
        self._acknowledged = False

    def reset(self) -> None:
        self._takeover_started_at = None
        self._acknowledged = False

    def acknowledge_takeover(self) -> None:
        self._acknowledged = True

    def evaluate(
        self,
        state: Any,
        *,
        driver_present: bool = True,
        now: Optional[float] = None,
    ) -> FallbackDecision:
        state_value = getattr(state, "value", state)
        state_value = str(state_value).upper()
        current = monotonic() if now is None else now

        if state_value == "CLOSED":
            self.reset()
            return FallbackDecision(
                FallbackAction.NORMAL, state_value, driver_present, False,
                0.0, "Circuit breaker is CLOSED; normal perception continues.", False
            )

        if state_value == "HALF_OPEN":
            if not driver_present:
                return self._safe_pull_over(
                    state_value, driver_present,
                    "Circuit breaker is HALF_OPEN and no driver is present."
                )
            return self._request_or_escalate(
                state_value, driver_present, current,
                "Circuit breaker is HALF_OPEN; driver takeover is requested."
            )

        if state_value == "OPEN":
            if not driver_present:
                return self._safe_pull_over(
                    state_value, driver_present,
                    "Circuit breaker is OPEN and no driver is present."
                )
            return self._request_or_escalate(
                state_value, driver_present, current,
                "Circuit breaker is OPEN; automated prediction is not trusted."
            )

        self.reset()
        return FallbackDecision(
            FallbackAction.SAFE_PULL_OVER, state_value, driver_present, False,
            0.0, "Unknown breaker state; fail-safe to minimum-risk maneuver.", True
        )

    def _request_or_escalate(
        self, state: str, driver_present: bool, now: float, reason: str
    ) -> FallbackDecision:
        if self._takeover_started_at is None:
            self._takeover_started_at = now
            self._acknowledged = False

        if self._acknowledged:
            return FallbackDecision(
                FallbackAction.TAKEOVER_REQUESTED, state, driver_present, True,
                0.0, "Driver acknowledged the takeover request.", True
            )

        elapsed = max(0.0, now - self._takeover_started_at)
        remaining = max(0.0, self.config.takeover_timeout_s - elapsed)

        if remaining <= 0:
            return self._safe_pull_over(
                state, driver_present,
                reason + " Takeover was not acknowledged within 10 seconds."
            )

        return FallbackDecision(
            FallbackAction.TAKEOVER_REQUESTED, state, driver_present, False,
            round(remaining, 2), reason, True
        )

    def _safe_pull_over(
        self, state: str, driver_present: bool, reason: str
    ) -> FallbackDecision:
        return FallbackDecision(
            FallbackAction.SAFE_PULL_OVER, state, driver_present,
            self._acknowledged, 0.0, reason, True
        )


def decision_to_dict(decision: FallbackDecision) -> dict:
    """Convert a decision to a JSON/WebSocket-safe payload."""
    return {
        "action": decision.action.value,
        "state": decision.state,
        "driver_present": decision.driver_present,
        "acknowledged": decision.acknowledged,
        "countdown_s": decision.countdown_s,
        "reason": decision.reason,
        "critical": decision.critical,
    }
