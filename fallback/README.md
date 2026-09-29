# Additive Vehicle Fallback Subsystem

This directory is intentionally isolated from the existing Metamorphic Circuit Breaker.

## State policy

| Existing CB state | Fallback behavior |
|---|---|
| CLOSED | No fallback. Normal automated perception continues. |
| HALF_OPEN | Driver takeover request + 5-second acknowledgement window. |
| OPEN + driver present | Driver takeover request + 5-second acknowledgement window. |
| OPEN + driver absent | Immediate minimum-risk maneuver (safe pull-over). |
| OPEN + takeover timeout | Minimum-risk maneuver. |

The fallback layer does not calculate CBI, alter thresholds, execute metamorphic
relations, modify the model, or change circuit-breaker state.

## Test

From the repository root:

    python -m unittest fallback.test_controller

## Integration

A caller can pass the existing DiagnosticReport.state directly:

    decision = controller.evaluate(report.state, driver_present=True)

The result is serializable with decision_to_dict() for a UI or telemetry channel.
