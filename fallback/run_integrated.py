"""Run the existing framework through the additive fallback boundary.

This is a new entry point. Existing project entry points remain unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.circuit_breaker import MetamorphicCircuitBreaker
from src.models.mobilenet_adapter import MobileNetAdapter

from fallback.controller import decision_to_dict
from fallback.integration import CircuitBreakerWithFallback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--driver-absent",
        action="store_true",
        help="Simulate driverless mode for the fallback test.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")

    model = MobileNetAdapter()
    existing_cb = MetamorphicCircuitBreaker(model)
    system = CircuitBreakerWithFallback(existing_cb)

    result = system.evaluate(
        Image.open(image_path).convert("RGB"),
        driver_present=not args.driver_absent,
    )

    print("=== EXISTING METAMORPHIC CIRCUIT BREAKER ===")
    print(f"State       : {result.report.state.value}")
    print(f"Action      : {result.report.action}")
    print(f"CBI         : {result.report.cbi:.4f}")
    print(f"Prediction  : {result.report.baseline_label}")
    print(f"Confidence  : {result.report.baseline_confidence:.4f}")

    print("\n=== ADDITIVE FALLBACK ===")
    for key, value in decision_to_dict(result.fallback).items():
        print(f"{key:15}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
