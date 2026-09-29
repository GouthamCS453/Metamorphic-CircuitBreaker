"""Run the existing circuit breaker and then apply the additive fallback.

This is deliberately a separate entry point. It does not modify the circuit
breaker, model adapter, metamorphic relations, thresholds, or Streamlit app.

Examples:
    python fallback/run_demo.py --image path/to/sign.png --driver-present
    python fallback/run_demo.py --image path/to/sign.png --driver-absent
    python fallback/run_demo.py --image path/to/sign.png --driver-present --ack
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fallback.controller import FallbackController, decision_to_dict
from src.circuit_breaker import MetamorphicCircuitBreaker
from src.models.mobilenet_adapter import MobileNetAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description="Metamorphic CB fallback demo")
    parser.add_argument("--image", required=True, help="Traffic-sign image")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--driver-present", action="store_true")
    group.add_argument("--driver-absent", action="store_true")
    parser.add_argument(
        "--ack",
        action="store_true",
        help="Simulate driver acknowledgement before the fallback check",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")

    driver_present = not args.driver_absent

    model = MobileNetAdapter()
    breaker = MetamorphicCircuitBreaker(model)
    report = breaker.evaluate(Image.open(image_path).convert("RGB"))

    fallback = FallbackController()
    if args.ack:
        fallback.evaluate(
            report.state, driver_present=driver_present, now=0.0
        )
        fallback.acknowledge_takeover()

    decision = fallback.evaluate(
        report.state, driver_present=driver_present, now=1.0
    )

    print(json.dumps({
        "circuit_breaker": {
            "state": report.state.value,
            "action": report.action,
            "cbi": report.cbi,
            "baseline_label": report.baseline_label,
            "baseline_confidence": report.baseline_confidence,
        },
        "fallback": decision_to_dict(decision),
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
