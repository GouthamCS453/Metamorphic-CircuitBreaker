"""
web_app/backend/app/config.py
Dynamic configuration store for the Metamorphic Circuit Breaker parameters.
Maintains a singleton in-memory config that all routes read/write.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


from pathlib import Path


def _find_project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()
PRESETS_DIR = PROJECT_ROOT / "data" / "presets"


@dataclass
class CBConfig:
    alpha: float = 0.50
    beta: float = 0.35
    gamma: float = 0.15
    tau_fam: float = 0.35
    theta_warn: float = 0.25
    theta_trip: float = 0.55

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self):
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"alpha + beta + gamma must sum to 1.0 (got {total:.4f})")
        if self.theta_warn >= self.theta_trip:
            raise ValueError("theta_warn must be less than theta_trip")

    def to_src_config(self):
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.circuit_breaker import CircuitBreakerConfig
        return CircuitBreakerConfig(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            tau_fam=self.tau_fam,
            theta_warn=self.theta_warn,
            theta_trip=self.theta_trip,
        )


CLINICAL_PROFILES = {
    "oncology": {
        "label": "High-Risk Oncology Clinic",
        "description": "Zero tolerance — trips aggressively even on single-family instability.",
        "config": CBConfig(alpha=0.55, beta=0.30, gamma=0.15, tau_fam=0.35, theta_warn=0.20, theta_trip=0.50),
    },
    "rural": {
        "label": "Rural Clinic (Low Capacity)",
        "description": "Requires multi-family proof before tripping — reduces triage overload.",
        "config": CBConfig(alpha=0.40, beta=0.45, gamma=0.15, tau_fam=0.35, theta_warn=0.30, theta_trip=0.60),
    },
    "telehealth": {
        "label": "Telehealth / Smartphone Cameras",
        "description": "Higher beta to separate sensor noise from genuine model collapse.",
        "config": CBConfig(alpha=0.45, beta=0.40, gamma=0.15, tau_fam=0.35, theta_warn=0.25, theta_trip=0.55),
    },
    "calibrated": {
        "label": "Calibrated Model",
        "description": "Higher gamma — temperature-scaled confidence is more trustworthy.",
        "config": CBConfig(alpha=0.45, beta=0.30, gamma=0.25, tau_fam=0.35, theta_warn=0.25, theta_trip=0.55),
    },
}

# Global singleton config
_active_config = CBConfig()


def get_active_config() -> CBConfig:
    return _active_config


def set_active_config(cfg: CBConfig) -> None:
    global _active_config
    _active_config = cfg
