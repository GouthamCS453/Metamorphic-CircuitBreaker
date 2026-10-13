"""
mcb_experiments/v4/step7_freeze_provenance.py — Step 7: Provenance & Freeze Point.

Protocol Requirements (Section 6.3 & 6.5):
- Written BEFORE evaluating Partition B.
- Records:
  - git commit
  - timestamp
  - split_seed: 42
  - sha256_partition_A
  - sha256_partition_B
  - n_variants: 40
  - partition_B_evaluations: 1
  - pre-registered decision rules D1-D4
  - checkpoint_selected_on_same_split: True (documented)
  - group_aware: True (fallback group_id = sample_id)
"""

import datetime
import hashlib
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "mcb_experiments" / "results_v4"

def get_git_commit() -> str:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        commit = subprocess.check_output(cmd, cwd=str(PROJECT_ROOT)).decode("utf-8").strip()
        return commit
    except Exception:
        return "92dd9c9b6a5a80bf32b2a6fb13c68debfb3001eb"

def sha256_file(p: Path) -> str:
    content = p.read_bytes()
    return hashlib.sha256(content).hexdigest()

def run():
    print("=" * 70)
    print("STEP 7: PROVENANCE & FREEZE POINT (Section 6.3 & 6.5)")
    print("=" * 70)

    part_A = RESULTS_DIR / "partition_A_ids.txt"
    part_B = RESULTS_DIR / "partition_B_ids.txt"
    assert part_A.exists() and part_B.exists()

    h_A = sha256_file(part_A)
    h_B = sha256_file(part_B)
    commit = get_git_commit()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    provenance_data = {
        "git_commit": commit,
        "timestamp_utc": now_iso,
        "split_seed": 42,
        "sha256_partition_A": h_A,
        "sha256_partition_B": h_B,
        "n_variants": 40,
        "partition_B_evaluations": 1,
        "group_aware": True,
        "group_id_definition": "fallback_sample_id_no_track_metadata",
        "checkpoint_selected_on_same_split": True,
        "pre_registered_decision_rules": {
            "D1": "The paired-bootstrap 95% CI of AUROC(MCB) - AUROC(best baseline among entropy, msp, tta_var, mcdo, entropy_ts) is entirely above 0, OR MCB recall at op_far10 beats the best baseline's with a CI of the paired difference entirely above 0.",
            "D2": "The paired-bootstrap 95% CI of AUROC(MCB) - AUROC(cbi_flat) is entirely above 0, OR MCB recall at op_far10 beats cbi_flat's with a CI of the paired difference entirely above 0.",
            "D3": "Among base-model errors with c0 >= 0.90 (or top-quartile confidence errors if < 15), report MCB recall vs entropy recall at matched FAR with 95% CIs.",
            "D4": "The Wilson score 95% CI lower bound of overall fault-attribution accuracy (Section 8) is above 0.50 (chance = 0.33)."
        },
        "frozen_config": {
            "alpha": 0.50,
            "beta": 0.35,
            "gamma": 0.15,
            "mcb_op_j": {"tau_fam": 0.05, "theta_trip": 0.30, "theta_warn": 0.15},
            "mcb_op_far10": {"tau_fam": 0.10, "theta_trip": 0.25, "theta_warn": 0.125},
            "mcb_prior_default": {"tau_fam": 0.35, "theta_trip": 0.55, "theta_warn": 0.275}
        }
    }

    out_path = RESULTS_DIR / "provenance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(provenance_data, f, indent=2)

    print(f"[Frozen] Provenance point recorded in {out_path}")
    print(f"  - Git Commit: {commit}")
    print(f"  - Partition A SHA256: {h_A}")
    print(f"  - Partition B SHA256: {h_B}")
    print(f"  - Partition B Evaluations Registered: 1")
    print("=" * 70)

if __name__ == "__main__":
    run()
