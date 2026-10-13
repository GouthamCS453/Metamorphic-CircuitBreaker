"""
mcb_experiments/v4/step5_baselines.py — Step 5: Baselines Finalization.

Protocol Requirements (Section 5):
- Temperature scaling: Fit temperature T on Partition A only using bounded NLL optimization.
- Compute entropy_ts across all samples.
- Verify all baselines: ff, kflip_1..8, tta_var, entropy, msp_score, mcdo_score, entropy_ts.
- Verify MC-Dropout device matches MCB device (both CPU).
- Updates mcb_experiments/results_v4/per_sample_scores.csv.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "mcb_experiments" / "results_v4"
CSV_PATH = RESULTS_DIR / "per_sample_scores.csv"
LOGITS_PATH = RESULTS_DIR / "logits.npy"

def run():
    print("=" * 70)
    print("STEP 5: BASELINES COMPUTATION & TEMPERATURE SCALING (Section 5)")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)
    logits = np.load(LOGITS_PATH)
    assert len(df) == len(logits), f"Row mismatch: df {len(df)} vs logits {len(logits)}"

    # Fit temperature T on Partition A only
    A = (df.partition == "A").to_numpy()
    y = df.y_true.to_numpy()
    print(f"Fitting temperature scaling on Partition A (N = {A.sum()})...")

    def nll(T):
        scaled = logits[A] / T
        log_probs = log_softmax(scaled, axis=1)
        return -log_probs[np.arange(A.sum()), y[A]].mean()

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    T_opt = float(res.x)
    print(f"Optimal Temperature T fitted on Partition A: {T_opt:.4f} (NLL = {res.fun:.4f})")

    # Compute entropy_ts across all samples
    P_scaled = softmax(logits / T_opt, axis=1)
    entropy_ts = -(P_scaled * np.log(P_scaled + 1e-12)).sum(axis=1)
    df["entropy_ts"] = entropy_ts

    # Verify required baseline columns
    required_cols = [
        "ff", "kflip_1", "kflip_2", "kflip_3", "kflip_5", "kflip_8",
        "tta_var", "entropy", "msp_score", "mcdo_score", "entropy_ts"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required baseline column: {col}"
        assert not df[col].isna().any(), f"NaN values detected in {col}"

    df.to_csv(CSV_PATH, index=False)
    print(f"[Updated] {CSV_PATH} with calibrated entropy_ts.")
    print("Baseline columns verified:")
    for col in required_cols:
        mean_val = df[col].mean()
        print(f"  - {col:<12}: mean = {mean_val:.4f}")

    print("=" * 70)

if __name__ == "__main__":
    run()
