"""
mcb_experiments/v4/step4_partitions.py — Step 4: Group-Aware Stratified Partitions.

Protocol Requirements (Section 4.3):
- Stratified group 2-fold split on y_true + "_" + correct, seed=42.
- Splits into Partition A and Partition B.
- Writes partition_A_ids.txt and partition_B_ids.txt in mcb_experiments/results_v4/.
- Updates mcb_experiments/results_v4/per_sample_scores.csv with 'partition' column.
- MANDATORY STOP: Assert Partition B errors >= 50.
"""

import hashlib
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "mcb_experiments" / "results_v4"
CSV_PATH = RESULTS_DIR / "per_sample_scores.csv"

def make_partitions(df, seed=42):
    strat = df.y_true.astype(str) + "_" + df.correct.astype(str)
    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    a, b = next(sgkf.split(df, strat, groups=df.group_id))
    df = df.copy()
    df["partition"] = "B"
    df.iloc[a, df.columns.get_loc("partition")] = "A"
    return df

def run():
    print("=" * 70)
    print("STEP 4: GROUP-AWARE STRATIFIED PARTITIONING (Section 4.3)")
    print("=" * 70)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from {CSV_PATH}")

    df = make_partitions(df, seed=42)
    df.to_csv(CSV_PATH, index=False)
    print(f"Updated {CSV_PATH} with 'partition' column.")

    hashes = {}
    error_counts = {}
    sizes = {}

    for p in "AB":
        sub = df[df.partition == p]
        ids = "\n".join(sorted(sub.sample_id.astype(str)))
        out_path = RESULTS_DIR / f"partition_{p}_ids.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(ids)
        h = hashlib.sha256(ids.encode()).hexdigest()
        n_err = int((sub.correct == 0).sum())
        hashes[p] = h
        error_counts[p] = n_err
        sizes[p] = len(sub)
        print(f"Partition {p}: N = {len(sub):5d} | Errors = {n_err:4d} | SHA256: {h}")

    # MANDATORY STOP CHECK
    print("\n" + "=" * 70)
    print("MANDATORY STOP CHECK: Partition B Base-Model Errors >= 50")
    print("=" * 70)
    b_errors = error_counts["B"]
    print(f"Partition B Base-Model Errors: {b_errors}")
    if b_errors < 50:
        print(f"[STOP TRIGGERED] Partition B has fewer than 50 errors ({b_errors} < 50)!")
        sys.exit(1)

    print(f"[VERIFICATION PASSED] Partition B has {b_errors} errors (>= 50 threshold satisfied).")
    print("=" * 70)

if __name__ == "__main__":
    run()
