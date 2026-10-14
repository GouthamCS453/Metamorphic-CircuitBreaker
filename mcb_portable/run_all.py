"""
run_all.py - Run the whole protocol for ONE (model, dataset) pair.

Order:  step0 extract -> step1 calibrate (freeze) -> A batch -> B external -> C shift -> D ablation -> E family
        -> F latency -> G attribution.   Stage 1 must succeed before any test runs.

Usage
  python run_all.py                              # uses member_config.py
  python run_all.py --config demo/demo_config.py # smoke test on synthetic data
  python run_all.py --n 1500                     # stratified subset
  python run_all.py --skip C,G                   # skip tests (letters)
  python run_all.py --only A,B                   # run only these tests (after extract + calibrate)
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb.analysis import base_parser  # noqa: E402

import run_A_batch, run_B_external, run_C_shift, run_D_ablation, run_E_family, run_F_latency, run_G_attribution  # noqa: E402
import step0_extract, step1_calibrate  # noqa: E402

TESTS = [("A", run_A_batch), ("B", run_B_external), ("C", run_C_shift), ("D", run_D_ablation),
         ("E", run_E_family), ("F", run_F_latency), ("G", run_G_attribution)]


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n", type=int, default=None, help="stratified subset size for the dataset (0 = all)")
    ap.add_argument("--n-shift", type=int, default=100)
    ap.add_argument("--skip", default="", help="comma list of test letters to skip, e.g. C,G")
    ap.add_argument("--only", default="", help="comma list of test letters to run")
    ap.add_argument("--skip-extract", action="store_true", help="reuse existing arrays.npz")
    args = ap.parse_args()
    common = ["--config", args.config] + (["--run-dir", args.run_dir] if args.run_dir else [])
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    t0 = time.time()
    status = {}

    if not args.skip_extract:
        print("\n=== STEP 0: extract (one inference pass) ===")
        step0_extract.main(common + (["--n", str(args.n)] if args.n is not None else []))
    print("\n=== STEP 1: calibrate & freeze parameters ===")
    step1_calibrate.main(common)

    extra = {"C": ["--n-shift", str(args.n_shift)]}
    for letter, mod in TESTS:
        if letter in skip or (only and letter not in only):
            status[letter] = "skipped"
            continue
        print(f"\n=== TEST {letter} ===")
        try:
            mod.main(common + extra.get(letter, []))
            status[letter] = "ok"
        except SystemExit as e:  # missing optional inputs (e.g. no external images)
            status[letter] = f"skipped: {e}"
            print(f"[skip] {letter}: {e}")
        except Exception:
            status[letter] = "FAILED"
            traceback.print_exc()
    print("\n=== SUMMARY ===")
    for k, v in status.items():
        print(f"  Test {k}: {v}")
    print(f"Done in {time.time() - t0:.0f}s. Results are in the run folder (see results/<run_id>/).")


if __name__ == "__main__":
    main()
