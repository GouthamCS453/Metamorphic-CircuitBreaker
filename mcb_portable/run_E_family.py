"""
run_E_family.py - TEST E: family / transform vulnerability profile.

Which perturbation axes does this model fail on?  Reports per-family instability (IR_k) for all / correct /
erroneous predictions, how well each family separates errors (AUROC), per-transform flip rates for errors vs
correct, per-class views and cross-family coupling.  Offline (uses arrays.npz), whole evaluation split.

Usage:  python run_E_family.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb.analysis import base_parser, load_calibration, load_run, make_ctx  # noqa: E402
from mcb.io_utils import ci_str, f3, md_table, pct, save_result  # noqa: E402
from mcb.metrics import bootstrap_auroc, wilson  # noqa: E402
from mcb.scoring import family_ir, signals  # noqa: E402
from mcb.transforms import FAMILIES  # noqa: E402


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--all-splits", action="store_true", help="use calibration+evaluation data (descriptive profile)")
    args = ap.parse_args(argv)
    ctx = make_ctx(args)
    run = load_run(ctx)
    calib = load_calibration(ctx)
    m = np.ones(len(run.y_err), bool) if args.all_splits else (run.split == "eval")
    P, y, lab = run.P[m], run.y_err[m], run.labels[m]
    warnings = list(calib.get("warnings", []))
    if y.sum() < 20:
        warnings.append(f"Only {int(y.sum())} errors in the profiled data; per-family differences are indicative only.")

    out = {}
    for sg in ("flip", "drift"):
        ir = family_ir(signals(P, sg), run.meta, None, 0.7)
        d = {}
        for f in FAMILIES:
            v = ir[f]
            a, lo, hi = bootstrap_auroc(y, v, n_boot=500, seed=ctx.seed)
            d[f] = dict(mean_all=float(v.mean()), mean_correct=float(v[y == 0].mean()),
                        mean_error=float(v[y == 1].mean()) if y.sum() else None,
                        frac_nonzero=float((v > 0).mean()), auroc_error_detection=a, auroc_ci=[lo, hi])
        out[sg] = d
    coupling = np.corrcoef(np.column_stack([family_ir(signals(P, "drift"), run.meta, None, 0.7)[f] for f in FAMILIES]).T)

    S = signals(P, "flip")
    tr = []
    for j, mt in enumerate(run.meta):
        e = S[y == 1, j].mean() if y.sum() else float("nan")
        c = S[y == 0, j].mean()
        tr.append(dict(id=mt["id"], family=mt["family"], type=mt["type"], severity=mt["severity"],
                       flip_rate_error=float(e), flip_rate_correct=float(c), difference=float(e - c)))
    types = {}
    for t in tr:
        types.setdefault((t["family"], t["type"]), []).append(t)
    type_rows = [dict(family=k[0], type=k[1], flip_rate_error=float(np.mean([x["flip_rate_error"] for x in v])),
                      flip_rate_correct=float(np.mean([x["flip_rate_correct"] for x in v]))) for k, v in types.items()]

    cls = []
    ir_flip = family_ir(S, run.meta, None, 0.7)
    for ci, cn in enumerate(ctx.class_names):
        mm = lab == ci
        if mm.sum() == 0:
            continue
        cls.append(dict(cls=cn, n=int(mm.sum()), error_rate=float(y[mm].mean()),
                        **{f"ir_{f}": float(ir_flip[f][mm].mean()) for f in FAMILIES}))
    zero_flip = dict(all=float((S.sum(1) == 0).mean()), errors=float((S[y == 1].sum(1) == 0).mean()) if y.sum() else None,
                     correct=float((S[y == 0].sum(1) == 0).mean()),
                     errors_with_zero_flips=int(((S.sum(1) == 0) & (y == 1)).sum()), n_errors_total=int(y.sum()))
    results = dict(n=int(m.sum()), n_errors=int(y.sum()), family_profile=out, per_transform=tr, per_type=type_rows,
                   per_class=cls, family_coupling_corr=coupling.tolist(), coupling_order=FAMILIES, zero_flip=zero_flip)

    md = [f"Profiled samples: **{int(m.sum())}** ({int(y.sum())} base-model errors). `flip` = original binary signal; `drift` = continuous.", ""]
    for sg in ("flip", "drift"):
        md += [f"## Family instability ({sg} signal)", "",
               md_table(["Family", "Mean IR (all)", "Correct", "Errors", "Share > 0", "AUROC for error detection [95% CI]"],
                        [[f, f3(out[sg][f]["mean_all"]), f3(out[sg][f]["mean_correct"]), f3(out[sg][f]["mean_error"]),
                          pct(out[sg][f]["frac_nonzero"]), ci_str(out[sg][f]["auroc_error_detection"], *out[sg][f]["auroc_ci"])]
                         for f in FAMILIES]), ""]
    md += ["## Zero-flip blindness", "",
           f"- Samples with **no flip at all** across the whole bank: {pct(zero_flip['all'])} overall; correct {pct(zero_flip['correct'])}; "
           f"errors {pct(zero_flip['errors']) if zero_flip['errors'] is not None else 'n/a'} "
           f"({zero_flip['errors_with_zero_flips']} of {int(y.sum())} errors).",
           "- These errors are invisible to the binary-flip signal; the continuous `drift` signal exists to address them.", "",
           "## Per transform type (flip rate)", "",
           md_table(["Family", "Type", "Flip rate | errors", "Flip rate | correct"],
                    [[r["family"], r["type"], pct(r["flip_rate_error"]), pct(r["flip_rate_correct"])] for r in type_rows]), "",
           "## Per individual transform (sorted by error-vs-correct difference)", "",
           md_table(["Transform", "Errors", "Correct", "Difference"],
                    [[t["id"], pct(t["flip_rate_error"]), pct(t["flip_rate_correct"]), pct(t["difference"])]
                     for t in sorted(tr, key=lambda x: -(x["difference"] if x["difference"] == x["difference"] else -1))]), "",
           "## Per class", "",
           md_table(["Class", "N", "Error rate"] + [f"IR {f}" for f in FAMILIES],
                    [[c["cls"], c["n"], pct(c["error_rate"])] + [f3(c[f"ir_{f}"]) for f in FAMILIES] for c in cls]), "",
           "## Cross-family coupling (correlation of drift IR)", "",
           md_table([""] + FAMILIES, [[FAMILIES[i]] + [f3(coupling[i, j], 2) for j in range(3)] for i in range(3)]), "",
           "_High off-diagonal values mean a failure on one axis co-occurs with failures on the others (multi-axis sensitivity)._"]
    save_result(ctx, "E_family", "E_family", "Test E - Family & Transform Vulnerability Profile", results, "\n".join(md),
                warnings=warnings)


if __name__ == "__main__":
    main()
