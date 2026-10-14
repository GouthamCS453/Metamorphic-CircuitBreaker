"""
run_G_attribution.py - TEST G (EXPLORATORY): can MCB attribute a degradation to the right family?

Reuses the degraded-image arrays from Test C.  For images the model classified correctly when clean, the
family instability change  dIR_k = IR_k(degraded) - IR_k(clean)  is computed; the predicted family is
argmax_k dIR_k.  Ties and negligible changes are reported explicitly (never silently resolved), which fixes
the first-family tie-break bias of the earlier protocol.  Shifts map to families:
  geometric: rotate | photometric: darken, color_cast | sensor_noise: blur, noise, jpeg

Usage:  python run_G_attribution.py        (run run_C_shift.py first)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb.analysis import base_parser, load_calibration, load_run, make_ctx  # noqa: E402
from mcb.io_utils import f3, md_table, pct, save_result  # noqa: E402
from mcb.metrics import binom_sf, wilson  # noqa: E402
from mcb.scoring import family_ir, signals, states, variant_cbi  # noqa: E402
from mcb.shifts import SHIFTS  # noqa: E402
from mcb.transforms import FAMILIES  # noqa: E402

COLS = FAMILIES + ["tie", "none"]


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--min-severity", type=int, default=2)
    ap.add_argument("--min-delta", type=float, default=0.05, help="smallest dIR to count as a detection")
    ap.add_argument("--variant", default="selected")
    args = ap.parse_args(argv)
    ctx = make_ctx(args)
    calib = load_calibration(ctx)
    run = load_run(ctx)
    if not ctx.arrays_shift_npz.exists():
        sys.exit("Run run_C_shift.py first (it produces the degraded-image arrays).")
    z = np.load(ctx.arrays_shift_npz, allow_pickle=False)
    ids = [str(x) for x in z["ids"]]
    pos = {s: i for i, s in enumerate(run.ids)}
    rows_idx = np.array([pos[i] for i in ids])
    params = calib["variants"][args.variant]["params"]
    clean_P = run.P[rows_idx]
    lab = run.labels[rows_idx]
    clean_ok = clean_P[:, 0].argmax(1) == lab

    def ir_of(P):
        return np.column_stack([family_ir(signals(P, params["signal"]), run.meta, None, params["mean_w"])[f] for f in FAMILIES])

    ir_clean = ir_of(clean_P)
    th = calib["variants"][args.variant]["profiles"]["far10"]
    conf = {f: {c: 0 for c in COLS} for f in FAMILIES}
    flagged = {f: [0, 0] for f in FAMILIES}
    per_shift = {}
    for key in [k for k in z.files if k.startswith("probs__")]:
        _, sname, lvl = key.split("__")
        lvl = int(lvl)
        if lvl < args.min_severity:
            continue
        P = z[key].astype(np.float64)
        if P.shape[0] != len(ids):
            continue
        fam_true = SHIFTS[sname]["family"]
        d = ir_of(P) - ir_clean
        s = variant_cbi(calib["variants"][args.variant], P, run.meta)
        st = states(s, th["theta_warn"], th["theta_trip"])
        ok_after = P[:, 0].argmax(1) == lab
        for i in np.where(clean_ok)[0]:
            order = np.sort(d[i])[::-1]
            if order[0] < args.min_delta:
                pred = "none"
            elif order[0] - order[1] < 1e-9:
                pred = "tie"
            else:
                pred = FAMILIES[int(d[i].argmax())]
            conf[fam_true][pred] += 1
            flagged[fam_true][0] += int(st[i] >= 1)
            flagged[fam_true][1] += 1
            per_shift.setdefault(sname, dict(family=fam_true, n=0, correct=0, none=0, tie=0))
            ps = per_shift[sname]
            ps["n"] += 1
            ps["correct"] += int(pred == fam_true)
            ps["none"] += int(pred == "none")
            ps["tie"] += int(pred == "tie")
    total = sum(sum(v.values()) for v in conf.values())
    correct = sum(conf[f][f] for f in FAMILIES)
    detected = sum(conf[f][c] for f in FAMILIES for c in FAMILIES + ["tie"])
    acc, lo, hi = wilson(correct, total)
    chance = 1 / len(FAMILIES)
    pval = binom_sf(correct, total, chance) if total else float("nan")
    cond_acc = correct / max(detected, 1)
    warnings = ["EXPLORATORY: attribution is not a core claim. Metamorphic relations measure local sensitivity around "
                "the (already degraded) image, so a degradation need not raise the instability of its own family."]
    if total < 150:
        warnings.append(f"Only {total} attribution trials; interpret with caution.")
    results = dict(min_severity=args.min_severity, min_delta=args.min_delta, variant=args.variant, trials=total,
                   accuracy=acc, accuracy_ci=[lo, hi], chance=chance, p_value_vs_chance=pval,
                   accuracy_given_detected=cond_acc, detected_fraction=detected / max(total, 1),
                   confusion=conf, flagged_by_breaker=dict(zip(flagged, [{"flagged": a, "n": b} for a, b in flagged.values()])),
                   per_shift=per_shift,
                   interpretation=("above chance" if (not np.isnan(lo) and lo > chance) else "not above chance"))
    md = ["## 1. Method", "",
          f"Clean-correct images x shifts of severity >= {args.min_severity}. Predicted family = argmax of the change in family "
          f"instability (dIR); `none` if the largest dIR < {args.min_delta}; `tie` if two families share the maximum. "
          f"Variant `{args.variant}`.", "",
          "## 2. Confusion (rows = injected family)", "",
          md_table(["Injected \\ Predicted"] + COLS + ["Accuracy"],
                   [[f] + [conf[f][c] for c in COLS] + [pct(conf[f][f] / max(sum(conf[f].values()), 1))] for f in FAMILIES]), "",
          "## 3. Summary", "",
          f"- Trials: **{total}**; attribution accuracy **{pct(acc)}** (Wilson 95% [{pct(lo)}, {pct(hi)}]) vs chance {pct(chance)}; "
          f"one-sided binomial p = {f3(pval, 4)} -> **{results['interpretation']}**.",
          f"- Share of trials where any family reacted (detected): {pct(results['detected_fraction'])}; "
          f"accuracy among detected trials: {pct(cond_acc)}.", "",
          "## 4. Per shift", "",
          md_table(["Shift", "True family", "Trials", "Correct family", "Tie", "None"],
                   [[s, p["family"], p["n"], pct(p["correct"] / max(p["n"], 1)), p["tie"], p["none"]] for s, p in per_shift.items()])]
    save_result(ctx, "G_attribution", "G_attribution", "Test G - Fault Attribution (Exploratory)", results, "\n".join(md),
                warnings=warnings)


if __name__ == "__main__":
    main()
