"""
run_B_external.py - TEST B: external / in-the-wild images - plain model vs the framework.

Uses the thresholds FROZEN in calibrated_config.json (never tuned on the external images).
For every external image it shows what the plain model predicted, whether it was wrong, and whether
(a) a plain softmax-confidence flag and (b) the MCB flagged it - i.e. which errors the plain model
silently made that the framework caught (and which it missed).

Setup:  external/images/*.jpg|png   +   external/labels.csv  (filename,label  - class name or index)
        then:  python step0_extract.py --only external   ;   python run_B_external.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb.analysis import base_parser, load_calibration, load_run, make_ctx  # noqa: E402
from mcb.io_utils import f3, md_table, pct, pct_ci, save_result  # noqa: E402
from mcb.metrics import confusion  # noqa: E402
from mcb.scoring import baseline_scores, states, variant_cbi  # noqa: E402

PROFILES = ["far05", "far10", "far20", "balanced"]


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--profile", default="far10", help="profile used for the per-image table")
    args = ap.parse_args(argv)
    ctx = make_ctx(args)
    calib = load_calibration(ctx)
    if not ctx.arrays_external_npz.exists():
        sys.exit("No external arrays. Add external/images + external/labels.csv, then run "
                 "`python step0_extract.py --only external`.")
    ext_samples = pd.read_csv(ctx.run_dir / "samples_external.csv")
    ext = load_run(ctx, ctx.arrays_external_npz, ext_samples.assign(split="ext"))
    internal = load_run(ctx)
    ev = internal.split == "eval"
    y = ext.y_err
    n = len(y)
    warnings = list(calib.get("warnings", []))
    if n < 60:
        warnings.append(f"Small external set (n={n}): confidence intervals are wide; report counts, not just rates.")
    if y.sum() == 0:
        warnings.append("The plain model made no errors on the external set - nothing to intercept.")

    base = baseline_scores(ext.P)
    cbi = {k: variant_cbi(v, ext.P, ext.meta) for k, v in calib["variants"].items()}
    base_int = baseline_scores(internal.P)
    cbi_int = {k: variant_cbi(v, internal.P, internal.meta) for k, v in calib["variants"].items()}

    # ---- per profile comparison
    tables = {}
    for p in PROFILES:
        rows = []
        for vn, v in calib["variants"].items():
            th = v["profiles"][p]
            st = states(cbi[vn], th["theta_warn"], th["theta_trip"])
            rows.append(dict(method=f"mcb_{vn}", rule="OPEN", **confusion(y, st == 2)))
            rows.append(dict(method=f"mcb_{vn}", rule="OPEN+HALF_OPEN", **confusion(y, st >= 1)))
        for b in ("msp", "entropy", "tta_entropy", "kflip"):
            rows.append(dict(method=b, rule="flag", **confusion(y, base[b] >= calib["baselines"][b][p])))
        tables[p] = rows

    # ---- matched flag-rate comparison (label-free matching): give the plain softmax flag the same
    #      number of flags as the MCB so the comparison is not about threshold aggressiveness.
    sel = calib["variants"]["selected"]["profiles"][args.profile]
    st_sel = states(cbi["selected"], sel["theta_warn"], sel["theta_trip"])
    m = int((st_sel == 2).sum())
    matched = {}
    for b in ("msp", "entropy"):
        flag = np.zeros(n, bool)
        if m > 0:
            flag[np.argsort(-base[b], kind="stable")[:m]] = True
        matched[b] = confusion(y, flag)
    matched["mcb_selected"] = confusion(y, st_sel == 2)
    matched["flags_given"] = m

    # ---- per image
    fl = {}
    for vn, v in calib["variants"].items():
        th = v["profiles"][args.profile]
        fl[vn] = states(cbi[vn], th["theta_warn"], th["theta_trip"])
    msp_flag = base["msp"] >= calib["baselines"]["msp"][args.profile]
    ent_flag = base["entropy"] >= calib["baselines"]["entropy"][args.profile]
    per_image = []
    for i, sid in enumerate(ext.ids):
        per_image.append(dict(
            image=sid, label=ctx.class_names[ext.labels[i]], pred=ctx.class_names[ext.pred[i]],
            confidence=float(ext.c0[i]), error=bool(y[i]),
            cbi_selected=float(cbi["selected"][i]), state_selected=["CLOSED", "HALF_OPEN", "OPEN"][int(fl["selected"][i])],
            state_flip_only=["CLOSED", "HALF_OPEN", "OPEN"][int(fl["flip_only"][i])] if "flip_only" in fl else None,
            msp_flag=bool(msp_flag[i]), entropy_flag=bool(ent_flag[i]),
            high_confidence_error=bool(y[i] and ext.c0[i] >= 0.75)))
    pd.DataFrame(per_image).to_csv(ctx.run_dir / "B_external_per_image.csv", index=False)
    err = np.array([r["error"] for r in per_image])
    mcb_open = fl["selected"] == 2
    caught_mcb_missed_msp = [r["image"] for r, a, b in zip(per_image, mcb_open, msp_flag) if r["error"] and a and not b]
    caught_msp_missed_mcb = [r["image"] for r, a, b in zip(per_image, mcb_open, msp_flag) if r["error"] and b and not a]
    silent = [r["image"] for r in per_image if r["high_confidence_error"]]
    silent_caught = [r["image"] for r in per_image if r["high_confidence_error"] and (mcb_open[ext.ids.index(r["image"])])]

    # ---- domain-shift indicator vs internal evaluation split
    shift = dict(
        mean_confidence_external=float(ext.c0.mean()), mean_confidence_internal=float(internal.c0[ev].mean()),
        plain_accuracy_external=float(1 - y.mean()), plain_accuracy_internal=float(1 - internal.y_err[ev].mean()),
        mean_cbi_external=float(cbi["selected"].mean()), mean_cbi_internal=float(cbi_int["selected"][ev].mean()),
        trip_rate_external=float(mcb_open.mean()),
        trip_rate_internal=float((states(cbi_int["selected"][ev], sel["theta_warn"], sel["theta_trip"]) == 2).mean()),
        msp_flag_rate_external=float(msp_flag.mean()),
        msp_flag_rate_internal=float((base_int["msp"][ev] >= calib["baselines"]["msp"][args.profile]).mean()))

    results = dict(n_external=n, n_errors=int(y.sum()), plain_accuracy=float(1 - y.mean()), profile_for_table=args.profile,
                   comparison_by_profile=tables, matched_flag_rate=matched, interception=dict(
                       errors=int(err.sum()), caught_by_mcb=int((err & mcb_open).sum()), caught_by_msp=int((err & msp_flag).sum()),
                       caught_by_mcb_missed_by_msp=caught_mcb_missed_msp, caught_by_msp_missed_by_mcb=caught_msp_missed_mcb,
                       high_confidence_errors=silent, high_confidence_errors_caught_by_mcb=silent_caught),
                   domain_shift_indicator=shift, per_image=per_image)

    md = ["## 1. Plain model on the external images", "",
          f"- Images: **{n}**; plain-model accuracy **{pct(1 - y.mean())}** -> **{int(y.sum())} silent errors** without a monitor.",
          f"- Internal evaluation accuracy for reference: {pct(shift['plain_accuracy_internal'])} "
          f"(mean confidence internal {f3(shift['mean_confidence_internal'])} vs external {f3(shift['mean_confidence_external'])}).",
          f"- Thresholds are the FROZEN calibration profiles; nothing was tuned on these images.", ""]
    for p in PROFILES:
        md += [f"## 2.{PROFILES.index(p) + 1} Plain model vs framework - profile `{p}`", "",
               md_table(["Method", "Rule", "Errors caught (TP)", "Recall [95% CI]", "False alarms (FP)", "FAR [95% CI]", "Silent errors left"],
                        [[r["method"], r["rule"], f"{r['TP']}/{r['TP'] + r['FN']}", pct_ci(r["recall"], r["recall_ci"]),
                          f"{r['FP']}/{r['FP'] + r['TN']}", pct_ci(r["far"], r["far_ci"]), r["FN"]] for r in tables[p]]), ""]
    md += [f"## 3. Matched flag-rate comparison (profile `{args.profile}`)", "",
           f"The plain softmax detectors are given exactly as many flags as the MCB raised ({m}), chosen as their "
           f"highest-uncertainty images - so differences reflect ranking quality, not threshold aggressiveness.", "",
           md_table(["Method", "Flags", "Errors caught", "False alarms"],
                    [[k, matched[k]["TP"] + matched[k]["FP"], matched[k]["TP"], matched[k]["FP"]] for k in ("mcb_selected", "msp", "entropy")]), "",
           "## 4. Errors the plain model made silently", "",
           f"- Errors caught by MCB but NOT by the softmax flag: **{len(caught_mcb_missed_msp)}** {caught_mcb_missed_msp}",
           f"- Errors caught by the softmax flag but NOT by MCB: **{len(caught_msp_missed_mcb)}** {caught_msp_missed_mcb}",
           f"- High-confidence (>=0.75) errors: **{len(silent)}**, of which MCB caught **{len(silent_caught)}** {silent_caught}", "",
           "## 5. Domain-shift indicator", "",
           md_table(["Quantity", "Internal (eval split)", "External"],
                    [["Plain accuracy", pct(shift["plain_accuracy_internal"]), pct(shift["plain_accuracy_external"])],
                     ["Mean confidence", f3(shift["mean_confidence_internal"]), f3(shift["mean_confidence_external"])],
                     ["Mean CBI (selected)", f3(shift["mean_cbi_internal"]), f3(shift["mean_cbi_external"])],
                     ["MCB trip rate", pct(shift["trip_rate_internal"]), pct(shift["trip_rate_external"])],
                     ["Softmax flag rate", pct(shift["msp_flag_rate_internal"]), pct(shift["msp_flag_rate_external"])]]), "",
           f"## 6. Per-image audit (profile `{args.profile}`)", "",
           md_table(["Image", "Truth", "Pred", "Conf", "CBI", "MCB state", "Softmax flag", "Outcome"],
                    [[r["image"], r["label"], r["pred"], f3(r["confidence"], 2), f3(r["cbi_selected"]), r["state_selected"],
                      "FLAG" if r["msp_flag"] else "-",
                      ("ERROR - caught by MCB" if r["state_selected"] == "OPEN" else "ERROR - MISSED") if r["error"]
                      else ("false alarm" if r["state_selected"] == "OPEN" else "ok")] for r in per_image])]
    save_result(ctx, "B_external", "B_external", "Test B - External Images: Plain Model vs Framework",
                results, "\n".join(md), warnings=warnings)


if __name__ == "__main__":
    main()
