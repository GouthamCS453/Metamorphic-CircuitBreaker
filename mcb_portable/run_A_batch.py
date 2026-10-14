"""
run_A_batch.py - TEST A: full batch benchmark (A) + baseline comparison (A1) + calibration study (A2).

Everything is evaluated on the EVALUATION split with the thresholds FROZEN by step1_calibrate.py.

Usage:  python run_A_batch.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb.analysis import base_parser, critical_mask, load_calibration, load_run, make_ctx  # noqa: E402
from mcb.calibration import FAR_PROFILES  # noqa: E402
from mcb.io_utils import ci_str, f3, md_table, pct, pct_ci, save_result  # noqa: E402
from mcb.metrics import (bootstrap_auroc, confusion, mean_ci, risk_coverage, roc_recall_at_far,  # noqa: E402
                         threshold_for_far)
from mcb.scoring import BASELINE_NAMES, baseline_scores, states, variant_cbi  # noqa: E402

PROFILES = ["far05", "far10", "far20", "balanced"]
MIN_EVAL_ERRORS = 30


def pctile(score, ref):
    return np.searchsorted(np.sort(ref), score, side="right") / max(len(ref), 1)


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--lc-repeats", type=int, default=20, help="repeats for the calibration-size learning curve")
    args = ap.parse_args(argv)
    ctx = make_ctx(args)
    run, calib = load_run(ctx), load_calibration(ctx)
    manifest = ctx.manifest()
    cal, ev = run.split == "cal", run.split == "eval"
    y = run.y_err
    yE, yC = y[ev], y[cal]
    warnings = list(calib.get("warnings", []))
    if yE.sum() < MIN_EVAL_ERRORS:
        warnings.append(f"UNDERPOWERED: evaluation split has only {int(yE.sum())} errors (< {MIN_EVAL_ERRORS}); "
                        "confidence intervals are wide - treat conclusions as indicative only.")

    base = baseline_scores(run.P)
    cbi = {n: variant_cbi(v, run.P, run.meta) for n, v in calib["variants"].items()}
    score_all = {**{f"mcb_{n}": s for n, s in cbi.items()}, **base}
    mcb_names = [f"mcb_{n}" for n in cbi]

    # ---------------------------------------------------------------- summary
    summary = dict(n_total=len(y), n_cal=int(cal.sum()), n_eval=int(ev.sum()),
                   plain_accuracy_eval=float(1 - yE.mean()), plain_accuracy_cal=float(1 - yC.mean()),
                   eval_errors=int(yE.sum()), eval_correct=int((1 - yE).sum()),
                   subset=dict(dataset_size=manifest.get("dataset_size"), n_sampled=manifest.get("n_sampled"),
                               subset_fraction=manifest.get("subset_fraction")))
    cls_rows = []
    for ci_, cname in enumerate(ctx.class_names):
        m = ev & (run.labels == ci_)
        if m.sum():
            cls_rows.append(dict(cls=cname, n=int(m.sum()), errors=int(y[m].sum()), error_rate=float(y[m].mean())))
    summary["per_class_error"] = cls_rows

    # ------------------------------------------------- operating-point tables
    op = {p: [] for p in PROFILES}
    for p in PROFILES:
        for vn, v in calib["variants"].items():
            th = v["profiles"][p]
            st = states(cbi[vn][ev], th["theta_warn"], th["theta_trip"])
            op[p].append(dict(method=f"mcb_{vn}", rule="OPEN (trip)", theta=th["theta_trip"], **confusion(yE, st == 2)))
            op[p].append(dict(method=f"mcb_{vn}", rule="OPEN+HALF_OPEN", theta=th["theta_warn"], **confusion(yE, st >= 1)))
        for b in BASELINE_NAMES:
            thr = calib["baselines"][b][p]
            op[p].append(dict(method=b, rule="flag", theta=thr, **confusion(yE, base[b][ev] >= thr)))

    # ----------------------------------------------- threshold-free (AUROC etc)
    auc_rows = {}
    for n, s in score_all.items():
        pt, lo, hi = bootstrap_auroc(yE, s[ev], n_boot=args.n_boot, seed=ctx.seed)
        rc = risk_coverage(yE, s[ev])
        auc_rows[n] = dict(auroc=pt, ci=[lo, hi], **rc,
                           oracle_recall_at_far={f"{int(f * 100)}": roc_recall_at_far(yE, s[ev], f) for f in (0.05, 0.10, 0.20)})
    plain_aurc = float(yE.mean())
    base_only = {n: auc_rows[n]["auroc"] for n in BASELINE_NAMES}
    best_base = max(base_only, key=lambda k: -1 if np.isnan(base_only[k]) else base_only[k])
    diffs = {}
    for mn in mcb_names:
        for bn in [best_base, "msp", "entropy"]:
            d, lo, hi = bootstrap_auroc(yE, score_all[mn][ev], score_all[bn][ev], n_boot=args.n_boot, seed=ctx.seed)
            diffs[f"{mn}_minus_{bn}"] = dict(diff=d, ci=[lo, hi], significant_gain=bool(lo > 0),
                                              significant_loss=bool(hi < 0))

    # ------------------------------------------------------- complementarity
    sel = calib["variants"]["selected"]["profiles"]["far10"]
    st_sel = states(cbi["selected"][ev], sel["theta_warn"], sel["theta_trip"])
    mcb_flag = st_sel == 2
    comp = {}
    for bn in ("msp", "entropy"):
        bf = base[bn][ev] >= calib["baselines"][bn]["far10"]
        e = yE == 1
        comp[bn] = dict(errors=int(e.sum()), both=int((mcb_flag & bf & e).sum()),
                        mcb_only=int((mcb_flag & ~bf & e).sum()), baseline_only=int((~mcb_flag & bf & e).sum()),
                        neither=int((~mcb_flag & ~bf & e).sum()),
                        false_alarms_mcb=int((mcb_flag & ~e).sum()), false_alarms_baseline=int((bf & ~e).sum()))
    combined = {}
    for bn in ("entropy", "msp"):
        comb_all = (pctile(cbi["selected"], cbi["selected"][cal]) + pctile(base[bn], base[bn][cal])) / 2
        thr = threshold_for_far(comb_all[cal & (y == 0)], 0.10)
        a, lo, hi = bootstrap_auroc(yE, comb_all[ev], n_boot=args.n_boot, seed=ctx.seed)
        d1 = bootstrap_auroc(yE, comb_all[ev], cbi["selected"][ev], n_boot=args.n_boot, seed=ctx.seed)
        d2 = bootstrap_auroc(yE, comb_all[ev], base[bn][ev], n_boot=args.n_boot, seed=ctx.seed)
        combined[f"mcb+{bn}"] = dict(auroc=a, ci=[lo, hi], diff_vs_mcb=dict(diff=d1[0], ci=[d1[1], d1[2]]),
                                     diff_vs_baseline=dict(diff=d2[0], ci=[d2[1], d2[2]]),
                                     far10=confusion(yE, comb_all[ev] >= thr))

    # --------------------------------------------------------- error slices
    slices = {}
    for name, mask in (("high_conf_errors_c0>=0.75", (y == 1) & (run.c0 >= 0.75)),
                       ("high_conf_errors_c0>=0.90", (y == 1) & (run.c0 >= 0.90))):
        m = mask[ev]
        if m.sum() == 0:
            slices[name] = dict(n=0)
            continue
        rec = {}
        for vn, v in calib["variants"].items():
            th = v["profiles"]["far10"]
            st = states(cbi[vn][ev], th["theta_warn"], th["theta_trip"])
            rec[f"mcb_{vn}_open"] = float((st[m] == 2).mean())
            rec[f"mcb_{vn}_any"] = float((st[m] >= 1).mean())
        for b in ("msp", "entropy", "tta_entropy"):
            rec[b] = float((base[b][ev][m] >= calib["baselines"][b]["far10"]).mean())
        slices[name] = dict(n=int(m.sum()), recall_at_far10=rec)
    crit = critical_mask(ctx, run.labels)
    if crit is not None and (crit[ev] & (yE == 1)).sum() > 0:
        m = crit[ev] & (yE == 1)
        rec = {}
        for vn, v in calib["variants"].items():
            th = v["profiles"]["far10"]
            st = states(cbi[vn][ev], th["theta_warn"], th["theta_trip"])
            rec[f"mcb_{vn}_open"] = float((st[m] == 2).mean())
        for b in ("msp", "entropy"):
            rec[b] = float((base[b][ev][m] >= calib["baselines"][b]["far10"]).mean())
        slices["critical_class_errors"] = dict(n=int(m.sum()), recall_at_far10=rec)

    # ----------------------------------------------- per-class MCB recall
    th = sel
    for r in summary["per_class_error"]:
        m = ev & (run.labels == ctx.class_names.index(r["cls"])) & (y == 1)
        r["mcb_selected_open_recall_far10"] = (float((states(cbi["selected"][m], th["theta_warn"], th["theta_trip"]) == 2).mean())
                                               if m.sum() else None)

    # ----------------------------------------------------- A2: calibration study
    targets = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    curve = []
    for t in targets:
        row = dict(target_far=t)
        for name, s in (("mcb_selected", cbi["selected"]), ("mcb_flip_only", cbi.get("flip_only", cbi["selected"])),
                        ("msp", base["msp"]), ("entropy", base["entropy"])):
            thr = threshold_for_far(s[cal & (y == 0)], t)
            c = confusion(yE, s[ev] >= thr)
            row[name] = dict(realised_far=c["far"], recall=c["recall"])
        curve.append(row)
    rng = np.random.default_rng(ctx.seed)
    cal_idx = np.where(cal)[0]
    sizes = [s for s in (25, 50, 100, 200, 400, 800) if s < len(cal_idx)] + [len(cal_idx)]
    learning = []
    for k in sizes:
        fars, recs, fars_m, recs_m = [], [], [], []
        for _ in range(args.lc_repeats if k < len(cal_idx) else 1):
            sub = rng.choice(cal_idx, k, replace=False) if k < len(cal_idx) else cal_idx
            cor = sub[y[sub] == 0]
            if len(cor) < 10:
                continue
            for s, F, R in ((cbi["selected"], fars, recs), (base["msp"], fars_m, recs_m)):
                c = confusion(yE, s[ev] >= threshold_for_far(s[cor], 0.10))
                F.append(c["far"])
                R.append(c["recall"])
        if fars:
            learning.append(dict(n_cal=int(k), mcb_far_mean=float(np.mean(fars)), mcb_far_sd=float(np.std(fars)),
                                 mcb_recall_mean=float(np.mean(recs)), mcb_recall_sd=float(np.std(recs)),
                                 msp_far_mean=float(np.mean(fars_m)), msp_far_sd=float(np.std(fars_m)),
                                 msp_recall_mean=float(np.mean(recs_m)), msp_recall_sd=float(np.std(recs_m))))

    # ------------------------------------------------------------ per-sample CSV
    out = pd.DataFrame(dict(sample_id=run.ids, split=run.split, label=run.labels, pred=run.pred, c0=run.c0,
                            error=y, **{f"cbi_{n}": s for n, s in cbi.items()}, **{k: base[k] for k in base}))
    (ctx.run_dir / "A_batch").mkdir(exist_ok=True)
    out.to_csv(ctx.run_dir / "A_batch" / "per_sample.csv", index=False)

    # ----------------------------------------------------------- claims (neutral)
    claims = dict(
        best_baseline_by_auroc=best_base,
        mcb_selected_auroc_minus_best_baseline=diffs[f"mcb_selected_minus_{best_base}"],
        mcb_catches_errors_baselines_miss_at_far10={bn: comp[bn]["mcb_only"] for bn in comp},
        combined_improves_over_both=bool(
            combined["mcb+entropy"]["diff_vs_mcb"]["ci"][0] > 0 and combined["mcb+entropy"]["diff_vs_baseline"]["ci"][0] > 0),
    )
    results = dict(summary=summary, operating_points=op, threshold_free=auc_rows, plain_model_aurc=plain_aurc,
                   paired_auroc_differences=diffs, complementarity_far10=comp, combined_detectors=combined,
                   error_slices=slices, calibration_study=dict(operating_curve=curve, learning_curve=learning),
                   claims=claims, frozen_config_hash=manifest.get("calibration_hash"),
                   data_provenance=dict(n_requested=manifest.get("n_requested"), dataset_size=manifest.get("dataset_size"),
                                        underpowered=bool(yE.sum() < MIN_EVAL_ERRORS)))

    # ------------------------------------------------------------------ markdown
    md = [f"## 1. Data and plain-model performance", "",
          f"- Samples: **{summary['n_total']}** (calibration {summary['n_cal']} / evaluation {summary['n_eval']}); "
          f"dataset size {summary['subset']['dataset_size']}.",
          f"- Plain-model accuracy on evaluation split: **{pct(summary['plain_accuracy_eval'])}** "
          f"({summary['eval_errors']} errors = silent errors without a monitor).", ""]
    for p in PROFILES:
        tl = "balanced (Youden-optimal on calibration)" if p == "balanced" else f"target FAR {pct(FAR_PROFILES[p], 0)}"
        md += [f"## 2.{PROFILES.index(p) + 1} Operating point: {tl}", "",
               md_table(["Method", "Rule", "Recall [95% CI]", "FAR [95% CI]", "Flag rate", "Accuracy of auto-approved", "Silent errors left"],
                        [[r["method"], r["rule"], pct_ci(r["recall"], r["recall_ci"]), pct_ci(r["far"], r["far_ci"]),
                          pct(r["flag_rate"]), pct(r["accepted_accuracy"]), r["silent_errors"]] for r in op[p]]), ""]
    md += ["## 3. Threshold-free comparison (A1)", "",
           md_table(["Detector", "AUROC [95% CI]", "AURC (lower better)", "Acc @90% coverage", "Oracle recall @5/10/20% FAR"],
                    [[n, ci_str(r["auroc"], *r["ci"]), f3(r["aurc"]), pct(r["acc_at_cov_90"]),
                      " / ".join(pct(r["oracle_recall_at_far"][k], 0) for k in ("5", "10", "20"))]
                     for n, r in auc_rows.items()]), "",
           f"_Plain model (no rejection) risk = {f3(plain_aurc)}. Oracle recall thresholds on the evaluation data itself "
           f"(upper bound); deployable numbers are in section 2._", "",
           "### Paired AUROC differences (bootstrap)", "",
           md_table(["Comparison", "AUROC diff [95% CI]", "Verdict"],
                    [[k, ci_str(v["diff"], *v["ci"]),
                      "MCB significantly better" if v["significant_gain"] else
                      "MCB significantly worse" if v["significant_loss"] else "no significant difference"]
                     for k, v in diffs.items()]), "",
           f"_Best baseline is picked by evaluation AUROC (`{best_base}`), which is conservative against MCB._", "",
           "## 4. Complementarity at 10% FAR (selected MCB, OPEN) (A1)", "",
           md_table(["Baseline", "Errors", "Both catch", "MCB only", "Baseline only", "Neither", "False alarms MCB / baseline"],
                    [[bn, c["errors"], c["both"], c["mcb_only"], c["baseline_only"], c["neither"],
                      f"{c['false_alarms_mcb']} / {c['false_alarms_baseline']}"] for bn, c in comp.items()]), "",
           "### Combined detector (rank-average of calibration percentiles)", "",
           md_table(["Detector", "AUROC [95% CI]", "vs MCB alone", "vs baseline alone", "Recall / FAR @10% FAR"],
                    [[n, ci_str(c["auroc"], *c["ci"]), ci_str(c["diff_vs_mcb"]["diff"], *c["diff_vs_mcb"]["ci"]),
                      ci_str(c["diff_vs_baseline"]["diff"], *c["diff_vs_baseline"]["ci"]),
                      f"{pct(c['far10']['recall'])} / {pct(c['far10']['far'])}"] for n, c in combined.items()]), "",
           "## 5. Error slices (recall at the 10% FAR profile)", ""]
    for name, s in slices.items():
        if s.get("n", 0) == 0:
            md += [f"- **{name}**: no such errors in the evaluation split.", ""]
            continue
        md += [f"**{name}** (n={s['n']})", "",
               md_table(["Detector", "Recall"], [[k, pct(v)] for k, v in s["recall_at_far10"].items()]), ""]
    md += ["## 6. Calibration study (A2)", "",
           "### 6.1 Operating curve: target FAR -> realised FAR / recall on held-out evaluation data", "",
           md_table(["Target FAR", "MCB selected (FAR / recall)", "MCB flip-only", "MSP", "Entropy"],
                    [[pct(r["target_far"], 0)] + [f"{pct(r[k]['realised_far'])} / {pct(r[k]['recall'])}"
                                                  for k in ("mcb_selected", "mcb_flip_only", "msp", "entropy")] for r in curve]), "",
           "### 6.2 How much calibration data is needed? (target FAR 10%, mean +/- sd over random subsets)", "",
           md_table(["Calibration samples", "MCB realised FAR", "MCB recall", "MSP realised FAR", "MSP recall"],
                    [[r["n_cal"], f"{pct(r['mcb_far_mean'])} +/- {pct(r['mcb_far_sd'])}",
                      f"{pct(r['mcb_recall_mean'])} +/- {pct(r['mcb_recall_sd'])}",
                      f"{pct(r['msp_far_mean'])} +/- {pct(r['msp_far_sd'])}",
                      f"{pct(r['msp_recall_mean'])} +/- {pct(r['msp_recall_sd'])}"] for r in learning]), "",
           "## 7. Per-class view", "",
           md_table(["Class", "N", "Errors", "Error rate", "MCB recall (OPEN, 10% FAR)"],
                    [[r["cls"], r["n"], r["errors"], pct(r["error_rate"]), pct(r["mcb_selected_open_recall_far10"])
                      if r["mcb_selected_open_recall_far10"] is not None else "-"] for r in cls_rows])]
    save_result(ctx, "A_batch", "A_batch", "Test A - Full Batch Benchmark, Baseline Comparison & Calibration Study",
                results, "\n".join(md), warnings=warnings)


if __name__ == "__main__":
    main()
