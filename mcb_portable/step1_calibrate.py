"""
step1_calibrate.py - STAGE 0: find the ideal parameters and thresholds, then FREEZE them.

Uses the CALIBRATION split only.  Writes calibrated_config.json, which every later test loads.
Searches signal type (flip/drift/hybrid), CBI weights and tau_fam; derives theta_trip / theta_warn for
target false-alarm profiles (far05 / far10 / far20) plus a Youden-optimal 'balanced' profile; calibrates
the baseline detectors the same way; estimates generalisation with nested cross-validation.

Two variants are frozen:
  selected  : best configuration over all signals (the proposed calibrated framework)
  flip_only : best configuration restricted to the ORIGINAL binary-flip signal (paper's original MCB)

Usage:  python step1_calibrate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb.analysis import base_parser, load_run, make_ctx  # noqa: E402
from mcb.calibration import (FAR_PROFILES, SIGNALS, TAU_GRID, WEIGHT_CONFIGS, baseline_thresholds,  # noqa: E402
                             build_grid, make_profiles, nested_cv, select_config)
from mcb.io_utils import f3, md_table, pct, save_result, sha256_obj, write_json  # noqa: E402
from mcb.metrics import auroc, bootstrap_auroc  # noqa: E402
from mcb.scoring import baseline_scores  # noqa: E402


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--signals", default=",".join(SIGNALS), help="comma list from flip,drift,hybrid")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args(argv)
    ctx = make_ctx(args)
    sigs = tuple(s.strip() for s in args.signals.split(",") if s.strip())

    run = load_run(ctx)
    cal = run.split == "cal"
    P, y = run.P[cal], run.y_err[cal]
    n_err, n_cor = int(y.sum()), int((1 - y).sum())
    warnings = []
    if n_err < 10 or n_cor < 30:
        warnings.append(f"LOW DATA: calibration split has only {n_err} errors / {n_cor} correct samples; "
                        "thresholds and the selected configuration are unreliable. Use a larger --n.")
    if n_err == 0 or n_cor == 0:
        raise SystemExit("Calibration split needs both errors and correct predictions.")

    configs, cbi_mat = build_grid(P, run.meta, sigs=sigs)
    aucs = np.array([auroc(y, cbi_mat[i]) for i in range(len(configs))])
    base_cal = baseline_scores(P)
    ref = {k: base_cal[k] for k in ("msp", "entropy")}
    cv = nested_cv(cbi_mat, y, ref, folds=args.folds, repeats=args.repeats, seed=ctx.seed)

    variants, md_var = {}, []
    for vname, allowed in (("selected", None), ("flip_only", ("flip",))):
        if allowed and "flip" not in sigs:
            continue
        sel = select_config(configs, aucs, y, cbi_mat, allowed, seed=ctx.seed)
        c = configs[sel["index"]]
        profiles = make_profiles(cbi_mat[sel["index"]], y, seed=ctx.seed)
        auc, lo, hi = bootstrap_auroc(y, cbi_mat[sel["index"]], n_boot=300, seed=ctx.seed)
        variants[vname] = dict(
            params=dict(signal=c["signal"], weights_name=c["wname"], alpha=c["alpha"], beta=c["beta"],
                        gamma=c["gamma"], tau_fam=c["tau_fam"], mean_w=0.70),
            selection=sel, calibration_auroc=dict(point=auc, lo=lo, hi=hi), profiles=profiles)
        md_var.append((vname, c, sel, (auc, lo, hi), profiles))

    baselines = baseline_thresholds(base_cal, y)
    order = np.argsort(-np.nan_to_num(aucs, nan=-1))[:12]
    grid_rows = [[configs[i]["signal"], configs[i]["wname"], configs[i]["tau_fam"], f3(aucs[i])] for i in order]
    pd.DataFrame([{**{k: v for k, v in c.items() if k != "tau_idx"}, "auroc_cal": a}
                  for c, a in zip(configs, aucs)]).to_csv(ctx.run_dir / "calibration_grid.csv", index=False)

    cfg_json = dict(
        schema_version="1.0", run_id=ctx.run_id, frozen=True,
        calibration_data=dict(n=int(cal.sum()), n_errors=n_err, n_correct=n_cor,
                              sample_ids_hash=sha256_obj(sorted(np.array(run.ids)[cal].tolist()))),
        search_space=dict(signals=list(sigs), weight_configs=[w[0] for w in WEIGHT_CONFIGS], tau_grid=TAU_GRID,
                          far_profiles=FAR_PROFILES, mean_w=0.70),
        variants=variants, baselines=baselines, nested_cv=cv, warnings=warnings)
    write_json(ctx.calibration_json, cfg_json)
    ctx.update_manifest(calibration_hash=sha256_obj(cfg_json["variants"]))

    md = ["## Stage 0 - Parameter search and threshold calibration", "",
          f"Calibration split: **{int(cal.sum())}** samples ({n_err} base-model errors, {n_cor} correct). "
          f"Evaluation data is never touched here. Search space: signals `{', '.join(sigs)}`, "
          f"{len(WEIGHT_CONFIGS)} weight configs, {len(TAU_GRID)} tau_fam values = {len(configs)} configurations.", "",
          "### Selected configurations", ""]
    for vname, c, sel, (auc, lo, hi), prof in md_var:
        md += [f"**`{vname}`** -> signal `{c['signal']}`, weights `{c['wname']}` "
               f"(alpha={c['alpha']}, beta={c['beta']}, gamma={c['gamma']}), tau_fam={c['tau_fam']}; "
               f"calibration AUROC {f3(auc)} [{f3(lo)}, {f3(hi)}]; "
               f"{sel['n_candidates_within_1se']} configs within 1 SE (SE={f3(sel['se'], 4)}), "
               f"plateau AUROC {f3(sel['plateau_auroc'])} (best single config {f3(sel['best_auroc'])}).", ""]
        rows = []
        for pn, p in prof.items():
            cc = p["calibration_confusion"]
            ci = p["theta_trip_ci"]
            rows.append([pn, "-" if p["target_far"] is None else pct(p["target_far"], 0), f3(p["theta_trip"], 4),
                         "-" if ci is None else f"[{f3(ci[0], 4)}, {f3(ci[1], 4)}]", f3(p["theta_warn"], 4),
                         pct(cc["recall"]), pct(cc["far"])])
        md += [md_table(["Profile", "Target FAR", "theta_trip", "theta_trip 95% CI", "theta_warn",
                         "Cal recall", "Cal FAR"], rows), ""]
    md += ["### Honest generalisation estimate (nested cross-validation)", "",
           f"Selecting the best configuration on training folds and scoring on the held-out fold "
           f"({cv['n_folds_used']} folds): CBI held-out AUROC = **{f3(cv['heldout_auroc_mean'])} +/- "
           f"{f3(cv['heldout_auroc_sd'])}** vs MSP {f3(cv['reference_heldout_auroc']['msp'])} and entropy "
           f"{f3(cv['reference_heldout_auroc']['entropy'])} on the same folds. "
           f"Same configuration chosen in {pct(cv['selection_stability'], 0)} of folds.", "",
           "### Top configurations by calibration AUROC", "",
           md_table(["Signal", "Weights", "tau_fam", "Cal AUROC"], grid_rows), "",
           "Full grid: `calibration_grid.csv`. Thresholds for baselines (msp, entropy, margin, tta_entropy, "
           "tta_var, kflip) are derived the same way and stored in `calibrated_config.json`."]
    results = dict(calibration_data=cfg_json["calibration_data"], variants=variants, nested_cv=cv,
                   baselines=baselines, search_space=cfg_json["search_space"],
                   top_configs=[dict(signal=configs[i]["signal"], weights=configs[i]["wname"],
                                     tau_fam=configs[i]["tau_fam"], auroc=aucs[i]) for i in order])
    save_result(ctx, "STAGE0_calibration", "0_calibration", "Stage 0 - Calibration & Parameter Selection",
                results, "\n".join(md), warnings=warnings)


if __name__ == "__main__":
    main()
