"""
run_D_ablation.py - TEST D: ablations of the framework's design choices.

Each specification re-tunes ONLY tau_fam on the calibration split (fair to every variant), then is evaluated on
the evaluation split with a far10 threshold derived from calibration.  Differences are paired-bootstrap
AUROC differences against the reference (calibrated 'selected' configuration).

Groups: signal type | hierarchical vs flat | weight configs | component removal | level-2 mix |
        severity weighting | family removal | (optional) bank strength.

Usage:  python run_D_ablation.py
        python run_D_ablation.py --bank-scales 0.5,1.5 --bank-n 300     # adds the bank-strength sweep (re-runs the model)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb.analysis import base_parser, load_calibration, load_run, make_ctx  # noqa: E402
from mcb.calibration import TAU_GRID, WEIGHT_CONFIGS  # noqa: E402
from mcb.inference import collect_probs  # noqa: E402
from mcb.io_utils import ci_str, f3, md_table, pct, save_result  # noqa: E402
from mcb.metrics import auroc, bootstrap_auroc, confusion, threshold_for_far  # noqa: E402
from mcb.scoring import compute_cbi, signals, split_probs  # noqa: E402
from mcb.transforms import FAMILIES, bank_hash, bank_meta, build_bank  # noqa: E402

UNIFORM = {"mild": 1.0, "moderate": 1.0, "severe": 1.0}
INVERTED = {"mild": 0.3, "moderate": 0.6, "severe": 1.0}


def make_specs(ref):
    p = ref["params"]
    base = dict(signal=p["signal"], alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"], mean_w=0.7,
                sev_weights=None, flat=False, families=None)
    S = []

    def add(group, name, **kw):
        S.append(dict(group=group, name=name, **{**base, **kw}))

    add("reference", f"reference ({p['signal']}, {p['weights_name']})")
    for sg in ("flip", "drift", "hybrid"):
        add("A1 signal type", f"signal = {sg}", signal=sg)
    add("A2 structure", "flat (no hierarchy / no CFS)", flat=True)
    for wn, a, b, g in WEIGHT_CONFIGS:
        add("A3 weights", f"weights = {wn} ({a}/{b}/{g})", alpha=a, beta=b, gamma=g)
    add("A4 components", "no cross-family spread (beta=0)", alpha=0.85, beta=0.0, gamma=0.15)
    add("A4 components", "no softmax term (gamma=0)", alpha=0.60, beta=0.40, gamma=0.0)
    add("A4 components", "softmax only (1-c0)", alpha=0.0, beta=0.0, gamma=1.0)
    add("A4 components", "peak instability only", alpha=1.0, beta=0.0, gamma=0.0)
    add("A4 components", "spread only", alpha=0.0, beta=1.0, gamma=0.0)
    for mw in (1.0, 0.5, 0.0):
        add("A5 level-2 mix", f"mean weight = {mw}" + (" (pure mean)" if mw == 1 else " (pure peak)" if mw == 0 else ""), mean_w=mw)
    add("A6 severity weights", "uniform (1/1/1)", sev_weights=UNIFORM)
    add("A6 severity weights", "inverted (severe highest)", sev_weights=INVERTED)
    for f in FAMILIES:
        add("A7 family removal", f"drop {f}", families=[x for x in FAMILIES if x != f])
    return S


def eval_spec(spec, Pc, yc, Pe, ye, meta, ref_scores_e=None, n_boot=500, seed=0):
    sig_c, sig_e = signals(Pc, spec["signal"]), signals(Pe, spec["signal"])
    _, c0c = split_probs(Pc)
    _, c0e = split_probs(Pe)
    params = {k: spec[k] for k in ("alpha", "beta", "gamma", "mean_w", "sev_weights", "flat", "families")}
    best_tau, best_auc = TAU_GRID[0], -1.0
    for tau in ([None] if spec["flat"] else TAU_GRID):
        s = compute_cbi(sig_c, c0c, meta, {**params, "tau_fam": tau if tau is not None else 0.2})["cbi"]
        a = auroc(yc, s)
        if a > best_auc:
            best_auc, best_tau = a, (tau if tau is not None else 0.2)
    sc = compute_cbi(sig_c, c0c, meta, {**params, "tau_fam": best_tau})["cbi"]
    se = compute_cbi(sig_e, c0e, meta, {**params, "tau_fam": best_tau})["cbi"]
    thr = threshold_for_far(sc[yc == 0], 0.10)
    conf = confusion(ye, se >= thr)
    a, lo, hi = bootstrap_auroc(ye, se, n_boot=n_boot, seed=seed)
    out = dict(tau_fam=best_tau, cal_auroc=best_auc, eval_auroc=a, eval_auroc_ci=[lo, hi],
               recall_far10=conf["recall"], far_far10=conf["far"])
    if ref_scores_e is not None:
        d, dl, dh = bootstrap_auroc(ye, se, ref_scores_e, n_boot=n_boot, seed=seed)
        out["diff_vs_reference"] = dict(diff=d, ci=[dl, dh])
    return out, se


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--variant", default="selected", help="calibrated variant used as the reference")
    ap.add_argument("--bank-scales", default="", help="comma list, e.g. 0.5,1.5 (re-runs the model)")
    ap.add_argument("--bank-n", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args(argv)
    ctx = make_ctx(args)
    run, calib = load_run(ctx), load_calibration(ctx)
    ref = calib["variants"][args.variant]
    cal, ev = run.split == "cal", run.split == "eval"
    y = run.y_err
    Pc, Pe, yc, ye = run.P[cal], run.P[ev], y[cal], y[ev]
    warnings = list(calib.get("warnings", []))
    if ye.sum() < 30:
        warnings.append(f"UNDERPOWERED: only {int(ye.sum())} evaluation errors; ablation differences are unlikely to be significant.")

    specs = make_specs(ref)
    rows, ref_scores = [], None
    for sp in specs:
        res, se = eval_spec(sp, Pc, yc, Pe, ye, run.meta, ref_scores, args.n_boot, ctx.seed)
        if sp["group"] == "reference":
            ref_scores = se
        rows.append(dict(group=sp["group"], name=sp["name"], **res))

    bank_rows = []
    if args.bank_scales.strip():
        scales = [float(s) for s in args.bank_scales.split(",")] + [1.0]
        rng = np.random.default_rng(ctx.seed)
        half = args.bank_n // 2
        sel_cal = rng.choice(np.where(cal)[0], min(half, cal.sum()), replace=False)
        sel_ev = rng.choice(np.where(ev)[0], min(half, ev.sum()), replace=False)
        sel = np.concatenate([sel_cal, sel_ev])
        ids = [run.ids[i] for i in sel]
        paths = dict(zip(run.df["sample_id"], run.df["image_path"]))
        model = ctx.cfg.load_model()
        ref_spec = specs[0]
        ref_e = None
        for sc in sorted(set(scales)):
            if sc == 1.0:
                P_all = run.P[sel]
            else:
                bank = build_bank(sc)
                cp = ctx.run_dir / f"_bank_scale_{sc}.npz"
                _, P_all, _ = collect_probs(model, [(i, paths[i]) for i in ids], bank, cp, bank_hash(sc),
                                            load_image=getattr(ctx.cfg, "load_image", None), desc=f"bank x{sc}")
            m = bank_meta(build_bank(sc))
            yy = (P_all[:, 0].argmax(1) != run.labels[sel]).astype(int)
            n_c = len(sel_cal)
            res, se = eval_spec(ref_spec, P_all[:n_c], yy[:n_c], P_all[n_c:], yy[n_c:], m, ref_e, args.n_boot, ctx.seed)
            bank_rows.append(dict(scale=sc, n=len(ids), eval_errors=int(yy[n_c:].sum()), **res))

    results = dict(reference=args.variant, ablations=rows, bank_strength=bank_rows,
                   protocol="tau_fam re-tuned on calibration split for every row; far10 threshold from calibration; "
                            "AUROC differences are paired bootstrap vs the reference row.")
    md = ["## Protocol", "",
          "Every row re-tunes only `tau_fam` on the calibration split, then reports held-out evaluation AUROC and the recall/FAR obtained "
          "with a calibration-derived 10%-FAR threshold. *Diff vs reference* is a paired bootstrap difference; a CI that excludes 0 "
          "marks a real effect.", ""]
    groups = []
    for r in rows:
        if r["group"] not in groups:
            groups.append(r["group"])
    for g in groups:
        md += [f"### {g}", "",
               md_table(["Specification", "tau_fam", "Eval AUROC [95% CI]", "Diff vs reference [95% CI]", "Recall @10% FAR", "Realised FAR"],
                        [[r["name"], r["tau_fam"] if "flat" not in r["name"] else "-", ci_str(r["eval_auroc"], *r["eval_auroc_ci"]),
                          "-" if "diff_vs_reference" not in r else ci_str(r["diff_vs_reference"]["diff"], *r["diff_vs_reference"]["ci"]),
                          pct(r["recall_far10"]), pct(r["far_far10"])] for r in rows if r["group"] == g]), ""]
    if bank_rows:
        md += ["### A8 Transform-bank strength (range sensitivity)", "",
               md_table(["Bank scale", "N", "Eval errors", "Eval AUROC [95% CI]", "Recall @10% FAR", "Realised FAR"],
                        [[r["scale"], r["n"], r["eval_errors"], ci_str(r["eval_auroc"], *r["eval_auroc_ci"]),
                          pct(r["recall_far10"]), pct(r["far_far10"])] for r in bank_rows]), "",
               "_Scale multiplies every perturbation magnitude (1.0 = nominal bank). The reference configuration is re-calibrated per scale._"]
    save_result(ctx, "D_ablation", "D_ablation", "Test D - Ablations", results, "\n".join(md), warnings=warnings)


if __name__ == "__main__":
    main()
