"""
run_C_shift.py - TEST C: graded distribution-shift test.

Degrades a subset of evaluation images with 6 realistic shifts (blur, noise, jpeg, darken, color_cast, rotate)
at 3 severities, re-runs the full bank on every degraded image, and tracks - as severity increases -
accuracy, softmax confidence, over-confidence, flag rates and error-detection quality of MCB vs plain softmax.
Thresholds are the FROZEN clean-data profile (default far10).  A second analysis shows how a small amount
of shifted-domain data can RE-CALIBRATE the thresholds (random-half calibration, repeated).

Usage:  python run_C_shift.py --n-shift 100
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb.analysis import base_parser, load_calibration, load_run, make_ctx  # noqa: E402
from mcb.inference import collect_probs  # noqa: E402
from mcb.io_utils import f3, md_table, pct, save_result  # noqa: E402
from mcb.metrics import auroc, confusion, spearman, threshold_for_far  # noqa: E402
from mcb.scoring import baseline_scores, split_probs, states, variant_cbi  # noqa: E402
from mcb.shifts import SHIFTS, make_shift  # noqa: E402
from mcb.transforms import bank_hash, build_bank  # noqa: E402


def eval_condition(P, labels, calib, meta, profile):
    pred, c0 = split_probs(P)
    y = (pred != labels).astype(int)
    base = baseline_scores(P)
    out = dict(n=int(len(y)), errors=int(y.sum()), accuracy=float(1 - y.mean()), mean_confidence=float(c0.mean()),
               overconfidence_gap=float(c0.mean() - (1 - y.mean())))
    have = y.sum() >= 1 and (1 - y).sum() >= 1
    for vn in ("selected", "flip_only"):
        if vn not in calib["variants"]:
            continue
        v = calib["variants"][vn]
        s = variant_cbi(v, P, meta)
        th = v["profiles"][profile]
        st = states(s, th["theta_warn"], th["theta_trip"])
        d = dict(flag_rate_open=float((st == 2).mean()), flag_rate_any=float((st >= 1).mean()), mean_cbi=float(s.mean()),
                 auroc=auroc(y, s) if have and y.sum() >= 3 else None)
        if have:
            c = confusion(y, st == 2)
            d.update(recall=c["recall"], far=c["far"])
        out[f"mcb_{vn}"] = d
    for b in ("msp", "entropy"):
        s = base[b]
        thr = calib["baselines"][b][profile]
        d = dict(flag_rate=float((s >= thr).mean()), auroc=auroc(y, s) if have and y.sum() >= 3 else None)
        if have:
            c = confusion(y, s >= thr)
            d.update(recall=c["recall"], far=c["far"])
        out[b] = d
    return out


def recalibration(P, labels, calib, meta, repeats, seed):
    """Calibrate far10 thresholds on a random half of the (shifted) samples, evaluate on the other half."""
    pred, c0 = split_probs(P)
    y = (pred != labels).astype(int)
    rng = np.random.default_rng(seed)
    base = baseline_scores(P)
    s_mcb = variant_cbi(calib["variants"]["selected"], P, meta)
    res = {"mcb": [], "msp": []}
    for _ in range(repeats):
        perm = rng.permutation(len(y))
        a, b = perm[: len(y) // 2], perm[len(y) // 2:]
        if (y[b] == 1).sum() < 2 or (y[a] == 0).sum() < 10:
            continue
        for name, s in (("mcb", s_mcb), ("msp", base["msp"])):
            thr = threshold_for_far(s[a][y[a] == 0], 0.10)
            c = confusion(y[b], s[b] >= thr)
            res[name].append((c["recall"], c["far"]))
    if not res["mcb"]:
        return None
    return {k: dict(recall=float(np.mean([r[0] for r in v])), far=float(np.mean([r[1] for r in v])), repeats=len(v))
            for k, v in res.items()}


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--n-shift", type=int, default=100, help="evaluation images to degrade")
    ap.add_argument("--shifts", default="all", help="comma list of shifts or 'all'")
    ap.add_argument("--profile", default="far10")
    ap.add_argument("--recal-repeats", type=int, default=30)
    ap.add_argument("--flush-every", type=int, default=25)
    args = ap.parse_args(argv)
    ctx = make_ctx(args)
    calib = load_calibration(ctx)
    run = load_run(ctx)
    ev_idx = np.where(run.split == "eval")[0]
    rng = np.random.default_rng(ctx.seed)
    pick = np.sort(rng.choice(ev_idx, min(args.n_shift, len(ev_idx)), replace=False))
    ids = [run.ids[i] for i in pick]
    paths = dict(zip(run.df["sample_id"], run.df["image_path"]))
    shifts = list(SHIFTS) if args.shifts == "all" else [s.strip() for s in args.shifts.split(",")]
    bank, bid = build_bank(1.0), bank_hash(1.0)
    model = ctx.cfg.load_model()
    loader = getattr(ctx.cfg, "load_image", None)

    store = {}
    if ctx.arrays_shift_npz.exists():
        z = np.load(ctx.arrays_shift_npz, allow_pickle=False)
        if str(z["bank_id"]) == bid and [str(x) for x in z["ids"]] == ids:
            store = {k: z[k] for k in z.files if k.startswith("probs__")}
            print(f"[shift] resuming with {len(store)} cached conditions")

    def save():
        np.savez(ctx.arrays_shift_npz, ids=np.array(ids), bank_id=np.array(bid), **store)

    for sname in shifts:
        for lvl in (1, 2, 3):
            key = f"probs__{sname}__{lvl}"
            if key in store:
                continue
            _, P, failed = collect_probs(model, [(i, paths[i]) for i in ids], bank,
                                         ctx.run_dir / "_tmp_shift_cache.npz", bid + f"_{sname}{lvl}",
                                         load_image=loader, preprocess=make_shift(sname, lvl),
                                         flush_every=args.flush_every, desc=f"{sname} L{lvl}")
            (ctx.run_dir / "_tmp_shift_cache.npz").unlink(missing_ok=True)
            if failed:
                print(f"[warn] {len(failed)} images failed under {sname} L{lvl}")
            store[key] = P.astype(np.float32)
            save()

    clean_P = run.P[pick]
    lab = run.labels[pick]
    conditions, rows_md = {}, []
    conditions["clean"] = dict(severity=0, **eval_condition(clean_P, lab, calib, run.meta, args.profile))
    trends = {}
    for sname in shifts:
        seq = [(0, clean_P)] + [(l, store[f"probs__{sname}__{l}"].astype(np.float64)) for l in (1, 2, 3)]
        flag_mcb, flag_msp, conf, acc = [], [], [], []
        for lvl, P in seq:
            if P.shape[0] != len(lab):
                continue  # failed images -> skip condition rather than misalign
            ec = conditions["clean"] if lvl == 0 else eval_condition(P, lab, calib, run.meta, args.profile)
            if lvl > 0:
                ec = dict(severity=lvl, **ec)
                ec["recalibrated_far10"] = recalibration(P, lab, calib, run.meta, args.recal_repeats, ctx.seed)
                conditions[f"{sname}__{lvl}"] = ec
            flag_mcb.append(ec["mcb_selected"]["flag_rate_open"])
            flag_msp.append(ec["msp"]["flag_rate"])
            conf.append(ec["mean_confidence"])
            acc.append(ec["accuracy"])
        sev = list(range(len(flag_mcb)))
        trends[sname] = dict(family=SHIFTS[sname]["family"], spearman_mcb_flag_rate=spearman(sev, flag_mcb),
                             spearman_msp_flag_rate=spearman(sev, flag_msp),
                             accuracy_drop=float(acc[0] - acc[-1]), confidence_drop=float(conf[0] - conf[-1]),
                             mcb_flag_rate_change=float(flag_mcb[-1] - flag_mcb[0]),
                             msp_flag_rate_change=float(flag_msp[-1] - flag_msp[0]))
    rec_rows = []
    for k, c in conditions.items():
        if k == "clean":
            continue
        rc = c.get("recalibrated_far10")
        rec_rows.append([k, f"{pct(c['mcb_selected'].get('recall'))} / {pct(c['mcb_selected'].get('far'))}",
                         f"{pct(c['msp'].get('recall'))} / {pct(c['msp'].get('far'))}",
                         "n/a" if not rc else f"{pct(rc['mcb']['recall'])} / {pct(rc['mcb']['far'])}",
                         "n/a" if not rc else f"{pct(rc['msp']['recall'])} / {pct(rc['msp']['far'])}"])

    warnings = list(calib.get("warnings", []))
    if len(ids) < 80:
        warnings.append(f"Only {len(ids)} images were degraded; per-condition rates are noisy.")
    results = dict(n_images=len(ids), profile=args.profile, shifts=shifts, conditions=conditions, trends=trends,
                   notes="Accuracy here is on a random (not error-enriched) subset of the evaluation split.")
    md = ["## 1. Setup", "",
          f"{len(ids)} evaluation images, {len(shifts)} shifts x 3 severities, thresholds frozen from clean calibration "
          f"(profile `{args.profile}`). Flag rate = share of images flagged OPEN (MCB) or above the frozen threshold (softmax).", "",
          "## 2. Trend with severity (Spearman rho of flag rate vs severity 0-3)", "",
          md_table(["Shift", "Family", "Accuracy drop", "Confidence drop", "MCB flag rate: change / rho", "Softmax flag rate: change / rho"],
                   [[s, t["family"], pct(t["accuracy_drop"]), f3(t["confidence_drop"]),
                     f"{pct(t['mcb_flag_rate_change'])} / {f3(t['spearman_mcb_flag_rate'], 2)}",
                     f"{pct(t['msp_flag_rate_change'])} / {f3(t['spearman_msp_flag_rate'], 2)}"] for s, t in trends.items()]), "",
          "_A confidence drop far smaller than the accuracy drop means the model stays over-confident under shift._", "",
          "## 3. Per-condition detail", "",
          md_table(["Condition", "Acc", "Mean conf", "Over-conf gap", "MCB flag rate", "Softmax flag rate", "AUROC MCB", "AUROC softmax"],
                   [[k, pct(c["accuracy"]), f3(c["mean_confidence"]), f3(c["overconfidence_gap"]),
                     pct(c["mcb_selected"]["flag_rate_open"]), pct(c["msp"]["flag_rate"]),
                     f3(c["mcb_selected"]["auroc"]) if c["mcb_selected"]["auroc"] is not None else "n/a",
                     f3(c["msp"]["auroc"]) if c["msp"]["auroc"] is not None else "n/a"] for k, c in conditions.items()]), "",
          "## 4. Frozen-clean vs re-calibrated thresholds (recall / FAR at target 10% FAR)", "",
          "Re-calibration derives thresholds from a random half of the shifted samples and scores the other half "
          f"(mean of up to {args.recal_repeats} repeats).", "",
          md_table(["Condition", "MCB frozen", "Softmax frozen", "MCB re-calibrated", "Softmax re-calibrated"], rec_rows)]
    save_result(ctx, "C_shift", "C_shift", "Test C - Graded Distribution-Shift Test", results, "\n".join(md), warnings=warnings)


if __name__ == "__main__":
    main()
