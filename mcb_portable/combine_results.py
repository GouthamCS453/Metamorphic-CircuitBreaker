"""
combine_results.py - Merge the results of several members' runs into ONE report for the paper.

Usage
  python combine_results.py                          # scans results/*/
  python combine_results.py path/runA path/runB ...  # explicit run folders (e.g. collected from teammates)
  python combine_results.py --allow-incomplete       # also include underpowered / smoke runs (flagged)

Outputs  COMBINED/COMBINED.json  and  COMBINED/COMBINED.md
Results are reported PER (model, dataset) and summarised across runs; raw counts are never pooled across different models.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb import SCHEMA_VERSION  # noqa: E402
from mcb.io_utils import ROOT, clean, f3, md_table, pct, write_json  # noqa: E402


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def g(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k) if isinstance(d, dict) else None
    return default if d is None else d


def op_row(A, method, rule, profile="far10"):
    for r in g(A, "results", "operating_points", profile, default=[]):
        if r["method"] == method and r["rule"] == rule:
            return r
    return None


def summarise(run_dir: Path):
    man = load(run_dir / "run_manifest.json") or {}
    A = load(run_dir / "A_batch" / "results.json")
    B = load(run_dir / "B_external" / "results.json")
    C = load(run_dir / "C_shift" / "results.json")
    D = load(run_dir / "D_ablation" / "results.json")
    F = load(run_dir / "F_latency" / "results.json")
    G = load(run_dir / "G_attribution" / "results.json")
    cal = load(run_dir / "0_calibration" / "results.json")
    s = dict(run_id=man.get("run_id", run_dir.name), model=man.get("model"), dataset=man.get("dataset"),
             dataset_version=man.get("dataset_version"), schema_version=man.get("schema_version"),
             bank_id=man.get("bank_id"), n_sampled=man.get("n_sampled"), dataset_size=man.get("dataset_size"),
             warnings=[], tests_present=[k for k, v in dict(A=A, B=B, C=C, D=D, F=F, G=G).items() if v])
    if A:
        r = A["results"]
        sm = r["summary"]
        sel = op_row(A, "mcb_selected", "OPEN (trip)")
        sel_any = op_row(A, "mcb_selected", "OPEN+HALF_OPEN")
        flip = op_row(A, "mcb_flip_only", "OPEN (trip)")
        msp = op_row(A, "msp", "flag")
        ent = op_row(A, "entropy", "flag")
        tf = r["threshold_free"]
        best = r["claims"]["best_baseline_by_auroc"]
        dd = r["claims"]["mcb_selected_auroc_minus_best_baseline"]
        s["A"] = dict(
            n_eval=sm["n_eval"], eval_errors=sm["eval_errors"], plain_accuracy=sm["plain_accuracy_eval"],
            auroc_mcb_selected=tf["mcb_selected"]["auroc"], auroc_mcb_flip_only=g(tf, "mcb_flip_only", "auroc"),
            auroc_msp=tf["msp"]["auroc"], auroc_entropy=tf["entropy"]["auroc"],
            best_baseline=best, auroc_diff_vs_best_baseline=dd["diff"], auroc_diff_ci=dd["ci"],
            far10=dict(mcb_selected=sel and dict(recall=sel["recall"], far=sel["far"]),
                       mcb_selected_any=sel_any and dict(recall=sel_any["recall"], far=sel_any["far"]),
                       mcb_flip_only=flip and dict(recall=flip["recall"], far=flip["far"]),
                       msp=msp and dict(recall=msp["recall"], far=msp["far"]),
                       entropy=ent and dict(recall=ent["recall"], far=ent["far"])),
            mcb_only_errors_vs_msp=g(r, "complementarity_far10", "msp", "mcb_only"),
            msp_only_errors=g(r, "complementarity_far10", "msp", "baseline_only"),
            combined_auroc=g(r, "combined_detectors", "mcb+entropy", "auroc"),
            underpowered=g(r, "data_provenance", "underpowered", default=False))
        if s["A"]["underpowered"]:
            s["warnings"].append("test A underpowered (few evaluation errors)")
    if B:
        r = B["results"]
        i = r["interception"]
        s["B"] = dict(n=r["n_external"], plain_errors=i["errors"], caught_by_mcb=i["caught_by_mcb"], caught_by_msp=i["caught_by_msp"],
                      mcb_only=len(i["caught_by_mcb_missed_by_msp"]), msp_only=len(i["caught_by_msp_missed_by_mcb"]),
                      high_conf_errors=len(i["high_confidence_errors"]),
                      high_conf_errors_caught=len(i["high_confidence_errors_caught_by_mcb"]),
                      plain_accuracy=r["plain_accuracy"])
    if C:
        tr = C["results"]["trends"]
        s["C"] = dict(mean_spearman_mcb=float(np.nanmean([t["spearman_mcb_flag_rate"] or np.nan for t in tr.values()])),
                      mean_spearman_msp=float(np.nanmean([t["spearman_msp_flag_rate"] or np.nan for t in tr.values()])),
                      mean_mcb_flag_rate_change=float(np.mean([t["mcb_flag_rate_change"] for t in tr.values()])),
                      mean_msp_flag_rate_change=float(np.mean([t["msp_flag_rate_change"] for t in tr.values()])),
                      mean_accuracy_drop=float(np.mean([t["accuracy_drop"] for t in tr.values()])),
                      mean_confidence_drop=float(np.mean([t["confidence_drop"] for t in tr.values()])))
    if F:
        s["F"] = dict(total_ms=F["results"]["components"]["total"]["mean_ms"], base_ms=F["results"]["components"]["base"]["mean_ms"],
                      overhead=F["results"]["overhead_factor_vs_single_pass"], device=F["results"]["device"])
    if G:
        s["G"] = dict(accuracy=G["results"]["accuracy"], ci=G["results"]["accuracy_ci"], interpretation=G["results"]["interpretation"])
    if D:
        rows = {r["name"]: r for r in D["results"]["ablations"]}
        s["D"] = dict(signal_auroc={r["name"]: r["eval_auroc"] for r in D["results"]["ablations"] if r["group"] == "A1 signal type"},
                      flat_vs_hier=g(rows.get("flat (no hierarchy / no CFS)"), "diff_vs_reference"))
    if cal:
        v = cal["results"]["variants"]["selected"]
        s["calibration"] = dict(params=v["params"], theta_trip_far10=v["profiles"]["far10"]["theta_trip"],
                                nested_cv_auroc=cal["results"]["nested_cv"]["heldout_auroc_mean"])
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run folders (default: results/*)")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "COMBINED"))
    args = ap.parse_args()
    dirs = [Path(r) for r in args.runs] or sorted(p for p in (ROOT / "results").glob("*") if (p / "run_manifest.json").exists())
    if not dirs:
        raise SystemExit("No run folders found.")
    runs, excluded = [], []
    for d in dirs:
        s = summarise(d)
        if s["schema_version"] != SCHEMA_VERSION:
            excluded.append((s["run_id"], f"schema {s['schema_version']} != {SCHEMA_VERSION}"))
            continue
        if "A" not in s:
            excluded.append((s["run_id"], "no Test A results"))
            continue
        if s["A"]["underpowered"] and not args.allow_incomplete:
            excluded.append((s["run_id"], "underpowered (use --allow-incomplete to include)"))
            continue
        runs.append(s)
    banks = {r["bank_id"] for r in runs}
    notes = []
    if len(banks) > 1:
        notes.append(f"WARNING: runs used different transform banks {sorted(map(str, banks))}; results are not directly comparable.")

    def agg(fn):
        v = [fn(r) for r in runs]
        v = [x for x in v if x is not None and x == x]
        return dict(mean=float(np.mean(v)), median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)), n=len(v)) if v else None

    aggregate = dict(
        n_runs=len(runs),
        auroc_mcb_selected=agg(lambda r: r["A"]["auroc_mcb_selected"]),
        auroc_mcb_flip_only=agg(lambda r: r["A"]["auroc_mcb_flip_only"]),
        auroc_msp=agg(lambda r: r["A"]["auroc_msp"]),
        auroc_entropy=agg(lambda r: r["A"]["auroc_entropy"]),
        auroc_diff_vs_best_baseline=agg(lambda r: r["A"]["auroc_diff_vs_best_baseline"]),
        recall_far10_mcb=agg(lambda r: g(r["A"], "far10", "mcb_selected", "recall")),
        recall_far10_msp=agg(lambda r: g(r["A"], "far10", "msp", "recall")),
        recall_far10_flip_only=agg(lambda r: g(r["A"], "far10", "mcb_flip_only", "recall")),
        mcb_only_errors=agg(lambda r: r["A"]["mcb_only_errors_vs_msp"]),
        shift_spearman_mcb=agg(lambda r: g(r, "C", "mean_spearman_mcb")),
        shift_spearman_msp=agg(lambda r: g(r, "C", "mean_spearman_msp")),
        overhead_factor=agg(lambda r: g(r, "F", "overhead")))
    wins = sum(1 for r in runs if r["A"]["auroc_diff_ci"][0] is not None and r["A"]["auroc_diff_ci"][0] > 0)
    losses = sum(1 for r in runs if r["A"]["auroc_diff_ci"][1] is not None and r["A"]["auroc_diff_ci"][1] < 0)
    aggregate["auroc_vs_best_baseline_significant_wins_losses_ties"] = [wins, losses, len(runs) - wins - losses]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "COMBINED.json", dict(generated=datetime.now().isoformat(timespec="seconds"), schema_version=SCHEMA_VERSION,
                                            runs=runs, aggregate=aggregate, excluded=excluded, notes=notes))
    md = ["# Combined Results Across Models and Datasets", "",
          f"_Generated {datetime.now():%Y-%m-%d %H:%M} | runs included: {len(runs)} | excluded: {len(excluded)}_", ""]
    md += [f"> {n}" for n in notes] + ([""] if notes else [])
    if excluded:
        md += ["**Excluded runs:** " + "; ".join(f"`{a}` ({b})" for a, b in excluded), ""]
    md += ["## 1. Error detection on held-out data (Test A)", "",
           md_table(["Run (model / dataset)", "Eval N / errors", "Plain acc", "AUROC MCB (sel / flip)", "AUROC MSP / entropy",
                     "MCB - best baseline [95% CI]", "Recall@10%FAR MCB / MSP", "FAR MCB / MSP", "MCB-only errors"],
                    [[f"{r['model']} / {r['dataset']}", f"{r['A']['n_eval']} / {r['A']['eval_errors']}", pct(r["A"]["plain_accuracy"]),
                      f"{f3(r['A']['auroc_mcb_selected'])} / {f3(r['A']['auroc_mcb_flip_only'])}",
                      f"{f3(r['A']['auroc_msp'])} / {f3(r['A']['auroc_entropy'])}",
                      f"{f3(r['A']['auroc_diff_vs_best_baseline'])} [{f3(r['A']['auroc_diff_ci'][0])}, {f3(r['A']['auroc_diff_ci'][1])}]",
                      f"{pct(g(r['A'], 'far10', 'mcb_selected', 'recall'))} / {pct(g(r['A'], 'far10', 'msp', 'recall'))}",
                      f"{pct(g(r['A'], 'far10', 'mcb_selected', 'far'))} / {pct(g(r['A'], 'far10', 'msp', 'far'))}",
                      r["A"]["mcb_only_errors_vs_msp"]] for r in runs]), ""]
    ext = [r for r in runs if "B" in r]
    if ext:
        md += ["## 2. External images: plain model vs framework (Test B)", "",
               md_table(["Run", "N", "Plain acc", "Silent errors", "Caught by MCB", "Caught by softmax flag", "MCB-only", "Softmax-only", "High-conf errors caught"],
                        [[f"{r['model']} / {r['dataset']}", r["B"]["n"], pct(r["B"]["plain_accuracy"]), r["B"]["plain_errors"], r["B"]["caught_by_mcb"],
                          r["B"]["caught_by_msp"], r["B"]["mcb_only"], r["B"]["msp_only"],
                          f"{r['B']['high_conf_errors_caught']}/{r['B']['high_conf_errors']}"] for r in ext]), ""]
    sh = [r for r in runs if "C" in r]
    if sh:
        md += ["## 3. Behaviour under graded shift (Test C)", "",
               md_table(["Run", "Mean accuracy drop", "Mean confidence drop", "MCB flag-rate change (rho)", "Softmax flag-rate change (rho)"],
                        [[f"{r['model']} / {r['dataset']}", pct(r["C"]["mean_accuracy_drop"]), f3(r["C"]["mean_confidence_drop"]),
                          f"{pct(r['C']['mean_mcb_flag_rate_change'])} ({f3(r['C']['mean_spearman_mcb'], 2)})",
                          f"{pct(r['C']['mean_msp_flag_rate_change'])} ({f3(r['C']['mean_spearman_msp'], 2)})"] for r in sh]), ""]
    md += ["## 4. Calibrated configurations and cost", "",
           md_table(["Run", "Selected signal / weights / tau", "theta_trip (10% FAR)", "Nested-CV AUROC", "Latency (ms) / overhead", "Attribution acc (G)"],
                    [[f"{r['model']} / {r['dataset']}",
                      f"{g(r, 'calibration', 'params', 'signal')} / {g(r, 'calibration', 'params', 'weights_name')} / {g(r, 'calibration', 'params', 'tau_fam')}",
                      f3(g(r, "calibration", "theta_trip_far10"), 4), f3(g(r, "calibration", "nested_cv_auroc")),
                      f"{f3(g(r, 'F', 'total_ms'), 0)} / {f3(g(r, 'F', 'overhead'), 1)}x" if "F" in r else "n/a",
                      pct(g(r, "G", "accuracy")) if "G" in r else "n/a"] for r in runs]), "",
           "## 5. Summary across runs", ""]
    rows = []
    for k, v in aggregate.items():
        if isinstance(v, dict):
            rows.append([k, f3(v["mean"]), f3(v["median"]), f"{f3(v['min'])} - {f3(v['max'])}", v["n"]])
    md += [md_table(["Metric", "Mean", "Median", "Range", "Runs"], rows), "",
           f"MCB AUROC vs best baseline across runs (significant wins / losses / ties): **{wins} / {losses} / {len(runs) - wins - losses}**.", "",
           "_Aggregates are descriptive across independent model/dataset pairs; do not read them as one pooled experiment._"]
    (out / "COMBINED.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[OK] combined {len(runs)} run(s) -> {out / 'COMBINED.md'} and COMBINED.json")


if __name__ == "__main__":
    main()
