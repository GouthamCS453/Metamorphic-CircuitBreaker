"""
run_F_latency.py - TEST F: deployment cost of the circuit breaker.

Times, per image: base forward pass | variant generation | batched forward over all variants | CBI scoring |
end-to-end.  Reports mean, sd, p50, p95 and the overhead factor vs a plain single forward pass.

Usage:  python run_F_latency.py --runs 30 --warmup 5
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb.analysis import base_parser, load_calibration, make_ctx  # noqa: E402
from mcb.inference import build_variants, default_loader  # noqa: E402
from mcb.io_utils import environment, f3, md_table, save_result  # noqa: E402
from mcb.scoring import compute_cbi, signals, split_probs  # noqa: E402
from mcb.transforms import bank_meta, build_bank  # noqa: E402


def stats(v):
    v = np.asarray(v) * 1000.0
    return dict(mean_ms=float(v.mean()), sd_ms=float(v.std()), p50_ms=float(np.percentile(v, 50)),
                p95_ms=float(np.percentile(v, 95)), n=int(len(v)))


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args(argv)
    ctx = make_ctx(args)
    calib = load_calibration(ctx)
    samples = pd.read_csv(ctx.samples_csv)
    ev = samples[samples["split"] == "eval"]
    paths = ev["image_path"].tolist()
    rng = np.random.default_rng(ctx.seed)
    model = ctx.cfg.load_model()
    sync = getattr(model, "sync", lambda: None)
    loader = getattr(ctx.cfg, "load_image", None) or default_loader
    bank, meta = build_bank(1.0), bank_meta(build_bank(1.0))
    params = calib["variants"]["selected"]["params"]
    T = dict(base=[], generate=[], batch=[], score=[], total=[])
    for it in range(args.warmup + args.runs):
        img = loader(paths[int(rng.integers(len(paths)))])
        t0 = time.perf_counter()
        model.predict_proba([img])
        sync()
        t1 = time.perf_counter()
        variants, _ = build_variants(img, bank)
        t2 = time.perf_counter()
        P = model.predict_proba(variants)
        sync()
        t3 = time.perf_counter()
        P3 = P[None]
        _, c0 = split_probs(P3)
        compute_cbi(signals(P3, params["signal"]), c0, meta, params)
        t4 = time.perf_counter()
        if it >= args.warmup:
            T["base"].append(t1 - t0)
            T["generate"].append(t2 - t1)
            T["batch"].append(t3 - t2)
            T["score"].append(t4 - t3)
            T["total"].append((t2 - t1) + (t3 - t2) + (t4 - t3))
    res = {k: stats(v) for k, v in T.items()}
    overhead = res["total"]["mean_ms"] / max(res["base"]["mean_ms"], 1e-9)
    device = getattr(model, "device_name", "unknown")
    results = dict(device=device, runs=args.runs, warmup=args.warmup, n_variants=len(bank) + 1, components=res,
                   overhead_factor_vs_single_pass=overhead, environment=environment())
    names = dict(base="Single forward pass (plain model)", generate=f"Generate {len(bank)} variants",
                 batch=f"Batched forward ({len(bank) + 1} images)", score="CBI scoring + state machine",
                 total="MCB total (generate + batch + score)")
    md = [f"Device: **{device}**; {args.runs} timed runs after {args.warmup} warm-up runs; {len(bank) + 1} images per breaker call.", "",
          md_table(["Component", "Mean (ms)", "SD", "P50", "P95"],
                   [[names[k], f3(res[k]["mean_ms"], 2), f3(res[k]["sd_ms"], 2), f3(res[k]["p50_ms"], 2), f3(res[k]["p95_ms"], 2)]
                    for k in ("base", "generate", "batch", "score", "total")]), "",
          f"**Overhead vs a plain single forward pass: {overhead:.1f}x.** The scoring logic itself costs {res['score']['mean_ms']:.2f} ms; "
          "cost is dominated by the batched forward over the variants (can be reduced with a smaller bank, fp16, or "
          "running the breaker only on predictions the plain model is not already very sure about)."]
    save_result(ctx, "F_latency", "F_latency", "Test F - Latency", results, "\n".join(md))


if __name__ == "__main__":
    main()
