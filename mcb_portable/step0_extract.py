"""
step0_extract.py - ONE inference pass over the dataset (and the external images).

Samples the dataset (stratified, optional), splits it into a CALIBRATION half and an EVALUATION half,
runs the model on the original image + all 25 metamorphic variants, and caches the full softmax
tensor P (N, 26, C) in arrays.npz.  Everything else is offline numpy on that file.

Usage
  python step0_extract.py                      # default: N from member_config (or all)
  python step0_extract.py --n 1500             # stratified subset of 1500 images
  python step0_extract.py --only external      # only the external images
Resumable: re-running continues from the cache.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb.analysis import base_parser, make_ctx  # noqa: E402
from mcb.inference import collect_probs  # noqa: E402
from mcb.io_utils import ROOT, sha256_obj  # noqa: E402
from mcb.transforms import bank_hash, build_bank  # noqa: E402


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.copy()
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby("label"):
        k = max(1, int(round(n * len(g) / len(df))))
        parts.append(g.iloc[rng.permutation(len(g))[:min(k, len(g))]])
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def assign_split(df: pd.DataFrame, seed: int, cal_frac: float = 0.5) -> pd.DataFrame:
    """Stratified (and group-aware if a `group` column exists) calibration/evaluation split."""
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    df["split"] = "eval"
    if "group" in df.columns:
        groups = df.groupby("group")["label"].first()
        for _, gs in groups.groupby(groups):
            names = list(gs.index)
            perm = rng.permutation(len(names))
            ncal = int(round(len(names) * cal_frac))
            for j in perm[:ncal]:
                df.loc[df["group"] == names[j], "split"] = "cal"
    else:
        for _, idx in df.groupby("label").groups.items():
            idx = list(idx)
            perm = rng.permutation(len(idx))
            ncal = int(round(len(idx) * cal_frac))
            for j in perm[:ncal]:
                df.loc[idx[j], "split"] = "cal"
    return df


def load_external(ctx):
    cfg = ctx.cfg
    ext_dir = Path(getattr(cfg, "EXTERNAL_DIR", ROOT / "external"))
    labels_csv = ext_dir / "labels.csv"
    img_dir = ext_dir / "images"
    if not img_dir.exists():
        img_dir = ext_dir          # flat layout: images next to labels.csv
    if not labels_csv.exists() or not img_dir.exists():
        return None
    lab = pd.read_csv(labels_csv)
    rows = []
    for r in lab.itertuples():
        v = str(r.label).strip()
        if v.lstrip("-").isdigit():
            li = int(v)
        elif v in ctx.class_names:
            li = ctx.class_names.index(v)
        else:
            print(f"[warn] unknown label '{v}' for {r.filename}; skipped")
            continue
        p = img_dir / str(r.filename)
        if p.exists():
            rows.append(dict(sample_id=Path(r.filename).stem, image_path=str(p), label=li))
        else:
            print(f"[warn] missing external image {p}")
    return pd.DataFrame(rows)


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--n", type=int, default=None, help="stratified subset size (0 = all images)")
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--only", choices=["dataset", "external", "both"], default="both")
    ap.add_argument("--flush-every", type=int, default=25)
    args = ap.parse_args(argv)
    ctx = make_ctx(args)
    cfg = ctx.cfg
    n_req = args.n if args.n is not None else int(getattr(cfg, "N_SAMPLES", 0))

    model = cfg.load_model()
    bank = build_bank(1.0)
    bid = bank_hash(1.0)
    loader = getattr(cfg, "load_image", None)
    manifest_updates = dict(bank_id=bid, n_transforms=len(bank), bank_scale=1.0)

    if args.only in ("dataset", "both"):
        full = cfg.get_dataset().copy()
        if "image_path" not in full.columns or "label" not in full.columns:
            sys.exit("get_dataset() must return a DataFrame with columns image_path,label (optional: group)")
        full["label"] = full["label"].astype(int)
        if "sample_id" not in full.columns:
            full["sample_id"] = [Path(p).stem for p in full["image_path"]]
        if full["sample_id"].duplicated().any():
            full["sample_id"] = full["image_path"].astype(str)
        full["sample_id"] = full["sample_id"].astype(str)
        sub = stratified_sample(full, n_req, ctx.seed)
        sub = assign_split(sub, ctx.seed, args.cal_frac)
        id_hash = sha256_obj(sorted(sub["sample_id"].tolist()))
        old = ctx.manifest()
        if old.get("sample_ids_hash") not in (None, id_hash) and ctx.arrays_npz.exists():
            print("[cache] sample set changed -> discarding old arrays.npz and calibration")
            ctx.arrays_npz.unlink()
            if ctx.calibration_json.exists():
                ctx.calibration_json.unlink()
        sub.to_csv(ctx.samples_csv, index=False)
        items = list(zip(sub["sample_id"], sub["image_path"]))
        ids, P, failed = collect_probs(model, items, bank, ctx.arrays_npz, bid, load_image=loader,
                                       flush_every=args.flush_every, desc="dataset")
        manifest_updates.update(
            sample_ids_hash=id_hash, dataset_size=int(len(full)), n_requested=int(n_req),
            n_sampled=int(len(sub)), n_processed=int(len(ids)), n_failed=int(len(failed)),
            subset_fraction=float(len(sub) / max(len(full), 1)),
            n_cal=int((sub["split"] == "cal").sum()), n_eval=int((sub["split"] == "eval").sum()))
        acc = float((P[:, 0].argmax(1) == sub.set_index("sample_id").loc[ids, "label"].to_numpy()).mean())
        print(f"[dataset] processed {len(ids)} images | plain accuracy = {acc:.4f}")

    if args.only in ("external", "both"):
        ext = load_external(ctx)
        if ext is None or len(ext) == 0:
            print("[external] no external/images + labels.csv found -> skipping Test B data")
        else:
            items = list(zip(ext["sample_id"], ext["image_path"]))
            ids, P, failed = collect_probs(model, items, bank, ctx.arrays_external_npz, bid,
                                           load_image=loader, flush_every=args.flush_every, desc="external")
            ext.to_csv(ctx.run_dir / "samples_external.csv", index=False)
            manifest_updates.update(n_external=int(len(ids)), n_external_failed=int(len(failed)))
    ctx.update_manifest(**manifest_updates)


if __name__ == "__main__":
    main()
