"""
mcb/inference.py - Run the model on the original image + all transformed variants, with resumable cache.

Output tensor P has shape (N, V+1, C); P[:,0] = original image, P[:,1:] = the V bank variants.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except Exception:  # tqdm optional
    def tqdm(it, **kw):
        return it


def default_loader(path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def build_variants(img: Image.Image, bank) -> Tuple[List[Image.Image], int]:
    out, failures = [img], 0
    for t in bank:
        try:
            out.append(t.fn(img))
        except Exception as exc:  # a failed transform becomes an unchanged copy (never a false flip)
            failures += 1
            print(f"[warn] transform {t.id} failed: {exc}", file=sys.stderr)
            out.append(img.copy())
    return out, failures


def collect_probs(model, items: Sequence[Tuple[str, str]], bank, cache_path: Path, bank_id: str,
                  load_image: Optional[Callable] = None, preprocess: Optional[Callable] = None,
                  flush_every: int = 25, desc: str = "inference"):
    """items: list of (sample_id, path). Resumes from cache_path if it matches bank_id.

    Returns (ids, probs, failed_ids) where probs is aligned with ids (failed images omitted).
    """
    load_image = load_image or default_loader
    cache_path = Path(cache_path)
    done_ids: List[str] = []
    done_probs: List[np.ndarray] = []
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if str(z["bank_id"]) == bank_id:
            done_ids = [str(x) for x in z["ids"]]
            done_probs = [p for p in z["probs"]]
            print(f"[{desc}] resuming: {len(done_ids)} samples already cached")
        else:
            print(f"[{desc}] cache bank mismatch -> discarding {cache_path.name}")
    done = set(done_ids)
    todo = [it for it in items if str(it[0]) not in done]
    failed: List[str] = []
    total_fail = 0

    def flush():
        if done_ids:
            np.savez(cache_path, ids=np.array(done_ids), probs=np.stack(done_probs).astype(np.float32),
                     bank_id=np.array(bank_id))

    for k, (sid, path) in enumerate(tqdm(todo, desc=desc), start=1):
        try:
            img = load_image(path)
            if preprocess is not None:
                img = preprocess(img)
            variants, nf = build_variants(img, bank)
            total_fail += nf
            probs = np.asarray(model.predict_proba(variants), dtype=np.float64)
            if probs.shape[0] != len(variants):
                raise ValueError(f"predict_proba returned {probs.shape[0]} rows for {len(variants)} images")
            done_ids.append(str(sid))
            done_probs.append(probs)
        except Exception as exc:
            failed.append(str(sid))
            print(f"[warn] skipped {sid}: {exc}", file=sys.stderr)
        if k % flush_every == 0:
            flush()
    flush()
    order = {sid: i for i, sid in enumerate(done_ids)}
    keep = [str(it[0]) for it in items if str(it[0]) in order]
    P = np.stack([done_probs[order[s]] for s in keep]).astype(np.float64) if keep else np.zeros((0, 0, 0))
    if total_fail:
        print(f"[{desc}] WARNING: {total_fail} transform failures were replaced by unchanged images")
    return keep, P, failed
