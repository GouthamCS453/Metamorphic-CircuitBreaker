"""
mcb_experiments/v4/test2_batch_benchmark.py — Full Held-Out Pool Cache Generation & Verification.

Authoritative Protocol:
- Evaluates full held-out validation pool (N = 11,763 from checkpoints/val_split.csv).
- Evaluates original image + all 40 transform variants dynamically loaded from transform_bank.yaml.
- Deterministic per-image transform seeding: seed = zlib.crc32(sample_id.encode()).
- Extracts baseline logits (saved to logits.npy) and pre-logits for head-only MC-Dropout (mcdo_score).
- Computes baseline metrics: c0, entropy, msp_score = 1 - c0.
- Computes variant predictions (vpred_000..vpred_039) and flips (flip_000..flip_039).
- Computes safety-critical errors: critical_error = 1 if y_true in {13, 14, 17, 26, 27, 28} and correct == 0.
- Computes Level 1 type scores (9 columns), Level 2 family instability (3 columns), ff, kflips, and tta_var.
- MANDATORY STOP CHECK: Verifies batched CBI == reference CBI on 50 random samples (|diff| < 1e-6).
- Exports:
  - mcb_experiments/results_v4/per_sample_scores.csv
  - mcb_experiments/results_v4/logits.npy
"""

from __future__ import annotations

import json
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mobilenet_adapter import MobileNetAdapter
from src.utils import BEST_MODEL_PATH, IMAGENET_MEAN, IMAGENET_STD
from src.circuit_breaker import (
    MetamorphicCircuitBreaker,
    CircuitBreakerConfig,
    DiagnosticReport,
    SingleTestResult,
)
from src.metamorphic_families import (
    HIERARCHICAL_TEST_MATRIX,
    SEVERITY_WEIGHTS,
    FAMILIES,
    MetamorphicTest,
)

RESULTS_DIR = PROJECT_ROOT / "mcb_experiments" / "results_v4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS_PATH = PROJECT_ROOT / "variants.json"
with open(VARIANTS_PATH, "r", encoding="utf-8") as f:
    VARIANTS = json.load(f)

N_VARIANTS = len(VARIANTS)
CRITICAL_CLASSES = {13, 14, 17, 26, 27, 28}


def type_scores_batched(flips: np.ndarray) -> Dict[Tuple[str, str], np.ndarray]:
    """Eq. 2: Level 1 normalization per transform type."""
    groups: Dict[Tuple[str, str], Dict[str, List[int]]] = {}
    for j, v in enumerate(VARIANTS):
        groups.setdefault((v["family"], v["type"]), {}).setdefault(v["tier"], []).append(j)
    out = {}
    for key, tiers in groups.items():
        num = sum(SEVERITY_WEIGHTS[t] * flips[:, idx].mean(1) for t, idx in tiers.items())
        denom = sum(SEVERITY_WEIGHTS[t] for t in tiers)
        out[key] = num / denom
    return out


def family_ir_batched(flips: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[Tuple[str, str], np.ndarray]]:
    """Eq. 3: Level 2 normalization: unweighted mean over exactly 3 types per family."""
    s = type_scores_batched(flips)
    ir = {}
    for f in FAMILIES:
        ms = [v for (fam, _), v in s.items() if fam == f]
        assert len(ms) == 3, f"family {f} has {len(ms)} transform types, need 3"
        ir[f] = np.mean(ms, axis=0)
    return ir, s


def cbi_batched(ir: Dict[str, np.ndarray], c0: np.ndarray, tau_fam: float = 0.35, a: float = 0.50, b: float = 0.35, g: float = 0.15) -> np.ndarray:
    """Eq. 4 and 5."""
    M = np.stack([ir[f] for f in FAMILIES], 1)
    cfs = (M >= tau_fam).mean(1)
    return a * M.max(1) + b * cfs + g * (1.0 - c0)


def process_sample_transforms(row: Dict[str, Any], to_tensor) -> Tuple[torch.Tensor, str, int]:
    """Generates original + 40 transformed image tensors for one sample at model-input resolution."""
    sample_id = str(row["image"])
    img_path = row["image_path"]
    y_true = int(row["label"])
    seed = zlib.crc32(sample_id.encode())
    
    img = Image.open(img_path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    t_imgs = [test(img, seed=seed) for test in HIERARCHICAL_TEST_MATRIX]
    
    # Stack 41 tensors: index 0 is original, 1..40 are variants
    tensors = torch.stack([to_tensor(img)] + [to_tensor(t) for t in t_imgs])
    return tensors, sample_id, y_true


@torch.no_grad()
def compute_mcdo_score(feat: torch.Tensor, head: torch.nn.Module, p: float = 0.2, T: int = 10) -> np.ndarray:
    """Section 5: B5 MC-Dropout head-only predictive entropy on pooled features."""
    # feat: (B, 1024)
    B = feat.shape[0]
    passes = []
    for _ in range(T):
        dropped = F.dropout(feat, p=p, training=True)
        logits = head(dropped)
        probs = F.softmax(logits, dim=-1)
        passes.append(probs)
    mean_probs = torch.stack(passes, dim=0).mean(dim=0)  # (B, C)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-12)).sum(dim=-1)
    return entropy.cpu().numpy()


def run_batch_benchmark(batch_size: int = 16, num_workers: int = 6):
    print("=" * 70)
    print("STEP 3: FULL HELD-OUT POOL CACHE GENERATION (GTSRB MobileNetV3-Small)")
    print("=" * 70)

    val_csv = PROJECT_ROOT / "checkpoints" / "val_split.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Missing {val_csv}")

    df_val = pd.read_csv(val_csv)
    # Resolve image paths if relative
    def _res(p: str) -> str:
        pth = Path(p)
        if pth.is_absolute() and pth.exists():
            return str(pth)
        cand = PROJECT_ROOT / p
        if cand.exists():
            return str(cand)
        cand2 = PROJECT_ROOT / "data" / "gtsrb" / p
        if cand2.exists():
            return str(cand2)
        return str(pth)

    df_val["image_path"] = df_val["image_path"].apply(_res)
    total_samples = len(df_val)
    print(f"Loaded held-out pool: {total_samples} samples from {val_csv}")

    # Check for existing checkpoint/cache to resume if needed
    cache_csv = RESULTS_DIR / "per_sample_scores.csv"
    logits_npy = RESULTS_DIR / "logits.npy"
    
    # Initialize model
    print("Loading MobileNetV3-Small model...")
    adapter = MobileNetAdapter(checkpoint_path=str(BEST_MODEL_PATH))
    model = adapter.model
    model.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    records_list = df_val.to_dict("records")
    checkpoint_file = RESULTS_DIR / "cache_checkpoint.npz"

    all_sample_ids: List[str] = []
    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    all_c0: List[float] = []
    all_entropy: List[float] = []
    all_msp: List[float] = []
    all_mcdo: List[float] = []
    all_vpred: List[List[int]] = []
    all_logits: List[np.ndarray] = []

    start_batch = 0
    if checkpoint_file.exists():
        print(f"Found existing checkpoint: {checkpoint_file}. Resuming cache generation...")
        ckpt = np.load(checkpoint_file, allow_pickle=True)
        all_sample_ids = ckpt["sids"].tolist()
        all_y_true = ckpt["y_true"].tolist()
        all_y_pred = ckpt["y_pred"].tolist()
        all_c0 = ckpt["c0"].tolist()
        all_entropy = ckpt["entropy"].tolist()
        all_msp = ckpt["msp"].tolist()
        all_mcdo = ckpt["mcdo"].tolist()
        all_vpred = ckpt["vpred"].tolist()
        all_logits = ckpt["logits"].tolist()
        ckpt.close()
        done_samples = len(all_sample_ids)
        start_batch = done_samples // batch_size
        print(f"Resumed from {done_samples} samples (batch {start_batch})")

    t_start = time.time()
    n_batches = (total_samples + batch_size - 1) // batch_size
    print(f"Executing batch inference from batch {start_batch} to {n_batches} (batch_size={batch_size})...\n", flush=True)

    for b_idx in range(start_batch, n_batches):
        b_start = b_idx * batch_size
        b_end = min(total_samples, b_start + batch_size)
        batch_records = records_list[b_start:b_end]

        # Multi-threaded image loading and transform generation at 224x224
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            batch_data = list(executor.map(lambda r: process_sample_transforms(r, to_tensor), batch_records))

        # batch_tensors: list of (41, 3, 224, 224) tensors -> shape (B * 41, 3, 224, 224)
        B = len(batch_data)
        stacked = torch.cat([d[0] for d in batch_data], dim=0)
        sids = [d[1] for d in batch_data]
        y_trues = [d[2] for d in batch_data]

        with torch.no_grad():
            # Extract features for original images (indices 0, 41, 82, ...)
            orig_indices = [i * 41 for i in range(B)]
            orig_tensors = stacked[orig_indices]
            
            # Features and mcdo
            feats = model.forward_head(model.forward_features(orig_tensors), pre_logits=True)
            mcdo_vals = compute_mcdo_score(feats, model.classifier, p=0.2, T=10)

            # Full forward pass for all 41 * B images
            all_preds_logits = model(stacked)
            all_probs = F.softmax(all_preds_logits, dim=-1)
            all_preds = all_preds_logits.argmax(dim=-1).cpu().numpy()
            probs_np = all_probs.cpu().numpy()
            logits_np = all_preds_logits.cpu().numpy()

        for i in range(B):
            base_idx = i * 41
            orig_logit = logits_np[base_idx]
            orig_prob = probs_np[base_idx]
            y_p = int(all_preds[base_idx])
            c0_val = float(orig_prob[y_p])
            ent_val = float(-(orig_prob * np.log(orig_prob + 1e-12)).sum())
            msp_val = float(1.0 - c0_val)
            mcdo_val = float(mcdo_vals[i])
            vpreds = all_preds[base_idx + 1 : base_idx + 41].tolist()

            all_sample_ids.append(sids[i])
            all_y_true.append(y_trues[i])
            all_y_pred.append(y_p)
            all_c0.append(c0_val)
            all_entropy.append(ent_val)
            all_msp.append(msp_val)
            all_mcdo.append(mcdo_val)
            all_vpred.append(vpreds)
            all_logits.append(orig_logit)

        # Checkpoint every 25 batches
        if (b_idx + 1) % 25 == 0:
            np.savez(
                checkpoint_file,
                sids=np.array(all_sample_ids),
                y_true=np.array(all_y_true),
                y_pred=np.array(all_y_pred),
                c0=np.array(all_c0),
                entropy=np.array(all_entropy),
                msp=np.array(all_msp),
                mcdo=np.array(all_mcdo),
                vpred=np.array(all_vpred),
                logits=np.array(all_logits),
            )

        # Progress reporting every 5 batches
        if (b_idx + 1) % 5 == 0 or (b_idx + 1) == n_batches:
            elapsed = time.time() - t_start
            done = len(all_sample_ids)
            rate = done / elapsed if elapsed > 0 else 0
            eta_min = (total_samples - done) / rate / 60.0 if rate > 0 else 0
            print(f"[{done:5d} / {total_samples}] Processed ({done/total_samples*100:5.1f}%) | {rate:5.2f} img/s | Elapsed: {elapsed/60:4.1f}m | ETA: {eta_min:4.1f}m", flush=True)

    # Clean up checkpoint file upon successful completion
    try:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
    except Exception:
        pass

    # Convert to arrays and DataFrames
    print("\nInference complete. Compiling cached metrics...")
    all_y_true_np = np.array(all_y_true)
    all_y_pred_np = np.array(all_y_pred)
    all_c0_np = np.array(all_c0)
    all_vpred_np = np.array(all_vpred)  # (N, 40)
    all_logits_np = np.array(all_logits)  # (N, 43)

    correct_np = (all_y_pred_np == all_y_true_np).astype(int)
    flips_np = (all_vpred_np != all_y_pred_np[:, None]).astype(int)

    # Critical errors: safety-relevant sign misclassified
    critical_error_np = np.array([
        1 if (yt in CRITICAL_CLASSES and c == 0) else 0
        for yt, c in zip(all_y_true_np, correct_np)
    ], dtype=int)

    # Compute Level 1 (type scores) and Level 2 (family IR)
    ir_dict, s_dict = family_ir_batched(flips_np)

    # Compute baseline scores
    ff_np = flips_np.mean(axis=1)
    kflip_1 = (flips_np.sum(axis=1) >= 1).astype(int)
    kflip_2 = (flips_np.sum(axis=1) >= 2).astype(int)
    kflip_3 = (flips_np.sum(axis=1) >= 3).astype(int)
    kflip_5 = (flips_np.sum(axis=1) >= 5).astype(int)
    kflip_8 = (flips_np.sum(axis=1) >= 8).astype(int)

    # TTA categorical variance (original image included as variant 0)
    all_preds_with_orig = np.concatenate([all_y_pred_np[:, None], all_vpred_np], axis=1)  # (N, 41)
    C_classes = 43
    tta_counts = np.stack([(all_preds_with_orig == c).sum(1) for c in range(C_classes)], 1) / 41.0
    tta_var_np = 1.0 - tta_counts.max(axis=1)

    # Build DataFrame
    df_cache = pd.DataFrame({
        "sample_id": all_sample_ids,
        "group_id": all_sample_ids,  # GTSRB fallback: group_id = sample_id
        "y_true": all_y_true_np,
        "y_pred": all_y_pred_np,
        "correct": correct_np,
        "c0": all_c0_np,
        "entropy": all_entropy,
        "msp_score": all_msp,
        "mcdo_score": all_mcdo,
        "critical_error": critical_error_np,
        "ff": ff_np,
        "kflip_1": kflip_1,
        "kflip_2": kflip_2,
        "kflip_3": kflip_3,
        "kflip_5": kflip_5,
        "kflip_8": kflip_8,
        "tta_var": tta_var_np,
        "ir_geometric": ir_dict["geometric"],
        "ir_photometric": ir_dict["photometric"],
        "ir_sensor_noise": ir_dict["sensor_noise"],
    })

    # Add 9 score_<family>_<type> columns
    for (fam, t_type), sc in s_dict.items():
        df_cache[f"score_{fam}_{t_type}"] = sc

    # Add vpred_000..vpred_039 and flip_000..flip_039
    for j in range(N_VARIANTS):
        df_cache[f"vpred_{j:03d}"] = all_vpred_np[:, j]
        df_cache[f"flip_{j:03d}"] = flips_np[:, j]

    # Save cache and logits
    df_cache.to_csv(cache_csv, index=False)
    np.save(logits_npy, all_logits_np)
    print(f"\n[Saved] {cache_csv} ({len(df_cache)} rows, {len(df_cache.columns)} columns)")
    print(f"[Saved] {logits_npy} ({all_logits_np.shape})")

    # ─── MANDATORY STOP CHECK: Batched CBI == Reference CBI on 50 samples ────
    print("\n" + "=" * 70)
    print("MANDATORY STOP VERIFICATION: Reference CBI vs Batched CBI (50 Samples)")
    print("=" * 70)

    cb_ref = MetamorphicCircuitBreaker(model=adapter, config=CircuitBreakerConfig())
    np.random.seed(42)
    sample_indices = np.random.choice(total_samples, size=50, replace=False)

    ir_sub, _ = family_ir_batched(flips_np[sample_indices])
    batched_cbis = cbi_batched(ir_sub, all_c0_np[sample_indices], tau_fam=cb_ref.config.tau_fam)

    max_cbi_diff = 0.0
    for idx_pos, orig_idx in enumerate(sample_indices):
        flips_i = flips_np[orig_idx]
        c0_i = all_c0_np[orig_idx]

        # Construct reference SingleTestResult list
        test_results = []
        for j, test in enumerate(HIERARCHICAL_TEST_MATRIX):
            test_results.append(
                SingleTestResult(
                    test_id=test.id,
                    test_name=test.name,
                    family=test.family,
                    transform_type=test.transform_type,
                    severity=test.severity,
                    weight=test.weight,
                    pred_idx=int(all_vpred_np[orig_idx, j]),
                    pred_label="",
                    confidence=0.5,
                    flipped=bool(flips_i[j]),
                )
            )

        # Reference engine logic
        type_groups = {}
        for r in test_results:
            key = (r.family, r.transform_type)
            type_groups.setdefault(key, []).append(r)
        
        type_scores = {}
        family_to_types = {f: [] for f in FAMILIES}
        for (fam, t_type), tests in type_groups.items():
            w_fail = sum(t.weight for t in tests if t.flipped)
            w_tot = sum(t.weight for t in tests)
            score_m = w_fail / w_tot if w_tot > 0 else 0.0
            type_scores[t_type] = float(score_m)
            if t_type not in family_to_types[fam]:
                family_to_types[fam].append(t_type)

        family_instability = {}
        for fam in FAMILIES:
            t_types = family_to_types[fam]
            family_instability[fam] = float(sum(type_scores[t] for t in t_types) / len(t_types))

        peak_fam_score = max(family_instability.values())
        cfs = sum(1 for ir in family_instability.values() if ir >= cb_ref.config.tau_fam) / len(FAMILIES)
        ref_cbi = cb_ref.config.alpha * peak_fam_score + cb_ref.config.beta * cfs + cb_ref.config.gamma * (1.0 - c0_i)
        ref_cbi = float(max(0.0, min(1.0, ref_cbi)))

        diff = abs(ref_cbi - batched_cbis[idx_pos])
        if diff > max_cbi_diff:
            max_cbi_diff = diff

    print(f"Measured Maximum Absolute Difference on 50 Samples: {max_cbi_diff:.2e}")
    if max_cbi_diff >= 1e-6:
        print(f"[STOP TRIGGERED] Batched CBI differs from reference CBI by {max_cbi_diff} >= 1e-6!")
        sys.exit(1)

    print("[VERIFICATION PASSED] Batched CBI matches reference CBI identically (|diff| < 1e-6).")
    print("=" * 70)


if __name__ == "__main__":
    run_batch_benchmark(batch_size=16, num_workers=6)
