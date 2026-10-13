"""
mcb_experiments/v4/test6_latency.py — Step 10A: Component Latency Benchmarking.

Protocol Requirements (Section 9.1):
- 20 warm-up runs, then 100 timed runs per component.
- Device strictly identified: Intel CPU (CPU device only; never mixed with GPU).
- Components timed separately:
  - T_base: one forward pass on single original image
  - T_gen: generate all 40 variants for one image
  - T_batch: batched forward inference over all 40 variants
  - T_cbi: CBI scoring computation
  - T_total: full MCB pipeline (T_base + T_gen + T_batch + T_cbi)
  - T_mcdo: head-only MC-Dropout (10 head passes on pooled features)
  - T_entropy: softmax entropy computation
- Metrics: Mean, SD, P50, P95 in milliseconds.
- Generates:
  - mcb_experiments/results_v4/latency_breakdown.png
  - mcb_experiments/results_v4/latency_results.json
"""

import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "mcb_experiments" / "results_v4"
VARIANTS_PATH = PROJECT_ROOT / "variants.json"

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mobilenet_adapter import MobileNetAdapter
from src.utils import BEST_MODEL_PATH, IMAGENET_MEAN, IMAGENET_STD
from src.metamorphic_families import HIERARCHICAL_TEST_MATRIX, SEVERITY_WEIGHTS, FAMILIES

DEVICE_NAME = "Intel CPU"

def run_latency_benchmark():
    print("=" * 70)
    print("STEP 10A: COMPONENT LATENCY BENCHMARK (Section 9.1)")
    print(f"Device: {DEVICE_NAME}")
    print("=" * 70)

    with open(VARIANTS_PATH, "r", encoding="utf-8") as f:
        variants = json.load(f)

    # Initialize model
    adapter = MobileNetAdapter(checkpoint_path=str(BEST_MODEL_PATH))
    model = adapter.model
    model.eval()
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    # Sample a test image
    val_csv = PROJECT_ROOT / "checkpoints" / "val_split.csv"
    val_df = pd.read_csv(val_csv)
    sample_row = val_df.iloc[0]
    img_path = Path(sample_row["image_path"])
    if not img_path.is_absolute():
        img_path = PROJECT_ROOT / img_path
    sample_id = str(sample_row["image"])
    img_224 = Image.open(img_path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)

    # 1. Warm-up (20 runs)
    print("Executing 20 warm-up runs across all components...")
    for _ in range(20):
        t_img = to_tensor(img_224).unsqueeze(0)
        with torch.no_grad():
            _ = model(t_img)
            feat = model.forward_head(model.forward_features(t_img), pre_logits=True)
            _ = model.classifier(feat)

    # 2. Benchmark 100 Timed Runs
    N_RUNS = 100
    times = {
        "T_base": [],
        "T_gen": [],
        "T_batch": [],
        "T_cbi": [],
        "T_total": [],
        "T_mcdo": [],
        "T_entropy": [],
    }

    print(f"Executing {N_RUNS} timed runs per component...\n")

    for run_idx in range(N_RUNS):
        # T_base: single original image forward pass
        t_tensor = to_tensor(img_224).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            orig_logits = model(t_tensor)
        t_base = (time.perf_counter() - t0) * 1000.0
        times["T_base"].append(t_base)

        # T_entropy: softmax entropy
        t0 = time.perf_counter()
        probs = torch.softmax(orig_logits, dim=-1)
        _ = float(-(probs * torch.log(probs + 1e-12)).sum())
        t_ent = (time.perf_counter() - t0) * 1000.0
        times["T_entropy"].append(t_ent)

        # T_mcdo: head-only MC-Dropout (10 head passes)
        t0 = time.perf_counter()
        with torch.no_grad():
            feat = model.forward_head(model.forward_features(t_tensor), pre_logits=True)
            passes = []
            for _ in range(10):
                d = torch.nn.functional.dropout(feat, p=0.2, training=True)
                p = torch.softmax(model.classifier(d), dim=-1)
                passes.append(p)
            mean_p = torch.stack(passes, dim=0).mean(dim=0)
            _ = float(-(mean_p * torch.log(mean_p + 1e-12)).sum())
        t_mcdo = (time.perf_counter() - t0) * 1000.0
        times["T_mcdo"].append(t_mcdo)

        # T_gen: generate all 40 variants
        t0 = time.perf_counter()
        t_imgs = [test(img_224, seed=42) for test in HIERARCHICAL_TEST_MATRIX]
        batch_tensors = torch.stack([to_tensor(t) for t in t_imgs])
        t_gen = (time.perf_counter() - t0) * 1000.0
        times["T_gen"].append(t_gen)

        # T_batch: batched forward inference over 40 variants
        t0 = time.perf_counter()
        with torch.no_grad():
            v_logits = model(batch_tensors)
            v_preds = v_logits.argmax(dim=-1).cpu().numpy()
        t_batch = (time.perf_counter() - t0) * 1000.0
        times["T_batch"].append(t_batch)

        # T_cbi: scoring
        base_pred = int(orig_logits.argmax(dim=-1).item())
        c0 = float(probs[0, base_pred].item())
        flips = (v_preds != base_pred).astype(int)
        t0 = time.perf_counter()
        groups = {}
        for j, v in enumerate(variants):
            groups.setdefault((v["family"], v["type"]), {}).setdefault(v["tier"], []).append(j)
        scores = {}
        for key, tiers in groups.items():
            num = sum(SEVERITY_WEIGHTS[t] * (flips[idx].mean()) for t, idx in tiers.items())
            denom = sum(SEVERITY_WEIGHTS[t] for t in tiers)
            scores[key] = num / denom
        ir = {}
        for f in FAMILIES:
            ms = [scores[key] for key in scores if key[0] == f]
            ir[f] = float(np.mean(ms))
        cfs = sum(1 for f in FAMILIES if ir[f] >= 0.05) / len(FAMILIES)
        _ = 0.50 * max(ir.values()) + 0.35 * cfs + 0.15 * (1.0 - c0)
        t_cbi = (time.perf_counter() - t0) * 1000.0
        times["T_cbi"].append(t_cbi)

        # T_total: full pipeline
        times["T_total"].append(t_base + t_gen + t_batch + t_cbi)

    # 3. Report Metrics
    summary = {}
    print(f"{'Component':<12} | {'Mean (ms)':<10} | {'SD (ms)':<8} | {'P50 (ms)':<8} | {'P95 (ms)':<8}")
    print("-" * 55)
    for comp in ["T_base", "T_gen", "T_batch", "T_cbi", "T_total", "T_mcdo", "T_entropy"]:
        arr = np.array(times[comp])
        m = float(np.mean(arr))
        s = float(np.std(arr))
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        summary[comp] = {
            "mean": round(m, 2),
            "sd": round(s, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
        }
        print(f"{comp:<12} | {m:10.2f} | {s:8.2f} | {p50:8.2f} | {p95:8.2f}")

    # Plot Latency Breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    components = ["T_base", "T_gen", "T_batch", "T_cbi"]
    means = [summary[c]["mean"] for c in components]
    labels = ["Base Forward\n(T_base)", "Variant Gen\n(T_gen)", "Batch Forward\n(T_batch)", "CBI Scoring\n(T_cbi)"]
    bars = ax.bar(labels, means, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.0, f"{h:.1f} ms", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Latency (milliseconds)")
    ax.set_title(f"MCB Latency Breakdown on {DEVICE_NAME} (Total: {summary['T_total']['mean']:.1f} ms)")
    plt.tight_layout()
    chart_png = RESULTS_DIR / "latency_breakdown.png"
    plt.savefig(chart_png, dpi=300)
    plt.close()
    print(f"\n[Saved] {chart_png}")

    # Save latency_results.json
    out_json = {
        "hardware_device": DEVICE_NAME,
        "n_warmup": 20,
        "n_runs": 100,
        "latency_ms": summary,
    }
    json_path = RESULTS_DIR / "latency_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"[Saved] {json_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_latency_benchmark()
