# Experimental Testing Suite — Results Tables

**Executed:** 2026-10-06  
**Dataset:** 30 labeled web-testbed images  
**Runtime:** CPU-only Windows execution with TensorFlow/Keras 2.21.0, PyTorch 2.14.1, and OpenCV 5.0.0

## Table 1 — Seven-Test Summary

| Test | Evaluation | Samples / scope | Main result | Status |
|---|---|---:|---|---|
| 1 | In-the-wild web comparison | 30 images | 5 OPEN, 3 HALF-OPEN, 22 CLOSED; MCB safety recall **26.32%**; false-trip rate **0%** | Executed |
| 2 | Held-out batch benchmark | 30-image provisional web batch | MCB TP=5, TN=11, FP=0, FN=14; F2 **0.3086** | Provisional: not a held-out split |
| 3 | Baseline safety paradigm comparison | 5 methods | Proposed MCB F2 **0.3086**; naive 1-flip F2 **0.7576**; MCB false-trip rate **0%** | Executed with scope notes |
| 4 | Family vulnerability profiling | 22 transformations, 3 families | Mean IR: geometric **0.1860**, photometric **0.1520**, sensor/noise **0.2114** | Executed |
| 5 | Threshold and weight sweep | 119 grid points, 5 formulas | Best measured point: tau_fam **0.20**, theta_trip **0.10**, Youden J **0.5167** | Executed |
| 6 | Computational overhead and latency | 30 images | Mean total **9821.00 ms**, P50 **10634.10 ms**, P95 **12748.53 ms** | Executed, CPU-only |
| 7 | Grad-CAM attention-shift audit | 5 OPEN cases; 3 panels generated | 3 paired Grad-CAM panels saved | Executed |

## Table 2 — Test 1 Web Comparison

| Measure | Value |
|---|---:|
| Total images | 30 |
| Baseline-correct images | 11 |
| Baseline errors | 19 |
| OPEN states | 5 |
| HALF-OPEN states | 3 |
| CLOSED states | 22 |
| True positives | 5 |
| True negatives | 11 |
| False positives | 0 |
| False negatives | 14 |
| Safety recall | 26.32% |
| False-trip rate | 0.00% |
| Safety precision | 100.00% |
| F2-score | 0.3086 |

## Table 3 — Test 2 Batch Metrics

| Metric | Value |
|---|---:|
| Batch size | 30 |
| TP | 5 |
| TN | 11 |
| FP | 0 |
| FN | 14 |
| Safety recall | 26.32% |
| False-trip rate | 0.00% |
| Safety precision | 100.00% |
| F2-score | 0.3086 |

**Scope note:** This is a provisional web-batch benchmark. It is not a separate held-out split from training data.

## Table 4 — Test 3 Baseline Comparison

| Method | TP | TN | FP | FN | Safety recall | False-trip rate | Precision | F2-score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standalone model | 0 | 11 | 0 | 19 | 0.00% | 0.00% | 0.00% | 0.0000 |
| Naive 1-flip MT | 15 | 3 | 8 | 4 | 78.95% | 72.73% | 65.22% | 0.7576 |
| Softmax entropy | 3 | 11 | 0 | 16 | 15.79% | 0.00% | 100.00% | 0.1899 |
| MC-Dropout, 10 passes | 3 | 11 | 0 | 16 | 15.79% | 0.00% | 100.00% | 0.1899 |
| Proposed hierarchical MCB | 5 | 11 | 0 | 14 | 26.32% | 0.00% | 100.00% | 0.3086 |

## Table 5 — Test 4 Family Profiling

| Transformation family | Mean hierarchical IR | Mean flat family score |
|---|---:|---:|
| Geometric | 0.1860 | 0.2015 |
| Photometric | 0.1520 | 0.1520 |
| Sensor / noise | 0.2114 | 0.2114 |

The test matrix contains 22 transformations with severity weights: mild **1.0**, moderate **0.6**, and severe **0.3**.

## Table 6 — Test 5 Threshold Grid

| Selected parameter | Value |
|---|---:|
| Grid points evaluated | 119 |
| Best family cutoff, tau_fam | 0.20 |
| Best trip cutoff, theta_trip | 0.10 |
| Best Youden J | 0.5167 |
| Recall at best point | 78.95% |
| False-trip rate at best point | 27.27% |
| Precision at best point | 83.33% |
| F2-score at best point | 0.7979 |

### Formula Weight Ablation

| Configuration | Alpha | Beta | Gamma | AUROC |
|---|---:|---:|---:|---:|
| Balanced formula | 0.50 | 0.35 | 0.15 | 0.7943 |
| Worst-only | 0.85 | 0.00 | 0.15 | 0.7656 |
| Spread-dominant | 0.15 | 0.70 | 0.15 | 0.8325 |
| High-confidence trust | 0.25 | 0.25 | 0.50 | 0.8182 |
| Pure black-box | 0.60 | 0.40 | 0.00 | 0.7368 |

## Table 7 — Test 6 Latency

| Runtime component | Mean milliseconds |
|---|---:|
| Baseline inference | 1302.90 |
| Transform generation | 590.65 |
| Batched inference | 7927.23 |
| Circuit-breaker logic | 0.23 |
| Total end-to-end | 9821.00 |
| Total P50 | 10634.10 |
| Total P95 | 12748.53 |

**Hardware note:** GPU was not used. TensorFlow reported that native Windows GPU support is unavailable in this environment.

## Table 8 — Test 7 Grad-CAM Panels

| Sample | Transformation | Baseline class | Flipped class | Output |
|---|---|---|---|---|
| web_0026 | Rotation 5 deg | tomato__late_blight | tomato__leaf_mold | `outputs/experimental_gradcam/web_0026_GEOM_ROT_MILD.png` |
| web_0016 | Rotation 5 deg | Cassava__Brown_Streak_Disease_(CBSD) | tomato__tomato_yellow_leaf_curl_virus | `outputs/experimental_gradcam/web_0016_GEOM_ROT_MILD.png` |
| web_0030 | Rotation 10 deg | Rice__LeafBlast | Cassava__Healthy | `outputs/experimental_gradcam/web_0030_GEOM_ROT_MOD.png` |

## Interpretation Notes

- The proposed MCB produced zero false trips on this web set, but its safety recall was only 26.32% at the default OPEN threshold.
- Naive one-flip monitoring caught more errors but generated a 72.73% false-trip rate, illustrating alert fatigue.
- The best threshold-grid point was much more sensitive than the default configuration and consequently incurred false trips.
- Test 2 should be repeated on a genuinely held-out labeled split before being used as the paper's primary benchmark.

The complete machine-readable records remain in `outputs/experimental_suite_results.json`.
