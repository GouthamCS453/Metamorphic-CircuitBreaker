# Held-Out Experimental Results Table

Executed on the structured dataset `test` split. The split contains 5,741 images; this run used a deterministic stratified sample of 100 images across the available classes because the complete seven-test suite is CPU-intensive.

## Summary

| Test | Evaluation | Result | Notes |
|---|---|---:|---|
| 1 | Held-out baseline and MCB comparison | 83/100 baseline predictions correct | 17 baseline errors; 8 OPEN and 10 HALF_OPEN states |
| 2 | Held-out batch benchmark | MCB F2 **0.1974** | Executed on structured test data |
| 3 | Runtime monitor comparison | Entropy F2 **0.3205** | Best F2 among the tested fixed monitors |
| 4 | Family profiling | Sensor/noise IR **0.1792** | Highest mean instability family |
| 5 | Threshold and weight sweep | Best Youden J **0.2927** | `tau_fam=0.45`, `theta_trip=0.50` |
| 6 | Latency | Mean **11,852.99 ms** | P50 11,307.86 ms; P95 12,419.03 ms; CPU-only |
| 7 | Grad-CAM audit | **3 panels** | 8 OPEN cases were available |

## Test 1 and Test 2 Confusion Matrices

The strict table treats only `OPEN` as a trip. `HALF_OPEN` is reported separately because it represents advisory review rather than automatic blocking.

| Accounting policy | TP | TN | FP | FN | Safety recall | False-trip rate | Precision | F2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict OPEN only | 3 | 78 | 5 | 14 | 17.65% | 6.02% | 37.50% | 0.1974 |
| Conservative OPEN + HALF_OPEN intervention | 6 | 71 | 12 | 11 | 35.29% | 14.46% | 33.33% | 0.3488 |

## Test 3: Monitor Comparison

| Method | TP | TN | FP | FN | Safety recall | False-trip rate | Precision | F2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standalone model | 0 | 83 | 0 | 17 | 0.00% | 0.00% | 0.00% | 0.0000 |
| Naive one-flip breaker | 12 | 15 | 68 | 5 | 70.59% | 81.93% | 15.00% | 0.4054 |
| Softmax entropy | 5 | 78 | 5 | 12 | 29.41% | 6.02% | 50.00% | **0.3205** |
| MC Dropout, 10 passes | 3 | 76 | 7 | 14 | 17.65% | 8.43% | 30.00% | 0.1923 |
| Hierarchical MCB | 3 | 78 | 5 | 14 | 17.65% | 6.02% | 37.50% | 0.1974 |

## Test 4: Family Profiling

| Transformation family | Hierarchical mean IR | Flat mean score |
|---|---:|---:|
| Geometric | 0.0561 | 0.0580 |
| Photometric | 0.0570 | 0.0570 |
| Sensor/noise | **0.1792** | **0.1792** |

## Test 5: Threshold and Weight Sweep

| Best parameter | Value |
|---|---:|
| Family threshold (`tau_fam`) | 0.45 |
| Trip threshold (`theta_trip`) | 0.50 |
| Youden J | 0.2927 |
| TP / TN / FP / FN | 6 / 78 / 5 / 11 |
| Safety recall / false-trip rate | 35.29% / 6.02% |
| Precision / F2 | 54.55% / 0.3797 |

| Weighting variant | AUROC |
|---|---:|
| Balanced | 0.6215 |
| Worst-family only | 0.5819 |
| Spread-dominant | 0.7484 |
| High-confidence trust | **0.8278** |
| Pure black-box | 0.4929 |

## Test 7: Grad-CAM Outputs

Generated panels are in `outputs/experimental_gradcam/`. The JSON record lists each panel and its baseline/transformed class pair.

## Interpretation

On this held-out sample, softmax entropy produced the highest F2 among the fixed monitor comparisons. The high-confidence-trust weighting variant produced the highest AUROC in the ablation. These results are separate from the earlier 30-image web-testbed results and should not be pooled with them. The 100-image run is useful for engineering comparison, but a paper-level claim should use the full 5,741-image test split or a predeclared 500–1,000-image benchmark with confidence intervals.
