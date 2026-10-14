# Test A - Full Batch Benchmark, Baseline Comparison & Calibration Study

_Run `convnext-base__isic2019` | model: **ConvNeXt-Base** | dataset: **ISIC2019** | generated 2026-10-12T00:04:41_
## 1. Data and plain-model performance

- Samples: **1499** (calibration 750 / evaluation 749); dataset size 7600.
- Plain-model accuracy on evaluation split: **82.4%** (132 errors = silent errors without a monitor).

## 2.1 Operating point: target FAR 5%

| Method | Rule | Recall [95% CI] | FAR [95% CI] | Flag rate | Accuracy of auto-approved | Silent errors left |
| --- | --- | --- | --- | --- | --- | --- |
| mcb_selected | OPEN (trip) | 28.8% [21.8%, 37.0%] | 5.0% [3.6%, 7.0%] | 9.2% | 86.2% | 94 |
| mcb_selected | OPEN+HALF_OPEN | 50.0% [41.6%, 58.4%] | 11.8% [9.5%, 14.6%] | 18.6% | 89.2% | 66 |
| mcb_flip_only | OPEN (trip) | 28.8% [21.8%, 37.0%] | 5.8% [4.2%, 8.0%] | 9.9% | 86.1% | 94 |
| mcb_flip_only | OPEN+HALF_OPEN | 47.7% [39.4%, 56.2%] | 12.5% [10.1%, 15.3%] | 18.7% | 88.7% | 69 |
| msp | flag | 29.5% [22.4%, 37.8%] | 4.7% [3.3%, 6.7%] | 9.1% | 86.3% | 93 |
| entropy | flag | 25.8% [19.1%, 33.8%] | 5.2% [3.7%, 7.2%] | 8.8% | 85.7% | 98 |
| margin | flag | 29.5% [22.4%, 37.8%] | 5.3% [3.8%, 7.4%] | 9.6% | 86.3% | 93 |
| tta_entropy | flag | 22.0% [15.8%, 29.8%] | 4.9% [3.4%, 6.9%] | 7.9% | 85.1% | 103 |
| tta_var | flag | 6.8% [3.6%, 12.5%] | 5.7% [4.1%, 7.8%] | 5.9% | 82.6% | 123 |
| kflip | flag | 15.9% [10.6%, 23.1%] | 3.9% [2.6%, 5.7%] | 6.0% | 84.2% | 111 |

## 2.2 Operating point: target FAR 10%

| Method | Rule | Recall [95% CI] | FAR [95% CI] | Flag rate | Accuracy of auto-approved | Silent errors left |
| --- | --- | --- | --- | --- | --- | --- |
| mcb_selected | OPEN (trip) | 50.0% [41.6%, 58.4%] | 11.8% [9.5%, 14.6%] | 18.6% | 89.2% | 66 |
| mcb_selected | OPEN+HALF_OPEN | 67.4% [59.0%, 74.8%] | 20.6% [17.6%, 24.0%] | 28.8% | 91.9% | 43 |
| mcb_flip_only | OPEN (trip) | 47.7% [39.4%, 56.2%] | 12.5% [10.1%, 15.3%] | 18.7% | 88.7% | 69 |
| mcb_flip_only | OPEN+HALF_OPEN | 65.9% [57.5%, 73.4%] | 19.8% [16.8%, 23.1%] | 27.9% | 91.7% | 45 |
| msp | flag | 47.7% [39.4%, 56.2%] | 11.5% [9.2%, 14.3%] | 17.9% | 88.8% | 69 |
| entropy | flag | 40.9% [32.9%, 49.4%] | 10.5% [8.4%, 13.2%] | 15.9% | 87.6% | 78 |
| margin | flag | 51.5% [43.1%, 59.9%] | 12.3% [10.0%, 15.1%] | 19.2% | 89.4% | 64 |
| tta_entropy | flag | 38.6% [30.8%, 47.2%] | 9.7% [7.6%, 12.3%] | 14.8% | 87.3% | 81 |
| tta_var | flag | 15.2% [10.0%, 22.2%] | 8.9% [6.9%, 11.4%] | 10.0% | 83.4% | 112 |
| kflip | flag | 31.8% [24.5%, 40.2%] | 9.1% [7.1%, 11.6%] | 13.1% | 86.2% | 90 |

## 2.3 Operating point: target FAR 20%

| Method | Rule | Recall [95% CI] | FAR [95% CI] | Flag rate | Accuracy of auto-approved | Silent errors left |
| --- | --- | --- | --- | --- | --- | --- |
| mcb_selected | OPEN (trip) | 67.4% [59.0%, 74.8%] | 20.6% [17.6%, 24.0%] | 28.8% | 91.9% | 43 |
| mcb_selected | OPEN+HALF_OPEN | 90.2% [83.9%, 94.2%] | 39.5% [35.8%, 43.5%] | 48.5% | 96.6% | 13 |
| mcb_flip_only | OPEN (trip) | 65.9% [57.5%, 73.4%] | 19.8% [16.8%, 23.1%] | 27.9% | 91.7% | 45 |
| mcb_flip_only | OPEN+HALF_OPEN | 90.2% [83.9%, 94.2%] | 38.4% [34.7%, 42.3%] | 47.5% | 96.7% | 13 |
| msp | flag | 67.4% [59.0%, 74.8%] | 21.4% [18.3%, 24.8%] | 29.5% | 91.9% | 43 |
| entropy | flag | 65.9% [57.5%, 73.4%] | 21.7% [18.6%, 25.1%] | 29.5% | 91.5% | 45 |
| margin | flag | 66.7% [58.3%, 74.1%] | 20.9% [17.9%, 24.3%] | 29.0% | 91.7% | 44 |
| tta_entropy | flag | 61.4% [52.8%, 69.2%] | 20.9% [17.9%, 24.3%] | 28.0% | 90.5% | 51 |
| tta_var | flag | 32.6% [25.2%, 41.0%] | 17.0% [14.3%, 20.2%] | 19.8% | 85.2% | 89 |
| kflip | flag | 51.5% [43.1%, 59.9%] | 20.4% [17.4%, 23.8%] | 25.9% | 88.5% | 64 |

## 2.4 Operating point: balanced (Youden-optimal on calibration)

| Method | Rule | Recall [95% CI] | FAR [95% CI] | Flag rate | Accuracy of auto-approved | Silent errors left |
| --- | --- | --- | --- | --- | --- | --- |
| mcb_selected | OPEN (trip) | 78.0% [70.2%, 84.2%] | 26.6% [23.2%, 30.2%] | 35.6% | 94.0% | 29 |
| mcb_selected | OPEN+HALF_OPEN | 93.9% [88.5%, 96.9%] | 50.1% [46.1%, 54.0%] | 57.8% | 97.5% | 8 |
| mcb_flip_only | OPEN (trip) | 81.8% [74.4%, 87.5%] | 30.3% [26.8%, 34.0%] | 39.4% | 94.7% | 24 |
| mcb_flip_only | OPEN+HALF_OPEN | 96.2% [91.4%, 98.4%] | 50.1% [46.1%, 54.0%] | 58.2% | 98.4% | 5 |
| msp | flag | 79.5% [71.9%, 85.5%] | 29.2% [25.7%, 32.9%] | 38.1% | 94.2% | 27 |
| entropy | flag | 84.8% [77.8%, 90.0%] | 31.6% [28.1%, 35.4%] | 41.0% | 95.5% | 20 |
| margin | flag | 82.6% [75.2%, 88.1%] | 32.1% [28.5%, 35.9%] | 41.0% | 94.8% | 23 |
| tta_entropy | flag | 82.6% [75.2%, 88.1%] | 33.1% [29.5%, 36.9%] | 41.8% | 94.7% | 23 |
| tta_var | flag | 87.9% [81.2%, 92.4%] | 47.5% [43.6%, 51.4%] | 54.6% | 95.3% | 16 |
| kflip | flag | 51.5% [43.1%, 59.9%] | 20.4% [17.4%, 23.8%] | 25.9% | 88.5% | 64 |

## 3. Threshold-free comparison (A1)

| Detector | AUROC [95% CI] | AURC (lower better) | Acc @90% coverage | Oracle recall @5/10/20% FAR |
| --- | --- | --- | --- | --- |
| mcb_selected | 0.826 [0.791, 0.860] | 0.056 | 86.6% | 29% / 42% / 66% |
| mcb_flip_only | 0.824 [0.789, 0.857] | 0.057 | 86.1% | 28% / 42% / 67% |
| msp | 0.828 [0.794, 0.861] | 0.056 | 86.5% | 30% / 42% / 67% |
| entropy | 0.820 [0.786, 0.853] | 0.057 | 86.5% | 24% / 41% / 64% |
| margin | 0.825 [0.791, 0.860] | 0.056 | 86.4% | 30% / 43% / 66% |
| tta_entropy | 0.812 [0.776, 0.847] | 0.059 | 86.4% | 22% / 39% / 60% |
| tta_var | 0.718 [0.678, 0.755] | 0.080 | 83.4% | 7% / 17% / 38% |
| kflip | 0.667 [0.616, 0.714] | 0.134 | 85.6% | 19% / 32% / 40% |

_Plain model (no rejection) risk = 0.176. Oracle recall thresholds on the evaluation data itself (upper bound); deployable numbers are in section 2._

### Paired AUROC differences (bootstrap)

| Comparison | AUROC diff [95% CI] | Verdict |
| --- | --- | --- |
| mcb_selected_minus_msp | -0.002 [-0.006, 0.002] | no significant difference |
| mcb_selected_minus_entropy | 0.006 [-0.003, 0.016] | no significant difference |
| mcb_flip_only_minus_msp | -0.004 [-0.010, 0.003] | no significant difference |
| mcb_flip_only_minus_entropy | 0.004 [-0.008, 0.017] | no significant difference |

_Best baseline is picked by evaluation AUROC (`msp`), which is conservative against MCB._

## 4. Complementarity at 10% FAR (selected MCB, OPEN) (A1)

| Baseline | Errors | Both catch | MCB only | Baseline only | Neither | False alarms MCB / baseline |
| --- | --- | --- | --- | --- | --- | --- |
| msp | 132 | 61 | 5 | 2 | 64 | 73 / 71 |
| entropy | 132 | 52 | 14 | 2 | 64 | 73 / 65 |

### Combined detector (rank-average of calibration percentiles)

| Detector | AUROC [95% CI] | vs MCB alone | vs baseline alone | Recall / FAR @10% FAR |
| --- | --- | --- | --- | --- |
| mcb+entropy | 0.825 [0.792, 0.859] | -0.001 [-0.006, 0.004] | 0.005 [0.000, 0.010] | 44.7% / 10.0% |
| mcb+msp | 0.828 [0.793, 0.861] | 0.002 [-0.001, 0.004] | -0.000 [-0.002, 0.002] | 47.7% / 11.3% |

## 5. Error slices (recall at the 10% FAR profile)

**high_conf_errors_c0>=0.75** (n=20)

| Detector | Recall |
| --- | --- |
| mcb_selected_open | 0.0% |
| mcb_selected_any | 0.0% |
| mcb_flip_only_open | 0.0% |
| mcb_flip_only_any | 0.0% |
| msp | 0.0% |
| entropy | 0.0% |
| tta_entropy | 0.0% |

**high_conf_errors_c0>=0.90** (n=2)

| Detector | Recall |
| --- | --- |
| mcb_selected_open | 0.0% |
| mcb_selected_any | 0.0% |
| mcb_flip_only_open | 0.0% |
| mcb_flip_only_any | 0.0% |
| msp | 0.0% |
| entropy | 0.0% |
| tta_entropy | 0.0% |

**critical_class_errors** (n=71)

| Detector | Recall |
| --- | --- |
| mcb_selected_open | 45.1% |
| mcb_flip_only_open | 40.8% |
| msp | 45.1% |
| entropy | 42.3% |

## 6. Calibration study (A2)

### 6.1 Operating curve: target FAR -> realised FAR / recall on held-out evaluation data

| Target FAR | MCB selected (FAR / recall) | MCB flip-only | MSP | Entropy |
| --- | --- | --- | --- | --- |
| 1% | 1.3% / 9.1% | 0.8% / 6.8% | 1.0% / 7.6% | 1.1% / 10.6% |
| 2% | 2.6% / 13.6% | 1.5% / 11.4% | 2.4% / 14.4% | 2.4% / 12.9% |
| 5% | 5.0% / 28.8% | 5.8% / 28.8% | 4.7% / 29.5% | 5.2% / 25.8% |
| 10% | 11.8% / 50.0% | 12.5% / 47.7% | 11.5% / 47.7% | 10.5% / 40.9% |
| 15% | 16.9% / 58.3% | 17.0% / 54.5% | 16.7% / 59.8% | 17.0% / 56.8% |
| 20% | 20.6% / 67.4% | 19.8% / 65.9% | 21.4% / 67.4% | 21.7% / 65.9% |
| 30% | 28.2% / 79.5% | 29.3% / 80.3% | 30.8% / 83.3% | 30.1% / 82.6% |

### 6.2 How much calibration data is needed? (target FAR 10%, mean +/- sd over random subsets)

| Calibration samples | MCB realised FAR | MCB recall | MSP realised FAR | MSP recall |
| --- | --- | --- | --- | --- |
| 25 | 13.1% +/- 8.4% | 48.6% +/- 19.4% | 12.9% +/- 8.2% | 48.0% +/- 19.1% |
| 50 | 11.7% +/- 4.7% | 47.3% +/- 11.5% | 11.9% +/- 4.6% | 47.5% +/- 11.4% |
| 100 | 13.4% +/- 3.5% | 51.6% +/- 7.7% | 13.5% +/- 3.3% | 51.9% +/- 7.8% |
| 200 | 10.5% +/- 2.7% | 45.5% +/- 7.4% | 10.9% +/- 2.6% | 44.9% +/- 7.6% |
| 400 | 11.5% +/- 1.6% | 48.1% +/- 4.8% | 11.6% +/- 1.6% | 47.5% +/- 4.3% |
| 750 | 11.8% +/- 0.0% | 50.0% +/- 0.0% | 11.5% +/- 0.0% | 47.7% +/- 0.0% |

## 7. Per-class view

| Class | N | Errors | Error rate | MCB recall (OPEN, 10% FAR) |
| --- | --- | --- | --- | --- |
| MEL | 134 | 46 | 34.3% | 39.1% |
| NV | 381 | 21 | 5.5% | 57.1% |
| BCC | 99 | 10 | 10.1% | 60.0% |
| AK | 25 | 15 | 60.0% | 53.3% |
| BKL | 77 | 21 | 27.3% | 57.1% |
| DF | 7 | 4 | 57.1% | 50.0% |
| VASC | 7 | 0 | 0.0% | - |
| SCC | 19 | 15 | 78.9% | 53.3% |
