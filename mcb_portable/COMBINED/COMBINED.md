# Combined Results Across Models and Datasets

_Generated 2026-10-12 12:34 | runs included: 1 | excluded: 0_

## 1. Error detection on held-out data (Test A)

| Run (model / dataset) | Eval N / errors | Plain acc | AUROC MCB (sel / flip) | AUROC MSP / entropy | MCB - best baseline [95% CI] | Recall@10%FAR MCB / MSP | FAR MCB / MSP | MCB-only errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ConvNeXt-Base / ISIC2019 | 749 / 132 | 82.4% | 0.826 / 0.824 | 0.828 / 0.820 | -0.002 [-0.006, 0.002] | 50.0% / 47.7% | 11.8% / 11.5% | 5 |

## 2. External images: plain model vs framework (Test B)

| Run | N | Plain acc | Silent errors | Caught by MCB | Caught by softmax flag | MCB-only | Softmax-only | High-conf errors caught |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ConvNeXt-Base / ISIC2019 | 40 | 45.0% | 22 | 5 | 4 | 1 | 0 | 0/6 |

## 3. Behaviour under graded shift (Test C)

| Run | Mean accuracy drop | Mean confidence drop | MCB flag-rate change (rho) | Softmax flag-rate change (rho) |
| --- | --- | --- | --- | --- |
| ConvNeXt-Base / ISIC2019 | 7.7% | 0.008 | 2.0% (0.42) | 3.0% (0.56) |

## 4. Calibrated configurations and cost

| Run | Selected signal / weights / tau | theta_trip (10% FAR) | Nested-CV AUROC | Latency (ms) / overhead | Attribution acc (G) |
| --- | --- | --- | --- | --- | --- |
| ConvNeXt-Base / ISIC2019 | drift / softmax_trust / 0.5 | 0.2513 | 0.786 | 3005 / 21.9x | 3.2% |

## 5. Summary across runs

| Metric | Mean | Median | Range | Runs |
| --- | --- | --- | --- | --- |
| auroc_mcb_selected | 0.826 | 0.826 | 0.826 - 0.826 | 1 |
| auroc_mcb_flip_only | 0.824 | 0.824 | 0.824 - 0.824 | 1 |
| auroc_msp | 0.828 | 0.828 | 0.828 - 0.828 | 1 |
| auroc_entropy | 0.820 | 0.820 | 0.820 - 0.820 | 1 |
| auroc_diff_vs_best_baseline | -0.002 | -0.002 | -0.002 - -0.002 | 1 |
| recall_far10_mcb | 0.500 | 0.500 | 0.500 - 0.500 | 1 |
| recall_far10_msp | 0.477 | 0.477 | 0.477 - 0.477 | 1 |
| recall_far10_flip_only | 0.477 | 0.477 | 0.477 - 0.477 | 1 |
| mcb_only_errors | 5.000 | 5.000 | 5.000 - 5.000 | 1 |
| shift_spearman_mcb | 0.418 | 0.418 | 0.418 - 0.418 | 1 |
| shift_spearman_msp | 0.563 | 0.563 | 0.563 - 0.563 | 1 |
| overhead_factor | 21.945 | 21.945 | 21.945 - 21.945 | 1 |

MCB AUROC vs best baseline across runs (significant wins / losses / ties): **0 / 0 / 1**.

_Aggregates are descriptive across independent model/dataset pairs; do not read them as one pooled experiment._
