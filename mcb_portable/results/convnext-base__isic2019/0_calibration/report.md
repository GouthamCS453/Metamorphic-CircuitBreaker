# Stage 0 - Calibration & Parameter Selection

_Run `convnext-base__isic2019` | model: **ConvNeXt-Base** | dataset: **ISIC2019** | generated 2026-10-12T00:04:38_
## Stage 0 - Parameter search and threshold calibration

Calibration split: **750** samples (139 base-model errors, 611 correct). Evaluation data is never touched here. Search space: signals `flip, drift, hybrid`, 5 weight configs, 8 tau_fam values = 120 configurations.

### Selected configurations

**`selected`** -> signal `drift`, weights `softmax_trust` (alpha=0.25, beta=0.25, gamma=0.5), tau_fam=0.5; calibration AUROC 0.788 [0.749, 0.830]; 45 configs within 1 SE (SE=0.0215), plateau AUROC 0.788 (best single config 0.788).

| Profile | Target FAR | theta_trip | theta_trip 95% CI | theta_warn | Cal recall | Cal FAR |
| --- | --- | --- | --- | --- | --- | --- |
| far05 | 5% | 0.2998 | [0.2843, 0.3077] | 0.2513 | 22.3% | 4.9% |
| far10 | 10% | 0.2513 | [0.2405, 0.2709] | 0.1920 | 40.3% | 10.0% |
| far20 | 20% | 0.1920 | [0.1755, 0.2078] | 0.1138 | 57.6% | 20.0% |
| balanced | - | 0.1568 | - | 0.0889 | 74.1% | 28.0% |

**`flip_only`** -> signal `flip`, weights `softmax_trust` (alpha=0.25, beta=0.25, gamma=0.5), tau_fam=0.5; calibration AUROC 0.788 [0.749, 0.830]; 32 configs within 1 SE (SE=0.0210), plateau AUROC 0.788 (best single config 0.788).

| Profile | Target FAR | theta_trip | theta_trip 95% CI | theta_warn | Cal recall | Cal FAR |
| --- | --- | --- | --- | --- | --- | --- |
| far05 | 5% | 0.3617 | [0.3219, 0.4860] | 0.2619 | 25.2% | 4.9% |
| far10 | 10% | 0.2619 | [0.2454, 0.2976] | 0.1855 | 42.4% | 10.0% |
| far20 | 20% | 0.1855 | [0.1683, 0.2065] | 0.1065 | 54.7% | 20.0% |
| balanced | - | 0.1347 | - | 0.0824 | 74.8% | 30.8% |

### Honest generalisation estimate (nested cross-validation)

Selecting the best configuration on training folds and scoring on the held-out fold (15 folds): CBI held-out AUROC = **0.786 +/- 0.019** vs MSP 0.793 and entropy 0.778 on the same folds. Same configuration chosen in 53% of folds.

### Top configurations by calibration AUROC

| Signal | Weights | tau_fam | Cal AUROC |
| --- | --- | --- | --- |
| drift | softmax_trust | 0.5 | 0.788 |
| drift | softmax_trust | 0.4 | 0.788 |
| flip | softmax_trust | 0.5 | 0.788 |
| flip | softmax_trust | 0.4 | 0.788 |
| drift | softmax_trust | 0.3 | 0.788 |
| flip | softmax_trust | 0.3 | 0.787 |
| hybrid | softmax_trust | 0.5 | 0.787 |
| drift | softmax_trust | 0.15 | 0.787 |
| hybrid | softmax_trust | 0.4 | 0.786 |
| drift | softmax_trust | 0.2 | 0.786 |
| hybrid | softmax_trust | 0.3 | 0.785 |
| flip | softmax_trust | 0.2 | 0.785 |

Full grid: `calibration_grid.csv`. Thresholds for baselines (msp, entropy, margin, tta_entropy, tta_var, kflip) are derived the same way and stored in `calibrated_config.json`.
