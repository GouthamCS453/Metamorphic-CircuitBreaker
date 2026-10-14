# Test D - Ablations

_Run `convnext-base__isic2019` | model: **ConvNeXt-Base** | dataset: **ISIC2019** | generated 2026-10-12T01:36:02_
## Protocol

Every row re-tunes only `tau_fam` on the calibration split, then reports held-out evaluation AUROC and the recall/FAR obtained with a calibration-derived 10%-FAR threshold. *Diff vs reference* is a paired bootstrap difference; a CI that excludes 0 marks a real effect.

### reference

| Specification | tau_fam | Eval AUROC [95% CI] | Diff vs reference [95% CI] | Recall @10% FAR | Realised FAR |
| --- | --- | --- | --- | --- | --- |
| reference (drift, softmax_trust) | 0.4 | 0.826 [0.791, 0.862] | - | 50.0% | 11.8% |

### A1 signal type

| Specification | tau_fam | Eval AUROC [95% CI] | Diff vs reference [95% CI] | Recall @10% FAR | Realised FAR |
| --- | --- | --- | --- | --- | --- |
| signal = flip | 0.5 | 0.824 [0.789, 0.859] | -0.002 [-0.007, 0.004] | 47.7% | 12.5% |
| signal = drift | 0.4 | 0.826 [0.791, 0.862] | 0.000 [0.000, 0.000] | 50.0% | 11.8% |
| signal = hybrid | 0.5 | 0.824 [0.788, 0.858] | -0.002 [-0.007, 0.002] | 48.5% | 12.6% |

### A2 structure

| Specification | tau_fam | Eval AUROC [95% CI] | Diff vs reference [95% CI] | Recall @10% FAR | Realised FAR |
| --- | --- | --- | --- | --- | --- |
| flat (no hierarchy / no CFS) | - | 0.823 [0.788, 0.859] | -0.003 [-0.005, -0.001] | 50.0% | 11.7% |

### A3 weights

| Specification | tau_fam | Eval AUROC [95% CI] | Diff vs reference [95% CI] | Recall @10% FAR | Realised FAR |
| --- | --- | --- | --- | --- | --- |
| weights = balanced (0.5/0.35/0.15) | 0.15 | 0.797 [0.763, 0.833] | -0.029 [-0.046, -0.014] | 32.6% | 9.6% |
| weights = pure_mt (0.6/0.4/0.0) | 0.02 | 0.679 [0.634, 0.718] | -0.147 [-0.193, -0.107] | 18.2% | 11.2% |
| weights = peak_focus (0.7/0.2/0.1) | 0.15 | 0.770 [0.734, 0.805] | -0.056 [-0.080, -0.033] | 26.5% | 10.0% |
| weights = softmax_trust (0.25/0.25/0.5) | 0.4 | 0.826 [0.791, 0.862] | 0.000 [0.000, 0.000] | 50.0% | 11.8% |
| weights = spread (0.3/0.6/0.1) | 0.15 | 0.799 [0.765, 0.835] | -0.027 [-0.043, -0.012] | 32.6% | 9.6% |

### A4 components

| Specification | tau_fam | Eval AUROC [95% CI] | Diff vs reference [95% CI] | Recall @10% FAR | Realised FAR |
| --- | --- | --- | --- | --- | --- |
| no cross-family spread (beta=0) | 0.02 | 0.779 [0.744, 0.815] | -0.047 [-0.068, -0.025] | 25.8% | 8.9% |
| no softmax term (gamma=0) | 0.02 | 0.679 [0.634, 0.718] | -0.147 [-0.193, -0.107] | 18.2% | 11.2% |
| softmax only (1-c0) | 0.02 | 0.828 [0.794, 0.863] | 0.002 [-0.002, 0.006] | 47.7% | 11.5% |
| peak instability only | 0.02 | 0.682 [0.639, 0.723] | -0.144 [-0.187, -0.098] | 15.9% | 10.7% |
| spread only | 0.02 | 0.676 [0.628, 0.714] | -0.150 [-0.195, -0.113] | 0.0% | 0.0% |

### A5 level-2 mix

| Specification | tau_fam | Eval AUROC [95% CI] | Diff vs reference [95% CI] | Recall @10% FAR | Realised FAR |
| --- | --- | --- | --- | --- | --- |
| mean weight = 1.0 (pure mean) | 0.4 | 0.826 [0.791, 0.862] | 0.000 [-0.001, 0.001] | 50.0% | 12.0% |
| mean weight = 0.5 | 0.4 | 0.826 [0.791, 0.862] | -0.000 [-0.001, 0.001] | 50.0% | 11.8% |
| mean weight = 0.0 (pure peak) | 0.4 | 0.825 [0.790, 0.861] | -0.001 [-0.003, 0.001] | 50.0% | 11.2% |

### A6 severity weights

| Specification | tau_fam | Eval AUROC [95% CI] | Diff vs reference [95% CI] | Recall @10% FAR | Realised FAR |
| --- | --- | --- | --- | --- | --- |
| uniform (1/1/1) | 0.5 | 0.824 [0.789, 0.860] | -0.002 [-0.004, -0.001] | 47.7% | 11.7% |
| inverted (severe highest) | 0.5 | 0.821 [0.786, 0.857] | -0.005 [-0.008, -0.002] | 46.2% | 11.3% |

### A7 family removal

| Specification | tau_fam | Eval AUROC [95% CI] | Diff vs reference [95% CI] | Recall @10% FAR | Realised FAR |
| --- | --- | --- | --- | --- | --- |
| drop geometric | 0.4 | 0.822 [0.788, 0.858] | -0.004 [-0.006, -0.001] | 48.5% | 11.8% |
| drop photometric | 0.4 | 0.827 [0.792, 0.863] | 0.001 [-0.000, 0.003] | 49.2% | 11.7% |
| drop sensor_noise | 0.15 | 0.828 [0.796, 0.865] | 0.002 [-0.002, 0.007] | 49.2% | 11.8% |

