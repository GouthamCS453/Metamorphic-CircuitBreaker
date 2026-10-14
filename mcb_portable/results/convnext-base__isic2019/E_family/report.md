# Test E - Family & Transform Vulnerability Profile

_Run `convnext-base__isic2019` | model: **ConvNeXt-Base** | dataset: **ISIC2019** | generated 2026-10-12T01:36:02_
Profiled samples: **749** (132 base-model errors). `flip` = original binary signal; `drift` = continuous.

## Family instability (flip signal)

| Family | Mean IR (all) | Correct | Errors | Share > 0 | AUROC for error detection [95% CI] |
| --- | --- | --- | --- | --- | --- |
| geometric | 0.047 | 0.024 | 0.157 | 10.7% | 0.610 [0.572, 0.652] |
| photometric | 0.048 | 0.031 | 0.130 | 15.6% | 0.638 [0.596, 0.685] |
| sensor_noise | 0.053 | 0.041 | 0.111 | 19.9% | 0.587 [0.545, 0.629] |

## Family instability (drift signal)

| Family | Mean IR (all) | Correct | Errors | Share > 0 | AUROC for error detection [95% CI] |
| --- | --- | --- | --- | --- | --- |
| geometric | 0.023 | 0.019 | 0.040 | 91.1% | 0.714 [0.665, 0.761] |
| photometric | 0.026 | 0.025 | 0.034 | 97.6% | 0.607 [0.552, 0.655] |
| sensor_noise | 0.028 | 0.027 | 0.032 | 90.4% | 0.562 [0.502, 0.615] |

## Zero-flip blindness

- Samples with **no flip at all** across the whole bank: 74.1% overall; correct 79.6%; errors 48.5% (64 of 132 errors).
- These errors are invisible to the binary-flip signal; the continuous `drift` signal exists to address them.

## Per transform type (flip rate)

| Family | Type | Flip rate | errors | Flip rate | correct |
| --- | --- | --- | --- |
| geometric | rotation | 9.1% | 1.9% |
| geometric | scale | 12.4% | 1.9% |
| geometric | flip | 16.7% | 1.9% |
| photometric | brightness | 10.6% | 2.9% |
| photometric | contrast | 11.6% | 1.4% |
| photometric | saturation | 13.9% | 4.2% |
| sensor_noise | blur | 11.1% | 2.8% |
| sensor_noise | noise | 14.4% | 8.0% |
| sensor_noise | jpeg | 7.8% | 2.2% |

## Per individual transform (sorted by error-vs-correct difference)

| Transform | Errors | Correct | Difference |
| --- | --- | --- | --- |
| PHOTO_CONTRAST_SEV | 17.4% | 1.9% | 15.5% |
| GEOM_FLIP_MOD | 16.7% | 1.9% | 14.7% |
| NOISE_BLUR_SEV | 19.7% | 5.0% | 14.7% |
| GEOM_SCALE_SEV | 16.7% | 2.8% | 13.9% |
| PHOTO_SAT_SEV | 18.2% | 5.7% | 12.5% |
| PHOTO_SAT_MOD | 15.9% | 4.2% | 11.7% |
| PHOTO_BRIGHT_SEV | 15.9% | 5.2% | 10.7% |
| PHOTO_CONTRAST_MOD | 12.1% | 1.5% | 10.7% |
| GEOM_SCALE_MOD | 12.1% | 1.8% | 10.3% |
| PHOTO_BRIGHT_MOD | 12.1% | 2.9% | 9.2% |
| GEOM_ROT_MILD | 9.8% | 1.1% | 8.7% |
| GEOM_ROT_MOD | 9.8% | 1.8% | 8.1% |
| NOISE_GAUSS_MOD | 15.2% | 7.1% | 8.0% |
| NOISE_JPEG_SEV | 11.4% | 3.6% | 7.8% |
| NOISE_GAUSS_SEV | 22.0% | 14.7% | 7.2% |
| GEOM_SCALE_MILD | 8.3% | 1.3% | 7.0% |
| NOISE_BLUR_MOD | 9.1% | 2.8% | 6.3% |
| PHOTO_SAT_MILD | 7.6% | 2.6% | 5.0% |
| GEOM_ROT_SEV | 7.6% | 2.8% | 4.8% |
| NOISE_JPEG_MOD | 6.8% | 2.1% | 4.7% |
| PHOTO_CONTRAST_MILD | 5.3% | 0.6% | 4.7% |
| NOISE_JPEG_MILD | 5.3% | 0.8% | 4.5% |
| NOISE_BLUR_MILD | 4.5% | 0.5% | 4.1% |
| NOISE_GAUSS_MILD | 6.1% | 2.3% | 3.8% |
| PHOTO_BRIGHT_MILD | 3.8% | 0.6% | 3.1% |

## Per class

| Class | N | Error rate | IR geometric | IR photometric | IR sensor_noise |
| --- | --- | --- | --- | --- | --- |
| MEL | 134 | 34.3% | 0.077 | 0.078 | 0.085 |
| NV | 381 | 5.5% | 0.014 | 0.018 | 0.015 |
| BCC | 99 | 10.1% | 0.062 | 0.061 | 0.094 |
| AK | 25 | 60.0% | 0.178 | 0.171 | 0.136 |
| BKL | 77 | 27.3% | 0.076 | 0.053 | 0.092 |
| DF | 7 | 57.1% | 0.106 | 0.216 | 0.089 |
| VASC | 7 | 0.0% | 0.000 | 0.000 | 0.012 |
| SCC | 19 | 78.9% | 0.137 | 0.166 | 0.115 |

## Cross-family coupling (correlation of drift IR)

|  | geometric | photometric | sensor_noise |
| --- | --- | --- | --- |
| geometric | 1.00 | 0.43 | 0.42 |
| photometric | 0.43 | 1.00 | 0.56 |
| sensor_noise | 0.42 | 0.56 | 1.00 |

_High off-diagonal values mean a failure on one axis co-occurs with failures on the others (multi-axis sensitivity)._
