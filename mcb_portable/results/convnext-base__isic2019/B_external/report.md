# Test B - External Images: Plain Model vs Framework

_Run `convnext-base__isic2019` | model: **ConvNeXt-Base** | dataset: **ISIC2019** | generated 2026-10-12T00:04:41_

> **Warnings**
> - Small external set (n=40): confidence intervals are wide; report counts, not just rates.
## 1. Plain model on the external images

- Images: **40**; plain-model accuracy **45.0%** -> **22 silent errors** without a monitor.
- Internal evaluation accuracy for reference: 82.4% (mean confidence internal 0.739 vs external 0.670).
- Thresholds are the FROZEN calibration profiles; nothing was tuned on these images.

## 2.1 Plain model vs framework - profile `far05`

| Method | Rule | Errors caught (TP) | Recall [95% CI] | False alarms (FP) | FAR [95% CI] | Silent errors left |
| --- | --- | --- | --- | --- | --- | --- |
| mcb_selected | OPEN | 2/22 | 9.1% [2.5%, 27.8%] | 2/18 | 11.1% [3.1%, 32.8%] | 20 |
| mcb_selected | OPEN+HALF_OPEN | 5/22 | 22.7% [10.1%, 43.4%] | 6/18 | 33.3% [16.3%, 56.3%] | 17 |
| mcb_flip_only | OPEN | 2/22 | 9.1% [2.5%, 27.8%] | 4/18 | 22.2% [9.0%, 45.2%] | 20 |
| mcb_flip_only | OPEN+HALF_OPEN | 6/22 | 27.3% [13.2%, 48.2%] | 5/18 | 27.8% [12.5%, 50.9%] | 16 |
| msp | flag | 2/22 | 9.1% [2.5%, 27.8%] | 2/18 | 11.1% [3.1%, 32.8%] | 20 |
| entropy | flag | 4/22 | 18.2% [7.3%, 38.5%] | 1/18 | 5.6% [1.0%, 25.8%] | 18 |
| tta_entropy | flag | 2/22 | 9.1% [2.5%, 27.8%] | 1/18 | 5.6% [1.0%, 25.8%] | 20 |
| kflip | flag | 1/22 | 4.5% [0.8%, 21.8%] | 1/18 | 5.6% [1.0%, 25.8%] | 21 |

## 2.2 Plain model vs framework - profile `far10`

| Method | Rule | Errors caught (TP) | Recall [95% CI] | False alarms (FP) | FAR [95% CI] | Silent errors left |
| --- | --- | --- | --- | --- | --- | --- |
| mcb_selected | OPEN | 5/22 | 22.7% [10.1%, 43.4%] | 6/18 | 33.3% [16.3%, 56.3%] | 17 |
| mcb_selected | OPEN+HALF_OPEN | 11/22 | 50.0% [30.7%, 69.3%] | 8/18 | 44.4% [24.6%, 66.3%] | 11 |
| mcb_flip_only | OPEN | 6/22 | 27.3% [13.2%, 48.2%] | 5/18 | 27.8% [12.5%, 50.9%] | 16 |
| mcb_flip_only | OPEN+HALF_OPEN | 11/22 | 50.0% [30.7%, 69.3%] | 8/18 | 44.4% [24.6%, 66.3%] | 11 |
| msp | flag | 4/22 | 18.2% [7.3%, 38.5%] | 6/18 | 33.3% [16.3%, 56.3%] | 18 |
| entropy | flag | 5/22 | 22.7% [10.1%, 43.4%] | 4/18 | 22.2% [9.0%, 45.2%] | 17 |
| tta_entropy | flag | 5/22 | 22.7% [10.1%, 43.4%] | 4/18 | 22.2% [9.0%, 45.2%] | 17 |
| kflip | flag | 2/22 | 9.1% [2.5%, 27.8%] | 4/18 | 22.2% [9.0%, 45.2%] | 20 |

## 2.3 Plain model vs framework - profile `far20`

| Method | Rule | Errors caught (TP) | Recall [95% CI] | False alarms (FP) | FAR [95% CI] | Silent errors left |
| --- | --- | --- | --- | --- | --- | --- |
| mcb_selected | OPEN | 11/22 | 50.0% [30.7%, 69.3%] | 8/18 | 44.4% [24.6%, 66.3%] | 11 |
| mcb_selected | OPEN+HALF_OPEN | 19/22 | 86.4% [66.7%, 95.3%] | 9/18 | 50.0% [29.0%, 71.0%] | 3 |
| mcb_flip_only | OPEN | 11/22 | 50.0% [30.7%, 69.3%] | 8/18 | 44.4% [24.6%, 66.3%] | 11 |
| mcb_flip_only | OPEN+HALF_OPEN | 18/22 | 81.8% [61.5%, 92.7%] | 9/18 | 50.0% [29.0%, 71.0%] | 4 |
| msp | flag | 11/22 | 50.0% [30.7%, 69.3%] | 8/18 | 44.4% [24.6%, 66.3%] | 11 |
| entropy | flag | 11/22 | 50.0% [30.7%, 69.3%] | 8/18 | 44.4% [24.6%, 66.3%] | 11 |
| tta_entropy | flag | 9/22 | 40.9% [23.3%, 61.3%] | 8/18 | 44.4% [24.6%, 66.3%] | 13 |
| kflip | flag | 7/22 | 31.8% [16.4%, 52.7%] | 6/18 | 33.3% [16.3%, 56.3%] | 15 |

## 2.4 Plain model vs framework - profile `balanced`

| Method | Rule | Errors caught (TP) | Recall [95% CI] | False alarms (FP) | FAR [95% CI] | Silent errors left |
| --- | --- | --- | --- | --- | --- | --- |
| mcb_selected | OPEN | 11/22 | 50.0% [30.7%, 69.3%] | 8/18 | 44.4% [24.6%, 66.3%] | 11 |
| mcb_selected | OPEN+HALF_OPEN | 19/22 | 86.4% [66.7%, 95.3%] | 12/18 | 66.7% [43.7%, 83.7%] | 3 |
| mcb_flip_only | OPEN | 13/22 | 59.1% [38.7%, 76.7%] | 8/18 | 44.4% [24.6%, 66.3%] | 9 |
| mcb_flip_only | OPEN+HALF_OPEN | 19/22 | 86.4% [66.7%, 95.3%] | 11/18 | 61.1% [38.6%, 79.7%] | 3 |
| msp | flag | 13/22 | 59.1% [38.7%, 76.7%] | 8/18 | 44.4% [24.6%, 66.3%] | 9 |
| entropy | flag | 16/22 | 72.7% [51.8%, 86.8%] | 9/18 | 50.0% [29.0%, 71.0%] | 6 |
| tta_entropy | flag | 14/22 | 63.6% [43.0%, 80.3%] | 9/18 | 50.0% [29.0%, 71.0%] | 8 |
| kflip | flag | 7/22 | 31.8% [16.4%, 52.7%] | 6/18 | 33.3% [16.3%, 56.3%] | 15 |

## 3. Matched flag-rate comparison (profile `far10`)

The plain softmax detectors are given exactly as many flags as the MCB raised (11), chosen as their highest-uncertainty images - so differences reflect ranking quality, not threshold aggressiveness.

| Method | Flags | Errors caught | False alarms |
| --- | --- | --- | --- |
| mcb_selected | 11 | 5 | 6 |
| msp | 11 | 5 | 6 |
| entropy | 11 | 7 | 4 |

## 4. Errors the plain model made silently

- Errors caught by MCB but NOT by the softmax flag: **1** ['web_mel_04_ISIC_0000158']
- Errors caught by the softmax flag but NOT by MCB: **0** []
- High-confidence (>=0.75) errors: **6**, of which MCB caught **0** []

## 5. Domain-shift indicator

| Quantity | Internal (eval split) | External |
| --- | --- | --- |
| Plain accuracy | 82.4% | 45.0% |
| Mean confidence | 0.739 | 0.670 |
| Mean CBI (selected) | 0.141 | 0.177 |
| MCB trip rate | 18.6% | 27.5% |
| Softmax flag rate | 17.9% | 25.0% |

## 6. Per-image audit (profile `far10`)

| Image | Truth | Pred | Conf | CBI | MCB state | Softmax flag | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| web_mel_01_ISIC_0000070 | MEL | NV | 0.85 | 0.083 | CLOSED | - | ERROR - MISSED |
| web_mel_02_ISIC_0000076 | MEL | MEL | 0.37 | 0.348 | OPEN | FLAG | false alarm |
| web_mel_03_ISIC_0000144 | MEL | MEL | 0.37 | 0.322 | OPEN | FLAG | false alarm |
| web_mel_04_ISIC_0000158 | MEL | NV | 0.55 | 0.260 | OPEN | - | ERROR - caught by MCB |
| web_mel_05_ISIC_0009886 | MEL | MEL | 0.46 | 0.282 | OPEN | FLAG | false alarm |
| web_nv_01_ISIC_0000005 | NV | NV | 0.61 | 0.229 | HALF_OPEN | - | ok |
| web_nv_02_ISIC_0000033 | NV | NV | 0.51 | 0.288 | OPEN | FLAG | false alarm |
| web_nv_03_ISIC_0001109 | NV | NV | 0.85 | 0.074 | CLOSED | - | ok |
| web_nv_04_ISIC_0001111 | NV | NV | 0.53 | 0.235 | HALF_OPEN | - | ok |
| web_nv_05_ISIC_0001117 | NV | NV | 0.50 | 0.261 | OPEN | FLAG | false alarm |
| web_bcc_01_ISIC_0001120 | BCC | NV | 0.59 | 0.221 | HALF_OPEN | - | ERROR - MISSED |
| web_bcc_02_ISIC_0001149 | BCC | BCC | 0.82 | 0.106 | CLOSED | - | ok |
| web_bcc_03_ISIC_0001159 | BCC | BCC | 0.88 | 0.068 | CLOSED | - | ok |
| web_bcc_04_ISIC_0010515 | BCC | NV | 0.78 | 0.118 | CLOSED | - | ERROR - MISSED |
| web_bcc_05_ISIC_0011141 | BCC | BCC | 0.90 | 0.068 | CLOSED | - | ok |
| web_ak_01_ISIC_0010512 | AK | BKL | 0.55 | 0.233 | HALF_OPEN | - | ERROR - MISSED |
| web_ak_02_ISIC_0010889 | AK | NV | 0.93 | 0.038 | CLOSED | - | ERROR - MISSED |
| web_ak_03_ISIC_0011472 | AK | MEL | 0.59 | 0.209 | HALF_OPEN | - | ERROR - MISSED |
| web_ak_04_ISIC_0011477 | AK | BKL | 0.28 | 0.368 | OPEN | FLAG | ERROR - caught by MCB |
| web_ak_05_ISIC_0011552 | AK | BKL | 0.39 | 0.318 | OPEN | FLAG | ERROR - caught by MCB |
| web_bkl_01_ISIC_0001104 | BKL | BKL | 0.96 | 0.019 | CLOSED | - | ok |
| web_bkl_02_ISIC_0001113 | BKL | MEL | 0.56 | 0.235 | HALF_OPEN | - | ERROR - MISSED |
| web_bkl_03_ISIC_0001123 | BKL | BKL | 0.74 | 0.142 | CLOSED | - | ok |
| web_bkl_04_ISIC_0001144 | BKL | BKL | 0.94 | 0.029 | CLOSED | - | ok |
| web_bkl_05_ISIC_0001145 | BKL | MEL | 0.53 | 0.251 | HALF_OPEN | - | ERROR - MISSED |
| web_df_01_ISIC_0001114 | DF | NV | 0.45 | 0.278 | OPEN | FLAG | ERROR - caught by MCB |
| web_df_02_ISIC_0001130 | DF | NV | 0.69 | 0.156 | CLOSED | - | ERROR - MISSED |
| web_df_03_ISIC_0011410 | DF | NV | 0.51 | 0.260 | OPEN | FLAG | ERROR - caught by MCB |
| web_df_04_ISIC_0011433 | DF | NV | 0.72 | 0.152 | CLOSED | - | ERROR - MISSED |
| web_df_05_ISIC_0011478 | DF | NV | 0.77 | 0.119 | CLOSED | - | ERROR - MISSED |
| web_vasc_01_ISIC_0001115 | VASC | VASC | 0.96 | 0.022 | CLOSED | - | ok |
| web_vasc_02_ISIC_0001129 | VASC | VASC | 0.82 | 0.096 | CLOSED | - | ok |
| web_vasc_03_ISIC_0011574 | VASC | NV | 0.90 | 0.050 | CLOSED | - | ERROR - MISSED |
| web_vasc_04_ISIC_0011629 | VASC | VASC | 0.51 | 0.267 | OPEN | FLAG | false alarm |
| web_vasc_05_ISIC_0011637 | VASC | VASC | 0.84 | 0.090 | CLOSED | - | ok |
| web_scc_01_ISIC_0012279 | SCC | BKL | 0.79 | 0.123 | CLOSED | - | ERROR - MISSED |
| web_scc_02_ISIC_0012688 | SCC | BKL | 0.73 | 0.150 | CLOSED | - | ERROR - MISSED |
| web_scc_03_ISIC_0012689 | SCC | BKL | 0.74 | 0.142 | CLOSED | - | ERROR - MISSED |
| web_scc_04_ISIC_0012695 | SCC | BKL | 0.74 | 0.132 | CLOSED | - | ERROR - MISSED |
| web_scc_05_ISIC_0012751 | SCC | BKL | 0.60 | 0.219 | HALF_OPEN | - | ERROR - MISSED |
