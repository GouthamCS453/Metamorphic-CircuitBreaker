# Test C - Graded Distribution-Shift Test

_Run `convnext-base__isic2019` | model: **ConvNeXt-Base** | dataset: **ISIC2019** | generated 2026-10-12T01:35:59_
## 1. Setup

100 evaluation images, 6 shifts x 3 severities, thresholds frozen from clean calibration (profile `far10`). Flag rate = share of images flagged OPEN (MCB) or above the frozen threshold (softmax).

## 2. Trend with severity (Spearman rho of flag rate vs severity 0-3)

| Shift | Family | Accuracy drop | Confidence drop | MCB flag rate: change / rho | Softmax flag rate: change / rho |
| --- | --- | --- | --- | --- | --- |
| blur | sensor_noise | 2.0% | 0.013 | 2.0% / 0.74 | 4.0% / 0.95 |
| noise | sensor_noise | 13.0% | -0.016 | -2.0% / -0.40 | 0.0% / 0.11 |
| jpeg | sensor_noise | 9.0% | 0.030 | 6.0% / 0.80 | 8.0% / 0.77 |
| darken | photometric | 0.0% | -0.004 | -1.0% / -0.21 | -2.0% / -0.20 |
| color_cast | photometric | 18.0% | 0.016 | 5.0% / 0.63 | 5.0% / 0.80 |
| rotate | geometric | 4.0% | 0.006 | 2.0% / 0.95 | 3.0% / 0.95 |

_A confidence drop far smaller than the accuracy drop means the model stays over-confident under shift._

## 3. Per-condition detail

| Condition | Acc | Mean conf | Over-conf gap | MCB flag rate | Softmax flag rate | AUROC MCB | AUROC softmax |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 83.0% | 0.728 | -0.102 | 20.0% | 18.0% | 0.785 | 0.772 |
| blur__1 | 82.0% | 0.729 | -0.091 | 19.0% | 19.0% | 0.793 | 0.789 |
| blur__2 | 81.0% | 0.725 | -0.085 | 22.0% | 19.0% | 0.798 | 0.795 |
| blur__3 | 81.0% | 0.715 | -0.095 | 22.0% | 22.0% | 0.764 | 0.756 |
| noise__1 | 81.0% | 0.732 | -0.078 | 19.0% | 20.0% | 0.793 | 0.790 |
| noise__2 | 75.0% | 0.730 | -0.020 | 21.0% | 21.0% | 0.800 | 0.803 |
| noise__3 | 70.0% | 0.744 | 0.044 | 18.0% | 18.0% | 0.815 | 0.816 |
| jpeg__1 | 83.0% | 0.728 | -0.102 | 17.0% | 18.0% | 0.764 | 0.763 |
| jpeg__2 | 82.0% | 0.723 | -0.097 | 21.0% | 18.0% | 0.763 | 0.763 |
| jpeg__3 | 74.0% | 0.698 | -0.042 | 26.0% | 26.0% | 0.831 | 0.837 |
| darken__1 | 83.0% | 0.725 | -0.105 | 19.0% | 19.0% | 0.790 | 0.779 |
| darken__2 | 83.0% | 0.724 | -0.106 | 23.0% | 20.0% | 0.782 | 0.773 |
| darken__3 | 83.0% | 0.732 | -0.098 | 19.0% | 16.0% | 0.771 | 0.768 |
| color_cast__1 | 80.0% | 0.733 | -0.067 | 18.0% | 17.0% | 0.779 | 0.771 |
| color_cast__2 | 77.0% | 0.736 | -0.034 | 20.0% | 21.0% | 0.737 | 0.732 |
| color_cast__3 | 65.0% | 0.712 | 0.062 | 25.0% | 23.0% | 0.844 | 0.841 |
| rotate__1 | 82.0% | 0.724 | -0.096 | 21.0% | 20.0% | 0.767 | 0.768 |
| rotate__2 | 83.0% | 0.725 | -0.105 | 22.0% | 21.0% | 0.770 | 0.772 |
| rotate__3 | 79.0% | 0.723 | -0.067 | 22.0% | 21.0% | 0.825 | 0.834 |

## 4. Frozen-clean vs re-calibrated thresholds (recall / FAR at target 10% FAR)

Re-calibration derives thresholds from a random half of the shifted samples and scores the other half (mean of up to 30 repeats).

| Condition | MCB frozen | Softmax frozen | MCB re-calibrated | Softmax re-calibrated |
| --- | --- | --- | --- | --- |
| blur__1 | 61.1% / 9.8% | 61.1% / 9.8% | 55.2% / 12.5% | 57.6% / 12.6% |
| blur__2 | 63.2% / 12.3% | 57.9% / 9.9% | 56.7% / 9.9% | 56.4% / 9.7% |
| blur__3 | 52.6% / 14.8% | 52.6% / 14.8% | 38.4% / 9.7% | 37.4% / 9.6% |
| noise__1 | 42.1% / 13.6% | 42.1% / 14.8% | 39.1% / 12.5% | 39.3% / 12.3% |
| noise__2 | 48.0% / 12.0% | 52.0% / 10.7% | 48.4% / 12.8% | 53.8% / 12.8% |
| noise__3 | 40.0% / 8.6% | 40.0% / 8.6% | 50.1% / 13.1% | 52.2% / 12.4% |
| jpeg__1 | 35.3% / 13.3% | 41.2% / 13.3% | 31.8% / 12.0% | 33.7% / 11.6% |
| jpeg__2 | 44.4% / 15.9% | 44.4% / 12.2% | 35.1% / 12.5% | 38.4% / 11.7% |
| jpeg__3 | 57.7% / 14.9% | 57.7% / 14.9% | 47.0% / 12.3% | 49.3% / 12.2% |
| darken__1 | 52.9% / 12.0% | 52.9% / 12.0% | 52.9% / 10.8% | 49.2% / 11.1% |
| darken__2 | 58.8% / 15.7% | 52.9% / 13.3% | 53.6% / 11.2% | 49.3% / 11.5% |
| darken__3 | 58.8% / 10.8% | 52.9% / 8.4% | 47.9% / 10.4% | 55.3% / 10.6% |
| color_cast__1 | 45.0% / 11.2% | 45.0% / 10.0% | 49.8% / 10.7% | 49.7% / 10.8% |
| color_cast__2 | 39.1% / 14.3% | 39.1% / 15.6% | 24.8% / 12.1% | 22.7% / 12.1% |
| color_cast__3 | 57.1% / 7.7% | 54.3% / 6.2% | 65.9% / 12.6% | 63.3% / 12.7% |
| rotate__1 | 55.6% / 13.4% | 50.0% / 13.4% | 38.7% / 12.5% | 41.5% / 12.3% |
| rotate__2 | 58.8% / 14.5% | 58.8% / 13.3% | 45.1% / 12.1% | 45.1% / 11.9% |
| rotate__3 | 61.9% / 11.4% | 66.7% / 8.9% | 57.9% / 10.1% | 63.5% / 10.1% |
