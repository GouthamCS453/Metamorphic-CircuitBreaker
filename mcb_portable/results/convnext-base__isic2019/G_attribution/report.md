# Test G - Fault Attribution (Exploratory)

_Run `convnext-base__isic2019` | model: **ConvNeXt-Base** | dataset: **ISIC2019** | generated 2026-10-12T01:37:54_

> **Warnings**
> - EXPLORATORY: attribution is not a core claim. Metamorphic relations measure local sensitivity around the (already degraded) image, so a degradation need not raise the instability of its own family.
## 1. Method

Clean-correct images x shifts of severity >= 2. Predicted family = argmax of the change in family instability (dIR); `none` if the largest dIR < 0.05; `tie` if two families share the maximum. Variant `selected`.

## 2. Confusion (rows = injected family)

| Injected \ Predicted | geometric | photometric | sensor_noise | tie | none | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| geometric | 2 | 1 | 2 | 0 | 161 | 1.2% |
| photometric | 15 | 30 | 33 | 0 | 254 | 9.0% |
| sensor_noise | 12 | 16 | 0 | 0 | 470 | 0.0% |

## 3. Summary

- Trials: **996**; attribution accuracy **3.2%** (Wilson 95% [2.3%, 4.5%]) vs chance 33.3%; one-sided binomial p = 1.0000 -> **not above chance**.
- Share of trials where any family reacted (detected): 11.1%; accuracy among detected trials: 28.8%.

## 4. Per shift

| Shift | True family | Trials | Correct family | Tie | None |
| --- | --- | --- | --- | --- | --- |
| blur | sensor_noise | 166 | 0.0% | 0 | 158 |
| noise | sensor_noise | 166 | 0.0% | 0 | 155 |
| jpeg | sensor_noise | 166 | 0.0% | 0 | 157 |
| darken | photometric | 166 | 0.6% | 0 | 134 |
| color_cast | photometric | 166 | 17.5% | 0 | 120 |
| rotate | geometric | 166 | 1.2% | 0 | 161 |
