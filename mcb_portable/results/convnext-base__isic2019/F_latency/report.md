# Test F - Latency

_Run `convnext-base__isic2019` | model: **ConvNeXt-Base** | dataset: **ISIC2019** | generated 2026-10-12T01:37:54_
Device: **cpu**; 30 timed runs after 5 warm-up runs; 26 images per breaker call.

| Component | Mean (ms) | SD | P50 | P95 |
| --- | --- | --- | --- | --- |
| Single forward pass (plain model) | 136.93 | 10.87 | 134.98 | 160.69 |
| Generate 25 variants | 354.49 | 162.41 | 473.49 | 505.62 |
| Batched forward (26 images) | 2649.98 | 85.64 | 2637.57 | 2740.40 |
| CBI scoring + state machine | 0.48 | 0.13 | 0.49 | 0.65 |
| MCB total (generate + batch + score) | 3004.95 | 221.74 | 3095.89 | 3242.53 |

**Overhead vs a plain single forward pass: 21.9x.** The scoring logic itself costs 0.48 ms; cost is dominated by the batched forward over the variants (can be reduced with a smaller bank, fp16, or running the breaker only on predictions the plain model is not already very sure about).
