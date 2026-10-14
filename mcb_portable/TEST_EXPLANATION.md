# MCB Portable Test Suite - Detailed Test Explanation

## 0. Principles

- **Dynamic calibration, not hardcoded.** Stage 0 derives `theta_trip` and the other tunables from data. Every later test uses
  those frozen values.
- **Leak-free.** Images are split into a *calibration* set and a *held-out evaluation* set (grouped by `group` if given).
  Tunables are never fitted on the evaluation set or on the external images.
- **Matched-FAR comparison.** Detectors are compared at the same false-alarm rate (FAR = fraction of *correct* predictions
  flagged), or by AUROC. Comparing at different FARs is meaningless and misled earlier analyses.
- **Honest reporting.** Wilson and bootstrap 95% CIs, paired differences against baselines, and "underpowered" flags.
  No result is pre-committed as a win. If MCB ties softmax in-distribution, the report says so.
- **Prior N=250 ISIC finding (motivation).** At matched 10% FAR, CBI recall was 0.465 vs 0.442 for entropy: effectively a tie.
  72.8% of images had zero flips, and all 9 high-confidence errors had zero flips, so a binary-flip signal cannot catch them.
  This motivated the continuous `drift` signal. This is a post-hoc design change and is disclosed as such; Test D (ablation) shows what it contributes.
- **Cheap to run.** One inference pass saves an `N x 26 x C` array (original + 25 transforms). All tests then run offline.

## 1. Key concepts

| Term | Meaning |
| --- | --- |
| Transform bank | 25 label-preserving transforms in 3 families: 7 geometric, 9 photometric, 9 sensor-noise. Identified by `bank_id` hash. |
| `flip` signal | Per transform: did the predicted class change? (binary) |
| `drift` signal | Per transform: continuous change of the probability vector (catches confidently wrong cases with no flips). |
| `hybrid` signal | Combination of flip and drift. |
| Family IR | Instability rate per family, i.e. the weighted fraction of that family's transforms that flip/drift. |
| CBI | Composite Brittleness Index: a weighted combination of the family IRs, optionally blended with softmax trust (`softmax_trust` weights). |
| `theta_trip` | CBI threshold above which the breaker trips (prediction is flagged unreliable). |
| `tau` | Drift sensitivity threshold. |
| States | Pass / flagged by CBI relative to `theta_trip` (plus an intermediate warning band). |
| Baselines | `msp` (1 - max prob), `entropy`, `margin`, TTA entropy, TTA variance, k-flip count. |

## 2. Stage 0 - Calibration (`step1_calibrate.py`)

**Question:** What are the ideal `theta_trip`, signal type, weights and `tau` for *this* model and dataset?

**Method:**
1. Grid search over signal (flip / drift / hybrid), family weights, `tau`, using the calibration split only.
2. **Nested cross-validation** gives an honest estimate of the AUROC the selection procedure achieves.
3. Selection by the **1-SE rule + plateau**: choose the simplest configuration within one standard error of the best, and prefer
   stable plateaus over a sharp peak.
4. Derive **operating profiles**: `far05`, `far10`, `far20` (threshold set so FAR on calibration correct samples is 5/10/20%) and
   `balanced` (Youden). Also fits baseline thresholds at matching FARs.
5. Two variants are frozen: `selected` and `flip_only` (the original, hardcoded-style design, for comparison).

**Output:** `0_calibration/results.json`, `report.md`, `calibrated_config.json` (the frozen config used by every test).

**Interpretation:** A big gap between the calibration AUROC and nested-CV AUROC means over-fitting. Treat the nested-CV figure as the figure to cite.
Different models will produce different `theta_trip` values. That is the point (range-sensitivity calibration).

## 3. Test A - Full batch statistics (`run_A_batch.py`)

**Question:** On held-out data, how well does MCB detect the base model's errors compared with plain uncertainty measures?

**Metrics:** plain accuracy, confusion, AUROC (MCB-selected, MCB-flip-only, MSP, entropy, margin, TTA baselines) with bootstrap CIs;
the **paired AUROC difference** vs the best baseline with CI; recall of errors at matched 5/10/20% FAR; risk-coverage curve;
errors caught by MCB only / baseline only (complementarity); per-class results and critical-class recall; Wilson CIs.

- **A1:** operating-profile table (what each profile catches and what it flags wrongly).
- **A2:** the complementarity analysis: does combining MCB and softmax catch more than either alone.

**Favourable result:** CI of the AUROC difference includes or exceeds 0 *and* a non-trivial number of MCB-only catches. This supports
"complementary signal". A significantly *positive* difference would support "better than baselines", only if the data shows it.
**Unfavourable:** significantly negative difference. Report it plainly and rely on Tests B/C for the value argument.

## 4. Test B - External images, plain model vs framework (`run_B_external.py`)

**Question:** On ~40 out-of-source web images, does the framework catch errors that the plain model passes silently?

**Method:** The calibrated frozen config is applied, with **no retuning**. For each image: the plain model prediction (correct or not,
and its confidence) vs the framework verdict (trip or not).

**Metrics:** plain accuracy; "silent errors" (wrong and not flagged by a standard softmax-confidence flag at the same FAR);
errors caught by MCB, by softmax, MCB-only, softmax-only; high-confidence errors caught; false alarms on correct images; Wilson CIs.

**Caveat:** With ~40 images and a handful of errors, the CIs are wide. This test is a *demonstration* and not a proof. Do not
generalise from it alone, and report the counts together with Test A.

## 5. Test C - Graded distribution shift (`run_C_shift.py`)

**Question:** When the input degrades (blur, noise, JPEG, darkening, colour cast, rotation at 3 severities), does the MCB flag rate respond,
and how does it compare with softmax?

**Method:** Apply the shift to a subset (`--n-shift`), recompute the signals with the frozen config, and measure accuracy drop,
confidence drop, flag rate at each severity, and the Spearman correlation of flag rate with severity.

**Interpretation:** A model that gets worse under shift while its confidence stays flat is the situation in which a softmax-based breaker is blind.
A positive flag-rate response from MCB at that point is the favourable case. If softmax also responds strongly, the advantage is nil. Report either.
Writes `arrays_shift.npz` (needed by G).

## 6. Test D - Ablation (`run_D_ablation.py`)

**Question:** Which parts of the design matter: flip vs drift vs hybrid, hierarchical vs flat weighting, transform-bank scale, `tau`?

**Metrics:** AUROC and recall at matched FAR per variant with paired CIs against the selected configuration.
**Use:** Justifies each component honestly. Components with no gain (earlier data: hierarchy changed AUROC by about +/-0.001) should be reported as such.

## 7. Test E - Family and transform profile (`run_E_family.py`)

**Question:** Which transform families and individual transforms destabilise predictions, and how does that differ for errors and correct predictions?

**Metrics:** per-family and per-transform instability for errors vs correct, per-error-type and per-class profiles, family coupling
correlation, share of errors with zero flips.
**Use:** Explains *why* the signal works or doesn't, and supports the family-weight calibration claim (different models have different profiles).

## 8. Test F - Latency / cost (`run_F_latency.py`)

**Question:** What does the framework cost?
**Metrics:** ms per image for plain inference vs full-bank MCB, and the overhead factor (about 26x forward passes, minus batching gains).
Report it as the price of the added signal. The cost can be reduced via bank subsetting (see D).

## 9. Test G - Failure-cause attribution (`run_G_attribution.py`) - EXPLORATORY

**Question:** When a shifted image is mis-predicted, can the family IR profile tell which kind of corruption caused it?
**Method:** On Test C's shifted data, map the shift type to the expected family and check whether the family with the highest IR matches.
Ties and "no signal" are reported explicitly (they are not forced into one family; this fixes the earlier tie-break bias toward
"geometric").
**Status:** Exploratory. The numbers must not be quoted as a headline result. Low accuracy is possible and is informative in itself. An earlier "70/30 blame split" was
an unmeasured estimate and must not be used.

## 10. Result schema and merging

Each test writes `results.json` (fields: `schema_version`, `test`, `run_id`, `model`, `dataset`, `bank_id`, `n`, metrics...) and `report.md`.
`combine_results.py` reads the JSON and builds a cross-model/dataset table (Sections: error detection, external, shift, config/cost, and summary across runs),
so the paper can report mean/median/range across members' models and datasets.

## 11. Deliberately omitted

- **Grad-CAM audit (old Test H)** and **MC-Dropout baseline**: they require model-specific hooks and would break portability.
- No claim is pre-committed; the paper's framing should follow the results. A defensible framing based on the evidence so far:
  MCB is a *complementary, shift-aware, calibratable* reliability signal, not necessarily a better in-distribution error detector than softmax entropy.

## 12. Validation status

Pipeline verified end to end only on the synthetic demo (`demo/`). The real ConvNeXt/ISIC run has **not yet been executed**; run it with a small `--n`
first to check config and class-order correctness.
