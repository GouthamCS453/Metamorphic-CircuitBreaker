# Experimental Testing Suite Results

Executed: 2026-10-06T16:56:59.249461+00:00

## Scope

- All tests use the structured dataset 'test' split as a held-out labeled evaluation set.
- A deterministic stratified sample of 100 images was used from 5741 available images.
- Test 3 computes normalized softmax entropy and 10 stochastic dropout passes directly from the Keras model.
- Latency is CPU-only on Windows; TensorFlow reports that native Windows GPU support is unavailable.

## Results

- Test 1: 100 structured_dataset_test images; 83 baseline-correct; 8 OPEN trips; MCB safety recall=0.1765.
- Test 2: held-out status=EXECUTED_HELD_OUT_SPLIT; MCB F2=0.1974.
- Test 3: MCB F2=0.1974; entropy F2=0.3205; MC-Dropout F2=0.1923.
- Test 4: mean IR: geometric=0.0561, photometric=0.0570, sensor_noise=0.1792.
- Test 5: best grid point tau_fam=0.45, theta_trip=0.5, Youden J=0.2927.
- Test 6: total mean=11852.99 ms, P50=11307.86 ms, P95=12419.03 ms across 100 images.
- Test 7: generated 3 paired Grad-CAM panels in `outputs\experimental_gradcam`.

## Per-sample records

The complete sample-level records, sweep grid, metrics, and review scope notes are in `experimental_suite_results_heldout_test_100.json`.
