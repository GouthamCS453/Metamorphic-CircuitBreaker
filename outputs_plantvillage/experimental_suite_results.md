# Experimental Testing Suite Results

Executed: 2026-10-06T10:00:00.752857+00:00

## Scope

- Test 2 uses the labeled web testbed as a provisional batch because no separate held-out split is present.
- Test 3 computes normalized softmax entropy and 10 stochastic dropout passes directly from the Keras model.
- Latency is CPU-only on Windows; TensorFlow reports that native Windows GPU support is unavailable.

## Results

- Test 1: 30 web images; 11 baseline-correct; 5 OPEN trips; MCB safety recall=0.2632.
- Test 2: provisional web batch, not a held-out split; MCB F2=0.3086.
- Test 3: MCB F2=0.3086; entropy F2=0.1899; MC-Dropout F2=0.1899.
- Test 4: mean IR: geometric=0.1860, photometric=0.1520, sensor_noise=0.2114.
- Test 5: best grid point tau_fam=0.2, theta_trip=0.1, Youden J=0.5167.
- Test 6: total mean=9821.00 ms, P50=10634.10 ms, P95=12748.53 ms across 30 images.
- Test 7: generated 3 paired Grad-CAM panels in `outputs\experimental_gradcam`.

## Per-sample records

The complete sample-level records, sweep grid, metrics, and review scope notes are in `experimental_suite_results.json`.
