# MCB Portable Test Suite - Execution Guide

Runs the full Metamorphic Circuit Breaker (MCB) evaluation on **your model + your dataset** and writes
mergeable JSON + Markdown results. For what each test means, see `TEST_EXPLANATION.md`.

## 1. Setup (once)

```bash
pip install -r requirements.txt          # numpy, pandas, scipy, pillow, torch, tqdm, ...
```

Copy this folder (code only, no datasets or checkpoints inside) to your machine.

### Edit `member_config.py`
| Item | What to set |
| --- | --- |
| `MODEL_NAME`, `DATASET_NAME`, `DATASET_VERSION` | Labels used in the results and the run folder name (`results/<model>__<dataset>_<version>`). |
| `CLASS_NAMES` | In the model's output-index order. |
| `CRITICAL_CLASSES` | Optional, e.g. `["MEL","BCC","SCC"]`. |
| `N_SAMPLES` | Stratified subset size. **1500-2000 is recommended**; 0 = all images. |
| `load_model()` | Return an object with `predict_proba(list[PIL]) -> (n, C)` probabilities. Use `mcb.adapters.TorchClassifier` (logits) or `CallableClassifier`. Preprocessing must match your validation preprocessing. |
| `get_dataset()` | DataFrame with `image_path`, `label` (int), optional `group` (e.g. patient id). **Held-out images only** (not training data). |

`load_model()` and `get_dataset()` raise `NotImplementedError` until edited.

### External images (Test B)
```
external/images/*.jpg
external/labels.csv        # columns: filename,label   (see external/labels_TEMPLATE.csv)
```
About 40 web images of the same task, labelled by class name or index. Skip them and Test B is skipped.

## 2. Run everything

```bash
python run_all.py --n 1500
```
Order: `step0_extract` (one inference pass, cached) -> `step1_calibrate` (Stage 0) -> tests A..G.

| Flag | Meaning |
| --- | --- |
| `--config path.py` | Use a different config than `member_config.py` |
| `--n N` | Override `N_SAMPLES` |
| `--n-shift N` | Images used in the shift test C (default 100) |
| `--skip C,G` | Skip these tests |
| `--only A,B` | Run only these tests |
| `--skip-extract` | Reuse an existing `arrays.npz` |
| `--run-dir DIR` | Override the output folder |

### Individual steps
```bash
python step0_extract.py --config member_config.py            # add --only external to re-extract only the external set
python step1_calibrate.py --config member_config.py          # Stage 0, writes calibrated_config.json
python run_A_batch.py   --config member_config.py            # likewise B_external, C_shift, D_ablation, E_family, F_latency, G_attribution
```
Every test after Stage 0 **refuses to run without `calibrated_config.json`**. Extraction is resumable: if it is interrupted,
re-run the same command.

## 3. Demo smoke test (no model needed, ~30 s)
```bash
python demo/make_demo.py
python run_all.py --config demo/demo_config.py --n-shift 40
```
This uses a synthetic dataset and toy model. It only checks that the pipeline works; **do not cite its numbers**.

## 4. Output layout
```
results/<run_id>/
  run_manifest.json          model, dataset, bank_id, seeds, versions
  arrays.npz  samples.csv    cached model outputs (all tests run offline from these)
  0_calibration/             results.json, report.md, calibrated_config.json
  A_batch/  B_external/  C_shift/  D_ablation/  E_family/  F_latency/  G_attribution/
                             each: results.json + report.md
```
The `.npz` arrays are large; **share only the JSON/MD files and `run_manifest.json`** (or the whole folder minus `*.npz`).

## 5. Combining members' results
Each member sends their `results/<run_id>/` folder. Put them all under one `results/` directory, then:
```bash
python combine_results.py                          # all of results/*
python combine_results.py path/runA path/runB      # or list folders
python combine_results.py --allow-incomplete       # include runs missing some tests
```
Writes `COMBINED/COMBINED.md` and `COMBINED/COMBINED.json`. Runs with a different `bank_id` (transform bank) or
`schema_version` are flagged and excluded, so everyone must use the same code version.

## 6. Google Colab (T4) workflow
1. Zip **only** this folder (code only, a few hundred KB) and upload it to Drive. Keep the checkpoint on Drive separately.
2. In Colab: mount Drive, copy the zip, the checkpoint and the **selected subset of images** (not the full dataset) to `/content`, *then* unzip there.
   Copying first avoids slow reads through the Drive mount.
3. `pip install -r requirements.txt`, edit the config paths, `python run_all.py --n 1500`.
4. Download just `results/<run_id>` (without `*.npz`).

Rough cost: one inference pass is ~25 forward passes per image (original plus the 25-transform bank). For 1500 images that is about 40k forwards, roughly
a few minutes on a T4 for a ConvNeXt-Base-sized model. Everything after extraction is CPU-only and takes seconds to minutes.

## 7. Troubleshooting
| Symptom | Fix |
| --- | --- |
| `calibrated_config.json missing` | Run `step1_calibrate.py` (or `run_all.py`) first. |
| "underpowered" warning in a report | Too few errors in the evaluation set. Raise `--n` (aim for 200+ errors) before drawing conclusions. |
| Class order mismatch / garbage AUROC | Check `CLASS_NAMES` order and that preprocessing matches validation. |
| Out of memory | Lower `batch_size` in the adapter. |
| Want to redo only the analyses | `python run_all.py --skip-extract` (cache is reused). |
| Changed model/dataset | Use a new `MODEL_NAME`/`DATASET_*` so a new run folder is created. |
| Disable progress bars | `$env:TQDM_DISABLE="1"` (PowerShell) / `TQDM_DISABLE=1` (bash) |
