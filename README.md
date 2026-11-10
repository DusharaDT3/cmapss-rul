# Remaining Useful Life Prediction for Turbofan Engines (NASA C-MAPSS)

NB6057CEM Artificial Neural Networks — CW1 practical project.

- Demonstration video: https://drive.google.com/drive/folders/1sSR8g8CIEi_t4GLRz3SaDFI6rsfcLA5K?usp=sharing
- Source code: https://github.com/DusharaDT3/cmapss-rul

Predict the remaining useful life (RUL, in cycles) of a turbofan engine from observed sensor readings,
to support maintenance planning. Subset: **FD001** (one operating condition, one fault mode).

## Latest neural model: observed-history MLP (v2)

The new neural model beats both Random Forest comparisons on the reused FD001 test set.
Mean test RMSE over seeds 0, 1 and 2 (lower is better):

- **History neural MLP: 9.751 ± 0.433 cycles**; mean MAE 6.836 cycles.
- Newly trained history Random Forest: **10.928 ± 0.099 cycles**.
- Original Random Forest: **12.126 ± 0.128 cycles**.

This is a **19.6% reduction in RMSE** versus the original forest and **10.8%** versus
the new history forest, not an accuracy percentage. The CLI averages three neural models;
its ensemble RMSE is **9.641**, versus **10.916** for the equally sized history-forest ensemble.

Open [notebooks/05_neural_history.ipynb](notebooks/05_neural_history.ipynb) for executed outputs,
all tuning trials, learning curves and error analysis. Full details, metrics, predictions and
selected checkpoints are in [experiments/neural_history_v2/](experiments/neural_history_v2/README.md).

```bash
python scripts/predict_neural_history.py examples/engine_normal.csv
python -m unittest discover -s tests -v
```

The model is a PyTorch MLP: **113 inputs → 128 → 64 → 1**, with SiLU hidden activations,
dropout and 22,913 parameters per run. AdamW, a validation learning-rate scheduler and early
stopping are used. Twelve neural candidates and three forest candidates were selected on
validation data, with all selections frozen before this phase's test evaluation.

Supply the **complete observed history starting at cycle 1**, with at least 30 cycles.
Features include recent-window summaries, initial sensor levels, changes from those levels,
cumulative trends and the current observed cycle. All preprocessing uses training engines only;
future failure time, future observations and unit identity are excluded. The inference model
contains only neural networks. The original `scripts/predict.py` default still reproduces the
original Random Forest demonstration; use the new command above for v2.

V2 adds information beyond the original 30-cycle window. Its new forest comparator receives
that same history information plus ordered pooled-window features. Neural and forest tuning
budgets differ, and the official test set was already seen in the original and v1 phases.
These results are **exploratory**, not independent proof of generalisation or an architecture-only
improvement. Original coursework reports and the existing submission ZIP retain their original
results; disclose this additional experiment and OpenAI Codex assistance if including it.

Training, evaluation and notebook generation are implemented in `scripts/train_neural_history.py`,
`scripts/evaluate_neural_history.py` and `scripts/report_neural_history.py`. Completed experiments
are protected against retraining or overwriting their test evaluation.

## First neural upgrade (v1, earlier exploratory follow-up)

A new feature-based neural model has been trained in `experiments/neural_upgrade/`.
See the executed `notebooks/04_neural_upgrade.ipynb` and the experiment's `README.md` for
the full search, selected architecture, checkpoints and honest comparison. This work began
**after the original test results were known** and is additional to the original coursework protocol.
The three-run mean test RMSE improved from 13.40 (original MLP) and 13.27 (LSTM) to **12.69**;
in that phase, the random forest was better at **12.13**. These are descriptive results on the reused test set.

```bash
python scripts/predict.py examples/engine_normal.csv --model neural_upgrade
python -m unittest discover -s tests -p 'test_neural_upgrade.py' -v
```

The upgraded CLI averages three selected checkpoints (ensemble test RMSE 12.58);
the matching random-forest ensemble scores 12.11. The original inference default remains
`random_forest` to reproduce the original demonstration. Upgrade runs use embedded saved
preprocessing, two hidden layers (128, 64), 140 denoised window features and a bounded output.

Reproduction in a fresh experiment: `scripts/train_neural_upgrade.py`, then
`scripts/evaluate_neural_upgrade.py`, then `scripts/report_neural_upgrade.py`.
Completed runs are protected against further tuning/test overwrites. Original reports and
the existing submission ZIP describe the original experiment; this exploratory extension
must be disclosed if included in a later submission. OpenAI Codex assisted with this extension.

## Dataset

- Source: NASA Prognostics Center of Excellence, *Turbofan Engine Degradation Simulation Data Set* (C-MAPSS).
- Citation: Saxena, A., Goebel, K., Simon, D. and Eklund, N. (2008) 'Damage propagation modeling for aircraft
  engine run-to-failure simulation', *Proc. 1st Int. Conf. on Prognostics and Health Management (PHM08)*, Denver, CO.
- Simulated data, not real flight recordings. Raw files are downloaded by script and not redistributed here.

## Setup (Python 3.12, CPU only)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test (under a minute)

```bash
python scripts/smoke_test.py
```

## Full pipeline

```bash
python scripts/download_data.py                          # T1  fetch C-MAPSS into data/raw/
python scripts/prepare_data.py configs/data_fd001.json   # T1  engine-level split + train-only preprocessing
python scripts/train_numpy_mlp.py configs/numpy_mlp.json # T2  from-scratch NumPy MLP
python scripts/tune.py                                   # T3  pre-registered tuning plan -> configs/final_*.json
python scripts/run_final.py                              # T4a/c  3 seeds per model + dropout ablation (validation only)
python scripts/evaluate.py                               # T4/T5a freeze manifest, then the one test-set evaluation
python scripts/evaluate.py --verify-only                 # check frozen configs/checkpoints against the manifest
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb   # figures and tables
python scripts/word_count.py                             # report word counts using the brief's exclusions
python scripts/build_docs.py                             # rebuild report PDFs/Word files from docs/*.md
python scripts/package_submission.py                     # submission ZIP (add --without-tuning-checkpoints for a smaller file)
```

Inference on an unseen engine (T5b):

```bash
python scripts/predict.py examples/engine_normal.csv                 # normal input
python scripts/predict.py examples/engine_too_short.csv              # invalid: fewer than 30 cycles (exit code 2)
python scripts/predict.py examples/engine_missing_sensor.csv         # invalid: missing column (exit code 2)
python scripts/predict.py examples/engine_other_conditions_fd002.csv # out of scope: other operating conditions (exit code 3)
```

Browser demo — **Flightline / Engine RUL Lab** (local Python HTTP server, no external web services):

```bash
python scripts/demo_app.py
# Open http://localhost:8050
# Use --port 8051 if another process is already using the default port.
```

The normal engine runs when the page opens. The five sections support the CW1 demonstration:

- **Prediction lab:** run saved seed checkpoints on a supplied example or uploaded CSV, inspect the sensor chart, and export the prediction JSON. Invalid and out-of-scope examples can be selected directly.
- **Model architecture:** inspect window MLP, LSTM, history MLP and forest diagrams, layer shapes, parameter counts and saved configurations; explain the engine-level split.
- **Results & trace:** compare saved metrics, inspect a selected seed's configuration/checkpoint fingerprint, and connect its test metric to the current engine's live prediction. Checkpoint fingerprints identify files; they do not replace freeze-manifest verification.
- **Error analysis:** inspect five large errors per ensemble and replay an official test engine through the live models.
- **Demo guide:** follow a six-minute walkthrough, use the rehearsal timer, and switch to full-screen presentation.

Original results and the later history follow-up are labelled separately. A known test label appears only for the bundled normal example and explicitly replayed official test engines; uploaded engine IDs are never treated as known labels. Seed ranges are not confidence intervals. The UI is in `scripts/demo_ui/`; `src/demo_evidence.py` reads the saved evidence without retraining. The older submission ZIP is not automatically rebuilt when the app changes.

Further invalid examples (all exit code 2): `engine_empty.csv`, `engine_header_only.csv`,
`engine_negative_cycles.csv`, `engine_infinite_unit.csv`. The CLI rejects missing, non-numeric and
non-finite (±inf) values, non-integer or non-positive unit and cycle numbers, gaps, and files with more
than one engine.

## Layout

| Path | Contents |
|---|---|
| `configs/` | Data, model, tuning-plan and evaluation-protocol configurations; `final_*.json` are the frozen selections |
| `src/data.py` | Loading, RUL labelling, engine-level split, scaling, windowing |
| `src/numpy_mlp.py` | T2 NumPy MLP: forward, analytical backprop, mini-batch GD, gradient check |
| `src/models.py`, `src/train_torch.py` | T3 PyTorch window MLP and LSTM; training loop with early stopping |
| `src/features.py`, `src/metrics.py`, `src/inference.py` | Baseline features; RMSE / MAE / PHM08 score; checkpoint loading |
| `scripts/` | Command-line entry points (see above); `check_q1.py` verifies the CW2 Q1 arithmetic |
| `notebooks/01_data_audit.ipynb` | T1b data audit and exploratory figures |
| `notebooks/02_numpy_mlp.ipynb` | T2 shapes, gradient verification, loss curves, numerical diagnosis |
| `notebooks/03_results.ipynb` | T3–T5a architectures, learning curves, results, ablation, error and stress analysis |
| `data/splits/` | Saved split indices (engine IDs per partition) |
| `models/` | Preprocessing settings and every checkpoint (`<model>_seed<n>.pt/.pkl`) |
| `results/` | `results_table.csv`, `test_predictions.csv`, `freeze_manifest.json`, `metrics/*.json`, `figures/` |
| `examples/` | Inputs for the inference demonstration |
| `docs/` | Drafts of the CW1 technical record and CW2 report (`.md`, built to `docs/build/*.pdf/.docx` by `scripts/build_docs.py`), AI-use declaration, demo plan and narration, evidence and planning guides |

## Original coursework protocol (summary)

- Training engines are split **by unit** into 80 train / 20 validation (`split_seed = 42`); the official
  `test_FD001.txt` + `RUL_FD001.txt` form the original held-out test set. Test labels are read by
  evaluation entry points (`src/data.py` keeps train/validation and test loaders separate); earlier runs
  loaded them in memory via a shared loader without using them (see the technical record's T1 correction).
- Feature selection (drop channels with ≤ 2 distinct values) and min-max scaling are fitted on training engines only.
- Target: piecewise-linear RUL capped at 125 cycles; input: last 30 cycles × 14 sensors.
- Tuning budget and candidates fixed in `configs/tuning_plan.json`; selection on validation RMSE only.
- Final seeds 0, 1, 2; metrics and stress tests fixed in `configs/evaluation_protocol.json` before test evaluation.
- The explicitly labelled neural follow-ups reuse this already-seen test set for descriptive comparisons;
  they are separate from the original fixed-budget evaluation.

## Environment

Python 3.12.3, Linux, Intel Alder Lake CPU. The laptop also has an NVIDIA RTX 2050 (4 GB), but all reported
runs use the CPU build of PyTorch: the models are small enough that a GPU gives little speed-up, and CPU runs
avoid cuDNN nondeterminism. Exact package versions are in `requirements.txt`; each metrics JSON records the
Python / PyTorch / NumPy versions, platform and thread count of its run.

## Acknowledgements

No pretrained models or borrowed code. Libraries: NumPy, pandas, scikit-learn, PyTorch, Matplotlib.
Generative-AI assistance must be declared according to the institution's assessment policy.
