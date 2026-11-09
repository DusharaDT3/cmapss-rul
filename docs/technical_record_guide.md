# CW1 technical record: evidence map (write the record yourself, max 1500 words)

The brief asks for the record organised under T1–T6, a results table linking each model to its
configuration, seed, validation score and test score, and the requested plots. Below: where each piece of
evidence lives and the facts to cite. Numbers come from `results/` and the executed notebooks.

## Results table (required)
Use `results/results_table.csv` directly (9 rows: model, config, seed, checkpoint, val RMSE, test RMSE, MAE,
PHM08 score, uncapped RMSE). Add the NumPy MLP (T2) as a separate validation-only row:
`results/metrics/numpy_mlp_seed0.json`, val RMSE 12.99, test set not used.

## T1 Problem and data (~250 words)
- **You write T1a:** intended user (maintenance planner), target (RUL in cycles), unit of observation
  (one engine's last 30 cycles), practical value, licence/access terms from the NASA page, ethics
  (simulated data; safety-critical use; no personal data).
- Facts: 100 train / 100 test engines; 20,631 train rows; 0 missing, 0 duplicates, 0 cycle gaps;
  7 constant channels + s6 (2 values) dropped -> 14 sensors; lifetimes 128–362 cycles; shortest test engine 31 cycles.
- Figures: fig01 (lifetimes), fig02 (sensor trends), fig03 (correlation), fig04 (cap), fig05 (target distribution).
- Leakage: engine-level split 80/20 (`data/splits/fd001_split.json`); scaler and feature choice fitted on
  training engines only (`models/preprocess_fd001.json`); `cycle` not an input; test labels read only by
  `scripts/evaluate.py` after the loader split (earlier runs loaded them unused: see the T1 correction).

## T2 NumPy MLP (~250 words)
- 28 inputs (window mean + slope, standardised) -> 32 ReLU -> 1 linear; 961 params; ½·MSE.
- Loss 1.049 -> 0.0054; early-stopped at epoch 57 (best epoch 37); lr 0.05, batch 64, patience 20.
- Gradient check: 10 entries over W1, b1, W2, b2; eps 1e-5; batch 8; min |z| 0.0067 vs max perturbation 2.4e-5;
  max relative error 1.3e-10 (tolerance 1e-6, justified in notebook 02).
- Diagnosis: 16-example overfit 0.907 -> 4.3e-5; lr 0.5 diverges in epoch 1 and is detected; lr 0.005 slow.
- Unscaled inputs gave val RMSE 20.45, standardised 12.99 (slopes ~100x smaller than means).
- Figures: fig06, fig07, fig08.

## T3 Models (~300 words)
- Baseline: random forest on 56 window features (`src/features.py`), same windows, split and metric.
- Layer tables: notebook 03, section T3b. MLP 62,209 params; LSTM 53,825 params.
- Training control: Adam 1e-3, weight decay 1e-5, batch 256, dropout 0.1, early stopping (patience 10),
  best-epoch weights saved in `models/<model>_seed<n>.pt`. Learning curves: fig09.
- Fair comparison: `configs/tuning_plan.json` written before tuning; 11 trials in `results/metrics/tuning.json`;
  selection on validation RMSE, seed 0; frozen configs `configs/final_*.json`; hashes in `results/freeze_manifest.json`.
- Traceability: preparation (`src/data.py`), training (`src/train_torch.py`, `scripts/train_*.py`),
  evaluation (`scripts/evaluate.py`) are separate; environment recorded in every metrics JSON. No borrowed code or
  pretrained weights. **Declare AI assistance per your institution's policy.**

## T4 Experiments (~400 words)
- Seeds 0–2, fixed split. Val RMSE: RF 12.79±0.03, MLP 12.81±0.02, LSTM 12.83±0.11.
  Test RMSE: RF 12.13±0.13, MLP 13.40±0.44, LSTM 13.27±0.43. PHM08: 226 / 281 / 280.
- Metric justification: RMSE primary (cycles, penalises large errors); MAE + PHM08 (late errors worse).
- Plots: fig10 (predicted vs observed + residuals), fig11 (time-ordered validation trajectories).
- Ablation (MLP, dropout 0.1 -> 0): val RMSE 12.81 -> 12.67, consistent across all 3 seeds; claim not supported;
  confounders listed in notebook 03.
- Errors: 5 largest (engines 93, 45, 3 early; 67, 50 late with short histories) + 5 near-median (~±7 cycles); fig12.
- Stress tests (protocol in `configs/evaluation_protocol.json`, written before testing): short vs long history,
  n = 50 each; Gaussian noise sigma 0.01–0.10, fig13. RF degrades fastest under noise (24.9 at sigma 0.10 vs MLP 18.3).

## T5 Recommendation and use (~200 words)
- Recommendation and reasoning: final section of notebook 03 (RF for clean FD001-like data; MLP if noise
  is plausible; LSTM not justified). Timing: RF 55.7 ms, MLP 0.21 ms, LSTM 0.62 ms per engine (batch 1, CPU, 8 threads).
- CLI: `scripts/predict.py`; demo outputs below. Limitations: simulated data, one condition and fault mode,
  100 test engines, short-history weakness, cap assumption, no deployment claim.

## T6 Reproducibility (~100 words)
- README commands, `requirements.txt` pins, `scripts/smoke_test.py` (about 7 s), saved splits, seeds, configs,
  checkpoints, machine-readable metrics. Raw data fetched by `scripts/download_data.py`, not redistributed.
- Reproducibility evidence: the tuning trials were re-run from saved configs and reproduced the recorded
  validation scores (see `results/metrics/tuning_rerun_check.txt`).

## Inference demonstration outputs (T5b)
```
$ python scripts/predict.py examples/engine_normal.csv
Engine 31: 196 cycles observed, last cycle 196
Estimated remaining useful life: 11 cycles  (seed range 11-11, model: random_forest)
Interpretation: maintenance recommended now (predicted to fail within ~30 cycles)
   -> true RUL of test engine 31 is 8 cycles (RUL_FD001.txt line 31): prediction 3 cycles late

$ python scripts/predict.py examples/engine_too_short.csv          -> REJECTED (invalid input), exit code 2
$ python scripts/predict.py examples/engine_missing_sensor.csv     -> REJECTED (invalid input), exit code 2
$ python scripts/predict.py examples/engine_other_conditions_fd002.csv -> REJECTED (out of scope), exit code 3
```
