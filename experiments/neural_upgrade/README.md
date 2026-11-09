# Neural upgrade: exploratory follow-up

This experiment was requested after the original FD001 test results were already known. It does not replace the original coursework experiment or provide a fresh independent test.

Eight candidates were registered in configs/neural_upgrade_plan.json, each trained with seeds 0 and 1. Selection used mean validation RMSE on the original 20 validation engines. The winner was then also trained with seed 2. The test set was evaluated only after selection, with no subsequent tuning.

Selected: pooled_regularised, a 140 → 128 → 64 → 1 MLP with SiLU hidden activations, dropout 0.1, AdamW weight decay 0.001 and a sigmoid output multiplied by 125.

Each of the same 14 sensor channels supplies a window mean, least-squares slope, last-five-cycle mean, standard deviation and six ordered five-cycle means. The 140 features are standardised using training engines only. Two hidden layers provide 26,369 trainable parameters. No engine ID, total lifetime, cycle counter, test target or additional dataset is used as an input.

Validation RMSE: 12.090 ± 0.098 cycles. Test RMSE: 12.690 ± 0.197. Test MAE: 9.644. All mean/std figures are across three individual runs.

The mean test RMSE improved by 5.28% relative to the original MLP's 13.398. It also improved on the original LSTM's 13.268, but did NOT beat the random forest's 12.126.

The CLI averages all three selected checkpoints; its ensemble test RMSE is 12.584. The matching three-forest ensemble scores 12.110. This distinguishes averaging models from averaging their separate RMSE values.

## Use

```bash
python scripts/predict.py examples/engine_normal.csv --model neural_upgrade
```

See notebooks/04_neural_upgrade.ipynb for executed outputs, every search trial, learning curves, seed results and residuals. Checkpoints and their embedded preprocessing are selected_seed0.pt, selected_seed1.pt and selected_seed2.pt in this directory.

## Reproduction and evidence

Run train_neural_upgrade.py, evaluate_neural_upgrade.py and report_neural_upgrade.py in that order in a fresh experiment directory. Existing completed experiments reject additional tuning/evaluation. Do not delete evidence to silently repeat a test; use a new declared experiment instead. training_manifest.json and evaluation_manifest.json bind the plan, source and preprocessing to the runs.

## Limits

The validation set has already guided earlier work, and the official test set was already seen. These are descriptive improvements on that benchmark, not proof of independent generalisation. Input representation, activation, output bounds and optimiser all changed together, so the cause of the gain is not isolated. No new robustness ablation was performed. FD001 has one operating condition; this remains a simulation exercise. Original coursework reports and their results are preserved; any use of this follow-up must be disclosed as additional work after test inspection.

Implementation and evaluation of this follow-up used OpenAI Codex assistance.
