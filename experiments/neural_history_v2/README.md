# History neural model: second exploratory follow-up

This phase was explicitly requested after the original and first-upgrade test results were known. It is not a fresh held-out validation of generalisation.

## What changed

The model now uses the complete observed engine history from cycle 1. It uses only information available at the prediction endpoint: last-30-cycle summaries, first-ten-cycle sensor means, change from that initial level, cumulative means/slopes, and the observed cycle number. The cycle counter is observable; the future failure cycle and full run-to-failure lifetime are never features. Tests verify that changing future rows or target values cannot alter features.

The original split remains 80 training / 20 validation engines. Min-max sensor scaling and feature standardisation use training engines only. No test examples are used in fitting. Twelve neural candidates and three Random Forest candidates are evaluated with seeds 0 and 1; the separate winners add seed 2. Different search budgets are disclosed, so this does not isolate architecture alone.

Selected neural configuration: {'name': 'history_wide_linear', 'representation': 'history', 'hidden': [128, 64], 'dropout': 0.1, 'weight_decay': 0.001, 'output': 'linear'}. The history representation has 113 inputs; the two hidden layers have 128 and 64 units, SiLU activations and dropout. Output is linear, clipped to the same [0,125] target range at inference. Training uses AdamW, a validation learning-rate scheduler and early stopping.

The newly trained Random Forest receives the same causal history information and additionally the six ordered five-cycle sensor means (197 features); this gives the comparator at least the information available to the selected neural model. Neither new model uses the forest as a neural input or component.

## Measured results

Neural validation RMSE: 7.905 ± 0.132 cycles.
Neural test RMSE: 9.751 ± 0.433; MAE: 6.836.
History Random Forest test RMSE: 10.928 ± 0.099.
Original Random Forest test RMSE: 12.126.
The neural model has lower mean test RMSE than the newly trained history Random Forest.
The neural model beats the original Random Forest; relative RMSE change is 19.58% lower.

Three-model neural ensemble test RMSE (the CLI): 9.641; matching new forest ensemble: 10.916. Report individual-run averages separately from errors of averaged predictions.

## Run

```bash
python scripts/predict_neural_history.py examples/engine_normal.csv
python -m unittest discover -s tests -p 'test_neural_history.py' -v
```

Inputs must contain the full observed history starting at cycle 1 and at least 30 cycles. Providing only the last 30 cycles would lose the initial baseline and is rejected if its cycle counter does not start at 1.

See notebooks/05_neural_history.ipynb for all trials, learning curves, per-seed results, error analysis and comparisons. Selected weights and both preprocessing stages are embedded in selected_neural_seed0.pt through selected_neural_seed2.pt.

## Limits and reproducibility

Original results and upgrade-v1 remain frozen. Fresh training uses scripts/train_neural_history.py, followed by scripts/evaluate_neural_history.py and scripts/report_neural_history.py. Completed experiments refuse further tuning/evaluation. Hash manifests link source, training data, plans, configurations and selected checkpoints.

The original model comparison used only the last 30 cycles, while v2 adds initial conditions and all past observations. Improvement over the original forest therefore combines extra information with architecture/training changes; it cannot be attributed solely to a better neural architecture. The new history forest provides a stronger comparator, but the neural search has more candidates. The validation set is reused, and the official test set was seen in two prior phases. New independent evaluation is required. No new noise or subgroup robustness claim is made.

This is additional post-test exploratory work, not a replacement for the original fixed-budget coursework. OpenAI Codex assisted with implementation, training, tests and this record.
