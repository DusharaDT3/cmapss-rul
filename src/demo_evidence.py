"""Read-only presentation evidence from saved CW1 artifacts (never trains a model)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from torch import nn

from src import data
from src.models import count_params, layer_table

ROOT = data.ROOT


def artifact(path):
    p = ROOT / path
    digest = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': path, 'bytes': p.stat().st_size, 'sha256': digest.hexdigest()}


def build_evidence(runs, history, cfg):
    original = pd.read_csv(ROOT / 'results/results_table.csv')
    followup = pd.read_csv(ROOT / 'experiments/neural_history_v2/comparison.csv')
    labels = {'random_forest': 'Random forest', 'mlp': 'Window MLP', 'lstm': 'LSTM', 'history': 'History MLP'}
    rows = []
    for frame, phase in [(original, 'original'), (followup, 'exploratory')]:
        for name, group in frame.groupby('model', sort=False):
            label = labels.get(name, 'History MLP')
            if phase == 'exploratory' and name == 'random_forest':
                label = 'History random forest'
            rows.append({'model': label, 'phase': phase, 'n': len(group),
                         **{k: float(group[k].mean()) for k in ['val_rmse', 'test_rmse', 'test_mae']},
                         'std': float(group.test_rmse.std(ddof=1)),
                         'seeds': group.seed.astype(int).tolist()})
    architectures, traces = {}, {}
    for name, seed_runs in runs.items():
        run = seed_runs[0]
        conf = run.config
        if name == 'random_forest':
            layers = [['Input window', [1, 30, 14], 0], ['Mean / slope / last / std', [1, 56], 0],
                      [f'{conf["n_estimators"]} regression trees', [1, conf['n_estimators']], None], ['Tree mean', [1], 0]]
            nodes = sum(t.tree_.node_count for t in run.model.estimators_)
            blocks = [['30 × 14', 'Sensor window'], ['56', 'Summary features'], [str(conf['n_estimators']), 'Regression trees'], ['1', 'RUL in cycles']]
            description = 'Each sensor contributes mean, slope, last value and standard deviation. Tree predictions are averaged. Three seed forests are then averaged in the demo.'
            params = None
        else:
            layers = layer_table(run.model, cfg['window'], len(data.load_config(ROOT / cfg['preprocess_file'])['features']))
            params, nodes = count_params(run.model), None
            if name == 'mlp':
                blocks = [['30 × 14', 'Sensor window'], ['420', 'Flatten'], [str(conf['hidden'][0]), 'Linear · ReLU'], [str(conf['hidden'][1]), 'Linear · ReLU'], ['1', 'Linear output']]
                description = 'A feed-forward model reads the last 30 cycles as one vector. Dropout follows each ReLU during training; it is disabled for inference.'
            else:
                blocks = [['30 × 14', 'Sequence input'], [str(conf['hidden']), 'LSTM layer 1'], [str(conf['hidden']), 'LSTM layer 2'], [str(conf['hidden']), 'Last time step'], ['1', 'Linear head']]
                description = 'The recurrent model reads 30 time steps in order. Sigmoid gates and tanh updates preserve a cell state; the final hidden state feeds a linear head. Dropout is used between layers and before the head during training.'
        architectures[name] = {'name': labels[name], 'phase': 'original', 'layers': layers, 'blocks': blocks,
                               'parameters': params, 'tree_nodes': nodes, 'description': description, 'config': conf,
                               'postprocessing': 'Neural outputs are multiplied by 125; all original outputs are clipped at zero. The demo averages predictions from seeds 0, 1 and 2.'}
        records = []
        for run in seed_runs:
            row = original[(original.model == name) & (original.seed == run.seed)].iloc[0]
            records.append({'seed': run.seed, 'checkpoint': artifact(str(run.path.relative_to(ROOT))),
                            'metrics': {k: float(row[k]) for k in ['val_rmse', 'test_rmse', 'test_mae']}})
        traces[name] = {'config_path': f'configs/final_{name}.json', 'config': conf,
                        'preprocessing': artifact(cfg['preprocess_file']), 'records': records,
                        'metrics_path': 'results/results_table.csv', 'script': 'scripts/predict.py',
                        'command': f'python scripts/predict.py examples/engine_normal.csv --model {name}'}
    if history:
        run = history[0]
        n_in = run.model.net[0].in_features
        layers = [['Standardised history features', [1, n_in], 0]]
        width = n_in
        for layer in run.model.net:
            if isinstance(layer, nn.Linear):
                width = layer.out_features
            layers.append([type(layer).__name__, [1, width], sum(p.numel() for p in layer.parameters())])
        architectures['history'] = {'name': 'History MLP', 'phase': 'exploratory', 'layers': layers,
            'blocks': [[str(n_in), 'History features'], ['128', 'Linear · SiLU'], ['64', 'Linear · SiLU'], ['1', 'Linear output']],
            'parameters': count_params(run.model), 'tree_nodes': None, 'config': run.config,
            'description': 'Uses recent summaries, initial sensor levels, change from initial levels, cumulative means and slopes, and observed cycle. Only history up to the prediction time is used. SiLU and training dropout follow each hidden linear layer.',
            'postprocessing': 'Saved sensor scaling and feature standardisation → neural output clipped to [0, 1] → ×125 cycles → mean of three seed models.'}
        traces['history'] = {'config_path': 'experiments/neural_history_v2/selection.json', 'config': run.config,
            'preprocessing': {'path': 'Embedded sensor_scaler and scaler in each history checkpoint', 'sha256': None},
            'records': [{'seed': r.seed, 'checkpoint': artifact(f'experiments/neural_history_v2/selected_neural_seed{r.seed}.pt'),
                        'metrics': {k: float(followup[(followup.model == 'neural') & (followup.seed == r.seed)].iloc[0][k]) for k in ['val_rmse','test_rmse','test_mae']}}
                       for r in history], 'metrics_path': 'experiments/neural_history_v2/comparison.csv',
            'script': 'scripts/predict_neural_history.py',
            'command': 'python scripts/predict_neural_history.py examples/engine_normal.csv'}
    predictions = pd.read_csv(ROOT / 'results/test_predictions.csv')
    hist_predictions = pd.read_csv(ROOT / 'experiments/neural_history_v2/test_predictions.csv')
    failures = []
    for key in ['random_forest', 'mlp', 'lstm', 'history']:
        values = hist_predictions.set_index('unit').loc[predictions.unit, 'neural_ensemble'].to_numpy() if key == 'history' else predictions[[f'{key}_seed{s}' for s in [0,1,2]]].mean(axis=1).to_numpy()
        truth = np.minimum(predictions.true_rul.to_numpy(), cfg['rul_cap'])
        for i in np.argsort(abs(values-truth))[-5:][::-1]:
            failures.append({'model': key, 'unit': int(predictions.unit.iloc[i]), 'observed': int(predictions.observed_cycles.iloc[i]),
                             'true_rul': float(truth[i]), 'predicted': float(values[i]), 'error': float(values[i]-truth[i])})
    split = data.load_config(ROOT / cfg['split_file'])
    return {'models': architectures, 'traces': traces, 'results': rows, 'failures': failures,
            'split': {'train': len(split['train_units']), 'validation': len(split['val_units']), 'seed': split['split_seed'], 'file': cfg['split_file']},
            'test_engines': len(predictions), 'features': data.load_config(ROOT / cfg['preprocess_file'])['features'],
            'truth': {str(int(row.unit)): float(min(row.true_rul, cfg['rul_cap'])) for row in predictions.itertuples()}}
