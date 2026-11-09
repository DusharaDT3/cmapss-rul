"""Build an executed notebook and narrative from v2's saved results (no retraining)."""
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/neural_history_v2'


def main():
    result=json.loads((OUT/'test_results.json').read_text())
    selected=result['selection']['models']['neural']['config']
    nn=result['models']['neural'];rf=result['models']['random_forest']
    old=json.loads((ROOT/'results/metrics/test_metrics.json').read_text())
    prior=json.loads((ROOT/'experiments/neural_upgrade/test_results.json').read_text())
    best=nn['test_rmse']['mean']<rf['test_rmse']['mean']
    old_won=nn['test_rmse']['mean']<old['random_forest']['test_rmse']['mean']
    gain=100*(old['random_forest']['test_rmse']['mean']-nn['test_rmse']['mean'])/old['random_forest']['test_rmse']['mean']
    verdict=('The neural model has lower mean test RMSE than the newly trained history Random Forest.' if best else 'The newly trained history Random Forest still has lower mean test RMSE.')
    lines=['# History neural model: second exploratory follow-up','',
           'This phase was explicitly requested after the original and first-upgrade test results were known. It is not a fresh held-out validation of generalisation.','',
           '## What changed','',
           'The model now uses the complete observed engine history from cycle 1. It uses only information available at the prediction endpoint: last-30-cycle summaries, first-ten-cycle sensor means, change from that initial level, cumulative means/slopes, and the observed cycle number. The cycle counter is observable; the future failure cycle and full run-to-failure lifetime are never features. Tests verify that changing future rows or target values cannot alter features.','',
           'The original split remains 80 training / 20 validation engines. Min-max sensor scaling and feature standardisation use training engines only. No test examples are used in fitting. Twelve neural candidates and three Random Forest candidates are evaluated with seeds 0 and 1; the separate winners add seed 2. Different search budgets are disclosed, so this does not isolate architecture alone.','',
           f'Selected neural configuration: {selected}. The history representation has 113 inputs; the two hidden layers have 128 and 64 units, SiLU activations and dropout. Output is linear, clipped to the same [0,125] target range at inference. Training uses AdamW, a validation learning-rate scheduler and early stopping.','',
           'The newly trained Random Forest receives the same causal history information and additionally the six ordered five-cycle sensor means (197 features); this gives the comparator at least the information available to the selected neural model. Neither new model uses the forest as a neural input or component.','',
           '## Measured results','',
           f"Neural validation RMSE: {nn['val_rmse']['mean']:.3f} ± {nn['val_rmse']['std']:.3f} cycles.",
           f"Neural test RMSE: {nn['test_rmse']['mean']:.3f} ± {nn['test_rmse']['std']:.3f}; MAE: {nn['test_mae']['mean']:.3f}.",
           f"History Random Forest test RMSE: {rf['test_rmse']['mean']:.3f} ± {rf['test_rmse']['std']:.3f}.",
           f"Original Random Forest test RMSE: {old['random_forest']['test_rmse']['mean']:.3f}.",
           verdict,
           f"The neural model {'beats' if old_won else 'does not beat'} the original Random Forest; relative RMSE change is {gain:.2f}% lower.",'',
           f"Three-model neural ensemble test RMSE (the CLI): {nn['ensemble_test']['rmse']:.3f}; matching new forest ensemble: {rf['ensemble_test']['rmse']:.3f}. Report individual-run averages separately from errors of averaged predictions.",'',
           '## Run','',
           '```bash','python scripts/predict_neural_history.py examples/engine_normal.csv',
           "python -m unittest discover -s tests -p 'test_neural_history.py' -v",'```','',
           'Inputs must contain the full observed history starting at cycle 1 and at least 30 cycles. Providing only the last 30 cycles would lose the initial baseline and is rejected if its cycle counter does not start at 1.','',
           'See notebooks/05_neural_history.ipynb for all trials, learning curves, per-seed results, error analysis and comparisons. Selected weights and both preprocessing stages are embedded in selected_neural_seed0.pt through selected_neural_seed2.pt.','',
           '## Limits and reproducibility','',
           'Original results and upgrade-v1 remain frozen. Fresh training uses scripts/train_neural_history.py, followed by scripts/evaluate_neural_history.py and scripts/report_neural_history.py. Completed experiments refuse further tuning/evaluation. Hash manifests link source, training data, plans, configurations and selected checkpoints.','',
           'The original model comparison used only the last 30 cycles, while v2 adds initial conditions and all past observations. Improvement over the original forest therefore combines extra information with architecture/training changes; it cannot be attributed solely to a better neural architecture. The new history forest provides a stronger comparator, but the neural search has more candidates. The validation set is reused, and the official test set was seen in two prior phases. New independent evaluation is required. No new noise or subgroup robustness claim is made.','',
           'This is additional post-test exploratory work, not a replacement for the original fixed-budget coursework. OpenAI Codex assisted with implementation, training, tests and this record.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    nb=nbformat.v4.new_notebook()
    nb.cells=[
      md('# 05 — Neural model with causal engine history\n\nSecond exploratory follow-up after original and upgrade-v1 test results were known. No independent generalisation claim. Models are selected using validation only; all saved test results are reported without choosing a best test seed.'),
      code("from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nROOT=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()\nOUT=ROOT/'experiments/neural_history_v2'\nFIG=OUT/'figures';FIG.mkdir(exist_ok=True)\nread=lambda p:json.loads(Path(p).read_text())\nresult=read(OUT/'test_results.json')\nsearch=read(OUT/'search_results.json')\nselection=read(OUT/'selection.json')\nprint('Neural winner:',selection['models']['neural']['config'])\nprint('Forest winner:',selection['models']['random_forest']['config'])"),
      md('## Causal features and architecture\n\nFor each prediction, use only its observed prefix: 56 recent-window summaries, 14 initial means, 14 recent-minus-initial changes, 14 cumulative means, 14 cumulative slopes and one observed cycle feature: **113 inputs**. No unit identity, future rows or total engine lifetime.\n\nThe selected MLP has 128 and 64 SiLU hidden units, dropout, linear output and clipping to [0,125]. The new forest gets the same history plus 84 ordered pooled-window features. Extra information and unequal tuning budgets limit architecture-only conclusions.'),
      code("trials=pd.DataFrame([{'kind':r['kind'],'candidate':r['config']['name'],'seed 0 val RMSE':r['runs'][0]['val_rmse'],'seed 1 val RMSE':r['runs'][1]['val_rmse'],'mean val RMSE':r['mean_val_rmse']} for r in search])\ndisplay(trials.sort_values(['kind','mean val RMSE']).round(4))"),
      md('## Learning curves\n\nValidation error includes the predetermined clipping used by inference. Training error is computed before clipping, with dropout active. Best-epoch weights are restored.'),
      code("fig,ax=plt.subplots(figsize=(9,4))\nfor seed in result['seeds']:\n    r=read(OUT/f\"{selection['models']['neural']['config']['name']}_seed{seed}.json\")\n    line,=ax.plot(np.arange(1,len(r['history']['val_mse'])+1),125*np.sqrt(r['history']['val_mse']),label=f'Validation seed {seed}')\n    ax.plot(np.arange(1,len(r['history']['train_mse'])+1),125*np.sqrt(r['history']['train_mse']),'--',color=line.get_color(),alpha=.6,label=f'Training seed {seed}')\nax.set(xlabel='Completed epoch',ylabel='RMSE (cycles)',title='History MLP learning curves')\nax.legend(ncol=2);fig.tight_layout();fig.savefig(FIG/'learning_curves.png',dpi=150);plt.show()"),
      md('## Comparison on the already-seen FD001 test set\n\nCompare seed means to seed means and ensembles to equally sized ensembles. Both new models have more history information than the original models. These are descriptive results on 100 test engines.'),
      code("display(pd.read_csv(OUT/'comparison.csv').round(4))\nold=read(ROOT/'results/metrics/test_metrics.json')\nv1=read(ROOT/'experiments/neural_upgrade/test_results.json')['models']\nrows=[]\nfor label,r in [('Original RF',old['random_forest']),('Original MLP',old['mlp']),('Original LSTM',old['lstm']),('Upgrade v1',v1['neural_upgrade']),('History RF',result['models']['random_forest']),('History neural',result['models']['neural'])]:\n    rows.append({'model':label,'test RMSE mean':r['test_rmse']['mean'],'test RMSE SD':r['test_rmse']['std'],'test MAE mean':r['test_mae']['mean']})\nsummary=pd.DataFrame(rows).set_index('model');display(summary.round(4))\nfig,ax=plt.subplots(figsize=(10,4))\nax.bar(summary.index,summary['test RMSE mean'],yerr=summary['test RMSE SD'],capsize=4,color=['gray']*5+['steelblue'])\nax.set(xlabel='Model (history models receive additional causal information)',ylabel='Test RMSE (cycles), mean ± seed SD',title='FD001 exploratory comparison; lower RMSE is better')\nax.tick_params(axis='x',rotation=15);fig.tight_layout();fig.savefig(FIG/'comparison.png',dpi=150);plt.show()\nensembles=pd.DataFrame({name:r['ensemble_test'] for name,r in result['models'].items()}).T\ndisplay(ensembles.round(4))"),
      md('## Errors of the selected three-neural-model ensemble\n\nPositive residuals overestimate remaining life. The five largest errors are inspected without further tuning.'),
      code("pred=pd.read_csv(OUT/'test_predictions.csv');pred['error']=pred.neural_ensemble-pred.capped_rul\ndisplay(pred.loc[pred.error.abs().nlargest(5).index,['unit','true_rul','capped_rul','neural_ensemble','error']].round(3))\nfig,axes=plt.subplots(1,2,figsize=(10,4))\naxes[0].scatter(pred.capped_rul,pred.neural_ensemble,s=18);axes[0].plot([0,125],[0,125],'k--')\naxes[0].set(xlabel='True capped RUL (cycles)',ylabel='Predicted RUL (cycles)',title='History neural ensemble')\naxes[1].scatter(pred.capped_rul,pred.error,s=18);axes[1].axhline(0,color='black',linestyle='--')\naxes[1].set(xlabel='True capped RUL (cycles)',ylabel='Predicted − true (cycles)',title='Residuals on 100 engines')\nfig.tight_layout();fig.savefig(FIG/'errors.png',dpi=150);plt.show()"),
      md('## Interpretation\n\n'+verdict+f" The new neural mean test RMSE is {nn['test_rmse']['mean']:.3f}, versus {old['random_forest']['test_rmse']['mean']:.3f} for the original forest.\n\n"+
         'Full observed history adds useful information but also changes the original comparison. Results do not prove performance on real engines, other operating conditions or new independent data. The first two test evaluations were known before v2 began, and tuning budgets differ. All individual seeds and unsuccessful trials are retained.\n\n'+
         'Use: `python scripts/predict_neural_history.py examples/engine_normal.csv`. Supply complete observed history beginning at cycle 1. Predictions use neural networks only.'),
    ]
    nb.metadata.kernelspec={'name':'python3','display_name':'Python 3','language':'python'}
    path=ROOT/'notebooks/05_neural_history.ipynb';nbformat.write(nb,path)
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,path);print('Created',path);print(verdict)


if __name__=='__main__':main()
