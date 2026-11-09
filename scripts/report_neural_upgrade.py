"""Create a saved-output notebook and a concise record of the follow-up experiment.

Run after evaluate_neural_upgrade.py. Only reads recorded metrics; never retrains
or re-evaluates on test data. The notebook can be executed again to redraw figures.
"""
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/neural_upgrade"


def main():
    result = json.loads((OUT / "test_results.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    s = result["models"]
    new = s["neural_upgrade"]
    gain = 100 * (s["mlp"]["test_rmse"]["mean"] - new["test_rmse"]["mean"]) / s["mlp"]["test_rmse"]["mean"]
    lines = [
        "# Neural upgrade: exploratory follow-up", "",
        "This experiment was requested after the original FD001 test results were already known. "
        "It does not replace the original coursework experiment or provide a fresh independent test.", "",
        "Eight candidates were registered in configs/neural_upgrade_plan.json, each trained with seeds 0 and 1. "
        "Selection used mean validation RMSE on the original 20 validation engines. The winner was then also "
        "trained with seed 2. The test set was evaluated only after selection, with no subsequent tuning.", "",
        f"Selected: {selection['selected_config']['name']}, a 140 → 128 → 64 → 1 MLP with SiLU hidden activations, "
        "dropout 0.1, AdamW weight decay 0.001 and a sigmoid output multiplied by 125.", "",
        "Each of the same 14 sensor channels supplies a window mean, least-squares slope, last-five-cycle mean, "
        "standard deviation and six ordered five-cycle means. The 140 features are standardised using training "
        "engines only. Two hidden layers provide 26,369 trainable parameters. No engine ID, total lifetime, cycle "
        "counter, test target or additional dataset is used as an input.", "",
        f"Validation RMSE: {new['val_rmse']['mean']:.3f} ± {new['val_rmse']['std']:.3f} cycles. "
        f"Test RMSE: {new['test_rmse']['mean']:.3f} ± {new['test_rmse']['std']:.3f}. "
        f"Test MAE: {new['test_mae']['mean']:.3f}. All mean/std figures are across three individual runs.", "",
        f"The mean test RMSE improved by {gain:.2f}% relative to the original MLP's {s['mlp']['test_rmse']['mean']:.3f}. "
        f"It also improved on the original LSTM's {s['lstm']['test_rmse']['mean']:.3f}, "
        f"but did NOT beat the random forest's {s['random_forest']['test_rmse']['mean']:.3f}.", "",
        f"The CLI averages all three selected checkpoints; its ensemble test RMSE is {new['ensemble_test']['rmse']:.3f}. "
        f"The matching three-forest ensemble scores {s['random_forest']['ensemble_test']['rmse']:.3f}. "
        "This distinguishes averaging models from averaging their separate RMSE values.", "",
        "## Use", "",
        "```bash", "python scripts/predict.py examples/engine_normal.csv --model neural_upgrade", "```", "",
        "See notebooks/04_neural_upgrade.ipynb for executed outputs, every search trial, learning curves, "
        "seed results and residuals. Checkpoints and their embedded preprocessing are selected_seed0.pt, "
        "selected_seed1.pt and selected_seed2.pt in this directory.", "",
        "## Reproduction and evidence", "",
        "Run train_neural_upgrade.py, evaluate_neural_upgrade.py and report_neural_upgrade.py in that order "
        "in a fresh experiment directory. Existing completed experiments reject additional tuning/evaluation. "
        "Do not delete evidence to silently repeat a test; use a new declared experiment instead. "
        "training_manifest.json and evaluation_manifest.json bind the plan, source and preprocessing to the runs.", "",
        "## Limits", "",
        "The validation set has already guided earlier work, and the official test set was already seen. "
        "These are descriptive improvements on that benchmark, not proof of independent generalisation. "
        "Input representation, activation, output bounds and optimiser all changed together, so the cause of "
        "the gain is not isolated. No new robustness ablation was performed. FD001 has one operating condition; "
        "this remains a simulation exercise. Original coursework reports and their results are preserved; "
        "any use of this follow-up must be disclosed as additional work after test inspection.", "",
        "Implementation and evaluation of this follow-up used OpenAI Codex assistance.", "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    nb = nbformat.v4.new_notebook()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb.cells = [
        md("# 04 — Neural upgrade (exploratory follow-up)\n\n"
           "The original FD001 test results were known before this work. Selection uses validation only, "
           "but the test comparison below is **descriptive and not independent confirmation**. "
           "Original coursework artifacts are retained. This notebook reads saved results only."),
        code("from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
             "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
             "OUT = ROOT / 'experiments' / 'neural_upgrade'\n"
             "FIG = OUT / 'figures'\nFIG.mkdir(exist_ok=True)\n"
             "read = lambda name: json.loads((OUT / name).read_text())\n"
             "result, selection, search = read('test_results.json'), read('selection.json'), read('search_results.json')\n"
             "comparison = pd.read_csv(OUT / 'comparison.csv')\n"
             "print('Selected architecture:', selection['selected_config'])\n"
             "print('Trainable parameters:', selection['runs'][0]['parameters'])\n"
             "print('All scores use capped RUL (125 cycles); test engines =', result['n_test_engines'])"),
        md("## Fixed search and selection\n\nEight candidates × two tuning seeds; no test score used to rank candidates. "
           "The final winner adds seed 2. All trials, including the weaker CNNs, are retained."),
        code("trials = pd.DataFrame([{'candidate': r['config']['name'], 'architecture': r['config']['architecture'], "
             "'seed 0 val RMSE': r['seed_results'][0]['val_rmse'], 'seed 1 val RMSE': r['seed_results'][1]['val_rmse'], "
             "'mean val RMSE': r['mean_val_rmse']} for r in search])\n"
             "display(trials.sort_values('mean val RMSE').round(4))"),
        md("## Selected model\n\nEach 30 × 14 input becomes 140 train-standardised features: "
           "56 summary features plus 84 ordered five-cycle means. The neural network is "
           "140 → 128 (SiLU, dropout 0.1) → 64 (SiLU, dropout 0.1) → 1 (sigmoid) × 125. "
           "Training uses MSE, AdamW, validation learning-rate reduction and early stopping. "
           "The sigmoid bounds predictions between zero and the cap; it does not guarantee true remaining life."),
        code("fig, ax = plt.subplots(figsize=(9, 4))\n"
             "for seed in result['seeds']:\n"
             "    record = read(f\"{selection['selected_config']['name']}_seed{seed}.json\")\n"
             "    line, = ax.plot(np.sqrt(record['history']['val_mse']) * 125, label=f'seed {seed}: validation')\n"
             "    ax.plot(np.sqrt(record['history']['train_mse']) * 125, '--', color=line.get_color(), alpha=.6, label=f'seed {seed}: training')\n"
             "ax.set(xlabel='Epoch index (0 = first completed epoch)', ylabel='RMSE (cycles)', title='Selected neural model learning curves')\n"
             "ax.legend(ncol=2); fig.tight_layout(); fig.savefig(FIG / 'learning_curves.png', dpi=150); plt.show()"),
        md("## Individual runs and fair ensemble comparisons\n\nMean ± sample SD is calculated across "
           "three separately evaluated seeds. The CLI instead averages the three models' predictions. "
           "Every comparator has a matching three-model ensemble below."),
        code("display(comparison.round(4))\n"
             "summary = pd.DataFrame({name: {'val RMSE mean': v['val_rmse']['mean'], 'test RMSE mean': v['test_rmse']['mean'], "
             "'test RMSE SD': v['test_rmse']['std'], 'test MAE mean': v['test_mae']['mean'], "
             "'ensemble test RMSE': v['ensemble_test']['rmse'], 'ensemble inference ms (batch 1)': v['ensemble_inference_ms_batch1']} "
             "for name, v in result['models'].items()}).T\n"
             "display(summary.round(4))\n"
             "fig, ax = plt.subplots(figsize=(9, 4))\n"
             "ax.bar(summary.index, summary['test RMSE mean'], yerr=summary['test RMSE SD'], capsize=5, color=['gray','gray','gray','steelblue'])\n"
             "ax.set(xlabel='Model', ylabel='Test RMSE (cycles), mean ± SD across seeds', title='Already-seen FD001 test: descriptive follow-up comparison')\n"
             "fig.tight_layout(); fig.savefig(FIG / 'comparison.png', dpi=150); plt.show()"),
        md("## Prediction errors\n\nThe five largest errors are for the selected three-seed ensemble. "
           "Positive residuals overestimate remaining life. These examples do not guide any additional tuning."),
        code("pred = pd.read_csv(OUT / 'test_predictions.csv')\n"
             "pred['error'] = pred['neural_upgrade_ensemble'] - pred['capped_rul']\n"
             "display(pred.loc[pred.error.abs().nlargest(5).index, ['unit','true_rul','capped_rul','neural_upgrade_ensemble','error']].round(2))\n"
             "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
             "axes[0].scatter(pred.capped_rul, pred.neural_upgrade_ensemble, s=20)\n"
             "axes[0].plot([0,125], [0,125], 'k--'); axes[0].set(xlabel='True capped RUL (cycles)', ylabel='Predicted RUL (cycles)', title='Neural upgrade: three-seed ensemble')\n"
             "axes[1].scatter(pred.capped_rul, pred.error, s=20); axes[1].axhline(0, color='black', linestyle='--')\n"
             "axes[1].set(xlabel='True capped RUL (cycles)', ylabel='Predicted − true (cycles)', title='Residuals on 100 test engines')\n"
             "fig.tight_layout(); fig.savefig(FIG / 'prediction_errors.png', dpi=150); plt.show()"),
        md(f"## Interpretation\n\nThe upgraded neural model reduced mean test RMSE from the old MLP's "
           f"{s['mlp']['test_rmse']['mean']:.2f} to {new['test_rmse']['mean']:.2f} cycles ({gain:.1f}% lower RMSE). "
           f"It also beat the old LSTM ({s['lstm']['test_rmse']['mean']:.2f}), but the forest remains better "
           f"({s['random_forest']['test_rmse']['mean']:.2f}). No claim of beating all models is supported.\n\n"
           "Changes in features, output bounds, activation and optimiser were made together; an ablation would "
           "be needed to identify the source of improvement. The old test set is reused and the validation set "
           "has seen additional tuning, so new independent evaluation is required. This follow-up should be "
           "disclosed separately from the original fixed-budget coursework.\n\n"
           "Inference: `python scripts/predict.py examples/engine_normal.csv --model neural_upgrade`"),
    ]
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = ROOT / "notebooks/04_neural_upgrade.ipynb"
    nbformat.write(nb, path)
    NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, path)
    print("Wrote executed notebook:", path)
    print("Wrote experiment record:", OUT / "README.md")


if __name__ == "__main__":
    main()
