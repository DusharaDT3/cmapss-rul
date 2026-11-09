"""Train the frozen final configurations with seeds 0-2, then the dropout ablation (T4a, T4c).

Usage: python scripts/run_final.py
Requires configs/final_*.json from scripts/tune.py. Uses training and validation data only.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_baseline, train_torch_model  # noqa: E402
from src import data  # noqa: E402

SEEDS = [0, 1, 2]


def main():
    cfgs = {m: data.load_config(data.ROOT / "configs" / f"final_{m}.json") for m in ("random_forest", "mlp", "lstm")}
    val_rmse = {}
    for name, cfg in cfgs.items():
        runner = train_baseline if name == "random_forest" else train_torch_model
        val_rmse[name] = []
        for seed in SEEDS:
            rec = runner.run(cfg, seed)
            val_rmse[name].append(rec["val"]["rmse"])
            print(f"{rec['run']:22s} val RMSE {rec['val']['rmse']:.2f}", flush=True)

    # Ablation on the neural architecture with the better mean validation RMSE.
    arch = min(("mlp", "lstm"), key=lambda m: np.mean(val_rmse[m]))
    ablated = {**cfgs[arch], "dropout": 0.0, "name": f"{arch}_nodropout"}
    (data.ROOT / "configs" / f"ablation_{arch}_nodropout.json").write_text(json.dumps(ablated, indent=2))
    val_rmse[ablated["name"]] = []
    for seed in SEEDS:
        rec = train_torch_model.run(ablated, seed)
        val_rmse[ablated["name"]].append(rec["val"]["rmse"])
        print(f"{rec['run']:22s} val RMSE {rec['val']['rmse']:.2f}", flush=True)

    summary = {k: {"seeds": SEEDS, "val_rmse": v, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
               for k, v in val_rmse.items()}
    summary["ablation_architecture"] = arch
    (data.ROOT / "results" / "metrics" / "final_validation.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"{k:18s} val RMSE {v['mean']:.2f} +- {v['std']:.2f}")


if __name__ == "__main__":
    main()
