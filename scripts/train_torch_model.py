"""Train one PyTorch model configuration with one seed (T3b/T3c) and save checkpoint + metrics.

Usage: python scripts/train_torch_model.py configs/final_lstm.json --seed 0
Outputs: models/<run>.pt (weights + config) and results/metrics/<run>.json (history, validation metrics, environment).
The test set is not touched here; see scripts/evaluate.py.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data  # noqa: E402
from src.metrics import all_metrics  # noqa: E402
from src.train_torch import environment, fit, predict  # noqa: E402


def run(cfg, seed, tag=None):
    data_cfg = data.load_config(data.ROOT / cfg["data_config"])
    scale = float(data_cfg["rul_cap"])
    prepared = data.load_prepared(data_cfg)
    (X_tr, y_tr, _), (X_va, y_va, _) = prepared["train"], prepared["val"]

    model, history, info = fit(cfg, X_tr, y_tr / scale, X_va, y_va / scale, seed)
    val = all_metrics(y_va, predict(model, X_va) * scale)

    name = tag or f"{cfg['name']}_seed{seed}"
    torch.save({"state_dict": model.state_dict(), "config": cfg, "seed": seed,
                "window": X_tr.shape[1], "n_features": X_tr.shape[2], "target_scale": scale},
               data.ROOT / "models" / f"{name}.pt")
    record = {"model": cfg["name"], "run": name, "seed": seed, "config": cfg, "val": val, **info,
              "history": history, "environment": environment()}
    (data.ROOT / "results" / "metrics" / f"{name}.json").write_text(json.dumps(record, indent=2))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", help="override the run name (used for tuning and ablation runs)")
    args = ap.parse_args()
    rec = run(data.load_config(args.config), args.seed, args.tag)
    print(f"{rec['run']}: val RMSE {rec['val']['rmse']:.2f}  epochs {rec['epochs_completed']} "
          f"(best {rec['best_epoch']})  params {rec['n_params']}  {rec['train_seconds']:.0f}s")


if __name__ == "__main__":
    main()
