"""Train the from-scratch NumPy MLP (T2) and save weights, history and validation metrics.

Usage: python scripts/train_numpy_mlp.py configs/numpy_mlp.json
Targets are divided by the RUL cap so the network learns values in [0, 1];
reported RMSE is converted back to cycles. The test set is not used here.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data  # noqa: E402
from src.numpy_mlp import NumpyMLP, train, window_summary  # noqa: E402


def load_inputs(cfg):
    data_cfg = data.load_config(data.ROOT / cfg["data_config"])
    prepared = data.load_prepared(data_cfg)
    scale = float(data_cfg["rul_cap"])
    X_tr, y_tr, _ = prepared["train"]
    X_va, y_va, _ = prepared["val"]
    S_tr = window_summary(X_tr.astype(np.float64))
    S_va = window_summary(X_va.astype(np.float64))
    # Slopes are ~100x smaller than means; standardise with training-engine statistics only.
    mu, sd = S_tr.mean(axis=0), S_tr.std(axis=0)
    return (S_tr - mu) / sd, y_tr / scale, (S_va - mu) / sd, y_va / scale, scale, {"mu": mu, "sd": sd}


def main(config_path):
    cfg = data.load_config(config_path)
    X_tr, y_tr, X_va, y_va, scale, standardiser = load_inputs(cfg)

    model = NumpyMLP(X_tr.shape[1], cfg["n_hidden"], seed=cfg["seed"])
    history, info = train(model, X_tr, y_tr, X_va, y_va, cfg["lr"], cfg["batch_size"],
                          cfg["max_epochs"], cfg["patience"], seed=cfg["seed"])

    val_rmse = float(np.sqrt(np.mean((model.predict(X_va) - y_va) ** 2)) * scale)
    metrics = {
        "model": "numpy_mlp", "config": str(config_path), "seed": cfg["seed"], "n_params": int(model.n_params()),
        "initial_train_loss": float(history["train_loss"][0]), "final_train_loss": float(model.loss(model.forward(X_tr)[0], y_tr)),
        "val_rmse_cycles": val_rmse, **info, "history": [list(map(float, history[k])) for k in ("train_loss", "val_loss")],
    }
    for rel in (cfg["checkpoint"], cfg["metrics_file"]):
        (data.ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
    np.savez(data.ROOT / cfg["checkpoint"], **model.params, **standardiser, target_scale=scale)
    (data.ROOT / cfg["metrics_file"]).write_text(json.dumps(metrics, indent=2))

    print(f"inputs={X_tr.shape[1]} hidden={cfg['n_hidden']} params={model.n_params()}")
    print(f"loss: initial={metrics['initial_train_loss']:.4f} final={metrics['final_train_loss']:.4f} (restored best weights)")
    print(f"{info['status']} after {info['epochs_completed']} epochs (best epoch {info['best_epoch']})")
    print(f"validation RMSE = {val_rmse:.2f} cycles")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/numpy_mlp.json")
