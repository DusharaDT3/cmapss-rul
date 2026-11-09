"""Follow-up neural models with denoised window features and bounded RUL outputs.

The original coursework models and results remain a separate frozen experiment.
This module never reads raw data or test labels.
"""
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.numpy_mlp import window_summary


def features(X, representation):
    """Use only observations inside each input window; no unit/cycle/target features."""
    X = np.asarray(X, dtype=np.float32)
    if representation == "sequence":
        return X
    # Mean and slope plus recent denoised level and within-window variability.
    summary = np.concatenate([window_summary(X), X[:, -5:].mean(1), X.std(1)], axis=1)
    if representation == "summary":
        return summary
    if representation == "pooled":
        if X.shape[1] % 5:
            raise ValueError("pooled representation requires window length divisible by 5")
        blocks = X.reshape(len(X), X.shape[1] // 5, 5, X.shape[2]).mean(2)
        return np.concatenate([summary, blocks.reshape(len(X), -1)], axis=1)
    raise ValueError(f"Unknown representation: {representation}")


def fit_standardiser(train, representation):
    """Caller must pass training examples only."""
    F = features(train, representation)
    axes = (0, 1) if F.ndim == 3 else 0
    mean = F.mean(axis=axes, dtype=np.float64).astype(np.float32)
    std = F.std(axis=axes, dtype=np.float64).astype(np.float32)
    return {"mean": mean, "std": np.maximum(std, 1e-6)}


def transform(X, representation, scaler):
    return np.ascontiguousarray((features(X, representation) - scaler["mean"]) / scaler["std"], dtype=np.float32)


class FeatureMLP(nn.Module):
    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        layers = []
        for width in hidden:
            layers.extend([nn.Linear(n_in, width), nn.SiLU(), nn.Dropout(dropout)])
            n_in = width
        layers.extend([nn.Linear(n_in, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TemporalCNN(nn.Module):
    def __init__(self, window, sensors, channels, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(sensors, channels, 5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.Tanh(),
            nn.AvgPool1d(2), nn.Flatten(),
            nn.Linear(channels * (window // 2), hidden), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)


def build(cfg, input_shape):
    if cfg["architecture"] == "feature_mlp":
        return FeatureMLP(input_shape[0], cfg["hidden"], cfg["dropout"])
    if cfg["architecture"] == "temporal_cnn":
        return TemporalCNN(*input_shape, cfg["channels"], cfg["hidden"], cfg["dropout"])
    raise ValueError(cfg["architecture"])


@torch.no_grad()
def predict_scaled(model, X, batch_size=1024):
    model.eval()
    return np.concatenate([model(torch.from_numpy(X[i:i + batch_size])).numpy()
                           for i in range(0, len(X), batch_size)])


def train_one(cfg, settings, Xtr, ytr, Xva, yva, seed, cap):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build(cfg, Xtr.shape[1:])
    opt = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=settings["scheduler_factor"], patience=settings["scheduler_patience"], min_lr=settings["min_lr"])
    X, y = torch.from_numpy(Xtr), torch.from_numpy((ytr / cap).astype(np.float32))
    V, vy = torch.from_numpy(Xva), torch.from_numpy((yva / cap).astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, state, waited = float("inf"), 0, None, 0
    history = {"train_mse": [], "val_mse": [], "lr": []}
    start = time.perf_counter()
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(X), generator=generator)
        total = 0.0
        for idx in order.split(settings["batch_size"]):
            opt.zero_grad(set_to_none=True)
            loss = (model(X[idx]) - y[idx]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {cfg['name']}, seed {seed}, epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = sum((model(vx) - vt).square().sum().item()
                     for vx, vt in zip(V.split(1024), vy.split(1024))) / len(V)
        history["train_mse"].append(total / len(X))
        history["val_mse"].append(vl)
        history["lr"].append(opt.param_groups[0]["lr"])
        if vl < best:
            best, best_epoch, state, waited = vl, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
        scheduler.step(vl)
        if epoch % 25 == 0:
            print(f"  {cfg['name']} seed {seed}: epoch {epoch}, best validation RMSE {cap * best**0.5:.3f}", flush=True)
        if waited >= settings["patience"]:
            break
    model.load_state_dict(state)
    prediction = predict_scaled(model, Xva) * cap
    record = {"candidate": cfg["name"], "config": cfg, "seed": seed, "best_epoch": best_epoch,
              "epochs_completed": epoch, "training_seconds": time.perf_counter() - start,
              "val_rmse": float(np.sqrt(np.mean((prediction - yva)**2))),
              "val_mae": float(np.mean(np.abs(prediction - yva))),
              "parameters": sum(p.numel() for p in model.parameters()), "history": history}
    return model, record


class UpgradeRun:
    """Self-contained checkpoint: sensor scaler, feature scaler, model config and weights."""
    def __init__(self, path):
        self.path = Path(path)
        blob = torch.load(self.path, weights_only=False, map_location="cpu")
        self.config, self.seed = blob["config"], blob["seed"]
        self.scaler, self.sensor_scaler = blob["standardiser"], blob["sensor_scaler"]
        self.window, self.cap = blob["window"], blob["cap"]
        self.model = build(self.config, tuple(blob["input_shape"]))
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()

    def predict(self, sensor_scaled_windows):
        X = transform(sensor_scaled_windows, self.config["representation"], self.scaler)
        return predict_scaled(self.model, X) * self.cap
