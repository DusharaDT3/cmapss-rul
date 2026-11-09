"""Neural models using strictly causal observed engine histories (follow-up v2)."""
import copy
import time

import numpy as np
import torch
from torch import nn

from src import data
from src.neural_upgrade import features as window_features, predict_scaled
from src.train_torch import set_seed


def engine_features(values, cycles, end_indices, representation="history", window=30):
    """Rows at endpoints use ONLY their available prefix, including the initial ten cycles.

    values are already min-max scaled with saved training-engine statistics.
    end_indices are exclusive (e.g. 30 is the endpoint at cycle 30).
    """
    values = np.asarray(values, dtype=np.float64)
    cycles = np.asarray(cycles, dtype=np.float64)
    ends = np.asarray(end_indices, dtype=int)
    if len(values) < window or np.any(ends < window) or np.any(ends > len(values)):
        raise ValueError("Every endpoint needs at least 30 observed cycles")
    windows = np.stack([values[e-window:e] for e in ends]).astype(np.float32)
    local = window_features(windows, "pooled" if representation == "pooled_history" else "summary")
    initial = values[:10].mean(0)
    cumulative = np.cumsum(values, axis=0)
    # OLS against the actual observed cycle counter, using prefix sums only.
    cx = np.cumsum(cycles)
    cx2 = np.cumsum(cycles**2)
    cxy = np.cumsum(cycles[:, None] * values, axis=0)
    k = ends - 1
    denom = ends * cx2[k] - cx[k]**2
    slopes = (ends[:, None] * cxy[k] - cx[k, None] * cumulative[k]) / denom[:, None]
    means = cumulative[k] / ends[:, None]
    change = windows[:, -5:].mean(1) - initial
    initial_rows = np.broadcast_to(initial, (len(ends), len(initial)))
    return np.ascontiguousarray(np.concatenate([local, initial_rows, change, means, slopes,
                                                cycles[k, None] / 125.0], axis=1), dtype=np.float32)


def frame_features(frame, scaler, representation, window=30, last_only=False):
    frame = data.apply_scaler(frame, scaler)
    arrays, targets, units, endpoints = [], [], [], []
    for unit, g in frame.groupby("unit"):
        g = g.sort_values("cycle")
        ends = np.array([len(g)]) if last_only else np.arange(window, len(g) + 1)
        arrays.append(engine_features(g[scaler["features"]].to_numpy(), g.cycle.to_numpy(), ends, representation, window))
        if "rul" in g:
            targets.extend(g.rul.to_numpy()[ends - 1])
        units.extend([int(unit)] * len(ends))
        endpoints.extend(g.cycle.to_numpy()[ends - 1])
    return np.concatenate(arrays), np.asarray(targets, dtype=np.float32), np.asarray(units), np.asarray(endpoints)


def training_arrays(cfg, representation):
    split = data.load_config(data.ROOT / cfg["split_file"])
    sensor_scaler = data.load_config(data.ROOT / cfg["preprocess_file"])
    raw = data.add_train_rul(data.load_raw(cfg["subset"], "train"), cfg["rul_cap"])
    tr = frame_features(raw[raw.unit.isin(split["train_units"])], sensor_scaler, representation, cfg["window"])
    va = frame_features(raw[raw.unit.isin(split["val_units"])], sensor_scaler, representation, cfg["window"])
    return tr, va, sensor_scaler


def fit_scaler(X):
    return {"mean": X.mean(0, dtype=np.float64).astype(np.float32),
            "std": np.maximum(X.std(0, dtype=np.float64), 1e-6).astype(np.float32)}


def transform(X, scaler):
    return np.ascontiguousarray((X - scaler["mean"]) / scaler["std"], dtype=np.float32)


class HistoryMLP(nn.Module):
    def __init__(self, n_in, cfg):
        super().__init__()
        layers = []
        for n in cfg["hidden"]:
            layers += [nn.Linear(n_in, n), nn.SiLU(), nn.Dropout(cfg["dropout"])]
            n_in = n
        layers += [nn.Linear(n_in, 1)]
        if cfg["output"] == "sigmoid":
            layers += [nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(cfg, settings, Xtr, ytr, Xva, yva, seed, cap):
    set_seed(seed)
    m = HistoryMLP(Xtr.shape[1], cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=settings["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=settings["scheduler_patience"], min_lr=1e-5)
    X, y = torch.from_numpy(Xtr), torch.from_numpy((ytr / cap).astype(np.float32))
    V, target = torch.from_numpy(Xva), torch.from_numpy((yva / cap).astype(np.float32))
    gen = torch.Generator().manual_seed(seed)
    history = {"train_mse": [], "val_mse": [], "lr": []}
    best, epoch_best, state, waited = float("inf"), 0, None, 0
    start = time.perf_counter()
    for epoch in range(1, settings["max_epochs"] + 1):
        m.train(); total = 0.0
        for idx in torch.randperm(len(X), generator=gen).split(settings["batch_size"]):
            opt.zero_grad(set_to_none=True)
            loss = (m(X[idx]) - y[idx]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {cfg['name']}, seed {seed}")
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            total += loss.item() * len(idx)
        m.eval()
        with torch.no_grad():
            val = sum((m(b).clamp(0, 1) - t).square().sum().item()
                      for b, t in zip(V.split(1024), target.split(1024))) / len(V)
        history["train_mse"].append(total / len(X)); history["val_mse"].append(val)
        history["lr"].append(opt.param_groups[0]["lr"])
        if val < best:
            best, epoch_best, state, waited = val, epoch, copy.deepcopy(m.state_dict()), 0
        else:
            waited += 1
        scheduler.step(val)
        if waited >= settings["patience"]:
            break
    m.load_state_dict(state)
    p = predict_scaled(m, Xva).clip(0, 1) * cap
    return m, {"config": cfg, "seed": seed, "epochs_completed": epoch, "best_epoch": epoch_best,
               "val_rmse": float(np.sqrt(np.mean((p-yva)**2))), "val_mae": float(np.mean(abs(p-yva))),
               "parameters": sum(p.numel() for p in m.parameters()), "history": history,
               "training_seconds": time.perf_counter()-start}


class HistoryRun:
    def __init__(self, path):
        blob = torch.load(path, weights_only=False, map_location="cpu")
        self.kind, self.config, self.seed = blob["kind"], blob["config"], blob["seed"]
        self.scaler, self.sensor_scaler = blob["scaler"], blob["sensor_scaler"]
        self.cap, self.window = blob["cap"], blob["window"]
        if self.kind == "neural":
            self.model = HistoryMLP(blob["input_dim"], self.config)
            self.model.load_state_dict(blob["state_dict"]); self.model.eval()
        else:
            self.model = blob["model"]

    def predict_features(self, X):
        X = transform(X, self.scaler)
        if self.kind == "neural":
            return predict_scaled(self.model, X).clip(0, 1) * self.cap
        return self.model.predict(X).clip(0, self.cap)

    def predict_frame(self, frame, last_only=True):
        X, _, units, cycles = frame_features(frame, self.sensor_scaler, self.config["representation"], self.window, last_only)
        return self.predict_features(X), units, cycles
