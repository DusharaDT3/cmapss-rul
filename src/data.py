"""Data loading, labelling, leakage-safe splitting and windowing for C-MAPSS.

Every function that learns something from data (feature selection, scaling)
receives the TRAINING frame only. Validation and test frames are transformed
with the fitted settings, never used to fit them.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

SETTINGS = ["setting1", "setting2", "setting3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTINGS + SENSORS


def load_config(path):
    with open(path) as f:
        return json.load(f)


def load_raw(subset, part):
    """Load train_/test_ file for a subset such as 'FD001'. part is 'train' or 'test'."""
    path = RAW_DIR / f"{part}_{subset}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_test_rul(subset):
    """True RUL at the last observed cycle of each test engine (one row per unit, in unit order)."""
    rul = pd.read_csv(RAW_DIR / f"RUL_{subset}.txt", header=None, names=["rul"])
    rul["unit"] = np.arange(1, len(rul) + 1)
    return rul


def add_train_rul(df, cap):
    """Piecewise-linear target: RUL = last cycle - current cycle, clipped at `cap`.

    Valid only for run-to-failure (training) trajectories, where the last
    recorded cycle is the failure point.
    """
    df = df.copy()
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (last - df["cycle"]).clip(upper=cap)
    return df


def split_units(df, val_fraction, seed):
    """Split training ENGINES (not rows) into train/validation groups.

    Windows from one engine overlap and share its wear history, so a row-level
    split would leak near-identical samples into validation.
    """
    units = np.sort(df["unit"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = int(round(len(units) * val_fraction))
    val_units = np.sort(shuffled[:n_val])
    train_units = np.sort(shuffled[n_val:])
    return train_units.tolist(), val_units.tolist()


def select_features(train_df, min_unique):
    """Keep sensors with more than `min_unique` distinct values in the training split.

    Operational settings are excluded: FD001 has a single operating condition,
    so the settings only carry measurement noise.
    """
    return [c for c in SENSORS if train_df[c].nunique() > min_unique]


def fit_scaler(train_df, features):
    """Min-max parameters fitted on training engines only."""
    return {
        "features": features,
        "min": train_df[features].min().tolist(),
        "max": train_df[features].max().tolist(),
    }


def apply_scaler(df, scaler):
    df = df.copy()
    lo = np.array(scaler["min"])
    hi = np.array(scaler["max"])
    df[scaler["features"]] = (df[scaler["features"]].to_numpy() - lo) / (hi - lo)
    return df


def make_windows(df, features, window):
    """All sliding windows per engine for training/validation.

    Returns X of shape (n_windows, window, n_features), y = RUL at each window's
    last cycle, and the unit id of each window (for grouped analysis).
    """
    xs, ys, units = [], [], []
    for unit, g in df.groupby("unit"):
        values = g[features].to_numpy(dtype=np.float32)
        target = g["rul"].to_numpy(dtype=np.float32)
        for end in range(window, len(g) + 1):
            xs.append(values[end - window:end])
            ys.append(target[end - 1])
            units.append(unit)
    return np.stack(xs), np.array(ys), np.array(units)


def last_windows(df, features, window):
    """One window per engine ending at its last observed cycle (test-time input).

    Engines shorter than `window` are front-padded by repeating their first row.
    """
    xs, units = [], []
    for unit, g in df.groupby("unit"):
        values = g[features].to_numpy(dtype=np.float32)
        if len(values) < window:
            pad = np.repeat(values[:1], window - len(values), axis=0)
            values = np.vstack([pad, values])
        xs.append(values[-window:])
        units.append(unit)
    return np.stack(xs), np.array(units)


def load_prepared(config):
    """Rebuild train/validation arrays from saved split indices and preprocessing settings.

    Reads only the training file. Test data has its own loaders below so that
    training, tuning and validation code never opens the test files.
    """
    split = load_config(ROOT / config["split_file"])
    scaler = load_config(ROOT / config["preprocess_file"])
    features = scaler["features"]

    full = add_train_rul(load_raw(config["subset"], "train"), config["rul_cap"])
    full = apply_scaler(full, scaler)
    train = full[full["unit"].isin(split["train_units"])]
    val = full[full["unit"].isin(split["val_units"])]

    X_train, y_train, u_train = make_windows(train, features, config["window"])
    X_val, y_val, u_val = make_windows(val, features, config["window"])
    return {
        "features": features,
        "train": (X_train, y_train, u_train),
        "val": (X_val, y_val, u_val),
    }


def load_test_inputs(config):
    """Last window of every test engine (inputs only, no labels)."""
    scaler = load_config(ROOT / config["preprocess_file"])
    test = apply_scaler(load_raw(config["subset"], "test"), scaler)
    return last_windows(test, scaler["features"], config["window"])


def load_test_labels(config, units):
    """True (uncapped) RUL for the given test units. Called only by the final evaluation."""
    return load_test_rul(config["subset"]).set_index("unit").loc[units, "rul"].to_numpy(np.float32)
