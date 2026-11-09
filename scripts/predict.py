"""Predict remaining useful life for ONE engine from its cycle history (T5b).

Usage:
  python scripts/predict.py examples/engine_normal.csv
  python scripts/predict.py examples/engine_too_short.csv --model mlp

Input: a CSV with columns unit, cycle, setting1-3, s1-s21 (as in C-MAPSS), or a raw whitespace-separated
C-MAPSS text file, for a single engine. The saved preprocessing and all three seed checkpoints of the chosen
model are loaded; the reported value is their mean, with the seed spread as a rough stability indicator.

Exit codes: 0 prediction made, 2 invalid input, 3 input outside the model's scope.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data  # noqa: E402
from src.inference import load_runs  # noqa: E402

SCALED_LIMITS = (-0.25, 1.25)  # allowed range after min-max scaling with training-engine statistics


class InputError(Exception):
    code = 2


class OutOfScope(Exception):
    code = 3


def read_engine(path):
    path = Path(path)
    if not path.exists():
        raise InputError(f"file not found: {path}")
    text = path.read_text(errors="replace")
    if not text.strip():
        raise InputError(f"file is empty: {path}")
    try:
        if "," in text.lstrip()[:200]:
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, sep=r"\s+", header=None)
            if df.shape[1] != len(data.COLUMNS):
                raise InputError(f"raw C-MAPSS files need {len(data.COLUMNS)} columns, found {df.shape[1]}")
            df.columns = data.COLUMNS
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise InputError(f"could not parse {path.name}: {e}") from None
    if df.empty:
        raise InputError(f"file has a header but no data rows: {path}")
    return df


def validate(df, features, window):
    missing = [c for c in ["unit", "cycle"] + features if c not in df.columns]
    if missing:
        raise InputError(f"missing required columns: {missing}")
    numeric = df[["unit", "cycle"] + features].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise InputError(f"missing or non-numeric values in: {bad}")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    if not finite.all():
        bad = numeric.columns[~finite.all(axis=0)].tolist()
        raise InputError(f"non-finite values (inf) in: {bad}")
    if (numeric["cycle"] < 1).any() or (numeric["cycle"] % 1 != 0).any():
        raise InputError("cycle numbers must be positive whole numbers (C-MAPSS cycles start at 1)")
    if (numeric["unit"] < 1).any() or (numeric["unit"] % 1 != 0).any():
        raise InputError("unit (engine) numbers must be positive whole numbers")
    if numeric["unit"].nunique() != 1:
        raise InputError("file contains more than one engine; provide one engine per file")
    numeric = numeric.sort_values("cycle")
    if not (numeric["cycle"].diff().dropna() == 1).all():
        raise InputError("cycles must be consecutive (no gaps or duplicates)")
    if len(numeric) < window:
        raise InputError(f"need at least {window} consecutive cycles of history, got {len(numeric)}")
    return numeric


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--model", choices=["lstm", "mlp", "random_forest", "neural_upgrade"], default="random_forest")
    args = ap.parse_args()

    cfg = data.load_config(data.ROOT / "configs" / "data_fd001.json")
    scaler = data.load_config(data.ROOT / cfg["preprocess_file"])
    features, window, cap = scaler["features"], cfg["window"], cfg["rul_cap"]
    runs = None
    if args.model == "neural_upgrade":
        runs = load_runs(args.model)
        scaler, window, cap = runs[0].sensor_scaler, runs[0].window, runs[0].cap
        if any(r.sensor_scaler != scaler or r.window != window or r.cap != cap for r in runs[1:]):
            raise ValueError("Selected upgrade checkpoints use inconsistent preprocessing")
        features = scaler["features"]
    try:
        engine = validate(read_engine(args.file), features, window)
        scaled = data.apply_scaler(engine, scaler)[features].to_numpy(np.float32)[-window:]
        lo, hi = SCALED_LIMITS
        outside = [f for f, col in zip(features, scaled.T) if col.min() < lo or col.max() > hi]
        if outside:
            raise OutOfScope(f"sensor values far outside the FD001 training range ({outside}); this model was trained "
                             "on one operating condition and one fault mode and cannot be trusted for this engine")
    except (InputError, OutOfScope) as e:
        print(f"REJECTED ({'invalid input' if e.code == 2 else 'out of scope'}): {e}", file=sys.stderr)
        sys.exit(e.code)

    preds = np.array([run.predict(scaled[None])[0] for run in (runs or load_runs(args.model))])
    rul, spread = float(preds.mean()), float(preds.max() - preds.min())
    if rul < 30:
        advice = "maintenance recommended now (predicted to fail within ~30 cycles)"
    elif rul < 60:
        advice = "plan maintenance soon"
    else:
        advice = "no immediate action indicated"
    print(f"Engine {int(engine['unit'].iloc[0])}: {len(engine)} cycles observed, last cycle {int(engine['cycle'].iloc[-1])}")
    print(f"Estimated remaining useful life: {rul:.0f} cycles  (seed range {preds.min():.0f}-{preds.max():.0f}, model: {args.model})")
    if args.model == "neural_upgrade":
        print("Model: exploratory neural upgrade selected on validation data after the original benchmark results were known.")
    if rul >= cap - 10:
        print(f"Note: targets were capped at {cap}; values near {cap} mean 'healthy, at least ~{cap} cycles', not an exact count.")
    print(f"Interpretation: {advice}")
    print("Decision support only: trained on simulated NASA C-MAPSS FD001 data, not certified for real engines.")
    if spread > 20:
        print(f"Warning: the three seed models disagree by {spread:.0f} cycles; treat this estimate with extra caution.")


if __name__ == "__main__":
    main()
