"""Train the random-forest baseline (T3a) on the same engine split and windows as the neural models.

Usage: python scripts/train_baseline.py configs/baseline_rf.json [--seed 0] [--max-depth 8 --min-samples-leaf 1]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import sklearn
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data  # noqa: E402
from src.features import baseline_features  # noqa: E402
from src.metrics import all_metrics  # noqa: E402
from src.train_torch import environment  # noqa: E402


def run(cfg, seed, tag=None):
    data_cfg = data.load_config(data.ROOT / cfg["data_config"])
    prepared = data.load_prepared(data_cfg)
    (X_tr, y_tr, _), (X_va, y_va, _) = prepared["train"], prepared["val"]
    model = RandomForestRegressor(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                  min_samples_leaf=cfg["min_samples_leaf"], n_jobs=-1, random_state=seed)
    model.fit(baseline_features(X_tr), y_tr)
    val = all_metrics(y_va, model.predict(baseline_features(X_va)))

    name = tag or f"{cfg['name']}_seed{seed}"
    with open(data.ROOT / "models" / f"{name}.pkl", "wb") as f:
        pickle.dump({"model": model, "config": cfg, "seed": seed}, f)
    record = {"model": cfg["name"], "run": name, "seed": seed, "config": cfg, "val": val,
              "environment": {**environment(), "scikit_learn": sklearn.__version__}}
    (data.ROOT / "results" / "metrics" / f"{name}.json").write_text(json.dumps(record, indent=2))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-depth", type=lambda v: None if v == "None" else int(v))
    ap.add_argument("--min-samples-leaf", type=int)
    ap.add_argument("--tag")
    args = ap.parse_args()
    cfg = data.load_config(args.config)
    if args.max_depth is not None or args.min_samples_leaf is not None:
        cfg.update(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf or cfg["min_samples_leaf"])
    rec = run(cfg, args.seed, args.tag)
    print(f"{rec['run']}: val RMSE {rec['val']['rmse']:.2f}  MAE {rec['val']['mae']:.2f}")


if __name__ == "__main__":
    main()
