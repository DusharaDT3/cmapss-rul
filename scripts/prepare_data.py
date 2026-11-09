"""Create the engine-level split and fit preprocessing on training engines only.

Usage: python scripts/prepare_data.py configs/data_fd001.json
Outputs: split indices (data/splits/) and preprocessing settings (models/).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data  # noqa: E402


def main(config_path):
    cfg = data.load_config(config_path)
    train_full = data.load_raw(cfg["subset"], "train")
    train_units, val_units = data.split_units(train_full, cfg["val_fraction"], cfg["split_seed"])
    assert not set(train_units) & set(val_units), "engine appears in both partitions"

    train_only = train_full[train_full["unit"].isin(train_units)]
    features = data.select_features(train_only, cfg["min_unique"])
    scaler = data.fit_scaler(train_only, features)

    split = {
        "subset": cfg["subset"],
        "split_seed": cfg["split_seed"],
        "train_units": train_units,
        "val_units": val_units,
        "test": f"official test_{cfg['subset']}.txt (all units)",
    }
    scaler.update({"fitted_on": "train_units only", "rul_cap": cfg["rul_cap"], "window": cfg["window"]})

    for rel, obj in ((cfg["split_file"], split), (cfg["preprocess_file"], scaler)):
        out = data.ROOT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(obj, indent=2))
        print("Wrote", out.relative_to(data.ROOT))

    prepared = data.load_prepared(cfg)
    for name in ("train", "val"):
        X, y, u = prepared[name]
        print(f"{name:5s} X={X.shape} y={y.shape} engines={len(set(u))}")
    X_test, test_units = data.load_test_inputs(cfg)  # inputs only; test labels are not read here
    print(f"test  X={X_test.shape} engines={len(test_units)} (labels not loaded)")
    print("Features:", features)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/data_fd001.json")
