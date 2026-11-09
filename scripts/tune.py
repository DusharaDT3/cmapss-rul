"""Run the pre-registered tuning plan (configs/tuning_plan.json) and freeze the selected configurations.

Usage: python scripts/tune.py
Writes results/metrics/tuning.json (every trial) and configs/final_<model>.json (selected by validation RMSE).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_baseline, train_torch_model  # noqa: E402
from src import data  # noqa: E402

BASE = {"mlp": "configs/mlp.json", "lstm": "configs/lstm.json", "random_forest": "configs/baseline_rf.json"}


def main():
    plan = data.load_config(data.ROOT / "configs" / "tuning_plan.json")
    trials = []
    for model_name, candidates in plan["candidates"].items():
        base = data.load_config(data.ROOT / BASE[model_name])
        for i, overrides in enumerate(candidates):
            cfg = {**base, **overrides}
            tag = f"tune_{model_name}_{i}"
            runner = train_baseline if model_name == "random_forest" else train_torch_model
            rec = runner.run(cfg, seed=0, tag=tag)
            trials.append({"model": model_name, "trial": i, "overrides": overrides, "val_rmse": rec["val"]["rmse"],
                           "epochs_completed": rec.get("epochs_completed"), "n_params": rec.get("n_params")})
            print(f"{tag:22s} {json.dumps(overrides):45s} val RMSE {rec['val']['rmse']:.2f}", flush=True)

    selected = {}
    for model_name in plan["candidates"]:
        best = min((t for t in trials if t["model"] == model_name), key=lambda t: t["val_rmse"])
        cfg = {**data.load_config(data.ROOT / BASE[model_name]), **best["overrides"], "name": model_name,
               "selected_by": f"tuning trial {best['trial']} (lowest validation RMSE, seed 0)"}
        (data.ROOT / "configs" / f"final_{model_name}.json").write_text(json.dumps(cfg, indent=2))
        selected[model_name] = best
    (data.ROOT / "results" / "metrics" / "tuning.json").write_text(
        json.dumps({"plan": plan, "trials": trials, "selected": selected}, indent=2))
    print("Selected:", {k: v["overrides"] for k, v in selected.items()})


if __name__ == "__main__":
    main()
