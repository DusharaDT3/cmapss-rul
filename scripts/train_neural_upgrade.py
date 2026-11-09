"""Fixed-budget follow-up search. Training/validation only; preserves the original experiment.

Usage: python scripts/train_neural_upgrade.py
Resume skips only trials whose saved inputs/source/config hashes still match.
"""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data
from src.neural_upgrade import fit_standardiser, transform, train_one
from src.train_torch import environment

OUT = data.ROOT / "experiments" / "neural_upgrade"
PLAN = data.ROOT / "configs" / "neural_upgrade_plan.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = data.load_config(PLAN)
    cfg = data.load_config(data.ROOT / plan["data_config"])
    paths = [PLAN, Path(__file__), data.ROOT / "src/neural_upgrade.py", data.ROOT / "src/data.py",
             data.ROOT / "src/numpy_mlp.py", data.ROOT / plan["data_config"],
             data.ROOT / cfg["split_file"], data.ROOT / cfg["preprocess_file"],
             data.ROOT / "data/raw/train_FD001.txt"]
    hashes = {str(p.relative_to(data.ROOT)): digest(p) for p in paths}
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "training_manifest.json"
    if lock.exists():
        if data.load_config(lock)["files"] != hashes:
            raise SystemExit("Follow-up training inputs/source changed. Use a new experiment; existing results are protected.")
    else:
        lock.write_text(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(), "files": hashes,
                                   "plan": plan, "original_test_already_seen": True}, indent=2))
    if (OUT / "test_results.json").exists():
        raise SystemExit("Test evaluation already exists: this experiment is closed to further tuning.")
    torch.set_num_threads(plan["threads"])
    prepared = data.load_prepared(cfg)
    Xtr, ytr, _ = prepared["train"]
    Xva, yva, _ = prepared["val"]
    sensor_scaler = data.load_config(data.ROOT / cfg["preprocess_file"])
    cache = {}
    for representation in {c["representation"] for c in plan["candidates"]}:
        scaler = fit_standardiser(Xtr, representation)
        cache[representation] = (transform(Xtr, representation, scaler), transform(Xva, representation, scaler), scaler)

    def run(candidate, seed):
        tag = f"{candidate['name']}_seed{seed}"
        dest, metrics = OUT / f"{tag}.pt", OUT / f"{tag}.json"
        if dest.exists() and metrics.exists():
            rec = data.load_config(metrics)
            if rec["checkpoint_sha256"] != digest(dest):
                raise RuntimeError(f"Checkpoint changed: {dest}")
            return rec
        tr, va, scaler = cache[candidate["representation"]]
        model, rec = train_one(candidate, plan["training"], tr, ytr, va, yva, seed, cfg["rul_cap"])
        torch.save({"state_dict": model.state_dict(), "config": candidate, "seed": seed,
                    "standardiser": scaler, "sensor_scaler": sensor_scaler,
                    "window": cfg["window"], "cap": cfg["rul_cap"], "input_shape": list(tr.shape[1:]),
                    "experiment": plan["experiment"]}, dest)
        rec.update(environment=environment(), checkpoint_sha256=digest(dest))
        metrics.write_text(json.dumps(rec, indent=2))
        print(f"DONE {tag}: validation RMSE {rec['val_rmse']:.4f}; {rec['training_seconds']:.1f}s", flush=True)
        return rec

    rows = []
    for candidate in plan["candidates"]:
        records = [run(candidate, seed) for seed in plan["tuning_seeds"]]
        rows.append({"config": candidate, "seed_results": [{k: r[k] for k in ("seed", "val_rmse", "val_mae", "best_epoch", "epochs_completed", "training_seconds", "parameters")} for r in records],
                     "mean_val_rmse": float(np.mean([r["val_rmse"] for r in records]))})
        (OUT / "search_results.json").write_text(json.dumps(rows, indent=2))
    best = min(rows, key=lambda r: r["mean_val_rmse"])
    print(f"SELECTED {best['config']['name']} by mean validation RMSE {best['mean_val_rmse']:.4f}", flush=True)
    final = []
    for seed in plan["final_seeds"]:
        record = run(best["config"], seed)
        shutil.copyfile(OUT / f"{best['config']['name']}_seed{seed}.pt", OUT / f"selected_seed{seed}.pt")
        final.append({k: record[k] for k in ("seed", "val_rmse", "val_mae", "best_epoch", "epochs_completed", "parameters")})
    (OUT / "selection.json").write_text(json.dumps({"selected_config": best["config"], "selection_metric": "mean validation RMSE over tuning seeds 0 and 1", "final_seeds": plan["final_seeds"], "runs": final,
        "val_rmse_mean": float(np.mean([r["val_rmse"] for r in final])),
        "val_rmse_std": float(np.std([r["val_rmse"] for r in final], ddof=1)),
        "original_test_already_seen": True}, indent=2))


if __name__ == "__main__":
    main()
