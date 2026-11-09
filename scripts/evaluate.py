"""Final evaluation on the held-out test engines (T4a, T4b, T4d, T5a), following configs/evaluation_protocol.json.

Usage: python scripts/evaluate.py [--verify-only] [--refreeze]
Run once, after scripts/run_final.py. Records a freeze manifest (hashes of the final configs and
checkpoints) before the test set is read. If a manifest already exists it is never overwritten: the
current files are checked against it and the script stops on any mismatch (pass --refreeze only if a
deliberate, documented re-freeze is intended). --verify-only checks the manifest and exits without
reading the test set. Outputs:
  results/results_table.csv            one row per model and seed: config, seed, validation and test scores
  results/metrics/test_metrics.json    per-seed and mean +- std metrics, stress tests, timing
  results/test_predictions.csv         per-engine predictions for every run
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data  # noqa: E402
from src.inference import load_runs  # noqa: E402
from src.metrics import all_metrics, rmse  # noqa: E402

MODELS = ["random_forest", "mlp", "lstm"]
SEEDS = [0, 1, 2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


MANIFEST = data.ROOT / "results" / "freeze_manifest.json"


def frozen_files():
    files = [data.ROOT / "configs" / f"final_{m}.json" for m in MODELS]
    files += [data.ROOT / "models" / f"{m}_seed{s}{'.pkl' if m == 'random_forest' else '.pt'}" for m in MODELS for s in SEEDS]
    return {str(f.relative_to(data.ROOT)): sha256(f) for f in files}


def freeze_or_verify(refreeze=False):
    """Write the manifest on the first run; afterwards only verify it, so it cannot be silently replaced."""
    current = frozen_files()
    if MANIFEST.exists() and not refreeze:
        recorded = json.loads(MANIFEST.read_text())
        changed = [f for f in current if recorded["files"].get(f) != current[f]]
        if changed:
            sys.exit(f"Frozen files changed since {recorded['frozen_at']}: {changed}. "
                     "Refusing to evaluate; restore them or re-freeze deliberately with --refreeze.")
        print(f"Freeze manifest verified ({len(current)} files unchanged since {recorded['frozen_at']})")
        return
    if MANIFEST.exists():
        MANIFEST.rename(MANIFEST.with_name(f"freeze_manifest_superseded_{datetime.now():%Y%m%d%H%M%S}.json"))
    manifest = {"frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "note": "Written immediately before the first read of the test set",
                "files": current}
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print("Freeze manifest written")


def time_inference(run, X, batch, repeats=50, warmup=5):
    xb = X[:batch]
    for _ in range(warmup):
        run.predict(xb)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        run.predict(xb)
        times.append(time.perf_counter() - t)
    return float(np.median(times) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true", help="check the freeze manifest and exit")
    ap.add_argument("--refreeze", action="store_true", help="archive the old manifest and write a new one")
    args = ap.parse_args()
    freeze_or_verify(args.refreeze)
    if args.verify_only:
        return
    protocol = data.load_config(data.ROOT / "configs" / "evaluation_protocol.json")
    data_cfg = data.load_config(data.ROOT / "configs" / "data_fd001.json")
    cap = data_cfg["rul_cap"]
    X_va, y_va, _ = data.load_prepared(data_cfg)["val"]
    X_te, units = data.load_test_inputs(data_cfg)
    y_te_raw = data.load_test_labels(data_cfg, units)  # the only place test labels are read
    y_te = np.minimum(y_te_raw, cap)
    observed = data.load_raw("FD001", "test").groupby("unit")["cycle"].max().loc[units].to_numpy()
    short = observed < np.median(observed)

    rows, preds, results = [], {"unit": units, "observed_cycles": observed, "true_rul": y_te_raw}, {}
    rng_noise = np.random.default_rng(123)
    noise_draws = {s: [rng_noise.normal(0, s, X_te.shape).astype(np.float32) for _ in range(5)]
                   for s in (0.01, 0.02, 0.05, 0.10)}

    for name in MODELS:
        runs = load_runs(name, SEEDS)
        per_seed = []
        for run in runs:
            p_va, p_te = run.predict(X_va), run.predict(X_te)
            m = {"seed": run.seed, "val": all_metrics(y_va, p_va), "test": all_metrics(y_te, p_te),
                 "test_rmse_uncapped": rmse(y_te_raw, p_te),
                 "test_rmse_short_history": rmse(y_te[short], p_te[short]),
                 "test_rmse_long_history": rmse(y_te[~short], p_te[~short]),
                 "noise_rmse": {str(s): float(np.mean([rmse(y_te, run.predict(X_te + n)) for n in draws]))
                                for s, draws in noise_draws.items()}}
            per_seed.append(m)
            preds[f"{name}_seed{run.seed}"] = p_te
            rows.append({"model": name, "config": f"configs/final_{name}.json", "seed": run.seed,
                         "checkpoint": str(run.path.relative_to(data.ROOT)),
                         "val_rmse": m["val"]["rmse"], "test_rmse": m["test"]["rmse"], "test_mae": m["test"]["mae"],
                         "test_nasa_score": m["test"]["nasa_score"], "test_rmse_uncapped": m["test_rmse_uncapped"]})

        def agg(get):
            v = np.array([get(m) for m in per_seed])
            return {"mean": float(v.mean()), "std": float(v.std(ddof=1))}

        results[name] = {
            "per_seed": per_seed,
            "val_rmse": agg(lambda m: m["val"]["rmse"]),
            "test_rmse": agg(lambda m: m["test"]["rmse"]),
            "test_mae": agg(lambda m: m["test"]["mae"]),
            "test_nasa_score": agg(lambda m: m["test"]["nasa_score"]),
            "test_rmse_uncapped": agg(lambda m: m["test_rmse_uncapped"]),
            "test_rmse_short_history": agg(lambda m: m["test_rmse_short_history"]),
            "test_rmse_long_history": agg(lambda m: m["test_rmse_long_history"]),
            "noise_rmse": {s: agg(lambda m, s=s: m["noise_rmse"][s]) for s in per_seed[0]["noise_rmse"]},
            "n_params": None if name == "random_forest" else int(sum(p.numel() for p in runs[0].model.parameters())),
            "rf_total_nodes": int(sum(t.tree_.node_count for t in runs[0].model.estimators_)) if name == "random_forest" else None,
            "inference_ms": {"batch_1": time_inference(runs[0], X_te, 1), "batch_100": time_inference(runs[0], X_te, 100)},
        }

    results["subgroup_sizes"] = {"short_history": int(short.sum()), "long_history": int((~short).sum()),
                                 "median_observed_cycles": float(np.median(observed))}
    results["protocol"] = protocol
    pd.DataFrame(rows).to_csv(data.ROOT / "results" / "results_table.csv", index=False)
    pd.DataFrame(preds).to_csv(data.ROOT / "results" / "test_predictions.csv", index=False)
    (data.ROOT / "results" / "metrics" / "test_metrics.json").write_text(json.dumps(results, indent=2))

    for name in MODELS:
        r = results[name]
        print(f"{name:14s} val {r['val_rmse']['mean']:.2f}±{r['val_rmse']['std']:.2f}  "
              f"test RMSE {r['test_rmse']['mean']:.2f}±{r['test_rmse']['std']:.2f}  "
              f"score {r['test_nasa_score']['mean']:.0f}  {r['inference_ms']['batch_1']:.2f} ms/1")


if __name__ == "__main__":
    main()
