"""One descriptive test evaluation of the frozen follow-up model.

The FD001 test set was already inspected during the original coursework. This
comparison is exploratory, not an independent new test. Run once after training.
"""
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data
from src.inference import load_runs
from src.neural_upgrade import UpgradeRun
from src.metrics import all_metrics
from src.train_torch import environment
from scripts.train_neural_upgrade import OUT, digest


def main():
    if (OUT / "test_results.json").exists() or (OUT / "evaluation_manifest.json").exists():
        raise SystemExit("Follow-up test evaluation already started/completed. Read saved results; do not retune on this test set.")
    selection = data.load_config(OUT / "selection.json")
    training = data.load_config(OUT / "training_manifest.json")
    for rel, sha in training["files"].items():
        if digest(data.ROOT / rel) != sha:
            raise SystemExit(f"Training input/source changed: {rel}")
    original = data.load_config(data.ROOT / "results/freeze_manifest.json")
    for rel, sha in original["files"].items():
        if digest(data.ROOT / rel)[:len(sha)] != sha:
            raise SystemExit(f"Original experiment changed: {rel}")
    cfg = data.load_config(data.ROOT / "configs/data_fd001.json")
    seeds = selection["final_seeds"]
    paths = [OUT / "selection.json", *[OUT / f"selected_seed{s}.pt" for s in seeds],
             Path(__file__), data.ROOT / "src/inference.py", data.ROOT / "src/models.py",
             data.ROOT / "src/train_torch.py", data.ROOT / "src/features.py", data.ROOT / "src/metrics.py",
             data.ROOT / "results/freeze_manifest.json"]
    hashes = {**training["files"], **{str(p.relative_to(data.ROOT)): digest(p) for p in paths}}
    # Freeze before accessing test inputs/labels. Archive the actual original outcomes separately.
    (OUT / "evaluation_manifest.json").write_text(json.dumps({
        "frozen_at": datetime.now(timezone.utc).isoformat(), "files": hashes,
        "test_reused": True, "note": "Post-hoc follow-up; original FD001 test scores were already known.",
        "original_frozen_files": original["files"]}, indent=2))
    torch.set_num_threads(training["plan"]["threads"])
    Xva, yva, _ = data.load_prepared(cfg)["val"]
    Xte, units = data.load_test_inputs(cfg)
    raw_y = data.load_test_labels(cfg, units)
    yte = np.minimum(raw_y, cfg["rul_cap"])
    runs = {name: load_runs(name, seeds) for name in ("random_forest", "mlp", "lstm")}
    runs["neural_upgrade"] = [UpgradeRun(OUT / f"selected_seed{s}.pt") for s in seeds]
    rows, summaries = [], {}
    predictions = {"unit": units, "true_rul": raw_y, "capped_rul": yte}
    for name, models in runs.items():
        vp, tp = [], []
        for model in models:
            val, test = model.predict(Xva), model.predict(Xte)
            vp.append(val); tp.append(test)
            m, vm = all_metrics(yte, test), all_metrics(yva, val)
            rows.append({"model": name, "seed": model.seed, "val_rmse": vm["rmse"],
                         "test_rmse": m["rmse"], "test_mae": m["mae"], "test_nasa_score": m["nasa_score"]})
            predictions[f"{name}_seed{model.seed}"] = test
        own = [r for r in rows if r["model"] == name]
        summary = {metric: {"mean": float(np.mean([r[metric] for r in own])),
                            "std": float(np.std([r[metric] for r in own], ddof=1))}
                   for metric in ("val_rmse", "test_rmse", "test_mae", "test_nasa_score")}
        summary["ensemble_val"] = all_metrics(yva, np.mean(vp, axis=0))
        summary["ensemble_test"] = all_metrics(yte, np.mean(tp, axis=0))
        summary["ensemble_uncapped_test"] = all_metrics(raw_y, np.mean(tp, axis=0))
        predictions[f"{name}_ensemble"] = np.mean(tp, axis=0)
        # Include the whole three-checkpoint prediction cost, with all models loaded.
        for _ in range(3):
            for m in models:
                m.predict(Xte[:1])
        times = []
        for _ in range(10):
            start = time.perf_counter()
            np.mean([m.predict(Xte[:1]) for m in models], axis=0)
            times.append((time.perf_counter() - start) * 1000)
        summary["ensemble_inference_ms_batch1"] = float(np.median(times))
        summaries[name] = summary
        print(f"{name}: val {summary['val_rmse']['mean']:.3f}; test {summary['test_rmse']['mean']:.3f}; ensemble test {summary['ensemble_test']['rmse']:.3f}", flush=True)
    result = {"test_reused": True, "limitation": "All test comparisons are descriptive follow-up results on an already-seen benchmark; require new independent data for confirmation.",
              "selected_config": selection["selected_config"], "seeds": seeds, "models": summaries,
              "environment": environment(), "n_test_engines": len(units), "timing": "batch 1, 3 warmups, median 10 repeats, entire three-model ensemble, CPU"}
    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "test_predictions.csv", index=False)
    (OUT / "test_results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
