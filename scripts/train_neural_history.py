"""Train the predeclared v2 follow-up, without reading any test inputs or labels."""
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data
from src.neural_history import training_arrays, fit_scaler, transform, train_model
from src.train_torch import environment

OUT = data.ROOT / "experiments/neural_history_v2"
PLAN = data.ROOT / "configs/neural_history_plan.json"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    plan = data.load_config(PLAN); cfg = data.load_config(data.ROOT / plan["data_config"])
    files = [PLAN, Path(__file__), *[data.ROOT / f"src/{name}.py" for name in ("neural_history", "neural_upgrade", "numpy_mlp", "train_torch", "models", "data")],
             data.ROOT / plan["data_config"], data.ROOT / cfg["split_file"], data.ROOT / cfg["preprocess_file"], data.RAW_DIR / "train_FD001.txt"]
    hashes = {str(p.relative_to(data.ROOT)): digest(p) for p in files}
    OUT.mkdir(parents=True, exist_ok=True); lock = OUT / "training_manifest.json"
    if lock.exists():
        if data.load_config(lock)["files"] != hashes:
            raise SystemExit("Source/data changed since training started; create a new experiment rather than overwriting.")
    else:
        lock.write_text(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(), "plan": plan, "files": hashes, "test_already_seen": True}, indent=2))
    if (OUT / "evaluation_manifest.json").exists():
        raise SystemExit("Experiment closed to tuning after evaluation.")
    torch.set_num_threads(plan["threads"])
    cache = {}
    for rep in ("history", "pooled_history"):
        tr, va, sensor_scaler = training_arrays(cfg, rep)
        scaler = fit_scaler(tr[0]); cache[rep] = (transform(tr[0], scaler), tr[1], transform(va[0], scaler), va[1], scaler)

    def run(c, seed, kind):
        name = f"{c['name']}_seed{seed}"; ck = OUT / f"{name}.pt"; metric = OUT / f"{name}.json"
        if ck.exists() and metric.exists():
            r=data.load_config(metric)
            if digest(ck)!=r["checkpoint_sha256"]:raise RuntimeError(f"Changed checkpoint {ck}")
            return r
        X, y, V, vy, scaler = cache[c["representation"]]
        if kind == "neural":
            m, r = train_model(c, plan["training"], X, y, V, vy, seed, cfg["rul_cap"])
            weights = {"state_dict": m.state_dict()}
        else:
            start=time.perf_counter()
            m=RandomForestRegressor(n_estimators=c["n_estimators"], min_samples_leaf=c["min_samples_leaf"], max_depth=c["max_depth"], random_state=seed, n_jobs=2)
            m.fit(X,y); p=m.predict(V).clip(0,cfg["rul_cap"])
            r={"config":c,"seed":seed,"val_rmse":float(np.sqrt(np.mean((p-vy)**2))),"val_mae":float(np.mean(abs(p-vy))),"training_seconds":time.perf_counter()-start}
            weights={"model":m}
        torch.save({**weights,"kind":kind,"config":c,"seed":seed,"scaler":scaler,"sensor_scaler":sensor_scaler,"input_dim":X.shape[1],"window":cfg["window"],"cap":cfg["rul_cap"]},ck)
        r.update(environment=environment(),checkpoint_sha256=digest(ck));metric.write_text(json.dumps(r,indent=2))
        print(f"DONE {name}: validation RMSE {r['val_rmse']:.4f} ({r['training_seconds']:.1f}s)",flush=True)
        return r

    search=[]; selected={}
    for kind,key in (("neural","neural_candidates"),("random_forest","rf_candidates")):
        for c in plan[key]:
            runs=[run(c,s,kind) for s in plan["tuning_seeds"]]
            row={"kind":kind,"config":c,"runs":[{k:r[k] for k in ("seed","val_rmse","val_mae","training_seconds")} for r in runs],"mean_val_rmse":float(np.mean([r["val_rmse"] for r in runs]))}
            search.append(row);(OUT/'search_results.json').write_text(json.dumps(search,indent=2))
        best=min([r for r in search if r["kind"]==kind],key=lambda r:r["mean_val_rmse"])
        final=[]
        for seed in plan["final_seeds"]:
            r=run(best["config"],seed,kind);final.append({k:r[k] for k in ("seed","val_rmse","val_mae")})
            shutil.copyfile(OUT/f"{best['config']['name']}_seed{seed}.pt",OUT/f"selected_{kind}_seed{seed}.pt")
        selected[kind]={"config":best["config"],"runs":final,"val_rmse_mean":float(np.mean([r["val_rmse"] for r in final])),"val_rmse_std":float(np.std([r["val_rmse"] for r in final],ddof=1))}
        print('SELECTED',kind,selected[kind],flush=True)
    (OUT/'selection.json').write_text(json.dumps({"models":selected,"seeds":plan["final_seeds"],"test_already_seen":True},indent=2))


if __name__ == '__main__':main()
