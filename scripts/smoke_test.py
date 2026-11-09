"""Quick end-to-end check (under a minute on CPU): data pipeline, both architectures, NumPy MLP, inference CLI.

Usage: python scripts/smoke_test.py
Trains tiny throwaway models on a slice of the data; nothing in models/ or results/ is overwritten.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data  # noqa: E402
from src.numpy_mlp import NumpyMLP, gradient_check, window_summary  # noqa: E402
from src.train_torch import fit  # noqa: E402


def main():
    cfg = data.load_config(data.ROOT / "configs" / "data_fd001.json")
    p = data.load_prepared(cfg)
    (X_tr, y_tr, u_tr), (X_va, y_va, u_va) = p["train"], p["val"]
    X_te, _ = data.load_test_inputs(cfg)
    assert X_tr.shape[1:] == (30, 14) and X_te.shape == (100, 30, 14)
    assert not set(u_tr) & set(u_va), "engine leakage between train and validation"
    print("data pipeline ok", X_tr.shape, X_va.shape, X_te.shape)

    for arch in ("mlp", "lstm"):
        mcfg = data.load_config(data.ROOT / "configs" / f"{arch}.json") | {"max_epochs": 2}
        _, hist, _ = fit(mcfg, X_tr[:2000], y_tr[:2000] / 125, X_va[:500], y_va[:500] / 125, seed=0)
        assert np.isfinite(hist["val_loss"]).all()
        print(f"{arch} trains ok, val loss {hist['val_loss'][-1]:.4f}")

    S = window_summary(X_tr[:8].astype(np.float64))
    S = (S - S.mean(0)) / (S.std(0) + 1e-8)
    rows = gradient_check(NumpyMLP(28, 8, seed=0), S, y_tr[:8] / 125, [("W1", (0, 0)), ("b1", (1,)), ("W2", (2, 0)), ("b2", (0,))])
    print("numpy gradient check max rel error", f"{max(r['rel_error'] for r in rows):.1e}")

    if (data.ROOT / "models" / "lstm_seed0.pt").exists():
        cli = [sys.executable, str(data.ROOT / "scripts" / "predict.py")]
        ok = subprocess.run(cli + [str(data.ROOT / "examples" / "engine_normal.csv")], capture_output=True, text=True)
        bad = subprocess.run(cli + [str(data.ROOT / "examples" / "engine_too_short.csv")], capture_output=True, text=True)
        assert ok.returncode == 0 and bad.returncode == 2, (ok.stderr, bad.stderr)
        for name in ("engine_empty.csv", "engine_negative_cycles.csv", "engine_infinite_unit.csv"):
            r = subprocess.run(cli + [str(data.ROOT / "examples" / name)], capture_output=True, text=True)
            assert r.returncode == 2, (name, r.stderr)
        print("inference CLI ok:", ok.stdout.splitlines()[1])
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
