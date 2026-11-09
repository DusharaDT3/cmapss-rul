"""Checks for new feature transforms, bounded inference and checkpoint portability."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src import data
from src.neural_upgrade import build, features, fit_standardiser, transform, UpgradeRun


class NeuralUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(8).normal(size=(8, 30, 14)).astype(np.float32)

    def test_window_statistics_and_order(self):
        X = np.broadcast_to(np.arange(30, dtype=np.float32)[None, :, None], (2, 30, 14)).copy()
        F = features(X, "summary")
        np.testing.assert_allclose(F[:, :14], 14.5)
        np.testing.assert_allclose(F[:, 14:28], 1.0)
        np.testing.assert_allclose(F[:, 28:42], 27.0)
        P = features(X, "pooled")[:, 56:].reshape(2, 6, 14)
        np.testing.assert_allclose(P[0, :, 0], [2, 7, 12, 17, 22, 27])

    def test_train_only_loader_and_scaling(self):
        cfg = data.load_config(data.ROOT / "configs/data_fd001.json")
        original = data.load_raw
        def load(subset, part):
            self.assertEqual(part, "train")
            return original(subset, part)
        with patch.object(data, "load_raw", side_effect=load), patch.object(data, "load_test_rul", side_effect=AssertionError("test access")):
            arrays = data.load_prepared(cfg)
        self.assertFalse(set(arrays["train"][2]) & set(arrays["val"][2]))
        scaler = fit_standardiser(self.X, "summary")
        before = {k: v.copy() for k, v in scaler.items()}
        transform(self.X * 10000, "summary", scaler)
        for k in before:
            np.testing.assert_array_equal(before[k], scaler[k])

    def test_checkpoint_round_trip(self):
        candidates = data.load_config(data.ROOT / "configs/neural_upgrade_plan.json")["candidates"]
        for cfg in (candidates[0], candidates[2], candidates[5]):
            scaler = fit_standardiser(self.X, cfg["representation"])
            X = transform(self.X, cfg["representation"], scaler)
            model = build(cfg, X.shape[1:]).eval()
            with torch.no_grad():
                expected = model(torch.from_numpy(X)).numpy() * 125
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "model.pt"
                torch.save({"config": cfg, "seed": 0, "standardiser": scaler, "sensor_scaler": {},
                            "window": 30, "cap": 125, "input_shape": list(X.shape[1:]), "state_dict": model.state_dict()}, p)
                actual = UpgradeRun(p).predict(self.X)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
            self.assertTrue(np.all((actual >= 0) & (actual <= 125)))


if __name__ == "__main__":
    unittest.main()
