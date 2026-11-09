"""Load saved checkpoints and preprocessing, and predict RUL in cycles."""
import pickle

import numpy as np
import torch

from src import data
from src.features import baseline_features
from src.models import build_model
from src.train_torch import predict as torch_predict


class SavedModel:
    """Wraps a saved random forest (.pkl) or PyTorch (.pt) checkpoint; predict() returns cycles, clipped at 0."""

    def __init__(self, path):
        self.path = path
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                blob = pickle.load(f)
            self.kind, self.model, self.scale = "rf", blob["model"], 1.0
        else:
            blob = torch.load(path, weights_only=False)
            self.kind, self.scale = "torch", blob["target_scale"]
            self.model = build_model(blob["config"], blob["window"], blob["n_features"])
            self.model.load_state_dict(blob["state_dict"])
            self.model.eval()
        self.config, self.seed = blob["config"], blob["seed"]

    def predict(self, X):
        if self.kind == "rf":
            out = self.model.predict(baseline_features(X))
        else:
            out = torch_predict(self.model, X.astype(np.float32)) * self.scale
        return np.clip(out, 0.0, None)


def load_runs(model_name, seeds=(0, 1, 2)):
    if model_name == "neural_upgrade":
        from src.neural_upgrade import UpgradeRun
        return [UpgradeRun(data.ROOT / "experiments" / "neural_upgrade" / f"selected_seed{s}.pt") for s in seeds]
    ext = ".pkl" if model_name == "random_forest" else ".pt"
    return [SavedModel(data.ROOT / "models" / f"{model_name}_seed{s}{ext}") for s in seeds]
