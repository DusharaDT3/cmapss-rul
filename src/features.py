"""Hand-crafted window features for the non-neural baseline."""
import numpy as np

from src.numpy_mlp import window_summary


def baseline_features(X):
    """(N, T, F) windows -> (N, 4F): per-sensor mean, slope, last value and standard deviation."""
    return np.concatenate([window_summary(X), X[:, -1, :], X.std(axis=1)], axis=1)
