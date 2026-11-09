"""Evaluation metrics, in cycles."""
import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def nasa_score(y_true, y_pred):
    """PHM08 asymmetric score (Saxena et al., 2008); lower is better.

    d = predicted - true. Late predictions (d > 0, over-estimating life) are
    penalised more heavily (exp(d/10)) than early ones (exp(-d/13)).
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def all_metrics(y_true, y_pred):
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "nasa_score": nasa_score(y_true, y_pred)}
