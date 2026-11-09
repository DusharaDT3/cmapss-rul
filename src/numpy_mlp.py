"""One-hidden-layer MLP for RUL regression in plain NumPy (no automatic differentiation).

Batch convention (rows are examples):
    X  (N, D)   input batch
    W1 (D, H)   b1 (H,)     z = X @ W1 + b1      (N, H)  hidden pre-activation
                            h = ReLU(z)          (N, H)  hidden activation
    W2 (H, 1)   b2 (1,)     y_hat = h @ W2 + b2  (N, 1)  linear output
    L = 1/(2N) * sum((y_hat - y)^2)                       scalar loss

W1 here is the transpose of W in the column-vector form z = Wx + b used in CW2 Q1.
"""
import numpy as np


def window_summary(X):
    """Reduce (N, T, F) windows to (N, 2F): per-sensor window mean and least-squares slope."""
    t = np.arange(X.shape[1], dtype=X.dtype)
    t = t - t.mean()
    mean = X.mean(axis=1)
    slope = np.einsum("t,ntf->nf", t, X - mean[:, None, :]) / (t @ t)
    return np.concatenate([mean, slope], axis=1)


class NumpyMLP:
    def __init__(self, n_in, n_hidden, seed=0, dtype=np.float64):
        rng = np.random.default_rng(seed)
        # He initialisation keeps ReLU pre-activation variance roughly constant across layers.
        self.params = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_hidden)).astype(dtype),
            "b1": np.zeros(n_hidden, dtype=dtype),
            "W2": rng.normal(0.0, np.sqrt(2.0 / n_hidden), (n_hidden, 1)).astype(dtype),
            "b2": np.zeros(1, dtype=dtype),
        }

    def forward(self, X):
        p = self.params
        z = X @ p["W1"] + p["b1"]
        h = np.maximum(z, 0.0)
        y_hat = h @ p["W2"] + p["b2"]
        return y_hat, {"X": X, "z": z, "h": h}

    @staticmethod
    def loss(y_hat, y):
        return 0.5 * np.mean((y_hat - y.reshape(-1, 1)) ** 2)

    def backward(self, y_hat, y, cache):
        """Analytical gradients of the loss with respect to every parameter."""
        n = y_hat.shape[0]
        d_out = (y_hat - y.reshape(-1, 1)) / n           # dL/dy_hat        (N, 1)
        d_h = d_out @ self.params["W2"].T                # dL/dh            (N, H)
        d_z = d_h * (cache["z"] > 0)                     # ReLU derivative  (N, H)
        return {
            "W1": cache["X"].T @ d_z,                    # (D, H)
            "b1": d_z.sum(axis=0),                       # (H,)
            "W2": cache["h"].T @ d_out,                  # (H, 1)
            "b2": d_out.sum(axis=0),                     # (1,)
        }

    def step(self, grads, lr):
        for k in self.params:
            self.params[k] -= lr * grads[k]

    def predict(self, X):
        return self.forward(X)[0].ravel()

    def n_params(self):
        return sum(v.size for v in self.params.values())


def train(model, X, y, X_val, y_val, lr, batch_size, max_epochs, patience, seed=0, diverge_factor=1e3):
    """Mini-batch gradient descent with validation early stopping.

    Stopping rule: stop when validation loss has not improved for `patience` epochs
    (best weights are restored), or when training loss becomes non-finite or exceeds
    `diverge_factor` x its initial value (reported as divergence, not hidden).
    """
    rng = np.random.default_rng(seed)
    init_train = model.loss(model.forward(X)[0], y)
    history = {"train_loss": [init_train], "val_loss": [model.loss(model.forward(X_val)[0], y_val)]}
    best = (history["val_loss"][0], {k: v.copy() for k, v in model.params.items()}, 0)
    status, waited = "max_epochs", 0

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            idx = order[start:start + batch_size]
            y_hat, cache = model.forward(X[idx])
            model.step(model.backward(y_hat, y[idx], cache), lr)

        train_loss = model.loss(model.forward(X)[0], y)
        val_loss = model.loss(model.forward(X_val)[0], y_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if not np.isfinite(train_loss) or train_loss > diverge_factor * init_train:
            status = "diverged"
            break
        if val_loss < best[0]:
            best, waited = (val_loss, {k: v.copy() for k, v in model.params.items()}, epoch), 0
        else:
            waited += 1
            if waited >= patience:
                status = "early_stopped"
                break

    if status != "diverged":
        model.params = best[1]
    return history, {"status": status, "epochs_completed": epoch, "best_epoch": best[2], "best_val_loss": float(best[0])}


def gradient_check(model, X, y, entries, eps=1e-5):
    """Compare analytical gradients with central finite differences at chosen parameter entries.

    entries: list of (param_name, index_tuple). Returns one dict per entry.
    """
    y_hat, cache = model.forward(X)
    grads = model.backward(y_hat, y, cache)
    rows = []
    for name, idx in entries:
        p = model.params[name]
        original = p[idx]
        p[idx] = original + eps
        loss_plus = model.loss(model.forward(X)[0], y)
        p[idx] = original - eps
        loss_minus = model.loss(model.forward(X)[0], y)
        p[idx] = original
        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = grads[name][idx]
        abs_err = abs(analytic - numeric)
        rel_err = abs_err / max(abs(analytic), abs(numeric), 1e-12)
        rows.append({"param": name, "index": idx, "analytic": float(analytic), "numeric": float(numeric),
                     "abs_error": float(abs_err), "rel_error": float(rel_err)})
    return rows
