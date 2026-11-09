"""Training loop for the PyTorch models: mini-batches, Adam, validation early stopping, checkpointing."""
import copy
import platform
import random
import time

import numpy as np
import torch
from torch import nn

from src.models import build_model, count_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def environment():
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "torch_threads": torch.get_num_threads(), "device": "cpu"}


@torch.no_grad()
def predict(model, X, batch_size=1024):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + batch_size])) for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


def fit(cfg, X_tr, y_tr, X_va, y_va, seed):
    """Train one model. Targets must already be scaled to [0, 1] by the RUL cap.

    Stopping rule: keep the weights with the lowest validation loss; stop after
    `patience` epochs without improvement or at `max_epochs`.
    """
    set_seed(seed)
    model = build_model(cfg, X_tr.shape[1], X_tr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    Xv, yv = torch.from_numpy(X_va), torch.from_numpy(y_va)
    gen = torch.Generator().manual_seed(seed)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch, waited = float("inf"), None, 0, 0
    start = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(Xt), cfg["batch_size"]):
            idx = order[i:i + cfg["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        history["train_loss"].append(total / len(Xt))
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val, best_state, best_epoch, waited = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1
            if waited >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    info = {"epochs_completed": epoch, "best_epoch": best_epoch, "best_val_loss": best_val,
            "train_seconds": time.perf_counter() - start, "n_params": count_params(model)}
    return model, history, info
