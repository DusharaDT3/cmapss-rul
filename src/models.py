"""PyTorch models for T3. Both map a (batch, window, n_features) tensor to (batch,) RUL / cap."""
import torch
from torch import nn


class WindowMLP(nn.Module):
    """Feed-forward network on the flattened window: position-specific weights, no state or weight sharing across time."""

    def __init__(self, window, n_features, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers, n_in = [nn.Flatten()], window * n_features
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(dropout)]
            n_in = h
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTMRegressor(nn.Module):
    """Recurrent network: reads the window cycle by cycle and predicts from the final hidden state."""

    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


def build_model(cfg, window, n_features):
    arch = cfg["architecture"]
    if arch == "mlp":
        return WindowMLP(window, n_features, tuple(cfg["hidden"]), cfg["dropout"])
    if arch == "lstm":
        return LSTMRegressor(n_features, cfg["hidden"], cfg["layers"], cfg["dropout"])
    raise ValueError(f"unknown architecture {arch!r}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_table(model, window, n_features):
    """Rows of (layer, output shape, trainable params) for a batch of one window."""
    rows, x = [], torch.zeros(1, window, n_features)
    rows.append(("input", tuple(x.shape), 0))
    if isinstance(model, WindowMLP):
        for m in model.net:
            x = m(x)
            rows.append((m.__class__.__name__, tuple(x.shape), sum(p.numel() for p in m.parameters())))
    else:
        out, _ = model.lstm(x)
        rows.append((f"LSTM x{model.lstm.num_layers} (all steps)", tuple(out.shape), count_params(model.lstm)))
        last = out[:, -1]
        rows.append(("last time step + Dropout", tuple(last.shape), 0))
        rows.append(("Linear", tuple(model.head(last).shape), count_params(model.head)))
    return rows
