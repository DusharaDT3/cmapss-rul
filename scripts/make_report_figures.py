"""Diagrams for the CW2 report (Q1a XOR mapping, Q1b worked network). Usage: python scripts/make_report_figures.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Q1a: XOR is not linearly separable in input space, but is after two ReLU hidden units.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t = np.array([0, 1, 1, 0])
H = np.stack([np.maximum(X.sum(1), 0), np.maximum(X.sum(1) - 1, 0)], axis=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for a, pts, title, xl, yl in ((ax[0], X, "Input space: no single line separates the classes", "$x_1$", "$x_2$"),
                              (ax[1], H, "Hidden space: $h_1$=ReLU($x_1+x_2$), $h_2$=ReLU($x_1+x_2-1$)", "$h_1$", "$h_2$")):
    for cls, mk, col in ((0, "o", "tab:blue"), (1, "s", "tab:red")):
        a.scatter(*pts[t == cls].T, s=120, marker=mk, c=col, label=f"XOR = {cls}", zorder=3)
    a.set(title=title, xlabel=xl, ylabel=yl); a.grid(alpha=0.3)
for txt, (h1, h2) in (("x=(0,0)", (0, 0)), ("x=(0,1) and (1,0)", (1, 0)), ("x=(1,1)", (2, 1))):
    ax[1].annotate(txt, (h1, h2), textcoords="offset points", xytext=(8, 8))
hh = np.linspace(-0.2, 2.2, 10)
ax[1].plot(hh, (hh - 0.5) / 2, "k--", label="$h_1 - 2h_2 = 0.5$")
ax[0].legend(loc="center"); ax[1].legend(loc="upper left")
ax[0].set_xlim(-0.3, 1.3); ax[0].set_ylim(-0.3, 1.3); ax[1].set_xlim(-0.3, 2.3); ax[1].set_ylim(-0.3, 1.3)
fig.tight_layout(); fig.savefig(OUT / "figQ1a_xor.png", dpi=150)

# Q1b: the 2-2-1 network with forward values.
fig, ax = plt.subplots(figsize=(8, 4.2)); ax.axis("off")
nodes = {"x1": (0, 2.5), "x2": (0, 0.5), "h1": (3, 2.5), "h2": (3, 0.5), "y": (6, 1.5)}
labels = {"x1": "$x_1=1$", "x2": "$x_2=2$", "h1": "$z_1=0.1$\n$h_1=0.1$", "h2": "$z_2=0.5$\n$h_2=0.5$", "y": "$\\hat y=-0.01$\n$L=0.51005$"}
edges = [("x1", "h1", "$W_{11}$=0.2"), ("x2", "h1", "$W_{12}$=-0.1"), ("x1", "h2", "$W_{21}$=0.1"), ("x2", "h2", "$W_{22}$=0.3"), ("h1", "y", "$v_1$=0.4"), ("h2", "y", "$v_2$=-0.5")]
for a, b, w in edges:
    (x0, y0), (x1, y1) = nodes[a], nodes[b]
    ax.annotate("", (x1 - 0.45, y1), (x0 + 0.45, y0), arrowprops=dict(arrowstyle="->", color="gray"))
    f = 0.5 if y0 == y1 or a.startswith("h") else 0.28
    ax.text(x0 + f * (x1 - x0), y0 + f * (y1 - y0) + 0.12, w, fontsize=9, ha="center", color="tab:blue",
            bbox=dict(fc="white", ec="none", pad=0.5))
for k, (x, y) in nodes.items():
    ax.add_patch(plt.Circle((x, y), 0.45, fc="white", ec="k"))
    ax.text(x, y, labels[k], ha="center", va="center", fontsize=8.5)
ax.text(3, 3.35, "ReLU hidden layer, $b=[0.1,-0.2]^T$", ha="center"); ax.text(6, 2.35, "linear output, $c=0.2$", ha="center")
ax.set_xlim(-0.8, 7); ax.set_ylim(-0.2, 3.6); ax.set_aspect("equal")
ax.set_title("Q1 network (original parameters) with forward-pass values")
fig.tight_layout(); fig.savefig(OUT / "figQ1b_network.png", dpi=150)
print("saved", sorted(p.name for p in OUT.iterdir()))
