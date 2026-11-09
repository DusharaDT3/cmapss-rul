"""Checker for CW2 Q1b-c: compute the forward pass, gradients and one update for the given network,
so a hand calculation can be verified. The report must show the chain-rule derivation itself.

Network: z = Wx + b, h = ReLU(z), y_hat = v^T h + c, L = 0.5 (y_hat - y)^2
Usage: python scripts/check_q1.py
"""
import numpy as np

x = np.array([1.0, 2.0])
y = 1.0
W = np.array([[0.2, -0.1], [0.1, 0.3]])
b = np.array([0.1, -0.2])
v = np.array([0.4, -0.5])
c = 0.2
lr = 0.1


def forward(W, b, v, c):
    z = W @ x + b
    h = np.maximum(z, 0.0)
    y_hat = v @ h + c
    return z, h, y_hat, 0.5 * (y_hat - y) ** 2


z, h, y_hat, L = forward(W, b, v, c)
print("Forward pass")
print("  z      =", z)
print("  h      =", h)
print("  y_hat  =", round(y_hat, 6), "| prediction error y_hat - y =", round(y_hat - y, 6), "| loss L =", round(L, 6))

e = y_hat - y                        # dL/dy_hat
dv, dc = e * h, e
delta = e * v * (z > 0)              # hidden error signal: output error x outgoing weight x ReLU'(z)
dW, db = np.outer(delta, x), delta
print("Gradients at the original parameters")
print("  dL/dv =", dv, " dL/dc =", round(dc, 6))
print("  hidden error signals delta =", delta)
print("  dL/dW =\n", dW)
print("  dL/db =", db)

W2, b2, v2, c2 = W - lr * dW, b - lr * db, v - lr * dv, c - lr * dc
print("After one simultaneous update (lr = 0.1)")
print("  W =\n", W2)
print("  b =", b2, " v =", v2, " c =", round(c2, 6))
_, _, y_new, L_new = forward(W2, b2, v2, c2)
print("  new y_hat =", round(y_new, 6), " new loss =", round(L_new, 6))
