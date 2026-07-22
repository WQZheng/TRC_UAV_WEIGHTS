"""Minimal check: a differentiable QP via cvxpylayers, gradient flows.

Solves min 0.5||z - a||^2 s.t. Gz <= h, and checks d(loss)/d(a) exists.
Validates the implicit-differentiation backend (Proposition 2) before
building the full CBF-MPC layer.
"""
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

n = 4
z = cp.Variable(n)
a = cp.Parameter(n)
G = cp.Parameter((2, n))
h = cp.Parameter(2)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(z - a)), [G @ z <= h])
assert prob.is_dpp(), "problem is not DPP (required by cvxpylayers)"
layer = CvxpyLayer(prob, parameters=[a, G, h], variables=[z])

a_t = torch.tensor([1.0, 2.0, -1.0, 0.5], requires_grad=True,
                   dtype=torch.float64)
G_t = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=torch.float64)
h_t = torch.tensor([0.3, 0.3], dtype=torch.float64)

(z_sol,) = layer(a_t, G_t, h_t)
print("z* =", z_sol.detach().numpy().round(4))
loss = (z_sol ** 2).sum()
loss.backward()
print("d loss / d a =", a_t.grad.numpy().round(4))
assert a_t.grad is not None and torch.isfinite(a_t.grad).all()
print("OK: differentiable QP works, gradient flows through KKT")
