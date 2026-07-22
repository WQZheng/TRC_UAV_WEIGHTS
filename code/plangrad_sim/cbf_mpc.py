"""Differentiable safe MPC layer with CBF separation constraints.

Implements the planner of manuscript Section 4.3 as a differentiable
convex QP (cvxpylayers / Proposition 2). A double-integrator surrogate
over the planning horizon Hp:

    decision: a_{0..Hp-1} (commanded accelerations, R^3 each)
              eps_{i,k}   (per-neighbour, per-step CBF slacks >= 0)
    states:   p_k, v_k    propagated by the linear double integrator

Discrete-time CBF separation, linearised around the predicted neighbour
positions:
    h_{i,k} >= (1 - alpha) h_{i,k-1} - eps_{i,k},  eps >= 0
    h_{i,k} = nrm_{i,k} . p_k - b_{i,k} - d_sep
with nrm_{i,k} a unit direction (parameter) and
b_{i,k} = nrm_{i,k} . c_{i,k} precomputed in torch (keeps DPP). alpha is
a fixed float compiled into the program (the recursion would otherwise
create alpha^k parameter self-products, which DPP forbids).

Gradients of a_0 w.r.t. the predicted neighbour positions flow through
the KKT system -- exactly what Stage-2 task-aligned training needs.
"""
from __future__ import annotations
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class CBFMPCLayer:
    def __init__(self, n_neighbors: int, horizon: int = 8, dt: float = 0.2,
                 d_sep: float = 30.0, a_max: float = 8.0, alpha: float = 0.3,
                 w_track: float = 1.0, w_acc: float = 0.05,
                 w_slack: float = 1e3):
        self.N = n_neighbors
        self.Hp = horizon
        self.dt = dt
        self.d_sep = d_sep
        self.a_max = a_max
        self.alpha = float(alpha)
        self._build(w_track, w_acc, w_slack)

    def _build(self, w_track, w_acc, w_slack):
        N, Hp, dt = self.N, self.Hp, self.dt

        a = cp.Variable((Hp, 3), name="a")
        p = cp.Variable((Hp + 1, 3), name="p")
        v = cp.Variable((Hp + 1, 3), name="v")
        eps = cp.Variable((N, Hp), name="eps")

        p0 = cp.Parameter(3, name="p0")
        v0 = cp.Parameter(3, name="v0")
        p_ref = cp.Parameter((Hp + 1, 3), name="p_ref")
        nrm = cp.Parameter((N, Hp + 1, 3), name="nrm")
        bvec = cp.Parameter((N, Hp + 1), name="bvec")
        h0 = cp.Parameter((N,), name="h0")

        cons = [p[0] == p0, v[0] == v0]
        for k in range(Hp):
            cons += [p[k + 1] == p[k] + dt * v[k],
                     v[k + 1] == v[k] + dt * a[k]]
            cons += [cp.norm(a[k], "inf") <= self.a_max]

        for i in range(N):
            h_prev = h0[i]
            for k in range(1, Hp + 1):
                h_k = nrm[i, k, :] @ p[k] - bvec[i, k] - self.d_sep
                cons += [h_k >= (1 - self.alpha) * h_prev - eps[i, k - 1]]
                h_prev = h_k
        cons += [eps >= 0]

        obj = cp.Minimize(
            w_track * cp.sum_squares(p - p_ref)
            + w_acc * cp.sum_squares(a)
            + w_slack * cp.sum(eps))
        prob = cp.Problem(obj, cons)
        assert prob.is_dpp(), "CBF-MPC QP is not DPP"

        self.layer = CvxpyLayer(
            prob,
            parameters=[p0, v0, p_ref, nrm, bvec, h0],
            variables=[a, p, v, eps],
        )

    def solve(self, p0, v0, p_ref, neigh_pred, alpha=None):
        """Differentiable solve. Returns a0 [B,3], info dict {a,p,v,eps}."""
        ref_exp = p_ref.unsqueeze(1)
        diff = ref_exp - neigh_pred
        dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp_min(1e-6)
        nrm = diff / dist
        bvec = (nrm * neigh_pred).sum(-1)
        h0 = (nrm[:, :, 0, :] * (p0.unsqueeze(1) - neigh_pred[:, :, 0, :])
              ).sum(-1) - self.d_sep
        a_sol, p_sol, v_sol, eps_sol = self.layer(
            p0, v0, p_ref, nrm, bvec, h0)
        return a_sol[:, 0, :], {"a": a_sol, "p": p_sol, "v": v_sol,
                                "eps": eps_sol}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, Hp = 2, 2, 8
    layer = CBFMPCLayer(n_neighbors=N, horizon=Hp, dt=0.2, d_sep=30.0,
                        alpha=0.3)
    p0 = torch.zeros(B, 3, dtype=torch.float64)
    v0 = torch.zeros(B, 3, dtype=torch.float64); v0[:, 0] = 20.0
    t = torch.arange(Hp + 1, dtype=torch.float64) * 0.2
    p_ref = torch.zeros(B, Hp + 1, 3, dtype=torch.float64)
    p_ref[:, :, 0] = 20.0 * t
    neigh = torch.zeros(B, N, Hp + 1, 3, dtype=torch.float64)
    neigh[:, 0, :, 0] = 120.0 - 25.0 * t
    neigh[:, 1, :, 1] = 25.0
    neigh.requires_grad_(True)
    a0, info = layer.solve(p0, v0, p_ref, neigh)
    print("a0 =", a0.detach().numpy().round(3))
    (a0 ** 2).sum().backward()
    print("grad finite/nonzero:", bool(torch.isfinite(neigh.grad).all()),
          float(neigh.grad.abs().sum()) > 0)
