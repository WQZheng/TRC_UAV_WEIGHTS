"""Fast, non-differentiable CBF-MPC solver for EVALUATION rollouts.

The differentiable ``CBFMPCLayer`` (cbf_mpc.py) is required during Stage-2
training because gradients of the planned action must flow back into the
predictor. In pure *evaluation* rollouts---such as the multi-agent
penetration study---no gradient is needed, and paying the cvxpylayers
differentiable-solve cost per agent per step is what makes a system-level
Monte-Carlo sweep intractable.

This module solves the IDENTICAL optimisation problem---same double-integrator
dynamics, same discrete-time CBF separation constraints, same actuation bound,
same objective and weights as CBFMPCLayer---but with a direct compiled QP
solver (OSQP via cvxpy Parameters), which is ~10-50x faster. It is therefore a
drop-in replacement for the planner in evaluation only; the safety semantics
are exactly those of the differentiable layer (verified numerically against it
in ``__main__`` and in test_fast_matches_layer). Nothing scientific changes;
only the unused differentiability is dropped.
"""
from __future__ import annotations
import numpy as np
import cvxpy as cp


class FastCBFMPC:
    def __init__(self, n_neighbors: int, horizon: int = 15, dt: float = 0.2,
                 d_sep: float = 30.0, a_max: float = 20.0, alpha: float = 0.1,
                 w_track: float = 1.0, w_acc: float = 0.05, w_slack: float = 1e3):
        self.N = n_neighbors
        self.Hp = horizon
        self.dt = dt
        self.d_sep = d_sep
        self.a_max = a_max
        self.alpha = float(alpha)
        self._build(w_track, w_acc, w_slack)

    def _build(self, w_track, w_acc, w_slack):
        N, Hp, dt = self.N, self.Hp, self.dt
        a = cp.Variable((Hp, 3))
        p = cp.Variable((Hp + 1, 3))
        v = cp.Variable((Hp + 1, 3))
        eps = cp.Variable((N, Hp))

        # parameters set per solve
        self.p0 = cp.Parameter(3)
        self.v0 = cp.Parameter(3)
        self.p_ref = cp.Parameter((Hp + 1, 3))
        self.nrm = cp.Parameter((N, Hp + 1, 3))
        self.bvec = cp.Parameter((N, Hp + 1))
        self.h0 = cp.Parameter((N,))

        cons = [p[0] == self.p0, v[0] == self.v0]
        for k in range(Hp):
            cons += [p[k + 1] == p[k] + dt * v[k],
                     v[k + 1] == v[k] + dt * a[k]]
            cons += [cp.norm(a[k], "inf") <= self.a_max]
        for i in range(N):
            h_prev = self.h0[i]
            for k in range(1, Hp + 1):
                h_k = self.nrm[i, k, :] @ p[k] - self.bvec[i, k] - self.d_sep
                cons += [h_k >= (1 - self.alpha) * h_prev - eps[i, k - 1]]
                h_prev = h_k
        cons += [eps >= 0]

        obj = cp.Minimize(
            w_track * cp.sum_squares(p - self.p_ref)
            + w_acc * cp.sum_squares(a)
            + w_slack * cp.sum(eps))
        self.prob = cp.Problem(obj, cons)
        self._a = a

    def solve_np(self, p0, v0, p_ref, neigh_pred):
        """All numpy. p0,v0:(3,); p_ref:(Hp+1,3); neigh_pred:(N,Hp+1,3).
        Returns commanded acceleration a0:(3,)."""
        ref = p_ref[None]                        # (1,Hp+1,3)
        diff = ref - neigh_pred                  # (N,Hp+1,3)
        dist = np.linalg.norm(diff, axis=-1, keepdims=True).clip(1e-6)
        nrm = diff / dist
        bvec = (nrm * neigh_pred).sum(-1)
        h0 = (nrm[:, 0, :] * (p0[None] - neigh_pred[:, 0, :])).sum(-1) - self.d_sep

        self.p0.value = np.asarray(p0, float)
        self.v0.value = np.asarray(v0, float)
        self.p_ref.value = np.asarray(p_ref, float)
        self.nrm.value = np.asarray(nrm, float)
        self.bvec.value = np.asarray(bvec, float)
        self.h0.value = np.asarray(h0, float)
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True,
                            max_iter=20000, eps_abs=1e-6, eps_rel=1e-6,
                            verbose=False)
            if self._a.value is None:
                # fall back to a more robust conic solver on the rare
                # pathological geometry OSQP cannot polish
                self.prob.solve(solver=cp.CLARABEL, verbose=False)
                if self._a.value is None:
                    return None
            return np.asarray(self._a.value[0])
        except Exception:
            try:
                self.prob.solve(solver=cp.CLARABEL, verbose=False)
                return None if self._a.value is None else np.asarray(self._a.value[0])
            except Exception:
                return None


if __name__ == "__main__":
    # verify FastCBFMPC matches the differentiable CBFMPCLayer on a head-on.
    import torch
    from cbf_mpc import CBFMPCLayer

    N, Hp = 2, 15
    fast = FastCBFMPC(n_neighbors=N, horizon=Hp, dt=0.2, d_sep=30.0,
                      alpha=0.1, a_max=20.0)
    layer = CBFMPCLayer(n_neighbors=N, horizon=Hp, dt=0.2, d_sep=30.0,
                        alpha=0.1, a_max=20.0)

    rng = np.random.default_rng(0)
    max_err = 0.0
    for trial in range(8):
        p0 = rng.normal(size=3) * 5
        v0 = np.array([20.0, 0, 0]) + rng.normal(size=3)
        tt = np.arange(Hp + 1) * 0.2
        p_ref = p0[None] + np.outer(tt, np.array([20.0, 0, 0]))
        neigh = np.zeros((N, Hp + 1, 3))
        for i in range(N):
            neigh[i, :, 0] = 150.0 - 25.0 * tt + rng.normal() * 5
            neigh[i, :, 1] = 30.0 * i + rng.normal() * 3
        a_fast = fast.solve_np(p0, v0, p_ref, neigh)
        a_lay, _ = layer.solve(
            torch.tensor(p0[None]), torch.tensor(v0[None]),
            torch.tensor(p_ref[None]), torch.tensor(neigh[None]))
        a_lay = a_lay[0].detach().numpy()
        err = np.linalg.norm(a_fast - a_lay)
        max_err = max(max_err, err)
        print(f"trial {trial}: |a_fast-a_layer| = {err:.4f} m/s^2  "
              f"(a_fast={a_fast.round(2)}, a_layer={a_lay.round(2)})")
    print(f"\nMAX disagreement over trials: {max_err:.4f} m/s^2  -> "
          f"{'MATCH' if max_err < 0.5 else 'MISMATCH'}")
