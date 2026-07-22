"""Vanilla-MPC planner (NO control-barrier-function safety certificate).

Ablates the CBF safety layer of PlanGrad. Same differentiable convex QP,
same double-integrator surrogate over the planning horizon, same a_max
actuation limit and same reference-tracking objective as
plangrad_sim.cbf_mpc.CBFMPCLayer -- but the hard discrete-time CBF
separation constraints

    h_{i,k} >= (1-alpha) h_{i,k-1} - eps

are REMOVED. Collision avoidance is instead attempted only through a soft
linear repulsion term in the objective (penalise the planned position for
being inside a stand-off half-space in front of the predicted neighbour).
This is the canonical "MPC with a collision penalty but no safety
guarantee" baseline: it shows how much of PlanGrad's safety comes from the
CBF certificate rather than from the predictor or tracking controller.

The QP stays DPP/differentiable so it runs in the identical closed loop.
Drop-in: exposes `.Hp` and `.solve(p0,v0,p_ref,neigh_pred)` exactly like
CBFMPCLayer, so SafePolicy can reuse its acceleration->control mapping.

DPP note: like CBFMPCLayer pre-computes bvec = nrm . c in torch, we
pre-compute the stand-off offset  b_{i,k} = rep_dir . rep_pt  in torch so
the objective term is  pos(b - rep_dir . p)  (parameter . variable + scalar
parameter), which is DPP-compliant.
"""
from __future__ import annotations
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class VanillaMPCLayer:
    def __init__(self, n_neighbors: int, horizon: int = 15, dt: float = 0.2,
                 d_sep: float = 30.0, a_max: float = 20.0,
                 w_track: float = 1.0, w_acc: float = 0.05,
                 w_rep: float = 50.0):
        self.N = n_neighbors
        self.Hp = horizon
        self.dt = dt
        self.d_sep = d_sep
        self.a_max = a_max
        self.w_rep = w_rep
        self._build(w_track, w_acc, w_rep)

    def _build(self, w_track, w_acc, w_rep):
        N, Hp, dt = self.N, self.Hp, self.dt
        a = cp.Variable((Hp, 3), name="a")
        p = cp.Variable((Hp + 1, 3), name="p")
        v = cp.Variable((Hp + 1, 3), name="v")

        p0 = cp.Parameter(3, name="p0")
        v0 = cp.Parameter(3, name="v0")
        p_ref = cp.Parameter((Hp + 1, 3), name="p_ref")
        rep_dir = cp.Parameter((N, Hp + 1, 3), name="rep_dir")   # unit, ego-side
        rep_b = cp.Parameter((N, Hp + 1), name="rep_b")          # = dir . rep_pt

        cons = [p[0] == p0, v[0] == v0]
        for k in range(Hp):
            cons += [p[k + 1] == p[k] + dt * v[k],
                     v[k + 1] == v[k] + dt * a[k]]
            cons += [cp.norm(a[k], "inf") <= self.a_max]

        # soft repulsion: penalise being on the neighbour side of the
        # stand-off plane, i.e. pos(rep_b - rep_dir . p_k) > 0 when the ego
        # is closer to the neighbour than the d_sep stand-off point.
        rep = 0
        for i in range(N):
            for k in range(1, Hp + 1):
                rep += cp.pos(rep_b[i, k] - rep_dir[i, k, :] @ p[k])

        obj = cp.Minimize(w_track * cp.sum_squares(p - p_ref)
                          + w_acc * cp.sum_squares(a)
                          + w_rep * rep)
        prob = cp.Problem(obj, cons)
        assert prob.is_dpp(), "Vanilla-MPC QP is not DPP"
        self.layer = CvxpyLayer(prob,
                                parameters=[p0, v0, p_ref, rep_dir, rep_b],
                                variables=[a, p, v])

    def solve(self, p0, v0, p_ref, neigh_pred, alpha=None):
        # neigh_pred: [B,N,Hp+1,3]; ego ref expanded over neighbours
        ref_exp = p_ref.unsqueeze(1)                       # [B,1,Hp+1,3]
        diff = ref_exp - neigh_pred                        # ego - neighbour
        dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp_min(1e-6)
        rep_dir = diff / dist                              # unit, away from nbr
        rep_pt = neigh_pred + rep_dir * self.d_sep         # stand-off point
        rep_b = (rep_dir * rep_pt).sum(-1)                 # [B,N,Hp+1] scalar
        a_sol, p_sol, v_sol = self.layer(p0, v0, p_ref, rep_dir, rep_b)
        return a_sol[:, 0, :], {"a": a_sol, "p": p_sol, "v": v_sol,
                                "eps": torch.zeros(p0.shape[0], self.N,
                                                   self.Hp, dtype=p0.dtype,
                                                   device=p0.device)}
