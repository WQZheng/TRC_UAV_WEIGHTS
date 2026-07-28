"""Smoke test: end-to-end SafePolicy is differentiable into theta.

Builds predictor + CBF-MPC + SafePolicy, runs a short closed-loop rollout
with the real 6-DOF dynamics, computes a task-style loss on the realised
trajectory, and checks that gradients reach the predictor parameters
theta. This is the core requirement for Stage-2.

NOTE: each cvxpylayers step is ~6 s on CPU but ~0.1 s on GPU. On CPU keep
T small; on GPU this runs comfortably.
"""
from __future__ import annotations
import torch

from params import DEFAULT_PARAMS
from dynamics import EVTOLDynamics
from wind import UrbanWindField
from predictor import GMMTrajectoryPredictor
from cbf_mpc import CBFMPCLayer
from safe_policy import SafePolicy

DTYPE = torch.float64
DT, HP, T, N, B = 0.2, 6, 5, 1, 1


def main():
    torch.manual_seed(0)
    pred = GMMTrajectoryPredictor(T=30, K=5).to(DTYPE)
    mpc = CBFMPCLayer(n_neighbors=N, horizon=HP, dt=DT, d_sep=30.0,
                      alpha=0.4, a_max=10.0)
    policy = SafePolicy(pred, mpc)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE)
    wind = UrbanWindField(eta_w=0.3, dtype=DTYPE, seed=0)

    x = torch.zeros(B, 12, dtype=DTYPE); x[:, 3] = 25.0
    neigh_pos = torch.zeros(B, N, 3, dtype=DTYPE); neigh_pos[:, 0, 0] = 200.0
    neigh_hist = torch.zeros(B, N, 25, 3, dtype=DTYPE)
    neigh_hist[:, 0, :, 0] = torch.linspace(-0.05, 0.0, 25, dtype=DTYPE)

    min_sep = torch.full((B,), float("inf"), dtype=DTYPE)
    for t in range(T):
        p0, v0 = x[:, 0:3], x[:, 3:6]
        tt = torch.arange(HP + 1, dtype=DTYPE) * DT
        p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
        u, _ = policy(x, neigh_hist, neigh_pos, p_ref)
        x = dyn.step(x, u, wind.sample(p0), DT)
        neigh_pos = neigh_pos.clone()
        neigh_pos[:, 0, 0] = neigh_pos[:, 0, 0] - 25.0 * DT
        d = torch.linalg.norm(x[:, 0:3] - neigh_pos[:, 0, :], dim=-1)
        min_sep = torch.minimum(min_sep, d)

    loss = -min_sep.mean()
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item()
                for p in pred.parameters() if p.grad is not None)
    print(f"sum |grad theta| = {gnorm:.4e}")
    assert gnorm > 0, "no gradient reached predictor theta"
    print("OK: end-to-end gradient reaches the predictor (Stage-2 ready)")


if __name__ == "__main__":
    main()
