"""Smoke tests for the differentiable corridor simulator.

T1. Dynamics step + hover thrust balances gravity (zero drift).
T2. Closed-loop rollout runs.
T3. Gradients flow through the whole rollout into policy parameters
    (core requirement for end-to-end task-aligned training).
T4. GUAM trajectories load and serve as neighbour replay.
"""
from __future__ import annotations
import torch

from config import GUAM_MAT
from params import DEFAULT_PARAMS
from dynamics import EVTOLDynamics
from wind import UrbanWindField
from simulator import CorridorSimulator
from guam_data import GUAMTrajectories

DTYPE = torch.float64


def t1_dynamics_hover():
    print("[T1] dynamics + hover balance")
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE)
    x = torch.zeros(4, 12, dtype=DTYPE)
    u = torch.zeros(4, 4, dtype=DTYPE); u[:, 0] = DEFAULT_PARAMS.weight
    w = torch.zeros(4, 3, dtype=DTYPE)
    x1 = dyn.step(x, u, w, dt=0.2)
    dz = (x1[:, 2] - x[:, 2]).abs().max().item()
    dvz = x1[:, 5].abs().max().item()
    print(f"     |dz|={dz:.3e} m, |vz|={dvz:.3e} m/s (expect ~0)")
    assert dz < 1e-3 and dvz < 1e-3, "hover not balanced"
    print("     OK")


def t2_rollout():
    print("[T2] closed-loop rollout")
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE)
    wind = UrbanWindField(eta_w=0.5, dtype=DTYPE, seed=0)
    sim = CorridorSimulator(dyn, wind, dt=0.2, d_sep=30.0)
    B, N, T = 4, 2, 30
    x0 = torch.zeros(B, 12, dtype=DTYPE); x0[:, 3] = 20.0
    neigh = torch.zeros(B, N, T + 1, 3, dtype=DTYPE)
    neigh[:, 0, :, 0] = torch.linspace(200, 50, T + 1)
    neigh[:, 1, :, 1] = 60.0

    def hover_policy(x, neigh_pos, w_now):
        u = torch.zeros(x.shape[0], 4, dtype=DTYPE)
        u[:, 0] = DEFAULT_PARAMS.weight
        return u

    res = sim.rollout(x0, neigh, hover_policy, horizon=T)
    print(f"     ego_traj {tuple(res.ego_traj.shape)}, "
          f"controls {tuple(res.controls.shape)}")
    print("     OK")


def t3_gradient_flow():
    print("[T3] gradient flows through rollout into policy params")
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE)
    wind = UrbanWindField(eta_w=0.5, dtype=DTYPE, seed=1)
    sim = CorridorSimulator(dyn, wind, dt=0.2, d_sep=30.0)
    B, N, T = 3, 1, 25
    x0 = torch.zeros(B, 12, dtype=DTYPE); x0[:, 3] = 15.0
    neigh = torch.zeros(B, N, T + 1, 3, dtype=DTYPE)
    neigh[:, 0, :, 0] = torch.linspace(150, 30, T + 1)
    theta = torch.zeros(2, dtype=DTYPE, requires_grad=True)

    def policy(x, neigh_pos, w_now):
        u = torch.zeros(x.shape[0], 4, dtype=DTYPE)
        u[:, 0] = DEFAULT_PARAMS.weight + theta[0]
        u[:, 2] = theta[1]
        return u

    res = sim.rollout(x0, neigh, policy, horizon=T)
    loss = (-res.min_separation.mean() + 1e-6 * (res.controls ** 2).mean())
    loss.backward()
    g = theta.grad
    print(f"     d loss / d theta = {g.tolist()}")
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print("     OK (gradient is finite and non-zero)")


def t4_guam_neighbours():
    print("[T4] GUAM trajectories as neighbour replay")
    g = GUAMTrajectories(GUAM_MAT)
    tr = g.load_trajectory(0, samples_per_seg=10)
    print(f"     loaded GUAM traj 0: shape={tr.shape}")
    print("     OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    t1_dynamics_hover(); print()
    t2_rollout(); print()
    t3_gradient_flow(); print()
    t4_guam_neighbours()
    print("\nAll smoke tests passed.")
