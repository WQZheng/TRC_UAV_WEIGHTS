"""Differentiable closed-loop corridor simulator.

Ties together the eVTOL dynamics, the wind field, and neighbour replay
into a single differentiable rollout. A user-supplied `policy` callable
maps (ego_state, neighbor_states, wind_nowcast) -> control u; gradients
propagate through the entire rollout, which is what enables end-to-end
task-aligned training. This is the interactive environment that recorded
trajectories alone cannot provide.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from dynamics import EVTOLDynamics
from wind import UrbanWindField
from params import DEFAULT_PARAMS


@dataclass
class RolloutResult:
    ego_traj: torch.Tensor        # [B, T+1, 12]
    controls: torch.Tensor        # [B, T, 4]
    min_separation: torch.Tensor  # [B]
    collided: torch.Tensor        # [B] 1.0 if min_sep < d_sep


class CorridorSimulator:
    def __init__(self, dynamics: EVTOLDynamics, wind: UrbanWindField,
                 dt: float = 0.2, d_sep: float = 30.0):
        self.dyn = dynamics
        self.wind = wind
        self.dt = dt
        self.d_sep = d_sep

    def rollout(self, x0, neighbor_traj, policy, horizon):
        """Differentiable closed-loop rollout.

        x0:            [B, 12]
        neighbor_traj: [B, N, T+1, 3]
        policy:        callable(x, neigh_pos, w_nowcast) -> u [B,4]
        """
        B = x0.shape[0]
        x = x0
        ego_states = [x]
        controls = []
        min_sep = torch.full((B,), float("inf"), dtype=x.dtype,
                             device=x.device)

        for t in range(horizon):
            ego_pos = x[:, 0:3]
            neigh_pos = neighbor_traj[:, :, t, :]
            w_now = self.wind.nowcast(ego_pos)
            u = policy(x, neigh_pos, w_now)
            controls.append(u)
            w_true = self.wind.sample(ego_pos)
            x = self.dyn.step(x, u, w_true, self.dt)
            ego_states.append(x)
            d = torch.linalg.norm(x[:, None, 0:3] - neigh_pos, dim=-1)
            min_sep = torch.minimum(min_sep, d.min(dim=1).values)

        ego_traj = torch.stack(ego_states, dim=1)
        ctrl = torch.stack(controls, dim=1)
        collided = (min_sep < self.d_sep).to(x.dtype)
        return RolloutResult(ego_traj, ctrl, min_sep, collided)


def make_default_simulator(dt: float = 0.2, eta_w: float = 0.5,
                           device: str = "cpu", seed: int | None = 0):
    dyn = EVTOLDynamics(DEFAULT_PARAMS, device=device)
    wind = UrbanWindField(eta_w=eta_w, device=device, seed=seed)
    return CorridorSimulator(dyn, wind, dt=dt)
