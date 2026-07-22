"""Constant-Velocity (CV) trajectory predictor baseline.

The simplest, training-free neighbour predictor and a standard sanity
baseline in trajectory-prediction / conflict-detection literature: assume
the neighbour keeps its last observed velocity and extrapolate a straight
line. We expose it through the SAME interface as
plangrad_sim.predictor.GMMTrajectoryPredictor so it drops straight into
the unified SafePolicy + CBF-MPC closed loop with ZERO other changes ->
the ONLY thing that differs from PlanGrad is the predictor, so any safety
difference is attributable to prediction quality.

Output contract (identical to the GMM predictor; positions are /SCALE):
  alpha     [B, T, K]      mixture weights (all mass on mode 0)
  mu        [B, T, K, 3]   per-mode mean future displacement (recentred)
  log_sigma [B, T, K, 3]   per-mode log std (constant, isotropic)

Input contract:
  neigh_hist [B, 25, 3]    recentred, /SCALE neighbour history (last row ~0)

Velocity is the mean first-difference over the last `vel_window` history
steps (robust to a single noisy sample). Future displacements are
last_pos + v * k  for k = 1..T. Because the dataset is recentred so the
last observed position is ~0, mu[:,k] ~= v * (k+1).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ConstantVelocityPredictor(nn.Module):
    def __init__(self, T: int = 30, K: int = 5, vel_window: int = 5,
                 sigma: float = 0.05):
        super().__init__()
        self.T = T
        self.K = K
        self.vel_window = vel_window
        # constant log-sigma (sigma in /SCALE units; 0.05 -> 5 m at SCALE=100)
        self.log_sigma_val = float(torch.log(torch.tensor(sigma)))

    @torch.no_grad()
    def forward(self, neigh_hist: torch.Tensor) -> dict:
        # neigh_hist: [B, L, 3]
        B, L, _ = neigh_hist.shape
        dtype, device = neigh_hist.dtype, neigh_hist.device
        w = min(self.vel_window, L - 1)
        # mean per-step velocity over the last w history steps
        recent = neigh_hist[:, L - w - 1:, :]                  # [B, w+1, 3]
        vel = (recent[:, 1:, :] - recent[:, :-1, :]).mean(1)   # [B, 3]
        last = neigh_hist[:, -1, :]                            # [B, 3] (~0)

        ks = torch.arange(1, self.T + 1, dtype=dtype, device=device)  # [T]
        # mean future displacement: last + v*k   -> [B, T, 3]
        mu0 = last.unsqueeze(1) + vel.unsqueeze(1) * ks.view(1, self.T, 1)

        mu = mu0.unsqueeze(2).expand(B, self.T, self.K, 3).contiguous()
        alpha = torch.zeros(B, self.T, self.K, dtype=dtype, device=device)
        alpha[:, :, 0] = 1.0   # all probability mass on the single CV mode
        log_sigma = torch.full((B, self.T, self.K, 3), self.log_sigma_val,
                               dtype=dtype, device=device)
        return {"alpha": alpha, "mu": mu, "log_sigma": log_sigma}

    def eval(self):
        return self
