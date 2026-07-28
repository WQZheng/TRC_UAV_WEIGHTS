"""Urban wind field model (manuscript Section 5.2).

Two-tier representation: a mesoscale mean wind plus a corridor-scale
Gaussian random field (GRF) perturbation (approximated with random
Fourier features). A single multiplicative factor eta_w scales the
disturbance strength; eta_w is varied in the generalization studies.
Pure torch so it stays in the differentiable rollout's gradient path.
"""
from __future__ import annotations
import torch


class UrbanWindField:
    def __init__(self, mean_wind=(2.0, 0.0, 0.0), eta_w: float = 0.5,
                 corr_length: float = 30.0, gust_std: float = 3.0,
                 device: str = "cpu", dtype: torch.dtype = torch.float64,
                 seed: int | None = None):
        self.mean = torch.tensor(mean_wind, dtype=dtype, device=device)
        self.eta_w = eta_w
        self.corr_length = corr_length
        self.gust_std = gust_std
        self.device = device
        self.dtype = dtype
        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)
        self.n_modes = 8
        self._freqs = torch.randn(self.n_modes, 3, generator=self.gen,
                                  dtype=dtype, device=device) / corr_length
        self._phase = 2 * torch.pi * torch.rand(
            self.n_modes, generator=self.gen, dtype=dtype, device=device)
        self._amp = torch.randn(self.n_modes, 3, generator=self.gen,
                                dtype=dtype, device=device)

    def sample(self, pos: torch.Tensor) -> torch.Tensor:
        """Wind velocity at inertial positions pos [B,3] -> [B,3]."""
        proj = pos @ self._freqs.T + self._phase
        feats = torch.cos(proj)
        gust = feats @ self._amp
        gust = gust / (self.n_modes ** 0.5) * self.gust_std
        return self.mean.unsqueeze(0) + self.eta_w * gust

    def nowcast(self, pos: torch.Tensor, noise_std: float = 0.5):
        """Noisy wind nowcast delivered to the planner (Section 5.2)."""
        true_w = self.sample(pos)
        noise = noise_std * torch.randn(
            true_w.shape, generator=self.gen, dtype=self.dtype,
            device=self.device)
        return true_w + noise
