"""Trajectory predictor f_theta with a multi-modal Gaussian-mixture head.

Implements the predictor of manuscript Section 4.2: a GRU encoder over
kinematic features + a K-component Gaussian-mixture head outputting, per
future step k and mode m: alpha[k,m], mu[k,m,3], log_sigma[k,m,3].
"""
from __future__ import annotations
import torch
import torch.nn as nn

from dataset import add_kinematic_features


class GMMTrajectoryPredictor(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = 128, T: int = 30,
                 K: int = 5, n_layers: int = 2, out_dim: int = 3):
        super().__init__()
        self.T = T
        self.K = K
        self.out_dim = out_dim
        self.encoder = nn.GRU(in_dim, hidden, num_layers=n_layers,
                              batch_first=True)
        self.ctx = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_alpha = nn.Linear(hidden, T * K)
        self.head_mu = nn.Linear(hidden, T * K * out_dim)
        self.head_logsig = nn.Linear(hidden, T * K * out_dim)
        self.logsig_min, self.logsig_max = -6.0, 3.0

    def forward(self, hist_pos: torch.Tensor):
        """hist_pos: [B,L,3] -> dict alpha[B,T,K], mu[B,T,K,3],
        log_sigma[B,T,K,3]."""
        B = hist_pos.shape[0]
        feats = add_kinematic_features(hist_pos)
        _, h = self.encoder(feats)
        z = self.ctx(h[-1])
        alpha = torch.softmax(self.head_alpha(z).view(B, self.T, self.K),
                              dim=-1)
        mu = self.head_mu(z).view(B, self.T, self.K, self.out_dim)
        log_sigma = torch.clamp(
            self.head_logsig(z).view(B, self.T, self.K, self.out_dim),
            self.logsig_min, self.logsig_max)
        return {"alpha": alpha, "mu": mu, "log_sigma": log_sigma}


def gmm_nll(pred: dict, target: torch.Tensor) -> torch.Tensor:
    """NLL of target [B,T,3] under the diagonal-Gaussian mixture."""
    alpha = pred["alpha"]
    mu = pred["mu"]
    log_sigma = pred["log_sigma"]
    tgt = target.unsqueeze(2)
    var = torch.exp(2.0 * log_sigma)
    log_prob = -0.5 * (((tgt - mu) ** 2) / var
                       + 2.0 * log_sigma
                       + torch.log(torch.tensor(2.0 * torch.pi)))
    log_prob = log_prob.sum(dim=-1)
    log_mix = torch.log(alpha + 1e-9) + log_prob
    return -torch.logsumexp(log_mix, dim=-1).mean()


def best_mode_ade(pred: dict, target: torch.Tensor):
    """Returns (minADE, meanADE) in target units."""
    mu = pred["mu"]
    alpha = pred["alpha"]
    tgt = target.unsqueeze(2)
    disp = torch.linalg.norm(mu - tgt, dim=-1)
    ade_per_mode = disp.mean(dim=1)
    min_ade = ade_per_mode.min(dim=1).values.mean()
    weighted_mu = (alpha.unsqueeze(-1) * mu).sum(dim=2)
    mean_ade = torch.linalg.norm(weighted_mu - target, dim=-1).mean()
    return min_ade, mean_ade


if __name__ == "__main__":
    net = GMMTrajectoryPredictor(T=30, K=5)
    print(f"params: {sum(p.numel() for p in net.parameters())/1e3:.1f}K")
    hist = torch.randn(8, 25, 3)
    out = net(hist)
    tgt = torch.randn(8, 30, 3)
    print("nll", gmm_nll(out, tgt).item())
    print("minADE/meanADE", [v.item() for v in best_mode_ade(out, tgt)])
