"""Trajectory windowing dataset for Stage-1 predictor pretraining.

GUAM own-ship trajectories are long city-to-city cruises stored as
Bernstein paths with NO intrinsic time scale. We therefore (1) sample
each Bernstein path densely, (2) resample it to equal arc-length spacing
of step_m = cruise_speed * dt metres, giving each sample a physical
dt-second spacing, then (3) slice into (history L, future T) windows
recentred to the last observed position. Positions are in metres.

Optional yaw augmentation rotates each window about the vertical axis so
the predictor becomes orientation-invariant (needed so it generalises to
neighbours approaching from arbitrary headings).

Each sample:
    hist : (L, 3)  observed past positions (recentred, scaled)
    fut  : (T, 3)  future positions (recentred, scaled)
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

from guam_data import GUAMTrajectories, resample_by_arclength


class GUAMWindowDataset(Dataset):
    def __init__(self, mat_path: str, traj_indices, L: int = 25, T: int = 30,
                 dt: float = 0.2, cruise_speed: float = 50.0,
                 samples_per_seg: int = 200, stride: int = 5,
                 max_windows_per_traj: int = 40, dtype=torch.float32,
                 normalize_scale: float = 100.0, yaw_augment: bool = False,
                 n_yaw: int = 4, seed: int = 0):
        self.L, self.T = L, T
        self.dtype = dtype
        self.scale = normalize_scale
        self.step_m = cruise_speed * dt          # metres per timestep
        self.yaw_augment = yaw_augment
        self.n_yaw = n_yaw
        self._rng = np.random.default_rng(seed)
        loader = GUAMTrajectories(mat_path)
        self.samples = []

        win = L + T
        for idx in traj_indices:
            try:
                tr = loader.load_trajectory(idx, samples_per_seg=samples_per_seg)
            except Exception:
                continue
            tr = resample_by_arclength(tr, step_m=self.step_m)
            if tr.shape[0] < win + 1:
                continue
            count = 0
            for s in range(0, tr.shape[0] - win, stride):
                seg = tr[s:s + win]                       # (win, 3)
                anchor = seg[L - 1]
                seg = (seg - anchor) / self.scale         # recentre + scale
                hist = seg[:L]
                fut = seg[L:L + T]
                yaws = [0.0]
                if self.yaw_augment:
                    yaws = self._rng.uniform(0, 2 * np.pi, size=self.n_yaw)
                for yaw in yaws:
                    c, s2 = np.cos(yaw), np.sin(yaw)
                    Rz = np.array([[c, -s2, 0], [s2, c, 0], [0, 0, 1]])
                    h = hist @ Rz.T
                    f = fut @ Rz.T
                    self.samples.append((h.astype(np.float32),
                                         f.astype(np.float32)))
                count += 1
                if count >= max_windows_per_traj:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        hist, fut = self.samples[i]
        return torch.from_numpy(hist), torch.from_numpy(fut)


def add_kinematic_features(hist: torch.Tensor) -> torch.Tensor:
    """Append finite-difference velocity to positions: [B,L,3]->[B,L,6]."""
    vel = torch.zeros_like(hist)
    vel[:, 1:, :] = hist[:, 1:, :] - hist[:, :-1, :]
    return torch.cat([hist, vel], dim=-1)


if __name__ == "__main__":
    from config import GUAM_MAT
    ds = GUAMWindowDataset(GUAM_MAT, traj_indices=range(50), L=25, T=30)
    print(f"#windows from 50 trajectories = {len(ds)}")
    h, f = ds[0]
    print(f"hist {tuple(h.shape)}, fut {tuple(f.shape)}")
    print(f"hist last (should be ~0): {h[-1].tolist()}")
    fut_span_m = (f.max(0).values - f.min(0).values) * 100.0
    print(f"fut span (m): {fut_span_m.tolist()}")
