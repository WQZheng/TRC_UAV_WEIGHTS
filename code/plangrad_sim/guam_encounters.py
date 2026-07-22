"""GUAM-derived encounter generator (in-distribution, MEDIUM difficulty).

Builds encounters from REAL GUAM trajectory segments so the predictor
stays in-distribution. Difficulty is tuned to "medium": conflicts are
geometrically possible but AVOIDABLE with a reasonable planner, so that
(a) the baseline conflict rate is not pinned at 100%, and (b) prediction
quality has room to further improve safety -- the regime in which the
decoupling of ADE from operational safety can appear.

Per encounter:
  * ego  = a real GUAM window (its reference path = its own GUAM motion)
  * neighbour = another GUAM window, slowed, yaw-rotated and translated so
    it approaches the ego mid-rollout with a controlled closest-approach
    offset (so avoidance is feasible).

sample() returns:
  x0          [B,12], neigh_hist [B,1,L,3] (recentred+scaled),
  neigh_full  [B,1,T+He+1,3] (absolute), ego_ref [B,T+He+1,3] (absolute),
  neigh_fut   [B,1,30,3] (recentred+scaled ground-truth, for the ADE anchor)
"""
from __future__ import annotations
import numpy as np
import torch

from guam_data import GUAMTrajectories, resample_by_arclength

DTYPE = torch.float64
SCALE = 100.0


class GUAMEncounters:
    def __init__(self, mat_path, traj_indices, L=25, dt=0.2,
                 cruise_speed=50.0, seed=0,
                 neigh_speed_scale=0.45,
                 approach_offset=(18.0, 32.0),
                 extra_horizon=12):
        self.L = L
        self.dt = dt
        self.step_m = cruise_speed * dt
        self.neigh_speed_scale = neigh_speed_scale
        self.approach_offset = approach_offset
        self.extra_horizon = extra_horizon
        self.loader = GUAMTrajectories(mat_path)
        self.rng = np.random.default_rng(seed)
        self.pool = []
        for idx in traj_indices:
            try:
                tr = self.loader.load_trajectory(idx, samples_per_seg=200)
            except Exception:
                continue
            rs = resample_by_arclength(tr, step_m=self.step_m)
            if rs.shape[0] > L + 80:
                self.pool.append(rs.astype(np.float64))
        if not self.pool:
            raise RuntimeError("no usable GUAM trajectories in pool")

    def _rand_window(self, length):
        tr = self.pool[self.rng.integers(len(self.pool))]
        s = self.rng.integers(0, tr.shape[0] - length)
        return tr[s:s + length].copy()

    def sample(self, B, T, device):
        L = self.L
        He = self.extra_horizon
        need = L + T + He + 1
        x0 = torch.zeros(B, 12, dtype=DTYPE)
        neigh_hist = torch.zeros(B, 1, L, 3, dtype=DTYPE)
        neigh_full = torch.zeros(B, 1, T + He + 1, 3, dtype=DTYPE)
        ego_ref = torch.zeros(B, T + He + 1, 3, dtype=DTYPE)
        neigh_fut = torch.zeros(B, 1, 30, 3, dtype=DTYPE)

        for b in range(B):
            ego = self._rand_window(need)
            nei = self._rand_window(need)

            ego_now = ego[L - 1]
            ego_path = ego[L - 1:]
            ego_ref[b, :, :] = torch.tensor(ego_path[:T + He + 1])
            x0[b, 0:3] = torch.tensor(ego_now)
            v0 = (ego[L - 1] - ego[L - 2]) / self.dt
            x0[b, 3:6] = torch.tensor(v0)

            nei_now = nei[L - 1]
            nei_rel = (nei - nei_now) * self.neigh_speed_scale
            theta = self.rng.uniform(0.0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            nei_rot = nei_rel @ Rz.T

            mid_idx = min(T // 2, ego_path.shape[0] - 1)
            mid = ego_path[mid_idx]
            nei_mid_idx = L - 1 + T // 2
            nei_mid_rel = nei_rot[nei_mid_idx] - nei_rot[L - 1]
            off_mag = self.rng.uniform(*self.approach_offset)
            off_dir = self.rng.normal(size=3); off_dir[2] = 0.0
            off_dir = off_dir / (np.linalg.norm(off_dir) + 1e-9)
            offset = mid - nei_mid_rel + off_dir * off_mag
            offset[2] = 0.0
            nei_abs = nei_rot - nei_rot[L - 1] + offset

            neigh_full[b, 0, :, :] = torch.tensor(
                nei_abs[L - 1:L - 1 + T + He + 1])
            hist = nei_abs[:L] - nei_abs[L - 1]
            neigh_hist[b, 0, :, :] = torch.tensor(hist / SCALE)
            fut = (nei_abs[L:L + 30] - nei_abs[L - 1]) / SCALE
            neigh_fut[b, 0, :fut.shape[0], :] = torch.tensor(fut)

        return (x0.to(device), neigh_hist.to(device), neigh_full.to(device),
                ego_ref.to(device), neigh_fut.to(device))


if __name__ == "__main__":
    from config import GUAM_MAT
    enc = GUAMEncounters(GUAM_MAT, range(100), seed=1)
    print(f"pool size = {len(enc.pool)}")
    x0, nh, nf, ref, nfut = enc.sample(8, 20, "cpu")
    d = torch.linalg.norm(ref[:, :nf.shape[2], :] - nf[:, 0], dim=-1)
    print("min ref-vs-neighbour dist (m):",
          d.min(dim=1).values.round(decimals=1).tolist())
