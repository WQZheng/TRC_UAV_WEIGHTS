"""Loader for NASA GUAM Challenge-Problem own-ship trajectories.

Data_Set_1.mat (MATLAB v7.3 / HDF5) contains:
  own_traj      : (6, 3000) cell array of references; column = one of 3000
                  trajectories, each split into up to 6 Bernstein segments.
  own_traj_orig : the un-modified flight-plan version (we use own_traj).

Each segment is a (3, N) array of Bernstein/Bezier control points
(rows = x, y, z position in feet; columns = control points). We sample
each segment with the Bernstein basis, concatenate segments, convert
ft->m, and return an (T, 3) trajectory.

`resample_by_arclength` then gives the path a physical time scale.
"""
from __future__ import annotations
import math
import numpy as np
import h5py

FT_TO_M = 0.3048


def _bernstein_sample(ctrl: np.ndarray, n_samples: int) -> np.ndarray:
    """Sample a Bernstein curve from (3, N) control points -> (n_samples, 3)."""
    dim, n_ctrl = ctrl.shape
    degree = n_ctrl - 1
    s = np.linspace(0.0, 1.0, n_samples)
    out = np.zeros((n_samples, dim), dtype=np.float64)
    for j in range(n_ctrl):
        binom = math.comb(degree, j)
        basis = binom * (s ** j) * ((1.0 - s) ** (degree - j))
        out += np.outer(basis, ctrl[:, j])
    return out


class GUAMTrajectories:
    """Lazy reader over the 3000 GUAM own-ship trajectories."""

    def __init__(self, mat_path: str, which: str = "own_traj"):
        self.mat_path = mat_path
        self.which = which
        with h5py.File(mat_path, "r") as f:
            self._shape = f[which].shape  # (6 segments, 3000 trajs)
        self.n_segments, self.n_traj = self._shape

    def __len__(self) -> int:
        return self.n_traj

    def load_trajectory(self, idx: int, samples_per_seg: int = 40) -> np.ndarray:
        """Return one trajectory as an (T, 3) array of positions in metres."""
        if not (0 <= idx < self.n_traj):
            raise IndexError(idx)
        pieces = []
        with h5py.File(self.mat_path, "r") as f:
            refs = f[self.which]
            for seg in range(self.n_segments):
                ref = refs[seg, idx]
                ctrl = np.asarray(f[ref][()], dtype=np.float64)
                if ctrl.ndim != 2 or ctrl.shape[0] != 3 or ctrl.shape[1] < 2:
                    continue
                pieces.append(_bernstein_sample(ctrl, samples_per_seg))
        if not pieces:
            raise ValueError(f"trajectory {idx} has no valid segments")
        return np.concatenate(pieces, axis=0) * FT_TO_M

    def load_batch(self, indices, samples_per_seg: int = 40):
        return [self.load_trajectory(i, samples_per_seg) for i in indices]


def resample_by_arclength(traj, step_m):
    """Resample an (N,3) path to equal arc-length spacing of step_m metres.

    GUAM Bernstein paths have NO intrinsic time scale and unevenly spaced
    parameter points; resampling to equal arc-length spacing gives a
    physical time scale: with cruise speed v and timestep dt, choose
    step_m = v*dt so each output sample is dt apart in time. Returns (M,3).
    """
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < step_m:
        return traj.copy()
    n_out = int(total // step_m) + 1
    s_new = np.arange(n_out) * step_m
    out = np.zeros((n_out, 3), dtype=traj.dtype)
    for d in range(3):
        out[:, d] = np.interp(s_new, s, traj[:, d])
    return out


if __name__ == "__main__":
    from config import GUAM_MAT
    g = GUAMTrajectories(GUAM_MAT)
    print(f"#trajectories = {len(g)}, segments/traj = {g.n_segments}")
    for i in [0, 1, 2999]:
        tr = g.load_trajectory(i)
        print(f"traj {i}: shape={tr.shape}, "
              f"bbox span (m)={(tr.max(0) - tr.min(0)).round(1)}")