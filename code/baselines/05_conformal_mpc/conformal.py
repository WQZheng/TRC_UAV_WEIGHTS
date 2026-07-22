"""Split-conformal calibration of the predictor's neighbour-position error,
used to inflate the CBF-MPC separation margin with a finite-sample coverage
guarantee.

Method (standard split conformal prediction):
  1. On a CALIBRATION set of encounters DISJOINT from both training
     (0..2500) and the held-out eval set (2500..3000) -- we use
     range(2000,2500) -- run the FROZEN predictor and collect the
     nonconformity scores s = || mu_hat(k) - true(k) ||  (metres) over all
     look-ahead steps k within the planning horizon.
  2. The conformal radius at miscoverage delta is the
     ceil((n+1)(1-delta))/n empirical quantile of {s}. With prob >= 1-delta
     the true neighbour position lies within r_conf of the prediction
     (marginal coverage), so enforcing separation d_sep + r_conf gives a
     calibrated safety buffer.

This returns a single scalar r_conf (metres). Conformal-MPC then runs the
unchanged CBF-MPC with d_sep := d_sep + r_conf.
"""
from __future__ import annotations
import sys
import math
import torch

sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")
from config import GUAM_MAT                       # noqa: E402
from guam_encounters import GUAMEncounters        # noqa: E402
from seeding import set_seed                      # noqa: E402

SCALE = 100.0
CALIB_RANGE = range(2000, 2500)   # disjoint from train(0-2500 used)/eval(2500-3000)


@torch.no_grad()
def conformal_radius(predictor, delta=0.1, n_calib=200, horizon=15,
                     device="cuda", seed=12345):
    """Return the split-conformal radius r_conf (metres) at miscoverage delta."""
    set_seed(seed)
    gen = GUAMEncounters(GUAM_MAT, CALIB_RANGE, seed=seed)
    scores = []
    batch = 8
    T = 20
    for _ in range(max(1, n_calib // batch)):
        x0, nh, nf, _ref, nfut = gen.sample(batch, T, device)
        out = predictor(nh.reshape(batch, 25, 3))
        mean_traj = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)   # [B,30,3]
        h = min(horizon, nfut.shape[2])
        err = torch.linalg.norm(
            mean_traj[:, :h] - nfut[:, 0, :h, :], dim=-1) * SCALE       # [B,h]
        scores.append(err.reshape(-1).cpu())
    s = torch.cat(scores)
    n = s.numel()
    # finite-sample conformal quantile level
    q_level = min(1.0, math.ceil((n + 1) * (1 - delta)) / n)
    r_conf = torch.quantile(s, q_level).item()
    return r_conf, {"n_scores": int(n), "delta": delta, "q_level": q_level,
                    "mean_score": float(s.mean()), "max_score": float(s.max())}
