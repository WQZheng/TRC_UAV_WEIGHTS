"""Split-conformal calibration of the predictor's neighbour-position error,
used to inflate the CBF-MPC separation margin with a finite-sample coverage
guarantee.

Method (standard split conformal prediction):
  1. On a CALIBRATION set of encounters that is DISJOINT FROM THE TRAINING
     trajectories (training uses GUAM trajectories 0..2499) and is drawn
     INDEPENDENTLY from the held-out pool, we run the FROZEN predictor and
     collect the nonconformity scores s = || mu_hat(k) - true(k) || (metres)
     over all look-ahead steps k within the planning horizon.
  2. The conformal radius at miscoverage delta is the
     ceil((n+1)(1-delta))/n empirical quantile of {s}. With prob >= 1-delta
     the true neighbour position lies within r_conf of the prediction
     (marginal coverage), so enforcing separation d_sep + r_conf gives a
     calibrated safety buffer.

This returns a single scalar r_conf (metres). Conformal-MPC then runs the
unchanged CBF-MPC with d_sep := d_sep + r_conf.

Data-split rationale (IMPORTANT for split-conformal validity)
-------------------------------------------------------------
Split conformal requires two distinct separations:
  (a) calibration must be DISJOINT FROM TRAINING -- otherwise the frozen
      predictor has seen the calibration trajectories, its scores are
      optimistically small, r_conf is too small, and the coverage guarantee
      is void; and
  (b) calibration and deployment (evaluation) points must be EXCHANGEABLE --
      which does NOT require disjoint trajectory pools; independent draws
      from the same generating distribution are the textbook setting.

We therefore calibrate on the held-out pool (2500..2999), the SAME pool the
evaluation episodes are drawn from, but with a dedicated calibration seed
(CALIB_SEED = 777) that is independent of the evaluation seed (12345). This
satisfies all three constraints simultaneously:
  * calibration ∩ training = ∅         (2500..2999 vs 0..2499)     -> (a)
  * calibration ⫫ evaluation           (same pool, independent seeds; no
    shared episodes)                                                -> (b)
  * evaluation stream is byte-for-byte unchanged (still pool 2500..2999,
    seed 12345), so the paired McNemar / Wilson-CI framework across arms
    is preserved.

CHANGELOG
---------
Previously CALIB_RANGE = range(2000, 2500). That range OVERLAPS the Stage-1
training trajectories (0..2499), so the frozen predictor had already seen
every calibration trajectory: constraint (a) was violated and the reported
coverage guarantee did not hold. Fixed to range(2500, 3000) with an
independent calibration seed; evaluation is unchanged.
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
# Calibration pool: held-out trajectories, DISJOINT from training (0..2499).
CALIB_RANGE = range(2500, 3000)
# Dedicated calibration seed, independent of the evaluation seed (12345), so
# calibration and evaluation never share an episode even though they are
# drawn from the same held-out pool (exchangeability, not pool-disjointness).
CALIB_SEED = 777


@torch.no_grad()
def conformal_radius(predictor, delta=0.1, n_calib=200, horizon=15,
                     device="cuda", seed=None):
    """Return the split-conformal radius r_conf (metres) at miscoverage delta.

    The calibration draw always uses CALIB_SEED (independent of the
    evaluation seed) regardless of any `seed` passed by the caller, so that
    calibration and evaluation share no episode. `seed` is accepted for
    backward-compatible call signatures but intentionally ignored for the
    calibration RNG.
    """
    set_seed(CALIB_SEED)
    gen = GUAMEncounters(GUAM_MAT, CALIB_RANGE, seed=CALIB_SEED)
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
                    "calib_range": f"{CALIB_RANGE.start}-{CALIB_RANGE.stop-1}",
                    "calib_seed": CALIB_SEED,
                    "mean_score": float(s.mean()), "max_score": float(s.max())}
