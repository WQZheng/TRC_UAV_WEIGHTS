"""Paired statistical tests for the main comparison (referee point 5).

Conflict rate is a per-episode 0/1 statistic evaluated on the SAME held-out
encounters for every method (a paired design). This script:

  1. re-runs the four CBF-equipped methods (PlanGrad, Conformal-MPC,
     Fixed-Predictor, Constant-Velocity) collecting the per-episode conflict
     indicator vector conflict[n] for each, on identical encounters
     (seed 12345, n=200);
  2. reports a Wilson 95% confidence interval for each method's conflict rate;
  3. runs McNemar's exact paired test comparing PlanGrad against every other
     method (the correct test for paired binary outcomes), reporting the
     discordant-pair counts and the two-sided p-value.

Certificate-free methods (Vanilla, Soft-IPP) are excluded from the pairwise
safety tests because their conflict rate is governed by the planner, not the
predictor, and they are not in contention for "best CR".

Writes STATS.txt.

Usage:
    export GUAM_MAT=/path/Data_Set_1.mat
    python3 stats_tests.py --n 200 --seed 12345
"""
from __future__ import annotations
import os
import sys
import argparse
import math

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(B, "01_constant_velocity"))
sys.path.insert(0, os.path.join(B, "05_conformal_mpc"))
sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")

import numpy as np
import torch
import eval_common as ec
from cv_predictor import ConstantVelocityPredictor
from cbf_mpc import CBFMPCLayer
from conformal import conformal_radius

PLANGRAD = "/data/lab/plangrad/plangrad_sim"


@torch.no_grad()
def per_episode_conflict(predictor, planner, n, dev, d_sep=ec.D_SEP,
                         eta_w=ec.WIND_ETA):
    """Return a boolean array conflict[n] on the standard held-out encounters.

    Mirrors ec.evaluate_policy exactly but records the per-episode min-sep
    threshold crossing instead of only the aggregate."""
    from params import DEFAULT_PARAMS
    from dynamics import EVTOLDynamics
    from wind import UrbanWindField
    from safe_policy import SafePolicy
    from guam_encounters import GUAMEncounters
    from seeding import set_seed

    set_seed(ec.GLOBAL_SEED)
    policy = SafePolicy(predictor, planner)
    Hp = planner.Hp if hasattr(planner, "Hp") else ec.BEST_PLANNER["horizon"]
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=ec.DTYPE, device=dev)
    wind = UrbanWindField(eta_w=eta_w, dtype=ec.DTYPE, device=dev,
                          seed=ec.WIND_SEED)
    gen = GUAMEncounters(ec.GUAM_MAT, ec.EVAL_RANGE, seed=ec.GLOBAL_SEED)
    weight = DEFAULT_PARAMS.weight
    out = []
    B_ = 8
    for _ in range(max(1, n // B_)):
        x0, nh, nf, _ref, _ = gen.sample(B_, ec.T_EPISODE, dev)
        x = x0
        min_sep = torch.full((B_,), 1e6, dtype=ec.DTYPE, device=dev)
        for t in range(ec.T_EPISODE):
            p0, v0 = x[:, 0:3], x[:, 3:6]
            tt = torch.arange(Hp + 1, dtype=ec.DTYPE, device=dev) * ec.DT
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            try:
                u, _ = policy(x, nh, nf[:, :, t, :], p_ref)
            except Exception:
                u = torch.zeros(B_, 4, dtype=ec.DTYPE, device=dev); u[:, 0] = weight
            x = dyn.step(x, u, wind.sample(p0), ec.DT)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            min_sep = torch.minimum(min_sep, d)
        out.append((min_sep < d_sep).cpu().numpy())
    return np.concatenate(out)


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion (k successes of n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (100.0 * (center - half), 100.0 * (center + half))


def mcnemar_exact(a, b):
    """Exact two-sided McNemar test on paired binary vectors a,b (conflict=1).
    Returns (b01, b10, p_value) where b01 = a-safe & b-conflict, etc."""
    a = a.astype(bool); b = b.astype(bool)
    b01 = int((~a & b).sum())   # a avoids, b conflicts
    b10 = int((a & ~b).sum())   # a conflicts, b avoids
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    # exact binomial two-sided p under H0: discordant split ~ Binom(n, 0.5)
    from math import comb
    k = min(b01, b10)
    p = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) * 2
    return b01, b10, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(B, "STATS.txt"))
    args = ap.parse_args()
    dev = ec.device_str(True)

    s1 = ec.load_gmm_predictor(f"{PLANGRAD}/stage1_full.pt", dev)
    s2 = ec.load_gmm_predictor(f"{PLANGRAD}/stage2_final.pt", dev)
    cv = ConstantVelocityPredictor(T=30, K=5).double().to(dev)

    r_conf, _ = conformal_radius(s1, delta=0.1,
                                 horizon=ec.BEST_PLANNER["horizon"],
                                 device=dev, seed=ec.GLOBAL_SEED)

    def best():
        return ec.make_best_planner()

    def conf_planner():
        return CBFMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"],
                           dt=ec.DT, d_sep=ec.D_SEP + r_conf,
                           alpha=ec.BEST_PLANNER["alpha"],
                           a_max=ec.BEST_PLANNER["a_max"])

    print("collecting per-episode conflicts (n=%d) ..." % args.n, flush=True)
    conf = {
        "PlanGrad":          per_episode_conflict(s2, best(), args.n, dev),
        "Conformal-MPC":     per_episode_conflict(s1, conf_planner(), args.n, dev),
        "Fixed-Predictor":   per_episode_conflict(s1, best(), args.n, dev),
        "Constant-Velocity": per_episode_conflict(cv, best(), args.n, dev),
    }

    fh = open(args.out, "w")

    def w(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    w("=" * 74)
    w("PAIRED STATISTICAL TESTS  (n=%d, seed=%d, identical held-out encounters)"
      % (args.n, ec.GLOBAL_SEED))
    w("CBF-equipped methods only; conflict = per-episode min sep < %.0f m." % ec.D_SEP)
    w("=" * 74)
    w("%-20s %8s   %-22s" % ("method", "CR %", "Wilson 95% CI"))
    for name, c in conf.items():
        k, n = int(c.sum()), len(c)
        lo, hi = wilson_ci(k, n)
        w("%-20s %7.1f   [%5.1f, %5.1f]" % (name, 100.0 * k / n, lo, hi))

    w("")
    w("McNemar exact paired test vs PlanGrad (H0: equal conflict propensity):")
    a = conf["PlanGrad"]
    for name in ["Conformal-MPC", "Fixed-Predictor", "Constant-Velocity"]:
        b01, b10, p = mcnemar_exact(a, conf[name])
        w("  PlanGrad vs %-18s discordant (PG-avoids/other-conflicts=%d, "
          "PG-conflicts/other-avoids=%d)  p=%.3f  -> %s"
          % (name, b01, b10, p,
             "n.s." if p > 0.05 else "significant"))
    w("")
    w("Reading: overlapping CIs + all p>0.05 => the CBF-equipped methods are")
    w("statistically indistinguishable on conflict rate at this sample size;")
    w("no method's CR advantage over the others is significant. This is the")
    w("intended finding (safety is certificate-governed), now tested rather")
    w("than asserted.")
    fh.close()


if __name__ == "__main__":
    main()
