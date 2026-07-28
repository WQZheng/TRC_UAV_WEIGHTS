"""6 s lead-time anomaly diagnostic (referee point 2).

The lead-time sweep shows a NON-MONOTONE Stage-1 conflict-rate profile:
    3 s -> 1.5% | 6 s -> 12.0% | 10 s -> 0.5% | 20 s -> 0.0%
The manuscript's original "forecast error compounds over the longer window"
reading predicts 10 s/20 s should be WORSE, contradicting the data. This script
tests the competing hypothesis that the 6 s peak is an interaction between the
detection horizon and the point in the episode at which the FIRST avoidance
decision must be committed:

  For each detection horizon we roll out the SAME encounters under the Stage-1
  predictor and record, per episode,
    (i)  the step t* at which the CBF constraint first becomes active
         (first commanded lateral/vertical acceleration above a threshold), and
    (ii) the Stage-1 predictor's HP-window displacement error at t*
         (how wrong the forecast the planner acts on actually is).

  If the 6 s peak is caused by the first avoidance falling on a
  high-error part of the forecast, mean error-at-first-avoidance should peak at
  6 s and be lower at 3/10/20 s, tracking the CR profile. If instead the error
  is flat across horizons, the peak has another cause and the manuscript should
  not claim the mechanism.

Usage:
    export GUAM_MAT=/data/lab/plangrad/GUAM/Challenge_Problems/Data_Set_1.mat
    python3 diag_leadtime_6s.py --n 200 --seed 12345 \
        --stage1 stage1_full.pt --horizons 3,6,10,20 --out LEADTIME_DIAG.txt
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from config import GUAM_MAT
from seeding import set_seed
from predictor import GMMTrajectoryPredictor
from fast_cbf_mpc import FastCBFMPC
from dynamics import EVTOLDynamics
from params import DEFAULT_PARAMS
from wind import UrbanWindField
from eval_leadtime import (LeadTimeEncounters, predict_meantraj, SCALE, DTYPE,
                           HP, DT, D_SEP, AMAX, ALPHA)


def load_pred(path, dev):
    net = GMMTrajectoryPredictor(T=30, K=5).double().to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()
    return net


@torch.no_grad()
def diagnose(gen, pred, dev, n, T):
    """Return (mean_err_at_first_avoid_m, frac_episodes_with_avoid,
    mean_first_avoid_step)."""
    planner = FastCBFMPC(n_neighbors=1, horizon=HP, dt=DT, d_sep=D_SEP,
                         a_max=AMAX, alpha=ALPHA)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=dev)
    wind = UrbanWindField(eta_w=0.3, dtype=DTYPE, device=dev, seed=7)
    weight = DEFAULT_PARAMS.weight
    errs, steps, navoid = [], [], 0
    B = 8
    for _ in range(n // B):
        x0, nh, nf, _ref, _ = gen.sample(B, T, dev)
        x = x0
        first_avoid = [-1] * B
        avoid_err = [np.nan] * B
        for t in range(T):
            p0 = x[:, 0:3]; v0 = x[:, 3:6]
            tt = torch.arange(HP + 1, dtype=DTYPE, device=dev) * DT
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            u = torch.zeros(B, 4, dtype=DTYPE, device=dev); u[:, 0] = weight
            for b in range(B):
                npred = predict_meantraj(pred, nh[b, 0], nf[b, 0, t, :])[None]
                a0 = planner.solve_np(p0[b].cpu().numpy(), v0[b].cpu().numpy(),
                                      p_ref[b].cpu().numpy(), npred)
                if a0 is None:
                    continue
                # lateral/vertical accel magnitude = avoidance signal
                lat = float(np.hypot(a0[0], a0[1]))
                if first_avoid[b] < 0 and lat > 1.0:
                    first_avoid[b] = t
                    # Stage-1 HP-window forecast error vs true neighbour future
                    hpix = [min(t + k, nf.shape[2] - 1) for k in range(HP + 1)]
                    true = nf[b, 0, hpix, :].cpu().numpy()
                    avoid_err[b] = float(
                        np.linalg.norm(npred[0] - true, axis=-1).mean())
                a0 = torch.tensor(a0, dtype=DTYPE, device=dev)
                m = DEFAULT_PARAMS.mass; g = DEFAULT_PARAMS.g
                f_des = m * a0.clone(); f_des[2] += m * g
                thrust = torch.linalg.norm(f_des).clamp_min(1.0)
                ax = f_des[0] / thrust; ay = f_des[1] / thrust
                tilt = 0.45
                roll = torch.clamp(-ay, -tilt, tilt)
                pitch = torch.clamp(ax, -tilt, tilt)
                eta = x[b, 6:9]; om = x[b, 9:12]
                I = torch.tensor(DEFAULT_PARAMS.inertia_diag, dtype=DTYPE, device=dev)
                mom = torch.stack([(2.0 * (roll - eta[0]) - 1.5 * om[0]) * I[0],
                                   (2.0 * (pitch - eta[1]) - 1.5 * om[1]) * I[1],
                                   (-1.5 * om[2]) * I[2]])
                mmax = DEFAULT_PARAMS.max_body_moment
                mom = torch.clamp(mom, -mmax, mmax)
                u[b, 0] = thrust; u[b, 1:4] = mom
            x = dyn.step(x, u, wind.sample(x[:, 0:3]), DT)
        for b in range(B):
            if first_avoid[b] >= 0:
                navoid += 1
                steps.append(first_avoid[b])
                errs.append(avoid_err[b])
    if not errs:
        return float("nan"), 0.0, float("nan")
    return (float(np.nanmean(errs)) * SCALE, navoid / n,
            float(np.mean(steps)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--stage1", default="stage1_full.pt")
    ap.add_argument("--horizons", default="3,6,10,20")
    ap.add_argument("--out", default="LEADTIME_DIAG.txt")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    s1 = load_pred(args.stage1, dev)
    horizons = [float(x) for x in args.horizons.split(",")]

    lines = []
    def w(s=""):
        print(s); lines.append(s)
    w("6 s lead-time anomaly diagnostic (Stage-1 predictor)")
    w("n=%d seed=%d  HP=%d (%.1fs consumed window)" % (args.n, args.seed, HP, (HP)*DT))
    w("Testing whether the Stage-1 CR peak at 6 s tracks the forecast error")
    w("the planner acts on at the FIRST avoidance step.")
    w("")
    w("horizon(s) | ADE@first-avoid(m) | frac episodes avoiding | mean first-avoid step")
    for h in horizons:
        tcpa = int(round(h / DT))
        T = tcpa + 4
        gen = LeadTimeEncounters(GUAM_MAT, range(2500, 3000), seed=args.seed,
                                 t_cpa=tcpa)
        set_seed(args.seed)
        err, frac, mstep = diagnose(gen, s1, dev, args.n, T)
        w("%9.0f | %17.2f | %21.2f | %20.1f" % (h, err, frac, mstep))
    w("")
    w("READING: if ADE@first-avoid peaks at 6 s and is lower at 3/10/20 s,")
    w("the CR peak is an avoidance-timing/forecast-error interaction, not")
    w("monotone error compounding. If ADE@first-avoid is flat, the peak has")
    w("another cause and the manuscript must not attribute it to this mechanism.")
    open(args.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
