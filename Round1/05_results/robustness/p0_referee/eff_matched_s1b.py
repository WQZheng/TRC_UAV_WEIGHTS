"""[TODO-EFF] Per-episode CONTROL EFFORT (energy) of the matched pair, with the
paired difference and 95% CI, so that the "no detected increment" enumeration
covers all four run-time quantities (conflict, separation, EFFORT, yaw).

Matched control = Stage-1b (domain-adaptation only), exactly as in the paper and
in dev_matched_s1b.py / p2_mcnemar_dev.py; the appendix config-matched retrain
stage2_matched.pt is NOT the matched control here.

Effort metric = the manuscript's normalised control energy, identical to
eval_common.evaluate:
    thr_n = (u[:,0] - weight) / weight
    mom_n = u[:,1:4] / max_body_moment
    energy = sum_t ( thr_n^2 + ||mom_n||^2 )     (per episode, dimensionless)
Deploy rollout replicates p2_mcnemar_dev.rollout() verbatim (stage1b_domainadapt
vs stage2_final, alpha=0.1/Hp=15/a_max=20, d_sep=30, T=20, dt=0.2, eta_w=0.3,
batch=8, seed=12345, held-out 2500-3000) so it is a true paired design on the
SAME encounters.

Writes EFF_MATCHED_S1B.txt. Reports each arm's per-episode energy mean +/- SD and
the paired contrast Stage-2(final) - Stage-1b with a paired-t 95% CI (t_{n-1}).
If the CI includes 0, Stage-2 draws NO measurable extra control effort than the
matched control.
"""
from __future__ import annotations
import os
import torch
from math import sqrt

from seeding import set_seed
from config import GUAM_MAT
from params import DEFAULT_PARAMS
from dynamics import EVTOLDynamics
from wind import UrbanWindField
from predictor import GMMTrajectoryPredictor
from cbf_mpc import CBFMPCLayer
from safe_policy import SafePolicy
from guam_encounters import GUAMEncounters

OUT = "EFF_MATCHED_S1B.txt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
DSEP, N = 30.0, 200
ALPHA, HP, AMAX = 0.1, 15, 20.0
T, DT, ETA_W, BATCH = 20, 0.2, 0.3, 8


def w(s=""):
    with open(OUT, "a") as f:
        f.write(s + "\n")
    print(s, flush=True)


def load(p):
    n = GMMTrajectoryPredictor(T=30, K=5).double().to(DEV)
    n.load_state_dict(torch.load(p, map_location=DEV)); n.eval()
    return n


@torch.no_grad()
def rollout_energy(ckpt):
    """Deploy-config rollout; returns per-episode normalised control energy[N]."""
    set_seed(12345)
    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=12345)
    mpc = CBFMPCLayer(n_neighbors=1, horizon=HP, dt=DT, d_sep=DSEP,
                      alpha=ALPHA, a_max=AMAX)
    pol = SafePolicy(load(ckpt), mpc)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=DEV)
    wind = UrbanWindField(eta_w=ETA_W, dtype=DTYPE, device=DEV, seed=7)
    weight = DEFAULT_PARAMS.weight
    mmax = DEFAULT_PARAMS.max_body_moment
    energies = []
    for _ in range(N // BATCH):
        x0, nh, nf, _r, _f = gen.sample(BATCH, T, DEV)
        x = x0
        energy = torch.zeros(BATCH, dtype=DTYPE, device=DEV)
        for t in range(T):
            p0, v0 = x[:, 0:3], x[:, 3:6]
            tt = torch.arange(HP + 1, dtype=DTYPE, device=DEV) * DT
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            try:
                u, _ = pol(x, nh, nf[:, :, t, :], p_ref)
            except Exception:
                u = torch.zeros(BATCH, 4, dtype=DTYPE, device=DEV); u[:, 0] = weight
            # normalised control effort, identical to eval_common.evaluate
            thr_n = (u[:, 0] - weight) / weight
            mom_n = u[:, 1:4] / mmax
            energy = energy + thr_n ** 2 + (mom_n ** 2).sum(-1)
            x = dyn.step(x, u, wind.sample(p0), DT)
        energies.append(energy.cpu())
    return torch.cat(energies)[:N]


def paired_ci(diff, z_is_t=True):
    n = len(diff)
    m = float(diff.mean())
    sd = float(diff.std(unbiased=True))
    se = sd / sqrt(n)
    if z_is_t:
        try:
            from scipy import stats
            tcrit = float(stats.t.ppf(0.975, n - 1))
        except Exception:
            tcrit = 1.972
    else:
        tcrit = 1.96
    return m, sd, se, m - tcrit * se, m + tcrit * se, tcrit


def main():
    open(OUT, "w").close()
    w("[TODO-EFF] per-episode control EFFORT (normalised energy), matched pair")
    w("  n=%d seed=12345; effort = sum_t (thr_n^2 + ||mom_n||^2), dimensionless" % N)
    w("  deploy config: alpha=0.1 Hp=15 a_max=20 d_sep=30 T=20 dt=0.2 eta_w=0.3")
    w("  matched control = Stage-1b (domain-adaptation only)")
    w("")
    e1b = rollout_energy("stage1b_domainadapt.pt")
    e2 = rollout_energy("stage2_final.pt")

    def ms(v):
        return float(v.mean()), float(v.std(unbiased=True))

    m1b, s1b = ms(e1b)
    m2, s2 = ms(e2)
    w("  Stage-1b (matched control)  effort = %.4f +/- %.4f" % (m1b, s1b))
    w("  Stage-2  (final)            effort = %.4f +/- %.4f" % (m2, s2))
    w("")
    diff = e2 - e1b   # paired, per episode
    md, sdd, se, lo, hi, tc = paired_ci(diff)
    w("  paired contrast Stage-2(final) - Stage-1b (per-episode effort):")
    w("    mean paired diff = %+.4f   SD=%.4f  SE=%.4f  t_crit(0.975,df=%d)=%.3f"
      % (md, sdd, se, N - 1, tc))
    w("    95%% CI = [%+.4f, %+.4f]" % (lo, hi))
    if lo <= 0 <= hi:
        w("    -> CI includes 0: Stage-2 draws NO measurable extra control effort")
        w("       than the matched control Stage-1b.")
    elif lo > 0:
        w("    -> CI excludes 0, positive: Stage-2 uses MORE effort.")
    else:
        w("    -> CI excludes 0, negative: Stage-2 uses LESS effort.")


if __name__ == "__main__":
    main()
