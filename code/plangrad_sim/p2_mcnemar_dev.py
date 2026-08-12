"""[TODO-1BP] + [TODO-DEV]: deploy-config paired McNemar (Stage-1b vs Stage-2)
with paired-difference CIs, and route (cross-track) deviation of the matched
pair.

Deploy / main-table config (reproduces BEST.txt): CBF-MPC alpha=0.1, Hp=15,
a_max=20, d_sep=30, T=20, dt=0.2, eta_w=0.3, n=200, seed=12345, pool 2500-3000.
Evaluation-only; nothing retrained. Main model stays stage2_final.pt;
stage2_matched.pt is the appendix matched control.

[TODO-1BP] The main-table STATS only lists the four common-planner arms
(PlanGrad/Conformal/Fixed/CV); Stage-1b's DEPLOY-config paired McNemar vs
Stage-2 (discordant b,c read from per-episode conflicts, NOT inferred from the
net CR difference) was never in the main text. We collect per-episode min-sep
for Stage-1, Stage-1b, Stage-2 on the identical stream and emit:
  * per-arm CR + Wilson 95% CI,
  * the S1b-vs-S2 (and S1-vs-S2) paired 2x2 + exact McNemar p,
  * paired-difference 95% CI on per-episode min-sep for each comparison
    (a metric-level effect size to accompany the binary McNemar).

[TODO-DEV] Route deviation: at each step the straight-line reference is
p_ref(k) = p0 + v0*k*dt (the same reference the planner tracks). Cross-track
deviation = distance from the realized position to that reference LINE
(perpendicular component), capturing lateral detour / "yaw to buy performance".
We report mean +/- SD of the per-episode max cross-track deviation and of the
per-episode mean cross-track deviation, for Stage-1, Stage-2, and the matched
control Stage-2(matched). Sharp motivation: does Stage-2 buy its behaviour with
a larger lateral excursion than the matched control?

Writes P2_MCNEMAR_DEV.txt.
"""
from __future__ import annotations
import torch
from math import comb, sqrt

from seeding import set_seed
from config import GUAM_MAT
from params import DEFAULT_PARAMS
from dynamics import EVTOLDynamics
from wind import UrbanWindField
from predictor import GMMTrajectoryPredictor
from cbf_mpc import CBFMPCLayer
from safe_policy import SafePolicy
from guam_encounters import GUAMEncounters

OUT = "P2_MCNEMAR_DEV.txt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
DSEP, N = 30.0, 200
ALPHA, HP, AMAX = 0.1, 15, 20.0
T, DT, ETA_W, BATCH = 20, 0.2, 0.3, 8


def w(s):
    with open(OUT, "a") as f:
        f.write(s + "\n")
    print(s, flush=True)


def load(p):
    n = GMMTrajectoryPredictor(T=30, K=5).double().to(DEV)
    n.load_state_dict(torch.load(p, map_location=DEV)); n.eval()
    return n


@torch.no_grad()
def rollout(ckpt):
    """Deploy-config rollout. Returns per-episode:
       min_sep[N], max_xtrack[N], mean_xtrack[N] (metres)."""
    set_seed(12345)
    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=12345)
    mpc = CBFMPCLayer(n_neighbors=1, horizon=HP, dt=DT, d_sep=DSEP,
                      alpha=ALPHA, a_max=AMAX)
    pol = SafePolicy(load(ckpt), mpc)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=DEV)
    wind = UrbanWindField(eta_w=ETA_W, dtype=DTYPE, device=DEV, seed=7)

    mins, xmax, xmean = [], [], []
    for _ in range(N // BATCH):
        x0, nh, nf, _r, _f = gen.sample(BATCH, T, DEV)
        x = x0
        p_start = x[:, 0:3].clone()
        v_start = x[:, 3:6].clone()
        vdir = v_start / torch.linalg.norm(v_start, dim=-1, keepdim=True).clamp_min(1e-6)
        min_sep = torch.full((BATCH,), 1e6, dtype=DTYPE, device=DEV)
        xt_max = torch.zeros(BATCH, dtype=DTYPE, device=DEV)
        xt_sum = torch.zeros(BATCH, dtype=DTYPE, device=DEV)
        for t in range(T):
            p0, v0 = x[:, 0:3], x[:, 3:6]
            tt = torch.arange(HP + 1, dtype=DTYPE, device=DEV) * DT
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            try:
                u, _ = pol(x, nh, nf[:, :, t, :], p_ref)
            except Exception:
                u = torch.zeros(BATCH, 4, dtype=DTYPE, device=DEV)
                u[:, 0] = DEFAULT_PARAMS.weight
            x = dyn.step(x, u, wind.sample(p0), DT)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            min_sep = torch.minimum(min_sep, d)
            # cross-track: perpendicular distance of realized pos from the
            # straight-line INITIAL reference through p_start along vdir
            rel = x[:, 0:3] - p_start
            along = (rel * vdir).sum(-1, keepdim=True) * vdir
            xtrack = torch.linalg.norm(rel - along, dim=-1)
            xt_max = torch.maximum(xt_max, xtrack)
            xt_sum = xt_sum + xtrack
        mins.append(min_sep.cpu())
        xmax.append(xt_max.cpu())
        xmean.append((xt_sum / T).cpu())
    return (torch.cat(mins)[:N], torch.cat(xmax)[:N], torch.cat(xmean)[:N])


def mcnemar_exact_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - hw), 100 * (c + hw))


def paired_diff_ci(a, b, z=1.96):
    """95% CI on mean(a-b) via normal approx (n=200)."""
    diff = (a - b)
    n = diff.numel()
    m = float(diff.mean()); sd = float(diff.std(unbiased=True))
    se = sd / sqrt(n)
    return m, (m - z * se, m + z * se)


def paired_table(cx, cy):
    a = int((cx & cy).sum()); b = int((cx & ~cy).sum())
    c = int((~cx & cy).sum()); d = int((~cx & ~cy).sum())
    return a, b, c, d


if __name__ == "__main__":
    open(OUT, "w").close()
    w("P2  deploy-config McNemar (S1b vs S2) + route deviation (TODO-1BP/DEV)")
    w("  CBF-MPC alpha=%.1f Hp=%d a_max=%.0f d_sep=%.0f, n=%d seed=12345" %
      (ALPHA, HP, AMAX, DSEP, N))
    w("")

    sep1, xm1, xa1 = rollout("stage1_full.pt")
    sep1b, xm1b, xa1b = rollout("stage1b_domainadapt.pt")
    sep2, xm2, xa2 = rollout("stage2_final.pt")
    sepM, xmM, xaM = rollout("stage2_matched.pt")

    for nm, s in [("Stage-1", sep1), ("Stage-1b", sep1b),
                  ("Stage-2", sep2), ("Stage-2(matched)", sepM)]:
        k = int((s < DSEP).sum())
        lo, hi = wilson(k, N)
        w("  %-17s CR=%5.1f%%  Wilson95=[%4.1f,%5.1f]  (%d/%d)" %
          (nm, 100.0 * k / N, lo, hi, k, N))
    w("")

    # ---- [TODO-1BP] paired McNemar, deploy config ----
    w("=== [TODO-1BP] deploy-config paired McNemar (per-episode) ===")
    for aname, sa, bname, sb in [("Stage-1b", sep1b, "Stage-2", sep2),
                                 ("Stage-1", sep1, "Stage-2", sep2)]:
        ca = sa < DSEP; cb = sb < DSEP
        A, B, C, D = paired_table(ca, cb)
        p = mcnemar_exact_two_sided(B, C)
        md, (lo, hi) = paired_diff_ci(sb, sa)   # min-sep(S2) - min-sep(other)
        w("  %s vs %s:" % (aname, bname))
        w("    2x2 [rows %s]: both-conf=%d  %s-only=%d  %s-only=%d  both-safe=%d"
          % (aname, A, aname, B, bname, C, D))
        w("    discordant b(%s-conf/%s-safe)=%d  c(%s-safe/%s-conf)=%d  McNemar p=%.4g"
          % (aname, bname, B, aname, bname, C, p))
        w("    paired min-sep diff (S2 - %s) = %+.2f m  95%% CI [%+.2f, %+.2f]"
          % (aname, md, lo, hi))
        w("")

    # ---- [TODO-DEV] route (cross-track) deviation ----
    w("=== [TODO-DEV] route cross-track deviation (metres) ===")
    w("  reported as mean +/- SD over %d episodes" % N)
    def ms(v):
        return float(v.mean()), float(v.std(unbiased=True))
    for nm, vmax, vmean in [("Stage-1", xm1, xa1), ("Stage-2", xm2, xa2),
                            ("Stage-2(matched)", xmM, xaM)]:
        mmx, smx = ms(vmax); mmn, smn = ms(vmean)
        w("  %-17s  per-episode MAX xtrack = %5.2f +/- %5.2f m   MEAN xtrack = %5.2f +/- %5.2f m"
          % (nm, mmx, smx, mmn, smn))
    w("")
    # matched-pair contrast + paired CI on max xtrack (the sharp question)
    md, (lo, hi) = paired_diff_ci(xm2, xmM)
    w("  matched-pair contrast: Stage-2(final) - Stage-2(matched) on MAX xtrack")
    w("    = %+.2f m  95%% CI [%+.2f, %+.2f]" % (md, lo, hi))
    mdc, (loc, hic) = paired_diff_ci(xm2, xm1)
    w("  Stage-2(final) - Stage-1 on MAX xtrack = %+.2f m  95%% CI [%+.2f, %+.2f]"
      % (mdc, loc, hic))
    w("  READ: if the matched-pair CI includes 0, Stage-2 does NOT buy behaviour")
    w("        with a larger lateral excursion than its matched control.")
