"""P0-2: EPISODE-LEVEL directional error + direct paired test (referee #7b).

Referee point #7b: the existing e_parallel analysis (diag_error_direction.py)
pools ALL critical-window steps (~7 steps/episode x 200 = ~1400) into one list
and computes SEM = sd/sqrt(1400). Those 7 within-episode steps are highly
correlated, so the effective n is ~200 episodes, not 1400 steps; the step-level
SEM is understated by ~sqrt(7)~2.6x and the step-level significance may not
survive clustering. #7b also asks for the never-done DIRECT paired test
e_par(S2) - e_par(S1b) on the same episodes.

This script:
  (1) reproduces the exact geometry of diag_error_direction.measure but returns
      a PER-EPISODE critical-window e_par mean and toward-ego fraction (length-N
      vectors), for Stage-1 / Stage-1b / Stage-2, on the SAME encounter stream
      (seed 12345, range(2500,3000)) so the three arms are episode-aligned;
  (2) reports each arm's e_par at EPISODE level: mean, SEM = sd/sqrt(N_epi),
      t = mean/SEM, and whether |mean| > 2*SEM (cluster-robust "significance");
  (3) runs the DIRECT paired comparison S2 - S1b (and S2 - S1) over the N paired
      episodes: mean paired diff, paired-t p, and Wilcoxon signed-rank p.

The critical window per episode is defined by the closest-approach step of the
GUAM reference vs true neighbour (|idx - CPA| <= 3); it depends only on ref and
true neighbour, NOT on the model, so the three arms share an identical window
per episode and are legitimately paired.

Decision rule (computed, not assumed):
  * If per-arm episode-level e_par stays |mean|>2*SEM AND the paired S2-S1b diff
    is significant (both t and Wilcoxon) -> the conservative-bias fingerprint
    SURVIVES clustering; 5.4 keeps an inferential claim.
  * If either fails -> DOWNGRADE 5.4 to descriptive ("direction consistent,
    not significant under episode clustering").

Writes P0_ERRDIR_EPISODE.txt. Retrains nothing; evaluation-only.
"""
from __future__ import annotations
import argparse
import math
import numpy as np
import torch

from config import GUAM_MAT
from predictor import GMMTrajectoryPredictor
from guam_encounters import GUAMEncounters
from seeding import set_seed

DTYPE = torch.float64
SCALE = 100.0


@torch.no_grad()
def per_episode(net, gen, n, T, device):
    """Return per-episode critical-window (e_par_mean, toward_ego_frac, ade_mean)
    as three length-N numpy arrays, plus the flat step-level e_par list (for the
    step-vs-episode SEM comparison)."""
    epi_par, epi_fneg, epi_ade = [], [], []
    flat_par = []
    batch = 8
    for _ in range(max(1, n // batch)):
        x0, nh, nf, ref, nfut = gen.sample(batch, T, device)
        out = net(nh.reshape(batch, 25, 3))
        mu = out["mu"]; alpha = out["alpha"]
        mean_pred = (alpha.unsqueeze(-1) * mu).sum(2)
        h = min(30, nfut.shape[2])
        nei_origin = nf[:, 0, 0, :].unsqueeze(1)
        pred_abs = mean_pred[:, :h] * SCALE + nei_origin
        true_abs = nf[:, 0, 1:h + 1, :]
        ego_abs = ref[:, 1:h + 1, :]
        err = pred_abs - true_abs
        axis = true_abs - ego_abs
        axis_n = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-6)
        e_par = (err * axis_n).sum(-1)          # [B,h] signed metres
        ade = torch.linalg.norm(err, dim=-1)    # [B,h]
        d = torch.linalg.norm(ego_abs - true_abs, dim=-1)
        ca = d.argmin(dim=1)
        idx = torch.arange(h, device=device).unsqueeze(0)
        crit = (idx - ca.unsqueeze(1)).abs() <= 3
        for b in range(batch):
            cm = crit[b]
            vals = e_par[b, cm]
            if vals.numel() == 0:
                continue
            epi_par.append(float(vals.mean().cpu()))
            epi_fneg.append(float((vals < 0).float().mean().cpu()))
            epi_ade.append(float(ade[b, cm].mean().cpu()))
            flat_par += vals.cpu().tolist()
    return (np.array(epi_par), np.array(epi_fneg), np.array(epi_ade),
            np.array(flat_par))


def paired_t(diff):
    """Two-sided paired t-test p-value on a 1-D diff array (H0: mean=0)."""
    n = len(diff)
    m = float(np.mean(diff)); sd = float(np.std(diff, ddof=1))
    if sd == 0:
        return m, 0.0, (0.0 if m == 0 else 1e-300)
    se = sd / math.sqrt(n)
    t = m / se
    # p via scipy if available, else normal approx
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    except Exception:
        from math import erfc, sqrt
        p = float(erfc(abs(t) / sqrt(2)))
    return m, t, p


def wilcoxon_p(diff):
    try:
        from scipy import stats
        nz = diff[diff != 0]
        if len(nz) == 0:
            return 1.0
        return float(stats.wilcoxon(nz, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(p):
        net = GMMTrajectoryPredictor(T=30, K=5).double().to(device)
        net.load_state_dict(torch.load(p, map_location=device)); net.eval()
        return net

    lines = []
    def w(s=""):
        print(s, flush=True); lines.append(s)

    w("P0-2  EPISODE-LEVEL directional error + paired test (referee #7b)")
    w("  n=%d seed=%d, critical window |idx-CPA|<=3 (model-independent)" %
      (args.n, args.seed))
    w("  e_par<0 = predicted neighbour CLOSER to ego than truth (conservative)")
    w("")

    res = {}
    for name, ck in args.models:
        set_seed(args.seed)
        g = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=args.seed)
        epi_par, epi_fneg, epi_ade, flat = per_episode(load(ck), g, args.n,
                                                        args.T, device)
        res[name] = epi_par
        n_epi = len(epi_par)
        mean = float(np.mean(epi_par))
        sd_epi = float(np.std(epi_par, ddof=1))
        sem_epi = sd_epi / math.sqrt(n_epi)
        # step-level SEM for contrast (the OLD, understated one)
        sem_step = float(np.std(flat, ddof=1)) / math.sqrt(len(flat))
        sig = "YES" if abs(mean) > 2 * sem_epi else "NO"
        w("%s:" % name)
        w("  EPISODE-level e_par: mean=%+.3f m  SEM=%.3f (n_epi=%d)  "
          "|mean|>2SEM ? %s" % (mean, sem_epi, n_epi, sig))
        w("    t = mean/SEM = %+.2f   toward-ego frac (episode mean) = %.1f%%" %
          (mean / sem_epi if sem_epi else float('nan'),
           100 * float(np.mean(epi_fneg))))
        w("    [contrast] OLD step-level SEM = %.3f over %d steps "
          "(understated ~%.1fx)" %
          (sem_step, len(flat), (sem_epi / sem_step if sem_step else float('nan'))))
        w("")

    # direct paired comparisons on aligned episodes
    def cmp(aname, bname):
        da = res[aname]; db = res[bname]
        m = min(len(da), len(db))
        diff = da[:m] - db[:m]
        md, t, pt = paired_t(diff)
        pw = wilcoxon_p(diff)
        w("PAIRED  %s - %s  (n=%d):" % (aname, bname, m))
        w("  mean paired diff = %+.3f m   paired-t: t=%+.2f  p=%.4g   "
          "Wilcoxon p=%.4g" % (md, t, pt, pw))
        sig = (pt < 0.05) and (not math.isnan(pw) and pw < 0.05)
        w("  -> %s at episode level (both t and Wilcoxon < 0.05 ? %s)" %
          ("SIGNIFICANT" if sig else "NOT significant", "yes" if sig else "no"))
        w("")

    w("=== DIRECT PAIRED TESTS (the never-done e_par(S2)-e_par(S1b)) ===")
    cmp("Stage-2 (TASL)", "Stage-1b (domain-adapt)")
    cmp("Stage-2 (TASL)", "Stage-1")

    open(args.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--out", default="P0_ERRDIR_EPISODE.txt")
    args = ap.parse_args()
    args.models = [("Stage-1", "stage1_full.pt"),
                   ("Stage-1b (domain-adapt)", "stage1b_domainadapt.pt"),
                   ("Stage-2 (TASL)", "stage2_final.pt")]
    run(args)
