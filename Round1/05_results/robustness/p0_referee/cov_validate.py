"""[TODO-COV] Empirical pooled coverage validation on the EVALUATION stream.

The clean calibration (conformal_FIXED.py) fixed the conformal radius on the
held-out pool 2500-2999 with calibration seed 777 (independent of eval seed
12345), giving r_conf = 18.93 m at target miscoverage delta=0.1 (target coverage
90%), from n_scores=3000 calibration scores. Section 04 promised a coverage
validation but the results chapter never reported the EMPIRICAL pooled coverage
actually attained on the evaluation stream. This script supplies it.

We reproduce the eval encounter stream EXACTLY (frozen stage1_full.pt, pool
2500-3000, seed 12345, n=200, planning horizon Hp=15) and, for every look-ahead
step k<=Hp, compute the nonconformity score s = ||mu_hat(k) - true_neighbour(k)||
(metres) -- the SAME score conformal_FIXED calibrates on. Empirical pooled
coverage = fraction of eval scores with s <= r_conf. A calibrated split-conformal
predictor should attain >= 90% here (marginal coverage), validating the buffer.

Reports: target coverage, r_conf, empirical pooled coverage on eval, n eval
scores, n calibration scores (from the clean json). Writes COVERAGE_VALID.txt.
Frozen predictor, evaluation-only.
"""
from __future__ import annotations
import torch
from config import GUAM_MAT
from predictor import GMMTrajectoryPredictor
from guam_encounters import GUAMEncounters
from seeding import set_seed

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 100.0
R_CONF = 18.93201805200722     # from conformal_mpc_result_CLEAN.json
DELTA = 0.1                    # target miscoverage -> target coverage 90%
N_CALIB = 3000                # calibration scores (clean json)
HP, N, T = 15, 200, 20
OUT = "COVERAGE_VALID.txt"


def w(s):
    with open(OUT, "a") as f:
        f.write(s + "\n")
    print(s, flush=True)


@torch.no_grad()
def main():
    open(OUT, "w").close()
    net = GMMTrajectoryPredictor(T=30, K=5).double().to(DEV)
    net.load_state_dict(torch.load("stage1_full.pt", map_location=DEV)); net.eval()
    # EVAL stream: seed 12345 (the deployment stream, independent of calib 777)
    set_seed(12345)
    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=12345)
    scores = []
    batch = 8
    for _ in range(N // batch):
        x0, nh, nf, _ref, nfut = gen.sample(batch, T, DEV)
        out = net(nh.reshape(batch, 25, 3))
        mean_traj = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)   # [B,30,3]
        h = min(HP, nfut.shape[2])
        err = torch.linalg.norm(mean_traj[:, :h] - nfut[:, 0, :h, :], dim=-1) * SCALE
        scores.append(err.reshape(-1).cpu())
    s = torch.cat(scores)
    n_eval = s.numel()
    cover = float((s <= R_CONF).float().mean()) * 100.0
    w("EMPIRICAL POOLED COVERAGE VALIDATION (referee TODO-COV)")
    w("  frozen predictor stage1_full.pt; eval stream pool 2500-3000 seed 12345")
    w("  nonconformity score s = ||mu_hat(k) - true(k)|| over k<=Hp=%d, n_ep=%d" % (HP, N))
    w("")
    w("  target coverage (1 - delta)          = %.1f%%" % (100 * (1 - DELTA)))
    w("  conformal radius r_conf (calibrated) = %.2f m" % R_CONF)
    w("  calibration scores (independent)     = %d" % N_CALIB)
    w("  evaluation scores (pooled)           = %d" % n_eval)
    w("  EMPIRICAL POOLED COVERAGE on eval    = %.1f%%" % cover)
    w("")
    if cover >= 100 * (1 - DELTA):
        w("  -> coverage attained (>= target): the conformal buffer is calibrated;")
        w("     enforcing d_sep + r_conf gives the stated marginal guarantee on")
        w("     the held-out evaluation stream.")
    else:
        w("  -> coverage BELOW target on eval by %.1f pts; report honestly as" %
          (100 * (1 - DELTA) - cover))
        w("     an empirical shortfall (finite-sample / mild distribution shift).")
    w("")
    w("  [extra] eval score stats: mean=%.2f m  median=%.2f m  p90=%.2f m  max=%.2f m"
      % (float(s.mean()), float(s.median()),
         float(torch.quantile(s, 0.9)), float(s.max())))


if __name__ == "__main__":
    main()
