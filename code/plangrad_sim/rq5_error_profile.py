"""RQ5 critical/inert ERROR profile for Stage-1, Stage-1b, Stage-2 (single
consistent harness). Matches the manuscript's error-based reallocation table
(critical = within 3 steps of closest approach; inert = the rest), reporting
the mean absolute displacement error (not variance) in each region.

This lets us re-attribute RQ5 honestly once Stage-1b (domain-adaptation control)
is in play: is Stage-1b's error also concentrated in the critical region, or is
the critical-region concentration specific to the task-aligned loss?

Usage:
    export GUAM_MAT=/data/lab/plangrad/GUAM/Challenge_Problems/Data_Set_1.mat
    python3 rq5_error_profile.py --n 200 --seed 12345 \
        --models stage1_full.pt stage1b_domainadapt.pt stage2_final.pt \
        --out RQ5_PROFILE.txt
"""
from __future__ import annotations
import argparse
import torch

from config import GUAM_MAT
from predictor import GMMTrajectoryPredictor
from guam_encounters import GUAMEncounters
from seeding import set_seed

DTYPE = torch.float64
SCALE = 100.0


@torch.no_grad()
def measure(net, gen, n, T, device):
    crit_sum, inert_sum, ade_sum, ntot = 0.0, 0.0, 0.0, 0
    crit_cnt, inert_cnt = 0, 0
    batch = 8   # n//8 = exactly 200 encounters (matches main-table protocol)
    for _ in range(max(1, n // batch)):
        x0, nh, nf, ref, nfut = gen.sample(batch, T, device)
        out = net(nh.reshape(batch, 25, 3))
        mu = out["mu"]; alpha = out["alpha"]
        mean_pred = (alpha.unsqueeze(-1) * mu).sum(2)
        h = min(30, nfut.shape[2])
        gt = nfut[:, 0, :h, :]
        err = torch.linalg.norm(mean_pred[:, :h] - gt, dim=-1)   # [B,h]
        ade_sum += err.mean().item() * SCALE * batch
        ntot += batch
        # closest-approach step in the closed-loop reference (same as
        # diagnose_decoupling): critical = within 3 steps of CA.
        d = torch.linalg.norm(ref[:, 1:h + 1, :] - nf[:, 0, 1:h + 1, :], dim=-1)
        ca = d.argmin(dim=1)
        idx = torch.arange(h, device=device).unsqueeze(0)
        crit_mask = (idx - ca.unsqueeze(1)).abs() <= 3
        inert_mask = ~crit_mask
        crit_sum += (err * crit_mask).sum().item() * SCALE
        inert_sum += (err * inert_mask).sum().item() * SCALE
        crit_cnt += crit_mask.sum().item()
        inert_cnt += inert_mask.sum().item()
    return {"ADE_m": ade_sum / ntot,
            "err_critical_m": crit_sum / max(crit_cnt, 1),
            "err_inert_m": inert_sum / max(inert_cnt, 1)}


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(p):
        net = GMMTrajectoryPredictor(T=30, K=5).double().to(device)
        net.load_state_dict(torch.load(p, map_location=device)); net.eval()
        return net

    lines = []
    def w(s=""):
        print(s); lines.append(s)

    w("RQ5 critical/inert ERROR profile (n=%d seed=%d)" % (args.n, args.seed))
    w("critical = within 3 steps of closest approach; inert = rest")
    w("%-26s %10s %12s %10s %10s" %
      ("model", "ADE(m)", "critical(m)", "inert(m)", "crit/inert"))
    for ck in args.models:
        set_seed(args.seed)
        g = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=args.seed)
        m = measure(load(ck), g, args.n, args.T, device)
        ratio = m["err_critical_m"] / max(m["err_inert_m"], 1e-9)
        w("%-26s %10.2f %12.2f %10.2f %10.3f" %
          (ck, m["ADE_m"], m["err_critical_m"], m["err_inert_m"], ratio))
    open(args.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["stage1_full.pt", "stage1b_domainadapt.pt",
                             "stage2_final.pt"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--out", default="RQ5_PROFILE.txt")
    run(ap.parse_args())
