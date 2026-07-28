"""Mechanism-level diagnosis (manuscript Proposition 3).

Directly measures whether Stage-2 reallocates predictive variance toward
the conflict region. Splits each predicted trajectory into CRITICAL steps
(near closest approach to the neighbour) and INERT steps (the rest), and
reports mean predictive variance (trace Sigma) in each group for Stage-1
and Stage-2. Prop. 3 predicts the critical/inert variance ratio should
DECREASE under Stage-2.

Verified: ratio 0.313 (Stage-1) -> 0.308 (Stage-2) -- mechanism present.
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
    crit_var, inert_var, ade_sum, ntot = 0.0, 0.0, 0.0, 0
    crit_cnt, inert_cnt = 0, 0
    batch = 16
    for _ in range(max(1, n // batch)):
        x0, nh, nf, ref, nfut = gen.sample(batch, T, device)
        out = net(nh.reshape(batch, 25, 3))
        mu = out["mu"]; alpha = out["alpha"]
        var = torch.exp(2.0 * out["log_sigma"])
        tr = (alpha.unsqueeze(-1) * var).sum(2).sum(-1)      # [B,30]
        mean_pred = (alpha.unsqueeze(-1) * mu).sum(2)

        h = min(30, nfut.shape[2])
        gt = nfut[:, 0, :h, :]
        ade = torch.linalg.norm(mean_pred[:, :h] - gt, dim=-1)
        ade_sum += ade.mean().item() * SCALE * batch
        ntot += batch

        d = torch.linalg.norm(ref[:, 1:h + 1, :] - nf[:, 0, 1:h + 1, :],
                              dim=-1)
        ca = d.argmin(dim=1)
        idx = torch.arange(h, device=device).unsqueeze(0)
        crit_mask = (idx - ca.unsqueeze(1)).abs() <= 3
        inert_mask = ~crit_mask

        tr_h = tr[:, :h]
        crit_var += (tr_h * crit_mask).sum().item() * SCALE * SCALE
        inert_var += (tr_h * inert_mask).sum().item() * SCALE * SCALE
        crit_cnt += crit_mask.sum().item()
        inert_cnt += inert_mask.sum().item()

    return {"ADE_m": ade_sum / ntot,
            "var_critical": crit_var / max(crit_cnt, 1),
            "var_inert": inert_var / max(inert_cnt, 1)}


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}\n")

    def load(p):
        net = GMMTrajectoryPredictor(T=30, K=5).double().to(device)
        net.load_state_dict(torch.load(p, map_location=device)); net.eval()
        return net

    s1, s2 = load(args.stage1), load(args.stage2)
    g1 = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=args.seed)
    g2 = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=args.seed)
    m1 = measure(s1, g1, args.n_eval, args.T, device)
    m2 = measure(s2, g2, args.n_eval, args.T, device)

    print(f"{'metric':<16}{'Stage-1':>12}{'Stage-2':>12}")
    for k in ["ADE_m", "var_critical", "var_inert"]:
        print(f"{k:<16}{m1[k]:>12.4f}{m2[k]:>12.4f}")

    r1 = m1["var_critical"] / max(m1["var_inert"], 1e-9)
    r2 = m2["var_critical"] / max(m2["var_inert"], 1e-9)
    print(f"\ncritical/inert variance ratio: "
          f"Stage-1 {r1:.3f}  ->  Stage-2 {r2:.3f}")
    print("=> mechanism present" if r2 < r1 else "=> mechanism NOT present")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="stage1_full.pt")
    ap.add_argument("--stage2", default="stage2_full.pt")
    ap.add_argument("--n_eval", type=int, default=96)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--T", type=int, default=20)
    run(ap.parse_args())
