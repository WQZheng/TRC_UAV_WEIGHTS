"""Route-A scale sweep: fine-tune Stage-2 from the same stage1_full.pt at
several training-pool sizes, evaluate each on the held-out set
(2500-3000). Helps tell apart "scale" vs "loss config" issues.

NOTE: uses eval n=64 for speed; conflict rate has high variance at that
size, so confirm the chosen config with final_compare.py at n>=200.

Writes SCAN_RESULT.txt incrementally.
"""
import argparse
import time
import torch

from seeding import set_seed
from config import GUAM_MAT
from params import DEFAULT_PARAMS
from dynamics import EVTOLDynamics
from wind import UrbanWindField
from predictor import GMMTrajectoryPredictor
from cbf_mpc import CBFMPCLayer
from safe_policy import SafePolicy
from guam_encounters import GUAMEncounters
import train_stage2 as T2
import eval_stage1_vs_stage2 as EV

OUT = "SCAN_RESULT.txt"
DTYPE = torch.float64


def make_args(**kw):
    d = dict(beta=0.3, w_coll=3.0, w_delay=0.05, w_energy=0.01,
             w_lead=0.5, w_ade=0.3)
    d.update(kw)
    return argparse.Namespace(**d)


def train_pool(stage1, pool, iters, seed, device):
    set_seed(seed)
    gen = GUAMEncounters(GUAM_MAT, range(pool), seed=seed)
    pred = GMMTrajectoryPredictor(T=30, K=5).to(DTYPE).to(device)
    pred.load_state_dict(torch.load(stage1, map_location=device))
    mpc = CBFMPCLayer(n_neighbors=1, horizon=8, dt=0.2, d_sep=30.0,
                      alpha=0.4, a_max=10.0)
    policy = SafePolicy(pred, mpc)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=device)
    wind = UrbanWindField(eta_w=0.3, dtype=DTYPE, device=device, seed=0)
    opt = torch.optim.Adam(pred.parameters(), lr=1e-4)
    args = make_args()
    for it in range(1, iters + 1):
        x0, nh, nf, _r, nfut = gen.sample(16, 20, device)
        try:
            loss, info = T2.rollout_loss(policy, dyn, wind, x0, nh, nf,
                                         nfut, 20, 0.2, args)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 5.0)
            opt.step()
        except Exception:
            continue
    return pred


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed, stage1 = 12345, "stage1_full.pt"
    pools, iters, n_eval = [300, 800, 1500, 2500], 50, 64

    set_seed(seed)
    s1 = GMMTrajectoryPredictor(T=30, K=5).double().to(device)
    s1.load_state_dict(torch.load(stage1, map_location=device))
    s1.eval()
    g = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=seed)
    m1 = EV.evaluate(s1, g, n_eval, 20, 8, 0.2, 0.3, device)

    with open(OUT, "w") as f:
        f.write("Route-A scale sweep (seed=%d, iters=%d, n_eval=%d)\n"
                % (seed, iters, n_eval))
        f.write("Stage-1 baseline: CR=%.1f%%  minSep=%.1f m  ADE=%.2f m\n\n"
                % (m1["conflict_rate_%"], m1["mean_min_sep_m"], m1["ADE_m"]))
        f.write("%-8s %10s %10s %10s\n" % ("pool", "CR%", "minSep", "ADE"))

    for pool in pools:
        t0 = time.time()
        pred = train_pool(stage1, pool, iters, seed, device)
        pred.eval()
        ge = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=seed)
        m = EV.evaluate(pred, ge, n_eval, 20, 8, 0.2, 0.3, device)
        line = ("%-8d %10.1f %10.1f %10.2f   (%.0fs)\n"
                % (pool, m["conflict_rate_%"], m["mean_min_sep_m"],
                   m["ADE_m"], time.time() - t0))
        with open(OUT, "a") as f:
            f.write(line)
        print(line, end="", flush=True)


if __name__ == "__main__":
    main()
