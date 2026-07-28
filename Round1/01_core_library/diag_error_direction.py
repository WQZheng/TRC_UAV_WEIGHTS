"""Directional decomposition of prediction error (referee follow-up on RQ5).

Question: Stage-2 (TASL) carries a LARGER critical-region displacement error than
the pure-domain-adaptation control Stage-1b, yet closed-loop safety is identical.
What is the NATURE of that extra error? Is it isotropic noise, or a systematic
CONSERVATIVE bias -- i.e. does the task-aligned predictor systematically place the
neighbour CLOSER to the ego along the ego-neighbour line than it really is,
functionally an implicit, learned safety margin (cf. Conformal-MPC's explicit
18.6 m buffer)?

Method: for each future step k, build the unit vector u_k pointing FROM the ego
reference position TO the true neighbour position (the "conflict axis"). Project
the prediction error (pred_abs - true_neighbour) onto u_k:
    e_par = <pred - true, u>   (signed, along ego->neighbour axis)
    e_perp = || (pred - true) - e_par*u ||   (magnitude off-axis)
Sign convention:
    e_par > 0  => predicted neighbour is FARTHER from ego than truth (optimistic)
    e_par < 0  => predicted neighbour is CLOSER  to ego than truth (CONSERVATIVE)
We report mean signed e_par (with its std) and mean |e_perp| in the critical
region (within 3 steps of closest approach) and in the inert region, for
Stage-1, Stage-1b and Stage-2. A significantly NEGATIVE critical e_par for
Stage-2 (and near-zero for Stage-1b) is the "learned safety margin" signature.

All from the single final harness (n=200, seed 12345).

Usage:
    export GUAM_MAT=/data/lab/plangrad/GUAM/Challenge_Problems/Data_Set_1.mat
    python3 diag_error_direction.py --n 200 --seed 12345 --out ERRDIR.txt
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from config import GUAM_MAT
from predictor import GMMTrajectoryPredictor
from guam_encounters import GUAMEncounters
from seeding import set_seed

DTYPE = torch.float64
SCALE = 100.0


@torch.no_grad()
def measure(net, gen, n, T, device):
    par_c, par_i, perp_c, perp_i = [], [], [], []
    ade_c, ade_i = [], []
    batch = 8   # n//8 = exactly 200 encounters (matches main-table protocol)
    for _ in range(max(1, n // batch)):
        x0, nh, nf, ref, nfut = gen.sample(batch, T, device)
        out = net(nh.reshape(batch, 25, 3))
        mu = out["mu"]; alpha = out["alpha"]
        mean_pred = (alpha.unsqueeze(-1) * mu).sum(2)               # [B,30,3] scaled/recentred
        h = min(30, nfut.shape[2])
        # convert prediction to ABSOLUTE frame: recentred-by nei_abs[L-1] and /SCALE
        # nei origin = neigh_full[:,0,0,:] (absolute neighbour at first future index
        # corresponds to nei_abs[L-1]); neigh_fut is (nei_abs[L:...] - nei_abs[L-1])/SCALE
        nei_origin = nf[:, 0, 0, :].unsqueeze(1)                    # [B,1,3] absolute
        pred_abs = mean_pred[:, :h] * SCALE + nei_origin            # [B,h,3] absolute
        true_abs = nf[:, 0, 1:h + 1, :]                             # [B,h,3] absolute neighbour future
        ego_abs = ref[:, 1:h + 1, :]                                # [B,h,3] absolute ego reference
        err = pred_abs - true_abs                                   # [B,h,3] ABSOLUTE metres
        # conflict axis: unit vector ego -> true neighbour
        axis = true_abs - ego_abs
        axis_n = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-6)
        e_par = (err * axis_n).sum(-1)                              # [B,h] signed, metres
        e_perp = torch.linalg.norm(err - e_par.unsqueeze(-1) * axis_n, dim=-1)  # [B,h] metres
        ade = torch.linalg.norm(err, dim=-1)                       # [B,h] metres
        # critical vs inert by closest-approach step of the closed-loop reference
        d = torch.linalg.norm(ego_abs - true_abs, dim=-1)
        ca = d.argmin(dim=1)
        idx = torch.arange(h, device=device).unsqueeze(0)
        crit = (idx - ca.unsqueeze(1)).abs() <= 3
        inert = ~crit
        for b in range(batch):
            cm = crit[b]; im = inert[b]
            par_c += e_par[b, cm].cpu().tolist()
            par_i += e_par[b, im].cpu().tolist()
            perp_c += e_perp[b, cm].cpu().tolist()
            perp_i += e_perp[b, im].cpu().tolist()
            ade_c += ade[b, cm].cpu().tolist()
            ade_i += ade[b, im].cpu().tolist()
    m = lambda v: (float(np.mean(v)), float(np.std(v)))
    fneg = lambda v: float(np.mean([1.0 if x < 0 else 0.0 for x in v]))
    return {"par_c": m(par_c), "par_i": m(par_i),
            "perp_c": m(perp_c), "perp_i": m(perp_i),
            "ade_c": m(ade_c), "ade_i": m(ade_i),
            "fneg_c": fneg(par_c), "fneg_i": fneg(par_i),
            "n_c": len(par_c)}


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(p):
        net = GMMTrajectoryPredictor(T=30, K=5).double().to(device)
        net.load_state_dict(torch.load(p, map_location=device)); net.eval()
        return net

    lines = []
    def w(s=""):
        print(s); lines.append(s)
    w("DIRECTIONAL ERROR DECOMPOSITION (n=%d seed=%d)" % (args.n, args.seed))
    w("e_par = <pred-true, ego->neighbour unit>: >0 optimistic (farther from ego), "
      "<0 CONSERVATIVE (closer to ego). All in metres.")
    w("Convention: e_par<0 = predicted neighbour CLOSER to ego than truth "
      "(conservative). SEM = sd/sqrt(n); |mean|>2*SEM ~ significant.")
    for name, ck in args.models:
        set_seed(args.seed)
        g = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=args.seed)
        r = measure(load(ck), g, args.n, args.T, device)
        import math
        sem_c = r["par_c"][1] / math.sqrt(max(r["n_c"], 1))
        w("")
        w("%s:" % name)
        w("  critical: e_par mean=%+.2f m  (SEM %.2f, sd %.2f)  frac(e_par<0)=%.1f%%  |e_perp|=%.2f  ADE=%.2f m"
          % (r["par_c"][0], sem_c, r["par_c"][1], 100 * r["fneg_c"],
             r["perp_c"][0], r["ade_c"][0]))
        w("  inert   : e_par mean=%+.2f m  frac(e_par<0)=%.1f%%  ADE=%.2f m"
          % (r["par_i"][0], 100 * r["fneg_i"], r["ade_i"][0]))
    open(args.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--out", default="ERRDIR.txt")
    args = ap.parse_args()
    args.models = [("Stage-1", "stage1_full.pt"),
                   ("Stage-1b (domain-adapt)", "stage1b_domainadapt.pt"),
                   ("Stage-2 (TASL)", "stage2_final.pt")]
    run(args)
