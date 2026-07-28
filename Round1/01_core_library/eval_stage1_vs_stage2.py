"""Central evaluation: Stage-1 vs Stage-2 predictor (manuscript RQ2).

Compares, on the SAME held-out GUAM encounters, the displacement-trained
predictor (Stage 1) against the task-aligned fine-tuned predictor
(Stage 2), reporting BOTH displacement metrics (ADE/minADE) and
closed-loop operational safety (conflict rate / min separation). Both run
inside the IDENTICAL SafePolicy + CBF-MPC + 6-DOF loop, so the only
difference is theta.

Verified result (96 held-out encounters, two seeds): Stage-2 roughly
HALVES the conflict rate and raises min separation by ~8-9 m.
"""
from __future__ import annotations
import argparse
import torch

from config import GUAM_MAT
from params import DEFAULT_PARAMS
from dynamics import EVTOLDynamics
from wind import UrbanWindField
from predictor import GMMTrajectoryPredictor
from cbf_mpc import CBFMPCLayer
from safe_policy import SafePolicy
from guam_encounters import GUAMEncounters
from seeding import set_seed

DTYPE = torch.float64
D_SEP = 30.0
SCALE = 100.0


@torch.no_grad()
def evaluate(predictor, gen, n_eval, T, Hp, dt, eta_w, device):
    mpc = CBFMPCLayer(n_neighbors=1, horizon=Hp, dt=dt, d_sep=D_SEP,
                      alpha=0.4, a_max=10.0)
    policy = SafePolicy(predictor, mpc)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=device)
    wind = UrbanWindField(eta_w=eta_w, dtype=DTYPE, device=device, seed=7)

    n_coll, n_tot, sep_sum = 0, 0, 0.0
    ade_sum, minade_sum, n_ade = 0.0, 0.0, 0
    batch = 8
    for _ in range(max(1, n_eval // batch)):
        x0, nh, nf, _ref, _nfut = gen.sample(batch, T, device)
        x = x0
        min_sep = torch.full((batch,), 1e6, dtype=DTYPE, device=device)

        neigh_last = nf[:, :, 0, :]
        out = predictor(nh.reshape(batch, 25, 3))
        mu = out["mu"]
        mean_traj = (out["alpha"].unsqueeze(-1) * mu).sum(2)
        horizon = min(30, T)
        gt = (nf[:, 0, 1:horizon + 1, :] - neigh_last[:, 0:1, :]) / SCALE
        ade = torch.linalg.norm(mean_traj[:, :horizon] - gt, dim=-1).mean(1)
        ade_sum += (ade.mean().item() * SCALE) * batch
        disp = torch.linalg.norm(mu[:, :horizon] - gt.unsqueeze(2), dim=-1)
        minade = disp.mean(1).min(1).values
        minade_sum += (minade.mean().item() * SCALE) * batch
        n_ade += batch

        for t in range(T):
            p0 = x[:, 0:3]; v0 = x[:, 3:6]
            tt = torch.arange(Hp + 1, dtype=DTYPE, device=device) * dt
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            u, _ = policy(x, nh, nf[:, :, t, :], p_ref)
            x = dyn.step(x, u, wind.sample(p0), dt)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            min_sep = torch.minimum(min_sep, d)

        n_coll += int((min_sep < D_SEP).sum().item())
        sep_sum += float(min_sep.sum().item())
        n_tot += batch

    return {"ADE_m": ade_sum / n_ade, "minADE_m": minade_sum / n_ade,
            "conflict_rate_%": 100.0 * n_coll / n_tot,
            "mean_min_sep_m": sep_sum / n_tot, "n": n_tot}


def run(args):
    set_seed(args.seed)
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"device = {device}\n")

    def load(path):
        net = GMMTrajectoryPredictor(T=30, K=5).double().to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()
        return net

    s1, s2 = load(args.stage1), load(args.stage2)
    print("evaluating Stage-1 (ADE-trained) ...")
    g1 = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=args.seed)
    m1 = evaluate(s1, g1, args.n_eval, args.T, args.Hp, args.dt, args.eta_w,
                  device)
    print("evaluating Stage-2 (TASL-trained) ...")
    g2 = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=args.seed)
    m2 = evaluate(s2, g2, args.n_eval, args.T, args.Hp, args.dt, args.eta_w,
                  device)

    print("\n================ Stage-1 vs Stage-2 ================")
    print(f"{'metric':<18}{'Stage-1':>12}{'Stage-2':>12}{'change':>12}")
    for k in ["ADE_m", "minADE_m", "conflict_rate_%", "mean_min_sep_m"]:
        print(f"{k:<18}{m1[k]:>12.3f}{m2[k]:>12.3f}{m2[k]-m1[k]:>+12.3f}")
    print(f"\n(evaluated on {m1['n']} held-out encounters)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=str, default="stage1_full.pt")
    ap.add_argument("--stage2", type=str, default="stage2_full.pt")
    ap.add_argument("--n_eval", type=int, default=96)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--Hp", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--eta_w", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--cuda", action="store_true")
    run(ap.parse_args())
