"""Baseline 04 -- Soft-IPP: task-aligned joint training WITHOUT a CBF
safety certificate (a DIPP-style integrated predict-and-plan baseline).

This is the fair joint-training counterpart to PlanGrad's Stage-2. It is
IDENTICAL to plangrad_sim.train_stage2 in every respect -- same Stage-1
initialisation (stage1_full.pt), same differentiable closed-loop rollout,
same TASL surrogate loss (collision + delay + energy - lead + ADE anchor),
same iters / batch / T / Hp / learning rate / data pool / seed 12345 --
EXCEPT the planner inside the loop is the soft-penalty Vanilla-MPC
(no control-barrier-function constraints) instead of CBF-MPC.

Rationale: differentiable integrated planning methods (DIPP and kin)
back-propagate a task loss through a planner that encodes obstacles as a
SOFT cost, not a hard safety certificate. By keeping everything else fixed
and only removing the CBF certificate from BOTH training and evaluation,
the Soft-IPP vs PlanGrad gap isolates the value of training/optimising
through a hard safety layer rather than a soft penalty.

Output: soft_joint.pt  (the jointly-trained predictor for this baseline).

Reproducibility: seed 12345; GUAM trajectories 0..n_traj_pool for the
encounter pool (training range, no overlap with eval 2500..3000).
"""
from __future__ import annotations
import os
import sys
import argparse
import torch

sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import GUAM_MAT                      # noqa: E402
from params import DEFAULT_PARAMS                # noqa: E402
from dynamics import EVTOLDynamics               # noqa: E402
from wind import UrbanWindField                  # noqa: E402
from predictor import GMMTrajectoryPredictor     # noqa: E402
from safe_policy import SafePolicy               # noqa: E402
from guam_encounters import GUAMEncounters       # noqa: E402
from seeding import set_seed                     # noqa: E402
from vanilla_mpc import VanillaMPCLayer          # noqa: E402

DTYPE = torch.float64
D_SEP = 30.0


def rollout_loss(policy, dyn, wind, x0, nh, nf, nfut, T, dt, args):
    """Differentiable rollout + TASL loss (identical form to train_stage2)."""
    B = x0.shape[0]
    device = x0.device
    x = x0
    Hp = policy.Hp

    out = policy.f(nh.reshape(B, nh.shape[2], 3))
    mean_pred = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)
    h_anchor = min(30, nfut.shape[2])
    ade_anchor = torch.linalg.norm(
        mean_pred[:, :h_anchor] - nfut[:, 0, :h_anchor, :], dim=-1).mean()

    min_sep = torch.full((B,), 1e6, dtype=DTYPE, device=device)
    energy = torch.zeros(B, dtype=DTYPE, device=device)
    soft_coll = torch.zeros(B, dtype=DTYPE, device=device)
    lead_acc = torch.zeros(B, dtype=DTYPE, device=device)
    weight = DEFAULT_PARAMS.weight
    mmax = DEFAULT_PARAMS.max_body_moment

    for t in range(T):
        p0 = x[:, 0:3]
        v0 = x[:, 3:6]
        tt = torch.arange(Hp + 1, dtype=DTYPE, device=device) * dt
        p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
        u, _ = policy(x, nh, nf[:, :, t, :], p_ref)
        x = dyn.step(x, u, wind.sample(p0), dt)

        thr_n = (u[:, 0] - weight) / weight
        mom_n = u[:, 1:4] / mmax
        energy = energy + thr_n ** 2 + (mom_n ** 2).sum(-1)

        d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
        min_sep = torch.minimum(min_sep, d)
        margin = d - D_SEP
        soft_coll = soft_coll + torch.sigmoid(-args.beta * margin)
        lead_acc = lead_acc + torch.sigmoid(args.beta * margin)

    soft_coll = soft_coll / T
    lead = lead_acc / T
    delay = (x[:, 1] ** 2 + x[:, 2] ** 2).sqrt() / 100.0

    loss = (args.w_coll * soft_coll.mean()
            + args.w_delay * delay.mean()
            + args.w_energy * energy.mean()
            - args.w_lead * lead.mean()
            + args.w_ade * ade_anchor)
    info = {"min_sep": min_sep.detach(), "soft_coll": soft_coll.mean().item(),
            "energy": energy.mean().item(), "lead": lead.mean().item(),
            "ade": ade_anchor.item()}
    return loss, info


def run(args):
    set_seed(args.seed)
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"device = {device}")

    gen = GUAMEncounters(GUAM_MAT, range(args.n_traj_pool), seed=args.seed)
    pred = GMMTrajectoryPredictor(T=30, K=5).to(DTYPE).to(device)
    pred.load_state_dict(torch.load(args.stage1, map_location=device))
    print(f"loaded Stage-1 weights from {args.stage1}")

    planner = VanillaMPCLayer(n_neighbors=1, horizon=args.Hp, dt=args.dt,
                              d_sep=D_SEP, a_max=args.a_max, w_rep=args.w_rep)
    policy = SafePolicy(pred, planner)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=device)
    wind = UrbanWindField(eta_w=args.eta_w, dtype=DTYPE, device=device, seed=0)
    opt = torch.optim.Adam(pred.parameters(), lr=args.lr)

    print("\nstep | loss | mean_min_sep(m) | soft_coll | energy | lead | ade")
    for it in range(1, args.iters + 1):
        x0, nh, nf, _ref, nfut = gen.sample(args.batch, args.T, device)
        try:
            loss, info = rollout_loss(policy, dyn, wind, x0, nh, nf, nfut,
                                      args.T, args.dt, args)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 5.0)
            opt.step()
        except Exception as e:
            print(f"{it:4d} | SKIP (solver issue: {type(e).__name__})")
            continue
        if it % args.log_every == 0 or it == 1:
            print(f"{it:4d} | {loss.item():7.3f} | "
                  f"{info['min_sep'].mean().item():7.2f} | "
                  f"{info['soft_coll']:.3f} | {info['energy']:.2f} | "
                  f"{info['lead']:.3f} | ade={info['ade']:.3f}")

    torch.save(pred.state_dict(), args.out)
    print(f"\nsaved Soft-IPP predictor -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=str,
                    default="/data/lab/plangrad/plangrad_sim/stage1_full.pt")
    ap.add_argument("--out", type=str, default=os.path.join(HERE,
                    "soft_joint.pt"))
    ap.add_argument("--Hp", type=int, default=15)        # match best planner
    ap.add_argument("--a_max", type=float, default=20.0)
    ap.add_argument("--w_rep", type=float, default=50.0)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--eta_w", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)     # train_stage2 sweet spot
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--w_coll", type=float, default=3.0)
    ap.add_argument("--w_delay", type=float, default=0.05)
    ap.add_argument("--w_energy", type=float, default=0.01)
    ap.add_argument("--w_lead", type=float, default=0.5)
    ap.add_argument("--w_ade", type=float, default=0.3)
    ap.add_argument("--n_traj_pool", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--cuda", action="store_true")
    run(ap.parse_args())
