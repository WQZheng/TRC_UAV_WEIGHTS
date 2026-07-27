"""Stage-2 joint task-aligned fine-tuning (manuscript Algorithm 1, Stage 2).

Loads the (yaw-augmented) Stage-1 predictor, wraps it in SafePolicy
(predictor -> CBF-MPC -> 6-DOF dynamics), runs differentiable closed-loop
rollouts on GUAM-derived medium-difficulty encounters, and fine-tunes
theta under the task-aligned surrogate loss (TASL, Section 4.4):

  L_TASL = w_coll  * smooth_collision
         + w_delay * schedule_deviation
         + w_energy* control_energy (normalised)
         - w_lead  * conflict_warning_lead_time
         + w_ade   * ADE-anchor   (keeps the predictor faithful)

Verified config (RTX 4090):
  --iters 70 --batch 16 --T 20 --Hp 8 --lr 1e-4
  --w_coll 3.0 --w_lead 0.5 --w_ade 0.12
gives ~halved conflict rate vs Stage-1 (see eval_stage1_vs_stage2.py).
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


def rollout_loss(policy, dyn, wind, x0, neigh_hist, neigh_full, neigh_fut,
                 T, dt, args):
    """Differentiable rollout + TASL loss."""
    B = x0.shape[0]
    device = x0.device
    x = x0
    Hp = policy.Hp

    # ADE anchor: keep the predictor faithful to neighbour motion
    out = policy.f(neigh_hist.reshape(B, neigh_hist.shape[2], 3))
    mean_pred = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)
    h_anchor = min(30, neigh_fut.shape[2])
    ade_anchor = torch.linalg.norm(
        mean_pred[:, :h_anchor] - neigh_fut[:, 0, :h_anchor, :], dim=-1).mean()

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
        u, _ = policy(x, neigh_hist, neigh_full[:, :, t, :], p_ref)
        w_true = wind.sample(p0)
        x = dyn.step(x, u, w_true, dt)

        thr_n = (u[:, 0] - weight) / weight
        mom_n = u[:, 1:4] / mmax
        energy = energy + thr_n ** 2 + (mom_n ** 2).sum(-1)

        d = torch.linalg.norm(x[:, 0:3] - neigh_full[:, 0, t + 1, :], dim=-1)
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
    info = {"min_sep": min_sep.detach(),
            "soft_coll": soft_coll.mean().item(),
            "energy": energy.mean().item(),
            "lead": lead.mean().item(), "ade": ade_anchor.item()}
    return loss, info


def run(args):
    set_seed(args.seed)
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"device = {device}")

    gen = GUAMEncounters(GUAM_MAT, range(args.n_traj_pool), seed=args.seed)
    pred = GMMTrajectoryPredictor(T=30, K=5).to(DTYPE).to(device)
    pred.load_state_dict(torch.load(args.stage1, map_location=device))
    print(f"loaded Stage-1 weights from {args.stage1}")

    mpc = CBFMPCLayer(n_neighbors=1, horizon=args.Hp, dt=args.dt,
                      d_sep=D_SEP, alpha=args.alpha, a_max=args.a_max)
    policy = SafePolicy(pred, mpc)
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
    print(f"\nsaved Stage-2 predictor -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=str, default="stage1_predictor_aug.pt")
    ap.add_argument("--out", type=str, default="stage2_v2.pt")
    ap.add_argument("--Hp", type=int, default=8)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--a_max", type=float, default=10.0)
    ap.add_argument("--eta_w", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--w_coll", type=float, default=3.0)
    ap.add_argument("--w_delay", type=float, default=0.05)
    ap.add_argument("--w_energy", type=float, default=0.01)
    ap.add_argument("--w_lead", type=float, default=0.5)
    ap.add_argument("--w_ade", type=float, default=0.3)
    ap.add_argument("--n_traj_pool", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--cuda", action="store_true")
    run(ap.parse_args())
