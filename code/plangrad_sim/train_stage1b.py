"""Stage-1b: domain-adaptation control for the task-alignment attribution
(referee point 4, follow-up).

Purpose
-------
Stage-2 is fine-tuned in the CLOSED LOOP on encounter-distribution data, so it
has seen the encounter distribution that Stage-1 never saw. That raises a
confound: the 20.90 -> 4.32 m ADE drop might be plain DOMAIN ADAPTATION (just
training on encounter data with a displacement loss), not task alignment.

Stage-1b isolates it: take the SAME Stage-1 checkpoint and fine-tune it on the
IDENTICAL encounter rollout data (same generator, trajectory pool 0-2500, same
seed), for the IDENTICAL number of iterations, learning rate and batch size as
Stage-2, but with a PURE displacement objective L_ADE only -- no collision,
lead-time, energy or schedule terms, and no differentiable planner in the loss.
Everything that differs between Stage-1b and Stage-2 is then exactly the
task-aligned surrogate loss.

Interpretation of the comparison (Stage-1b vs Stage-2):
  * If Stage-1b's ADE also drops to ~4 m AND its critical-region error profile
    matches Stage-2 -> the gain is closed-loop DOMAIN ADAPTATION; the
    "task-alignment" story is downgraded (paper still stands on decoupling +
    certificate).
  * If Stage-1b's overall ADE is close but its critical-region / 6 s-horizon
    behaviour is worse than Stage-2 -> task alignment's value survives in a
    sharper form (domain adaptation buys overall accuracy; task alignment buys
    the critical-region ALLOCATION). Best outcome.
  * If Stage-1b barely moves -> the current narrative stands as-is.

This mirrors Stage-2's ADE-anchor computation EXACTLY (same GMM mixture-mean
displacement over the min(30, horizon) window), so the only removed ingredient
is the task-aligned part of the loss.

Usage (matches the deployed Stage-2 command exactly except for the loss):
    export GUAM_MAT=/data/lab/plangrad/GUAM/Challenge_Problems/Data_Set_1.mat
    python3 train_stage1b.py --cuda --seed 12345 \
        --stage1 stage1_full.pt \
        --iters 70 --batch 16 --T 20 --lr 1e-4 \
        --n_traj_pool 2500 --out stage1b_domainadapt.pt
"""
from __future__ import annotations
import argparse
import torch

from config import GUAM_MAT
from predictor import GMMTrajectoryPredictor
from guam_encounters import GUAMEncounters
from seeding import set_seed

DTYPE = torch.float64


def ade_anchor_loss(pred, neigh_hist, neigh_fut):
    """EXACTLY Stage-2's ADE anchor: GMM mixture-mean displacement error over
    the min(30, horizon) future window. This is the pure L_ADE objective."""
    B = neigh_hist.shape[0]
    out = pred(neigh_hist.reshape(B, neigh_hist.shape[2], 3))
    mean_pred = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)
    h = min(30, neigh_fut.shape[2])
    return torch.linalg.norm(
        mean_pred[:, :h] - neigh_fut[:, 0, :h, :], dim=-1).mean()


def run(args):
    set_seed(args.seed)
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"device = {device}")

    # IDENTICAL encounter generator, pool and seed as Stage-2 (train_stage2.py)
    gen = GUAMEncounters(GUAM_MAT, range(args.n_traj_pool), seed=args.seed)
    pred = GMMTrajectoryPredictor(T=30, K=5).to(DTYPE).to(device)
    pred.load_state_dict(torch.load(args.stage1, map_location=device))
    print(f"loaded Stage-1 weights from {args.stage1}")
    print("Stage-1b: PURE L_ADE fine-tune on encounter rollout data "
          "(no TASL terms, no planner in loss).")

    # IDENTICAL optimizer / lr as Stage-2
    opt = torch.optim.Adam(pred.parameters(), lr=args.lr)

    print("\nstep | L_ADE (scaled units) | ADE(m) on this batch")
    SCALE = 100.0
    for it in range(1, args.iters + 1):
        # IDENTICAL sampling call as Stage-2: same batch, same T, same seed
        # stream, so Stage-1b sees the very same encounters Stage-2 saw.
        x0, nh, nf, _ref, nfut = gen.sample(args.batch, args.T, device)
        loss = ade_anchor_loss(pred, nh, nfut)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), 5.0)
        opt.step()
        if it % args.log_every == 0 or it == 1:
            print(f"{it:4d} | {loss.item():10.4f} | {loss.item()*SCALE:7.2f}")

    torch.save(pred.state_dict(), args.out)
    print(f"\nsaved Stage-1b (domain-adaptation) predictor -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=str, default="stage1_full.pt")
    ap.add_argument("--out", type=str, default="stage1b_domainadapt.pt")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=70)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_traj_pool", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--cuda", action="store_true")
    run(ap.parse_args())
