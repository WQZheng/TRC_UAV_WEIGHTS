"""Stage-1 predictor pretraining (manuscript Algorithm 1, Stage 1).

Trains f_theta on GUAM own-ship trajectories with a displacement + NLL
objective: Loss = w_nll * GMM-NLL + w_ade * minADE-K. Cosine LR schedule;
best checkpoint (by 0.5*minADE + 0.5*meanADE) saved.

Use --yaw_augment so the predictor is orientation-invariant (needed for
the Stage-2 encounters, where neighbours approach from arbitrary
headings).

Example (GPU):
    python3 train_stage1.py --n_traj 400 --epochs 40 --batch 256 \
        --lr 8e-4 --yaw_augment --cuda --out stage1_predictor_aug.pt
"""
from __future__ import annotations
import argparse
import time
import torch
from torch.utils.data import DataLoader, random_split

from config import GUAM_MAT
from dataset import GUAMWindowDataset
from predictor import GMMTrajectoryPredictor, gmm_nll, best_mode_ade
from seeding import set_seed

SCALE = 100.0  # must match GUAMWindowDataset.normalize_scale


def run(args):
    set_seed(args.seed)
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"device = {device}")

    print("building dataset ...")
    t0 = time.time()
    ds = GUAMWindowDataset(GUAM_MAT, traj_indices=range(args.n_traj),
                           L=args.L, T=args.T, normalize_scale=SCALE,
                           yaw_augment=args.yaw_augment, seed=args.seed)
    print(f"  #windows = {len(ds)}  (built in {time.time()-t0:.1f}s)")

    n_val = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    net = GMMTrajectoryPredictor(T=args.T, K=args.K).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_metric = float("inf")
    best_min = best_mean = float("nan")

    def evaluate():
        net.eval()
        tot_min, tot_mean, n = 0.0, 0.0, 0
        with torch.no_grad():
            for hist, fut in val_dl:
                hist, fut = hist.to(device), fut.to(device)
                mn, me = best_mode_ade(net(hist), fut)
                bs = hist.shape[0]
                tot_min += mn.item() * bs
                tot_mean += me.item() * bs
                n += bs
        return (tot_min / n) * SCALE, (tot_mean / n) * SCALE

    print("\nepoch | train_loss | val_minADE(m) | val_meanADE(m)")
    for ep in range(1, args.epochs + 1):
        net.train()
        run_loss, nb = 0.0, 0
        for hist, fut in train_dl:
            hist, fut = hist.to(device), fut.to(device)
            pred = net(hist)
            loss = (args.w_nll * gmm_nll(pred, fut)
                    + args.w_ade * best_mode_ade(pred, fut)[0])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            run_loss += loss.item()
            nb += 1
        v_min, v_mean = evaluate()
        print(f"{ep:5d} | {run_loss/nb:10.4f} | {v_min:13.3f} | {v_mean:13.3f}")
        sched.step()
        combined = 0.5 * v_min + 0.5 * v_mean
        if combined < best_metric:
            best_metric = combined
            torch.save(net.state_dict(), args.out)
            best_min, best_mean = v_min, v_mean

    print(f"\nsaved BEST predictor weights -> {args.out}")
    print(f"best val minADE = {best_min:.3f} m, meanADE = {best_mean:.3f} m")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_traj", type=int, default=400)
    ap.add_argument("--L", type=int, default=25)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--w_nll", type=float, default=1.0)
    ap.add_argument("--w_ade", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yaw_augment", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--out", type=str, default="stage1_predictor.pt")
    run(ap.parse_args())
