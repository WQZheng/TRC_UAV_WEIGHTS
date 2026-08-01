"""Penetration-rate sweep driver.

Sweeps the equipped fraction p in {0, 25, 50, 75, 100}% at FIXED total demand,
running several Monte-Carlo replicates per p (conflict is a rare per-agent
event, so we need replicates for statistical power). Reports, for each p and
separately for the equipped / unequipped / all sub-populations: per-agent
conflict rate, throughput (passes/min), and mean delay (s). Writes PENETRATION.txt.

Usage:
    python3 run_penetration.py --reps 12 --horizon 600 --arrival 0.30 \
        --stage2 stage2_final.pt --seed 12345
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from seeding import set_seed
from predictor import GMMTrajectoryPredictor
from penetration_sim_censor import PenetrationCorridor


def load_pred(path, dev):
    net = GMMTrajectoryPredictor(T=30, K=5).double().to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()
    return net


def agg(runs, key):
    """Aggregate a metric across MC replicates (mean +/- std of the group)."""
    vals = [r[key]["conflict_rate"] for r in runs]
    thr = [r[key]["throughput"] for r in runs]
    dly = [r[key]["mean_delay"] for r in runs]
    ns = [r[key]["n"] for r in runs]
    pas = [r[key]["passed"] for r in runs]
    tos = [r[key]["n_timeout"] for r in runs]
    lat = [r[key]["n_lateral"] for r in runs]
    dcen = [r[key]["mean_delay_censored"] for r in runs]
    v = np.array(vals, float); t = np.array(thr, float); d = np.array(dly, float)
    dc = np.array(dcen, float)
    ntot = int(np.sum(ns)); ptot = int(np.sum(pas))
    ntoi = int(np.sum(tos)); nlai = int(np.sum(lat))
    comp = 100.0 * ptot / ntot if ntot else float("nan")   # completion %
    cens = int(np.sum([r[key].get("n_censored_in_corridor", 0) for r in runs]))
    fin  = int(np.sum([r[key].get("n_finished_postwarmup", r[key]["n"]) for r in runs]))
    return (np.nanmean(v), np.nanstd(v), np.nanmean(t), np.nanstd(t),
            np.nanmean(d), np.nanstd(d), ntot, ptot, comp,
            ntoi, nlai, np.nanmean(dc), cens, fin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--arrival", type=float, default=0.30)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--stage2", default="stage2_final.pt")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="PENETRATION.txt")
    ap.add_argument("--ps", default="0,25,50,75,100")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    pred = load_pred(args.stage2, dev)
    ps = [int(x) / 100.0 for x in args.ps.split(",")]

    fh = open(args.out, "w")

    def w(s=""):
        print(s, flush=True)
        fh.write(s + "\n"); fh.flush()

    w("=" * 78)
    w("MARKET-PENETRATION SWEEP  (fixed demand, vary equipped fraction p)")
    w("arrival=%.2f/step/end, horizon=%d, warmup=%d, reps=%d, K=%d, seed=%d"
      % (args.arrival, args.horizon, args.warmup, args.reps, args.K, args.seed))
    w("Equipped = PlanGrad-UAV (predictor+CBF-MPC); Unequipped = ORCA.")
    w("CR = per-agent conflict rate %%; Thr = passes/min; Delay = s.")
    w("=" * 78)

    for p in ps:
        runs = []
        for rep in range(args.reps):
            sim = PenetrationCorridor(pred, dev, K=args.K,
                                      arrival_rate=args.arrival,
                                      seed=args.seed + 1000 * rep + int(p * 97))
            m = sim.run(p_equip=p, horizon_steps=args.horizon,
                        warmup=args.warmup)
            runs.append(m)
        cra, crs, thr, thrs, dly, dlys, na, npa, compa, ntoa, nlaa, dcena, censa, fina = agg(runs, "all")
        cre, cres, thre, thres, dlye, dlyes, nae, npe, compe, ntoe, nlae, dcene, cense, fine = agg(runs, "equipped")
        cru, crus, thru, thrus, dlyu, dlyus, nau, npu, compu, ntou, nlau, dcenu, censu, finu = agg(runs, "unequipped")
        w("")
        w("### p = %d%% equipped" % int(p * 100))
        w("  ALL        : CR=%5.1f+-%4.1f%%  Thr=%5.2f+-%4.2f/min  Delay=%5.1f+-%4.1fs  (n=%d  pass=%d  completion=%4.1f%%  discard=%4.1f%%)"
          % (cra, crs, thr, thrs, dly, dlys, na, npa, compa, 100.0-compa))
        w("               discard cause: timeout=%d  lateral=%d  |  "
          "censored-delay UPPER bound (passers real + timeouts@cap, laterals excluded) = %5.1fs"
          % (ntoa, nlaa, dcena))
        _tot_all = fina + censa
        _cfrac = 100.0 * censa / _tot_all if _tot_all else float("nan")
        w("               RIGHT-CENSORING: still-in-corridor at sim end = %d  "
          "(finished post-warmup = %d ; censored share = %.1f%% of %d)"
          % (censa, fina, _cfrac, _tot_all))
        w("  EQUIPPED   : CR=%5.1f+-%4.1f%%  Thr=%5.2f+-%4.2f/min  Delay=%5.1f+-%4.1fs  (n=%d  pass=%d  completion=%4.1f%%  discard=%4.1f%%)"
          % (cre, cres, thre, thres, dlye, dlyes, nae, npe, compe, 100.0-compe))
        w("  UNEQUIPPED : CR=%5.1f+-%4.1f%%  Thr=%5.2f+-%4.2f/min  Delay=%5.1f+-%4.1fs  (n=%d  pass=%d  completion=%4.1f%%  discard=%4.1f%%)"
          % (cru, crus, thru, thrus, dlyu, dlyus, nau, npu, compu, 100.0-compu))

    w("")
    w("=" * 78)
    w("READING GUIDE")
    w(" System safety : ALL conflict rate vs p (does equipping the fleet make")
    w("   the whole corridor safer, and is the relation monotone?).")
    w(" Externality   : UNEQUIPPED conflict rate vs p. If it falls as p rises,")
    w("   equipped aircraft confer a POSITIVE externality on unequipped ones.")
    w(" Throughput    : ALL / group throughput and delay vs p (is safety")
    w("   bought at a throughput cost, or are both improved?).")
    w("=" * 78)
    fh.close()


if __name__ == "__main__":
    main()
