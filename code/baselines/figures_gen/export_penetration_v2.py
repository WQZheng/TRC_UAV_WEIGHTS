#!/usr/bin/env python3
"""Export per-replication corridor penetration data for Figure 13.

WHY THIS SCRIPT EXISTS
  The published penetration tables print mean +/- SD only. The per-replication
  values existed in memory inside run_penetration.py -- collected in a list,
  aggregated, and dropped -- so the six points behind every mean were never
  written to disk. Six replications is few enough that a mean alone invites
  over-reading: the review flagged the mid-penetration conflict-rate bump as a
  descriptive pattern with no statistical support, and the honest way to show
  that is to draw the six points and let their spread speak. So the sweep is
  re-run with the individual replications retained.

  Re-running is legitimate because the seeding is a deterministic function of
  the replication index and the equipped fraction:
      seed = base_seed + 1000 * rep + int(p * 97)
  (run_penetration.py:78). Every replication is therefore independently
  reproducible, and the aggregate of the re-run must reproduce the published
  table. That reproduction is asserted here, per configuration, and the script
  refuses to write output if any published mean is not recovered.

  NOTE ON n: the published n is the SUM of per-replication counts, not a mean
  (run_penetration.py:38, np.sum(ns)). A single replication yields roughly
  n/reps agents, so comparing one replication's n against the table looks like
  a factor-six discrepancy when it is only the aggregation convention.

METRIC DEFINITIONS come from penetration_sim.py and are not re-derived here.
  conflict rate : per-agent percentage, an agent counts once if it ever came
                  within the separation standard of another agent
  throughput    : passes per minute over the post-warmup window
  delay         : seconds relative to the free-flow corridor transit
  discard       : an agent that never passed, split by cause into lateral exit
                  and timeout. Every published configuration reports timeout=0,
                  so discards are lateral exits and the panel can be drawn as a
                  two-way split. That is asserted, not assumed.

PROVENANCE
  high demand  arrival=0.16/step/end, PENETRATION.txt and PENETRATION_DISC2.txt
  low demand   arrival=0.06/step/end, PENETRATION_LOW.txt and
               PENETRATION_LOW_DISC2.txt
  both         horizon=400, warmup=100, reps=6, K=3, seed=12345,
               equipped = PlanGrad-UAV (predictor stage2_final.pt + CBF-MPC),
               unequipped = ORCA.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = "/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim"
# The completion panel needs the discard CAUSE, and only the disc2 variant of
# the simulator records it (penetration_sim_disc2.py:69, a per-agent `reason`
# field, with n_lateral / n_timeout in its metrics dict). The base
# penetration_sim.py returns `passed` but not why an agent failed to pass, so
# it cannot support a two-way completion split. Importing the base module and
# reading a nonexistent key would silently yield NaN, which is exactly the kind
# of quiet degradation the assertions below are meant to make impossible.
DISC2_DIR = ("/data/lab/TRC_UAV_WEIGHTS/Round1/05_results/robustness/"
             "p0_referee")
WEIGHTS = "/data/lab/TRC_UAV_WEIGHTS/plangrad_sim/stage2_final.pt"
OUT_DIR = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")

PS = [0.0, 0.25, 0.50, 0.75, 1.0]
GROUPS = ["all", "equipped", "unequipped"]

# Published means the re-run must reproduce. Source files named above.
# demand -> p -> group -> (CR, Thr, Delay)
PUBLISHED = {
    "high": {
        0.00: {"all": (50.3, 44.83, 6.5), "unequipped": (50.3, 44.83, 6.5)},
        0.25: {"all": (57.2, 44.83, 4.3), "equipped": (49.8, 13.33, 1.3),
               "unequipped": (59.7, 31.50, 5.6)},
        0.50: {"all": (53.3, 45.33, 3.9), "equipped": (55.8, 19.33, 1.7),
               "unequipped": (51.7, 26.00, 5.5)},
        0.75: {"all": (52.1, 46.83, 3.0), "equipped": (51.2, 34.50, 2.4),
               "unequipped": (53.7, 12.33, 4.4)},
        1.00: {"all": (47.7, 47.67, 2.6), "equipped": (47.7, 47.67, 2.6)},
    },
    "low": {
        0.00: {"all": (34.4, 18.67, 2.9), "unequipped": (34.4, 18.67, 2.9)},
        0.25: {"all": (30.5, 18.50, 2.3), "equipped": (18.3, 4.33, 1.7),
               "unequipped": (34.3, 14.17, 2.5)},
        0.50: {"all": (36.7, 18.00, 3.1), "equipped": (30.8, 6.83, 2.1),
               "unequipped": (42.7, 11.17, 3.6)},
        0.75: {"all": (23.2, 15.17, 1.6), "equipped": (20.1, 10.67, 1.1),
               "unequipped": (29.2, 4.50, 2.4)},
        1.00: {"all": (24.3, 16.83, 1.4), "equipped": (24.3, 16.83, 1.4)},
    },
}
# Published lateral-exit counts, ALL group, from the DISC2 files. timeout=0
# everywhere, which is what licenses the two-way completion split.
PUB_LATERAL = {"high": {0.00: 0, 0.25: 4, 0.50: 12, 0.75: 13, 1.00: 25},
               "low": {0.00: 1, 0.25: 3, 0.50: 10, 0.75: 11, 1.00: 11}}
ARRIVAL = {"high": 0.16, "low": 0.06}


def load_predictor(dev):
    sys.path.insert(0, SIM_DIR)
    from predictor import GMMTrajectoryPredictor
    net = GMMTrajectoryPredictor(T=30, K=5).double().to(dev)
    sd = torch.load(WEIGHTS, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict(sd)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--demands", default="high,low")
    ap.add_argument("--tol_cr", type=float, default=0.15,
                    help="tolerance in pp when reproducing a published CR")
    ap.add_argument("--out", default="penetration_v2.npz")
    args = ap.parse_args()

    sys.path.insert(0, SIM_DIR)
    sys.path.insert(0, DISC2_DIR)
    from penetration_sim_disc2 import PenetrationCorridor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_predictor(dev)
    demands = args.demands.split(",")

    store = {}
    t_start = time.time()
    for dem in demands:
        arrival = ARRIVAL[dem]
        for p in PS:
            per_rep = {g: {"cr": [], "thr": [], "dly": [], "n": []}
                       for g in GROUPS}
            lateral = []
            timeout = []
            passed = []
            total = []
            for rep in range(args.reps):
                seed = args.seed + 1000 * rep + int(p * 97)
                sim = PenetrationCorridor(net, dev, K=args.K,
                                          arrival_rate=arrival, seed=seed)
                m = sim.run(p_equip=p, horizon_steps=args.horizon,
                            warmup=args.warmup)
                for g in GROUPS:
                    per_rep[g]["cr"].append(m[g]["conflict_rate"])
                    per_rep[g]["thr"].append(m[g]["throughput"])
                    per_rep[g]["dly"].append(m[g]["mean_delay"])
                    per_rep[g]["n"].append(m[g]["n"])
                a = m["all"]
                # Fail loudly if the simulator variant lacks the cause split,
                # rather than storing NaN and discovering it at figure time.
                for need in ("n_lateral", "n_timeout", "passed", "n"):
                    assert need in a, (
                        f"metrics dict has no {need!r}; wrong simulator "
                        f"variant imported?")
                lateral.append(a["n_lateral"])
                timeout.append(a["n_timeout"])
                passed.append(a["passed"])
                total.append(a["n"])
                el = time.time() - t_start
                print(f"  {dem:4s} p={p:4.2f} rep {rep}  "
                      f"CR={m['all']['conflict_rate']:5.1f}%  "
                      f"n={m['all']['n']:3d}  [{el / 60:5.1f} min]",
                      flush=True)
            for g in GROUPS:
                for k, v in per_rep[g].items():
                    store[f"{dem}__p{int(round(p * 100)):03d}__{g}__{k}"] = \
                        np.array(v, float)
            store[f"{dem}__p{int(round(p * 100)):03d}__lateral"] = \
                np.array(lateral, float)
            store[f"{dem}__p{int(round(p * 100)):03d}__timeout"] = \
                np.array(timeout, float)
            store[f"{dem}__p{int(round(p * 100)):03d}__passed"] = \
                np.array(passed, float)
            store[f"{dem}__p{int(round(p * 100)):03d}__total"] = \
                np.array(total, float)

    # ---- reproduce the published tables, or refuse to write -----------------
    bad = []
    for dem in demands:
        for p in PS:
            key = int(round(p * 100))
            for g, (cr, thr, dly) in PUBLISHED[dem][p].items():
                got_cr = float(np.nanmean(store[f"{dem}__p{key:03d}__{g}__cr"]))
                got_thr = float(np.nanmean(store[f"{dem}__p{key:03d}__{g}__thr"]))
                got_dly = float(np.nanmean(store[f"{dem}__p{key:03d}__{g}__dly"]))
                if abs(got_cr - cr) > args.tol_cr:
                    bad.append(f"{dem} p={p} {g} CR {got_cr:.2f} != {cr}")
                if abs(got_thr - thr) > 0.02:
                    bad.append(f"{dem} p={p} {g} Thr {got_thr:.3f} != {thr}")
                if abs(got_dly - dly) > 0.06:
                    bad.append(f"{dem} p={p} {g} Delay {got_dly:.2f} != {dly}")
            # Every discard is a lateral exit. Asserted, because the two-way
            # completion panel is only honest if timeout is genuinely zero.
            to = store[f"{dem}__p{key:03d}__timeout"]
            if to.sum() != 0:
                bad.append(f"{dem} p={p} timeout={to.sum()} != 0, so discards "
                           f"are not all lateral exits and the completion "
                           f"panel cannot be a two-way split")
            lat = store[f"{dem}__p{key:03d}__lateral"]
            want = PUB_LATERAL[dem][p]
            if int(lat.sum()) != want:
                bad.append(f"{dem} p={p} lateral {int(lat.sum())} != {want}")
            # passed + discarded must exhaust the agents; a missing category
            # would make the completion share meaningless.
            tot = store[f"{dem}__p{key:03d}__total"]
            ps_ = store[f"{dem}__p{key:03d}__passed"]
            if not np.allclose(ps_ + lat + to, tot):
                bad.append(f"{dem} p={p} passed+lateral+timeout != n")
    if bad:
        raise AssertionError(
            "re-run does not reproduce the published penetration tables, so "
            "the per-replication values cannot be trusted:\n  "
            + "\n  ".join(bad))

    store["ps"] = np.array(PS, float)
    store["reps"] = args.reps
    store["seed"] = args.seed
    store["horizon"] = args.horizon
    store["warmup"] = args.warmup
    store["K"] = args.K
    store["demands"] = np.array(demands)
    store["arrival"] = np.array([ARRIVAL[d] for d in demands], float)

    out = os.path.join(OUT_DIR, args.out)
    np.savez(out, **store)
    print(f"\nwrote {out}")
    print(f"  reproduced every published mean for {demands} "
          f"({len(PS)} penetrations x {args.reps} replications each)")
    for dem in demands:
        for p in PS:
            key = int(round(p * 100))
            cr = store[f"{dem}__p{key:03d}__all__cr"]
            print(f"  {dem:4s} p={p:4.2f}  ALL CR per rep: "
                  + " ".join(f"{v:5.1f}" for v in cr)
                  + f"  mean {np.nanmean(cr):5.1f}  sd {np.nanstd(cr):4.1f}")


if __name__ == "__main__":
    main()
