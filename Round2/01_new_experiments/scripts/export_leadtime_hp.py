#!/usr/bin/env python3
"""Planner-horizon sensitivity of the detection-lead-time result.

WHY THIS EXPERIMENT EXISTS
  The deployment planner uses H_p=15 with dt=0.2 s, so its planning horizon is
  H_p*dt = 3.0 s exactly. The published lead-time sweep collapses at 3 s
  (Stage-2: 82.5% at 1 s, 11.0% at 2 s, 0.0% at 3 s and beyond). The transition
  therefore coincides numerically with the planner's own horizon, and the
  manuscript cannot claim that surveillance infrastructure "need only guarantee
  3 s of warning" until that coincidence is ruled out: 3 s may be special only
  because the planner looks 3 s ahead.

  The confound is geometric, not incidental. eval_leadtime.py sweeps lead time by
  moving closest approach to t_cpa = h/dt while holding H_p fixed, so for h <= 3
  the CPA sits inside the planner's horizon from the very first step, and h=3 is
  the largest lead time that is both fully visible at t=0 and uses the whole
  horizon. The collapse from 1 s to 3 s can be read entirely as "actionable time
  grew from 1 s to 3 s", with no claim about surveillance implied.

  A simpler version of the confound is already refuted by the published data:
  if 3 s were special only because the horizon ends there, conflicts should
  reappear once CPA falls outside the horizon, yet Stage-2 stays at 0.0% for
  every h >= 3. A receding CPA still enters the rolling horizon later and still
  leaves 3 s to act. What remains untested, and what this script tests, is
  whether the LOCATION of the transition tracks H_p*dt.

WHAT IT DISCRIMINATES
  If the collapse moves with the horizon -- near 1.6 s for H_p=8, near 5.0 s for
  H_p=25 -- then 3 s is a property of the planner, and the operational claim must
  be restated as a design relation: detection lead time must cover the planner's
  horizon. That is a more useful statement than a magic number.
  If all three horizons collapse near 3 s, the threshold is set by the encounter
  geometry and closing dynamics rather than the horizon, and the original claim
  survives considerably strengthened.

WHY A SEPARATE SCRIPT
  eval_leadtime.py hard-codes HP=15 as a module constant used in five places
  (lines 149, 162, 175, 182, 258). Parameterising it in place would touch the
  path that produced Figure 10, which is finished and pushed. This script
  imports that module and reuses its encounter generator, planner wiring and
  inner control law verbatim, overriding only the horizon. Nothing in
  eval_leadtime.py is modified.

  Because the reused functions read the module-level HP, the override is done by
  rebinding eval_leadtime.HP around each call. That is deliberate and asserted:
  the value is checked before and restored after, so a failure cannot silently
  leave the module in a mutated state for a later call.

PROVENANCE
  Predictors stage2_final.pt and stage1_full.pt from Round1/04_weights.
  Held-out encounters 2500-2999, n=200, seed 12345, eta_w=0.3, wind seed 7,
  d_sep=30 m, gamma=0.1, a_max=20, dt=0.2. gamma is the manuscript's name for
  what the code calls alpha. Rollout length T = t_cpa + H_p + 2, so it depends
  on both the lead time and the horizon; the oracle arm receives true neighbour
  futures over the same horizon.
"""
import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch

SIM = "/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim"
W = "/data/lab/TRC_UAV_WEIGHTS/Round1/04_weights"
OUT_DIR = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
TXT = "/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim/LEADTIME_HP_SENS.txt"

sys.path.insert(0, SIM)
import eval_leadtime as EL          # noqa: E402
from seeding import set_seed        # noqa: E402

# Published Stage-2 lead-time curve at the deployment horizon, from
# leadtime_v2.npz / LEADTIME.txt. The H_p=15 column of this sweep must
# reproduce it, or the override plumbing is wrong.
PUB_H15_S2 = {1.0: 82.5, 2.0: 11.0, 3.0: 0.0, 4.0: 0.0, 5.0: 0.0, 7.0: 0.0}
PUB_H15_S1 = {1.0: 86.0, 2.0: 13.0, 3.0: 1.5, 4.0: 7.0, 5.0: 10.0, 7.0: 8.5}


@contextlib.contextmanager
def horizon(hp):
    """Temporarily rebind eval_leadtime.HP, restoring it unconditionally.

    The reused rollout reads HP from module scope in five places, so this is
    the only way to sweep it without editing that file. The original value is
    asserted on entry and restored in a finally block: a raised exception must
    not leave a mutated horizon behind for the next configuration.
    """
    old = EL.HP
    assert old == 15, f"eval_leadtime.HP was {old}, expected the pristine 15"
    EL.HP = int(hp)
    try:
        yield
    finally:
        EL.HP = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hps", default="8,15,25")
    ap.add_argument("--horizons", default="1,2,3,4,5,7")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--arms", default="stage2,stage1,oracle")
    ap.add_argument("--out", default="leadtime_hp_v2.npz")
    args = ap.parse_args()

    hps = [int(x) for x in args.hps.split(",")]
    hs = [float(x) for x in args.horizons.split(",")]
    arms = args.arms.split(",")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    lines = []

    def w(s=""):
        print(s, flush=True)
        lines.append(s)

    preds = {}
    if "stage2" in arms:
        preds["stage2"] = EL.load_pred(os.path.join(W, "stage2_final.pt"), dev)
    if "stage1" in arms:
        preds["stage1"] = EL.load_pred(os.path.join(W, "stage1_full.pt"), dev)

    w("PLANNER-HORIZON SENSITIVITY OF THE LEAD-TIME RESULT")
    w(f"  n={args.n} held-out 2500-2999, seed={args.seed}, eta_w=0.3, "
      f"wind seed 7, d_sep=30 m, gamma=0.1, a_max=20, dt={EL.DT}")
    w("  Lead time is swept by moving closest approach to t_cpa = h/dt; the")
    w("  planner horizon H_p is swept independently. Rollout length")
    w("  T = t_cpa + H_p + 2, so it grows with both.")
    w("  QUESTION: does the collapse sit at 3 s regardless of H_p, or does it")
    w("  track H_p*dt? The latter would mean 3 s is a property of the planner")
    w("  and not a surveillance requirement.")
    w("=" * 78)

    store = {}
    t0 = time.time()
    for hp in hps:
        w()
        w(f"H_p = {hp}  (planning horizon {hp * EL.DT:.1f} s)")
        w(f"  {'h(s)':>5s} {'T':>4s} " + " ".join(f"{a:>9s}" for a in arms)
          + f" {'act-lim':>8s} {'pred-lim':>9s}")
        for h in hs:
            t_cpa = int(round(h / EL.DT))
            with horizon(hp):
                T = t_cpa + EL.HP + 2

                def mk():
                    return EL.LeadTimeEncounters(
                        EL.GUAM_MAT, range(2500, 3000), seed=args.seed,
                        t_cpa=t_cpa)

                got = {}
                for a in arms:
                    set_seed(args.seed)
                    if a == "oracle":
                        c = EL.rollout_condition(mk(), "oracle", dev,
                                                 args.n, T)
                    else:
                        c = EL.rollout_condition(mk(), "stage", dev, args.n, T,
                                                 pred=preds[a])
                    got[a] = c
                    store[f"conflict__{a}__hp{hp}__h{h:g}"] = c
                    store[f"cr__{a}__hp{hp}__h{h:g}"] = np.array(
                        [100.0 * c.mean()])
            # attribution among Stage-2 conflicts, same definition as fig10
            al = pl = -1
            if "stage2" in got and "oracle" in got:
                al = int((got["stage2"] & got["oracle"]).sum())
                pl = int((got["stage2"] & ~got["oracle"]).sum())
                store[f"attrib__hp{hp}__h{h:g}"] = np.array([al, pl])
            w(f"  {h:5.1f} {T:4d} "
              + " ".join(f"{100.0 * got[a].mean():8.1f}%" for a in arms)
              + (f" {al:8d} {pl:9d}" if al >= 0 else "")
              + f"   [{(time.time() - t0) / 60:5.1f} min]")

    assert EL.HP == 15, f"module horizon left mutated at {EL.HP}"

    # ---- the H_p=15 column must reproduce the published curve ---------------
    if 15 in hps:
        bad = []
        for h in hs:
            for arm, pub in (("stage2", PUB_H15_S2), ("stage1", PUB_H15_S1)):
                if arm not in arms or h not in pub:
                    continue
                k = f"cr__{arm}__hp15__h{h:g}"
                got = float(store[k][0])
                if abs(got - pub[h]) > 0.01:
                    bad.append(f"{arm} h={h}: {got:.1f}% != published "
                               f"{pub[h]}%")
        if bad:
            raise AssertionError(
                "the H_p=15 column does not reproduce the published lead-time "
                "sweep, so the horizon override is not running the same "
                "experiment:\n  " + "\n  ".join(bad))
        w()
        w("  control: the H_p=15 column reproduces the published lead-time "
          "curve exactly")

    # ---- where does the collapse sit for each H_p? --------------------------
    w()
    w("TRANSITION LOCATION (first lead time at which Stage-2 reaches 0%)")
    trans = {}
    for hp in hps:
        first = None
        for h in hs:
            k = f"cr__stage2__hp{hp}__h{h:g}"
            if k in store and float(store[k][0]) == 0.0:
                first = h
                break
        trans[hp] = first
        w(f"  H_p={hp:2d}  horizon {hp * EL.DT:.1f} s  ->  collapse at "
          + (f"{first:.0f} s" if first is not None
             else "not reached within the swept grid"))
        store[f"transition__hp{hp}"] = np.array(
            [np.nan if first is None else first])

    w()
    vals = [trans[hp] for hp in hps if trans[hp] is not None]
    if len(vals) == len(hps) and len(set(vals)) == 1:
        w(f"  READ: the collapse sits at {vals[0]:.0f} s for every H_p tested,")
        w("  so its location is NOT set by the planning horizon. The threshold")
        w("  reflects the encounter geometry and closing dynamics, and the")
        w("  surveillance reading survives.")
    else:
        w("  READ: the collapse location DIFFERS across H_p, so 3 s is a")
        w("  property of the deployment planner rather than a surveillance")
        w("  requirement. The operational claim must be restated as a design")
        w("  relation -- detection lead time must cover the planner horizon --")
        w("  and the bare '3 s of warning' wording must not be used.")
        for hp in hps:
            t = trans[hp]
            w(f"    H_p={hp:2d}: horizon {hp * EL.DT:.1f} s, collapse "
              + (f"{t:.0f} s" if t is not None else "beyond grid"))

    store["hps"] = np.array(hps)
    store["horizons_s"] = np.array(hs)
    store["arms"] = np.array(arms)
    store["dt"] = np.array([EL.DT])
    store["n"] = np.array([args.n])

    out = os.path.join(OUT_DIR, args.out)
    np.savez(out, **store)
    w()
    w(f"wrote {out}")
    with open(TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {TXT}")

    with open(os.path.join(OUT_DIR, "leadtime_hp_v2.json"), "w") as f:
        json.dump({
            "dt": EL.DT, "n": args.n, "seed": args.seed,
            "hps": hps, "horizons_s": hs,
            "planning_horizon_s": {str(hp): hp * EL.DT for hp in hps},
            "cr": {f"{a}__hp{hp}__h{h:g}":
                   float(store[f"cr__{a}__hp{hp}__h{h:g}"][0])
                   for hp in hps for h in hs for a in arms
                   if f"cr__{a}__hp{hp}__h{h:g}" in store},
            "transition_s": {str(hp): (None if trans[hp] is None
                                       else float(trans[hp])) for hp in hps},
        }, f, indent=2)
    print("wrote leadtime_hp_v2.json")


if __name__ == "__main__":
    main()
