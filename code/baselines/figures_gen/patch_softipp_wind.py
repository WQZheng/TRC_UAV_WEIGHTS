#!/usr/bin/env python3
"""Add the missing Soft-IPP arm to wind_sweep_v2.npz.

Why this exists: the sweep ran 5 arms, not 6. export_robustness.py wrapped the
Soft-IPP construction in try/except and pointed at
`{PLANGRAD_DIR}/soft_joint.pt`, but that checkpoint lives in the baseline
directory (`code/baselines/04_soft_ipp/soft_joint.pt`). The except branch logged
a WARN and dropped the arm, so a wrong path became a silently missing arm
instead of a crash. That is the real defect -- a self-check cannot fire on data
that was never produced.

This script only APPENDS the four Soft-IPP cells. Every existing key is copied
through byte-for-byte and re-verified afterwards, so the five arms that already
passed their assertions cannot be disturbed.
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np

BASE = "/data/lab/TRC_UAV_WEIGHTS/code/baselines"
sys.path.insert(0, f"{BASE}/common")
sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")
sys.path.insert(0, f"{BASE}/02_vanilla_mpc")

import eval_common as ec                                   # noqa: E402
from vanilla_mpc import VanillaMPCLayer                     # noqa: E402
from safe_policy import SafePolicy                          # noqa: E402

OUT = f"{BASE}/figures_gen/fig_data/wind_sweep_v2.npz"
WEIGHTS = f"{BASE}/04_soft_ipp/soft_joint.pt"
GRID = [0.3, 0.5, 1.0, 1.5]
DEV = ec.device_str(True)
ARM = "Soft-IPP"

# Published Soft-IPP nominal values (Table 1) for a sanity read, not an
# assertion: the sweep's eta=0.3 point is the canonical operating point.
TABLE_CR = 53.0
TABLE_EFFORT = 16.975


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    if not os.path.exists(WEIGHTS):
        sys.exit(f"missing {WEIGHTS}")
    old = dict(np.load(OUT, allow_pickle=True))
    arms = [str(a) for a in old["arms"]]
    if ARM in arms:
        sys.exit(f"{ARM} already present; refusing to overwrite")
    log(f"existing arms: {arms}")

    pred = ec.load_gmm_predictor(WEIGHTS, DEV)
    mpc = VanillaMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"],
                          dt=ec.DT, d_sep=ec.D_SEP,
                          a_max=ec.BEST_PLANNER["a_max"], w_rep=50.0)
    arm = dict(predictor=pred, planner=mpc,
               policy=SafePolicy(pred, mpc), ade_predictor=pred)

    new = {}
    for eta in GRID:
        m = ec.evaluate_policy(n=200, device=DEV, eta_w=eta, **arm)
        c = (m["minsep_per_ep"] < ec.D_SEP)
        k = f"{ARM}__eta{eta}"
        new[f"conflict__{k}"] = c
        new[f"minsep__{k}"] = m["minsep_per_ep"]
        new[f"scalars__{k}"] = np.array(
            [m["CR_%"], m["minSep_m"], m["ADE_m"], m["Energy"]])
        log(f"  eta={eta:<4} CR={m['CR_%']:5.1f}  minSep={m['minSep_m']:6.2f}  "
            f"ADE={m['ADE_m']:6.2f}  E={m['Energy']:6.2f}")
        # CR_% must be the reduction of the very vector we store
        assert abs(m["CR_%"] - 100.0 * c.mean()) < 1e-9

    cr03 = float(new[f"scalars__{ARM}__eta0.3"][0])
    e03 = float(new[f"scalars__{ARM}__eta0.3"][3])
    log(f"canonical point eta=0.3: CR={cr03:.1f} (Table 1 {TABLE_CR})  "
        f"Energy={e03:.3f} (Table 1 {TABLE_EFFORT})")
    if abs(cr03 - TABLE_CR) > 1e-9:
        log(f"  NOTE: CR differs from Table 1 by {cr03 - TABLE_CR:+.1f} pp")

    merged = {**old, **new}
    merged["arms"] = np.array(arms + [ARM])
    np.savez_compressed(OUT, **merged)
    log(f"wrote {OUT}")

    # ---- re-verify: nothing pre-existing may have moved ----
    chk = np.load(OUT, allow_pickle=True)
    for k, v in old.items():
        if k == "arms":
            continue
        assert np.array_equal(chk[k], v), f"pre-existing key {k} changed!"
    log(f"all {len(old)-1} pre-existing arrays byte-identical")

    crs = [float(chk[f"scalars__{ARM}__eta{e}"][0]) for e in GRID]
    mss = [float(chk[f"scalars__{ARM}__eta{e}"][1]) for e in GRID]
    log(f"{ARM}: dCR={max(crs)-min(crs):.1f} pp  "
        f"dMinSep={max(mss)-min(mss):.3f} m")

    log("--- full sweep, all six arms ---")
    worst_cr = worst_ms = 0.0
    for a in [str(x) for x in chk["arms"]]:
        c = [float(chk[f"scalars__{a}__eta{e}"][0]) for e in GRID]
        s = [float(chk[f"scalars__{a}__eta{e}"][1]) for e in GRID]
        worst_cr = max(worst_cr, max(c) - min(c))
        worst_ms = max(worst_ms, max(s) - min(s))
        log(f"  {a:20s} " + " ".join(f"{x:5.1f}" for x in c)
            + f"   dCR={max(c)-min(c):.1f} pp  dMinSep={max(s)-min(s):.3f} m")
    log(f"ACROSS ALL SIX ARMS: dCR <= {worst_cr:.1f} pp "
        f"({worst_cr/100*200:.0f} of 200 episodes), dMinSep <= {worst_ms:.3f} m")


if __name__ == "__main__":
    main()
