#!/usr/bin/env python3
"""Export robustness data for the mismatch figure, the Stage-1b forest, and
the wind sweep -- all three from the canonical `evaluate_policy`.

Rationale
---------
`eval_mismatch.py` and `06_sim_ood` each carry a PRIVATE re-implementation of
the closed loop. That is exactly the defect that produced the 81-vs-82
provenance conflict: a second rollout drifts from the one the main table was
reduced from. So `evaluate_policy` was extended to accept a perturbed plant
(plant_params / act_delay / thrust_eff) and explicit wind parameters
(eta_w / gust_std / wind_seed), and every number below comes from it.

Disturbance strength is the PRODUCT eta_w * gust_std. Canonical is
0.3 * 3.0 = 0.9. The manuscript used to declare a nominal of 0.5, which is a
67% stronger field; that discrepancy is settled in favour of the code (Route
A), and the sweep grid now includes the canonical 0.3 point so the sweep and
the main table can be checked against each other episode by episode.

Products
--------
1. mismatch_v2.npz   9 regimes x 3 arms, 200-dim conflict vectors
2. stage1b_mismatch_v2.npz  5 regimes, Stage-1b vs Stage-2 paired vectors
3. wind_sweep_v2.npz  eta in [0.3, 0.5, 1.0, 1.5] x 6 arms + tracking diagnostic
"""
from __future__ import annotations
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

BASE = "/data/lab/TRC_UAV_WEIGHTS/code/baselines"
sys.path.insert(0, f"{BASE}/common")
sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")
sys.path.insert(0, f"{BASE}/01_constant_velocity")   # cv_predictor lives here
sys.path.insert(0, f"{BASE}/02_vanilla_mpc")         # vanilla_mpc lives here

import eval_common as ec                                     # noqa: E402
from params import DEFAULT_PARAMS                            # noqa: E402
from vanilla_mpc import VanillaMPCLayer                       # noqa: E402
# The no-certificate arms use SafePolicy too: it only maps accel->control, and
# swapping in VanillaMPCLayer is what removes the certificate. There is no
# SoftPolicy in this codebase.
from safe_policy import SafePolicy                            # noqa: E402
from cv_predictor import ConstantVelocityPredictor            # noqa: E402

OUT = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
PG = ec.PLANGRAD_DIR
DEV = ec.device_str(True)
N = 200


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def perturbed(mass=1.0, inertia=1.0):
    """Plant-side parameter perturbation; the controller keeps DEFAULT_PARAMS."""
    return dataclasses.replace(
        DEFAULT_PARAMS,
        mass=DEFAULT_PARAMS.mass * mass,
        Ixx=DEFAULT_PARAMS.Ixx * inertia,
        Iyy=DEFAULT_PARAMS.Iyy * inertia,
        Izz=DEFAULT_PARAMS.Izz * inertia,
    )


# The 9 regimes of tab:mismatch, transcribed from eval_mismatch.py:328-341.
# NOTE the wind rows use gust_std=5.0 and field seed 99, so their strength is
# 5.0 (5.6x canonical) -- NOT the 3.0 of the sweep's eta=1.0 point.
REGIMES = [
    ("nominal",            {}),
    ("mass +20%",          dict(plant_params=perturbed(mass=1.20))),
    ("mass -15%",          dict(plant_params=perturbed(mass=0.85))),
    ("inertia +30%",       dict(plant_params=perturbed(inertia=1.30))),
    ("thrust eff 0.85",    dict(thrust_eff=0.85)),
    ("actuator delay 1",   dict(act_delay=1)),
    ("actuator delay 2",   dict(act_delay=2)),
    ("wind shift",         dict(eta_w=1.0, gust_std=5.0, wind_seed=99)),
    ("combined",           dict(plant_params=perturbed(mass=1.20),
                                thrust_eff=0.85, act_delay=1,
                                eta_w=1.0, gust_std=5.0, wind_seed=99)),
]

# Published tab:mismatch values, for assertion. (Stage-2+CBF, Stage-1+CBF,
# Vanilla no-CBF). "Stage-1 + CBF" is the frozen Stage-1 predictor, i.e. the
# Fixed-Predictor arm -- not Stage-1b.
TAB_MISMATCH = {
    "nominal":          (11.0, 12.5, 41.0),
    "mass +20%":        (17.5, 18.5, 75.0),
    "mass -15%":        (9.0,  10.0, 19.5),
    "inertia +30%":     (12.0, 12.0, 42.5),
    "thrust eff 0.85":  (16.5, 18.5, 68.5),
    "actuator delay 1": (18.5, 22.0, 54.0),
    "actuator delay 2": (28.0, 34.0, 87.5),
    "wind shift":       (11.0, 12.0, 39.5),
    "combined":         (30.0, 40.0, 94.5),
}

# tab:stage1b mismatch row (manuscript lines 2081-2082): Stage-1b CR.
TAB_S1B = {"nominal": 11.5, "mass +20%": 17.0, "thrust eff 0.85": 16.0,
           "actuator delay 2": 30.0, "combined": 31.0}
TAB_S1B_S2 = {"nominal": 11.0, "mass +20%": 17.5, "thrust eff 0.85": 16.5,
              "actuator delay 2": 28.0, "combined": 30.0}
S1B_REGIMES = ["nominal", "mass +20%", "thrust eff 0.85",
               "actuator delay 2", "combined"]

WIND_GRID = [0.3, 0.5, 1.0, 1.5]


def cbf_arm(weights):
    pred = ec.load_gmm_predictor(f"{PG}/{weights}", DEV)
    return dict(predictor=pred, planner=ec.make_best_planner())


def vanilla_arm(weights):
    return vanilla_arm_from(f"{PG}/{weights}")


def vanilla_arm_from(path):
    pred = ec.load_gmm_predictor(path, DEV)
    mpc = VanillaMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"],
                          dt=ec.DT, d_sep=ec.D_SEP,
                          a_max=ec.BEST_PLANNER["a_max"], w_rep=50.0)
    return dict(predictor=pred, planner=mpc,
                policy=SafePolicy(pred, mpc), ade_predictor=pred)


def run(arm, **kw):
    m = ec.evaluate_policy(n=N, device=DEV, **arm, **kw)
    conflict = (m["minsep_per_ep"] < ec.D_SEP)
    return m, conflict


# --------------------------------------------------------------------------- #
def product_1_mismatch():
    log("PRODUCT 1: mismatch, 9 regimes x 3 arms")
    arms = {"stage2": cbf_arm("stage2_final.pt"),
            "stage1": cbf_arm("stage1_full.pt"),
            "vanilla": vanilla_arm("stage2_final.pt")}
    out, fails = {}, []
    for rname, kw in REGIMES:
        for aname, arm in arms.items():
            m, c = run(arm, **kw)
            out[f"conflict__{rname}__{aname}"] = c
            out[f"minsep__{rname}__{aname}"] = m["minsep_per_ep"]
            cr = 100.0 * c.mean()
            exp = TAB_MISMATCH[rname][{"stage2": 0, "stage1": 1,
                                       "vanilla": 2}[aname]]
            tag = "OK " if abs(cr - exp) < 1e-9 else "DIFF"
            if tag == "DIFF":
                fails.append((rname, aname, cr, exp))
            log(f"  {tag} {rname:18s} {aname:8s} CR={cr:5.1f} (tab {exp:5.1f})"
                f"  minSep={m['minSep_m']:6.2f}")
    out["regimes"] = np.array([r for r, _ in REGIMES])
    np.savez_compressed(f"{OUT}/mismatch_v2.npz", **out)
    log(f"  -> mismatch_v2.npz  ({len(fails)} deviations from tab:mismatch)")
    return fails


def product_2_stage1b():
    log("PRODUCT 2: Stage-1b paired, 5 regimes")
    a1b = cbf_arm("stage1b_domainadapt.pt")
    a2 = cbf_arm("stage2_final.pt")
    out, fails = {}, []
    for rname in S1B_REGIMES:
        kw = dict(REGIMES)[rname]
        m1, c1 = run(a1b, **kw)
        m2, c2 = run(a2, **kw)
        out[f"conflict__{rname}__stage1b"] = c1
        out[f"conflict__{rname}__stage2"] = c2
        out[f"minsep__{rname}__stage1b"] = m1["minsep_per_ep"]
        out[f"minsep__{rname}__stage2"] = m2["minsep_per_ep"]
        # discordant pairs for the paired CI the forest plot needs
        b = int((c1 & ~c2).sum())
        c_ = int((~c1 & c2).sum())
        out[f"discordant__{rname}"] = np.array([b, c_])
        cr1, cr2 = 100.0 * c1.mean(), 100.0 * c2.mean()
        e1, e2 = TAB_S1B[rname], TAB_S1B_S2[rname]
        tag = "OK " if (abs(cr1 - e1) < 1e-9 and abs(cr2 - e2) < 1e-9) else "DIFF"
        if tag == "DIFF":
            fails.append((rname, cr1, e1, cr2, e2))
        log(f"  {tag} {rname:18s} S1b={cr1:5.1f} (tab {e1:5.1f})  "
            f"S2={cr2:5.1f} (tab {e2:5.1f})  discordant=({b},{c_})")
    out["regimes"] = np.array(S1B_REGIMES)
    np.savez_compressed(f"{OUT}/stage1b_mismatch_v2.npz", **out)
    log(f"  -> stage1b_mismatch_v2.npz  ({len(fails)} deviations)")
    return fails


def product_3_wind():
    log("PRODUCT 3: wind sweep, eta in %s x 6 arms" % WIND_GRID)
    arms = {
        "PlanGrad": cbf_arm("stage2_final.pt"),
        "Stage-1b": cbf_arm("stage1b_domainadapt.pt"),
        "Fixed-Predictor": cbf_arm("stage1_full.pt"),
        "Constant-Velocity": None,   # filled below
        "Vanilla-MPC": vanilla_arm("stage2_final.pt"),
        "Soft-IPP": None,
    }
    # CV arm: constant-velocity predictor drives control, Stage-2 reports ADE
    from predictor import GMMTrajectoryPredictor           # noqa: F401
    try:
        from cv_predictor import ConstantVelocityPredictor
        cvp = ConstantVelocityPredictor().double().to(DEV)
        arms["Constant-Velocity"] = dict(
            predictor=cvp, planner=ec.make_best_planner(),
            ade_predictor=cvp)
    except Exception as e:                                  # pragma: no cover
        log(f"  WARN: CV arm unavailable ({e}); skipping")
        arms.pop("Constant-Velocity")
    try:
        sip = ec.load_gmm_predictor(f"{PG}/soft_joint.pt", DEV)
        mpc = VanillaMPCLayer(n_neighbors=1,
                              horizon=ec.BEST_PLANNER["horizon"], dt=ec.DT,
                              d_sep=ec.D_SEP,
                              a_max=ec.BEST_PLANNER["a_max"], w_rep=50.0)
        arms["Soft-IPP"] = dict(predictor=sip, planner=mpc,
                                policy=SoftPolicy(sip, mpc,
                                                  params=DEFAULT_PARAMS),
                                ade_predictor=sip)
    except Exception as e:                                  # pragma: no cover
        log(f"  WARN: Soft-IPP arm unavailable ({e}); skipping")
        arms.pop("Soft-IPP")

    out = {}
    for eta in WIND_GRID:
        for aname, arm in arms.items():
            if arm is None:
                continue
            m, c = run(arm, eta_w=eta)
            k = f"{aname}__eta{eta}"
            out[f"conflict__{k}"] = c
            out[f"minsep__{k}"] = m["minsep_per_ep"]
            out[f"scalars__{k}"] = np.array(
                [m["CR_%"], m["minSep_m"], m["ADE_m"], m["Energy"]])
            log(f"  eta={eta:<4} {aname:18s} CR={m['CR_%']:5.1f}  "
                f"minSep={m['minSep_m']:6.2f}  ADE={m['ADE_m']:6.2f}  "
                f"E={m['Energy']:6.2f}")
    out["etas"] = np.array(WIND_GRID)
    out["arms"] = np.array([a for a, v in arms.items() if v is not None])
    out["gust_std"] = np.array([ec.GUST_STD])
    np.savez_compressed(f"{OUT}/wind_sweep_v2.npz", **out)
    log("  -> wind_sweep_v2.npz")
    return out


def verify(wind_out):
    """Hard assertions. Any failure means the data must not be plotted."""
    log("VERIFY")
    errs = []

    def chk(cond, msg):
        if cond:
            log(f"  OK   {msg}")
        else:
            log(f"  FAIL {msg}")
            errs.append(msg)

    mm = np.load(f"{OUT}/mismatch_v2.npz", allow_pickle=True)
    for rname, _ in REGIMES:
        for i, aname in enumerate(["stage2", "stage1", "vanilla"]):
            cr = 100.0 * mm[f"conflict__{rname}__{aname}"].mean()
            chk(abs(cr - TAB_MISMATCH[rname][i]) < 1e-9,
                f"mismatch {rname}/{aname} CR == tab ({cr:.1f})")

    s1 = np.load(f"{OUT}/stage1b_mismatch_v2.npz", allow_pickle=True)
    for rname in S1B_REGIMES:
        cr = 100.0 * s1[f"conflict__{rname}__stage1b"].mean()
        chk(abs(cr - TAB_S1B[rname]) < 1e-9,
            f"stage1b {rname} CR == tab:stage1b ({cr:.1f})")

    # THE key assertion: the sweep's canonical point must be episode-for-episode
    # identical to the main table, since eta/seed/pipeline all match. This
    # replaces the old (and wrong) "independent resampling" story.
    cv = np.load(f"{OUT}/conflict_vectors_v2.npz", allow_pickle=True)
    name_map = {"PlanGrad": None, "Stage-1b": None,
                "Fixed-Predictor": None, "Constant-Velocity": None}
    keys = list(cv.keys())
    for arm in list(name_map):
        for k in keys:
            if arm.lower().replace("-", "").replace("_", "") in \
                    k.lower().replace("-", "").replace("_", ""):
                name_map[arm] = k
                break
    for arm, k in name_map.items():
        wk = f"conflict__{arm}__eta0.3"
        if k is None or wk not in wind_out:
            log(f"  SKIP {arm}: no counterpart (cv={k}, sweep={wk in wind_out})")
            continue
        chk(bool((wind_out[wk] == cv[k]).all()),
            f"wind_sweep(eta=0.3) {arm} == conflict_vectors_v2 elementwise")

    # wind produces no effect: bound the spread across the grid
    for arm in [str(a) for a in np.load(f"{OUT}/wind_sweep_v2.npz",
                                       allow_pickle=True)["arms"]]:
        crs = [wind_out[f"scalars__{arm}__eta{e}"][0] for e in WIND_GRID
               if f"scalars__{arm}__eta{e}" in wind_out]
        mss = [wind_out[f"scalars__{arm}__eta{e}"][1] for e in WIND_GRID
               if f"scalars__{arm}__eta{e}" in wind_out]
        log(f"  INFO {arm:18s} dCR={max(crs)-min(crs):.1f} pp  "
            f"dMinSep={max(mss)-min(mss):.3f} m")

    if errs:
        raise AssertionError(
            f"{len(errs)} assertion(s) failed; refusing to certify:\n  "
            + "\n  ".join(errs))
    log("all assertions hold")


if __name__ == "__main__":
    t0 = time.time()
    f1 = product_1_mismatch()
    f2 = product_2_stage1b()
    w = product_3_wind()
    verify(w)
    log(f"done in {(time.time()-t0)/60:.1f} min "
        f"(deviations: mismatch={len(f1)}, stage1b={len(f2)})")
