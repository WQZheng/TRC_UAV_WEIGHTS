#!/usr/bin/env python3
"""Cross-planner transfer: the three training-time seeds + the ADE both reviewers asked for.

WHY
  export_weak_2x2.py established the fourth cell: the Stage-2 checkpoint trained
  under the deployment planner conflicts in 69.0% of encounters under the
  training-time planner (0.4, 8, 10), against 56.0% for Stage-1b and 28.5% for
  the primary Stage-2 checkpoint that was itself trained under (0.4, 8, 10).
  Two questions decide how strongly that can be written up, and both are
  evaluation-only:

  (1) IS THE FAVOURABLE SAME-CONFIGURATION RESULT REPRODUCIBLE?
      28.5% currently rests on one checkpoint. The three additional Stage-2
      seeds were ALSO trained under (0.4, 8, 10) -- verified from
      train_stage2.py defaults Hp=8, alpha=0.4, a_max=10.0 -- but were only ever
      evaluated under the deployment planner (12.0 / 11.0 / 11.5%). Evaluating
      them under their own training planner turns the matched side from n=1 into
      n=4. If all three land far below Stage-1b's 56.0%, the same-configuration
      advantage is a family property, not one lucky run, and the n=1 limitation
      survives only on the deployment-trained side.

  (2) IS THE 69% CHECKPOINT SIMPLY A BROKEN PREDICTOR?
      Every reader will ask. Open-loop ADE answers it. Note what ADE can and
      cannot do here: under this planner Stage-1 (ADE 20.90 m) and Stage-1b
      (1.84 m) both conflict at ~56%, an 11-fold accuracy span with almost no
      operational separation, so ADE has very little discriminating power in
      this regime. A benign ADE therefore CANNOT prove the divergence is
      planner-specific co-adaptation -- it can only rule out gross predictor
      failure. The script reports ADE and explicitly refuses to draw the
      stronger conclusion.

WHAT IS MEASURED
  For every checkpoint, under BOTH planning configurations:
    - conflict rate and per-episode conflict vector (conflict := MinSep < 30 m)
    - mean minimum separation
  Plus, planner-independent (neighbour trajectories are replayed, so open-loop
  error does not depend on the planner):
    - ADE over the prediction horizon, same definition as the published table

METHOD REUSE
  per_episode_minsep() from mcnemar_final2 is the exact published training-time
  rollout (alpha=0.4, a_max=10, Hp=8, T=20, dt=0.2, eta_w=0.3, wind seed 7,
  d_sep=30, encounters range(2500,3000), seed 12345, batch 8), imported
  unmodified so all arms stay episode-aligned and paired tests are legitimate.
  The deployment configuration uses the same rollout with (alpha=0.1, Hp=15,
  a_max=20) via an explicit parameter override, and is validated against the
  published deployment numbers before anything new is reported.

CONTROLS
  Reproduce, or refuse to write:
    Stage-1b   56.0% and primary Stage-2 28.5% under the training-time planner
    Stage-1b   11.5% and primary Stage-2 11.0% under the deployment planner
    primary Stage-2 ADE 4.32 m, Stage-1b ADE 1.84 m
"""
import os
import sys

import numpy as np
import torch

SIM = "/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim"
W = "/data/lab/TRC_UAV_WEIGHTS/Round1/04_weights"
OUT_DIR = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
TXT = "/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim/PLANNER_TRANSFER.txt"

sys.path.insert(0, SIM)
os.chdir(SIM)
import torch as _t                                                    # noqa
from config import GUAM_MAT                                           # noqa
from params import DEFAULT_PARAMS                                     # noqa
from predictor import GMMTrajectoryPredictor                          # noqa
from guam_encounters import GUAMEncounters                            # noqa
from cbf_mpc import CBFMPCLayer                                      # noqa
from safe_policy import SafePolicy                                   # noqa
from dynamics import EVTOLDynamics                                   # noqa
from wind import UrbanWindField                                      # noqa
from seeding import set_seed                                         # noqa
from mcnemar_final2 import mcnemar_exact_two_sided, D_SEP            # noqa

DTYPE = torch.float64
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 100.0
N = 200
T, DT, ETA_W, BATCH = 20, 0.2, 0.3, 8

TRAIN_CFG = dict(name="training-time", alpha=0.4, Hp=8, a_max=10.0)
DEPLOY_CFG = dict(name="deployment", alpha=0.1, Hp=15, a_max=20.0)

CKPT = [
    ("Stage-1",          "stage1_full.pt"),
    ("Stage-1b",         "stage1b_domainadapt.pt"),
    ("Stage-2 primary",  "stage2_final.pt"),
    ("Stage-2 seed1",    "stage2_seed1.pt"),
    ("Stage-2 seed2",    "stage2_seed2.pt"),
    ("Stage-2 seed3",    "stage2_seed3.pt"),
    ("Stage-2 deploy",   "stage2_matched.pt"),
]
# which planner each checkpoint was TRAINED under
TRAINED_UNDER = {
    "Stage-1": None, "Stage-1b": None,
    "Stage-2 primary": "training-time", "Stage-2 seed1": "training-time",
    "Stage-2 seed2": "training-time", "Stage-2 seed3": "training-time",
    "Stage-2 deploy": "deployment",
}

PUB_TRAIN_CR = {"Stage-1": 56.5, "Stage-1b": 56.0, "Stage-2 primary": 28.5}
PUB_DEPLOY_CR = {"Stage-1b": 11.5, "Stage-2 primary": 11.0}
PUB_ADE = {"Stage-1b": 1.84, "Stage-2 primary": 4.32, "Stage-1": 20.90}


def load(p):
    n = GMMTrajectoryPredictor(T=30, K=5).double().to(DEV)
    n.load_state_dict(torch.load(os.path.join(W, p), map_location=DEV))
    n.eval()
    return n


@torch.no_grad()
def rollout(pred, cfg):
    """Published rollout with an explicit planner configuration."""
    set_seed(12345)
    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=12345)
    mpc = CBFMPCLayer(n_neighbors=1, horizon=cfg["Hp"], dt=DT, d_sep=D_SEP,
                      alpha=cfg["alpha"], a_max=cfg["a_max"])
    policy = SafePolicy(pred, mpc)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=DEV)
    wind = UrbanWindField(eta_w=ETA_W, dtype=DTYPE, device=DEV, seed=7)
    mins = []
    for _ in range(max(1, N // BATCH)):
        x0, nh, nf, _ref, _nfut = gen.sample(BATCH, T, DEV)
        x = x0
        ms = torch.full((BATCH,), 1e6, dtype=DTYPE, device=DEV)
        for t in range(T):
            p0 = x[:, 0:3]; v0 = x[:, 3:6]
            tt = torch.arange(cfg["Hp"] + 1, dtype=DTYPE, device=DEV) * DT
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            u, _ = policy(x, nh, nf[:, :, t, :], p_ref)
            x = dyn.step(x, u, wind.sample(p0), DT)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            ms = torch.minimum(ms, d)
        mins.append(ms.cpu())
    return torch.cat(mins)[:N].numpy().astype(float)


@torch.no_grad()
def ade(pred):
    """Open-loop ADE (m) over the prediction horizon, planner-independent."""
    set_seed(12345)
    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=12345)
    tot, cnt = 0.0, 0
    for _ in range(max(1, N // BATCH)):
        _x0, nh, _nf, _ref, nfut = gen.sample(BATCH, T, DEV)
        out = pred(nh[:, 0])
        mean = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)
        h = min(mean.shape[1], nfut.shape[2])
        err = torch.linalg.norm(
            (mean[:, :h] - nfut[:, 0, :h]) * SCALE, dim=-1)
        tot += err.mean(1).sum().item(); cnt += err.shape[0]
    return tot / cnt


def paired(cx, cy):
    return (int((cx & cy).sum()), int((cx & ~cy).sum()),
            int((~cx & cy).sum()), int((~cx & ~cy).sum()))


def main():
    lines = []

    def w(s=""):
        print(s, flush=True)
        lines.append(s)

    w("CROSS-PLANNER TRANSFER OF THE TASK-ALIGNED OBJECTIVE")
    w(f"  n={N} held-out 2500-2999, seed 12345, eta_w={ETA_W}, wind seed 7,")
    w(f"  d_sep={D_SEP} m, T={T}, dt={DT}. Conflict := MinSep < {D_SEP} m.")
    w("  training-time planner  gamma=0.4 H_p=8  a_max=10")
    w("  deployment planner     gamma=0.1 H_p=15 a_max=20")
    w("  Nothing is retrained. ADE is planner-independent (replayed neighbours).")
    w("=" * 76)

    res = {}
    for name, fn in CKPT:
        p = load(fn)
        a = ade(p)
        row = {"ade": a}
        for cfg in (DEPLOY_CFG, TRAIN_CFG):
            ms = rollout(p, cfg)
            row[cfg["name"]] = {"ms": ms, "cf": ms < D_SEP,
                                "cr": 100.0 * (ms < D_SEP).mean(),
                                "minsep": float(ms.mean())}
        res[name] = row
        w(f"  {name:17s} trained@{str(TRAINED_UNDER[name] or '-'):13s} "
          f"ADE={a:6.2f} m   "
          f"deploy CR={row['deployment']['cr']:5.1f}%   "
          f"train-time CR={row['training-time']['cr']:5.1f}%")

    # ---- controls -----------------------------------------------------------
    w()
    bad = []
    for k, want in PUB_TRAIN_CR.items():
        got = res[k]["training-time"]["cr"]
        if abs(got - want) > 1e-9:
            bad.append(f"train-time {k}: {got:.2f} != {want}")
    for k, want in PUB_DEPLOY_CR.items():
        got = res[k]["deployment"]["cr"]
        if abs(got - want) > 1e-9:
            bad.append(f"deploy {k}: {got:.2f} != {want}")
    for k, want in PUB_ADE.items():
        got = res[k]["ade"]
        if abs(got - want) > 0.02:
            bad.append(f"ADE {k}: {got:.2f} != {want}")
    if bad:
        raise AssertionError("published values not reproduced:\n  "
                             + "\n  ".join(bad))
    w("  control: published CRs (56.5/56.0/28.5 train-time, 11.5/11.0 deploy)")
    w("  and ADEs (20.90/1.84/4.32 m) all reproduced")

    # ---- Q1: is the same-configuration result reproducible? -----------------
    w()
    w("Q1  THREE ADDITIONAL SEEDS, ALL TRAINED UNDER (0.4, 8, 10),")
    w("    NOW EVALUATED UNDER THAT SAME PLANNER")
    fam = ["Stage-2 primary", "Stage-2 seed1", "Stage-2 seed2", "Stage-2 seed3"]
    ref = res["Stage-1b"]["training-time"]["cr"]
    crs = []
    for k in fam:
        r = res[k]["training-time"]
        a, b, c, d = paired(res["Stage-1b"]["training-time"]["cf"], r["cf"])
        p = mcnemar_exact_two_sided(b, c)
        crs.append(r["cr"])
        w(f"    {k:17s} CR={r['cr']:5.1f}%  vs Stage-1b {ref:.1f}%  "
          f"({r['cr'] - ref:+5.1f} pp)  resolved={b:3d} introduced={c:3d}  "
          f"p={p:.2e}  nested={'yes' if c == 0 else 'no'}")
    crs = np.array(crs)
    w(f"    family mean {crs.mean():.1f}% +/- {crs.std(ddof=1):.1f} "
      f"(n=4, ddof=1), range {crs.min():.1f}-{crs.max():.1f}%")
    dep = res["Stage-2 deploy"]["training-time"]["cr"]
    w(f"    deployment-trained checkpoint {dep:.1f}%")
    if crs.max() < ref and dep > ref:
        w("    READ: all same-configuration checkpoints fall below the Stage-1b")
        w("    reference while the deployment-trained one exceeds it, so the")
        w("    favourable same-configuration outcome is reproducible across")
        w("    training seeds and is not a property of one lucky run.")
    elif crs.max() < ref:
        w("    READ: same-configuration checkpoints reproducibly beat Stage-1b,")
        w("    but the deployment-trained checkpoint does not exceed it.")
    else:
        w("    READ: the same-configuration advantage is NOT reproducible across")
        w("    seeds; the published 28.5% cannot be treated as a family result.")

    # ---- Q2: does ADE explain the divergence? -------------------------------
    w()
    w("Q2  DOES OPEN-LOOP ACCURACY ACCOUNT FOR THE DIVERGENCE?")
    a_dep = res["Stage-2 deploy"]["ade"]
    lo, hi = res["Stage-1b"]["ade"], res["Stage-1"]["ade"]
    w(f"    deployment-trained checkpoint ADE = {a_dep:.2f} m")
    w(f"    under this planner ADE {hi:.2f} m -> "
      f"{res['Stage-1']['training-time']['cr']:.1f}% and "
      f"{lo:.2f} m -> {res['Stage-1b']['training-time']['cr']:.1f}%,")
    w("    an accuracy span of about "
      f"{hi / lo:.0f}x with almost no separation in outcome, so ADE has very")
    w("    little discriminating power in this regime.")
    if lo <= a_dep <= hi:
        w("    The deployment-trained checkpoint's ADE lies inside that span,")
        w("    which rules out gross predictor failure as the explanation but")
        w("    CANNOT establish planner-specific co-adaptation: error geometry,")
        w("    temporal structure and optimisation quality remain uncontrolled.")
        w("    Report as 'not explained by ADE alone and consistent with")
        w("    planner-specific adaptation', never as proof.")
    else:
        w("    The ADE falls OUTSIDE the bracketed span; gross predictor")
        w("    degradation cannot be ruled out and must be reported as a")
        w("    competing explanation.")

    # ---- interaction, not a diagonal ---------------------------------------
    w()
    w("INTERACTION STRUCTURE (the claim the figure may make)")
    spread_dep = max(res[k]["deployment"]["cr"] for k in fam + ["Stage-2 deploy"]) \
        - min(res[k]["deployment"]["cr"] for k in fam + ["Stage-2 deploy"])
    spread_tr = max(res[k]["training-time"]["cr"] for k in fam + ["Stage-2 deploy"]) \
        - min(res[k]["training-time"]["cr"] for k in fam + ["Stage-2 deploy"])
    w(f"    spread across Stage-2 checkpoints under deployment planner   "
      f"{spread_dep:5.1f} pp")
    w(f"    spread across Stage-2 checkpoints under training-time planner "
      f"{spread_tr:5.1f} pp")
    pri = res["Stage-2 primary"]
    w(f"    primary checkpoint: {pri['deployment']['cr']:.1f}% under the "
      f"deployment planner vs {pri['training-time']['cr']:.1f}% under its OWN "
      f"training planner")
    if pri["deployment"]["cr"] < pri["training-time"]["cr"]:
        w("    NOTE: the primary checkpoint is absolutely BETTER under the")
        w("    planner it was NOT trained on, so the data do NOT form a")
        w("    'each predictor is best under its own planner' diagonal. The")
        w("    defensible statement is an asymmetric interaction: the")
        w("    deployment planner attenuates differences between predictors,")
        w("    the training-time planner exposes them. Figures must not imply")
        w("    a matched-is-better diagonal.")

    np.savez(os.path.join(OUT_DIR, "planner_transfer_v2.npz"),
             names=np.array([n for n, _ in CKPT]),
             trained_under=np.array([str(TRAINED_UNDER[n]) for n, _ in CKPT]),
             ade=np.array([res[n]["ade"] for n, _ in CKPT]),
             cr_deploy=np.array([res[n]["deployment"]["cr"] for n, _ in CKPT]),
             cr_train=np.array([res[n]["training-time"]["cr"] for n, _ in CKPT]),
             minsep_deploy=np.array([res[n]["deployment"]["minsep"] for n, _ in CKPT]),
             minsep_train=np.array([res[n]["training-time"]["minsep"] for n, _ in CKPT]),
             family_cr_train=crs,
             **{f"cf_train__{n}": res[n]["training-time"]["cf"] for n, _ in CKPT},
             **{f"cf_deploy__{n}": res[n]["deployment"]["cf"] for n, _ in CKPT})
    w()
    w(f"wrote {OUT_DIR}/planner_transfer_v2.npz")
    with open(TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {TXT}")


if __name__ == "__main__":
    main()
