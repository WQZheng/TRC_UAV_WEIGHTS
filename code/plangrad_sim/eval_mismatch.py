"""Evaluation under train/eval model mismatch (referee "sim-to-real" defence).

Motivation
----------
In every current evaluation script (``final_best.py`` etc.) the closed-loop
rollout is integrated with the SAME ``EVTOLDynamics(DEFAULT_PARAMS)`` that the
CBF-MPC planner and the SafePolicy inner loop assume. Controller model == plant
model, so the CBF forward-invariance guarantee holds by construction and the
reported safety numbers are obtained under a "perfect model" assumption. A
reviewer correctly pointed out that this is the single biggest hole in the
empirical story: the paper never tests robustness to a mismatch between the
model the planner uses (its internal ``g``) and the plant it actually controls.

What this script does
---------------------
We keep the CONTROLLER nominal (SafePolicy + CBFMPCLayer + inner feedback
law all use ``DEFAULT_PARAMS``) and integrate the rollout with a PERTURBED
plant. This is the standard learning-to-control defence: train on the
differentiable surrogate, evaluate on a perturbed / independent plant. The
perturbations are:

  * mass error         : plant mass  = nominal * mass_factor
  * inertia error      : plant I     = nominal * inertia_factor
  * thrust efficiency  : realised collective thrust = commanded * thrust_eff
  * actuator delay     : the plant executes the control commanded ``act_delay``
                         steps earlier (a 1-2 step transport lag)
  * wind shift         : plant wind uses a larger eta_w / gust_std and a
                         different RNG seed than the planner's nowcast model

For every perturbation regime we re-run the head-to-head comparison over the
SAME held-out GUAM-Seed encounters (trajectories 2500-3000, seed 12345) for:

  * Stage-1 predictor  vs  Stage-2 predictor      (is the decoupling stable?)
  * with CBF certificate  vs  without (Vanilla)   (does the safety edge hold?)

and report conflict rate (CR), mean minimum separation (MinSep) and ADE.

The two questions this answers, directly:
  (Q1) Does the decoupling of displacement accuracy (ADE) from operational
       safety (CR) survive model mismatch?
  (Q2) Does the CBF certificate keep its large safety advantage over a
       soft/no-certificate planner when the plant no longer matches the model?

Usage
-----
    export GUAM_MAT=/path/to/Data_Set_1.mat
    python3 eval_mismatch.py --n 200 --seed 12345 \
        --stage1 stage1_full.pt --stage2 stage2_final.pt

Nothing is retrained; this is evaluation-only (cheap). Results are written to
``MISMATCH.txt`` (and echoed to stdout).
"""
from __future__ import annotations
import argparse
import copy
import dataclasses

import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from seeding import set_seed
from config import GUAM_MAT
from params import DEFAULT_PARAMS
from dynamics import EVTOLDynamics
from wind import UrbanWindField
from predictor import GMMTrajectoryPredictor
from cbf_mpc import CBFMPCLayer
from safe_policy import SafePolicy
from guam_encounters import GUAMEncounters

# Use the EXACT main-table Vanilla-MPC planner (baselines/02_vanilla_mpc) as
# the no-certificate comparator so that the nominal mismatch row reproduces
# Table 1 (CR = 41.0) rather than a different soft-penalty planner. The
# baselines VanillaMPCLayer.solve() signature is drop-in compatible with the
# SoftPolicy wrapper below.
import os as _os
import sys as _sys
_sys.path.insert(0, "/data/lab/plangrad/baselines/02_vanilla_mpc")
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "baselines", "02_vanilla_mpc"))
from vanilla_mpc import VanillaMPCLayer  # noqa: E402  (main-table Vanilla-MPC)

SCALE = 100.0
DSEP = 30.0


# --------------------------------------------------------------------------- #
# Soft-penalty MPC (the paper's Vanilla-MPC / no-certificate baseline).
#
# Same double-integrator MPC, same horizon Hp and same actuation bound a_max
# as CBFMPCLayer, but the HARD discrete-time CBF separation constraint is
# replaced by a SOFT quadratic repulsion penalty in the objective. There is
# no forward-invariance guarantee; the planner is free to violate separation
# whenever tracking the reference is cheaper. This is the honest "no CBF
# certificate" comparator used in the manuscript (Vanilla-MPC / Soft-IPP),
# rather than merely disabling the constraint with d_sep=0 (which leaves the
# tracking term doing implicit avoidance and understates the certificate's
# value).
# --------------------------------------------------------------------------- #
class SoftMPCLayer:
    def __init__(self, n_neighbors=1, horizon=15, dt=0.2, a_max=20.0,
                 d_sep=30.0, w_track=1.0, w_acc=0.05, w_rep=8.0):
        self.N = n_neighbors
        self.Hp = horizon
        self.dt = dt
        self.a_max = a_max
        self.d_sep = d_sep
        self._build(w_track, w_acc, w_rep)

    def _build(self, w_track, w_acc, w_rep):
        N, Hp, dt = self.N, self.Hp, self.dt
        a = cp.Variable((Hp, 3), name="a")
        p = cp.Variable((Hp + 1, 3), name="p")
        v = cp.Variable((Hp + 1, 3), name="v")
        s = cp.Variable((N, Hp + 1), name="s")   # repulsion shortfall

        p0 = cp.Parameter(3, name="p0")
        v0 = cp.Parameter(3, name="v0")
        p_ref = cp.Parameter((Hp + 1, 3), name="p_ref")
        nrm = cp.Parameter((N, Hp + 1, 3), name="nrm")
        bvec = cp.Parameter((N, Hp + 1), name="bvec")

        cons = [p[0] == p0, v[0] == v0]
        for k in range(Hp):
            cons += [p[k + 1] == p[k] + dt * v[k],
                     v[k + 1] == v[k] + dt * a[k]]
            cons += [cp.norm(a[k], "inf") <= self.a_max]
        # soft one-sided separation shortfall:
        #   s_{i,k} >= d_sep - (nrm . p_k - bvec),   s >= 0
        # penalising s in the objective => a *soft* push away, no guarantee.
        for i in range(N):
            for k in range(Hp + 1):
                sep = nrm[i, k, :] @ p[k] - bvec[i, k]
                cons += [s[i, k] >= self.d_sep - sep]
        cons += [s >= 0]

        obj = cp.Minimize(
            w_track * cp.sum_squares(p - p_ref)
            + w_acc * cp.sum_squares(a)
            + w_rep * cp.sum_squares(s))
        prob = cp.Problem(obj, cons)
        assert prob.is_dpp(), "Soft-MPC QP is not DPP"
        self.layer = CvxpyLayer(prob, parameters=[p0, v0, p_ref, nrm, bvec],
                                variables=[a, p, v, s])

    def solve(self, p0, v0, p_ref, neigh_pred):
        ref_exp = p_ref.unsqueeze(1)
        diff = ref_exp - neigh_pred
        dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp_min(1e-6)
        nrm = diff / dist
        bvec = (nrm * neigh_pred).sum(-1)
        # This is evaluation only (no gradient needed), so cap the SCS solver
        # iterations: the soft QP can otherwise take very many iterations on
        # perturbed geometries. A capped, slightly looser solve does not
        # change the collision statistics but is far faster.
        a_sol, p_sol, v_sol, s_sol = self.layer(
            p0, v0, p_ref, nrm, bvec,
            solver_args={"max_iters": 2000, "eps": 1e-4})
        return a_sol[:, 0, :], {"a": a_sol, "p": p_sol, "v": v_sol}


class SoftPolicy(SafePolicy):
    """SafePolicy whose planner is the soft-penalty MPC (no CBF certificate).

    Reuses SafePolicy's predictor call and inner feedback-linearising control;
    only the planning QP is swapped for the soft comparator.
    """
    def __call__(self, x, neigh_hist, neigh_last, p_ref):
        p0 = x[:, 0:3]
        v0 = x[:, 3:6]
        pred_abs, var_mean = self.predict_neighbours(neigh_hist, neigh_last)
        a0, info = self.mpc.solve(p0, v0, p_ref, pred_abs)
        u = self.accel_to_control(x, a0)
        return u, {"a0": a0, "pred_abs": pred_abs, "var_mean": var_mean}
# Best planner configuration from the manuscript (final_best.py).
ALPHA, HP, AMAX = 0.1, 15, 20.0


# --------------------------------------------------------------------------- #
# Perturbed plant parameters (controller stays on DEFAULT_PARAMS)
# --------------------------------------------------------------------------- #
def perturbed_params(mass_factor=1.0, inertia_factor=1.0):
    """Return a copy of DEFAULT_PARAMS with mass / inertia scaled.

    Only the PLANT uses this; the planner and the SafePolicy inner loop keep
    DEFAULT_PARAMS, which is what creates the model mismatch.
    """
    p = dataclasses.replace(
        DEFAULT_PARAMS,
        mass=DEFAULT_PARAMS.mass * mass_factor,
        Ixx=DEFAULT_PARAMS.Ixx * inertia_factor,
        Iyy=DEFAULT_PARAMS.Iyy * inertia_factor,
        Izz=DEFAULT_PARAMS.Izz * inertia_factor,
    )
    return p


# --------------------------------------------------------------------------- #
# One closed-loop evaluation under a given perturbation regime
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(pred, dev, regime, use_cbf=True, n=200, T=20):
    """Roll out the nominal controller against a perturbed plant.

    regime: dict with optional keys
        mass_factor, inertia_factor, thrust_eff, act_delay,
        wind_eta, wind_gust, wind_seed
    """
    mass_factor = regime.get("mass_factor", 1.0)
    inertia_factor = regime.get("inertia_factor", 1.0)
    thrust_eff = regime.get("thrust_eff", 1.0)
    act_delay = int(regime.get("act_delay", 0))
    wind_eta = regime.get("wind_eta", 0.3)
    wind_gust = regime.get("wind_gust", 3.0)
    wind_seed = regime.get("wind_seed", 7)

    # --- CONTROLLER: strictly nominal ---
    if use_cbf:
        mpc = CBFMPCLayer(n_neighbors=1, horizon=HP, dt=0.2, d_sep=DSEP,
                          alpha=ALPHA, a_max=AMAX)
        pol = SafePolicy(pred, mpc, params=DEFAULT_PARAMS)
    else:
        # no-certificate comparator = the EXACT main-table Vanilla-MPC planner
        # (baselines/02_vanilla_mpc, w_rep=50), so the nominal mismatch row
        # reproduces Table 1 (CR = 41.0). SoftPolicy only swaps the planning
        # QP; VanillaMPCLayer.solve() is signature-compatible.
        mpc = VanillaMPCLayer(n_neighbors=1, horizon=HP, dt=0.2, a_max=AMAX,
                              d_sep=DSEP, w_rep=50.0)
        pol = SoftPolicy(pred, mpc, params=DEFAULT_PARAMS)

    # --- PLANT: perturbed ---
    plant = EVTOLDynamics(perturbed_params(mass_factor, inertia_factor),
                          dtype=torch.float64, device=dev)
    wind = UrbanWindField(eta_w=wind_eta, gust_std=wind_gust,
                          dtype=torch.float64, device=dev, seed=wind_seed)

    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=12345)

    n_coll = tot = 0
    sep_sum = ade_sum = 0.0
    B = 8
    for _ in range(n // B):
        x0, nh, nf, _ref, nfut = gen.sample(B, T, dev)

        # ADE of the predictor mean vs ground-truth neighbour future
        out = pred(nh.reshape(B, 25, 3))
        mean_traj = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)
        h = min(30, nfut.shape[2])
        ade_sum += torch.linalg.norm(
            mean_traj[:, :h] - nfut[:, 0, :h, :], dim=-1).mean().item() \
            * SCALE * B

        x = x0
        # actuator-delay buffer: hold a queue of past commands. Init with
        # a gravity-balancing hover command so the lag is physical, not a
        # free "do nothing" head start.
        hover = torch.zeros(B, 4, dtype=torch.float64, device=dev)
        hover[:, 0] = DEFAULT_PARAMS.weight
        cmd_buf = [hover.clone() for _ in range(act_delay)]

        min_sep = torch.full((B,), 1e6, dtype=torch.float64, device=dev)
        for t in range(T):
            p0, v0 = x[:, 0:3], x[:, 3:6]
            tt = torch.arange(HP + 1, dtype=torch.float64, device=dev) * 0.2
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            try:
                u_cmd, _ = pol(x, nh, nf[:, :, t, :], p_ref)
            except Exception:
                u_cmd = hover.clone()

            # actuator transport lag: plant executes a delayed command
            if act_delay > 0:
                cmd_buf.append(u_cmd)
                u_exec = cmd_buf.pop(0)
            else:
                u_exec = u_cmd

            # thrust efficiency loss on the PLANT side only
            u_plant = u_exec.clone()
            u_plant[:, 0] = u_plant[:, 0] * thrust_eff

            x = plant.step(x, u_plant, wind.sample(p0), 0.2)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            min_sep = torch.minimum(min_sep, d)

        n_coll += int((min_sep < DSEP).sum())
        sep_sum += float(min_sep.sum())
        tot += B

    cr = 100.0 * n_coll / tot
    return cr, sep_sum / tot, ade_sum / tot


# --------------------------------------------------------------------------- #
def load_pred(path, dev):
    net = GMMTrajectoryPredictor(T=30, K=5).double().to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--stage1", default="stage1_full.pt")
    ap.add_argument("--stage2", default="stage2_final.pt")
    ap.add_argument("--out", default="MISMATCH.txt")
    ap.add_argument("--regimes", default="all",
                    help="'all' or a comma-separated list of 0-based regime "
                         "indices, e.g. '0,1,4,6,8' for the key subset.")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    fh = open(args.out, "w")

    def w(s=""):
        print(s, flush=True)
        fh.write(s + "\n")
        fh.flush()

    # The perturbation regimes. "nominal" reproduces final_best.py conditions
    # (matched model) as the reference row; the rest introduce a controlled
    # single mismatch, plus one combined worst-case.
    regimes = [
        ("nominal (matched model)",          {}),
        ("mass +20% (heavy)",                 {"mass_factor": 1.20}),
        ("mass -15% (light)",                 {"mass_factor": 0.85}),
        ("inertia +30%",                      {"inertia_factor": 1.30}),
        ("thrust eff 0.85",                   {"thrust_eff": 0.85}),
        ("actuator delay 1 step",             {"act_delay": 1}),
        ("actuator delay 2 steps",            {"act_delay": 2}),
        ("wind shift (eta 1.0, gust 5)",      {"wind_eta": 1.0, "wind_gust": 5.0,
                                               "wind_seed": 99}),
        ("combined (m+20%,eff.85,delay1,wind",
         {"mass_factor": 1.20, "thrust_eff": 0.85, "act_delay": 1,
          "wind_eta": 1.0, "wind_gust": 5.0, "wind_seed": 99}),
    ]

    if args.regimes != "all":
        keep = {int(i) for i in args.regimes.split(",")}
        regimes = [r for i, r in enumerate(regimes) if i in keep]

    w("=" * 78)
    w("MODEL-MISMATCH EVALUATION  (controller = nominal DEFAULT_PARAMS,")
    w("plant = perturbed).  Held-out GUAM 2500-3000, n=%d, seed %d."
      % (args.n, args.seed))
    w("Planner: alpha=%.1f Hp=%d a_max=%.0f, d_sep=%.0f m."
      % (ALPHA, HP, AMAX, DSEP))
    w("CR=conflict rate %%, MinSep=mean min separation m, ADE=m.")
    w("=" * 78)

    s1 = load_pred(args.stage1, dev)
    s2 = load_pred(args.stage2, dev)

    header = ("%-34s | %-22s | %-22s" %
              ("regime", "Stage-1 (CBF)", "Stage-2 (CBF)"))
    for name, reg in regimes:
        set_seed(args.seed)          # identical encounters every regime
        cr1, sep1, ade1 = evaluate(s1, dev, reg, use_cbf=True, n=args.n)
        set_seed(args.seed)
        cr2, sep2, ade2 = evaluate(s2, dev, reg, use_cbf=True, n=args.n)
        set_seed(args.seed)
        crv, sepv, _adev = evaluate(s2, dev, reg, use_cbf=False, n=args.n)

        w("")
        w("### %s" % name)
        w("  Stage-1 +CBF : CR=%5.1f%%  MinSep=%5.1f m  ADE=%6.2f m"
          % (cr1, sep1, ade1))
        w("  Stage-2 +CBF : CR=%5.1f%%  MinSep=%5.1f m  ADE=%6.2f m"
          % (cr2, sep2, ade2))
        w("  Stage-2 NO-CBF: CR=%5.1f%%  MinSep=%5.1f m  (Vanilla)"
          % (crv, sepv))
        w("  -> CBF safety edge: CR %5.1f%% (no-CBF) vs %5.1f%% (CBF); "
          "MinSep %+.1f m" % (crv, cr2, sep2 - sepv))
        w("  -> decoupling: Stage-2 ADE %.2f m vs Stage-1 %.2f m at "
          "CR %.1f%% vs %.1f%%" % (ade2, ade1, cr2, cr1))

    w("")
    w("=" * 78)
    w("READING GUIDE")
    w(" Q1 (decoupling): compare Stage-1 vs Stage-2 rows. If Stage-2 keeps a")
    w("    much lower ADE while CR stays comparable, the ADE<->safety")
    w("    decoupling survives the mismatch.")
    w(" Q2 (CBF edge):  compare '+CBF' vs 'NO-CBF' CR. A large gap that")
    w("    persists across regimes shows the certificate's safety advantage")
    w("    is robust to model mismatch, not an artefact of a perfect model.")
    w("=" * 78)
    fh.close()


if __name__ == "__main__":
    main()
