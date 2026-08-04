"""Unified, reproducible closed-loop evaluation shared by ALL baselines.

This is the SINGLE source of truth for how every method in baselines/ is
scored, so the comparison is apples-to-apples and bit-for-bit reproducible.

It is a faithful re-implementation of the evaluation in
  plangrad_sim/final_best.py
(same held-out GUAM encounters range(2500,3000), same wind seed 7, same
"best" planner alpha=0.1 / Hp=15 / a_max=20, same d_sep=30, dt=0.2, T=20,
n=200, global seed 12345), extended with two extra operational metrics
the manuscript reports (lead time and control energy).

A "method" is defined by exactly TWO swappable pieces:
  * a PREDICTOR  : callable(neigh_hist[B,25,3]) -> dict{alpha,mu,log_sigma}
                   (same contract as plangrad_sim.predictor.GMMTrajectoryPredictor)
  * a PLANNER    : a CBFMPCLayer-like object with __call__ used by SafePolicy,
                   OR a full custom policy (see `policy_factory`).

Most baselines only change ONE of these, so safety stays attributable.

Metrics (per held-out encounter, then aggregated):
  CR%        : conflict rate = fraction of episodes whose min separation < d_sep
  minSep (m) : mean over episodes of the per-episode minimum ego-neighbour distance
  ADE (m)    : mean displacement error of the predictor's mean trajectory
               vs the true neighbour future (manuscript predictor metric)
  LeadT (s)  : conflict-warning lead time -- mean seconds before the closest
               approach at which the predicted closest-approach distance first
               drops below d_sep (0 if never predicted). Higher = earlier warning.
  Energy     : mean normalised control effort  sum_t (thr_n^2 + ||mom_n||^2)
               (thr_n,mom_n normalised by weight / max_body_moment), as in
               train_stage2.py. Lower = smoother / cheaper control.

All randomness is fixed: set_seed(12345); GUAMEncounters(range(2500,3000),
seed=12345); UrbanWindField(seed=7). On the same GPU + library versions the
numbers are reproducible run-to-run.
"""
from __future__ import annotations
import sys
import os
import json
import torch
import numpy as np

# make plangrad_sim importable no matter where this is run from
PLANGRAD_DIR = "/data/lab/plangrad/plangrad_sim"
if PLANGRAD_DIR not in sys.path:
    sys.path.insert(0, PLANGRAD_DIR)

from params import DEFAULT_PARAMS              # noqa: E402
from dynamics import EVTOLDynamics             # noqa: E402
from wind import UrbanWindField                # noqa: E402
from predictor import GMMTrajectoryPredictor   # noqa: E402
from cbf_mpc import CBFMPCLayer                # noqa: E402
from safe_policy import SafePolicy            # noqa: E402
from guam_encounters import GUAMEncounters    # noqa: E402
from config import GUAM_MAT                    # noqa: E402
from seeding import set_seed                   # noqa: E402

# ------------------------------------------------------------------ constants
DTYPE = torch.float64
D_SEP = 30.0          # separation threshold (m), identical to final_best.py
SCALE = 100.0         # predictor output scale, identical to final_best.py
DT = 0.2
T_EPISODE = 20        # closed-loop horizon (steps)
EVAL_RANGE = range(2500, 3000)   # held-out trajectories (no overlap w/ training)
WIND_SEED = 7
WIND_ETA = 0.3
GLOBAL_SEED = 12345

# "best" planner config (manuscript headline / scan_planner.py winner)
BEST_PLANNER = dict(alpha=0.1, horizon=15, a_max=20.0)


def device_str(use_cuda=True):
    return "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"


def load_gmm_predictor(weights_path, device, T=30, K=5):
    """Load a GMMTrajectoryPredictor checkpoint (float64), eval mode."""
    net = GMMTrajectoryPredictor(T=T, K=K).double().to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()
    return net


def make_best_planner(n_neighbors=1):
    """The tuned CBF-MPC planner used for the headline comparison."""
    return CBFMPCLayer(n_neighbors=n_neighbors, horizon=BEST_PLANNER["horizon"],
                       dt=DT, d_sep=D_SEP, alpha=BEST_PLANNER["alpha"],
                       a_max=BEST_PLANNER["a_max"])


@torch.no_grad()
def evaluate_policy(predictor, planner, n=200, device="cuda",
                    d_sep=D_SEP, T=T_EPISODE, seed=GLOBAL_SEED,
                    policy=None, ade_predictor=None, eta_w=WIND_ETA):
    """Run the closed loop on n held-out encounters and return metrics.

    Args
      predictor : predictor module (contract above). Used inside SafePolicy
                  UNLESS a custom `policy` is supplied.
      planner   : CBF-MPC-like layer for SafePolicy (ignored if `policy` given).
      policy    : optional custom callable(x, nh, neigh_now, p_ref)->(u,info)
                  replacing SafePolicy entirely (e.g. Vanilla-MPC baseline).
      ade_predictor : optional separate module used ONLY to report ADE
                  (when the control predictor has no GMM head, e.g. CV).
    """
    set_seed(seed)
    if policy is None:
        policy = SafePolicy(predictor, planner)
        Hp = planner.Hp if hasattr(planner, "Hp") else BEST_PLANNER["horizon"]
    else:
        Hp = getattr(policy, "Hp", BEST_PLANNER["horizon"])

    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=device)
    wind = UrbanWindField(eta_w=eta_w, dtype=DTYPE, device=device,
                          seed=WIND_SEED)
    gen = GUAMEncounters(GUAM_MAT, EVAL_RANGE, seed=seed)

    weight = DEFAULT_PARAMS.weight
    mmax = DEFAULT_PARAMS.max_body_moment

    ade_net = ade_predictor if ade_predictor is not None else predictor

    n_coll = tot = 0
    ms_all, en_all = [], []   # per-episode min_sep / effort (figure provenance)
    sep_sum = 0.0
    ade_sum = 0.0
    lead_sum = 0.0
    energy_sum = 0.0
    batch = 8
    # Guard against the sampling artefact that silently drops encounters when
    # n is not a multiple of the batch size (n//batch truncates). All reported
    # numbers use exactly n=200; a non-conforming n would change ADE/CR by the
    # dropped tail and break cross-table consistency.
    assert n % batch == 0, (
        f"n_eval={n} must be a multiple of batch={batch} so that exactly n "
        f"encounters are evaluated (n//batch truncation would drop the tail).")

    for _ in range(max(1, n // batch)):
        x0, nh, nf, _ref, nfut = gen.sample(batch, T, device)

        # ---- ADE of the (reporting) predictor's mean trajectory ----
        try:
            out = ade_net(nh.reshape(batch, 25, 3))
            mean_traj = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)
            h = min(30, nfut.shape[2])
            ade = torch.linalg.norm(
                mean_traj[:, :h] - nfut[:, 0, :h, :], dim=-1).mean().item()
            ade_sum += ade * SCALE * batch
        except Exception:
            ade_sum += float("nan")

        # ---- closed-loop rollout ----
        x = x0
        min_sep = torch.full((batch,), 1e6, dtype=DTYPE, device=device)
        energy = torch.zeros(batch, dtype=DTYPE, device=device)
        dist_hist = torch.zeros(batch, T, dtype=DTYPE, device=device)

        for t in range(T):
            p0, v0 = x[:, 0:3], x[:, 3:6]
            tt = torch.arange(Hp + 1, dtype=DTYPE, device=device) * DT
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            try:
                u, _ = policy(x, nh, nf[:, :, t, :], p_ref)
            except Exception:
                u = torch.zeros(batch, 4, dtype=DTYPE, device=device)
                u[:, 0] = weight
            x = dyn.step(x, u, wind.sample(p0), DT)

            thr_n = (u[:, 0] - weight) / weight
            mom_n = u[:, 1:4] / mmax
            energy = energy + thr_n ** 2 + (mom_n ** 2).sum(-1)

            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            dist_hist[:, t] = d
            min_sep = torch.minimum(min_sep, d)

        # ---- lead time: seconds before closest approach that distance first
        #      crossed below d_sep (per episode) ----
        for b in range(batch):
            dh = dist_hist[b]
            t_close = int(torch.argmin(dh).item())
            below = (dh < d_sep).nonzero().flatten()
            if len(below) > 0 and dh[t_close] < d_sep:
                t_first = int(below[0].item())
                lead_sum += max(0.0, (t_close - t_first) * DT)

        n_coll += int((min_sep < d_sep).sum().item())
        sep_sum += float(min_sep.sum().item())
        energy_sum += float(energy.sum().item())
        ms_all.append(min_sep.detach().cpu().numpy().copy())
        en_all.append(energy.detach().cpu().numpy().copy())
        tot += batch

    return {
        "CR_%": 100.0 * n_coll / tot,
        "minSep_m": sep_sum / tot,
        "ADE_m": ade_sum / tot,
        "LeadT_s": lead_sum / tot,
        "Energy": energy_sum / tot,
        "n": tot,
        # Per-episode arrays so that figures and the main table are provenance-
        # identical BY CONSTRUCTION: CR_% == 100*mean(minsep_per_ep < d_sep).
        "minsep_per_ep": np.concatenate(ms_all),
        "effort_per_ep": np.concatenate(en_all),
    }


def write_result(out_dir, method_name, model_name, metrics, extra=None):
    """Persist a baseline's result as both result.txt and result.json."""
    os.makedirs(out_dir, exist_ok=True)
    # Split per-episode arrays out of the JSON payload: they go to a sidecar
    # .npz so result.json stays scalar-only (and JSON-serialisable), while the
    # figures read the very same arrays the scalars were reduced from.
    arrays = {k: v for k, v in metrics.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    if arrays:
        np.savez(os.path.join(out_dir, "per_episode.npz"), **arrays)
        ms = arrays.get("minsep_per_ep")
        if ms is not None:
            # Provenance self-check: the table CR must equal the figure CR.
            cr_arr = 100.0 * float((ms < D_SEP).sum()) / len(ms)
            assert abs(cr_arr - scalars["CR_%"]) < 1e-9, (
                f"CR mismatch: scalar {scalars['CR_%']} vs array {cr_arr}")
    payload = {"method": method_name, "eval_model": model_name,
               "n": scalars["n"], "seed": GLOBAL_SEED,
               "planner": BEST_PLANNER, "d_sep": D_SEP,
               "eval_range": [EVAL_RANGE.start, EVAL_RANGE.stop],
               "metrics": scalars}
    if extra:
        payload.update(extra)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(out_dir, "result.txt"), "w") as f:
        f.write(f"method      : {method_name}\n")
        f.write(f"eval_model  : {model_name}\n")
        f.write(f"n / seed    : {scalars['n']} / {GLOBAL_SEED}\n")
        f.write(f"planner     : {BEST_PLANNER}  d_sep={D_SEP}\n")
        f.write(f"eval_range  : {EVAL_RANGE.start}-{EVAL_RANGE.stop} (held-out)\n")
        f.write("-" * 48 + "\n")
        f.write(f"CR (%)      : {scalars['CR_%']:.2f}\n")
        f.write(f"minSep (m)  : {scalars['minSep_m']:.2f}\n")
        f.write(f"ADE (m)     : {scalars['ADE_m']:.2f}\n")
        f.write(f"LeadT (s)   : {scalars['LeadT_s']:.3f}\n")
        f.write(f"Energy      : {scalars['Energy']:.3f}\n")
    print(f"[saved] {out_dir}/result.txt + result.json")
