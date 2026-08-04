"""Zero-slack feasibility re-solve: is the residual conflict actuation-limited
or merely slack-associated?

The main attribution claims residual conflicts are "actuation-limited"
because the CBF-MPC slack variable eps was active (non-zero) at the conflict.
But a non-zero slack only shows the solver CHOSE to relax the constraint for
cost reasons -- it does not prove the hard problem was INFEASIBLE. To decide
between "actuation-limited" (physically cannot avoid) and merely
"slack-associated", we re-solve, at every step of every conflict episode, the
SAME double-integrator CBF-MPC QP but with the slack REMOVED (eps == 0), as a
pure feasibility problem with the prediction held fixed.

  * eps=0 INFEASIBLE at a conflict step  -> actuation-limited (justified).
  * eps=0 FEASIBLE everywhere            -> slack-associated only (the solver
                                            used slack for cost, not necessity).

Protocol mirrors the deployment evaluation exactly: best planner
(alpha=0.1, Hp=15, a_max=20), d_sep=30, held-out 2500-3000, seed 12345, n=200.
Predictions come from the deployed Stage-2 predictor and are detached
(held fixed) -- we test the planner's feasibility, not the predictor.
"""
import os
import torch
import numpy as np
import cvxpy as cp
from seeding import set_seed
from config import GUAM_MAT
from params import DEFAULT_PARAMS
from dynamics import EVTOLDynamics
from wind import UrbanWindField
from predictor import GMMTrajectoryPredictor
from cbf_mpc import CBFMPCLayer
from safe_policy import SafePolicy
from guam_encounters import GUAMEncounters

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
DSEP = 30.0
N = 200
ALPHA, HP, AMAX = 0.1, 15, 20.0
DT = 0.2
CKPT = "stage2_final.pt"
OUT = "ZERO_SLACK_FEAS.txt"


def load(p):
    n = GMMTrajectoryPredictor(T=30, K=5).double().to(DEV)
    n.load_state_dict(torch.load(p, map_location=DEV))
    n.eval()
    return n


def build_noslack_qp(Hp, dt, a_max, d_sep):
    """Same double-integrator CBF-MPC as cbf_mpc.py but with eps REMOVED
    (hard CBF, eps==0). Returns a cvxpy Problem + parameter handles.
    alpha is a fixed float baked into the recursion (as in cbf_mpc.py)."""
    a = cp.Variable((Hp, 3))
    p = cp.Variable((Hp + 1, 3))
    v = cp.Variable((Hp + 1, 3))
    p0 = cp.Parameter(3)
    v0 = cp.Parameter(3)
    p_ref = cp.Parameter((Hp + 1, 3))
    nrm = cp.Parameter((Hp + 1, 3))
    bvec = cp.Parameter((Hp + 1,))
    h0 = cp.Parameter(1)
    cons = [p[0] == p0, v[0] == v0]
    for k in range(Hp):
        cons += [p[k + 1] == p[k] + dt * v[k],
                 v[k + 1] == v[k] + dt * a[k]]
        cons += [cp.norm(a[k], "inf") <= a_max]
    h_prev = h0[0]
    for k in range(1, Hp + 1):
        h_k = nrm[k, :] @ p[k] - bvec[k] - d_sep
        cons += [h_k >= (1 - ALPHA) * h_prev]   # NO slack term
        h_prev = h_k
    obj = cp.Minimize(cp.sum_squares(p - p_ref) + 0.05 * cp.sum_squares(a))
    return cp.Problem(obj, cons), (p0, v0, p_ref, nrm, bvec, h0), (a, p, v)


def noslack_feasible(prob, params, p0v, v0v, p_refv, neigh_pred_v):
    """Return True if the eps=0 QP is feasible at this step."""
    p0, v0, p_ref, nrm, bvec, h0 = params
    diff = p_refv - neigh_pred_v                    # [Hp+1,3]
    dist = np.linalg.norm(diff, axis=-1, keepdims=True).clip(1e-6)
    nrm_v = diff / dist
    bvec_v = (nrm_v * neigh_pred_v).sum(-1)
    h0_v = float((nrm_v[0] * (p0v - neigh_pred_v[0])).sum() - DSEP)
    p0.value = p0v; v0.value = v0v; p_ref.value = p_refv
    nrm.value = nrm_v; bvec.value = bvec_v; h0.value = np.array([h0_v])
    try:
        prob.solve(solver=cp.ECOS, abstol=1e-6, reltol=1e-6, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            return None
    return prob.status in ("optimal", "optimal_inaccurate")



# ============================================================================
# Solver-independent reformulation: instead of asking ECOS "is the eps=0 QP
# feasible?" (a status flag, tolerance-dependent), we solve the ALWAYS-FEASIBLE
# program that minimises the total CBF relaxation needed. The step is
# hard-infeasible iff that minimum relaxation exceeds a PHYSICAL threshold.
# This turns a binary solver verdict into a continuous, comparable quantity.
# ============================================================================
def build_minslack_qp(Hp, dt, a_max, d_sep):
    a = cp.Variable((Hp, 3))
    p = cp.Variable((Hp + 1, 3))
    v = cp.Variable((Hp + 1, 3))
    eps = cp.Variable(Hp, nonneg=True)          # per-step CBF relaxation
    p0 = cp.Parameter(3); v0 = cp.Parameter(3)
    p_ref = cp.Parameter((Hp + 1, 3))
    nrm = cp.Parameter((Hp + 1, 3)); bvec = cp.Parameter((Hp + 1,))
    h0 = cp.Parameter(1)
    cons = [p[0] == p0, v[0] == v0]
    for k in range(Hp):
        cons += [p[k + 1] == p[k] + dt * v[k],
                 v[k + 1] == v[k] + dt * a[k]]
        cons += [cp.norm(a[k], "inf") <= a_max]
    h_prev = h0[0]
    for k in range(1, Hp + 1):
        h_k = nrm[k, :] @ p[k] - bvec[k] - d_sep
        cons += [h_k + eps[k - 1] >= (1 - ALPHA) * h_prev]
        h_prev = h_k
    # Lexicographic intent: relaxation dominates; tracking only breaks ties.
    obj = cp.Minimize(1e6 * cp.sum(eps)
                      + cp.sum_squares(p - p_ref) + 0.05 * cp.sum_squares(a))
    return cp.Problem(obj, cons), (p0, v0, p_ref, nrm, bvec, h0), eps


def minslack_value(prob, params, epsvar, p0v, v0v, p_refv, neigh_pred_v):
    """Return the minimum total CBF relaxation (metres) needed at this step."""
    p0, v0, p_ref, nrm, bvec, h0 = params
    diff = p_refv - neigh_pred_v
    dist = np.linalg.norm(diff, axis=-1, keepdims=True).clip(1e-6)
    nrm_v = diff / dist
    bvec_v = (nrm_v * neigh_pred_v).sum(-1)
    h0_v = float((nrm_v[0] * (p0v - neigh_pred_v[0])).sum() - DSEP)
    p0.value = p0v; v0.value = v0v; p_ref.value = p_refv
    nrm.value = nrm_v; bvec.value = bvec_v; h0.value = np.array([h0_v])
    for sv in (cp.ECOS, cp.CLARABEL, cp.SCS):
        try:
            prob.solve(solver=sv, verbose=False)
            if epsvar.value is not None:
                return float(np.sum(np.maximum(epsvar.value, 0.0)))
        except Exception:
            continue
    return float("nan")


@torch.no_grad()
def main():
    set_seed(12345)
    pred = load(CKPT)
    mpc = CBFMPCLayer(n_neighbors=1, horizon=HP, dt=DT, d_sep=DSEP,
                      alpha=ALPHA, a_max=AMAX)
    pol = SafePolicy(pred, mpc)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=DEV)
    wind = UrbanWindField(eta_w=0.3, dtype=DTYPE, device=DEV, seed=7)
    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=12345)
    prob, params, _ = build_noslack_qp(HP, DT, AMAX, DSEP)
    mprob, mparams, mepsv = build_minslack_qp(HP, DT, AMAX, DSEP)

    T = 20
    n_conf = 0
    n_conf_with_infeas = 0        # conflicts having >=1 eps=0-infeasible step
    n_conf_all_feas = 0           # conflicts where eps=0 feasible at every step
    per_conf_infeas_steps = []
    per_conf_step_infeas = []   # 22x20 raster: per-conflict 0/1 per rollout step
    per_conf_episode_id = []    # global episode index in 0..N-1 (row alignment)
    per_conf_cpa_step = []      # realized closest-approach step per conflict
    per_conf_step_slack = []    # (n_conf, T) continuous min CBF relaxation [m]
    for _bi in range(N // 8):
        x0, nh, nf, _r, _f = gen.sample(8, T, DEV)
        # closed-loop rollout with the real (slacked) planner
        x = x0
        min_sep = torch.full((8,), 1e6, dtype=DTYPE, device=DEV)
        cpa_step = torch.zeros(8, dtype=torch.long, device=DEV)
        traj_x = [x0.clone()]
        for t in range(T):
            p0 = x[:, 0:3]; v0 = x[:, 3:6]
            tt = torch.arange(HP + 1, dtype=DTYPE, device=DEV) * DT
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            try:
                u, _ = pol(x, nh, nf[:, :, t, :], p_ref)
            except Exception:
                u = torch.zeros(8, 4, dtype=DTYPE, device=DEV)
                u[:, 0] = DEFAULT_PARAMS.weight
            x = dyn.step(x, u, wind.sample(p0), DT)
            traj_x.append(x.clone())
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            _upd = d < min_sep
            cpa_step = torch.where(_upd, torch.full_like(cpa_step, t), cpa_step)
            min_sep = torch.minimum(min_sep, d)

        conflict = (min_sep < DSEP).cpu().numpy()
        cpa_np = cpa_step.cpu().numpy()
        for b in range(8):
            if not conflict[b]:
                continue
            n_conf += 1
            infeas_steps = 0
            step_row = [0] * T
            slack_row = [float('nan')] * T
            for t in range(T):
                xb = traj_x[t][b]
                p0v = xb[0:3].cpu().numpy()
                v0v = xb[3:6].cpu().numpy()
                tt = (np.arange(HP + 1) * DT)
                p_refv = p0v[None, :] + v0v[None, :] * tt[:, None]
                # fixed prediction: neighbour mean over the horizon
                out = pred(nh[b:b+1].reshape(1, 25, 3))
                mean_traj = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)[0]
                horizon = min(HP + 1, mean_traj.shape[0])
                nb_last = nf[b, 0, t, :].cpu().numpy()
                npred = np.tile(nb_last, (HP + 1, 1))
                add = (mean_traj[:horizon].cpu().numpy() * 100.0)
                npred[:horizon] = nb_last[None, :] + add
                feas = noslack_feasible(prob, params, p0v, v0v, p_refv, npred)
                if feas is False:
                    infeas_steps += 1
                    step_row[t] = 1
                slack_row[t] = minslack_value(mprob, mparams, mepsv,
                                              p0v, v0v, p_refv, npred)
            per_conf_infeas_steps.append(infeas_steps)
            per_conf_step_infeas.append(step_row)
            per_conf_step_slack.append(slack_row)
            per_conf_episode_id.append(_bi * 8 + b)
            per_conf_cpa_step.append(int(cpa_np[b]))
            if infeas_steps > 0:
                n_conf_with_infeas += 1
            else:
                n_conf_all_feas += 1

    lines = []
    def w(s):
        print(s, flush=True); lines.append(s)
    w("ZERO-SLACK FEASIBILITY RE-SOLVE  (best planner a=%.1f Hp=%d amax=%.0f, "
      "n=%d, seed 12345)" % (ALPHA, HP, AMAX, N))
    w("checkpoint: %s   d_sep=%.0f m" % (CKPT, DSEP))
    w("-" * 60)
    w("total conflict episodes                    : %d" % n_conf)
    w("  with >=1 eps=0-INFEASIBLE step           : %d  (actuation-limited)"
      % n_conf_with_infeas)
    w("  eps=0 FEASIBLE at every step             : %d  (slack-associated only)"
      % n_conf_all_feas)
    if n_conf:
        w("  fraction actuation-limited               : %.1f%%"
          % (100.0 * n_conf_with_infeas / n_conf))
    if per_conf_infeas_steps:
        arr = np.array(per_conf_infeas_steps)
        _figdd = os.environ.get('FIGDATA_DIR',
            '/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data')
        os.makedirs(_figdd, exist_ok=True)
        np.save(os.path.join(_figdd, 'infeasible_steps.npy'), arr)
        w('  [dumped %d per-conflict infeasible-step counts -> infeasible_steps.npy]' % arr.size)
        raster = np.array(per_conf_step_infeas, dtype=np.uint8)  # (n_conf, T)
        np.save(os.path.join(_figdd, 'infeasibility_raster.npy'), raster)
        # ---- attribution_v2.npz: raster + row alignment keys (FIG05) ----
        np.savez(os.path.join(_figdd, 'attribution_v2.npz'),
                 episode_ids=np.asarray(per_conf_episode_id, dtype=np.int64),
                 infeasible=raster.astype(bool),
                 cpa_step=np.asarray(per_conf_cpa_step, dtype=np.int64),
                 min_slack=np.asarray(per_conf_step_slack, dtype=np.float64))
        w('  [dumped attribution_v2.npz: episode_ids/infeasible/cpa_step, '
          'infeasible.sum()=%d]' % int(raster.sum()))
        w('  [dumped %dx%d per-step infeasibility raster -> infeasibility_raster.npy]' % raster.shape)
        w("  infeasible steps per conflict (mean/max) : %.2f / %d"
          % (arr.mean(), arr.max()))
    w("-" * 60)
    if n_conf and n_conf_with_infeas == n_conf:
        w("=> EVERY conflict has an eps=0-infeasible step: "
          "'actuation-limited' is justified.")
    elif n_conf and n_conf_with_infeas == 0:
        w("=> NO conflict is eps=0-infeasible: downgrade to 'slack-associated'.")
    else:
        w("=> MIXED: report the split; use 'slack-associated' unless the "
          "actuation-limited fraction is overwhelming.")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
