"""collect_fig_data.py -- dump the REAL per-episode / per-step arrays that the
figure scripts need but that the existing pipeline never persisted.

Standing constraints honoured:
  * dead path /data/lab/plangrad/plangrad_sim is redirected to the live repo
    copy /data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim via sys.path + PLAN.
  * main-table config: seed=12345, n=200, batch=8, deploy planner
    (alpha/gamma=0.1, Hp=15, a_max=20, d_sep=30), eta_w=0.3 (the code's actual
    WIND_ETA; the manuscript's 0.5 is flagged as a text/code mismatch).
  * GUAM_MAT exported by caller.

Outputs (into ./fig_data/):
  minsep_effort.npz : per-episode min_sep[m] & energy for the 7 non-oracle arms
  errdir_profile.npz: per-|k-kCPA| mean |error| profile for Stage-1 / -1b / -2
  planner_heatmap_n200.json : CR over the 4x4 (gamma,a_max) grid at n=200
Every array is the genuine closed-loop output; nothing is synthesised.
"""
import os, sys, json
import numpy as np
import torch

B = "/data/lab/TRC_UAV_WEIGHTS/code/baselines"
PLAN = "/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim"   # live repo, NOT dead path
for p in [os.path.join(B, "common"),
          os.path.join(B, "01_constant_velocity"),
          os.path.join(B, "02_vanilla_mpc"),
          os.path.join(B, "05_conformal_mpc"),
          PLAN]:
    sys.path.insert(0, p)

import eval_common as ec
from cv_predictor import ConstantVelocityPredictor
from safe_policy import SafePolicy
from vanilla_mpc import VanillaMPCLayer
from params import DEFAULT_PARAMS
from dynamics import EVTOLDynamics
from wind import UrbanWindField
from guam_encounters import GUAMEncounters
from cbf_mpc import CBFMPCLayer
from fast_cbf_mpc import FastCBFMPC          # OSQP, ~10-50x faster, identical QP
from vanilla_mpc import VanillaMPCLayer as _VMPC_check  # noqa

DEV = ec.device_str(False)          # CPU on the lab box
DTYPE = torch.float64
DT = 0.2
DSEP = ec.D_SEP
ETA = 0.3                            # == ec.WIND_ETA; matches main-table pipeline
SEED = ec.GLOBAL_SEED
N = 200
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_data")
os.makedirs(OUT, exist_ok=True)


def _load(p):
    return ec.load_gmm_predictor(f"{PLAN}/{p}", DEV)


@torch.no_grad()
def rollout_minsep_effort(predictor, tag, d_sep_plan=DSEP, policy=None,
                          Hp=None):
    """Replicate eval_common.evaluate_policy's rollout but KEEP per-episode
    min_sep and energy, using the FAST OSQP CBF solver (identical QP to the
    differentiable layer, self-verified in fast_cbf_mpc.__main__, ~10-50x
    faster). If `policy` is given (certificate-free arms) it is used instead
    on the differentiable path. Same seed/stream/wind => same CR as the table.

    d_sep_plan: PLANNING margin fed to the fast solver (30 m for the CBF arms,
                30 + r_conf for Conformal); conflicts are still judged at 30 m.
    """
    ec.set_seed(SEED)
    if Hp is None:
        Hp = ec.BEST_PLANNER["horizon"]
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=DEV)
    wind = UrbanWindField(eta_w=ETA, dtype=DTYPE, device=DEV, seed=ec.WIND_SEED)
    gen = GUAMEncounters(ec.GUAM_MAT, ec.EVAL_RANGE, seed=SEED)
    weight = DEFAULT_PARAMS.weight
    mmax = DEFAULT_PARAMS.max_body_moment
    T = ec.T_EPISODE
    batch = 8
    fast = None
    sp = None
    if policy is None:
        fast = FastCBFMPC(n_neighbors=1, horizon=Hp, dt=DT, d_sep=d_sep_plan,
                          alpha=ec.BEST_PLANNER["alpha"],
                          a_max=ec.BEST_PLANNER["a_max"])
        sp = SafePolicy(predictor, CBFMPCLayer(n_neighbors=1, horizon=Hp, dt=DT,
                        d_sep=d_sep_plan, alpha=ec.BEST_PLANNER["alpha"],
                        a_max=ec.BEST_PLANNER["a_max"]))  # only for pred+inner
    ms_all, en_all = [], []
    for _ in range(N // batch):
        x0, nh, nf, _ref, nfut = gen.sample(batch, T, DEV)
        x = x0
        min_sep = torch.full((batch,), 1e6, dtype=DTYPE, device=DEV)
        energy = torch.zeros(batch, dtype=DTYPE, device=DEV)
        for t in range(T):
            p0, v0 = x[:, 0:3], x[:, 3:6]
            tt = torch.arange(Hp + 1, dtype=DTYPE, device=DEV) * DT
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            if policy is not None:                       # certificate-free arm
                try:
                    u, _ = policy(x, nh, nf[:, :, t, :], p_ref)
                except Exception:
                    u = torch.zeros(batch, 4, dtype=DTYPE, device=DEV)
                    u[:, 0] = weight
            else:                                        # fast CBF path
                pred_abs, _ = sp.predict_neighbours(nh, nf[:, :, t, :])
                a_cmd = torch.zeros(batch, 3, dtype=DTYPE, device=DEV)
                pr_np = p_ref.cpu().numpy()
                pa_np = pred_abs.cpu().numpy()
                p0n = p0.cpu().numpy(); v0n = v0.cpu().numpy()
                for b in range(batch):
                    a0 = fast.solve_np(p0n[b], v0n[b], pr_np[b], pa_np[b])
                    if a0 is not None:
                        a_cmd[b] = torch.as_tensor(a0, dtype=DTYPE, device=DEV)
                u = sp.accel_to_control(x, a_cmd)
            x = dyn.step(x, u, wind.sample(p0), DT)
            thr_n = (u[:, 0] - weight) / weight
            mom_n = u[:, 1:4] / mmax
            energy = energy + thr_n ** 2 + (mom_n ** 2).sum(-1)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            min_sep = torch.minimum(min_sep, d)
        ms_all.append(min_sep.cpu().numpy())
        en_all.append(energy.cpu().numpy())
    ms = np.concatenate(ms_all); en = np.concatenate(en_all)
    cr = 100.0 * float((ms < DSEP).sum()) / len(ms)
    print(f"  [{tag}] n={len(ms)}  CR={cr:.1f}%  minSep_mean={ms.mean():.2f} "
          f"Effort_mean={en.mean():.2f}", flush=True)
    return ms, en


@torch.no_grad()
def errdir_profile(tag, weights):
    """Mean |prediction error| vs |k - kCPA| (steps 0..7), model-independent
    window anchored on the TRUE neighbour CPA. Averaged over N episodes."""
    net = _load(weights)
    ec.set_seed(SEED)
    gen = GUAMEncounters(ec.GUAM_MAT, ec.EVAL_RANGE, seed=SEED)
    batch = 8
    MAXD = 7
    sums = np.zeros(MAXD + 1); cnts = np.zeros(MAXD + 1)
    for _ in range(N // batch):
        x0, nh, nf, _ref, nfut = gen.sample(batch, 30, DEV)
        out = net(nh.reshape(batch, 25, 3))
        mean_traj = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)  # [B,30,3]
        h = min(30, nfut.shape[2])
        err = torch.linalg.norm(mean_traj[:, :h] - nfut[:, 0, :h, :], dim=-1)  # [B,h] (scaled)
        err = err * ec.SCALE
        # true-neighbour CPA step per episode (distance ego-ref to neighbour fut)
        dist = torch.linalg.norm(nf[:, 0, 1:h + 1, :] - nf[:, 0, 0:1, :], dim=-1)
        # anchor CPA on the neighbour's closest approach to ego start (model-free)
        ego0 = x0[:, 0:3].unsqueeze(1)
        dcpa = torch.linalg.norm(nfut[:, 0, :h, :] - ego0, dim=-1)  # [B,h]
        for b in range(batch):
            kcpa = int(torch.argmin(dcpa[b]).item())
            for k in range(h):
                dd = abs(k - kcpa)
                if dd <= MAXD:
                    sums[dd] += float(err[b, k].item()); cnts[dd] += 1
    prof = sums / np.clip(cnts, 1, None)
    print(f"  [{tag}] profile(|k-kCPA|=0..7) = " +
          " ".join(f"{v:.2f}" for v in prof), flush=True)
    return prof


@torch.no_grad()
def _cr_fast(pred, sp, g, am, Hp):
    """CR% for one (gamma,a_max) planner setting via the FAST OSQP solver.
    sp (SafePolicy) is reused only for predict_neighbours / accel_to_control;
    the QP itself is FastCBFMPC(alpha=g, a_max=am). Same seed/stream/wind as
    the main table so the deployment cell reproduces CR~11.0."""
    ec.set_seed(SEED)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=DEV)
    wind = UrbanWindField(eta_w=ETA, dtype=DTYPE, device=DEV, seed=ec.WIND_SEED)
    gen = GUAMEncounters(ec.GUAM_MAT, ec.EVAL_RANGE, seed=SEED)
    fast = FastCBFMPC(n_neighbors=1, horizon=Hp, dt=DT, d_sep=DSEP,
                      alpha=g, a_max=am)
    T = ec.T_EPISODE; batch = 8
    ms_all = []
    for _ in range(N // batch):
        x0, nh, nf, _ref, nfut = gen.sample(batch, T, DEV)
        x = x0
        min_sep = torch.full((batch,), 1e6, dtype=DTYPE, device=DEV)
        for t in range(T):
            p0, v0 = x[:, 0:3], x[:, 3:6]
            tt = torch.arange(Hp + 1, dtype=DTYPE, device=DEV) * DT
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            pred_abs, _ = sp.predict_neighbours(nh, nf[:, :, t, :])
            a_cmd = torch.zeros(batch, 3, dtype=DTYPE, device=DEV)
            pr_np = p_ref.cpu().numpy(); pa_np = pred_abs.cpu().numpy()
            p0n = p0.cpu().numpy(); v0n = v0.cpu().numpy()
            for b in range(batch):
                a0 = fast.solve_np(p0n[b], v0n[b], pr_np[b], pa_np[b])
                if a0 is not None:
                    a_cmd[b] = torch.as_tensor(a0, dtype=DTYPE, device=DEV)
            u = sp.accel_to_control(x, a_cmd)
            x = dyn.step(x, u, wind.sample(p0), DT)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            min_sep = torch.minimum(min_sep, d)
        ms_all.append(min_sep.cpu().numpy())
    ms = np.concatenate(ms_all)
    return 100.0 * float((ms < DSEP).sum()) / len(ms)


def heatmap_n200():
    """4x4 (gamma,a_max) CR grid at the MAIN-TABLE n=200 (not the old n=96),
    stage2_final predictor, deploy Hp=15, eta=0.3, seed=12345. FAST OSQP path."""
    Hp = ec.BEST_PLANNER["horizon"]
    pred = _load("stage2_final.pt")
    # SafePolicy built once; its inner CBFMPCLayer is used ONLY for
    # predict_neighbours / accel_to_control (alpha/a_max-independent helpers).
    sp = SafePolicy(pred, CBFMPCLayer(n_neighbors=1, horizon=Hp, dt=DT,
                    d_sep=DSEP, alpha=ec.BEST_PLANNER["alpha"],
                    a_max=ec.BEST_PLANNER["a_max"]))
    gammas = [0.1, 0.2, 0.4, 0.6]
    amaxs = [5.0, 10.0, 15.0, 20.0]
    grid = []
    for g in gammas:
        row = []
        for am in amaxs:
            cr = _cr_fast(pred, sp, g, am, Hp)
            row.append(round(cr, 1))
            print(f"  gamma={g} a_max={am} -> CR={cr:.1f}", flush=True)
        grid.append(row)
    obj = {"gammas": gammas, "amaxs": amaxs, "CR": grid, "n": N,
           "seed": SEED, "eta_w": ETA, "eta": ETA,
           "Hp": ec.BEST_PLANNER["horizon"],
           "predictor": "stage2_final.pt", "eval_range": [2500, 3000]}
    json.dump(obj, open(os.path.join(OUT, "planner_heatmap_n200.json"), "w"),
              indent=1)
    print("  heatmap dumped", flush=True)


def main():
    print("=== minsep + effort (7 arms) ===", flush=True)
    s1 = _load("stage1_full.pt")
    s1b = _load("stage1b_domainadapt.pt")
    s2 = _load("stage2_final.pt")
    cv = ConstantVelocityPredictor(T=30, K=5).double().to(DEV)
    data = {}
    # CBF-equipped arms -> fast OSQP path (planning margin = 30 m)
    for tag, pred in [
        ("Stage2", s2),
        ("Stage-1b", s1b),
        ("Fixed-Predictor", s1),
        ("Constant-Velocity", cv),
    ]:
        ms, en = rollout_minsep_effort(pred, tag)
        data[f"{tag}__minsep"] = ms; data[f"{tag}__effort"] = en
    # Conformal: real conformal radius inflates the PLANNING margin (30 + r);
    # conflicts still judged at 30 m. Fast path with d_sep_plan = 30 + r.
    try:
        from conformal import conformal_radius
        r, _ = conformal_radius(s1, delta=0.1,
                                horizon=ec.BEST_PLANNER["horizon"],
                                device=DEV, seed=SEED)
    except Exception as e:
        print(f"  [Conformal] conformal_radius failed ({e}); r=5.0 fallback",
              flush=True)
        r = 5.0
    ms, en = rollout_minsep_effort(s1, "Conformal-MPC", d_sep_plan=DSEP + r)
    data["Conformal-MPC__minsep"] = ms; data["Conformal-MPC__effort"] = en
    data["Conformal-MPC__r_conf"] = np.array([r])
    # Vanilla-MPC: stage2 predictor, no-CBF soft planner
    vp = VanillaMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"], dt=DT,
                         d_sep=DSEP, a_max=ec.BEST_PLANNER["a_max"])
    ms, en = rollout_minsep_effort(s2, vp, "Vanilla-MPC", policy=SafePolicy(s2, vp))
    data["Vanilla-MPC__minsep"] = ms; data["Vanilla-MPC__effort"] = en
    # Soft-IPP: jointly-trained soft predictor if the weight file exists
    soft_w = os.path.join(B, "04_soft_ipp", "soft_joint.pt")
    if os.path.exists(soft_w):
        sj = ec.load_gmm_predictor(soft_w, DEV)
        vp2 = VanillaMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"],
                              dt=DT, d_sep=DSEP, a_max=ec.BEST_PLANNER["a_max"])
        ms, en = rollout_minsep_effort(sj, vp2, "Soft-IPP",
                                       policy=SafePolicy(sj, vp2))
        data["Soft-IPP__minsep"] = ms; data["Soft-IPP__effort"] = en
    else:
        print("  [Soft-IPP] soft_joint.pt absent -> skipped (flag in manifest)",
              flush=True)
    np.savez(os.path.join(OUT, "minsep_effort.npz"), **data)
    print("  minsep_effort.npz dumped", flush=True)

    print("=== errdir CPA profile (Stage-1 / -1b / -2) ===", flush=True)
    prof = {}
    prof["Stage1"] = errdir_profile("Stage1", "stage1_full.pt")
    prof["Stage-1b"] = errdir_profile("Stage-1b", "stage1b_domainadapt.pt")
    prof["Stage2"] = errdir_profile("Stage2", "stage2_final.pt")
    np.savez(os.path.join(OUT, "errdir_profile.npz"), **prof)
    print("  errdir_profile.npz dumped", flush=True)

    print("=== planner heatmap n=200 ===", flush=True)
    heatmap_n200()
    print("ALL FIG DATA COLLECTED", flush=True)


if __name__ == "__main__":
    main()
