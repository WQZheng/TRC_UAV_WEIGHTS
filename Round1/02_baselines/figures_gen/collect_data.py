"""Collect all raw arrays the RQ figures need that are not already in the
baselines' result.json / ood_results.json. Writes JSON artifacts into
figures_gen/data/. Everything uses the unified pipeline + seed 12345.

Artifacts:
  minsep_<method>.json  : per-episode min-separation array (fig 3 CDF)
  rollout_s1.json/_s2   : one representative closed-loop rollout, ego +
                          neighbour xy + separation-over-time (fig 2)
  rq5_profile.json      : prediction error vs |step - closest-approach|
                          for Stage-1 / Stage-2 (fig 4)
  attribution.json      : conflict attribution (slack-active vs pred-error)
                          for Stage-1 / Stage-2 (fig 7)
  planner_heatmap.json  : CR over an alpha x a_max grid (fig 6)
"""
import os
import sys
import json
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
sys.path.insert(0, os.path.join(B, "common"))
sys.path.insert(0, os.path.join(B, "01_constant_velocity"))
sys.path.insert(0, os.path.join(B, "02_vanilla_mpc"))
sys.path.insert(0, os.path.join(B, "05_conformal_mpc"))
sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")

import eval_common as ec
from params import DEFAULT_PARAMS
from dynamics import EVTOLDynamics
from wind import UrbanWindField
from cbf_mpc import CBFMPCLayer
from safe_policy import SafePolicy
from guam_encounters import GUAMEncounters
from config import GUAM_MAT
from seeding import set_seed
from cv_predictor import ConstantVelocityPredictor
from vanilla_mpc import VanillaMPCLayer

DEV = ec.device_str(True)
DTYPE = torch.float64
PLAN = "/data/lab/plangrad/plangrad_sim"
DSEP = 30.0
SCALE = 100.0
N = 200


def dump(name, obj):
    with open(os.path.join(DATA, name), "w") as f:
        json.dump(obj, f)
    print(f"[data] {name}")


# ---------- per-episode min-sep arrays (fig 3) + a rollout (fig 2) ----------
@torch.no_grad()
def collect_minsep(predictor, planner, tag, policy=None, save_rollout=False):
    set_seed(ec.GLOBAL_SEED)
    if policy is None:
        policy = SafePolicy(predictor, planner)
    Hp = getattr(policy, "Hp", ec.BEST_PLANNER["horizon"])
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=DEV)
    wind = UrbanWindField(eta_w=0.3, dtype=DTYPE, device=DEV, seed=7)
    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=ec.GLOBAL_SEED)
    T = 20
    minseps = []
    rollout = None
    for bi in range(N // 8):
        x0, nh, nf, _r, _f = gen.sample(8, T, DEV)
        x = x0
        ms = torch.full((8,), 1e6, dtype=DTYPE, device=DEV)
        ego_xy, nbr_xy, seps = [], [], []
        for t in range(T):
            p0, v0 = x[:, 0:3], x[:, 3:6]
            tt = torch.arange(Hp + 1, dtype=DTYPE, device=DEV) * 0.2
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            u, _ = policy(x, nh, nf[:, :, t, :], p_ref)
            x = dyn.step(x, u, wind.sample(p0), 0.2)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            ms = torch.minimum(ms, d)
            if save_rollout and bi == 0:
                ego_xy.append(x[0, 0:2].tolist())
                nbr_xy.append(nf[0, 0, t + 1, 0:2].tolist())
                seps.append(float(d[0]))
        minseps += ms.tolist()
        if save_rollout and bi == 0:
            rollout = {"ego_xy": ego_xy, "nbr_xy": nbr_xy, "sep": seps,
                       "dt": 0.2, "d_sep": DSEP}
    dump(f"minsep_{tag}.json", {"minsep": minseps, "d_sep": DSEP})
    if save_rollout:
        dump(f"rollout_{tag}.json", rollout)


# ---------- RQ5 error profile (fig 4) ----------
@torch.no_grad()
def collect_profile():
    def load(p):
        return ec.load_gmm_predictor(f"{PLAN}/{p}", DEV)

    def prof(net):
        set_seed(ec.GLOBAL_SEED)
        gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=ec.GLOBAL_SEED)
        bv = torch.zeros(8); bc = torch.zeros(8)
        for _ in range(N // 16):
            x0, nh, nf, ref, nfut = gen.sample(16, 20, DEV)
            out = net(nh.reshape(16, 25, 3))
            mean_pred = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)
            h = min(30, nfut.shape[2])
            err = torch.linalg.norm(mean_pred[:, :h] - nfut[:, 0, :h, :],
                                    dim=-1) * SCALE
            d = torch.linalg.norm(ref[:, 1:h + 1, :] - nf[:, 0, 1:h + 1, :],
                                  dim=-1)
            ca = d.argmin(1)
            idx = torch.arange(h, device=DEV).unsqueeze(0)
            dca = (idx - ca.unsqueeze(1)).abs().clamp(max=7).cpu()
            for bk in range(8):
                m = (dca == bk)
                bv[bk] += (err.cpu() * m).sum().item()
                bc[bk] += int(m.sum())
        return (bv / bc.clamp(min=1)).tolist()
    dump("rq5_profile.json",
         {"buckets": list(range(8)),
          "stage1": prof(load("stage1_full.pt")),
          "stage2": prof(load("stage2_final.pt"))})


# ---------- conflict attribution (fig 7) ----------
@torch.no_grad()
def collect_attribution():
    def load(p):
        return ec.load_gmm_predictor(f"{PLAN}/{p}", DEV)

    def attr(ckpt):
        set_seed(ec.GLOBAL_SEED)
        pred = load(ckpt)
        mpc = CBFMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"],
                          dt=0.2, d_sep=DSEP, alpha=ec.BEST_PLANNER["alpha"],
                          a_max=ec.BEST_PLANNER["a_max"])
        pol = SafePolicy(pred, mpc)
        dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=DEV)
        wind = UrbanWindField(eta_w=0.3, dtype=DTYPE, device=DEV, seed=7)
        gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=ec.GLOBAL_SEED)
        T = 20; n_coll = slack_only = pred_only = both = tot = 0
        for _ in range(N // 8):
            x0, nh, nf, _r, _f = gen.sample(8, T, DEV)
            x = x0
            ms = torch.full((8,), 1e6, dtype=DTYPE, device=DEV)
            mxs = torch.zeros(8, dtype=DTYPE, device=DEV)
            mxe = torch.zeros(8, dtype=DTYPE, device=DEV)
            for t in range(T):
                p0, v0 = x[:, 0:3], x[:, 3:6]
                tt = torch.arange(mpc.Hp + 1, dtype=DTYPE, device=DEV) * 0.2
                p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
                u, aux = pol(x, nh, nf[:, :, t, :], p_ref)
                pe = torch.linalg.norm(
                    aux["pred_abs"][:, 0, 1, :]
                    - nf[:, 0, min(t + 1, nf.shape[2] - 1), :], dim=-1)
                mxe = torch.maximum(mxe, pe)
                mxs = torch.maximum(
                    mxs, aux["slack"].reshape(8, -1).max(1).values)
                x = dyn.step(x, u, wind.sample(p0), 0.2)
                d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
                ms = torch.minimum(ms, d)
            coll = ms < DSEP
            s = mxs > 1.0
            e = mxe > 20.0
            n_coll += int(coll.sum())
            slack_only += int((coll & s & ~e).sum())
            pred_only += int((coll & e & ~s).sum())
            both += int((coll & s & e).sum())
            tot += 8
        return {"n_conflicts": n_coll, "slack_only": slack_only,
                "pred_only": pred_only, "both": both,
                "neither": n_coll - slack_only - pred_only - both,
                "total_episodes": tot}
    dump("attribution.json",
         {"Stage-1": attr("stage1_full.pt"),
          "Stage-2": attr("stage2_final.pt")})


# ---------- planner alpha x a_max heatmap (fig 6) ----------
@torch.no_grad()
def collect_heatmap():
    pred = ec.load_gmm_predictor(f"{PLAN}/stage2_final.pt", DEV)
    alphas = [0.1, 0.2, 0.4, 0.6]
    amaxs = [5.0, 10.0, 15.0, 20.0]
    grid = []
    nh_grid = 96  # smaller n for the 16-cell sweep to keep runtime sane
    for a in alphas:
        row = []
        for am in amaxs:
            pl = CBFMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"],
                             dt=0.2, d_sep=DSEP, alpha=a, a_max=am)
            m = ec.evaluate_policy(pred, pl, n=nh_grid, device=DEV)
            row.append(m["CR_%"])
            print(f"  alpha={a} a_max={am} -> CR={m['CR_%']:.1f}")
        grid.append(row)
    dump("planner_heatmap.json",
         {"alphas": alphas, "amaxs": amaxs, "CR": grid, "n": nh_grid})


def main():
    print("=== min-sep arrays + rollouts ===")
    s1 = ec.load_gmm_predictor(f"{PLAN}/stage1_full.pt", DEV)
    s2 = ec.load_gmm_predictor(f"{PLAN}/stage2_final.pt", DEV)
    cv = ConstantVelocityPredictor(T=30, K=5).double().to(DEV)
    best = ec.make_best_planner
    collect_minsep(s2, best(), "PlanGrad", save_rollout=True)
    collect_minsep(s1, best(), "Fixed-Predictor", save_rollout=True)
    collect_minsep(cv, best(), "Constant-Velocity")
    # Conformal: inflate margin (real conformal radius)
    from conformal import conformal_radius
    r, _ = conformal_radius(s1, delta=0.1, horizon=ec.BEST_PLANNER["horizon"],
                            device=DEV, seed=ec.GLOBAL_SEED)
    plc = CBFMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"],
                      dt=0.2, d_sep=DSEP + r, alpha=ec.BEST_PLANNER["alpha"],
                      a_max=ec.BEST_PLANNER["a_max"])
    collect_minsep(s1, plc, "Conformal-MPC")
    # no-CBF
    vp = VanillaMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"],
                         dt=0.2, d_sep=DSEP, a_max=ec.BEST_PLANNER["a_max"])
    collect_minsep(s2, vp, "Vanilla-MPC", policy=SafePolicy(s2, vp))

    print("=== rq5 profile ==="); collect_profile()
    print("=== attribution ==="); collect_attribution()
    print("=== planner heatmap ==="); collect_heatmap()
    print("ALL DATA COLLECTED")


if __name__ == "__main__":
    main()
