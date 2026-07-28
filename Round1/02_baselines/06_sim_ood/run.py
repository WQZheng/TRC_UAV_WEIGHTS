"""RQ4 / Sim-OOD: generalization under out-of-distribution wind strength.

Sweeps the wind scaling factor eta_w over {0.5 (nominal, ~Sim-Base),
1.0, 1.5 (Sim-OOD)} and evaluates every method under the unified
closed-loop evaluator (n=200, seed 12345, best CBF-MPC planner where
applicable). For each method and each eta_w it reports CR / MinSep /
LeadT / ADE / Energy.

Honesty note: in this codebase the encounter generator is single-
neighbour, so "Sim-OOD" here is realised as STRONGER WIND (eta_w up to
1.5, three to five times the 0.3-0.5 nominal gust scaling), not as
higher multi-neighbour traffic density, which the simulator does not
support. The wind shift is a genuine dynamics-level distribution shift:
the eVTOL must reject larger stochastic gusts while holding separation.

Methods evaluated (all real, reproducible implementations):
  PlanGrad        stage2_final.pt + CBF-MPC      (ours)
  Conformal-MPC   stage1_full.pt (frozen) + conformal-inflated CBF-MPC
  Fixed-Predictor stage1_full.pt + CBF-MPC
  Constant-Vel    analytic + CBF-MPC
  Vanilla-MPC     stage2_final.pt + soft planner (no CBF)
  Soft-IPP        soft_joint.pt + soft planner (no CBF)

Output: ./ood_results.json  and  ./ood_results.txt
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(B, "common"))
sys.path.insert(0, os.path.join(B, "01_constant_velocity"))
sys.path.insert(0, os.path.join(B, "02_vanilla_mpc"))
sys.path.insert(0, os.path.join(B, "05_conformal_mpc"))
sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")

import eval_common as ec                       # noqa: E402
from safe_policy import SafePolicy             # noqa: E402
from cbf_mpc import CBFMPCLayer                # noqa: E402
from cv_predictor import ConstantVelocityPredictor   # noqa: E402
from vanilla_mpc import VanillaMPCLayer        # noqa: E402
from conformal import conformal_radius         # noqa: E402

PLANGRAD = "/data/lab/plangrad/plangrad_sim"
import argparse
_ap=argparse.ArgumentParser()
_ap.add_argument('--n',type=int,default=200)
_ap.add_argument('--etas',default='0.5,1.0,1.5')
_a,_=_ap.parse_known_args()
ETAS=[float(x) for x in _a.etas.split(',')]
N=_a.n


def main():
    dev = ec.device_str(True)
    print(f"[RQ4 Sim-OOD] device={dev}  n={N}  eta_w sweep={ETAS}")

    s1 = ec.load_gmm_predictor(f"{PLANGRAD}/stage1_full.pt", dev)
    s2 = ec.load_gmm_predictor(f"{PLANGRAD}/stage2_final.pt", dev)
    cv = ConstantVelocityPredictor(T=30, K=5).double().to(dev)
    soft = ec.load_gmm_predictor(
        f"{B}/04_soft_ipp/soft_joint.pt", dev)

    # conformal radius is calibrated ONCE on the calibration split at
    # nominal wind, then held fixed across the OOD sweep (as a deployed
    # buffer would be).
    r_conf, cinfo = conformal_radius(s1, delta=0.1,
                                     horizon=ec.BEST_PLANNER["horizon"],
                                     device=dev, seed=ec.GLOBAL_SEED)
    print(f"conformal radius (fixed) = {r_conf:.2f} m")

    def best():
        return ec.make_best_planner()

    def conf_planner():
        return CBFMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"],
                           dt=ec.DT, d_sep=ec.D_SEP + r_conf,
                           alpha=ec.BEST_PLANNER["alpha"],
                           a_max=ec.BEST_PLANNER["a_max"])

    def soft_planner():
        return VanillaMPCLayer(n_neighbors=1,
                               horizon=ec.BEST_PLANNER["horizon"], dt=ec.DT,
                               d_sep=ec.D_SEP, a_max=ec.BEST_PLANNER["a_max"],
                               w_rep=50.0)

    # method registry: name -> closure(eta) -> metrics
    def run_cbf(pred, eta):
        return ec.evaluate_policy(pred, best(), n=N, device=dev, eta_w=eta)

    def run_conf(eta):
        pl = conf_planner()
        return ec.evaluate_policy(s1, pl, n=N, device=dev, d_sep=ec.D_SEP,
                                  eta_w=eta)

    def run_soft(pred, eta):
        pl = soft_planner()
        pol = SafePolicy(pred, pl)
        return ec.evaluate_policy(pred, pl, n=N, device=dev, policy=pol,
                                  ade_predictor=pred, eta_w=eta)

    methods = [
        ("PlanGrad (ours)",  "stage2_final + CBF-MPC", lambda e: run_cbf(s2, e)),
        ("Conformal-MPC",    "stage1 + conformal CBF", run_conf),
        ("Fixed-Predictor",  "stage1 + CBF-MPC",       lambda e: run_cbf(s1, e)),
        ("Constant-Velocity","CV + CBF-MPC",           lambda e: run_cbf(cv, e)),
        ("Vanilla-MPC",      "stage2 + soft (no CBF)", lambda e: run_soft(s2, e)),
        ("Soft-IPP",         "soft_joint + soft",      lambda e: run_soft(soft, e)),
    ]

    results = {"n": N, "seed": ec.GLOBAL_SEED, "etas": ETAS,
               "r_conf": r_conf, "methods": {}}
    for name, model, fn in methods:
        results["methods"][name] = {"model": model, "by_eta": {}}
        for eta in ETAS:
            m = fn(eta)
            results["methods"][name]["by_eta"][str(eta)] = m
            print(f"  {name:18s} eta={eta:<4} "
                  f"CR={m['CR_%']:5.1f}  minSep={m['minSep_m']:5.1f}  "
                  f"LeadT={m['LeadT_s']:.3f}  ADE={m['ADE_m']:5.2f}")

    with open(os.path.join(HERE, "ood_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(HERE, "ood_results.txt"), "w") as f:
        f.write(f"RQ4 Sim-OOD wind sweep (n={N}, seed={ec.GLOBAL_SEED}, "
                f"conformal r={r_conf:.2f} m)\n")
        f.write("eta_w: 0.5=nominal(~Sim-Base), 1.0/1.5=Sim-OOD\n")
        f.write("=" * 70 + "\n")
        for name, d in results["methods"].items():
            f.write(f"\n{name}  [{d['model']}]\n")
            f.write(f"  {'eta':>5} {'CR%':>7} {'minSep':>8} {'LeadT':>7} "
                    f"{'ADE':>7} {'Energy':>8}\n")
            for eta in ETAS:
                m = d["by_eta"][str(eta)]
                f.write(f"  {eta:>5} {m['CR_%']:>7.1f} {m['minSep_m']:>8.1f} "
                        f"{m['LeadT_s']:>7.3f} {m['ADE_m']:>7.2f} "
                        f"{m['Energy']:>8.1f}\n")
    print("\n[saved] ood_results.json + ood_results.txt")


if __name__ == "__main__":
    main()
