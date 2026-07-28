"""Baseline 05 -- Conformal-MPC: frozen predictor + CBF-MPC whose
separation margin is inflated by a split-conformal radius.

Evaluation model : Stage-1 predictor (stage1_full.pt), FROZEN. A split-
                   conformal calibration (on disjoint encounters 2000..2500)
                   gives a radius r_conf at miscoverage delta; the CBF-MPC
                   then enforces separation d_sep + r_conf (best planner
                   alpha=0.1, Hp=15, a_max=20).
Purpose          : the standard "uncertainty-aware safe planning" baseline
                   -- handle prediction uncertainty by an offline calibrated
                   margin instead of by task-aligned fine-tuning. Compared
                   with PlanGrad (which instead REALLOCATES predictive
                   variance via Stage-2), it shows whether a calibrated
                   static buffer matches learned task alignment.

Run:  python3 run.py --n 200 --delta 0.1
Result -> ./result.txt / ./result.json
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(os.path.dirname(HERE), "common")
sys.path.insert(0, COMMON)
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")

import eval_common as ec                  # noqa: E402
from cbf_mpc import CBFMPCLayer           # noqa: E402
from conformal import conformal_radius    # noqa: E402

PLANGRAD = "/data/lab/plangrad/plangrad_sim"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--delta", type=float, default=0.1)
    a = ap.parse_args()
    dev = ec.device_str(True)
    print(f"[05 Conformal-MPC] device={dev}  n={a.n}  delta={a.delta}")

    predictor = ec.load_gmm_predictor(f"{PLANGRAD}/stage1_full.pt", dev)

    r_conf, info = conformal_radius(predictor, delta=a.delta,
                                    horizon=ec.BEST_PLANNER["horizon"],
                                    device=dev, seed=ec.GLOBAL_SEED)
    print(f"conformal radius r_conf = {r_conf:.2f} m   {info}")

    d_sep_inflated = ec.D_SEP + r_conf
    planner = CBFMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"],
                          dt=ec.DT, d_sep=d_sep_inflated,
                          alpha=ec.BEST_PLANNER["alpha"],
                          a_max=ec.BEST_PLANNER["a_max"])

    # IMPORTANT: conflict is still judged at the TRUE operational d_sep (30 m),
    # not the inflated planning margin -- the buffer is a means, not the metric.
    metrics = ec.evaluate_policy(predictor, planner, n=a.n, device=dev,
                                 d_sep=ec.D_SEP)
    print(metrics)
    ec.write_result(HERE, "Conformal-MPC (calibrated margin) + frozen predictor",
                    "stage1_full.pt (frozen) + conformal-inflated CBF-MPC",
                    metrics, extra={"r_conf_m": r_conf, "delta": a.delta,
                                    "d_sep_planning": d_sep_inflated,
                                    "calib": info})


if __name__ == "__main__":
    main()
