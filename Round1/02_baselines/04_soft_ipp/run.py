"""Baseline 04 -- Soft-IPP evaluation: jointly-trained predictor + soft
Vanilla-MPC planner (NO CBF), evaluated in the unified closed loop.

Evaluation model : soft_joint.pt -- the predictor produced by train.py
                   (task-aligned joint training through a SOFT-penalty
                   planner, no CBF). Evaluated with the SAME soft Vanilla-MPC
                   planner it was trained with (a_max=20, Hp=15).
Purpose          : a fair DIPP-style integrated predict-and-plan baseline.
                   Compared with PlanGrad (stage2_final + CBF-MPC), the gap
                   shows the benefit of training & acting through a HARD
                   safety certificate rather than a soft cost.

NOTE: run train.py first to produce soft_joint.pt.

Run:  python3 run.py --n 200
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

import eval_common as ec                    # noqa: E402
from safe_policy import SafePolicy          # noqa: E402
from vanilla_mpc import VanillaMPCLayer     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--weights", type=str,
                    default=os.path.join(HERE, "soft_joint.pt"))
    ap.add_argument("--w_rep", type=float, default=50.0)
    a = ap.parse_args()
    dev = ec.device_str(True)
    print(f"[04 Soft-IPP] device={dev}  n={a.n}  weights={a.weights}")

    predictor = ec.load_gmm_predictor(a.weights, dev)
    planner = VanillaMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"],
                              dt=ec.DT, d_sep=ec.D_SEP,
                              a_max=ec.BEST_PLANNER["a_max"], w_rep=a.w_rep)
    policy = SafePolicy(predictor, planner)

    metrics = ec.evaluate_policy(predictor, planner, n=a.n, device=dev,
                                 policy=policy, ade_predictor=predictor)
    print(metrics)
    ec.write_result(HERE, "Soft-IPP (joint training, soft planner, no CBF)",
                    "soft_joint.pt (jointly trained); Vanilla-MPC planner",
                    metrics, extra={"w_rep": a.w_rep})


if __name__ == "__main__":
    main()
