"""Baseline 01 -- Constant-Velocity predictor + tuned CBF-MPC planner.

Evaluation model : ConstantVelocityPredictor (training-free) driving the
                   SAME best planner (alpha=0.1, Hp=15, a_max=20) inside
                   the unified SafePolicy closed loop.
Purpose          : isolates the value of a LEARNED predictor. Vs PlanGrad
                   (stage2_final) under the identical planner, the gap shows
                   how much the task-aligned learned predictor buys.

Run:  python3 run.py --n 200    (from this directory, after sourcing env.sh)
Result -> ./result.txt and ./result.json
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(os.path.dirname(HERE), "common")
sys.path.insert(0, COMMON)
sys.path.insert(0, HERE)

import eval_common as ec                 # noqa: E402
from cv_predictor import ConstantVelocityPredictor   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    dev = ec.device_str(True)
    print(f"[01 CV] device={dev}  n={a.n}")

    predictor = ConstantVelocityPredictor(T=30, K=5).double().to(dev)
    planner = ec.make_best_planner()

    metrics = ec.evaluate_policy(predictor, planner, n=a.n, device=dev)
    print(metrics)
    ec.write_result(HERE, "Constant-Velocity + CBF-MPC (best planner)",
                    "ConstantVelocityPredictor (no training)", metrics)


if __name__ == "__main__":
    main()
