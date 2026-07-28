"""Baseline 03 -- Fixed (displacement-only) predictor + tuned CBF-MPC.

Evaluation model : Stage-1 predictor (stage1_full.pt), i.e. trained ONLY
                   for displacement accuracy (GMM-NLL + minADE), with NO
                   task-aligned fine-tuning, driving the SAME best CBF-MPC
                   planner (alpha=0.1, Hp=15, a_max=20).
Purpose          : this is the "predict-then-plan with a frozen, accuracy-
                   trained predictor" baseline -- the de-facto standard
                   pipeline. The gap to PlanGrad (stage2_final, identical
                   planner) isolates exactly what the Stage-2 task-aligned
                   fine-tuning (the core contribution) adds on top.

Run:  python3 run.py --n 200
Result -> ./result.txt / ./result.json
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(os.path.dirname(HERE), "common")
sys.path.insert(0, COMMON)
sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")

import eval_common as ec               # noqa: E402

PLANGRAD = "/data/lab/plangrad/plangrad_sim"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    dev = ec.device_str(True)
    print(f"[03 Fixed-predictor] device={dev}  n={a.n}")

    predictor = ec.load_gmm_predictor(f"{PLANGRAD}/stage1_full.pt", dev)
    planner = ec.make_best_planner()

    metrics = ec.evaluate_policy(predictor, planner, n=a.n, device=dev)
    print(metrics)
    ec.write_result(HERE, "Fixed Stage-1 predictor + CBF-MPC (best planner)",
                    "stage1_full.pt (displacement-only, frozen)", metrics)


if __name__ == "__main__":
    main()
