"""Reference (NOT a baseline): our method PlanGrad = Stage-2 task-aligned
predictor (stage2_final.pt) + tuned CBF-MPC, scored by the SAME unified
evaluator so the results table has the method row to compare baselines to.
Writes into baselines/00_plangrad_reference/.
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")
import eval_common as ec   # noqa: E402

PLANGRAD = "/data/lab/plangrad/plangrad_sim"
OUT = os.path.join(os.path.dirname(HERE), "00_plangrad_reference")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    dev = ec.device_str(True)
    print(f"[REF PlanGrad] device={dev}  n={a.n}")
    predictor = ec.load_gmm_predictor(f"{PLANGRAD}/stage2_final.pt", dev)
    planner = ec.make_best_planner()
    metrics = ec.evaluate_policy(predictor, planner, n=a.n, device=dev)
    print(metrics)
    ec.write_result(OUT, "PlanGrad (OURS): Stage-2 + CBF-MPC (best planner)",
                    "stage2_final.pt + tuned CBF-MPC", metrics)


if __name__ == "__main__":
    main()
