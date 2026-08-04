"""Stage-1b (domain-adapted predictor) + tuned CBF-MPC, scored by the SAME
unified evaluator as every other arm.

Rationale: Stage-1b is reported in the manuscript (Outcome B: ADE 1.84 m at
CR 11.5%) and appears in the figures, but it previously had NO canonical
runner -- its per-episode arrays existed only inside figures_gen's private
re-implementation of the rollout. This runner closes that gap so the figure
and the table for Stage-1b share one provenance, exactly like the other arms.

Identical to 00_plangrad_reference except for the predictor weights.
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")
import eval_common as ec   # noqa: E402

PLANGRAD = "/data/lab/plangrad/plangrad_sim"
OUT = HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    dev = ec.device_str(True)
    print(f"[07 Stage-1b] device={dev}  n={a.n}")
    predictor = ec.load_gmm_predictor(f"{PLANGRAD}/stage1b_domainadapt.pt", dev)
    planner = ec.make_best_planner()
    metrics = ec.evaluate_policy(predictor, planner, n=a.n, device=dev)
    print(metrics)
    ec.write_result(OUT, "Stage-1b: domain-adapted predictor + CBF-MPC (best planner)",
                    "stage1b_domainadapt.pt + tuned CBF-MPC", metrics)


if __name__ == "__main__":
    main()
