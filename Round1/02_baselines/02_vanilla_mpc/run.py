"""Baseline 02 -- Vanilla-MPC (no CBF) + the SAME PlanGrad predictor.

Evaluation model : PlanGrad Stage-2 predictor (stage2_final.pt) driving a
                   Vanilla-MPC planner that has NO control-barrier-function
                   safety certificate -- collision avoidance only via a soft
                   repulsion penalty.
Purpose          : isolates the contribution of the CBF safety layer. We
                   deliberately keep the BEST predictor so any safety loss
                   is attributable to dropping the CBF certificate, not to a
                   worse predictor. Compare against PlanGrad (same predictor,
                   CBF-MPC planner).

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

import eval_common as ec                       # noqa: E402
from safe_policy import SafePolicy             # noqa: E402
from vanilla_mpc import VanillaMPCLayer        # noqa: E402

PLANGRAD = "/data/lab/plangrad/plangrad_sim"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--w_rep", type=float, default=50.0)
    a = ap.parse_args()
    dev = ec.device_str(True)
    print(f"[02 Vanilla-MPC] device={dev}  n={a.n}  w_rep={a.w_rep}")

    predictor = ec.load_gmm_predictor(f"{PLANGRAD}/stage2_final.pt", dev)
    planner = VanillaMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"],
                              dt=ec.DT, d_sep=ec.D_SEP,
                              a_max=ec.BEST_PLANNER["a_max"], w_rep=a.w_rep)
    # SafePolicy reuses its accel->control map; only the planner differs.
    policy = SafePolicy(predictor, planner)

    metrics = ec.evaluate_policy(predictor, planner, n=a.n, device=dev,
                                 policy=policy, ade_predictor=predictor)
    print(metrics)
    ec.write_result(HERE, "Vanilla-MPC (no CBF) + PlanGrad predictor",
                    "stage2_final.pt (predictor); Vanilla-MPC planner",
                    metrics, extra={"w_rep": a.w_rep})


if __name__ == "__main__":
    main()
