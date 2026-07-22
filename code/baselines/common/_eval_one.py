"""Evaluate ANY stage2 checkpoint under the unified evaluator (n=200,
best planner), so we can pick a single consistent PlanGrad model that
both (a) leads the operational table and (b) exhibits the Prop-3
variance-reallocation mechanism. Usage: python3 _eval_one.py --w stage2_v2.pt
"""
import argparse
import eval_common as ec

PLANGRAD = "/data/lab/plangrad/plangrad_sim"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    dev = ec.device_str(True)
    print(f"eval {a.w}  n={a.n}")
    net = ec.load_gmm_predictor(f"{PLANGRAD}/{a.w}", dev)
    planner = ec.make_best_planner()
    m = ec.evaluate_policy(net, planner, n=a.n, device=dev)
    print(m)


if __name__ == "__main__":
    main()
