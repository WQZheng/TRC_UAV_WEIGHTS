"""Sanity: eval_common must reproduce final_best.py's headline ordering.
Runs PlanGrad (stage2_final) and the Stage-1 reference under the unified
evaluator. Use --n 24 for a quick check, --n 200 for the headline.
"""
import argparse
import eval_common as ec

PLANGRAD = "/data/lab/plangrad/plangrad_sim"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    dev = ec.device_str(True)
    print("device", dev, "| n", a.n)

    planner = ec.make_best_planner()
    s1 = ec.load_gmm_predictor(f"{PLANGRAD}/stage1_full.pt", dev)
    s2 = ec.load_gmm_predictor(f"{PLANGRAD}/stage2_final.pt", dev)

    print("\n-- Stage-1 reference --")
    m1 = ec.evaluate_policy(s1, planner, n=a.n, device=dev)
    print(m1)
    print("\n-- PlanGrad (Stage-2, stage2_final) --")
    m2 = ec.evaluate_policy(s2, planner, n=a.n, device=dev)
    print(m2)

    print("\nExpected (final_best, n=200): Stage-1 CR~12.5 ADE~20.9 minSep~46.3 ;"
          " Stage-2 CR~12.0 ADE~10.2 minSep~48.4")


if __name__ == "__main__":
    main()
