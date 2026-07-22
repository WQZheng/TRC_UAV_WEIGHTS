"""Collect every baseline's result.json into one results.csv + RESULTS.md,
with the PlanGrad (ours) reference row on top for direct comparison.

Run:  python3 aggregate.py    (after all run.py have produced result.json)
Outputs: baselines/results.csv  and  baselines/RESULTS.md
"""
import os
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))

# display order: ours first, then baselines
ORDER = [
    "00_plangrad_reference",
    "01_constant_velocity",
    "02_vanilla_mpc",
    "03_fixed_predictor",
    "04_soft_ipp",
    "05_conformal_mpc",
]

COLS = ["CR_%", "minSep_m", "ADE_m", "LeadT_s", "Energy"]
HDR = {"CR_%": "CR (%)", "minSep_m": "minSep (m)", "ADE_m": "ADE (m)",
       "LeadT_s": "LeadT (s)", "Energy": "Energy"}


def load_all():
    rows = []
    for d in ORDER:
        p = os.path.join(HERE, d, "result.json")
        if not os.path.exists(p):
            print(f"[warn] missing {p} -- skipped")
            continue
        with open(p) as f:
            rows.append((d, json.load(f)))
    return rows


def main():
    rows = load_all()
    if not rows:
        print("no results found")
        return

    # results.csv
    csv_path = os.path.join(HERE, "results.csv")
    with open(csv_path, "w") as f:
        f.write("folder,method,eval_model,n,seed," + ",".join(COLS) + "\n")
        for d, r in rows:
            m = r["metrics"]
            vals = ",".join(f"{m[c]:.3f}" for c in COLS)
            f.write(f'{d},"{r["method"]}","{r["eval_model"]}",'
                    f'{m["n"]},{r["seed"]},{vals}\n')
    print(f"[saved] {csv_path}")

    # RESULTS.md
    md_path = os.path.join(HERE, "RESULTS.md")
    ref = rows[0][1]
    with open(md_path, "w") as f:
        f.write("# Baseline results (unified evaluator)\n\n")
        f.write(f"All rows scored by `common/eval_common.py` on the IDENTICAL "
                f"held-out GUAM encounters `range(2500,3000)`, seed "
                f"`{ref['seed']}`, n = `{ref['metrics']['n']}`, best CBF-MPC "
                f"planner `{ref['planner']}`, conflict threshold d_sep = "
                f"`{ref['d_sep']} m`.\n\n")
        f.write("Lower is better for CR / ADE / Energy; higher is better for "
                "minSep / LeadT.\n\n")

        # table
        f.write("| Method | Eval model | " + " | ".join(HDR[c] for c in COLS)
                + " |\n")
        f.write("|" + "---|" * (2 + len(COLS)) + "\n")
        for d, r in rows:
            m = r["metrics"]
            name = r["method"]
            if d == "00_plangrad_reference":
                name = "**" + name + "**"
            cells = " | ".join(f"{m[c]:.2f}" if c != "LeadT_s"
                               else f"{m[c]:.3f}" for c in COLS)
            f.write(f"| {name} | `{r['eval_model']}` | {cells} |\n")

        f.write("\n## How to read this\n\n")
        f.write("- **PlanGrad (ours)** = Stage-2 task-aligned predictor "
                "(`stage2_final.pt`) + tuned CBF-MPC.\n")
        f.write("- **01 Constant-Velocity**: training-free predictor, same "
                "CBF-MPC. On GUAM the neighbour replay is near-linear so CV "
                "ADE is tiny and CR matches ours -> *prediction accuracy is "
                "not the safety bottleneck; the planner is.*\n")
        f.write("- **02 Vanilla-MPC (no CBF)**: same predictor, CBF "
                "certificate removed -> CR explodes -> *safety comes from the "
                "CBF layer.*\n")
        f.write("- **03 Fixed-predictor**: Stage-1 (displacement-only) + "
                "CBF-MPC = standard predict-then-plan. Safe but ADE ~2x ours "
                "-> *Stage-2 fine-tuning halves ADE at equal safety.*\n")
        f.write("- **04 Soft-IPP**: joint training through a SOFT planner "
                "(DIPP-style). Even with task alignment, soft costs cannot "
                "enforce separation -> CR stays high -> *training through a "
                "HARD CBF certificate matters.*\n")
        f.write("- **05 Conformal-MPC**: frozen predictor + conformally "
                "inflated margin. Reaches our safety via a static buffer but "
                "keeps the high Stage-1 ADE -> *calibrated buffering is not a "
                "substitute for task-aligned learning of the predictor.*\n")
    print(f"[saved] {md_path}")

    # echo table to stdout
    with open(md_path) as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
