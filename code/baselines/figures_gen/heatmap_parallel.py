#!/usr/bin/env python3
"""Parallel FAST (OSQP) planner-CR heatmap for Figure 9.

Each (gamma, a_max) cell is a full n=200 closed-loop CR evaluation (~15 min
serial), so the 16-cell grid is run as 16 independent worker processes (the Lab
has 256 cores). Each worker computes ONE cell via collect_fig_data._cr_fast
(identical fast OSQP path used for the main arms, which reproduce the table),
prints its result, and the parent assembles fig_data/planner_heatmap_n200.json
in exactly the schema Figure 9 expects.

Thread caps per worker keep OSQP/numpy from oversubscribing across the 16 procs.
"""
import os, sys, json, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_data")
os.makedirs(OUT, exist_ok=True)
GAMMAS = [0.1, 0.2, 0.4, 0.6]
AMAXS = [5.0, 10.0, 15.0, 20.0]

WORKER = """
import os, sys
os.environ.setdefault("GUAM_MAT",
    "/data/lab/TRC_UAV_WEIGHTS/code/GUAM/Challenge_Problems/Data_Set_1.mat")
sys.path.insert(0, "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen")
sys.path.insert(0, "/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim")
import collect_fig_data as C
from safe_policy import SafePolicy
from cbf_mpc import CBFMPCLayer
g = float(sys.argv[1]); am = float(sys.argv[2])
Hp = C.ec.BEST_PLANNER["horizon"]
pred = C._load("stage2_final.pt")
sp = SafePolicy(pred, CBFMPCLayer(n_neighbors=1, horizon=Hp, dt=C.DT,
               d_sep=C.DSEP, alpha=C.ec.BEST_PLANNER["alpha"],
               a_max=C.ec.BEST_PLANNER["a_max"]))
cr = C._cr_fast(pred, sp, g, am, Hp)
print("CELL %.4f %.4f %.4f" % (g, am, cr), flush=True)
"""

def launch(g, am):
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[v] = "6"
    env["GUAM_MAT"] = ("/data/lab/TRC_UAV_WEIGHTS/code/GUAM/"
                       "Challenge_Problems/Data_Set_1.mat")
    lf = open(os.path.join(OUT, f"hm_{g}_{am}.log"), "w")
    p = subprocess.Popen([sys.executable, "-u", "-c", WORKER,
                          str(g), str(am)], stdout=lf, stderr=subprocess.STDOUT,
                         env=env)
    return p, lf

def main():
    t0 = time.time()
    procs = {}
    for g in GAMMAS:
        for am in AMAXS:
            procs[(g, am)] = launch(g, am)
    print(f"launched {len(procs)} cell workers", flush=True)
    CR = {}
    for (g, am), (p, lf) in procs.items():
        p.wait(); lf.close()
        # parse the cell's CR from its log
        val = None
        for line in open(os.path.join(OUT, f"hm_{g}_{am}.log")):
            if line.startswith("CELL"):
                val = float(line.split()[3])
        CR[(g, am)] = val
        print(f"  gamma={g} a_max={am} -> CR={val}", flush=True)
    grid = [[round(CR[(g, am)], 1) for am in AMAXS] for g in GAMMAS]
    obj = {"gammas": GAMMAS, "amaxs": AMAXS, "CR": grid, "n": 200,
           "seed": 12345, "eta_w": 0.3, "eta": 0.3,
           "Hp": 15, "predictor": "stage2_final.pt",
           "eval_range": [2500, 3000]}
    json.dump(obj, open(os.path.join(OUT, "planner_heatmap_n200.json"), "w"),
              indent=1)
    print(f"heatmap dumped in {time.time()-t0:.0f}s: {grid}", flush=True)

if __name__ == "__main__":
    main()
