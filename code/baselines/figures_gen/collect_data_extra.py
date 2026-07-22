"""Run ONLY the data pieces not yet saved (or saved as placeholder):
Conformal-MPC (real inflated margin), Vanilla-MPC, RQ5 profile,
attribution, planner heatmap. Reuses functions from collect_data.py.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import collect_data as cd
import eval_common as ec
from cbf_mpc import CBFMPCLayer
from safe_policy import SafePolicy
from vanilla_mpc import VanillaMPCLayer
from conformal import conformal_radius

PLAN = cd.PLAN
DEV = cd.DEV

s1 = ec.load_gmm_predictor(f"{PLAN}/stage1_full.pt", DEV)
s2 = ec.load_gmm_predictor(f"{PLAN}/stage2_final.pt", DEV)

print("=== Conformal-MPC (real) ===")
r, _ = conformal_radius(s1, delta=0.1, horizon=ec.BEST_PLANNER["horizon"],
                        device=DEV, seed=ec.GLOBAL_SEED)
print(f"r_conf={r:.2f}")
plc = CBFMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"], dt=0.2,
                  d_sep=cd.DSEP + r, alpha=ec.BEST_PLANNER["alpha"],
                  a_max=ec.BEST_PLANNER["a_max"])
cd.collect_minsep(s1, plc, "Conformal-MPC")

print("=== Vanilla-MPC ===")
vp = VanillaMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"], dt=0.2,
                     d_sep=cd.DSEP, a_max=ec.BEST_PLANNER["a_max"])
cd.collect_minsep(s2, vp, "Vanilla-MPC", policy=SafePolicy(s2, vp))

print("=== rq5 profile ==="); cd.collect_profile()
print("=== attribution ==="); cd.collect_attribution()
print("=== planner heatmap ==="); cd.collect_heatmap()
print("REST DONE")
