"""Clean final comparison: Stage-1 vs Stage-2 on a LARGE held-out set.

Conflict rate is a 0/1-per-episode metric with high variance, so a small
eval set (n=64) is misleading. Use n>=200. Writes FINAL2.txt with flush
so progress is visible.

Usage:
    python3 final_compare.py 200      # n_eval = 200
"""
import sys
import torch
from seeding import set_seed
from config import GUAM_MAT
from predictor import GMMTrajectoryPredictor
from guam_encounters import GUAMEncounters
import eval_stage1_vs_stage2 as EV

OUT = "FINAL2.txt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def w(line):
    with open(OUT, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load(p):
    n = GMMTrajectoryPredictor(T=30, K=5).double().to(DEV)
    n.load_state_dict(torch.load(p, map_location=DEV))
    n.eval()
    return n


def ev(ckpt):
    set_seed(12345)
    g = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=12345)
    return EV.evaluate(load(ckpt), g, N, 20, 8, 0.2, 0.3, DEV)


if __name__ == "__main__":
    open(OUT, "w").close()
    w("FINAL2 compare (seed=12345, held-out 2500-3000, n=%d)" % N)
    m1 = ev("stage1_full.pt")
    w("Stage-1   CR=%.1f%%  minSep=%.1f m  ADE=%.2f m  minADE=%.2f m"
      % (m1["conflict_rate_%"], m1["mean_min_sep_m"], m1["ADE_m"], m1["minADE_m"]))
    m2 = ev("stage2_final.pt")
    w("Stage-2   CR=%.1f%%  minSep=%.1f m  ADE=%.2f m  minADE=%.2f m"
      % (m2["conflict_rate_%"], m2["mean_min_sep_m"], m2["ADE_m"], m2["minADE_m"]))
    w("delta     CR %+.1f pts  minSep %+.1f m  ADE %+.2f m"
      % (m2["conflict_rate_%"] - m1["conflict_rate_%"],
         m2["mean_min_sep_m"] - m1["mean_min_sep_m"],
         m2["ADE_m"] - m1["ADE_m"]))
