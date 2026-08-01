#!/usr/bin/env python3
"""Standalone FAST (OSQP) planner-CR heatmap runner for Figure 9.

The main collector (collect_fig_data.py) loads its heatmap function into memory
at import time; a mid-run edit to the file does not affect the running process.
This standalone runner imports the *patched* fast heatmap_n200() fresh and dumps
fig_data/planner_heatmap_n200.json without re-running the (already-completed)
7-arm minsep/effort sweep or the errdir profile.

Run AFTER collect_fig_data.py has dumped minsep_effort.npz and
errdir_profile.npz (those are the slow differentiable/rollout parts we keep).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_fig_data as C

if __name__ == "__main__":
    t0 = time.time()
    print("=== FAST planner heatmap (standalone) ===", flush=True)
    C.heatmap_n200()
    print(f"heatmap done in {time.time() - t0:.0f}s", flush=True)
