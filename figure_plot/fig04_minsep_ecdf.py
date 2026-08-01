#!/usr/bin/env python3
"""Figure 4 -- empirical distributions of minimum realized separation.

Single-panel ECDF with an inset zoom around the 30 m standard.
  Main: ECDF of per-episode minimum separation (15-75 m) for all seven
        non-Oracle arms. Certificate arms = thin cool solid lines; Vanilla /
        Soft-IPP = thicker warm dashed. Thin red dashed line at x=30 m
        labelled only "30 m".
  Inset: zoom x in [25,35], y in [0.05,0.20], showing only Stage 2, Stage-1b,
        Stage 1, CV, Conformal (Vanilla/Soft omitted to avoid occlusion). No
        inset legend.

DATA PROVENANCE
  Per-episode minimum-separation arrays for the seven arms, produced on the
  Lab by figures_gen/collect_fig_data.py at n=200, seed 12345, eta_w=0.3,
  evtol encounters 2500-2999, deployment planner, using the OSQP fast CBF
  solver (identical QP to the differentiable layer; verified). Loaded from
  fig_data/minsep_effort.npz (keys "<arm>__minsep"). Arm CR reproduces the v9
  main table (Stage 2 ~11.0). The stale local baselines/figures_gen/data JSONs
  are NOT used (old dead path, PlanGrad CR=12.0, contradicts the 11.0 table).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures_generated")
os.makedirs(OUT, exist_ok=True)
NPZ = fs.find_data("fig_data/minsep_effort.npz",
                   "baselines/figures_gen/fig_data/minsep_effort.npz")

# collector arm keys -> canonical figstyle names
ARMMAP = {"Stage2": "Stage 2", "Stage-1b": "Stage-1b",
          "Fixed-Predictor": "Stage 1", "Constant-Velocity": "Constant-Velocity",
          "Conformal-MPC": "Conformal-MPC", "Vanilla-MPC": "Vanilla-MPC",
          "Soft-IPP": "Soft-IPP"}
INSET_ARMS = ["Stage 2", "Stage-1b", "Stage 1", "Constant-Velocity",
              "Conformal-MPC"]


def ecdf(x):
    xs = np.sort(np.asarray(x, float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def main():
    fs.set_rc()
    if not NPZ or not os.path.exists(NPZ):
        raise SystemExit("collector output minsep_effort.npz not found; "
                         "run figures_gen/collect_fig_data.py on the Lab first.")
    d = np.load(NPZ, allow_pickle=True)
    arms = {}
    for k in d.files:
        if k.endswith("__minsep"):
            raw = k[:-len("__minsep")]
            if raw in ARMMAP:
                arms[ARMMAP[raw]] = d[k]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name in fs.ordered(arms.keys()):
        xs, ys = ecdf(arms[name])
        s = fs.STYLE[name]
        warm = s["family"] == "free"
        ax.plot(xs, ys, color=s["color"], ls=s["ls"],
                lw=2.0 if warm else 1.3, label=name, zorder=3)
    ax.axvline(fs.THRESH, **fs.THRESH_KW)
    ax.annotate("30 m", (fs.THRESH, 0.9), textcoords="offset points",
                xytext=(3, 0), fontsize=8, color="#D55E00")
    ax.set_xlim(15, 75); ax.set_ylim(0, 1)
    ax.set_xlabel("Minimum realized separation (m)")
    ax.set_ylabel("Cumulative fraction of encounters")

    # inset zoom
    iax = ax.inset_axes([0.52, 0.14, 0.44, 0.42])
    for name in INSET_ARMS:
        if name in arms:
            xs, ys = ecdf(arms[name])
            s = fs.STYLE[name]
            iax.plot(xs, ys, color=s["color"], ls=s["ls"], lw=1.3)
    iax.axvline(fs.THRESH, **fs.THRESH_KW)
    iax.set_xlim(25, 35); iax.set_ylim(0.05, 0.20)
    iax.tick_params(labelsize=6.5)
    iax.set_title("zoom", fontsize=7, pad=2)

    # legend bottom, two rows, below axes
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4,
              frameon=False, fontsize=8, columnspacing=1.3, handletextpad=0.4)
    out = os.path.join(OUT, "fig04_minsep_ecdf.pdf")
    fig.savefig(out, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
