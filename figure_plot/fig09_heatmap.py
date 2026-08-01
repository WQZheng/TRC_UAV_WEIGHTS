#!/usr/bin/env python3
"""Figure 9 -- conflict rate across tested planner settings.

Single-panel heatmap. x = acceleration bound a_max (5/10/15/20 m/s^2),
y = barrier coefficient gamma (0.1/0.2/0.4/0.6, small at bottom -> large at
top). Each cell shows the integer CR%, with text colour auto-switched (white
on dark cells, black on light). Vertical colourbar (Conflict rate, CR %),
range fixed to the full tested span. Deployment configuration (gamma=0.1,
a_max=20) marked with a thick black rectangle; a small two-item legend
(Deployment / Training) sits below. Symbol is gamma (NOT the old alpha).

PROTOCOL AUDIT (completed before drawing, per the plan's precondition)
  The archived collect_data.collect_heatmap used stage2_final, seed 12345,
  Hp=15, evtol 2500-2999, paired generation -- BUT n=96 and default wind, and
  had never actually been run. This figure therefore requires the RE-RUN at
  n=200, eta_w=0.3 (matching the main table), produced on the Lab by
  figures_gen/collect_fig_data.py -> fig_data/planner_heatmap_n200.json.
  The stale local planner_heatmap.json (n=96, alpha axis) is NOT used.

  Heatmap Hp = 15 = deployment Hp, so the deployment cell is drawn on the
  grid. If a future grid uses a different Hp, mark deployment only and explain
  the training point in the caption text (not on the grid).

DATA PROVENANCE
  fig_data/planner_heatmap_n200.json with keys: gammas, amaxs, CR (2-D, rows
  indexed by gamma), n (=200), eta (=0.3). n=200, seed 12345, deployment
  planner family, evtol encounters 2500-2999.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures_generated")
os.makedirs(OUT, exist_ok=True)
JSON = fs.find_data("fig_data/planner_heatmap_n200.json",
                    "baselines/figures_gen/fig_data/planner_heatmap_n200.json")

DEPLOY = (0.1, 20.0)   # (gamma, a_max)


def main():
    fs.set_rc()
    if not JSON or not os.path.exists(JSON):
        raise SystemExit("collector planner_heatmap_n200.json not found; "
                         "run figures_gen/collect_fig_data.py on the Lab "
                         "(n=200, eta_w=0.3) first.")
    d = json.load(open(JSON))
    gammas = [float(g) for g in d["gammas"]]
    amaxs = [float(a) for a in d["amaxs"]]
    CR = np.asarray(d["CR"], float)          # shape (len(gammas), len(amaxs))
    n = d.get("n"); eta = d.get("eta")
    print(f"heatmap n={n} eta={eta} gammas={gammas} amaxs={amaxs}")

    # order rows small gamma at bottom -> large at top
    gi = np.argsort(gammas)
    gammas = [gammas[i] for i in gi]; CR = CR[gi]

    # main heatmap + slim marginal-mean strips (row means right, col means top).
    # NO contour / interpolated "CR=30%" boundary: on a 4x4 grid that would
    # require interpolating a smooth edge the discrete data cannot support.
    fig = plt.figure(figsize=(5.3, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.0, 0.7],
                          height_ratios=[0.7, 4.0], wspace=0.06, hspace=0.06)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    im = ax.imshow(CR, origin="lower", aspect="auto", cmap="YlOrRd",
                   vmin=float(np.floor(CR.min() / 10) * 10),
                   vmax=float(np.ceil(CR.max() / 10) * 10))
    ax.set_xticks(range(len(amaxs))); ax.set_xticklabels([f"{a:g}" for a in amaxs])
    ax.set_yticks(range(len(gammas))); ax.set_yticklabels([f"{g:g}" for g in gammas])
    ax.set_xlabel(r"Acceleration bound, $a_{\max}$ (m/s$^2$)")
    ax.set_ylabel(r"Barrier coefficient, $\gamma$")

    # marginal means (grey bars; a compact "which axis moves CR" summary)
    col_mean = CR.mean(0); row_mean = CR.mean(1)
    ax_top.bar(range(len(amaxs)), col_mean, width=0.7, color="0.55", lw=0)
    ax_top.set_ylabel("col\nmean", fontsize=6.6, labelpad=1)
    ax_top.tick_params(labelbottom=False, labelsize=6.2)
    ax_right.barh(range(len(gammas)), row_mean, height=0.7, color="0.55", lw=0)
    ax_right.set_xlabel("row mean", fontsize=6.6)
    ax_right.tick_params(labelleft=False, labelsize=6.2)
    for a_ in (ax_top, ax_right):
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
        a_.grid(False)

    vmid = 0.5 * (CR.min() + CR.max())
    for i in range(len(gammas)):
        for j in range(len(amaxs)):
            ax.text(j, i, f"{CR[i, j]:.0f}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if CR[i, j] > vmid else "0.1")

    # deployment cell rectangle
    try:
        dj = amaxs.index(DEPLOY[1]); di = gammas.index(DEPLOY[0])
        ax.add_patch(Rectangle((dj - 0.5, di - 0.5), 1, 1, fill=False,
                               ec="black", lw=2.4, zorder=5))
    except ValueError:
        print("WARNING: deployment cell not on grid; marking skipped")

    cax = ax.inset_axes([1.28, 0.0, 0.06, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Conflict rate, CR (%)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7)

    # small legend below (Deployment only; Training explained in caption)
    handles = [plt.Line2D([], [], marker="s", ls="none", mfc="none",
                          mec="black", mew=2.0, ms=11, label="Deployment")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=1, frameon=False, fontsize=8, handletextpad=0.5)

    out = os.path.join(OUT, "fig09_heatmap.pdf")
    fig.savefig(out, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
