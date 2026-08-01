#!/usr/bin/env python3
"""Figure 8 -- residual-conflict attribution under the deployment planner.

A two-branch diagnostic tree (NOT a bar chart):

        200 evaluated encounters
        /                       \\
  178 no conflict          22 conflict episodes
                            /               \\
              zero-slack re-solve      zero-error replay
              22/22 hard-infeasible    0/22 resolved
                            \\               /
                unresolved within the tested
                planning-and-control envelope

Colours: root deep blue, diagnostic nodes light blue, terminal node dark
grey-blue, the 178-no-conflict branch pale grey. No red (this is a diagnostic
logic diagram, not an error-category chart). Bottom-right inset: histogram of
the number of hard-infeasible steps per conflict episode (0..20), mean 17.8/20.

DATA PROVENANCE (authoritative)
  200 encounters; Stage 2 conflicts = 22; zero-slack re-solve 22/22 hard
  infeasible (mean 17.77/20); zero-error replay 0/22 resolved.
  Source: Round1/05_results/robustness/p0_referee/P1_ORACLE_CONFLICTS.txt
  and ZERO_SLACK_FEAS.txt (n=200, seed 12345, eta_w=0.3, deployment planner).
  Inset per-episode infeasible-step counts: collector/rerun
  fig_data/infeasible_steps.npy if present; otherwise the inset is drawn as a
  single annotated mean bar and the manifest flags it as pending re-dump.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures_generated")
os.makedirs(OUT, exist_ok=True)
INFEAS = fs.find_data("fig_data/infeasible_steps.npy",
                      "baselines/figures_gen/fig_data/infeasible_steps.npy")

C_ROOT = "#0B3D66"      # deep blue
C_DIAG = "#7FB2DD"      # light blue
C_TERM = "#3A4A5A"      # dark grey-blue
C_SIDE = "#C9CCD1"      # pale grey (no-conflict branch)
MEAN_INFEAS = 17.77


def box(ax, x, y, w, h, text, fc, tc="white", fs_=8.5):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.012,rounding_size=0.02",
                       fc=fc, ec="none", zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=fs_,
            zorder=4, linespacing=1.2)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-", color="0.4", lw=1.1), zorder=2)


def main():
    fs.set_rc()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # nodes
    box(ax, 5.0, 9.1, 3.4, 0.9, "200 evaluated encounters", C_ROOT)
    box(ax, 2.1, 7.0, 2.4, 0.8, "178 no conflict", C_SIDE, tc="0.15")
    box(ax, 6.8, 7.0, 2.6, 0.8, "22 conflict episodes", C_ROOT)
    box(ax, 4.6, 4.6, 2.9, 1.0,
        "zero-slack re-solve\n22/22 hard-infeasible", C_DIAG, tc="0.12")
    box(ax, 8.4, 4.6, 2.7, 1.0,
        "zero-error replay\n0/22 resolved", C_DIAG, tc="0.12")
    box(ax, 6.5, 1.9, 4.6, 1.1,
        "unresolved within the tested\nplanning-and-control envelope",
        C_TERM)

    # edges
    arrow(ax, 4.4, 8.75, 2.6, 7.4)
    arrow(ax, 5.7, 8.75, 6.6, 7.4)
    arrow(ax, 6.3, 6.6, 5.1, 5.1)
    arrow(ax, 7.3, 6.6, 8.2, 5.1)
    arrow(ax, 5.0, 4.1, 6.0, 2.45)
    arrow(ax, 8.2, 4.1, 7.2, 2.45)

    # inset histogram, bottom-right
    iax = fig.add_axes([0.63, 0.14, 0.30, 0.26])
    if INFEAS and os.path.exists(INFEAS):
        cnt = np.load(INFEAS)
        iax.hist(cnt, bins=np.arange(0, 22) - 0.5, color=C_DIAG,
                 edgecolor="0.3", lw=0.5)
        iax.set_ylabel("Conflict episodes", fontsize=7)
        src = "per-episode array"
    else:
        # only the mean is archived -> draw a single annotated marker
        iax.bar([MEAN_INFEAS], [22], width=1.2, color=C_DIAG, edgecolor="0.3")
        iax.set_ylabel("Conflict episodes", fontsize=7)
        src = "mean only (array pending)"
    iax.axvline(MEAN_INFEAS, color=C_TERM, ls="--", lw=1.0)
    iax.annotate(f"mean {MEAN_INFEAS:.1f}/20", (MEAN_INFEAS, 1.0),
                 xycoords=("data", "axes fraction"), xytext=(0, -2),
                 textcoords="offset points", ha="center", va="top",
                 fontsize=7, color=C_TERM)
    iax.set_xlim(0, 20)
    iax.set_xlabel("Hard-infeasible steps per episode", fontsize=7)
    iax.tick_params(labelsize=6.5)
    print("inset source:", src)

    out = os.path.join(OUT, "fig08_attribution.pdf")
    fig.savefig(out); print("wrote", out)


if __name__ == "__main__":
    main()
