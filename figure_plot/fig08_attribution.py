#!/usr/bin/env python3
"""Figure 8 -- residual-conflict attribution under the deployment planner.

Adjudicated hero visual: a 22x20 INFEASIBILITY RASTER (not a tree, not a
waterfall). Rows = the 22 residual-conflict episodes, columns = the 20 rollout
steps; a cell is dark when the zero-slack (eps=0) QP is hard-infeasible at that
step -- i.e. no admissible control exists even with the prediction held fixed.
The near-solid rows make "conflicts are actuation-limited, not
prediction-limited" self-evident: almost every conflict is infeasible for most
of its horizon (row mean 17.8/20).

  Centre : 22x20 raster, rows sorted by total infeasible steps (most-infeasible
           at the top). Dark = hard-infeasible, pale = feasible.
  Right   : per-episode row totals (horizontal bars, /20).
  Top     : per-step column totals (how many of the 22 conflicts are infeasible
           at that step).
  Left cards: the audit ledger as small number cards -- 200 encounters ->
           22 conflict -> 22/22 hard-infeasible, 0/22 resolved by zero-error
           replay. No branch lines, no red, no decomposition bar.

No in-figure title/conclusion text; fonts >= 8 pt; vector PDF; colour-blind-safe.

DATA PROVENANCE (authoritative)
  200 encounters; Stage 2 residual conflicts = 22; zero-slack re-solve 22/22
  hard infeasible (mean 17.77/20); zero-error replay 0/22 resolved. Sources:
  Round1/05_results/robustness/p0_referee/{P1_ORACLE_CONFLICTS,ZERO_SLACK_FEAS}
  .txt (n=200, seed 12345, eta_w=0.3, deployment planner). Per-step raster:
  fig_data/infeasibility_raster.npy (22x20 uint8, from
  plangrad_sim/zero_slack_feasibility.py; row sums verified mean 17.77).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures_generated")
os.makedirs(OUT, exist_ok=True)
RASTER = fs.find_data("fig_data/infeasibility_raster.npy",
                      "baselines/figures_gen/fig_data/infeasibility_raster.npy")

N_TOTAL, N_CONFLICT, N_INFEAS, N_RESOLVED, T = 200, 22, 22, 0, 20
MEAN_INFEAS = 17.77
C_DARK = "#0B3D66"     # hard-infeasible
C_PALE = "#E7ECF1"     # feasible
C_BAR = fs.STYLE["Stage 2"]["color"]
C_TEXT = "#3A4A5A"


def card(ax, x, y, w, h, big, small, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.004,rounding_size=0.02",
                 fc=fc, ec="none", transform=ax.transAxes, zorder=3,
                 clip_on=False))
    ax.text(x + w / 2, y + h * 0.62, big, transform=ax.transAxes, ha="center",
            va="center", fontsize=12, fontweight="bold", color="white", zorder=4)
    ax.text(x + w / 2, y + h * 0.24, small, transform=ax.transAxes, ha="center",
            va="center", fontsize=6.6, color="white", zorder=4)


def main():
    fs.set_rc()
    if not RASTER or not os.path.exists(RASTER):
        raise SystemExit("fig_data/infeasibility_raster.npy not found; run "
                         "plangrad_sim/zero_slack_feasibility.py on the Lab.")
    M = np.asarray(np.load(RASTER)).astype(int)
    assert M.shape == (N_CONFLICT, T), f"expected (22,20), got {M.shape}"
    order = np.argsort(M.sum(1))          # ascending; we draw top=most infeasible
    Mo = M[order]
    rowtot = Mo.sum(1)
    coltot = M.sum(0)

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.9, 3.0, 0.85],
                          height_ratios=[0.6, 3.0], wspace=0.06, hspace=0.06,
                          left=0.02, right=0.98, top=0.94, bottom=0.12)
    ax_cards = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_ras = fig.add_subplot(gs[1, 1])
    ax_right = fig.add_subplot(gs[1, 2])

    # ---- centre raster -----------------------------------------------------
    ax_ras.add_patch(Rectangle((0, 0), T, N_CONFLICT, fc=C_PALE, ec="none"))
    for r in range(N_CONFLICT):
        for c in np.where(Mo[r] == 1)[0]:
            ax_ras.add_patch(Rectangle((c, N_CONFLICT - 1 - r), 1, 1,
                             fc=C_DARK, ec="white", lw=0.25))
    ax_ras.set_xlim(0, T); ax_ras.set_ylim(0, N_CONFLICT)
    ax_ras.set_xticks([0.5, 4.5, 9.5, 14.5, 19.5])
    ax_ras.set_xticklabels([1, 5, 10, 15, 20], fontsize=7.5)
    ax_ras.set_yticks([]); ax_ras.set_xlabel("Rollout step (of 20)", fontsize=8.5)
    for s in ("left", "right", "top"):
        ax_ras.spines[s].set_visible(False)
    ax_ras.grid(False)
    fs.panel_label(ax_ras, "(a)", x=0.0, y=1.28)

    # ---- top: per-step column totals --------------------------------------
    ax_top.bar(np.arange(T) + 0.5, coltot, width=0.9, color=C_DARK, lw=0)
    ax_top.set_xlim(0, T); ax_top.set_ylim(0, N_CONFLICT)
    ax_top.set_xticks([]); ax_top.set_yticks([0, N_CONFLICT])
    ax_top.set_yticklabels(["0", "22"], fontsize=6.5)
    ax_top.set_ylabel("infeas.\nconflicts", fontsize=6.8, labelpad=1)
    for s in ("right", "top"):
        ax_top.spines[s].set_visible(False)
    ax_top.grid(False)

    # ---- right: per-episode row totals ------------------------------------
    ypos = np.arange(N_CONFLICT) + 0.5
    ax_right.barh(ypos, rowtot, height=0.82, color=C_BAR, lw=0)
    ax_right.axvline(T, color=C_TEXT, ls=":", lw=0.9)
    ax_right.axvline(MEAN_INFEAS, color=C_DARK, ls="--", lw=0.9)
    ax_right.set_ylim(0, N_CONFLICT); ax_right.set_xlim(0, T + 1)
    ax_right.set_yticks([]); ax_right.set_xticks([0, 10, 20])
    ax_right.tick_params(labelsize=6.5)
    ax_right.set_xlabel("steps\ninfeas. /20", fontsize=6.8)
    for s in ("left", "right", "top"):
        ax_right.spines[s].set_visible(False)
    ax_right.grid(axis="x", color="0.9", lw=0.5)
    ax_right.annotate(f"mean {MEAN_INFEAS:.1f}", (MEAN_INFEAS, N_CONFLICT),
                      xytext=(0, 2), textcoords="offset points", ha="center",
                      va="bottom", fontsize=6.5, color=C_DARK)

    # ---- left: audit ledger as number cards -------------------------------
    ax_cards.axis("off")
    fs.panel_label(ax_cards, "(b)", x=0.15, y=1.28)
    card(ax_cards, 0.02, 0.78, 0.96, 0.18, f"{N_TOTAL}", "encounters", C_TEXT)
    card(ax_cards, 0.02, 0.545, 0.96, 0.18, f"{N_CONFLICT}", "residual conflicts",
         C_BAR)
    card(ax_cards, 0.02, 0.31, 0.96, 0.18, f"{N_INFEAS}/{N_CONFLICT}",
         "hard-infeasible", C_DARK)
    card(ax_cards, 0.02, 0.075, 0.96, 0.18, f"{N_RESOLVED}/{N_CONFLICT}",
         "resolved by\nzero-error replay", "#8A9AA6")

    out = os.path.join(OUT, "fig08_attribution.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, "| raster", M.shape, "mean_infeas=%.2f" % M.sum(1).mean())


if __name__ == "__main__":
    main()
