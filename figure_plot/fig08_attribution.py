#!/usr/bin/env python3
"""Figure 8 -- residual-conflict attribution under the deployment planner.

v2: a DATA figure, not a logic tree. Two stacked panels sharing the story
"the 22 residual conflicts are actuation-limited, not prediction-limited":

(a) A horizontal DECOMPOSITION (waterfall) of the 200 evaluated encounters.
    The full bar is split by area: 178 no-conflict (pale grey) | 22 conflict
    (blue). The 22-conflict segment is then decomposed downward into the two
    diagnostic outcomes as proportional segments: zero-slack re-solve
    22/22 hard-infeasible (dark blue) and zero-error replay 0/22 resolved
    (hatched outline, zero width -> shown as a labelled zero tick). Segment
    lengths encode counts; no branch lines, no red.

(b) The CORE EVIDENCE as a raincloud: for each of the 22 conflict episodes,
    the number of planning steps (out of 20) at which the eps=0 QP is hard
    infeasible. Half-violin "cloud" + jittered raw "rain" (all 22 points) +
    median/IQR. The mass piles up near 20 (mean 17.77/20), i.e. conflicts are
    infeasible for most of their horizon -> actuation-limited. A thin vertical
    guide marks the full-horizon value (20).

No in-figure title/conclusion text; only (a)/(b) panel labels and axis labels.
Fonts >= 8 pt, vector PDF, colour-blind-safe.

DATA PROVENANCE (authoritative)
  200 encounters; Stage 2 residual conflicts = 22; zero-slack re-solve 22/22
  hard infeasible (mean 17.77/20); zero-error replay 0/22 resolved.
  Source: Round1/05_results/robustness/p0_referee/P1_ORACLE_CONFLICTS.txt and
  ZERO_SLACK_FEAS.txt (n=200, seed 12345, eta_w=0.3, deployment planner).
  Per-episode infeasible-step counts: fig_data/infeasible_steps.npy (22 values,
  produced by plangrad_sim/zero_slack_feasibility.py; verified mean 17.77).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures_generated")
os.makedirs(OUT, exist_ok=True)
INFEAS = fs.find_data("fig_data/infeasible_steps.npy",
                      "baselines/figures_gen/fig_data/infeasible_steps.npy")

N_TOTAL = 200
N_CONFLICT = 22
N_INFEAS = 22          # zero-slack re-solve: hard-infeasible
N_RESOLVED = 0         # zero-error replay: resolved
T_EPISODE = 20
MEAN_INFEAS = 17.77

C_NOCONF = "#C9CCD1"   # pale grey  (no conflict)
C_CONF = fs.STYLE["Stage 2"]["color"]   # Okabe-Ito blue
C_INFEAS = "#0B3D66"   # dark blue  (hard-infeasible outcome)
C_TERM = "#3A4A5A"     # dark grey-blue (guides / median)


def main():
    fs.set_rc()
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(6.6, 4.8),
        gridspec_kw=dict(height_ratios=[1.0, 1.35], hspace=0.55))

    # ---------------- (a) decomposition / waterfall ----------------
    # top row: the 200 encounters split into no-conflict | conflict
    y_top = 1.0
    axA.add_patch(Rectangle((0, y_top - 0.3), N_TOTAL - N_CONFLICT, 0.6,
                            fc=C_NOCONF, ec="white", lw=1.0, zorder=3))
    axA.add_patch(Rectangle((N_TOTAL - N_CONFLICT, y_top - 0.3), N_CONFLICT,
                            0.6, fc=C_CONF, ec="white", lw=1.0, zorder=3))
    axA.text((N_TOTAL - N_CONFLICT) / 2, y_top, f"{N_TOTAL - N_CONFLICT} no conflict",
             ha="center", va="center", color="0.15", fontsize=8.5, zorder=5)
    axA.text(N_TOTAL - N_CONFLICT / 2, y_top + 0.55, f"{N_CONFLICT} conflict",
             ha="center", va="bottom", color=C_CONF, fontsize=8.5,
             fontweight="bold", zorder=5)

    # connector: the 22-conflict segment (top) expands to the full width (bottom)
    axA.plot([N_TOTAL - N_CONFLICT, 0], [y_top - 0.3, 0.39],
             color=C_CONF, lw=0.8, alpha=0.5, zorder=2)
    axA.plot([N_TOTAL, N_TOTAL], [y_top - 0.3, 0.39],
             color=C_CONF, lw=0.8, alpha=0.5, zorder=2)

    # bottom row: decompose the 22 conflicts into the two diagnostic outcomes,
    # rescaled to the full axis width so the split is legible.
    scale = N_TOTAL / N_CONFLICT
    y_bot = 0.05
    axA.add_patch(Rectangle((0, y_bot - 0.3), N_INFEAS * scale, 0.6,
                            fc=C_INFEAS, ec="white", lw=1.0, zorder=3))
    axA.text(N_INFEAS * scale / 2, y_bot,
             f"zero-slack re-solve: {N_INFEAS}/{N_CONFLICT} hard-infeasible",
             ha="center", va="center", color="white", fontsize=8.5, zorder=5)
    # zero-error-replay resolved = 0 -> zero-width, shown as a labelled tick
    axA.plot([N_INFEAS * scale, N_INFEAS * scale], [y_bot - 0.34, y_bot + 0.34],
             color=C_TERM, lw=1.4, zorder=6)
    axA.annotate(f"zero-error replay: {N_RESOLVED}/{N_CONFLICT} resolved",
                 (N_INFEAS * scale, y_bot + 0.4), ha="right", va="bottom",
                 fontsize=8, color=C_TERM, zorder=6)

    axA.set_xlim(-2, N_TOTAL + 2)
    axA.set_ylim(-0.5, 1.9)
    axA.set_yticks([y_bot, y_top])
    axA.set_yticklabels(["of the 22\nconflicts", "of 200\nencounters"],
                        fontsize=7.5)
    axA.set_xlabel("Encounters (top) / conflicts rescaled to full width (bottom)",
                   fontsize=8)
    for s in ("left", "right", "top"):
        axA.spines[s].set_visible(False)
    axA.grid(False)
    fs.panel_label(axA, "(a)", x=-0.02, y=1.06)

    # ---------------- (b) raincloud of hard-infeasible steps ----------------
    if not INFEAS or not os.path.exists(INFEAS):
        raise SystemExit("fig_data/infeasible_steps.npy not found; run "
                         "plangrad_sim/zero_slack_feasibility.py on the Lab.")
    cnt = np.asarray(np.load(INFEAS), float)
    assert cnt.size == N_CONFLICT, f"expected {N_CONFLICT} counts, got {cnt.size}"

    fs.raincloud(axB, cnt, y0=0.0, color=C_CONF, width=0.55, jitter=0.16,
                 point_ms=3.2, seed=12345)
    axB.axvline(T_EPISODE, color=C_TERM, ls=":", lw=1.1, zorder=2)
    axB.annotate(f"full horizon = {T_EPISODE}", (T_EPISODE, 0.62),
                 ha="right", va="top", fontsize=7.5, color=C_TERM,
                 xytext=(-3, 0), textcoords="offset points")
    axB.axvline(MEAN_INFEAS, color=C_INFEAS, ls="--", lw=1.0, zorder=2)
    axB.annotate(f"mean {MEAN_INFEAS:.1f}", (MEAN_INFEAS, -0.32),
                 ha="center", va="top", fontsize=7.5, color=C_INFEAS)
    axB.set_xlim(0, T_EPISODE + 1.2)
    axB.set_ylim(-0.45, 0.75)
    axB.set_yticks([])
    axB.set_xlabel("Hard-infeasible planning steps per conflict episode "
                   f"(out of {T_EPISODE})")
    for s in ("left", "right", "top"):
        axB.spines[s].set_visible(False)
    axB.grid(axis="x", color="0.9", lw=0.5)
    fs.panel_label(axB, "(b)", x=-0.02, y=1.02)

    out = os.path.join(OUT, "fig08_attribution.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, "| n_conflict=", int(cnt.size),
          "mean_infeas=%.2f" % cnt.mean())


if __name__ == "__main__":
    main()
