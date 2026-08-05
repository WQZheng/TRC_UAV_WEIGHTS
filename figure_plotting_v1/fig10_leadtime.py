#!/usr/bin/env python3
"""Fig. 10 -- Conflict rate against available detection lead time.

Two stacked LINEAR panels sharing the abscissa.
  (a) full range 0-90%, which carries the main narrative: near-total conflict at
      one second, collapse by three, and four arms lying on top of each other
      from three seconds on.
  (b) magnified 0-14%, where the Stage-1 hump and the flat Stage-1b control are
      legible, with the mechanism inset.

Why not a log ordinate
----------------------
Fourteen of the plotted cells are exactly zero: Stage 2 and the oracle are at
0.0% for every horizon from three seconds on, and that zero IS the result. A log
ordinate cannot represent it. This is the case rule_5 reserves the veto for --
zero-reference semantics -- as opposed to Fig. 8, where the range was merely
large and the veto did not apply.

Hedging (imposed, not stylistic)
--------------------------------
The Stage-1 hump is labelled "consistent with predictor-induced spurious
avoidance". It is NOT called induced conflict: the experiment does not isolate
the trajectory-level mechanism. The sufficiency statement is scoped to Stage 2
and the oracle on the tested geometry, because Stage 1 is not zero at three
seconds and Stage-1b is not either at five to seven.

Mechanism inset
---------------
Two horizontal stacked bars, one per horizon where Stage-2 conflicts exist, with
the conflict count beside each: the proportions rest on 165 and 22 conflicts
respectively and are not equally trustworthy. From three seconds the attribution
is undefined because there are no conflicts left to attribute, which the inset
states in words rather than drawing an empty slot -- the disappearance of the
decomposition is the same fact as the transition. Bars are appropriate here
because this is a composition, not a trend.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs                           # noqa: E402

DATA = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
OUTD = os.environ.get("FIG_OUT_DIR",
                      "/data/lab/TRC_UAV_WEIGHTS/figures_v1")

# Colours come from the registry, never from literals here. The first draft of
# this figure hard-coded Stage-1 as #D55E00, which is Vanilla-MPC's vermillion,
# and gave the oracle #7A7A7A, which in this figure family is Stage-1's own grey.
# Both are now resolved by entity name: Stage-1 takes the prediction-space grey
# it already has in fig08/fig09, and the oracle takes #8E44AD, the colour it was
# given in fig05 panel (c) where the oracle-predictor replay first appears. Same
# entity, same colour, across figures.
ARMS = [("Stage1", "Stage 1 (context only)",
         fs.color("Stage-1", family="prediction"), "^", "--"),
        ("Stage-1b", "Stage-1b (matched control)", fs.color("Stage-1b"),
         "s", "-"),
        ("Stage2", "Stage 2 (PlanGrad)", fs.color("Stage-2"), "o", "-"),
        ("Oracle", "Oracle (true future)", fs.color("Oracle"), "D", ":")]

# The darker tone is the actuation-limited class here and in fig05's raster.
C_ACT = fs.C_ACT
C_PRED = fs.C_PRED
GREY = fs.GREY

TAB = {
    1.0:  {"Stage1": 86.0, "Stage-1b": 83.0, "Stage2": 82.5, "Oracle": 82.0},
    2.0:  {"Stage1": 13.0, "Stage-1b": 11.5, "Stage2": 11.0, "Oracle": 11.0},
    3.0:  {"Stage1":  1.5, "Stage-1b":  0.0, "Stage2":  0.0, "Oracle":  0.0},
    4.0:  {"Stage1":  7.0, "Stage-1b":  0.0, "Stage2":  0.0, "Oracle":  0.0},
    5.0:  {"Stage1": 10.0, "Stage-1b":  0.5, "Stage2":  0.0, "Oracle":  0.0},
    6.0:  {"Stage1": 12.0, "Stage-1b":  0.5, "Stage2":  0.0, "Oracle":  0.0},
    7.0:  {"Stage1":  8.5, "Stage-1b":  0.5, "Stage2":  0.0, "Oracle":  0.0},
    10.0: {"Stage1":  0.5, "Stage-1b":  0.0, "Stage2":  0.0, "Oracle":  0.0},
    20.0: {"Stage1":  0.0, "Stage-1b":  0.0, "Stage2":  0.0, "Oracle":  0.0},
}


def main():
    f = f"{DATA}/leadtime_v2.npz"
    if not os.path.exists(f):
        sys.exit(f"missing {f}; run export_leadtime_v2.py first")
    D = np.load(f, allow_pickle=True)
    h = D["horizons_s"]
    n_s2 = D["n_s2_conflicts"]
    act = D["act_limited"]
    prd = D["pred_limited"]

    # ---------------------------- self-checks ------------------------------
    errs = []
    for i, hh in enumerate(h):
        for tag, _l, _c, _m, _s in ARMS:
            got = D[f"cr__{tag}"][i]
            exp = TAB[float(hh)][tag]
            if abs(got - exp) > 0.051:
                errs.append(f"{hh:.0f}s {tag}: {got:.1f} != {exp:.1f}")
    i3 = int(np.where(h == 3.0)[0][0])
    for tag in ("Stage2", "Oracle"):
        if float(np.max(D[f"cr__{tag}"][i3:])) > 0.051:
            errs.append(f"{tag} not zero from 3 s on")
    if not (D["cr__Stage1"][int(np.where(h == 6.0)[0][0])]
            > D["cr__Stage1"][i3]):
        errs.append("Stage-1 hump absent")
    if float(np.max(D["cr__Stage-1b"][i3:])) > 1.01:
        errs.append("Stage-1b is not flat from 3 s on")
    for i, hh in enumerate(h):
        if act[i] + prd[i] != n_s2[i]:
            errs.append(f"{hh:.0f}s attribution does not sum to conflicts")
    if errs:
        raise AssertionError("fig10 self-check failed:\n  " + "\n  ".join(errs))
    print("fig10 self-check: all invariants hold")

    # One typography definition for the series; three earlier scripts disagreed
    # on the serif family, which is the drift this registry exists to stop.
    fs.set_rc()
    fig = plt.figure(figsize=(7.1, 5.0))
    # right is held back from 1.0 because the regime label and the hump note run
    # to the right-hand end of the axes; at 0.982 both were clipped by the canvas
    # edge, which the edge-ink check caught.
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.15], hspace=0.16,
                  left=0.093, right=0.947, top=0.955, bottom=0.098)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], sharex=axa)

    XMIN, XMAX = 0.4, 22.0
    for ax in (axa, axb):
        ax.set_xscale("log")
        ax.grid(color=fs.GRID, lw=0.55, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # Regime bands in neutral grey at low alpha. The first draft tinted the
    # 1-2 s band with #D55E00 and the >=3 s band with Stage-2's #0072B2, which
    # spent Vanilla-MPC's colour a second time and implied the last band somehow
    # belongs to Stage 2. A band marks an interval of the abscissa, which is a
    # property of the experiment and not of any arm, so it gets no arm colour.
    # Grey bands under grey Stage-1 lines are unambiguous at 4-7% alpha.
    for ax in (axa, axb):
        ax.axvspan(2.0, 3.0, color=GREY, alpha=0.070, lw=0, zorder=0)
        ax.axvspan(3.0, 22.0, color=GREY, alpha=0.040, lw=0, zorder=0)
    axa.axvspan(0.4, 2.0, color=GREY, alpha=0.110, lw=0, zorder=0)

    tr = axa.get_xaxis_transform()
    axa.text(0.88, 0.965, "manoeuvre\ncompetition", transform=tr, ha="center",
             va="top", fontsize=7.4, color=fs.INK)
    axa.text(2.45, 0.965, "transition", transform=tr, ha="center", va="top",
             fontsize=7.4, color=fs.INK)
    axa.text(11.5, 0.965,
             "sufficient for Stage 2 and oracle (tested geometry)",
             transform=tr, ha="center", va="top", fontsize=7.4,
             color=fs.INK)

    # ---- (a) full range ---------------------------------------------------
    for tag, lab, col, mk, ls in ARMS:
        axa.plot(h, D[f"cr__{tag}"], ls=ls, color=col, lw=1.5, marker=mk,
                 ms=4.4, mec="white", mew=0.7, label=lab, zorder=3)
    axa.set_ylim(-3.0, 92.0)
    axa.set_yticks([0, 20, 40, 60, 80])
    axa.set_ylabel("conflict rate (%)", fontsize=9)
    axa.tick_params(labelbottom=False, labelsize=8.4)
    axa.text(-0.072, 1.01, "(a)", transform=axa.transAxes, ha="left",
             va="bottom", fontsize=10, fontweight="bold")

    # ---- (b) magnified ----------------------------------------------------
    for tag, lab, col, mk, ls in ARMS:
        axb.plot(h, D[f"cr__{tag}"], ls=ls, color=col, lw=1.6, marker=mk,
                 ms=4.6, mec="white", mew=0.7, zorder=3)
    # Headroom to 17, not 14. The magnified panel only needs to reach 13.0 (the
    # two-second point) to show every value it is responsible for, but the two
    # notes need somewhere to sit: the only curve-free block at ylim 14 was
    # 159 px wide and the hump note measures 233 px, so no placement could avoid
    # running off the canvas. Raising the ceiling empties the whole band above
    # 13.3 for x > 2.05 s, which is wide enough. Data are untouched -- widening
    # a limit cannot clip (rule_7) -- and 17 is still far below (a)'s 92, so the
    # panel keeps its magnifying purpose.
    axb.set_ylim(-0.6, 17.0)
    axb.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
    axb.set_ylabel("conflict rate (%)", fontsize=9)
    axb.set_xlabel("available detection lead time (s)", fontsize=9)
    axb.set_xlim(XMIN, XMAX)
    axb.set_xticks([1, 2, 3, 4, 5, 6, 7, 10, 20])
    axb.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "10", "20"])
    axb.minorticks_off()
    axb.tick_params(labelsize=8.4)
    axb.text(-0.072, 1.01, "(b)", transform=axb.transAxes, ha="left",
             va="bottom", fontsize=10, fontweight="bold")

    # The legend lives in (b), not (a). Panel (a)'s free region -- right of three
    # seconds, above the floor traces -- is not tall enough for both the inset
    # and a four-entry legend: a legend anchored high enough to clear the
    # one-to-two-second descent grows downward into the inset, which is what the
    # overlay-versus-overlay check kept reporting. In (b) the descent from 86% to
    # 13% only enters the 14% window at x = 1.98 s, so the whole column left of
    # two seconds is empty over the panel's full height. The arms are drawn
    # identically in both panels, so one legend serves the figure.
    leg = axb.legend(handles=[
        Line2D([], [], color=c, ls=s, lw=1.6, marker=m, ms=4.6, mec="white",
               mew=0.7, label=l) for _t, l, c, m, s in ARMS],
        loc="upper left", bbox_to_anchor=(0.006, 0.795), frameon=False,
        fontsize=8.0, handlelength=2.3, labelspacing=0.30)

    # Both notes occupy the band above 13.3, which the raised ceiling leaves
    # empty for x > 2.05 s. Anchored at 2.15 s so each has the panel's full
    # remaining width instead of the 159 px stub on the right.
    # The hump note carries the arrow, whose curve reaches down to the 12%
    # marker; its window extent therefore spans far below the text itself, so the
    # zero note has to clear the arrow, not just the words. The hump note takes
    # the upper line and its arrow drops from the right-hand end.
    hump = axb.annotate(
        "Stage-1 hump: consistent with predictor-induced spurious avoidance",
        xy=(6.0, 12.2), xytext=(2.15, 16.8), fontsize=7.2, color=fs.INK,
        ha="left", va="top",
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color=fs.INK,
                        shrinkA=1.0, shrinkB=3.0,
                        connectionstyle="arc3,rad=-0.30"))
    # No arrow. A leader from this block down to the floor traces of Stage 2,
    # the oracle and Stage-1b would have to cross them to reach them, and the
    # annotation's window extent includes the arrow, so the occlusion check
    # counted it as a collision. The statement names its arms, which is enough.
    # Left column, under the legend, in axes fraction. The band above 13.3 is
    # taken by the hump note and its arrow, but the column left of two seconds is
    # empty over the panel's full height because the descent from 86% only enters
    # this window at 1.98 s. Wrapped to two lines to fit that column's width.
    zero = axb.text(
        0.008, 0.305, "Stage 2, oracle and Stage-1b at\nor below 0.5% "
        "from 3\u2009s onward", transform=axb.transAxes, fontsize=7.2,
        color=fs.INK, ha="left", va="top")

    # ---- mechanism inset: attribution of Stage-2 conflicts ---------------
    # Hosted in (a). With the legend moved to (b), the whole of (a) right of
    # three seconds and above the floor traces is free: three seconds is axes
    # fraction 0.503 on the log abscissa, and beyond it the highest curve is
    # 12.0% -> fraction 0.158.
    axi = axa.inset_axes([0.560, 0.315, 0.330, 0.400])
    rows = [i for i, v in enumerate(n_s2) if v > 0]
    ypos = np.arange(len(rows))[::-1]
    for y, i in zip(ypos, rows):
        tot = n_s2[i]
        fa = 100.0 * act[i] / tot
        fp = 100.0 * prd[i] / tot
        axi.barh([y], [fa], height=0.52, color=C_ACT, lw=0, zorder=3)
        axi.barh([y], [fp], left=[fa], height=0.52, color=C_PRED, lw=0,
                 zorder=3)
        axi.text(fa / 2.0, y, f"{fa:.0f}", ha="center", va="center",
                 fontsize=6.4, color="white", zorder=4)
        axi.text(101.5, y, f"$n={tot}$", ha="left", va="center", fontsize=6.4,
                 color=fs.INK)
    axi.set_yticks(ypos)
    axi.set_yticklabels([f"{h[i]:.0f}\u2009s" for i in rows], fontsize=6.8)
    axi.set_xlim(0, 128)
    axi.set_xticks([0, 50, 100])
    axi.set_xticklabels(["0", "50", "100"], fontsize=6.6)
    axi.set_xlabel("% of Stage-2 conflicts", fontsize=6.8, labelpad=1.5)
    axi.tick_params(length=2.0)
    axi.set_ylim(-1.15, len(rows) - 0.42)
    axi.text(0.0, -1.02, "no conflicts from 3\u2009s: attribution undefined",
             fontsize=6.3, color=fs.INK, ha="left", va="center")
    axi.text(0.0, len(rows) - 0.50,
             "actuation-limited (dark) vs prediction-limited",
             fontsize=6.3, color=fs.INK, ha="left", va="bottom")
    for s in ("top", "right"):
        axi.spines[s].set_visible(False)
    axi.set_facecolor(fs.PANEL_BG)

    # ---- geometric verification, all via the registry --------------------
    # These three checks used to be inlined per figure, which is how fig10 was
    # nearly shipped with a literal "\\%" on the canvas. They now live in
    # figstyle so every later figure inherits them unchanged.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    series = [D[f"cr__{tag}"] for tag, _l, _c, _m, _s in ARMS]
    sa = fs.curve_samples(axa, h, series)
    sb = fs.curve_samples(axb, h, series)
    fs.check_overlays([("legend (b)", leg.get_window_extent(r), sb),
                       ("inset in (a)", axi.get_tightbbox(r), sa),
                       ("hump note", hump.get_window_extent(r), sb),
                       ("zero note", zero.get_window_extent(r), sb)])
    fs.check_clipping(fig, (axa, axb), r)
    fs.check_escapes((axa, axb, axi), legends=(leg,))
    fs.assert_registered(*[c for _t, _l, c, _m, _s in ARMS],
                         C_ACT, C_PRED, GREY, fs.INK, fs.GRID, fs.PANEL_BG)
    print("colour check: every drawn colour is registered")

    for ext in ("pdf", "png"):
        p = f"{OUTD}/fig10_leadtime.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        print("saved:", p)

    print("\ncaption material:")
    print("  Stage-1 hump: " + " -> ".join(
        f"{D['cr__Stage1'][int(np.where(h==x)[0][0])]:.1f}"
        for x in (3.0, 4.0, 5.0, 6.0, 7.0)) + " over 3-7 s")
    print(f"  Stage-1b max from 3 s on: "
          f"{np.max(D['cr__Stage-1b'][i3:]):.1f}%")
    print(f"  attribution 1 s: {100.0*act[0]/n_s2[0]:.0f}% actuation-limited "
          f"of n={n_s2[0]}")
    print(f"  attribution 2 s: {100.0*act[1]/n_s2[1]:.0f}% actuation-limited "
          f"of n={n_s2[1]}")


if __name__ == "__main__":
    main()
