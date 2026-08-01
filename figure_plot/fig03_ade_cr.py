#!/usr/bin/env python3
"""Figure 3 -- ADE-CR decoupling and formal common-planner CR comparison.

The paper's signature figure. Two panels, full page width.
  (a) open-loop ADE (log x) vs closed-loop CR (%) scatter, all seven arms,
      vertical Wilson-95%-CI error bars on CR (ADE is a single held-out
      summary, so no horizontal bar). A faint outline rings the four
      common-planner arms (NO "safe band" text, per the plan).
  (b) common-planner CR point-range: four arms on the x-axis ordered by ADE
      (CV, Stage-1b, Stage-2, Stage-1), each a point + Wilson 95% CI, with the
      Cochran-Q and Holm-adjusted-p annotation top-right.

DATA PROVENANCE (all authoritative, no rerun needed)
  * CR %, ADE m  -> v9 manuscript Table tab:main-comparison
                    (05_experiments_results_v9.tex lines 332-350).
  * Wilson 95% CI -> baselines/STATS.txt (n=200, seed=12345, eta_w=0.3),
                    recomputed here with the identical Wilson formula so the
                    figure is self-contained; values match STATS.txt exactly.
  * Cochran Q / Holm-p -> STATS_COCHRAN (correct-arms) Q=5.000, df=3, p=0.172;
                    all Holm-adjusted pairwise McNemar p=1.00 (STATS.txt).
Conflict is per-episode min-sep < 30 m at n=200. eta_w = 0.3 (code WIND_ETA).

Run:  python3 fig03_ade_cr.py       (writes ../figures_generated/fig03_ade_cr.pdf)
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "figures_generated")
os.makedirs(OUT, exist_ok=True)

# ---- authoritative numbers (v9 Table tab:main-comparison) ------------------
# name : (ADE_m, CR_%, n_conflicts_for_Wilson)  with n = 200 throughout.
N = 200
ROW = {
    #                 ADE     CR%   conflicts (=round(CR/100*200))
    "Constant-Velocity": (0.83, 12.0, 24),
    "Stage-1b":          (1.84, 11.5, 23),
    "Stage 2":           (4.32, 11.0, 22),
    "Stage 1":           (20.90, 12.5, 25),   # "Fixed-Predictor" == Stage 1 predictor
    "Conformal-MPC":     (20.90, 11.5, 23),
    "Vanilla-MPC":       (4.32, 41.0, 82),
    "Soft-IPP":          (6.45, 53.0, 106),
}
# common-planner (formal RQ1) arms, ordered by ADE for panel (b)
COMMON = ["Constant-Velocity", "Stage-1b", "Stage 2", "Stage 1"]
COMMON_SHORT = {"Constant-Velocity": "CV", "Stage-1b": "Stage-1b",
                "Stage 2": "Stage-2", "Stage 1": "Stage-1"}


def main():
    fs.set_rc()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 3.1),
                                   gridspec_kw=dict(width_ratios=[1.55, 1.0],
                                                    wspace=0.32))

    # ---- panel (a): ADE vs CR scatter --------------------------------------
    for name in fs.ordered(ROW.keys()):
        ade, cr, k = ROW[name]
        lo, hi = fs.wilson_ci(k, N)
        kw = fs.marker_kw(name)
        axa.errorbar(ade, cr, yerr=[[cr - lo], [hi - cr]],
                     capsize=2.5, elinewidth=1.0, ls="none",
                     color=kw["color"], zorder=3)
        # common-planner arms larger + saturated; free/margin as encoded
        big = name in COMMON
        axa.plot(ade, cr, marker=kw["marker"], ls="none",
                 ms=8.5 if big else 6.5,
                 mfc=kw.get("mfc", kw["color"]),
                 mec=kw.get("mec", kw["color"]),
                 mew=kw.get("mew", 0.8), color=kw["color"],
                 label=name, zorder=4)

    # faint outline ringing the four common-planner points (no text/band)
    xs = [ROW[n][0] for n in COMMON]; ys = [ROW[n][1] for n in COMMON]
    cx, cy = np.mean(np.log10(xs)), np.mean(ys)
    from matplotlib.patches import Ellipse
    ell = Ellipse((10 ** cx, cy), width=10 ** (cx + 0.62) - 10 ** (cx - 0.62),
                  height=5.0, angle=0, fill=False, ls=(0, (4, 3)),
                  ec="0.45", lw=0.9, zorder=1)
    axa.add_patch(ell)

    axa.set_xscale("log")
    axa.set_xticks([1, 2, 5, 10, 20])
    axa.set_xticklabels(["1", "2", "5", "10", "20"])
    axa.set_xlim(0.6, 30)
    axa.set_ylim(0, 65)
    axa.set_xlabel("ADE (m, log scale)")
    axa.set_ylabel("Conflict rate, CR (%)")
    fs.panel_label(axa, "(a)")
    # legend outside, right of (a), single column, never over points
    axa.legend(loc="upper left", bbox_to_anchor=(1.005, 1.02),
               frameon=False, handletextpad=0.4, borderaxespad=0.0,
               labelspacing=0.35)

    # ---- panel (b): common-planner CR point-range --------------------------
    xpos = np.arange(len(COMMON))
    for i, name in enumerate(COMMON):
        _, cr, k = ROW[name]
        lo, hi = fs.wilson_ci(k, N)
        kw = fs.marker_kw(name)
        axb.errorbar(i, cr, yerr=[[cr - lo], [hi - cr]], capsize=3,
                     elinewidth=1.1, ls="none", color=kw["color"], zorder=3)
        axb.plot(i, cr, marker=kw["marker"], ls="none", ms=8,
                 mfc=kw.get("mfc", kw["color"]), mec=kw.get("mec", kw["color"]),
                 mew=kw.get("mew", 0.8), color=kw["color"], zorder=4)
        axb.annotate(f"{cr:.1f}%", (i, hi), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8)
    axb.set_xticks(xpos)
    axb.set_xticklabels([COMMON_SHORT[n] for n in COMMON])
    axb.set_ylim(5, 20)
    axb.set_ylabel("Conflict rate, CR (%)")
    axb.set_xlim(-0.5, len(COMMON) - 0.5)
    fs.panel_label(axb, "(b)")
    axb.text(0.97, 0.97, "Cochran Q: $p = 0.17$\nHolm-adj. pairwise $p = 1.00$",
             transform=axb.transAxes, ha="right", va="top", fontsize=8,
             linespacing=1.35)

    out = os.path.join(OUT, "fig03_ade_cr.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
