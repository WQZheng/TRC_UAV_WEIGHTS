#!/usr/bin/env python3
"""Figure 3 -- ADE-CR decoupling, formal common-planner CR, paired raster.

The paper's signature figure. THREE panels, full page width (reviewer-adjudicated
layout: keep the signature ADE-CR scatter, keep the four-arm formal comparison,
and add the paired-episode conflict raster that exposes the paired structure the
Cochran-Q / McNemar tests rely on).

  (a) open-loop ADE (log x) vs closed-loop CR (%) scatter, all seven arms,
      vertical Wilson-95%-CI bars on CR (ADE is a single held-out summary, so no
      horizontal bar). A faint hull rings the four common-planner arms. This is
      the signature relationship (open-loop accuracy does NOT predict
      closed-loop safety); it must not be replaced by a multi-metric raincloud.
  (b) common-planner CR point-range: four arms (CV, Stage-1b, Stage-2, Stage-1),
      each a point + Wilson 95% CI, with the Cochran-Q annotation. The Q test is
      scoped to these four common-planner arms ONLY (reviewer note): it is drawn
      in this panel, never on the eight-arm panel.
  (c) paired-episode conflict raster: 200 episodes x 4 common-planner arms, a
      cell is dark if that arm conflicted on that episode. Rows are sorted by
      conflict pattern (not episode id) so shared vs arm-specific conflicts are
      visible; per-pattern counts on the right. This is a statistical-structure
      panel supporting (b)'s paired tests -- not a second signature figure.

DATA PROVENANCE (all authoritative, no rerun needed)
  * CR %, ADE m  -> v9 Table tab:main-comparison (results_v9.tex 332-350).
  * Wilson 95% CI -> recomputed with the identical Wilson formula (matches
    baselines/STATS.txt; n=200, seed=12345, eta_w=0.3).
  * Cochran Q -> STATS_COCHRAN (four common-planner arms) Q=5.000, df=3,
    p=0.172; Holm-adjusted pairwise McNemar p=1.00 (STATS.txt).
  * Paired raster -> baselines/common/conflict_vectors_q2.npz, keys PlanGrad
    (=Stage 2), Stage-1b, Fixed-Predictor (=Stage 1), Constant-Velocity; each a
    200-vector of bool, per-arm mean reproduces the table CR (0.110/0.115/
    0.125/0.120). Conflict = per-episode min-sep < 30 m.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "figures_generated")
os.makedirs(OUT, exist_ok=True)
RASTER_NPZ = fs.find_data("baselines/common/conflict_vectors_q2.npz",
                          "code/baselines/common/conflict_vectors_q2.npz")

N = 200
ROW = {
    #                 ADE     CR%   conflicts (=round(CR/100*200))
    "Constant-Velocity": (0.83, 12.0, 24),
    "Stage-1b":          (1.84, 11.5, 23),
    "Stage 2":           (4.32, 11.0, 22),
    "Stage 1":           (20.90, 12.5, 25),
    "Conformal-MPC":     (20.90, 11.5, 23),
    "Vanilla-MPC":       (4.32, 41.0, 82),
    "Soft-IPP":          (6.45, 53.0, 106),
}
COMMON = ["Constant-Velocity", "Stage-1b", "Stage 2", "Stage 1"]
COMMON_SHORT = {"Constant-Velocity": "CV", "Stage-1b": "Stage-1b",
                "Stage 2": "Stage-2", "Stage 1": "Stage-1"}
# collector raster key -> canonical name
RKEY = {"Constant-Velocity": "Constant-Velocity", "Stage-1b": "Stage-1b",
        "Stage 2": "PlanGrad", "Stage 1": "Fixed-Predictor"}


def load_raster():
    """Return (M[200x4] bool, colnames) in COMMON order, or None if absent."""
    if not RASTER_NPZ or not os.path.exists(RASTER_NPZ):
        return None
    d = np.load(RASTER_NPZ, allow_pickle=True)
    cols = []
    for name in COMMON:
        k = RKEY[name]
        if k not in d.files:
            return None
        cols.append(np.asarray(d[k]).astype(int))
    return np.stack(cols, axis=1), COMMON


def main():
    fs.set_rc()
    fig = plt.figure(figsize=(7.6, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.95, 0.9], wspace=0.42)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[0, 2])

    # ---- panel (a): ADE vs CR scatter --------------------------------------
    for name in fs.ordered(ROW.keys()):
        ade, cr, k = ROW[name]
        lo, hi = fs.wilson_ci(k, N)
        kw = fs.marker_kw(name)
        axa.errorbar(ade, cr, yerr=[[cr - lo], [hi - cr]],
                     capsize=2.5, elinewidth=1.0, ls="none",
                     color=kw["color"], zorder=3)
        big = name in COMMON
        axa.plot(ade, cr, marker=kw["marker"], ls="none",
                 ms=8.5 if big else 6.5,
                 mfc=kw.get("mfc", kw["color"]), mec=kw.get("mec", kw["color"]),
                 mew=kw.get("mew", 0.8), color=kw["color"], label=name, zorder=4)

    xs = [ROW[n][0] for n in COMMON]; ys = [ROW[n][1] for n in COMMON]
    cx, cy = np.mean(np.log10(xs)), np.mean(ys)
    ell = Ellipse((10 ** cx, cy), width=10 ** (cx + 0.62) - 10 ** (cx - 0.62),
                  height=5.0, angle=0, fill=False, ls=(0, (4, 3)),
                  ec="0.45", lw=0.9, zorder=1)
    axa.add_patch(ell)
    axa.set_xscale("log")
    axa.set_xticks([1, 2, 5, 10, 20]); axa.set_xticklabels(["1", "2", "5", "10", "20"])
    axa.set_xlim(0.6, 30); axa.set_ylim(0, 65)
    axa.set_xlabel("ADE (m, log scale)")
    axa.set_ylabel("Conflict rate, CR (%)")
    fs.panel_label(axa, "(a)")
    axa.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False,
               handletextpad=0.35, borderaxespad=0.0, labelspacing=0.25,
               fontsize=7.2)

    # ---- panel (b): common-planner CR point-range (Cochran Q here only) ----
    for i, name in enumerate(COMMON):
        _, cr, k = ROW[name]
        lo, hi = fs.wilson_ci(k, N)
        kw = fs.marker_kw(name)
        fs.point_range(axb, i, cr, lo, hi, kw["color"], marker=kw["marker"],
                       ms=8, mfc=kw.get("mfc"), mec=kw.get("mec"),
                       mew=kw.get("mew", 0.8))
        axb.annotate(f"{cr:.1f}", (i, hi), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=7.5)
    axb.set_xticks(range(len(COMMON)))
    axb.set_xticklabels([COMMON_SHORT[n] for n in COMMON], rotation=20,
                        ha="right", fontsize=7.5)
    axb.set_ylim(5, 20); axb.set_xlim(-0.5, len(COMMON) - 0.5)
    axb.set_ylabel("Conflict rate, CR (%)")
    fs.panel_label(axb, "(b)")
    axb.text(0.96, 0.96, "Cochran $Q$: $p = 0.17$\n(4 common-planner arms)",
             transform=axb.transAxes, ha="right", va="top", fontsize=7.2,
             linespacing=1.3)

    # ---- panel (c): 200x4 paired conflict raster ---------------------------
    R = load_raster()
    if R is None:
        axc.text(0.5, 0.5, "conflict_vectors_q2.npz\nnot found", ha="center",
                 va="center", fontsize=8, color="0.4", transform=axc.transAxes)
        axc.axis("off")
    else:
        M, cols = R
        col_colors = [fs.STYLE[n]["color"] for n in cols]
        # sort rows by conflict pattern: episodes with more/earlier conflicts up.
        # key = (total conflicts, binary pattern) descending -> conflict-heavy top
        patt = M.dot(1 << np.arange(M.shape[1])[::-1])
        order = np.lexsort((patt, M.sum(1)))[::-1]
        fs.binary_raster(axc, M, col_colors, row_order=order)
        axc.set_xticks(np.arange(len(cols)) + 0.5)
        axc.set_xticklabels([COMMON_SHORT[n] for n in cols], rotation=20,
                            ha="right", fontsize=7.0)
        axc.set_yticks([0, M.shape[0]])
        axc.set_yticklabels(["200", "1"], fontsize=7)
        axc.set_ylabel("episode (sorted by pattern)", fontsize=8)
        for s in ("top", "right"):
            axc.spines[s].set_visible(False)
        axc.grid(False)
        # right-margin: how many episodes have >=1 conflict, and all-clear
        n_any = int((M.sum(1) > 0).sum())
        axc.annotate(f"{n_any}/200 with $\\geq$1 conflict",
                     (1.0, 1.015), xycoords="axes fraction", ha="right",
                     va="bottom", fontsize=7.0, color="0.25")
    fs.panel_label(axc, "(c)")

    out = os.path.join(OUT, "fig03_ade_cr.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, "| raster:", "yes" if R is not None else "MISSING")


if __name__ == "__main__":
    main()
