#!/usr/bin/env python3
"""Figure 11 -- conflict rate versus available detection lead time.

Line plot with a broken x-axis: the left panel covers 1-7 s (~75% width), the
right panel covers 10-20 s (~25% width), with standard axis-break marks. Four
curves: Stage 1, Stage-1b, Stage 2, Oracle (zero-error reference, black). A
very pale vertical grey band marks 2-3 s (labelled "2-3 s", no "critical
threshold" text). Stage-1b was not evaluated at 7/10/20 s, so its curve stops
at 6 s -- no fabricated connection, no dash points.

DATA PROVENANCE (authoritative)
  v9 Table tab:leadtime (lines 937-945), n=200, seed 12345, deployment
  planner. Dashes mark horizons not evaluated for Stage-1b. Values copied
  verbatim from the manuscript table (the full-sweep raw source file is not in
  the Lab archive, so the authoritative manuscript table is used directly, per
  the agreed plan).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "figures_generated")
os.makedirs(OUT, exist_ok=True)

# tab:leadtime  (lead-time s -> CR%).  None = not evaluated (Stage-1b).
LEAD = [1, 2, 3, 4, 5, 6, 7, 10, 20]
CR = {
    "Stage 1":  [86.0, 13.0, 1.5, 7.0, 10.0, 12.0, 8.5, 0.5, 0.0],
    "Stage-1b": [83.0, 11.5, 0.0, 0.0, 0.0, 0.0, None, None, None],
    "Stage 2":  [82.5, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Oracle":   [82.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
ORDER = ["Stage 2", "Stage-1b", "Stage 1", "Oracle"]


def _plot(ax, xs, ys, name):
    kw = fs.marker_kw(name)
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    ax.plot(xs, ys, marker=kw["marker"], ls=kw["ls"], color=kw["color"],
            ms=5, mfc=kw.get("mfc", kw["color"]), mec=kw.get("mec", kw["color"]),
            label=name)


def main():
    fs.set_rc()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True,
                                   gridspec_kw=dict(width_ratios=[3, 1],
                                                    wspace=0.06))
    for name in ORDER:
        pts = [(t, c) for t, c in zip(LEAD, CR[name]) if c is not None]
        xs = [t for t, _ in pts]; ys = [c for _, c in pts]
        # split at 7 s boundary
        xl = [x for x in xs if x <= 7]; yl = [c for x, c in zip(xs, ys) if x <= 7]
        xr = [x for x in xs if x >= 10]; yr = [c for x, c in zip(xs, ys) if x >= 10]
        _plot(axL, xl, yl, name)
        if xr:
            _plot(axR, xr, yr, name)

    # 2-3 s band on left
    axL.axvspan(2, 3, color="0.85", alpha=0.6, zorder=0)
    axL.text(2.5, 0.98, "2-3 s", transform=axL.get_xaxis_transform(),
             ha="center", va="top", fontsize=8, color="0.4")

    axL.set_xlim(0.5, 7.5); axR.set_xlim(9, 21)
    axL.set_ylim(0, 90)
    axL.set_xticks([1, 2, 3, 4, 5, 6, 7])
    axR.set_xticks([10, 20])
    axL.set_ylabel("Conflict rate, CR (%)")
    fig.supxlabel("Available detection lead time (s)", fontsize=10, y=0.02)

    # broken-axis marks
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.tick_params(left=False)
    dd = 0.012
    kwargs = dict(transform=axL.transAxes, color="k", clip_on=False, lw=0.9)
    axL.plot([1 - dd, 1 + dd], [-dd, +dd], **kwargs)
    axL.plot([1 - dd, 1 + dd], [1 - dd, 1 + dd], **kwargs)
    kwargs.update(transform=axR.transAxes)
    axR.plot([-dd * 3, +dd * 3], [-dd, +dd], **kwargs)
    axR.plot([-dd * 3, +dd * 3], [1 - dd, 1 + dd], **kwargs)

    # legend top-centre, four columns
    handles = [plt.Line2D([], [], color=fs.STYLE[n]["color"],
                          marker=fs.STYLE[n]["marker"], ls=fs.STYLE[n]["ls"],
                          ms=5, label=n) for n in ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=8.5, columnspacing=1.4,
               handletextpad=0.4)
    fig.subplots_adjust(top=0.88, bottom=0.16)
    out = os.path.join(OUT, "fig11_leadtime.pdf")
    fig.savefig(out, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
