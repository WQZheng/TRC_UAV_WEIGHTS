#!/usr/bin/env python3
"""Figure 5 -- control effort and closed-loop conflict rate.

Scatter of all seven non-Oracle arms: x = normalized control effort,
y = conflict rate (%). Certificate-free arms (Vanilla, Soft-IPP) sit at low
effort but high CR; certificate-equipped arms sit at higher effort but low CR.
No "safe band", no "low energy but unsafe" text, no conclusion sentence. CR
carries a Wilson 95% CI vertical bar; a horizontal effort SE bar is added only
if the collector supplies per-episode effort. Two faint grey arrows optionally
mark the certificate-free vs certificate-equipped clusters.

DATA PROVENANCE
  Effort (x) and CR (y) point estimates: v9 Table tab:main-comparison
    (lines 332-350) -- Effort 52.9/51.5/52.9/52.3/58.1/19.1/17.0 for
    CV/Fixed/Stage-1b/Stage2/Conformal/Vanilla/Soft; CR 12.0/12.5/11.5/11.0/
    11.5/41.0/53.0. n=200, seed 12345, eta_w=0.3.
  Wilson CI on CR recomputed here (matches STATS.txt for the common arms).
  Optional horizontal effort SE: collector fig_data/minsep_effort.npz
    "<arm>__effort" per-episode arrays -> SE = SD/sqrt(n); drawn if present.
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

# name -> (effort, CR%, conflicts)  from the v9 main table
ROW = {
    "Constant-Velocity": (52.9, 12.0, 24),
    "Stage 1":           (51.5, 12.5, 25),
    "Stage-1b":          (52.9, 11.5, 23),
    "Stage 2":           (52.3, 11.0, 22),
    "Conformal-MPC":     (58.1, 11.5, 23),
    "Vanilla-MPC":       (19.1, 41.0, 82),
    "Soft-IPP":          (17.0, 53.0, 106),
}
ARMMAP = {"Stage2": "Stage 2", "Fixed-Predictor": "Stage 1",
          "Constant-Velocity": "Constant-Velocity", "Stage-1b": "Stage-1b",
          "Conformal-MPC": "Conformal-MPC", "Vanilla-MPC": "Vanilla-MPC",
          "Soft-IPP": "Soft-IPP"}
N = 200


def effort_se():
    se = {}
    if NPZ and os.path.exists(NPZ):
        d = np.load(NPZ, allow_pickle=True)
        for k in d.files:
            if k.endswith("__effort"):
                raw = k[:-len("__effort")]
                if raw in ARMMAP:
                    arr = np.asarray(d[k], float)
                    se[ARMMAP[raw]] = arr.std(ddof=1) / np.sqrt(len(arr))
    return se


def main():
    fs.set_rc()
    se = effort_se()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name in fs.ordered(ROW.keys()):
        eff, cr, k = ROW[name]
        lo, hi = fs.wilson_ci(k, N)
        s = fs.STYLE[name]; kw = fs.marker_kw(name)
        ax.errorbar(eff, cr, yerr=[[cr - lo], [hi - cr]],
                    xerr=(1.96 * se[name] if name in se else None),
                    capsize=2.5, elinewidth=1.0, ls="none", color=s["color"],
                    zorder=3)
        ax.plot(eff, cr, marker=kw["marker"], ls="none", ms=8,
                color=s["color"], mfc=kw.get("mfc", s["color"]),
                mec=kw.get("mec", s["color"]), mew=kw.get("mew", 0.8),
                label=name, zorder=4)
    ax.set_xlabel("Normalized control effort")
    ax.set_ylabel("Conflict rate, CR (%)")
    ax.set_ylim(0, 65)
    ax.set_xlim(10, 65)
    fs.panel_label(ax, "")   # single panel, no letter
    # faint cluster arrows (encoding-first; minimal text)
    ax.annotate("certificate-free", (18, 47), (30, 58),
                arrowprops=dict(arrowstyle="->", color="0.6", lw=0.8),
                fontsize=7.5, color="0.5", ha="center")
    ax.annotate("certificate-equipped", (54, 12), (44, 25),
                arrowprops=dict(arrowstyle="->", color="0.6", lw=0.8),
                fontsize=7.5, color="0.5", ha="center")
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False,
              fontsize=8, handletextpad=0.4, labelspacing=0.35)
    out = os.path.join(OUT, "fig05_effort_cr.pdf")
    fig.savefig(out, bbox_inches="tight"); print("wrote", out,
                "| effort SE from collector:", bool(se))


if __name__ == "__main__":
    main()
