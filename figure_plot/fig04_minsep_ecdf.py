#!/usr/bin/env python3
"""Figure 4 -- empirical distributions of minimum realized separation.

v2 hero visual: a RIDGELINE (joyplot). Each non-Oracle arm gets one density
ridge of its per-episode minimum-separation sample, stacked bottom->top in the
fixed legend order, sharing a common x axis. A single thin red dashed line at
x = 30 m cuts vertically through every ridge, so the reader sees at a glance
that the certificate arms' mass sits to the RIGHT of the standard while the
certificate-free Vanilla ridge piles up on/left of it. Ridge fill uses the
per-method Okabe-Ito colour; certificate-free arms additionally carry a dashed
outline to preserve the family encoding. A slim ECDF inset (top-right) keeps
the quantitative tail readout around 30 m.

No in-figure title/conclusion text; only the "30 m" tick label and the (a)/(b)
free panel labels. Fonts >= 8 pt, vector PDF.

DATA PROVENANCE
  Per-episode minimum-separation arrays for the available arms, produced on the
  Lab by figures_gen/collect_fig_data.py at n=200, seed 12345, eta_w=0.3,
  evtol encounters 2500-2999, deployment planner, OSQP fast CBF solver
  (identical QP to the differentiable layer; verified). Loaded from
  fig_data/minsep_effort.npz (keys "<arm>__minsep"). frac(<30 m) per arm
  reproduces the v9 main-table conflict rate (Stage 2 11.0 %, Stage-1b 11.5 %,
  Stage 1 12.5 %, CV 12.0 %, Conformal 11.5 %, Vanilla 40.5 %). Soft-IPP has no
  weight on disk (soft_joint.pt absent) so it is honestly omitted -- the ridge
  set is drawn only for arms with real data. Stale local
  baselines/figures_gen/data JSONs are NOT used.
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
                arms[ARMMAP[raw]] = np.asarray(d[k], float)

    names = fs.ordered(arms.keys())          # bottom(free) .. top(cert), see below
    # stack so that certificate arms sit ABOVE the free arms: reverse the fixed
    # legend order (Oracle-side first) into bottom->top drawing order.
    draw = list(reversed(names))
    samples = [arms[n] for n in draw]
    colors = [fs.STYLE[n]["color"] for n in draw]
    labels = draw

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ys, (gx0, gx1) = fs.ridgeline(ax, samples, labels, colors,
                                  gap=0.9, width=1.45, lw=1.0)

    # certificate-free arms get a dashed ridge outline to keep the family code
    for i, n in enumerate(draw):
        if fs.STYLE[n]["family"] == "free":
            y0 = ys[i]
            grid = np.linspace(gx0, gx1, 320)
            from figstyle import _kde
            dd = _kde(arms[n], grid)
            dd = dd / dd.max() * 1.45 if dd.max() > 0 else dd
            ax.plot(grid, y0 + dd, color=fs.STYLE[n]["color"], lw=1.3,
                    ls="--", zorder=40)

    # median tick per ridge (small white-filled dot on the baseline)
    for i, n in enumerate(draw):
        med = float(np.median(arms[n]))
        ax.plot([med], [ys[i]], marker="o", ms=4.2, mfc="white",
                mec="0.15", mew=1.0, zorder=60, clip_on=False)

    # 30 m standard: one vertical red dashed line through all ridges
    ax.axvline(fs.THRESH, **fs.THRESH_KW, ymin=0, ymax=1)
    ax.annotate("30 m", (fs.THRESH, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -2), textcoords="offset points", ha="left",
                va="top", fontsize=8, color="#D55E00")

    ax.set_xlim(5, 78)
    ax.set_xlabel("Minimum realized separation (m)")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    fs.panel_label(ax, "(a)", x=-0.01, y=1.02)

    # ---- ECDF inset (top-right), quantitative tail readout near 30 m --------
    iax = ax.inset_axes([0.60, 0.60, 0.37, 0.36])
    for name in INSET_ARMS:
        if name in arms:
            xs, yy = ecdf(arms[name])
            s = fs.STYLE[name]
            iax.plot(xs, yy, color=s["color"], ls=s["ls"], lw=1.2)
    if "Vanilla-MPC" in arms:
        xs, yy = ecdf(arms["Vanilla-MPC"])
        s = fs.STYLE["Vanilla-MPC"]
        iax.plot(xs, yy, color=s["color"], ls="--", lw=1.6)
    iax.axvline(fs.THRESH, **fs.THRESH_KW)
    iax.set_xlim(20, 55); iax.set_ylim(0, 0.6)
    iax.set_xlabel("sep. (m)", fontsize=7, labelpad=1)
    iax.set_ylabel("ECDF", fontsize=7, labelpad=1)
    iax.tick_params(labelsize=6.5)
    iax.grid(True, color="0.9", lw=0.5)
    fs.panel_label(iax, "(b)", x=-0.04, y=1.03)

    out = os.path.join(OUT, "fig04_minsep_ecdf.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, "| ridges:", labels)


if __name__ == "__main__":
    main()
