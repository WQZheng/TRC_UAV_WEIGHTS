#!/usr/bin/env python3
"""Figure 4 -- empirical distributions of minimum realized separation.

Adjudicated layout: ECDF stays the MAIN panel (it reads the 30 m threshold
exactly -- Pr(min-sep < 30 m) is each arm's conflict rate -- which a ridgeline
cannot do without bandwidth-dependent smoothing). A compact half-violin inset
adds the distribution-shape view without displacing the quantitative main plot.

  Main: ECDF of per-episode minimum separation for the available arms.
        Certificate arms = cool solid; certificate-free = warm dashed (thicker).
        One thin red dashed line at 30 m labelled only "30 m".
  Inset 1 (lower-right): 30 m neighbourhood zoom of the ECDF (25-35 m), the
        certificate arms + CV + Conformal only (Vanilla omitted to avoid
        occlusion). Reads the near-threshold cumulative fraction.
  Inset 2 (upper-left): stacked half-violins of the same samples -- the
        distribution-shape "hero" glance, kept small and secondary.

No in-figure title/conclusion text; panel labels (a)/(b)/(c) only; fonts >= 8 pt.

DATA PROVENANCE
  Per-episode minimum-separation arrays produced on the Lab by
  figures_gen/collect_fig_data.py (n=200, seed 12345, eta_w=0.3, evtol
  2500-2999, deployment planner, OSQP fast CBF solver). fig_data/minsep_effort
  .npz keys "<arm>__minsep". frac(<30 m) reproduces each arm's main-table CR.
  Soft-IPP omitted (soft_joint.pt absent). Stale local JSONs NOT used.
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

    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    # ---- MAIN: ECDF --------------------------------------------------------
    order = fs.ordered(arms.keys())
    pr30 = {}
    for name in order:
        xs, ys = ecdf(arms[name])
        s = fs.STYLE[name]
        warm = s["family"] == "free"
        ax.plot(xs, ys, color=s["color"], ls=s["ls"],
                lw=2.0 if warm else 1.35, label=name, zorder=3)
        pr30[name] = float(np.mean(arms[name] < fs.THRESH))
    ax.axvline(fs.THRESH, **fs.THRESH_KW)
    # place the 30 m tag lower on the line so it clears the upper-left legend
    ax.annotate("30 m", (fs.THRESH, 0.60), xytext=(3, 0),
                textcoords="offset points", fontsize=8, color="#D55E00")
    ax.set_xlim(5, 82); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Minimum realized separation (m)")
    ax.set_ylabel("Cumulative fraction of encounters")
    fs.panel_label(ax, "(a)", x=-0.01, y=1.02)
    # legend in the upper-left dead space (ECDF curves are still ~0 there),
    # so it no longer collides with the lower-right zoom inset or its title.
    ax.legend(loc="upper left", frameon=False, fontsize=7.6,
              labelspacing=0.3, handletextpad=0.5, borderaxespad=0.8,
              bbox_to_anchor=(0.015, 0.985))

    # right-edge summary strip: Pr(min-sep < 30 m) = each arm's conflict rate,
    # pinning the quantitative threshold conclusion directly onto the figure.
    ax.text(1.015, 0.99, r"Pr($<$30 m)", transform=ax.transAxes, fontsize=7,
            color="0.3", ha="left", va="top")
    for i, name in enumerate(order):
        s = fs.STYLE[name]
        ax.text(1.015, 0.90 - i * 0.052, f"{100*pr30[name]:4.1f}%",
                transform=ax.transAxes, fontsize=7.2, color=s["color"],
                ha="left", va="top", family="monospace")

    # ---- INSET: single 30 m neighbourhood zoom (lower-right, clear area) ---
    iax = ax.inset_axes([0.60, 0.10, 0.34, 0.40])
    for name in INSET_ARMS:
        if name in arms:
            xs, ys = ecdf(arms[name])
            s = fs.STYLE[name]
            iax.plot(xs, ys, color=s["color"], ls=s["ls"], lw=1.3)
    if "Vanilla-MPC" in arms:
        xs, ys = ecdf(arms["Vanilla-MPC"])
        iax.plot(xs, ys, color=fs.STYLE["Vanilla-MPC"]["color"], ls="--", lw=1.7)
    iax.axvline(fs.THRESH, **fs.THRESH_KW)
    iax.set_xlim(24, 40); iax.set_ylim(0, 0.5)
    iax.set_title("30 m zoom", fontsize=7, pad=2)
    iax.tick_params(labelsize=6.5)
    iax.grid(True, color="0.9", lw=0.5)
    fs.panel_label(iax, "(b)", x=-0.05, y=1.06)

    out = os.path.join(OUT, "fig04_minsep_ecdf.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, "| Pr(<30m):",
          {k: round(100 * v, 1) for k, v in pr30.items()})


if __name__ == "__main__":
    main()
