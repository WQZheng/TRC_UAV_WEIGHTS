#!/usr/bin/env python3
"""Fig. 2 -- Realized separation distributions.

Standalone ECDF of per-episode minimum separation (n=200, seed 12345).
CBF-constrained planners (solid) compress the dangerous left tail below
the 30 m standard; certificate-free planners (dashed) cross early.

Data: lab minsep_effort.npz (OSQP fast path, same QP as differentiable
layer). Soft-IPC weight absent on Lab -> 6 arms only; caption notes this.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# (npz_key, display, color, has_cert, group)
ARMS = [
    ("Constant-Velocity", "Constant-Velocity", "#7A7A7A", True,  "common"),
    ("Stage-1b",          "Stage-1b",           "#009E73", True,  "common"),
    ("Stage2",            "Stage-2 (ours)",     "#0072B2", True,  "common"),
    ("Fixed-Predictor",   "Fixed-Predictor",    "#E69F00", True,  "common"),
    ("Conformal-MPC",     "Conformal-MPC",      "#56B4E9", True,  "other"),
    ("Vanilla-MPC",       "Vanilla-MPC",        "#D55E00", False, "other"),
]

D_SEP = 30.0

here = os.path.dirname(os.path.abspath(__file__))
npz_path = os.path.join(here, "..", "code", "baselines",
                        "figures_gen", "fig_data", "minsep_effort.npz")
d = np.load(npz_path, allow_pickle=True)


def ecdf(x):
    xs = np.sort(np.asarray(x, float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)

# -- loss-of-separation region shading (x < 30 m) --------------------
ax.axvspan(0, D_SEP, color="#C0392B", alpha=0.06, zorder=0)
ax.text(18, 0.93, "loss-of-separation\nregion", fontsize=7.5, color="#B03A2E",
        ha="center", va="top", style="italic", zorder=1)

# -- main ECDF curves -------------------------------------------------
for (key, disp, col, cert, group) in ARMS:
    npz_key = key + "__minsep"
    if npz_key not in d.files:
        continue
    xs, ys = ecdf(d[npz_key])
    ls = "-" if cert else "--"
    lw = 2.1 if key == "Stage2" else 1.7
    sat = 1.0 if group == "common" else 0.85
    ax.plot(xs, ys, color=col, linestyle=ls, linewidth=lw, alpha=sat,
            zorder=3)

# -- 30 m standard line ------------------------------------------------
ax.axvline(D_SEP, color="#C0392B", linestyle=":", linewidth=1.3, zorder=2.5)
ax.text(D_SEP + 0.6, 0.04, "30 m standard", color="#C0392B",
        fontsize=7.5, va="bottom", ha="left", zorder=4)

# -- upper-right zoom inset (26-35 m, 0.06-0.18) ----------------------
axins = ax.inset_axes([0.58, 0.55, 0.38, 0.42])
for (key, disp, col, cert, group) in ARMS:
    npz_key = key + "__minsep"
    if npz_key not in d.files:
        continue
    xs, ys = ecdf(d[npz_key])
    lw = 1.5 if key == "Stage2" else 1.1
    axins.plot(xs, ys, color=col, linestyle=("-" if cert else "--"),
               linewidth=lw, alpha=0.9, zorder=3)
axins.axvline(D_SEP, color="#C0392B", linestyle=":", linewidth=1.0, zorder=2)
axins.set_xlim(26, 35)
axins.set_ylim(0.0, 0.20)
axins.set_xlabel("MinSep (m)", fontsize=7)
axins.set_ylabel("ECDF", fontsize=7)
axins.tick_params(labelsize=6.5)
axins.set_title("30 m neighbourhood", fontsize=7, color="#555555")
ax.indicate_inset_zoom(axins, edgecolor="#BBBBBB")

# -- axes -------------------------------------------------------------
ax.set_xlabel("Minimum realised separation (m)")
ax.set_ylabel("Empirical CDF")
ax.set_xlim(8, 75)
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.6)
ax.set_axisbelow(True)

# -- external legend (right side, two groups) -------------------------
leg_handles = [
    Line2D([0], [0], color="#7A7A7A", lw=1.7, ls="-", label="Constant-Velocity"),
    Line2D([0], [0], color="#009E73", lw=1.7, ls="-", label="Stage-1b"),
    Line2D([0], [0], color="#0072B2", lw=2.1, ls="-", label="Stage-2 (ours)"),
    Line2D([0], [0], color="#E69F00", lw=1.7, ls="-", label="Fixed-Predictor"),
    Line2D([0], [0], color="#56B4E9", lw=1.7, ls="-", label="Conformal-MPC"),
    Line2D([0], [0], color="#D55E00", lw=1.7, ls="--", label="Vanilla-MPC"),
    Line2D([0], [0], color="#CC79A7", lw=1.7, ls="--", label="Soft-IPP (n/a)"),
    Line2D([0], [0], color="k", lw=1.7, ls="-", label="solid: CBF-constrained"),
    Line2D([0], [0], color="k", lw=1.7, ls="--", label="dashed: certificate-free"),
]
ax.legend(handles=leg_handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
          frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
          handletextpad=0.5, labelspacing=0.5, borderpad=0.6)

out_dir = os.path.normpath(os.path.join(here, "..", "figures_v1"))
os.makedirs(out_dir, exist_ok=True)
pdf = os.path.join(out_dir, "fig02_minsep_ecdf.pdf")
png = os.path.join(out_dir, "fig02_minsep_ecdf.png")
fig.savefig(pdf)
fig.savefig(png, dpi=300)
print("saved:", pdf)
print("saved:", png)
