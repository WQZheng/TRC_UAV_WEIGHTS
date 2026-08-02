#!/usr/bin/env python3
"""Fig. 1 -- Decoupling hero: displacement accuracy does not predict safety.

Single-panel ADE-CR scatter for 7 experimental arms (n=200, seed 12345,
held-out GUAM encounters 2500-2999, tuned CBF-MPC planner: alpha=0.1,
Hp=15, a_max=20).

Two orthogonal, non-redundant encoding dimensions:
  saturation  high = 4 common-planner arms (the paired comparison);
              low  = 3 other arms (margin planner / certificate-free).
  marker fill filled = CBF certificate present; open = certificate-free.

Wilson 95% CI on the conflict rate for every arm. The four common-planner
arms are statistically indistinguishable (Cochran Q = 5.00, df = 3,
p = 0.172; pairwise McNemar, Holm-adjusted p = 1.00) across a 25x ADE
range -- the central decoupling claim.

Authoritative source: lab BEST.txt / result.json / STATS.txt (n=200).
No external data file is read; every number is hard-coded from the
verified lab products.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# publication rc (matches the v4 figure-style hierarchy)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ----------------------------------------------------------------------
# authoritative numbers (lab-verified, n=200, seed 12345)
# (label, ADE_m, CR%, conflicts k, color, marker, group, has_cert)
# group 'common' = same tuned planner, only predictor differs
# ----------------------------------------------------------------------
ARMS = [
    ("Constant-Velocity", 0.8279, 12.0, 24, "#7A7A7A", "h", "common", True),
    ("Stage-1b",           1.8400, 11.5, 23, "#009E73", "D", "common", True),
    ("Stage-2 (ours)",     4.3212, 11.0, 22, "#0072B2", "o", "common", True),
    ("Fixed-Predictor",   20.8964, 12.5, 25, "#E69F00", "^", "common", True),
    ("Conformal-MPC",     20.8964, 11.5, 23, "#56B4E9", "s", "other",  True),
    ("Vanilla-MPC",        4.3212, 41.0, 82, "#D55E00", "v", "other",  False),
    ("Soft-IPP",           6.4486, 53.0,106, "#CC79A7", "p", "other",  False),
]
N = 200


def wilson_ci(k, n=N, z=1.959963984540054):
    """Wilson score 95% interval for a binomial proportion."""
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - half) * 100.0, (center + half) * 100.0


# label offset in points: (dx, dy, ha, draw_connector)
OFFS = {
    "Constant-Velocity": (-9, -17, "right", True),
    "Stage-1b":          (12,  -2, "left",  False),
    "Stage-2 (ours)":    (-7, -19, "right", True),
    "Fixed-Predictor":   (10,   9, "left",  False),
    "Conformal-MPC":     (10, -15, "left",  False),
    "Vanilla-MPC":       (12,  -3, "left",  False),
    "Soft-IPP":          (12,   4, "left",  False),
}

fig, ax = plt.subplots(figsize=(7.4, 5.2))

# -- shaded CR band for the certificate cluster -----------------------
ax.axhspan(11.0, 12.5, color="#BBBBBB", alpha=0.20, zorder=0)
ax.text(0.56, 11.75, "CR 11.0\u201312.5%", color="#555555", fontsize=8,
        ha="left", va="center", zorder=1)

# -- plot: others first (lower zorder), common arms on top ------------
for order in ("other", "common"):
    for (label, ade, cr, k, col, mk, group, cert) in ARMS:
        if group != order:
            continue
        common = (group == "common")
        sat = 1.0 if common else 0.42
        lo, hi = wilson_ci(k)
        ecol = col if common else "#9A9A9A"
        elw = 1.8 if common else 1.0
        # Wilson CI error bar
        ax.errorbar(ade, cr, yerr=[[cr - lo], [hi - cr]], fmt="none",
                    ecolor=ecol, elinewidth=elw, capsize=4, capthick=elw,
                    alpha=sat, zorder=2.5)
        # marker
        face = col if cert else "white"
        ms = 175 if label == "Stage-2 (ours)" else (135 if common else 80)
        ax.scatter([ade], [cr], s=ms, marker=mk, facecolors=face,
                   edgecolors=col, linewidths=(1.3 if cert else 1.7),
                   zorder=4, alpha=sat)

# -- direct labels with thin connectors where offset is large ----------
for (label, ade, cr, k, col, mk, group, cert) in ARMS:
    dx, dy, ha, conn = OFFS[label]
    common = (group == "common")
    tcol = "#222222" if common else "#666666"
    weight = "bold" if label == "Stage-2 (ours)" else "normal"
    ap = dict(arrowstyle="-", color="#9A9A9A", lw=0.6,
              shrinkA=2, shrinkB=3) if conn else None
    ax.annotate(label, xy=(ade, cr), xytext=(dx, dy),
                textcoords="offset points", ha=ha, va="center",
                fontsize=8 if common else 7.5, color=tcol, weight=weight,
                arrowprops=ap, zorder=6)

# -- 25x ADE range bracket along the bottom ----------------------------
y_br = 5.3
ax.annotate("", xy=(20.8964, y_br), xytext=(0.8279, y_br),
            arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.1,
                            shrinkA=0, shrinkB=0), zorder=3)
ax.text(np.sqrt(0.8279 * 20.8964), 4.0, "25\u00d7 ADE range",
        ha="center", va="top", fontsize=8, color="#444444", zorder=3)

# -- Cochran corner box (right) ---------------------------------------
ax.text(0.985, 0.965, "Cochran Q = 5.00\n(df = 3,  p = 0.172)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", lw=0.7),
        zorder=7)

# -- certificate legend (left) ----------------------------------------
leg_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#777777",
           markeredgecolor="#777777", markersize=9, label="CBF certificate (filled)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="#777777", markersize=9, markeredgewidth=1.6,
           label="certificate-free (open)"),
]
ax.legend(handles=leg_handles, loc="upper left", frameon=True,
          framealpha=0.92, edgecolor="#CCCCCC", handletextpad=0.4)

# -- axes -------------------------------------------------------------
ax.set_xscale("log")
ax.set_xlim(0.5, 35)
ax.set_ylim(0, 60)
ax.set_xlabel("Displacement error  ADE  (m, log scale)")
ax.set_ylabel("Conflict rate  CR  (%)")
ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#CCCCCC",
        alpha=0.7, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()

# -- save -------------------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.normpath(os.path.join(here, "..", "figures_v1"))
os.makedirs(out_dir, exist_ok=True)
pdf = os.path.join(out_dir, "fig01_ade_cr_decoupling.pdf")
png = os.path.join(out_dir, "fig01_ade_cr_decoupling.png")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, bbox_inches="tight", dpi=300)
print("saved:", pdf)
print("saved:", png)
