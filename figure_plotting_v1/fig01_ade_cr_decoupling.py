#!/usr/bin/env python3
"""Fig. 1 -- Decoupling hero: displacement accuracy does not predict safety.

Single-panel ADE-CR scatter for 7 experimental arms (n=200, seed 12345,
held-out GUAM encounters 2500-2999, tuned CBF-MPC planner: alpha=0.1,
Hp=15, a_max=20).

Cross argument:
  horizontal -- 4 common-planner arms span 25x ADE (0.83 -> 20.90 m) at
               CR 11.0-12.5%, Cochran Q=5.00 p=0.172 (no difference).
  vertical   -- Stage-2 vs Vanilla-MPC, same predictor (ADE=4.32 m),
               certificate removed: CR 11.0 -> 41.0%, +30.0 pp.

Authoritative source: lab BEST.txt / result.json / STATS.txt (n=200).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

# (label, ADE_m, CR%, conflicts k, color, marker, group, has_cert)
# group 'common' = same tuned planner, only predictor differs
# jitter applied below for the two arms sharing ADE=20.90
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
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - half) * 100.0, (center + half) * 100.0


# apply ~7% display jitter to the two ADE=20.90 arms (for visibility only)
ADE_JITTER = {"Fixed-Predictor": 0.93, "Conformal-MPC": 1.07}

fig, ax = plt.subplots(figsize=(7.6, 5.4))

# ---------- gray descriptive band: common-planner observed CR range ---
ax.axhspan(11.0, 12.5, xmin=0.0, xmax=1.0, color="#BBBBBB", alpha=0.16, zorder=0)

# ---------- vertical matched contrast Stage-2 -> Vanilla -------------
# bracket (two-end ticks) instead of arrow: it is a difference, not a flow
ax.plot([4.3212, 4.3212], [11.0, 41.0], linestyle=(0, (4, 3)),
        color="#888888", linewidth=1.0, zorder=1.6)
ax.plot([4.3212 - 0.18, 4.3212 + 0.18], [11.0, 11.0], color="#888888",
        linewidth=1.0, zorder=1.7)
ax.plot([4.3212 - 0.18, 4.3212 + 0.18], [41.0, 41.0], color="#888888",
        linewidth=1.0, zorder=1.7)
ax.text(4.55, 26.0, "Same predictor,\nCBF layer removed\n$\\Delta$CR = +30.0 pp",
        ha="left", va="center", fontsize=7.5, color="#555555", zorder=2)

# ---------- 25x ADE range bracket + common-planner tag (bottom) -------
y_bb = 5.5
ax.annotate("", xy=(20.8964, y_bb), xytext=(0.8279, y_bb),
            arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.1,
                            shrinkA=0, shrinkB=0), zorder=3)
ax.text(np.sqrt(0.8279 * 20.8964), 4.5,
        "25\u00d7 ADE range\ncommon planner, different predictors",
        ha="center", va="top", fontsize=8, color="#444444", zorder=3)

# ---------- plot arms ------------------------------------------------
for (label, ade, cr, k, col, mk, group, cert) in ARMS:
    common = (group == "common")
    sat = 1.0 if common else 0.42
    ade_p = ADE_JITTER.get(label, 1.0) * ade
    lo, hi = wilson_ci(k)
    ecol = col if common else "#9A9A9A"
    elw = 1.8 if common else 1.0
    ax.errorbar(ade_p, cr, yerr=[[cr - lo], [hi - cr]], fmt="none",
                ecolor=ecol, elinewidth=elw, capsize=4, capthick=elw,
                alpha=sat, zorder=2.5)
    face = col if cert else "white"
    ms = 95
    ax.scatter([ade_p], [cr], s=ms, marker=mk, facecolors=face,
               edgecolors=col, linewidths=(1.3 if cert else 1.7),
               zorder=4, alpha=sat)

# ---------- direct labels (uniform weight; no bolding) ----------------
LAB = {
    "Constant-Velocity": dict(xy=(0, 11),  ha="center", va="bottom", conn=False,
                              line2="training-free, most accurate"),
    "Stage-1b":          dict(xy=(12, 5),  ha="left",   va="center", conn=False),
    "Stage-2 (ours)":    dict(xy=(0, -20), ha="center", va="top",    conn=True),
    "Fixed-Predictor":   dict(xy=(-9, 8),  ha="right",  va="center", conn=True),
    "Conformal-MPC":     dict(xy=(-9, -12),ha="right",  va="center", conn=True),
    "Vanilla-MPC":       dict(xy=(13, 0),  ha="left",   va="center", conn=True),
    "Soft-IPP":          dict(xy=(13, 5),  ha="left",   va="center", conn=True),
}
for (label, ade, cr, k, col, mk, group, cert) in ARMS:
    common = (group == "common")
    sat = 1.0 if common else 0.42
    ade_p = ADE_JITTER.get(label, 1.0) * ade
    tcol = "#222222" if common else "#666666"
    cfg = LAB[label]
    ap = None
    if cfg["conn"]:
        ap = dict(arrowstyle="-", color="#9A9A9A", lw=0.5,
                  shrinkA=2, shrinkB=3)
    ax.annotate(label, xy=(ade_p, cr), xytext=cfg["xy"],
                textcoords="offset points", ha=cfg["ha"], va=cfg["va"],
                fontsize=8, color=tcol, arrowprops=ap, zorder=6)
    if cfg.get("line2"):
        ax.annotate(cfg["line2"], xy=(ade_p, cr),
                    xytext=(cfg["xy"][0], cfg["xy"][1] - 13),
                    textcoords="offset points", ha=cfg["ha"],
                    va="bottom", fontsize=6.8, color="#888888",
                    style="italic", zorder=6)

# ---------- Cochran corner box (right, lightweight) -------------------
ax.text(0.985, 0.965,
        "Common-planner comparison\nCochran\u2019s $Q$ = 5.00,  $df$ = 3\n"
        "$p$ = 0.172,  $n$ = 200\n4 common-planner arms",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#DDDDDD", lw=0.5),
        zorder=7)

# ---------- encoding legend (left, two rows only) --------------------
leg_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666",
           markeredgecolor="#666666", markersize=9,
           label="CBF-constrained planner"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="#666666", markersize=9, markeredgewidth=1.6,
           label="Certificate-free planner"),
]
ax.legend(handles=leg_handles, loc="upper left", frameon=True,
          framealpha=0.92, edgecolor="#CCCCCC", handletextpad=0.4)

# ---------- axes ------------------------------------------------------
ax.set_xscale("log")
ax.set_xlim(0.5, 35)
ax.set_ylim(0, 60)
ax.set_xticks([1, 2, 5, 10, 20])
ax.set_xticklabels([r"$1$", r"$2$", r"$5$", r"$10$", r"$20$"])
ax.set_xlabel("Average displacement error, ADE (m; log scale)")
ax.set_ylabel("Conflict rate, CR (%)")
ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#CCCCCC",
        alpha=0.7, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()

here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.normpath(os.path.join(here, "..", "figures_v1"))
os.makedirs(out_dir, exist_ok=True)
pdf = os.path.join(out_dir, "fig01_ade_cr_decoupling.pdf")
png = os.path.join(out_dir, "fig01_ade_cr_decoupling.png")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, bbox_inches="tight", dpi=300)
print("saved:", pdf)
print("saved:", png)
