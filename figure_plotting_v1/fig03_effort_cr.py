#!/usr/bin/env python3
"""Fig. 3 -- Control effort and conflict rate across planning regimes.

Standalone scatter: mean control effort vs conflict rate (n=200, seed 12345).
Certificate-free planners attain lower effort together with higher CR,
indicating weaker avoidance rather than efficiency.

Data provenance (unified): effort means and CR computed from lab
minsep_effort.npz (OSQP fast path, same QP as differentiable layer) for
the 6 arms present; Soft-IPC uses the main-table point (CR 53.0,
Effort 17.0) -- weight absent on Lab, no npz array, no error bar.
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
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

D_SEP = 30.0
N = 200

# (npz_key, display, color, marker, has_cert, group)
ARMS = [
    ("Constant-Velocity", "Constant-Velocity", "#7A7A7A", "h", True,  "common"),
    ("Stage-1b",          "Stage-1b",           "#009E73", "D", True,  "common"),
    ("Stage2",            "Stage-2 (ours)",     "#0072B2", "o", True,  "common"),
    ("Fixed-Predictor",   "Fixed-Predictor",   "#E69F00", "^", True,  "common"),
    ("Conformal-MPC",     "Conformal-MPC",     "#56B4E9", "s", True,  "other"),
    ("Vanilla-MPC",       "Vanilla-MPC",        "#D55E00", "v", False, "other"),
]

# Soft-IPC main-table point (weight absent, no npz)
SOFT_IPP = dict(display="Soft-IPP", color="#CC79A7", marker="p",
                has_cert=False, group="other",
                effort=16.97, cr=53.0, k=106)


def wilson_ci(k, n=N, z=1.959963984540054):
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - half) * 100.0, (center + half) * 100.0


here = os.path.dirname(os.path.abspath(__file__))
npz_path = os.path.join(here, "..", "code", "baselines",
                        "figures_gen", "fig_data", "minsep_effort.npz")
d = np.load(npz_path, allow_pickle=True)

# compute effort mean and CR from npz for 6 arms
points = []
for (key, disp, col, mk, cert, group) in ARMS:
    eff_key = key + "__effort"
    sep_key = key + "__minsep"
    effort_mean = float(np.mean(d[eff_key]))
    minsep = d[sep_key]
    cr = float(np.mean(minsep < D_SEP) * 100.0)
    k = int(np.sum(minsep < D_SEP))
    lo, hi = wilson_ci(k)
    points.append(dict(disp=disp, col=col, mk=mk, cert=cert, group=group,
                       effort=effort_mean, cr=cr, lo=lo, hi=hi))

# add Soft-IPC (no CI -- point only from main table)
so_lo, so_hi = wilson_ci(SOFT_IPP["k"])
points.append(dict(disp=SOFT_IPP["display"], col=SOFT_IPP["color"],
                   mk=SOFT_IPP["marker"], cert=SOFT_IPP["has_cert"],
                   group=SOFT_IPP["group"], effort=SOFT_IPP["effort"],
                   cr=SOFT_IPP["cr"], lo=so_lo, hi=so_hi))

# print verification
for p in points:
    print(f"  {p['disp']:20s} effort={p['effort']:6.2f}  CR={p['cr']:5.1f}%  "
          f"CI=[{p['lo']:.1f},{p['hi']:.1f}]")

fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)

# -- plot points (others first, common on top) ------------------------
for order in ("other", "common"):
    for p in points:
        if p["group"] != order:
            continue
        common = (p["group"] == "common")
        sat = 1.0 if common else 0.85
        # vertical Wilson CI only (no horizontal bars)
        ax.errorbar(p["effort"], p["cr"],
                    yerr=[[p["cr"] - p["lo"]], [p["hi"] - p["cr"]]],
                    fmt="none", ecolor=p["col"], elinewidth=1.3,
                    capsize=4, capthick=1.3, alpha=sat, zorder=2.5)
        face = p["col"] if p["cert"] else "white"
        ms = 95
        ax.scatter([p["effort"]], [p["cr"]], s=ms, marker=p["mk"],
                   facecolors=face, edgecolors=p["col"],
                   linewidths=(1.3 if p["cert"] else 1.7),
                   zorder=4, alpha=sat)

# -- short phrase labels (replace long sentences) ---------------------
ax.text(24, 48, "certificate-free\nlower effort, higher CR",
        fontsize=7.5, color="#666666", ha="left", va="center",
        style="italic", zorder=3)
ax.text(42, 20, "CBF-constrained\nsafer, but costlier control",
        fontsize=7.5, color="#666666", ha="left", va="center",
        style="italic", zorder=3)

# -- direct labels -----------------------------------------------------
LAB = {
    "Constant-Velocity": (10, -16, "center", "top"),
    "Stage-1b":          (11, 6,   "left",   "center"),
    "Stage-2 (ours)":    (-11, 7,  "right",  "center"),
    "Fixed-Predictor":   (11, -14, "left",   "top"),
    "Conformal-MPC":     (11, 8,   "left",   "center"),
    "Vanilla-MPC":       (-11, 5,  "right",  "center"),
    "Soft-IPP":          (11, 4,   "left",   "center"),
}
for p in points:
    dx, dy, ha, va = LAB[p["disp"]]
    tcol = "#222222" if p["group"] == "common" else "#666666"
    ap = dict(arrowstyle="-", color="#AAAAAA", lw=0.4, shrinkA=2, shrinkB=3)
    ax.annotate(p["disp"], xy=(p["effort"], p["cr"]), xytext=(dx, dy),
                textcoords="offset points", ha=ha, va=va, fontsize=7.5,
                color=tcol, arrowprops=ap, zorder=6)

# -- small encoding legend (left) -------------------------------------
leg_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666",
           markeredgecolor="#666666", markersize=9,
           label="CBF-constrained planner"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="#666666", markersize=9, markeredgewidth=1.6,
           label="Certificate-free planner"),
]
ax.legend(handles=leg_handles, loc="upper left", frameon=True,
          framealpha=0.9, edgecolor="#CCCCCC", handletextpad=0.4)

# -- axes -------------------------------------------------------------
ax.set_xlabel("Mean control effort")
ax.set_ylabel("Conflict rate, CR (%)")
ax.set_xlim(5, 70)
ax.set_ylim(0, 60)
ax.grid(linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.6)
ax.set_axisbelow(True)

out_dir = os.path.normpath(os.path.join(here, "..", "figures_v1"))
os.makedirs(out_dir, exist_ok=True)
pdf = os.path.join(out_dir, "fig03_effort_cr.pdf")
png = os.path.join(out_dir, "fig03_effort_cr.png")
fig.savefig(pdf)
fig.savefig(png, dpi=300)
print("saved:", pdf)
print("saved:", png)
