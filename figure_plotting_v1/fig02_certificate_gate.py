#!/usr/bin/env python3
"""Fig. 2 -- The certificate gate: what it holds back, and what safety costs.

Two-panel composite answering the two follow-up questions to Fig. 1:
  (a) ECDF of per-episode minimum separation -- WHERE the 11-12% vs 41-53%
      gap lives on the separation distribution (left tail crossing 30 m).
  (b) Effort vs CR scatter -- inverting the naive "low energy = better"
      reading: certificate-free arms sit low-effort/high-CR because they
      do not avoid; safety costs actuation.

Authoritative source: lab minsep_effort.npz (n=200, seed 12345, eta_w=0.3,
OSQP fast CBF path) + result.json main-table points. Soft-IPP weight
absent on Lab -> ECDF panel (a) omits Soft-IPP; panel (b) plots the
Soft-IPC point from the main table (CR 53, Effort 17.0).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

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

# ----------------------------------------------------------------------
# arm specs (color/marker/fill same convention as Fig. 1)
#   fill True  = CBF certificate present (solid line in ECDF)
#   fill False = certificate-free (dashed line in ECDF)
# ----------------------------------------------------------------------
# name_in_npz, display, color, marker, has_cert, group
ARMS = [
    ("Constant-Velocity", "Constant-Velocity", "#7A7A7A", "h", True,  "common"),
    ("Stage-1b",          "Stage-1b",           "#009E73", "D", True,  "common"),
    ("Stage2",            "Stage-2 (ours)",     "#0072B2", "o", True,  "common"),
    ("Fixed-Predictor",   "Fixed-Predictor",   "#E69F00", "^", True,  "common"),
    ("Conformal-MPC",     "Conformal-MPC",     "#56B4E9", "s", True,  "other"),
    ("Vanilla-MPC",       "Vanilla-MPC",        "#D55E00", "v", False, "other"),
    ("Soft-IPP",          "Soft-IPP",           "#CC79A7", "p", False, "other"),
]

# main-table points for panel (b) (effort_mean, effort_sd, CR%)
# authoritative: result.json + minsep_effort.npz effort arrays
TABLE = {
    "Constant-Velocity": dict(effort=52.87, sd=9.62, cr=12.0),
    "Stage-1b":          dict(effort=52.44, sd=9.80, cr=11.5),
    "Stage-2 (ours)":    dict(effort=52.35, sd=7.65, cr=11.0),
    "Fixed-Predictor":   dict(effort=51.47, sd=11.42, cr=12.5),
    "Conformal-MPC":     dict(effort=57.93, sd=10.68, cr=11.5),
    "Vanilla-MPC":       dict(effort=19.11, sd=9.71, cr=41.0),
    "Soft-IPP":          dict(effort=16.97, sd=8.91, cr=53.0),
}

D_SEP = 30.0
N = 200

here = os.path.dirname(os.path.abspath(__file__))
npz_path = os.path.join(here, "..", "code", "baselines",
                        "figures_gen", "fig_data", "minsep_effort.npz")
d = np.load(npz_path, allow_pickle=True)


def ecdf(x):
    xs = np.sort(np.asarray(x, float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


fig = plt.figure(figsize=(11.2, 5.0), constrained_layout=True)
gs = GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28, figure=fig)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

# ======================================================================
# Panel (a) -- ECDF of minimum separation
# ======================================================================
for (key, disp, col, mk, cert, group) in ARMS:
    npz_key = key + "__minsep"
    if npz_key not in d.files:
        continue  # Soft-IPP absent
    xs, ys = ecdf(d[npz_key])
    ls = "-" if cert else "--"
    lw = 1.7 if cert else 1.9
    sat = 1.0 if group == "common" else 0.85
    ax_a.plot(xs, ys, color=col, linestyle=ls, linewidth=lw, alpha=sat,
              label=disp, zorder=3)

# 30 m threshold
ax_a.axvline(D_SEP, color="#C0392B", linestyle=":", linewidth=1.2, zorder=2)
ax_a.text(D_SEP + 0.4, 0.04, "$d_{\\mathrm{sep}}$ = 30 m", color="#C0392B",
          fontsize=7.5, va="bottom", ha="left", zorder=4)

# CR readout at the threshold (annotate a couple to anchor the reading)
ax_a.annotate("", xy=(D_SEP, 0.11), xytext=(40, 0.11),
              arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6))
ax_a.text(40.5, 0.11, "CR = ECDF(30 m)", fontsize=7, color="#666666",
          va="center", ha="left")

# inset: 25-35 m zoom of certificate arms
axins = ax_a.inset_axes([0.55, 0.42, 0.40, 0.50])
for (key, disp, col, mk, cert, group) in ARMS:
    if not cert:
        continue
    npz_key = key + "__minsep"
    xs, ys = ecdf(d[npz_key])
    axins.plot(xs, ys, color=col, linestyle="-", linewidth=1.3,
               alpha=0.9, zorder=3)
axins.axvline(D_SEP, color="#C0392B", linestyle=":", linewidth=1.0, zorder=2)
axins.set_xlim(25, 35)
axins.set_ylim(0.0, 0.18)
axins.set_xlabel("min separation (m)", fontsize=7)
axins.set_ylabel("ECDF", fontsize=7)
axins.tick_params(labelsize=6.5)
axins.set_title("30 m neighbourhood", fontsize=7, color="#555555")
ax_a.indicate_inset_zoom(axins, edgecolor="#BBBBBB")

ax_a.set_xlabel("Minimum realised separation (m)")
ax_a.set_ylabel("Empirical CDF")
ax_a.set_xlim(5, 75)
ax_a.set_ylim(0, 1)
ax_a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_a.grid(linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.7)
ax_a.set_axisbelow(True)
ax_a.text(0.02, 0.98, "(a)", transform=ax_a.transAxes, fontsize=10,
          fontweight="bold", va="top", ha="left")

# ======================================================================
# Panel (b) -- Effort vs CR scatter
# ======================================================================
for (key, disp, col, mk, cert, group) in ARMS:
    t = TABLE[disp]
    face = col if cert else "white"
    sat = 1.0 if group == "common" else 0.85
    ax_b.errorbar(t["effort"], t["cr"],
                  xerr=t["sd"], fmt="none", ecolor=col, elinewidth=1.0,
                  capsize=3, capthick=1.0, alpha=0.6, zorder=2.5)
    ax_b.scatter([t["effort"]], [t["cr"]], s=90, marker=mk,
                  facecolors=face, edgecolors=col,
                  linewidths=(1.3 if cert else 1.7), zorder=4, alpha=sat)

# cluster annotations
ax_b.annotate("certificate-free:\nlow effort = no avoidance",
              xy=(17.5, 47), xytext=(28, 50), fontsize=7.5, color="#666666",
              ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))
ax_b.annotate("CBF-constrained:\nsafe operation costs effort",
              xy=(52, 11.75), xytext=(36, 24), fontsize=7.5, color="#666666",
              ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))

# direct labels (compact)
LAB_B = {
    "Constant-Velocity": (10, -14, "center", "top"),
    "Stage-1b":          (10, 8,   "left",   "center"),
    "Stage-2 (ours)":    (-10, 10, "right",  "center"),
    "Fixed-Predictor":   (10, -14, "left",   "top"),
    "Conformal-MPC":     (10, 10,  "left",   "center"),
    "Vanilla-MPC":       (-11, 6,  "right",  "center"),
    "Soft-IPP":          (11, 4,   "left",   "center"),
}
for (key, disp, col, mk, cert, group) in ARMS:
    t = TABLE[disp]
    dx, dy, ha, va = LAB_B[disp]
    tcol = "#222222" if group == "common" else "#666666"
    ax_b.annotate(disp, xy=(t["effort"], t["cr"]), xytext=(dx, dy),
                  textcoords="offset points", ha=ha, va=va, fontsize=7.5,
                  color=tcol, zorder=6)

ax_b.set_xlabel("Control effort  (mean $\\pm$ SD)")
ax_b.set_ylabel("Conflict rate, CR (%)")
ax_b.set_xlim(5, 70)
ax_b.set_ylim(0, 60)
ax_b.grid(linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.7)
ax_b.set_axisbelow(True)
ax_b.text(0.02, 0.98, "(b)", transform=ax_b.transAxes, fontsize=10,
          fontweight="bold", va="top", ha="left")

# shared encoding legend (bottom of panel b)
leg_handles = [
    Line2D([0], [0], color="#666666", linestyle="-", linewidth=1.7,
           label="CBF-constrained (solid)"),
    Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.7,
           label="Certificate-free (dashed)"),
]
ax_b.legend(handles=leg_handles, loc="upper left", frameon=True,
            framealpha=0.9, edgecolor="#CCCCCC", handletextpad=0.5)

out_dir = os.path.normpath(os.path.join(here, "..", "figures_v1"))
os.makedirs(out_dir, exist_ok=True)
pdf = os.path.join(out_dir, "fig02_certificate_gate.pdf")
png = os.path.join(out_dir, "fig02_certificate_gate.png")
fig.savefig(pdf)
fig.savefig(png, bbox_inches="tight", dpi=300)
print("saved:", pdf)
print("saved:", png)
