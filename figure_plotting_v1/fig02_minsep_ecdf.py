#!/usr/bin/env python3
"""Fig. 2 -- Realized separation distributions (standalone, revised layout).

ECDF of per-episode minimum separation (n=200, seed 12345), six arms.
CBF-constrained planners reduce the fraction of episodes crossing the
30 m standard (11.0-12.5% vs 41.0%); residual crossings extend as deep
as ~11 m, consistent with the hard-infeasible attribution (Sec. 4.4).
NOTE the wording: the certificate compresses the CROSSING FRACTION,
not the depth of the residual tail. Measured on this data, the residual
depth is essentially identical across arms (min 10.89-11.07 m for the
certificate arms vs 10.06 m for Vanilla-MPC, k<15 m = 1 episode for
every arm), and CONDITIONAL on crossing the certificate arms are in
fact slightly deeper (mean 22.6-23.2 m vs 26.5 m for Vanilla-MPC).
So "compresses the left tail" would be the wrong claim.

Data: minsep_effort_v2.npz -- per-episode arrays emitted directly by
baselines/common/eval_common.py::evaluate_policy, i.e. the SAME rollout
that produces the main table. See fig_data/PROVENANCE_v2.json.

PROVENANCE (resolved -- previous WARNING removed):
  The earlier npz reported Vanilla-MPC at 81/200 (40.5%) while Table 1
  reported 82/200 (41.0%). Cause: figures_gen/collect_fig_data.py did
  not call evaluate_policy but re-implemented its rollout, and solved
  the certificate arms with the OSQP fast path on CPU. VanillaMPCLayer
  is a cvxpylayers interior-point QP, so its finite tolerance biased
  every arm slightly low (minSep ~1e-3 relative, effort ~7e-3, always
  negative). Vanilla-MPC is the only arm whose minSep distribution is
  dense at the threshold (27 episodes within +-0.5 m of 30 m, ECDF
  slope 15.3 pp/m, vs 0-4 episodes and <1.3 pp/m for the certificate
  arms), so it was the only arm whose count could move: one episode at
  30.0252 m, 2.5 cm above the threshold. Re-running every arm through
  evaluate_policy on the differentiable path restores 82/200 = 41.0%
  and reproduces all seven arms' table values exactly. evaluate_policy
  now returns per-episode arrays and asserts
      CR_% == 100*mean(minsep_per_ep < d_sep)
  so the table and this figure can no longer diverge.

Layout notes (kept deliberate, do not "simplify" back):
  * inset lives lower-right (x>52 m, ECDF<0.46): no curve enters there;
  * no zoom connector lines (they crossed the data); the dashed source
    rectangle + matching inset frame carry the association;
  * "30 m standard" is rotated, left of the line, clear of Vanilla;
  * legend inside upper-left void; Stage-2 drawn last (top of cluster).
  * curves are drawn with step(where="post"), NOT plot(): a piecewise
    linear interpolation between order statistics makes the visual
    height at 30 m disagree with the true step-ECDF value (Stage-2 read
    11.48% instead of 11.00%), which would contradict the inset title.

Soft-IPP is available in minsep_effort_v2.npz (106/200 = 53.0%) but is
deliberately NOT drawn here: its ECDF reaches 0.53 while still left of
30 m, which runs through the upper-left legend void. Adding it requires
relocating the legend first.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- paths
# Repo-relative by default (this file lives in figure_plotting_v1/), so the
# script runs unchanged on the lab box and on a local checkout. Override with
# FIG_DATA_DIR / FIG_OUT_DIR when the data sits elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
DATA_DIR = os.environ.get(
    "FIG_DATA_DIR",
    os.path.join(_REPO, "code", "baselines", "figures_gen", "fig_data"))
OUT_DIR = os.environ.get("FIG_OUT_DIR", os.path.join(_REPO, "figures_v1"))

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

# (npz_key, display, color, has_cert)  -- plotting order; Stage2 LAST so
# it sits on top of the near-coincident certificate cluster.
ARMS = [
    ("Vanilla-MPC",       "Vanilla-MPC",        "#D55E00", False),
    ("Constant-Velocity", "Constant-Velocity",  "#7A7A7A", True),
    ("Fixed-Predictor",   "Fixed-Predictor",    "#E69F00", True),
    ("Conformal-MPC",     "Conformal-MPC",      "#56B4E9", True),
    ("Stage-1b",          "Stage-1b",           "#009E73", True),
    ("Stage2",            "Stage-2 (ours)",     "#0072B2", True),
]

D_SEP = 30.0
ZOOM_X = (26.0, 35.0)
ZOOM_Y = (0.0, 0.22)

d = np.load(os.path.join(DATA_DIR, "minsep_effort_v2.npz"), allow_pickle=True)


def ecdf(x):
    """Right-continuous empirical CDF, to be drawn with step(where='post').

    The leading (x[0], 0) vertex makes the curve start on the axis instead of
    at 1/n, and keeps the first jump at the first order statistic.
    """
    xs = np.sort(np.asarray(x, float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return np.concatenate([xs[:1], xs]), np.concatenate([[0.0], ys])


def draw_curves(target, lw_main=1.7, lw_ours=2.1):
    for (key, disp, col, cert) in ARMS:
        npz_key = key + "__minsep"
        if npz_key not in d.files:
            continue
        xs, ys = ecdf(d[npz_key])
        target.step(xs, ys, where="post", color=col,
                    linestyle="-" if cert else "--",
                    linewidth=lw_ours if key == "Stage2" else lw_main,
                    zorder=4 if key == "Stage2" else 3)


fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)

# -- loss-of-separation shading (x < 30 m) ----------------------------
ax.axvspan(0, D_SEP, color="#C0392B", alpha=0.06, zorder=0)
ax.text(18.5, 0.985, "loss-of-separation\nregion", fontsize=7.5,
        color="#B03A2E", ha="center", va="top", style="italic", zorder=1)

# -- ECDF curves ------------------------------------------------------
draw_curves(ax)

# -- 30 m standard: dotted line, rotated label left of it -------------
ax.axvline(D_SEP, color="#C0392B", linestyle=":", linewidth=1.3, zorder=2.5)
ax.text(D_SEP - 1.0, 0.60, "30 m separation standard", color="#C0392B",
        fontsize=7.5, rotation=90, ha="right", va="center", zorder=4)

# -- zoom source rectangle (NO connector lines: they cross the data) --
ax.add_patch(Rectangle((ZOOM_X[0], ZOOM_Y[0]),
                       ZOOM_X[1] - ZOOM_X[0], ZOOM_Y[1] - ZOOM_Y[0],
                       fill=False, edgecolor="#999999", linestyle=(0, (3, 2)),
                       linewidth=0.9, zorder=5))

# -- lower-right inset: threshold neighbourhood -----------------------
# region x>52 m, ECDF<0.46 contains no curve (all arms are above 0.55
# there); verified against the real data.
axins = ax.inset_axes([0.66, 0.08, 0.32, 0.38])
draw_curves(axins, lw_main=1.1, lw_ours=1.5)
axins.axvline(D_SEP, color="#C0392B", linestyle=":", linewidth=1.0, zorder=2)
axins.set_xlim(*ZOOM_X)
axins.set_ylim(*ZOOM_Y)
axins.set_xticks([26, 28, 30, 32, 34])
axins.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20])
axins.set_xlabel("min separation (m)", fontsize=6.5, labelpad=1.5)
axins.tick_params(labelsize=6.0, pad=1.5)
axins.set_title("26\u201335 m zoom \u2014 ECDF at 30 m = CR",
                fontsize=6.5, color="#555555", pad=2.5)
for sp in axins.spines.values():
    sp.set_edgecolor("#999999")
    sp.set_linestyle((0, (3, 2)))
    sp.set_linewidth(0.9)

# -- axes -------------------------------------------------------------
ax.set_xlabel("Minimum realised separation (m)")
ax.set_ylabel("Empirical CDF")
ax.set_xlim(8, 75)
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.6)
ax.set_axisbelow(True)

# -- legend inside the empty upper-left void --------------------------
leg_handles = [
    Line2D([0], [0], color="#0072B2", lw=2.1, ls="-", label="Stage-2 (ours)"),
    Line2D([0], [0], color="#009E73", lw=1.7, ls="-", label="Stage-1b"),
    Line2D([0], [0], color="#E69F00", lw=1.7, ls="-", label="Fixed-Predictor"),
    Line2D([0], [0], color="#7A7A7A", lw=1.7, ls="-", label="Constant-Velocity"),
    Line2D([0], [0], color="#56B4E9", lw=1.7, ls="-", label="Conformal-MPC"),
    Line2D([0], [0], color="#D55E00", lw=1.7, ls="--", label="Vanilla-MPC"),
    Line2D([0], [0], color="k", lw=1.5, ls="-", label="solid: CBF-constrained"),
    Line2D([0], [0], color="k", lw=1.5, ls="--", label="dashed: certificate-free"),
]
ax.legend(handles=leg_handles, loc="upper left", bbox_to_anchor=(0.015, 0.90),
          frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
          handletextpad=0.5, labelspacing=0.45, borderpad=0.55)

os.makedirs(OUT_DIR, exist_ok=True)
pdf = os.path.join(OUT_DIR, "fig02_minsep_ecdf.pdf")
png = os.path.join(OUT_DIR, "fig02_minsep_ecdf.png")
fig.savefig(pdf)
fig.savefig(png, dpi=300)
print("saved:", pdf)
print("saved:", png)

# -- provenance self-check: the drawn height at 30 m must equal the table CR
_prov = os.path.join(DATA_DIR, "PROVENANCE_v2.json")
_table = {}
if os.path.exists(_prov):
    import json
    _table = {k: v["metrics"]["CR_%"]
              for k, v in json.load(open(_prov))["arms"].items()}
for (key, _, _, _) in ARMS:
    k = key + "__minsep"
    if k in d.files:
        v = np.asarray(d[k], float)
        xs, ys = ecdf(v)
        drawn = ys[np.searchsorted(xs, D_SEP, side="right") - 1]
        cr = 100 * (v < D_SEP).mean()
        tab = _table.get(key)
        ok = "" if tab is None else ("  OK" if abs(cr - tab) < 1e-9 else "  MISMATCH")
        print(f"  {key:18s} ECDF(30m) = {cr:5.1f}%   drawn = {100*drawn:5.1f}%"
              f"   table = {'n/a' if tab is None else format(tab, '5.1f')}{ok}")
