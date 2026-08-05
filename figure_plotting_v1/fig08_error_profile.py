#!/usr/bin/env python3
"""Fig. 8 -- Prediction error relative to closest approach.

Neutral title by design. The earlier figure asserted that the planner-routed
objective concentrates accuracy where the constraints bind; the point estimates
refute that in its strong form, so the figure reports where each predictor
spends its accuracy and lets the inset carry the ratio.

Layout: two stacked LINEAR panels sharing the abscissa, height ratio 1:2.
  (a) Stage 1 alone, y ~ 13-14.5 m -- the flat context-only reference.
  (b) Stage-1b and Stage 2, y ~ 0-6 m -- the comparison of record.

Why not a single log ordinate
-----------------------------
Two separate arguments, kept separate on purpose.

  The fig05 log-axis veto does NOT apply here. That veto was semantic: fig05's
  entire claim rests on a zero reference (the rescue line), and a log ordinate
  has no zero, so it would have destroyed the argument through the coordinate
  system. An error profile carries no zero reference that must stay visible.
  RULE OF SCOPE: the log-axis red line is conditioned on zero-reference
  semantics, not on large dynamic range. Cite this when cross-range data
  recurs; do not re-litigate.

  A log ordinate is nevertheless rejected for an independent reason. The
  comparison of record here is multiplicative -- 1.04 vs 4.07 m critical error,
  ratios 0.499 vs 0.925 -- and a log axis compresses exactly that multiplicative
  gap into a constant visual offset. A factor of four would read as "not much".
  The point the reader must see is that the matched control is a whole band
  better inside the critical window, so both panels stay linear and each spans
  its own full range.

Unequal bucket counts: |k - k_CPA| = 0 has exactly one step per episode
(n = 200); every other bucket collects the two symmetric steps (n <= 400), with
mild attrition at the far end from horizon truncation. Bucket n is annotated on
the figure so the weighting convention is not left to inference.

Inset (lower panel, where its comparison pair lives): critical-to-inert ratio as
three point estimates, NOT bars. These are descriptive quantities with no
interval estimate, and bars would imply a precision never computed.

The profile is bound to the table by construction: the step-count-weighted mean
of the shaded buckets equals each arm's tabulated critical error.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.gridspec import GridSpec            # noqa: E402

DATA = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
OUTD = os.environ.get("FIG_OUT_DIR",
                      "/data/lab/TRC_UAV_WEIGHTS/figures_v1")

STYLE = {"Stage-1b": ("Stage-1b (matched control)", "#009E73", "s", "-"),
         "Stage2": ("Stage 2 (PlanGrad)", "#0072B2", "o", "-"),
         "Stage1": ("Stage 1 (context only)", "#7A7A7A", "^", "--")}

TAB_CRIT = {"Stage1": 13.53, "Stage-1b": 1.04, "Stage2": 4.07}
TAB_INERT = {"Stage1": 23.14, "Stage-1b": 2.08, "Stage2": 4.40}
TAB_RATIO = {"Stage1": 0.585, "Stage-1b": 0.499, "Stage2": 0.925}

GREY = "#8A8A8A"


def panel_label(ax, s, dx=-0.055, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, fontweight="bold")


def main():
    f = f"{DATA}/errdir_v2.npz"
    if not os.path.exists(f):
        sys.exit(f"missing {f}; run export_errdir_v2.py first")
    D = np.load(f, allow_pickle=True)
    buckets = D["buckets"]
    crit_w = int(D["crit_window"][0])

    # ---------------- self-checks (refuse to plot if any fails) -------------
    errs = []
    for tag in STYLE:
        prof = D[f"profile__{tag}"]
        cnts = D[f"counts__{tag}"]
        sel = buckets <= crit_w
        wc = float((prof[sel] * cnts[sel]).sum() / cnts[sel].sum())
        if abs(wc - TAB_CRIT[tag]) >= 5e-3:
            errs.append(f"{tag}: weighted critical {wc:.4f} != "
                        f"table {TAB_CRIT[tag]}")
        if D[f"epi_par__{tag}"].size != 200:
            errs.append(f"{tag}: epi_par n={D[f'epi_par__{tag}'].size} != 200")
        r = TAB_CRIT[tag] / TAB_INERT[tag]
        if abs(r - TAB_RATIO[tag]) > 6e-3:
            errs.append(f"{tag}: crit/inert {r:.4f} != table {TAB_RATIO[tag]}")
    c0 = D["counts__Stage2"]
    if int(c0[0]) != 200:
        errs.append(f"bucket 0 count {int(c0[0])} != 200 (one CPA step/episode)")
    if not all(int(v) <= 400 for v in c0[1:]):
        errs.append("some non-zero bucket exceeds 400 steps")
    if errs:
        raise AssertionError("fig08 self-check failed:\n  " + "\n  ".join(errs))
    print("fig08 self-check: all invariants hold")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 9, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.1, 4.35))
    gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.13,
                  left=0.098, right=0.985, top=0.945, bottom=0.135)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], sharex=axa)

    for ax in (axa, axb):
        ax.axvspan(-0.35, crit_w + 0.35, color="#000000", alpha=0.055, lw=0,
                   zorder=0)
        ax.grid(color="#E2E2E2", lw=0.55, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---- (a) Stage 1 alone -------------------------------------------------
    lab, col, mk, ls = STYLE["Stage1"]
    p1 = D["profile__Stage1"]
    axa.plot(buckets, p1, ls=ls, color=col, lw=1.5, marker=mk, ms=4.6,
             mec="white", mew=0.7, label=lab, zorder=3)
    axa.set_ylim(12.9, 14.7)
    axa.set_yticks([13.0, 13.5, 14.0, 14.5])
    axa.set_ylabel("error (m)", fontsize=8.6)
    axa.tick_params(labelbottom=False, labelsize=8.2)
    axa.legend(loc="upper left", bbox_to_anchor=(0.012, 1.0), frameon=False,
               fontsize=8.2, handlelength=2.4)
    axa.text(crit_w / 2.0, 0.955,
             f"critical window $|k-k_{{\\mathrm{{CPA}}}}|\\leq{crit_w}$",
             transform=axa.get_xaxis_transform(), ha="center", va="top",
             fontsize=7.6, color="#555555")
    panel_label(axa, "(a)")

    # ---- (b) Stage-1b and Stage 2 -----------------------------------------
    for tag in ("Stage2", "Stage-1b"):
        lab, col, mk, ls = STYLE[tag]
        axb.plot(buckets, D[f"profile__{tag}"], ls=ls, color=col, lw=1.6,
                 marker=mk, ms=4.8, mec="white", mew=0.7, label=lab, zorder=3)
    axb.set_ylim(0.0, 6.0)
    axb.set_yticks([0, 1, 2, 3, 4, 5, 6])
    axb.set_ylabel("mean $|$prediction error$|$ (m)", fontsize=9)
    axb.set_xlabel("steps from closest approach, "
                   "$|k-k_{\\mathrm{CPA}}|$", fontsize=9)
    axb.set_xlim(-0.5, buckets[-1] + 0.5)
    axb.set_xticks(buckets)
    axb.tick_params(labelsize=8.4)
    axb.legend(loc="upper left", bbox_to_anchor=(0.012, 0.995), frameon=False,
               fontsize=8.2, handlelength=2.4, labelspacing=0.35)
    panel_label(axb, "(b)")

    # bucket sample sizes, along the bottom of (b)
    for k, n in zip(buckets, D["counts__Stage2"]):
        axb.text(k, 0.022, f"{int(n)}", transform=axb.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=6.4, color="#777777")
    axb.text(-0.42, 0.022, "$n$:", transform=axb.get_xaxis_transform(),
             ha="left", va="bottom", fontsize=6.4, color="#777777")

    # ---- inset: crit/inert ratio, three points, never bars ----------------
    # Placed in the genuine vertical gap between the two curves on the right of
    # panel (b): over buckets 4-7 Stage-1b peaks at 1.194 m (axes fraction
    # 0.199) and Stage 2 bottoms at 4.600 m (fraction 0.767), leaving that band
    # empty. The inset's tight bounding box reaches 0.056 of the panel height
    # BELOW its axes frame (y-label plus tick text), which is why an axes origin
    # of 0.25 still collided; 0.300 puts the tight box at 0.244-0.605. The
    # occlusion check below enforces this rather than trusting the arithmetic.
    axi = axb.inset_axes([0.638, 0.300, 0.330, 0.295])
    order = ["Stage-1b", "Stage2", "Stage1"]
    for x, tag in enumerate(order):
        _l, col, mk, _s = STYLE[tag]
        axi.plot([x], [TAB_RATIO[tag]], marker=mk, ms=5.6, color=col,
                 mec="white", mew=0.7, zorder=3)
        axi.text(x, TAB_RATIO[tag] + 0.052, f"{TAB_RATIO[tag]:.3f}",
                 ha="center", va="bottom", fontsize=7.0, color="#333333")
    axi.axhline(1.0, color=GREY, lw=1.0, ls=":", zorder=1)
    axi.text(len(order) - 0.42, 1.0, "equal", ha="left", va="center",
             fontsize=6.6, color="#666666")
    axi.set_xticks(range(len(order)))
    axi.set_xticklabels(["S-1b", "S-2", "S-1"], fontsize=7.0)
    axi.set_ylim(0.36, 1.16)
    axi.set_yticks([0.5, 0.75, 1.0])
    axi.tick_params(labelsize=6.8, length=2.0)
    axi.set_ylabel("critical / inert", fontsize=7.2, labelpad=1.5)
    axi.set_xlim(-0.55, len(order) + 0.42)
    for s in ("top", "right"):
        axi.spines[s].set_visible(False)
    axi.grid(axis="y", color="#EDEDED", lw=0.5, zorder=0)
    axi.set_axisbelow(True)

    # ---- geometric occlusion check, in display coordinates ----------------
    # Every overlay (two legends, the ratio inset, the bucket-n row) is tested
    # against every plotted vertex and against the line segments joining them.
    # A pixel-colour heuristic cannot do this reliably: the curves span the full
    # panel width, so colour alone cannot distinguish "curve under the legend"
    # from "curve elsewhere in the same pixel rows".
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    overlays = {"legend (a)": axa.get_legend().get_window_extent(r),
                "legend (b)": axb.get_legend().get_window_extent(r),
                "ratio inset": axi.get_tightbbox(r)}

    curves = {"Stage1": (axa, D["profile__Stage1"]),
              "Stage-1b": (axb, D["profile__Stage-1b"]),
              "Stage2": (axb, D["profile__Stage2"])}

    def seg_hits_bbox(p0, p1, bb, steps=200):
        """Sample a segment densely and report whether any sample is inside."""
        for t in np.linspace(0.0, 1.0, steps):
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            if bb.x0 <= x <= bb.x1 and bb.y0 <= y <= bb.y1:
                return True
        return False

    occl = []
    for name, bb in overlays.items():
        host = axa if name.endswith("(a)") else axb
        for tag, (ax, prof) in curves.items():
            if ax is not host:
                continue
            pts = ax.transData.transform(np.column_stack([buckets, prof]))
            for i in range(len(pts)):
                if bb.x0 <= pts[i, 0] <= bb.x1 and bb.y0 <= pts[i, 1] <= bb.y1:
                    occl.append(f"{name} covers {tag} marker at "
                                f"|k-kCPA|={int(buckets[i])}")
                if i + 1 < len(pts) and seg_hits_bbox(pts[i], pts[i + 1], bb):
                    occl.append(f"{name} covers {tag} segment "
                                f"{int(buckets[i])}-{int(buckets[i+1])}")
        print(f"  overlay '{name}': x [{bb.x0:.0f},{bb.x1:.0f}] "
              f"y [{bb.y0:.0f},{bb.y1:.0f}]")
    if occl:
        raise AssertionError("fig08 occlusion check failed:\n  "
                             + "\n  ".join(sorted(set(occl))))
    print("fig08 occlusion check: no overlay touches any curve")

    for ext in ("pdf", "png"):
        p = f"{OUTD}/fig08_error_profile.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        print("saved:", p)

    print("\nprofile shapes (caption material):")
    for tag in ("Stage1", "Stage-1b", "Stage2"):
        pr = D[f"profile__{tag}"]
        print(f"  {tag:9s} b0={pr[0]:7.3f}  b7={pr[-1]:7.3f}  "
              f"outer/inner={pr[-1]/pr[0]:.3f}  "
              f"crit/inert={TAB_RATIO[tag]:.3f}")


if __name__ == "__main__":
    main()
