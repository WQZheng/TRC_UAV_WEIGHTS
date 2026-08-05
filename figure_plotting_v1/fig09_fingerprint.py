#!/usr/bin/env python3
"""Fig. 9 -- Task-aligned component of the prediction error, Stage 2 vs Stage-1b.

A Gardner-Altman estimation plot, chosen over a histogram or a raincloud because
the quantity of record is a PAIRED contrast on the same 200 encounters: the
paired difference and its interval belong on their own axis, aligned with the
arm-level data they are derived from.

  (a) paired episode means for the two arms, one line per encounter, plus the
      arm means. The abscissa carries only two levels, so no marker jitter is
      used; the connecting lines are the data structure.
  (b) the 200 paired differences against a difference axis whose zero is aligned
      with the Stage-1b mean of panel (a), with the mean difference and its
      bootstrap interval.

Outlier discipline
------------------
Encounter 93 is a lone geometric extreme: Stage-1 -54.09, Stage-1b -47.52,
Stage-2 -46.22 m, against a second-largest Stage-1b magnitude of only -1.08 m.
All three arms are extreme together, so the encounter itself is anomalous rather
than any one predictor. An ordinate spanning it would compress the other 199
encounters into a single line, so panel (a) focuses on [-2.2, +2.2] and marks
the encounter explicitly at the axis edge with its numeric values. This is not a
broken axis: the point is not hidden, it is annotated off-scale, and its paired
difference (+1.299 m) lies well inside the panel (b) range and is drawn there.

The encounter does not drive the contrast -- dropping it moves the paired mean
from -0.164 to -0.171 m and the p-value from 0.0074 to 0.0051 -- but it does
dominate the ARM-level dispersion that tab:errdir reports, which is why the
leave-one-out values are printed for the manuscript footnote.

Geometry inset
--------------
A ~1.5 cm inset defines the sign convention, five elements only: ego, true
neighbour, predicted neighbour, the line-of-sight axis, and the projection
marking e_parallel < 0 as "predicted closer to ego". It is an extension of the
axis label, not a concept schematic.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.gridspec import GridSpec            # noqa: E402
from matplotlib.patches import FancyArrowPatch      # noqa: E402
from scipy import stats                             # noqa: E402

DATA = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
OUTD = os.environ.get("FIG_OUT_DIR",
                      "/data/lab/TRC_UAV_WEIGHTS/figures_v1")

C_S1B = "#009E73"
C_S2 = "#0072B2"
GREY = "#8A8A8A"
# Panel (a) focuses on the bulk; encounter 93 is annotated off-scale. Panel (b)
# must span its data in full -- the paired differences reach -2.289 and +2.399 m
# -- because there is no outlier there to justify clipping, and a silently
# clipped point is hidden data. The two panels therefore carry different
# ordinate ranges, which the caption states.
YLO, YHI = -2.2, 2.2
YLO_B, YHI_B = -2.6, 2.6

TAB = {"Stage-1b": (-0.176, 0.239, 0.454), "Stage2": (-0.340, 0.235, 0.546)}
TAB_PAIRED = (-0.164, 0.007399, 0.001137)
TAB_CI = (-0.283, -0.045)


def main():
    f = f"{DATA}/errdir_v2.npz"
    if not os.path.exists(f):
        sys.exit(f"missing {f}; run export_errdir_v2.py first")
    D = np.load(f, allow_pickle=True)
    a = D["epi_par__Stage-1b"]
    b = D["epi_par__Stage2"]
    fa = D["epi_fneg__Stage-1b"]
    fb = D["epi_fneg__Stage2"]
    d = b - a
    n = d.size

    se = d.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    ci = (d.mean() - tc * se, d.mean() + tc * se)
    rng = np.random.default_rng(12345)
    bs = np.array([rng.choice(d, n, replace=True).mean() for _ in range(20000)])
    bci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

    # ---------------- self-checks: refuse to plot on mismatch ---------------
    errs = []
    for tag, v, fv in (("Stage-1b", a, fa), ("Stage2", b, fb)):
        m, s, fr = TAB[tag]
        if v.size != 200:
            errs.append(f"{tag}: n={v.size} != 200")
        if abs(v.mean() - m) > 5e-4:
            errs.append(f"{tag}: mean {v.mean():+.4f} != {m}")
        if abs(v.std(ddof=1) / np.sqrt(v.size) - s) > 5e-4:
            errs.append(f"{tag}: SEM != {s}")
        if abs(fv.mean() - fr) > 5e-4:
            errs.append(f"{tag}: toward-ego frac {fv.mean():.4f} != {fr}")
    if abs(d.mean() - TAB_PAIRED[0]) > 5e-4:
        errs.append(f"paired mean {d.mean():+.4f} != {TAB_PAIRED[0]}")
    if abs(stats.ttest_rel(b, a).pvalue - TAB_PAIRED[1]) > 5e-5:
        errs.append("paired-t p mismatch")
    if abs(stats.wilcoxon(b, a).pvalue - TAB_PAIRED[2]) > 5e-5:
        errs.append("Wilcoxon p mismatch")
    if (abs(ci[0] - TAB_CI[0]) > 6e-4) or (abs(ci[1] - TAB_CI[1]) > 6e-4):
        errs.append(f"t-CI [{ci[0]:+.4f},{ci[1]:+.4f}] != {TAB_CI}")
    # the outlier must be a single encounter, and must be shared by both arms
    out = np.where(np.abs(a) > 8)[0]
    if out.size != 1:
        errs.append(f"expected exactly 1 |e_par|>8 encounter, got {out.size}")
    if np.abs(b[out]).min() <= 8:
        errs.append("outlier encounter is not extreme in Stage 2 as well")
    if abs(d[out[0]]) > YHI:
        errs.append("outlier's paired difference falls outside panel (b)")
    # nothing in panel (b) may be silently clipped
    if d.min() < YLO_B or d.max() > YHI_B:
        errs.append(f"panel (b) range [{YLO_B},{YHI_B}] clips the differences "
                    f"[{d.min():+.4f},{d.max():+.4f}]")
    if errs:
        raise AssertionError("fig09 self-check failed:\n  " + "\n  ".join(errs))
    print("fig09 self-check: all invariants hold")
    io = int(out[0])

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 9, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.1, 4.0))
    gs = GridSpec(1, 2, width_ratios=[1.32, 1.0], wspace=0.30,
                  left=0.093, right=0.982, top=0.935, bottom=0.115)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])

    # ================= panel (a): paired episode means ======================
    x1, x2 = 0.0, 1.0
    inl = np.setdiff1d(np.arange(n), [io])
    for i in inl:
        axa.plot([x1, x2], [a[i], b[i]], color="#B9B9B9", lw=0.45, alpha=0.55,
                 zorder=1, solid_capstyle="round")
    axa.plot([x1] * inl.size, a[inl], ls="none", marker="s", ms=2.8,
             color=C_S1B, alpha=0.80, mec="none", zorder=3)
    axa.plot([x2] * inl.size, b[inl], ls="none", marker="o", ms=2.8,
             color=C_S2, alpha=0.80, mec="none", zorder=3)

    # arm means with SEM
    for x, v, col in ((x1, a, C_S1B), (x2, b, C_S2)):
        m = v.mean()
        s = v.std(ddof=1) / np.sqrt(v.size)
        axa.errorbar([x], [m], yerr=[s], color=col, marker="D", ms=5.4,
                     mec="white", mew=0.8, elinewidth=1.6, capsize=3.4,
                     capthick=1.2, zorder=5)
        axa.annotate(f"{m:+.3f}", (x, m), textcoords="offset points",
                     xytext=(13 if x == x1 else 13, -11), fontsize=7.6,
                     color=col, ha="left")

    axa.axhline(0.0, color=GREY, lw=0.9, ls="--", zorder=2)

    # the off-scale encounter, annotated rather than hidden
    for x, val, col in ((x1, a[io], C_S1B), (x2, b[io], C_S2)):
        axa.annotate(
            "", xy=(x, YLO + 0.045), xytext=(x, YLO + 0.40),
            arrowprops=dict(arrowstyle="-|>", lw=1.15, color=col,
                            shrinkA=0, shrinkB=0), zorder=6)
    axa.text(0.5, YLO + 0.44,
             f"encounter {io}: {a[io]:+.1f} / {b[io]:+.1f} m (off scale)",
             ha="center", va="bottom", fontsize=7.2, color="#444444",
             zorder=6)

    axa.set_xlim(-0.42, 1.42)
    axa.set_ylim(YLO, YHI)
    axa.set_xticks([x1, x2])
    axa.set_xticklabels(["Stage-1b\n(matched control)", "Stage 2\n(PlanGrad)"],
                        fontsize=8.4)
    axa.set_ylabel("episode-mean $e_\\parallel$ in critical window (m)",
                   fontsize=9)
    axa.tick_params(labelsize=8.4)
    axa.grid(axis="y", color="#EDEDED", lw=0.55, zorder=0)
    axa.set_axisbelow(True)
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)
    axa.text(-0.085, 1.015, "(a)", transform=axa.transAxes, ha="left",
             va="bottom", fontsize=10, fontweight="bold")
    axa.text(0.02, 0.145, "$e_\\parallel<0$: neighbour predicted closer to ego"
             " (conservative)", transform=axa.transAxes, ha="left", va="top",
             fontsize=7.2, color="#555555")

    # ---- geometry inset: five elements, sign convention only --------------
    # Placed in the empty upper band. Measured occupancy of panel (a): no point
    # and no connecting line reaches the [+1.8,+2.2] m band (axes fraction
    # 0.909-1.0), and the Stage-1b column is empty above +0.959 m (fraction
    # 0.718). The occlusion check below enforces the placement.
    axg = axa.inset_axes([0.035, 0.735, 0.360, 0.245])
    axg.set_xlim(0, 10)
    axg.set_ylim(0, 4.2)
    ego = (1.0, 1.0)
    tru = (8.4, 3.0)
    prd = (6.2, 2.4)
    axg.plot([ego[0], tru[0]], [ego[1], tru[1]], color=GREY, lw=0.8, ls=":",
             zorder=1)
    axg.plot(*ego, marker="^", ms=5.0, color="#333333", zorder=3)
    axg.plot(*tru, marker="o", ms=4.6, color="#333333", mfc="white", mew=1.0,
             zorder=3)
    axg.plot(*prd, marker="o", ms=4.6, color=C_S2, zorder=3)
    axg.add_patch(FancyArrowPatch(tru, prd, arrowstyle="-|>",
                                  mutation_scale=8, lw=1.2, color=C_S2,
                                  shrinkA=2.4, shrinkB=2.4, zorder=4))
    axg.text(ego[0] - 0.1, ego[1] - 0.72, "ego", fontsize=6.4, ha="left",
             color="#333333")
    axg.text(tru[0] + 0.25, tru[1] - 0.1, "true", fontsize=6.4, ha="left",
             va="center", color="#333333")
    axg.text(prd[0] - 0.35, prd[1] + 0.42, "pred.", fontsize=6.4, ha="center",
             color=C_S2)
    axg.text(7.0, 1.28, "$e_\\parallel<0$", fontsize=6.8, ha="center",
             color=C_S2)
    axg.set_xticks([])
    axg.set_yticks([])
    for s in axg.spines.values():
        s.set_color("#CFCFCF")
        s.set_linewidth(0.6)
    axg.set_facecolor("#FCFCFC")

    # ================= panel (b): paired differences ========================
    xd = 0.0
    axb.axhline(0.0, color=GREY, lw=1.0, ls="--", zorder=2)
    rj = np.random.default_rng(7).uniform(-0.052, 0.052, n)
    axb.plot(xd + rj, d, ls="none", marker="o", ms=2.9, color="#7A7A7A",
             alpha=0.50, mec="none", zorder=3)
    axb.plot([xd + rj[io]], [d[io]], ls="none", marker="o", ms=4.4,
             color="#333333", mfc="none", mew=1.0, zorder=4)
    axb.annotate(f"enc. {io}", (xd + rj[io], d[io]),
                 textcoords="offset points", xytext=(9, -2), fontsize=6.8,
                 color="#444444", va="center")

    # mean difference with its interval
    axb.errorbar([xd + 0.30], [d.mean()], yerr=[[d.mean() - bci[0]],
                                                [bci[1] - d.mean()]],
                 color=C_S2, marker="D", ms=6.2, mec="white", mew=0.9,
                 elinewidth=2.0, capsize=4.2, capthick=1.4, zorder=6)
    txt_ci = axb.annotate(f"{d.mean():+.3f} m\n[{bci[0]:+.3f}, {bci[1]:+.3f}]",
                          (xd + 0.30, d.mean()), textcoords="offset points",
                          xytext=(12, 4), fontsize=7.8, color=C_S2, ha="left")

    axb.set_xlim(-0.30, 0.92)
    axb.set_ylim(YLO_B, YHI_B)
    axb.set_xticks([xd + 0.15])
    axb.set_xticklabels(["Stage 2 $-$ Stage-1b\n(paired, $n=200$)"],
                        fontsize=8.4)
    axb.set_ylabel("paired difference in $e_\\parallel$ (m)", fontsize=9)
    axb.tick_params(labelsize=8.4)
    axb.grid(axis="y", color="#EDEDED", lw=0.55, zorder=0)
    axb.set_axisbelow(True)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.text(-0.115, 1.015, "(b)", transform=axb.transAxes, ha="left",
             va="bottom", fontsize=10, fontweight="bold")
    # In the empty right-hand column, clear of the jittered strip and of the
    # mean-difference marker. Verified by the occlusion check.
    txt_b = axb.text(0.995, 0.028,
                     f"{int((d<0).sum())}/{n} encounters more\nconservative "
                     "under Stage 2", transform=axb.transAxes, ha="right",
                     va="bottom", fontsize=7.2, color="#555555")

    # ---------------- occlusion check in display coordinates ---------------
    # Every overlay is tested against the plotted points AND, in panel (a),
    # against the densely sampled connecting lines, which are data here.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    pts_a = axa.transData.transform(
        np.column_stack([np.r_[np.full(inl.size, x1), np.full(inl.size, x2)],
                         np.r_[a[inl], b[inl]]]))
    p1 = axa.transData.transform(np.column_stack([np.full(inl.size, x1),
                                                  a[inl]]))
    p2 = axa.transData.transform(np.column_stack([np.full(inl.size, x2),
                                                  b[inl]]))
    tt = np.linspace(0.0, 1.0, 120)[None, :, None]
    segs = (p1[:, None, :] * (1 - tt) + p2[:, None, :] * tt).reshape(-1, 2)
    pts_b = axb.transData.transform(np.column_stack([xd + rj, d]))

    overlays = [("geometry inset", axg.get_tightbbox(r), axa,
                 np.vstack([pts_a, segs])),
                ("panel (b) note", txt_b.get_window_extent(r), axb, pts_b),
                ("mean-diff label", txt_ci.get_window_extent(r), axb, pts_b)]
    occl = []
    for name, bb, _ax, pts in overlays:
        hit = ((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1)
               & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1))
        print(f"  overlay '{name}': x [{bb.x0:.0f},{bb.x1:.0f}] "
              f"y [{bb.y0:.0f},{bb.y1:.0f}]  hits={int(hit.sum())}")
        if hit.any():
            occl.append(f"{name} covers {int(hit.sum())} data samples")
    if occl:
        raise AssertionError("fig09 occlusion check failed:\n  "
                             + "\n  ".join(occl))
    print("fig09 occlusion check: no overlay covers any point or line")

    for ext in ("pdf", "png"):
        p = f"{OUTD}/fig09_fingerprint.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        print("saved:", p)

    # ---------------- numbers for the manuscript ---------------------------
    keep = np.delete(np.arange(n), io)
    print("\nfor the leave-one-out footnote:")
    for tag, v in (("Stage-1b", a), ("Stage2", b)):
        m2 = v[keep]
        print(f"  {tag:9s} all: {v.mean():+.4f} +/- "
              f"{v.std(ddof=1)/np.sqrt(v.size):.4f}   "
              f"drop enc.{io}: {m2.mean():+.4f} +/- "
              f"{m2.std(ddof=1)/np.sqrt(m2.size):.4f}")
    d2 = d[keep]
    se2 = d2.std(ddof=1) / np.sqrt(d2.size)
    t2 = stats.t.ppf(0.975, d2.size - 1)
    print(f"  paired    all: {d.mean():+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] "
          f"p={stats.ttest_rel(b, a).pvalue:.5f}")
    print(f"            drop enc.{io}: {d2.mean():+.4f} "
          f"[{d2.mean()-t2*se2:+.4f},{d2.mean()+t2*se2:+.4f}] "
          f"p={stats.ttest_1samp(d2, 0).pvalue:.5f}")
    print(f"  bootstrap CI (20k, seed 12345) = "
          f"[{bci[0]:+.4f},{bci[1]:+.4f}]  t-CI = "
          f"[{ci[0]:+.4f},{ci[1]:+.4f}]")


if __name__ == "__main__":
    main()
