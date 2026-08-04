#!/usr/bin/env python3
"""Fig. 7 -- Matched operational differences: Stage 2 minus the Stage-1b control.

Replaces the tab:stage1b scorecard. Three stacked panels, height ratio 6:2:1,
because the three quantity classes carry different units and cannot share a
horizontal axis:

  (a) risk difference        percentage points   6 rows (nominal + 5 regimes)
  (b) distance differences   metres              2 rows
  (c) effort difference      dimensionless       1 row

Panels are stacked rather than placed side by side: their row sets are
different (6 / 2 / 1), so there is no shared label column to align, and a
horizontal arrangement would leave panels (b) and (c) mostly empty.

Naming and statistics follow the review adjudication. This is NOT an
equivalence plot: no equivalence margin was pre-specified and no TOST was run,
so the figure is titled "estimated matched differences" and no margin band is
drawn. Binary rows use exact conditional intervals for the paired risk difference,
consistent with the exact McNemar tests reported in the text; continuous rows
use the same critical values as the published sources (Student-t for min-sep
and effort, z for the max-offset diagnostic).

Scope is defined by quantity type, not by outcome: the figure carries closed-
loop operational metrics. ADE and critical-window error are prediction-space
quantities and live in Table 1 and the error-profile analysis.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.gridspec import GridSpec               # noqa: E402

DATA = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
OUTD = os.environ.get("FIG_OUT_DIR",
                      "/data/lab/TRC_UAV_WEIGHTS/figures_v1")

C_PT = "#0072B2"       # Stage-2 vs Stage-1b contrast colour
C_ZERO = "#8A8A8A"     # null line: neutral grey, NOT the red rescue line of
                       # Fig. 5 -- that red marks a physical threshold, this
                       # marks a null hypothesis.

REGIMES = [("nominal", "nominal"),
           ("mass +20%", "mass $+20\\%$"),
           ("thrust eff 0.85", "thrust eff.\\ $0.85$"),
           ("actuator delay 2", "actuator delay $2$"),
           ("combined", "combined")]

# Published targets for the hard self-check.
TGT_EFFORT = (-0.5496, -1.6824, +0.5832)      # EFF_MATCHED_S1B.txt
TGT_OFFSET = (+0.52, -1.11, +2.14)            # DEV_MATCHED_S1B.txt (2 d.p.)
TGT_MINSEP = (-0.04, -0.35, +0.27)            # tab:stage1b (2 d.p.)
TGT_DISC = {"nominal": (2, 1), "mass +20%": (1, 2), "thrust eff 0.85": (2, 3),
            "actuator delay 2": (5, 1), "combined": (2, 0)}


def exact_conditional_paired(b, c, n, alpha=0.05):
    """Exact conditional interval for the paired risk difference.

    Condition on the m = b + c discordant pairs; under conditioning
    b ~ Binomial(m, theta). A Clopper-Pearson interval on theta maps to the
    risk difference through d = (1 - 2*theta) * m / n, a monotone linear map,
    so the endpoints carry over directly (the negative slope swaps them).

    This is used in place of an unconditional approximation (e.g. Newcombe's
    method 10) for one reason: the manuscript reports these five comparisons
    with exact McNemar tests, and an unconditional interval is not guaranteed
    to agree with a conditional test. A Newcombe interval placed the delay-2
    row (b=5, c=1) entirely below zero while its exact McNemar p is 0.22 --
    figure and text would have contradicted each other. Agreement here is a
    constraint, not a preference, and is asserted below.

    At b = m the Clopper-Pearson upper limit for theta is exactly 1, so the
    lower endpoint of d coincides with the point estimate. That is correct
    boundary behaviour for an exact interval, not a defect, and the interval
    still covers zero for the discordant counts seen here.
    """
    from scipy import stats
    m = b + c
    if m == 0:
        return 0.0, 0.0, 0.0
    lo_t = 0.0 if b == 0 else stats.beta.ppf(alpha / 2, b, m - b + 1)
    hi_t = 1.0 if b == m else stats.beta.ppf(1 - alpha / 2, b + 1, m - b)
    d = (1 - 2 * (b / m)) * m / n
    return 100 * d, 100 * (1 - 2 * hi_t) * m / n, 100 * (1 - 2 * lo_t) * m / n


def mcnemar_exact_p(b, c):
    from scipy import stats
    m = b + c
    if m == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), m, 0.5).pvalue)


def paired_t(a, b):
    from scipy import stats
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return m, m - t * se, m + t * se


def paired_z(a, b, z=1.96):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    return m, m - z * se, m + z * se


def main():
    f = f"{DATA}/matched_pair_v2.npz"
    if not os.path.exists(f):
        sys.exit(f"missing {f}; run export_matched_pair_v2.py first")
    D = np.load(f, allow_pickle=True)

    # ---- assemble rows ----
    risk = []
    for rk, lab in REGIMES:
        c1 = D[f"conflict__{rk}__stage1b"].astype(bool)
        c2 = D[f"conflict__{rk}__stage2"].astype(bool)
        b = int((c1 & ~c2).sum())
        c = int((~c1 & c2).sum())
        m, lo, hi = exact_conditional_paired(b, c, c1.size)
        risk.append((lab, m, lo, hi, f"({b},{c})", b, c))

    ms = paired_t(D["nominal_minsep_stage2"], D["nominal_minsep_stage1b"])
    # max offset was published with z=1.96 (p2_mcnemar_dev.paired_diff_ci)
    mo = paired_z(D["nominal_maxoffset_stage2"], D["nominal_maxoffset_stage1b"])
    dist = [("min.\\ separation", *ms, "$n{=}200$"),
            ("max.\\ lateral offset", *mo, "$n{=}200$")]
    ef = paired_t(D["nominal_effort_stage2"], D["nominal_effort_stage1b"])
    eff = [("control effort", *ef, "$n{=}200$")]

    # ---- self-check ----
    errs = []
    for (lab, m, lo, hi, tag, b, c), (rk, _l) in zip(risk, REGIMES):
        b, c = TGT_DISC[rk]
        if tag != f"({b},{c})":
            errs.append(f"{rk}: discordant {tag} != ({b},{c})")
        if not (lo - 1e-12 <= m <= hi + 1e-12):
            errs.append(f"{rk}: point estimate outside its interval")
        # THE consistency constraint: the interval may exclude zero only when
        # the exact McNemar test is significant. This is what the earlier
        # Newcombe implementation violated.
        p = mcnemar_exact_p(b, c)
        if (lo <= 0 <= hi) != (p > 0.05):
            errs.append(f"{rk}: interval/{'excludes' if not (lo<=0<=hi) else 'spans'} "
                        f"zero disagrees with exact McNemar p={p:.4f}")
    for nm, got, tgt in [("effort", ef, TGT_EFFORT), ("minsep", ms, TGT_MINSEP),
                         ("maxoffset", mo, TGT_OFFSET)]:
        dp = 4 if nm == "effort" else 2
        for i, w in enumerate(("mean", "lo", "hi")):
            if abs(round(got[i], dp) - tgt[i]) > 10 ** (-dp) / 2:
                errs.append(f"{nm} {w}: {got[i]:.4f} != published {tgt[i]}")
    if errs:
        raise AssertionError("fig07 self-check failed:\n  " + "\n  ".join(errs))
    print("fig07 self-check: all invariants hold")

    # ---- draw ----
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 9, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.1, 4.5))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[6, 2, 1],
                  hspace=0.42, left=0.255, right=0.90, top=0.955, bottom=0.175)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    blocks = [
        (axes[0], risk, "risk difference (percentage points)", "(a)"),
        (axes[1], dist, "distance difference (m)", "(b)"),
        (axes[2], eff, "effort difference (dimensionless)", "(c)"),
    ]

    for ax, rows, xlab, tag in blocks:
        ys = np.arange(len(rows))[::-1]
        for y, row in zip(ys, rows):
            lab, m, lo, hi, note = row[0], row[1], row[2], row[3], row[4]
            ax.plot([lo, hi], [y, y], color=C_PT, lw=1.5,
                    solid_capstyle="butt", zorder=3)
            for xx in (lo, hi):
                ax.plot([xx, xx], [y - 0.16, y + 0.16], color=C_PT, lw=1.5,
                        zorder=3)
            ax.plot([m], [y], "o", ms=5.4, color=C_PT, mec="white", mew=0.8,
                    zorder=4)
            ax.text(1.014, y, note, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=7.6, color="#444444")
        ax.axvline(0.0, color=C_ZERO, lw=1.4, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8.4)
        ax.set_ylim(-0.62, len(rows) - 0.38)
        ax.set_xlabel(xlab, fontsize=8.6, labelpad=2.0)
        ax.grid(axis="x", color="#E4E4E4", lw=0.55, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8.0)
        # symmetric limits so the null line sits visually central
        lim = max(abs(v) for r in rows for v in (r[2], r[3])) * 1.30
        ax.set_xlim(-lim, lim)
        ax.text(-0.335, 1.0, tag, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=10, fontweight="bold")

    axes[0].text(0.5, 1.045,
                 "positive $=$ Stage 2 higher than the Stage-1b control",
                 transform=axes[0].transAxes, ha="center", va="bottom",
                 fontsize=8.2, color="#333333")
    # legend goes BELOW panel (c): inside panel (a) it overlapped the
    # combined row's interval, which reaches x = -1.0 .. +0.68 pp.
    axes[2].legend(handles=[
        Line2D([], [], color=C_PT, marker="o", ms=5.4, lw=1.5, mec="white",
               mew=0.8, label="paired difference, 95\\% CI"),
        Line2D([], [], color=C_ZERO, lw=1.4, label="no difference")],
        loc="upper center", bbox_to_anchor=(0.5, -0.62), ncol=2,
        frameon=False, fontsize=8.0, handletextpad=0.6, columnspacing=2.0)

    for ext in ("pdf", "png"):
        p = f"{OUTD}/fig07_matched.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("saved:", p)

    print("  (a) risk, pp:")
    for lab, m, lo, hi, tag, _b, _c in risk:
        print(f"      {lab:22s} {m:+5.2f} [{lo:+5.2f},{hi:+5.2f}] disc={tag}"
              f"{'  spans zero' if lo <= 0 <= hi else '  EXCLUDES ZERO'}")
    print(f"  (b) minSep    {ms[0]:+.4f} [{ms[1]:+.4f},{ms[2]:+.4f}]")
    print(f"      maxOffset {mo[0]:+.4f} [{mo[1]:+.4f},{mo[2]:+.4f}]")
    print(f"  (c) effort    {ef[0]:+.4f} [{ef[1]:+.4f},{ef[2]:+.4f}]")
    n_excl = sum(1 for r in risk if not (r[2] <= 0 <= r[3]))
    print(f"  rows whose interval excludes zero: {n_excl} of "
          f"{len(risk)+len(dist)+len(eff)}")


if __name__ == "__main__":
    main()
