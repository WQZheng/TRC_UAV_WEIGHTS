#!/usr/bin/env python3
"""Fig. 6 -- Certificate advantage under model mismatch.

One panel, nine rows, three arms. Each row is a perturbation regime from
tab:mismatch, kept in TABLE ORDER (the table's order is a semantic grouping:
nominal, mass, inertia, thrust, delay, wind, combined; text-table-figure
alignment is a frozen rule, so the rows are NOT re-sorted by conflict rate).

Each row carries the two certificate arms and the no-certificate comparator.
Keeping Stage-1 + CBF is the point of the figure: the two certificate arms stay
close to each other in every regime while the no-certificate arm runs away, so
the main claim -- safety comes from the certificate, not from predictor quality
-- is shown to survive plant mismatch rather than being a nominal-only artefact.

The wind rows are relabelled. Their disturbance is eta_w * gust_std = 1.0 * 5.0
= 5.0, i.e. 5.6x the canonical 0.3 * 3.0 = 0.9, and they use an independent
field seed (99 vs 7). The sweep's eta_w=1.0 point is a DIFFERENT regime
(strength 3.0, seed 7), so an "eta_w=1.0" label alone would be ambiguous.

Wind is deliberately absent as a panel: across a three-fold change in wind
scaling every arm moves by at most 0.5 pp in conflict rate and 0.05 m in mean
separation, which is sampling noise at n=200. Plotting a zero effect as a
trend line would invite the reader to read a slope that is not there.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.lines import Line2D                # noqa: E402

DATA = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
OUTD = os.environ.get(
    "FIG_OUT_DIR", "/data/lab/TRC_UAV_WEIGHTS/figures_v1")

# Manuscript-wide colour discipline.
C_S2 = "#0072B2"      # Stage-2 (PlanGrad)
C_S1 = "#E69F00"      # Stage-1 frozen predictor == Fixed-Predictor arm
C_VAN = "#D55E00"     # Vanilla, no certificate
C_GAP = "#BBBBBB"

ARMS = [("stage2", "Stage-2 + CBF", C_S2, "o"),
        ("stage1", "Stage-1 + CBF", C_S1, "s"),
        ("vanilla", "Stage-2, no CBF", C_VAN, "D")]

# Row order == tab:mismatch order. Labels spell out the wind strength.
ROWS = [
    ("nominal",          "nominal (matched plant)"),
    ("mass +20%",        "mass $+20\\%$"),
    ("mass -15%",        "mass $-15\\%$"),
    ("inertia +30%",     "inertia $+30\\%$"),
    ("thrust eff 0.85",  "thrust efficiency $0.85$"),
    ("actuator delay 1", "actuator delay $1$ step"),
    ("actuator delay 2", "actuator delay $2$ steps"),
    ("wind shift",       "wind $5.6\\times$ nominal, indep. field"),
    ("combined",         "combined (mass, thrust, delay, wind)"),
]

TAB = {  # published tab:mismatch, for the hard self-check
    "nominal": (11.0, 12.5, 41.0), "mass +20%": (17.5, 18.5, 75.0),
    "mass -15%": (9.0, 10.0, 19.5), "inertia +30%": (12.0, 12.0, 42.5),
    "thrust eff 0.85": (16.5, 18.5, 68.5), "actuator delay 1": (18.5, 22.0, 54.0),
    "actuator delay 2": (28.0, 34.0, 87.5), "wind shift": (11.0, 12.0, 39.5),
    "combined": (30.0, 40.0, 94.5),
}


def wilson(k, n, z=1.96):
    """Wilson score interval in percent; correct at the boundaries."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def load():
    f = f"{DATA}/mismatch_v2.npz"
    if not os.path.exists(f):
        sys.exit(f"missing {f}; run export_robustness.py first")
    d = np.load(f, allow_pickle=True)
    out = {}
    for rk, _ in ROWS:
        for ak, _l, _c, _m in ARMS:
            key = f"conflict__{rk}__{ak}"
            if key not in d:
                sys.exit(f"missing array {key}")
            v = d[key].astype(bool)
            out[(rk, ak)] = v
    return out


def verify(D):
    """Refuse to plot unless every vector reduces to the published table."""
    errs = []
    for rk, _ in ROWS:
        for i, (ak, lab, _c, _m) in enumerate(ARMS):
            v = D[(rk, ak)]
            if v.size != 200:
                errs.append(f"{rk}/{ak}: n={v.size} != 200")
            cr = 100.0 * v.mean()
            if abs(cr - TAB[rk][i]) >= 1e-9:
                errs.append(f"{rk}/{ak}: CR={cr:.4f} != tab {TAB[rk][i]}")
    # the certificate gap must be positive in every regime, else the figure's
    # premise is wrong and the caption would be a lie
    for rk, _ in ROWS:
        g = 100.0 * D[(rk, "vanilla")].mean() - 100.0 * D[(rk, "stage2")].mean()
        if g <= 0:
            errs.append(f"{rk}: certificate gap {g:.1f} pp is not positive")
    if errs:
        raise AssertionError("fig06 self-check failed:\n  " + "\n  ".join(errs))
    print("fig06 self-check: all invariants hold")


def main():
    D = load()
    verify(D)
    os.makedirs(OUTD, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 9, "axes.linewidth": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    n_rows = len(ROWS)
    fig, ax = plt.subplots(figsize=(7.1, 3.9))
    fig.subplots_adjust(left=0.30, right=0.895, top=0.985, bottom=0.205)

    ys = np.arange(n_rows)[::-1]          # first table row at the top

    for y, (rk, lab) in zip(ys, ROWS):
        cr = {ak: 100.0 * D[(rk, ak)].mean() for ak, _l, _c, _m in ARMS}
        # certificate gap: from the better certificate arm to the no-CBF arm
        lo = min(cr["stage2"], cr["stage1"])
        ax.plot([lo, cr["vanilla"]], [y, y], color=C_GAP, lw=4.0,
                solid_capstyle="round", zorder=1)
        for ak, _l, col, mk in ARMS:
            k = int(D[(rk, ak)].sum())
            a, b = wilson(k, D[(rk, ak)].size)
            ax.plot([a, b], [y, y], color=col, lw=1.1, alpha=0.75, zorder=2)
            ax.plot([cr[ak]], [y], marker=mk, ms=5.6, color=col,
                    mec="white", mew=0.7, zorder=3)
        # gap magnitude, right margin -- data, not commentary
        ax.text(1.017, y, f"{cr['vanilla'] - cr['stage2']:.1f}",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8.0, color="#333333")

    ax.text(1.017, ys[0] + 0.72, "$\\Delta$ pp",
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=8.0, color="#333333")

    # alternating row bands to keep the eye on one regime
    for i, y in enumerate(ys):
        if i % 2 == 1:
            ax.axhspan(y - 0.5, y + 0.5, color="#000000", alpha=0.030,
                       zorder=0, lw=0)

    ax.set_yticks(ys)
    ax.set_yticklabels([l for _r, l in ROWS], fontsize=8.4)
    ax.set_ylim(-0.65, n_rows - 0.35)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel("conflict rate (\\%),  $n=200$ held-out encounters per cell",
                  fontsize=9)
    ax.grid(axis="x", color="#DDDDDD", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [Line2D([], [], marker=m, ls="none", ms=5.6, color=c,
                      mec="white", mew=0.7, label=l)
               for _a, l, c, m in ARMS]
    handles.append(Line2D([], [], color=C_GAP, lw=4.0,
                          label="certificate gap"))
    handles.append(Line2D([], [], color="#555555", lw=1.1,
                          label="95\\% Wilson interval"))
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.155), ncol=3, frameon=False,
              fontsize=8.2, handletextpad=0.6, columnspacing=1.5)

    for ext in ("pdf", "png"):
        p = f"{OUTD}/fig06_robustness.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
        print("saved:", p)

    g = {rk: 100.0 * D[(rk, "vanilla")].mean() - 100.0 * D[(rk, "stage2")].mean()
         for rk, _ in ROWS}
    print(f"  certificate gap: nominal={g['nominal']:.1f} pp  "
          f"combined={g['combined']:.1f} pp  "
          f"min={min(g.values()):.1f}  max={max(g.values()):.1f}")
    cs = [100.0 * D[(rk, 'stage2')].mean() for rk, _ in ROWS]
    c1 = [100.0 * D[(rk, 'stage1')].mean() for rk, _ in ROWS]
    print(f"  certificate arms differ by at most "
          f"{max(abs(a - b) for a, b in zip(cs, c1)):.1f} pp across regimes")


if __name__ == "__main__":
    main()
