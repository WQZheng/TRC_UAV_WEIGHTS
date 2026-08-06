#!/usr/bin/env python3
"""Figure 13 -- corridor-scale market penetration, two demand levels.

A 2x4 grid of small multiples. Rows are demand levels, columns are metrics, and
panels are lettered (a)-(d) across the high-demand row and (e)-(h) across the
low-demand row so that the text and an appendix can cite them separately.

  columns   throughput | delay by group | conflict rate | lateral-exit share
  rows      high demand (arrival 0.16) | low demand (arrival 0.06)

WHY Y AXES ARE NOT SHARED ACROSS ROWS
  Throughput differs by a factor of 2.5 between demand levels (44.8 against
  18.7 passes/min) and the conflict rate by roughly 20 pp. A shared y axis would
  flatten the low-demand row into a horizontal line and destroy the very
  comparison the two rows exist to support. Each row sets its own limits; only
  the x axis, equipped fraction, is shared down a column (rule_7).

WHY PER-REPLICATION POINTS APPEAR ONLY FOR THE ORCA GROUP
  Six replications is few enough that a mean invites over-reading. The system
  group's dispersion is already carried by the main table's mean +/- SD, but the
  ORCA group's group-level dispersion lives in an appendix, so plotting its six
  points per penetration adds the uncertainty information the main text lacks
  rather than merely thinning ink. The system and equipped means are drawn as
  thin, low-saturation lines; the ORCA line plus points is the visual subject.

WHAT IS NOT DRAWN
  No significance marks anywhere. The mid-penetration elevation in the conflict
  rate is a descriptive pattern that six replications do not support, and the
  scatter of those replications is wider than the elevation itself -- which the
  points show without an annotation having to concede it. No timeout category in
  the completion panel: every configuration reports timeout=0, so discards are
  lateral exits, and that is asserted rather than assumed.

PROVENANCE
  penetration_v2.npz, re-run from the published sweep with per-replication
  values retained. horizon=400, warmup=100, reps=6, K=3, seed=12345,
  seed per replication = 12345 + 1000*rep + int(p*97), equipped = PlanGrad-UAV
  (stage2_final.pt + CBF-MPC), unequipped = ORCA. The re-run reproduces every
  published mean, SD and lateral-exit count in PENETRATION.txt,
  PENETRATION_LOW.txt and the DISC2 files; the exporter refuses to write
  otherwise.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

DATA = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
OUT = "/data/lab/TRC_UAV_WEIGHTS/figures_v1"
NPZ = os.environ.get("PEN_NPZ", "penetration_v2.npz")

# Equipped aircraft are PlanGrad-UAV, so they take the Stage-2 hue. ORCA is an
# unequipped comparator without a registered entity of its own; it uses the
# neutral grey that means "reference arm" everywhere in this figure family.
C_EQ = fs.color("Stage-2", family="prediction")
C_ORCA = fs.color("Constant-Velocity", family="arm")
C_ALL = fs.C_ACT
INK = fs.INK
GREY = fs.GREY

DEMANDS = [("high", "High demand", "arrival 0.16/step"),
           ("low", "Low demand", "arrival 0.06/step")]
LETTERS = {("high", 0): "(a)", ("high", 1): "(b)", ("high", 2): "(c)",
           ("high", 3): "(d)", ("low", 0): "(e)", ("low", 1): "(f)",
           ("low", 2): "(g)", ("low", 3): "(h)"}


def key(dem, p, *rest):
    return "__".join([dem, f"p{int(round(p * 100)):03d}"] + list(rest))


def load():
    d = np.load(os.path.join(DATA, NPZ), allow_pickle=True)
    ps = d["ps"]
    reps = int(d["reps"])
    assert reps == 6, reps
    assert list(ps) == [0.0, 0.25, 0.5, 0.75, 1.0], ps
    # Every discard is a lateral exit, in every configuration. The completion
    # panel is a two-way split only because of this.
    for dem, _, _ in DEMANDS:
        for p in ps:
            to = d[key(dem, p, "timeout")]
            assert to.sum() == 0, f"{dem} p={p} has {to.sum()} timeouts"
            tot = d[key(dem, p, "total")]
            assert np.allclose(d[key(dem, p, "passed")]
                               + d[key(dem, p, "lateral")], tot)
            for g in ("all", "equipped", "unequipped"):
                assert d[key(dem, p, g, "cr")].size == reps
    # The endpoints have no counterpart group by construction: nobody is
    # equipped at p=0 and nobody is unequipped at p=1. Those must be NaN, not
    # zero, or a mean line would dive to the floor at the edges.
    for dem, _, _ in DEMANDS:
        assert np.isnan(d[key(dem, 0.0, "equipped", "cr")]).all()
        assert np.isnan(d[key(dem, 1.0, "unequipped", "cr")]).all()
    return d, ps, reps


def series(d, dem, ps, group, field):
    """Per-penetration mean and the raw per-replication matrix."""
    mat = np.array([d[key(dem, p, group, field)] for p in ps], float)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(mat, axis=1)
    return mu, mat


def panel_throughput(ax, d, dem, ps):
    for grp, col, lw, lab in (("all", C_ALL, 1.9, "system"),
                              ("equipped", C_EQ, 1.2, "equipped"),
                              ("unequipped", C_ORCA, 1.2, "ORCA")):
        mu, _ = series(d, dem, ps, grp, "thr")
        ax.plot(100 * ps, mu, color=col, lw=lw, marker="o", ms=3.4,
                zorder=3, label=lab)
    ax.set_ylabel("throughput (passes/min)")


def panel_delay(ax, d, dem, ps):
    # ORCA is the subject: its points are the uncertainty information the main
    # text does not otherwise carry.
    for grp, col, lw, alpha in (("all", C_ALL, 1.1, 0.55),
                                ("equipped", C_EQ, 1.1, 0.55)):
        mu, _ = series(d, dem, ps, grp, "dly")
        ax.plot(100 * ps, mu, color=col, lw=lw, alpha=alpha, marker="o",
                ms=2.8, zorder=2)
    mu, mat = series(d, dem, ps, "unequipped", "dly")
    rng = np.random.default_rng(0)
    for i, p in enumerate(ps):
        v = mat[i]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        jit = (rng.random(v.size) - 0.5) * 5.4
        ax.plot(100 * p + jit, v, ls="none", marker="o", ms=2.6,
                mfc="none", mec=C_ORCA, mew=0.7, alpha=0.85, zorder=3)
    ax.plot(100 * ps, mu, color=C_ORCA, lw=2.1, marker="o", ms=4.0, zorder=4)
    ax.set_ylabel("delay (s)")


def panel_cr(ax, d, dem, ps):
    for grp, col, lw, alpha in (("all", C_ALL, 1.1, 0.55),
                                ("equipped", C_EQ, 1.1, 0.55)):
        mu, _ = series(d, dem, ps, grp, "cr")
        ax.plot(100 * ps, mu, color=col, lw=lw, alpha=alpha, marker="o",
                ms=2.8, zorder=2)
    mu, mat = series(d, dem, ps, "unequipped", "cr")
    rng = np.random.default_rng(1)
    for i, p in enumerate(ps):
        v = mat[i]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        jit = (rng.random(v.size) - 0.5) * 5.4
        ax.plot(100 * p + jit, v, ls="none", marker="o", ms=2.6,
                mfc="none", mec=C_ORCA, mew=0.7, alpha=0.85, zorder=3)
    ax.plot(100 * ps, mu, color=C_ORCA, lw=2.1, marker="o", ms=4.0, zorder=4)
    ax.set_ylabel("conflict rate (%)")


def panel_completion(ax, d, dem, ps):
    """Lateral-exit share of the aircraft entering the corridor.

    WHY THE COMPLETED SHARE IS NOT DRAWN ALONGSIDE
      Every configuration records timeout=0, so an aircraft either completes the
      corridor or leaves it laterally, and the two shares sum to exactly 100%.
      Plotting both would render one degree of freedom as two curves that are
      arithmetic mirrors, inviting a reader to treat them as two pieces of
      evidence. It would also invert the ink: the completed share occupies
      89-100% of the axis and the whole signal lives in the 0-11% remainder. So
      the panel draws the discard share, the quantity that actually varies, and
      the caption states that the completed share is its complement. That the
      split is two-way at all is an empirical finding, asserted in load().
    """
    lat, comp = [], []
    for p in ps:
        tot = d[key(dem, p, "total")].sum()
        lat.append(100.0 * d[key(dem, p, "lateral")].sum() / tot)
        comp.append(100.0 * d[key(dem, p, "passed")].sum() / tot)
    lat, comp = np.array(lat), np.array(comp)
    assert np.allclose(lat + comp, 100.0), "shares do not close to 100%"
    ax.plot(100 * ps, lat, color=C_ALL, lw=1.9, marker="o", ms=4.0, zorder=3)
    # Limits follow this row's own data, never the other row's (rule_7).
    top = float(np.nanmax(lat))
    ax.set_ylim(-0.45, max(top * 1.30, 1.0))
    ax.set_ylabel("lateral exit (%)")
    return lat


COLUMNS = [("throughput", panel_throughput), ("delay", panel_delay),
           ("conflict rate", panel_cr), ("lateral exit", panel_completion)]


def main():
    d, ps, reps = load()
    fs.set_rc()
    fig = plt.figure(figsize=(7.1, 4.2))
    gs = fig.add_gridspec(2, 4, left=0.082, right=0.988, top=0.905,
                          bottom=0.105, wspace=0.42, hspace=0.30)

    axes = {}
    lateral_shares = {}
    for r, (dem, row_lab, sub) in enumerate(DEMANDS):
        for c, (name, fn) in enumerate(COLUMNS):
            ax = fig.add_subplot(gs[r, c])
            axes[(dem, c)] = ax
            ret = fn(ax, d, dem, ps)
            if c == 3:
                lateral_shares[dem] = ret
            ax.set_xlim(-11, 111)
            ax.set_xticks([0, 25, 50, 75, 100])
            if r == 1:
                ax.set_xlabel("equipped fraction (%)")
            else:
                ax.set_xticklabels([])
            ax.grid(axis="y", color=fs.GRID, lw=0.55, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=6.4)
            ax.yaxis.label.set_size(6.8)
            ax.xaxis.label.set_size(6.8)
            ax.text(0.03, 0.955, LETTERS[(dem, c)], transform=ax.transAxes,
                    fontsize=7.6, fontweight="bold", va="top", color=INK)

    # Row labels live on the figure, not only in the caption, so a reader
    # scanning the grid knows which demand level a row reports without
    # leaving the image.
    row_notes = []
    for r, (dem, row_lab, sub) in enumerate(DEMANDS):
        ax0 = axes[(dem, 0)]
        bb = ax0.get_position()
        t = fig.text(0.004, bb.y1 + 0.012, f"{row_lab}  ({sub})",
                     fontsize=7.2, fontweight="bold", color=INK,
                     ha="left", va="bottom")
        row_notes.append((f"row label {dem}", t))

    handles = [
        Line2D([], [], color=C_ALL, lw=1.6, marker="o", ms=3.2,
               label="system (all aircraft)"),
        Line2D([], [], color=C_EQ, lw=1.6, marker="o", ms=3.2,
               label="equipped (PlanGrad-UAV)"),
        Line2D([], [], color=C_ORCA, lw=2.0, marker="o", ms=3.6,
               label="unequipped (ORCA)"),
        Line2D([], [], color=C_ORCA, ls="none", marker="o", ms=3.2,
               mfc="none", mew=0.7, label=f"ORCA replications ({reps} per point)"),
    ]
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, -0.004), ncol=4, fontsize=6.4,
                     frameon=False, handletextpad=0.5, columnspacing=1.5)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    # ---- self-check ---------------------------------------------------------
    # No axis may clip a drawn value, and each row sets its own limits.
    for (dem, c), ax in axes.items():
        lo, hi = ax.get_ylim()
        if c == 3:
            # The discard panel is not a group series; check it against the
            # shares it actually drew.
            lat = lateral_shares[dem]
            assert (lat >= lo - 1e-9).all() and (lat <= hi + 1e-9).all(), \
                f"panel {LETTERS[(dem, c)]} clips a lateral-exit share"
            continue
        field = {0: "thr", 1: "dly", 2: "cr"}[c]
        for grp in ("all", "equipped", "unequipped"):
            mu, mat = series(d, dem, ps, grp, field)
            vals = mat.ravel() if (c in (1, 2) and grp == "unequipped") else mu
            vals = vals[np.isfinite(vals)]
            assert (vals >= lo - 1e-9).all() and (vals <= hi + 1e-9).all(), \
                f"panel {LETTERS[(dem, c)]} clips {grp} {field}"
    for c in range(4):
        hi_lim = axes[("high", c)].get_ylim()
        lo_lim = axes[("low", c)].get_ylim()
        assert hi_lim != lo_lim, (
            f"column {c} shares a y range across demand levels, which would "
            f"flatten one of the rows")
    # The externality direction, pinned in both rows. The review found the
    # manuscript had this backwards: ORCA delay FALLS as equipage rises, so
    # equipping part of the fleet is a positive spillover onto the aircraft that
    # are not equipped, not a degradation of them. Assert the sign here so the
    # figure cannot ship carrying the inverted claim.
    PUB_ORCA_HIGH = [6.5, 5.6, 5.5, 4.4]
    for dem in ("high", "low"):
        mu_o, _ = series(d, dem, ps, "unequipped", "dly")
        f_ = mu_o[np.isfinite(mu_o)]
        assert f_[0] > f_[-1], (
            f"{dem} demand: ORCA delay does not fall with equipage "
            f"({f_[0]:.2f} -> {f_[-1]:.2f}); the positive-spillover reading "
            f"would be unsupported")
    mu_o, _ = series(d, "high", ps, "unequipped", "dly")
    finite = mu_o[np.isfinite(mu_o)]
    assert np.allclose(np.round(finite, 1), PUB_ORCA_HIGH, atol=0.06), (
        f"ORCA delay {np.round(finite, 1).tolist()} does not reproduce the "
        f"published sequence {PUB_ORCA_HIGH} quoted in the review")
    assert (np.diff(finite) <= 0).all(), "ORCA delay is not monotone"
    print("fig13 self-check: all invariants hold")
    print(f"  ORCA delay, high demand: "
          + " -> ".join(f"{v:.1f}" for v in finite) + " s")

    overlays = [(nm, t.get_window_extent(rend), np.empty((0, 2)))
                for nm, t in row_notes]
    overlays.append(("figure legend", leg.get_window_extent(rend),
                     np.empty((0, 2))))
    # Row labels and the legend sit outside every panel, so there is no curve
    # for them to cover; what must hold is that they do not cover each other or
    # any axes region.
    for nm, bb, _ in overlays:
        for (dem, c), ax in axes.items():
            ab = ax.get_window_extent()
            if (bb.x0 < ab.x1 and ab.x0 < bb.x1
                    and bb.y0 < ab.y1 and ab.y0 < bb.y1):
                raise AssertionError(f"{nm} overlaps panel {LETTERS[(dem, c)]}")
    fs.check_overlays(overlays, rend)
    fs.check_clipping(fig, list(axes.values()), rend)
    fs.check_escapes(list(axes.values()), legends=(leg,))
    fs.assert_registered(C_EQ, C_ORCA, C_ALL)
    print("colour check: every drawn colour is registered")

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"fig13_corridor.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"saved: {p}")
    print("edge ink:", fs.edge_ink(os.path.join(OUT, "fig13_corridor.png")))

    print()
    fs.report_pdf_text(os.path.join(OUT, "fig13_corridor.pdf"))

    print()
    print("caption material:")
    for dem, lab, sub in DEMANDS:
        mo, _ = series(d, dem, ps, "unequipped", "dly")
        f_ = mo[np.isfinite(mo)]
        cr, _ = series(d, dem, ps, "all", "cr")
        print(f"  {lab}: ORCA delay "
              + " -> ".join(f"{v:.1f}" for v in f_)
              + " s (positive spillover); system CR "
              + " -> ".join(f"{v:.1f}" for v in cr) + " %")
    print("  the mid-penetration elevation in the conflict rate is a "
          "descriptive pattern that the six replications do not support")
    print("  per-replication points are shown for the ORCA-controlled group "
          "only; system and equipped dispersion is tabulated")
    print("  every discarded aircraft is a lateral exit; no configuration "
          "recorded a timeout, so the completed share is the complement of "
          "the plotted discard share")
    for dem, lab, _ in DEMANDS:
        print(f"  {lab} lateral exit: "
              + " -> ".join(f"{v:.1f}" for v in lateral_shares[dem]) + " %")


if __name__ == "__main__":
    main()
