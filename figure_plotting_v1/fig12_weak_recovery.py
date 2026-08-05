#!/usr/bin/env python3
"""Figure 12 -- the task-aligned signal under a weak certificate.

ONE comparison, Stage-1b vs Stage-2, held fixed across TWO certificate
configurations. The certificate is what varies; the contrast is what does not.

  (a) two paired connectors on a single conflict-rate axis. The deployment
      certificate runs 11.5% -> 11.0% and is nearly horizontal; the weak
      certificate runs 56.0% -> 28.5% and drops steeply. Both endpoints carry
      Wilson intervals, and each connector is labelled with its discordant
      counts and exact McNemar p. The 44.5 pp difference in Stage-1b baseline
      is NOT hidden by plotting two risk differences side by side -- it is
      shown, because the headroom is part of the argument: a certificate that
      has already removed most conflicts leaves the signal nothing to act on.

  (b) the weak-certificate paired 2x2 as a transition matrix. Of Stage-1b's
      112 conflict episodes, 55 are resolved and 57 persist; of its 88
      conflict-free episodes, 0 become conflicts. That empty cell keeps its
      grid coordinate -- Stage-1b clear x Stage-2 conflict -- because it is the
      position of the void, not its count, that shows Stage-2's conflict set to
      be a proper subset. Every cell is labelled in place, so the panel needs
      no legend.

  (c) minimum separation under the weak certificate, three arms, mean with a
      95% interval, against the 30 m standard. Included because the conflict
      rate alone cannot distinguish "the marginal episodes were rescued" from
      "the whole distribution moved": the paired shift is +6.14 m
      [+5.45, +6.82]. The matched control's mean sits below the standard and
      Stage-2's above it.

DELIBERATE OMISSIONS
  No strong-certificate MinSep panel. That null is carried by the matched-pair
  figure and importing it would add a third data source for no new claim.
  No second transition matrix for adaptation-only. Panel (b) exists to show
  strict nesting; a second matrix would dilute it into a comparison of pairing
  structures. The adaptation contrast travels in the caption instead.
  No effect-size ladder across figures. Per the review, these deltas have
  different sample structures, baselines and pairing, and must not be ranked.

TWO DIFFERENT -0.5 pp NUMBERS
  strong-certificate S1b->S2 = -0.5 pp (b=2, c=1) is panel (a)'s upper line.
  weak-certificate   S1->S1b = -0.5 pp (b=24, c=23) is caption-only.
  Numerically identical, mechanically unrelated. The exporter keeps them under
  distinct names and this script never lets the second one reach an axis.

PROVENANCE
  weak    loose_minsep.pt, gamma=0.4 H_p=8 a_max=10, n=200, seed 12345.
  strong  conflict_vectors_v2.npz, gamma=0.1 H_p=15 a_max=20, n=200, seed 12345.
  The weak configuration is the point marked with a circled cross on the
  Figure 11 landscape; its H_p=8 lies off that H_p=15 surface, so the cell
  value there does not report these runs.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

DATA = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
OUT = "/data/lab/TRC_UAV_WEIGHTS/figures_v1"

C_S1B = fs.color("Stage-1b", family="prediction")
C_S2 = fs.color("Stage-2", family="prediction")
C_S1 = fs.color("Stage-1", family="prediction")
GREY = fs.GREY
INK = fs.INK


def load():
    d = np.load(os.path.join(DATA, "weak_recovery_v2.npz"), allow_pickle=True)
    # Re-assert the invariants the exporter established. A figure script that
    # trusts its input silently is how a stale npz reaches a page.
    assert tuple(d["table_weak"]) == (57, 55, 0, 88)
    assert tuple(d["table_strong"]) == (21, 2, 1, 176)
    assert int(d["table_weak"][2]) == 0, "weak side must be strictly nested"
    assert int(d["table_strong"][2]) > 0, "strong side is not nested"
    assert abs(float(d["delta_weak"]) + 27.5) < 1e-9
    assert abs(float(d["delta_strong"]) + 0.5) < 1e-9
    assert int(d["n"]) == 200
    return d


def panel_a(ax, d):
    """Two paired connectors on one CR axis."""
    cr_s, cr_w = d["cr_strong"], d["cr_weak"]
    wl_s, wl_w = d["wilson_strong"], d["wilson_weak"]
    x = np.array([0.0, 1.0])

    for cr, wl, col, lw in ((cr_s, wl_s, C_S2, 1.6),
                            (cr_w, wl_w, C_S2, 2.6)):
        ax.plot(x, cr, color=col, lw=lw, zorder=3, solid_capstyle="round")
        for xi, yi, (lo, hi) in zip(x, cr, wl):
            ax.plot([xi, xi], [lo, hi], color=col, lw=1.2, zorder=4,
                    solid_capstyle="butt")
            ax.plot(xi, yi, "o", color=col, ms=6.5, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.9)

    # Stage-1 under the weak certificate: carried as an unconnected reference
    # point so the three-arm decomposition is available without a fourth panel
    # or a second connector competing with the two the figure is about.
    ax.plot(-0.16, float(d["cr_weak_stage1"]), "o", color=C_S1, ms=4.2,
            zorder=5, clip_on=False)
    ax.annotate("Stage 1\n56.5", xy=(-0.16, float(d["cr_weak_stage1"])),
                xytext=(-0.30, float(d["cr_weak_stage1"]) - 7.5),
                fontsize=6.2, color=GREY, ha="center", va="top",
                linespacing=1.25, annotation_clip=False)

    a_s, b_s, c_s, _ = (int(v) for v in d["table_strong"])
    a_w, b_w, c_w, _ = (int(v) for v in d["table_weak"])

    # Both connector labels are arrowless. An annotation's window extent
    # includes its leader, so a note whose text clears the data still fails the
    # occlusion test through the arrow that by construction touches the line it
    # points at -- the first draft failed exactly that way, 28 samples covered
    # by a 56 px box holding 20 px of text. With the two connectors 45 pp apart
    # on the y axis there is no ambiguity about which label belongs to which.
    # Positions were solved on the real figure against connectors, Wilson
    # whiskers and each other, in axes fractions because that is what the API
    # consumes.
    notes = []
    n1 = ax.text(0.285, 0.165,
                 f"deployment certificate\n{float(d['delta_strong']):+.1f} pp"
                 f"   b={b_s}, c={c_s},  p = 1.0",
                 transform=ax.transAxes, fontsize=6.6, color=INK,
                 ha="left", va="bottom", linespacing=1.35)
    notes.append(("note strong", n1))

    # (0.100, 0.410) tied for nearest but sits BETWEEN the two connectors,
    # where it would read as labelling either. This candidate is the same
    # distance away and unambiguously below the steep one.
    n2 = ax.text(0.320, 0.725,
                 f"weak certificate\n{float(d['delta_weak']):+.1f} pp"
                 f"   b={b_w}, c={c_w},  p = 5.6\u00d710\u207b\u00b9\u2077",
                 transform=ax.transAxes, fontsize=6.6, color=INK,
                 ha="left", va="bottom", linespacing=1.35)
    notes.append(("note weak", n2))

    # The headroom the two baselines differ by. Drawn as a plain measured span,
    # not as a claim: the figure shows it, the text interprets it.
    ax.annotate("", xy=(-0.055, float(cr_w[0])), xytext=(-0.055, float(cr_s[0])),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=GREY,
                                shrinkA=0, shrinkB=0))
    n3 = ax.text(-0.075, float((cr_w[0] + cr_s[0]) / 2),
                 f"{float(d['baseline_gap']):.1f} pp\nheadroom",
                 fontsize=6.2, color=GREY, ha="right", va="center",
                 linespacing=1.25)
    notes.append(("note headroom", n3))

    ax.set_xlim(-0.34, 1.30)
    ax.set_ylim(0, 70)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stage-1b\n(matched control)", "Stage 2\n(task-aligned)"],
                       fontsize=7.0, linespacing=1.3)
    ax.set_ylabel("conflict rate (%)")
    ax.set_yticks([0, 20, 40, 60])
    ax.grid(axis="y", color=fs.GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.015, 0.965, "(a)", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", color=INK)
    return notes


def panel_b(ax, d):
    """Weak-certificate paired 2x2, drawn as a true transition matrix.

    The empty cell keeps its grid position. An earlier draft placed it outside
    the mosaic, which weakened the very thing the panel exists to show: it is
    the coordinate Stage-1b clear x Stage-2 conflict that is empty, and that
    emptiness is what makes Stage-2's conflict set a proper subset. Moved
    outside, "0" degrades into a footnote.

    Every cell is labelled in place, so the panel carries no legend.
    """
    a, b, c, dd = (int(v) for v in d["table_weak"])
    n = int(d["n"])
    row_conf, row_clear = a + b, c + dd
    col_conf, col_clear = a + c, b + dd
    assert (row_conf, row_clear) == (112, 88)
    assert (col_conf, col_clear) == (57, 143)
    assert a + b + c + dd == n

    # Fill encodes the transition, not the count: resolved is the Stage-2 hue,
    # conflict-under-both is the Stage-1b hue, and the unchanged-clear bulk is
    # neutral. Counts are read from the labels, never from area or saturation.
    spec = [
        (0, 1, a, "conflict\nunder both", C_S1B, "white"),
        (1, 1, b, "resolved\nby Stage 2", C_S2, "white"),
        (0, 0, c, None, None, None),                  # the empty coordinate
        (1, 0, dd, "clear\nunder both", "#E8EAEC", INK),
    ]
    for j, i, cnt, lab, fc, tc in spec:
        if fc is None:
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor="none",
                                   edgecolor=GREY, lw=0.9,
                                   linestyle=(0, (2.2, 1.8)), zorder=2))
            ax.text(j + 0.5, i + 0.56, "0", ha="center", va="center",
                    fontsize=9.5, color=GREY, zorder=3)
            ax.text(j + 0.5, i + 0.30, "newly\nintroduced", ha="center",
                    va="center", fontsize=6.0, color=GREY, linespacing=1.25,
                    zorder=3)
            continue
        ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fc, edgecolor="white",
                               lw=1.4, zorder=2))
        ax.text(j + 0.5, i + 0.58, f"{cnt}", ha="center", va="center",
                fontsize=9.5, color=tc, zorder=3)
        ax.text(j + 0.5, i + 0.31, lab, ha="center", va="center",
                fontsize=6.0, color=tc, linespacing=1.25, zorder=3)

    ax.set_xlim(-0.02, 2.02)
    ax.set_ylim(-0.02, 2.02)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels([f"conflict\n({col_conf})", f"clear\n({col_clear})"],
                       fontsize=6.8, linespacing=1.3)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels([f"clear\n({row_clear})", f"conflict\n({row_conf})"],
                       fontsize=6.8, linespacing=1.3)
    ax.set_xlabel("Stage 2", labelpad=2)
    ax.set_ylabel("Stage-1b", labelpad=2)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for s_ in ("top", "right", "bottom", "left"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.text(-0.10, 1.135, "(b)", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", color=INK)
    return None


def panel_c(ax, d):
    """Weak-certificate minimum separation, three arms."""
    m, lo, hi = d["minsep_mean"], d["minsep_lo"], d["minsep_hi"]
    d_sep = float(d["d_sep"])
    cols = [C_S1, C_S1B, C_S2]
    y = np.arange(3)[::-1]

    ax.axvline(d_sep, color=fs.GREY, lw=1.0, ls=(0, (3.2, 2.0)), zorder=1)
    ax.text(d_sep - 0.35, 2.42, f"{d_sep:.0f} m standard", fontsize=6.2,
            color=GREY, ha="right", va="center", rotation=90)

    for yi, mi, loi, hii, ci in zip(y, m, lo, hi, cols):
        ax.plot([loi, hii], [yi, yi], color=ci, lw=1.4, solid_capstyle="butt",
                zorder=3)
        ax.plot(mi, yi, "o", color=ci, ms=6.0, zorder=4,
                markeredgecolor="white", markeredgewidth=0.9)
        ax.text(hii + 0.7, yi, f"{mi:.1f}", fontsize=6.6, color=INK,
                ha="left", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(["Stage 1", "Stage-1b", "Stage 2"], fontsize=7.0)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(25.5, 37.5)
    ax.set_xticks([26, 30, 34])
    ax.set_xlabel("minimum separation (m)")
    ax.grid(axis="x", color=fs.GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.03, 0.975, "(c)", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", color=INK)


def main():
    d = load()
    fs.set_rc()
    fig = plt.figure(figsize=(7.1, 3.5))
    # top is 0.885, not 0.955: panel (b) puts its column label and tick labels
    # ABOVE the matrix, since a transition matrix reads row-then-column from the
    # top-left. Those labels need the headroom, and the clipping check caught
    # the first attempt with "Stage 2" 22 px past the canvas edge. Widening the
    # margin is the safe direction (rule_7).
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.92],
                          height_ratios=[1.0, 0.62],
                          left=0.115, right=0.985, top=0.885, bottom=0.135,
                          wspace=0.30, hspace=0.70)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    notes = panel_a(ax_a, d)
    panel_b(ax_b, d)
    panel_c(ax_c, d)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    # ---- self-check ---------------------------------------------------------
    cr_s, cr_w = d["cr_strong"], d["cr_weak"]
    assert cr_s[0] > cr_s[1] and cr_w[0] > cr_w[1]
    assert abs((cr_w[0] - cr_w[1]) / (cr_s[0] - cr_s[1])) > 50
    lo_a, hi_a = ax_a.get_ylim()
    for v in list(cr_s) + list(cr_w) + list(np.ravel(d["wilson_strong"])) \
            + list(np.ravel(d["wilson_weak"])) + [float(d["cr_weak_stage1"])]:
        assert lo_a <= v <= hi_a, f"panel (a) clips {v}"
    lo_c, hi_c = ax_c.get_xlim()
    for v in list(d["minsep_lo"]) + list(d["minsep_hi"]) + [float(d["d_sep"])]:
        assert lo_c <= v <= hi_c, f"panel (c) clips {v}"
    print("fig12 self-check: all invariants hold")

    # ---- occlusion: every overlay against panel (a)'s connectors ------------
    x = np.array([0.0, 1.0])
    pts_a = fs.curve_samples(ax_a, x, [np.asarray(cr_s, float),
                                       np.asarray(cr_w, float)])
    # Wilson whiskers are data too: a note that clears the connectors but sits
    # on an interval bar is still covering a measurement.
    whisk = []
    for xi, (lo, hi) in list(zip(x, d["wilson_strong"])) \
            + list(zip(x, d["wilson_weak"])):
        yy = np.linspace(lo, hi, 40)
        whisk.append(ax_a.transData.transform(
            np.column_stack([np.full_like(yy, xi), yy])))
    pts_a = np.vstack([pts_a] + whisk)

    overlays = [(nm, ar.get_window_extent(rend), pts_a) for nm, ar in notes]
    fs.check_overlays(overlays, rend)

    # Panel (b) carries no legend: every cell is labelled where it sits, so
    # there is nothing left for a key to say. What must be checked instead is
    # that each count and its caption stay inside their own grid cell -- a
    # number that drifts into a neighbour would silently re-attribute a
    # transition, claiming Stage 2 introduced the conflicts it resolved. Same
    # frontier as check_cell_labels guards for heatmaps, other side of it.
    fs.check_matrix_cells(ax_b, 2, 2, rend, per_cell=2)

    fs.check_clipping(fig, [ax_a, ax_b, ax_c], rend)
    fs.check_escapes([ax_a, ax_b, ax_c])

    # Free cross-validation of the strong side's lineage: its discordant pair
    # (b=2, c=1) is the same nominal pair frozen into the attribution and
    # forest figures. Agreement here means this figure really is reading the
    # v2 pipeline and not the superseded q2 vectors, which carry identical
    # values and would therefore pass every other check silently.
    assert (int(d["table_strong"][1]), int(d["table_strong"][2])) == (2, 1)
    print("lineage check: strong-side discordant pair (2,1) matches the "
          "nominal pair frozen in the attribution and forest figures, "
          "confirming the v2 conflict vectors")

    used = [C_S1, C_S1B, C_S2]
    fs.assert_registered(*used)
    print("colour check: every drawn colour is registered")

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"fig12_weak_recovery.{ext}")
        fig.savefig(p, dpi=300)
        print(f"saved: {p}")
    print("edge ink:", fs.edge_ink(os.path.join(OUT, "fig12_weak_recovery.png")))

    # Semantic pass. Geometry cannot see a label that reads wrong, only one that
    # sits wrong. Decoding is asserted clean; reading the strings is a human job.
    print()
    fs.report_pdf_text(os.path.join(OUT, "fig12_weak_recovery.pdf"))

    a_ad, b_ad, c_ad, d_ad = (int(v) for v in d["table_adaptation"])
    print()
    print("caption material:")
    print(f"  strong certificate (gamma=0.1, Hp=15, a_max=20): "
          f"{cr_s[0]:.1f}% -> {cr_s[1]:.1f}%, {float(d['delta_strong']):+.1f} pp, "
          f"b={int(d['table_strong'][1])}, c={int(d['table_strong'][2])}, p=1.0")
    print(f"  weak certificate (gamma=0.4, Hp=8, a_max=10): "
          f"{cr_w[0]:.1f}% -> {cr_w[1]:.1f}%, {float(d['delta_weak']):+.1f} pp, "
          f"b={int(d['table_weak'][1])}, c={int(d['table_weak'][2])}, p=5.6e-17")
    print(f"  approved wording: \"resolves {int(d['table_weak'][1])} of the "
          f"matched control's {int(d['table_weak'][0]) + int(d['table_weak'][1])} "
          f"conflicts and introduces none under the weak certificate\" -- the "
          f"scope qualifier is mandatory, because the strong side has "
          f"c={int(d['table_strong'][2])} and so \"introduces none\" is a "
          f"property of this configuration, not of Stage 2")
    print(f"  adaptation alone, weak certificate: {float(d['span_adaptation_only']):+.1f} pp, "
          f"b={b_ad}, c={c_ad}, p=1.0, {b_ad + c_ad} of {int(d['n'])} episodes "
          f"flipped in both directions")
    print(f"  baseline headroom at Stage-1b: {float(d['baseline_gap']):.1f} pp")
    print(f"  MinSep weak: " + ", ".join(
        f"{a}={m:.2f}" for a, m in zip(["S1", "S1b", "S2"], d["minsep_mean"]))
        + f"; paired S2-S1b {float(d['minsep_paired_delta']):+.2f} m "
          f"[{float(d['minsep_paired_lo']):+.2f}, "
          f"{float(d['minsep_paired_hi']):+.2f}]")
    print(f"  cross-reference: the configuration marked with a circled cross "
          f"on the Figure 11 landscape (Hp=8, off that Hp=15 surface)")


if __name__ == "__main__":
    main()
