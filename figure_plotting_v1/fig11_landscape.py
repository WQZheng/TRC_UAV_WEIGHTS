"""fig11 -- planner regime surface: which certificate knob governs safety.

Layout: a 4x4 discrete heatmap of conflict rate over (gamma, a_max), with the
a_max profile to its right and the gamma profile above it. No smoothing and no
contours: the grid has sixteen nodes and a continuous response surface would be
an interpolation the experiment never measured.

The two profile panels are deliberately asymmetric in what they show, because
the data are asymmetric: conflict rate falls 69.5 pp along a_max and moves at
most 10.5 pp along gamma (2.5 pp or less away from the weakest authority
column). The flat panel is the finding, not a failure of the panel.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.gridspec import GridSpec            # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs                               # noqa: E402

DATA = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
OUTD = os.environ.get("FIG_OUT_DIR",
                      "/data/lab/TRC_UAV_WEIGHTS/figures_v1")

# Expected surface, hard-coded so a regenerated npz cannot silently change the
# figure (same discipline as fig10's TAB).
EXPECT = np.array([[81.5, 30.5, 17.5, 11.0],
                   [80.0, 31.0, 18.0, 11.0],
                   [71.0, 29.0, 19.0, 12.0],
                   [72.5, 28.5, 18.5, 12.0]])


def main():
    f = f"{DATA}/landscape_v2.npz"
    if not os.path.exists(f):
        sys.exit(f"missing {f}; run export_landscape_v2.py first")
    D = np.load(f, allow_pickle=True)
    gam, amax, CR = D["gammas"], D["amaxs"], D["CR"]
    di, dj = D["deploy_ij"]
    wi, wj = D["weak_ij"]

    # ---------------------------- self-checks ------------------------------
    errs = []
    if not np.allclose(CR, EXPECT, atol=0.051):
        errs.append("surface does not match the expected 4x4 values")
    if abs(float(D["deploy_cr"]) - 11.0) > 0.051:
        errs.append(f"deployment CR {float(D['deploy_cr'])} != 11.0")
    if abs(float(D["stream_b_cr"]) - 12.0) > 0.051:
        errs.append(f"stream-B CR {float(D['stream_b_cr'])} != 12.0")
    if int(D["weak_Hp"]) == int(D["Hp_surface"]):
        errs.append("weak Hp equals surface Hp; the projection caveat is wrong")
    if float(D["span_gamma_max"]) >= float(D["span_amax"]):
        errs.append("gamma is no longer inert; profiles need redesigning")
    if errs:
        raise AssertionError("fig11 self-check failed:\n  " + "\n  ".join(errs))
    print("fig11 self-check: all invariants hold")

    fs.set_rc()
    fig = plt.figure(figsize=(7.1, 5.4))
    # Heatmap bottom-left, its two marginals sharing the corresponding axis.
    # width/height ratios give the map the dominant area; the marginals are
    # readable strips, not co-equal panels, because they summarise the map.
    # Panel (b) is given the left portion of the top row and the stream inset
    # occupies the top-right corner, which the 2x2 grid would otherwise leave
    # blank. The map keeps the dominant area.
    gs = GridSpec(2, 2, width_ratios=[1.0, 0.42], height_ratios=[0.40, 1.0],
                  wspace=0.055, hspace=0.055,
                  left=0.088, right=0.905, top=0.955, bottom=0.092)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_rgt = fig.add_subplot(gs[1, 1], sharey=ax_map)

    # ---- (a) the discrete surface -----------------------------------------
    # pcolormesh on explicit cell edges, not imshow with interpolation: every
    # cell is one measured configuration of n=200 episodes and must read as a
    # block. Edges are index-based so the unequal gamma spacing (0.1, 0.2, 0.4,
    # 0.6) does not distort cell areas into implying a sampling density.
    ny, nx = CR.shape
    mesh = ax_map.pcolormesh(np.arange(nx + 1), np.arange(ny + 1), CR,
                             cmap="magma_r", vmin=0.0, vmax=90.0,
                             edgecolors="white", linewidth=1.4, zorder=2)
    for i in range(ny):
        for j in range(nx):
            v = CR[i, j]
            # Text contrast against magma_r: dark cells (high CR) take white.
            ax_map.text(j + 0.5, i + 0.5, f"{v:.1f}", ha="center", va="center",
                        fontsize=8.6, zorder=4,
                        color="white" if v > 45.0 else fs.INK)
    ax_map.set_xticks(np.arange(nx) + 0.5)
    ax_map.set_xticklabels([f"{a:.0f}" for a in amax])
    ax_map.set_yticks(np.arange(ny) + 0.5)
    ax_map.set_yticklabels([f"{g:.1f}" for g in gam])
    ax_map.set_xlabel("control authority $a_{\\max}$ (m/s$^2$)")
    ax_map.set_ylabel("CBF decay coefficient $\\gamma$")
    ax_map.set_xlim(0, nx)
    ax_map.set_ylim(0, ny)
    ax_map.grid(False)
    ax_map.tick_params(length=0)

    # deployment node: on this surface, so it may be drawn as a cell.
    ax_map.add_patch(plt.Rectangle((dj, di), 1, 1, fill=False,
                                   ec=fs.color("Stage-2"), lw=2.6, zorder=5))
    # training / weak-certificate configuration: a PROJECTION. It ran at Hp=8
    # while this whole surface is Hp=15, so the cell underneath is not that
    # experiment's operating point. Drawn as an open dashed marker with no
    # arrow to the number, so nothing suggests the 29.0 in that cell is the
    # weak-configuration result.
    # Offset to the cell's lower-left corner rather than its centre: a glyph at
    # the centre lands exactly on that cell's own value (29.0), and the two would
    # be unreadable. The circle-and-cross reads as a coordinate datum in
    # engineering drawings -- "this is a position, not a measurement" -- which is
    # the required meaning. A slashed box was rejected: it reads as "this cell's
    # data are invalid", but the 29.0 is perfectly valid; what is invalid is only
    # reading it as the weak-configuration result.
    wx, wy = wj + 0.235, wi + 0.245
    mk_o, = ax_map.plot([wx], [wy], marker="o", ms=10.5, mfc="none",
                        mec=fs.INK, mew=1.6, ls="none", zorder=6)
    mk_x, = ax_map.plot([wx], [wy], marker="x", ms=5.4, color=fs.INK,
                        mew=1.6, ls="none", zorder=6)

    cax = fig.add_axes([0.917, 0.092, 0.020, 0.548])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("conflict rate (%)", fontsize=8.6, labelpad=2.0)
    cb.ax.tick_params(labelsize=8.0, length=2.0)
    cb.outline.set_linewidth(0.6)

    ax_map.text(-0.088, 1.005, "(a)", transform=ax_map.transAxes, ha="left",
                va="bottom", fontsize=10, fontweight="bold")

    # ---- (b) gamma profile, above ----------------------------------------
    # Four lines, one per authority column, against gamma. They are nearly
    # horizontal and that is the point; drawn on the same 0-90 scale as the
    # right-hand panel so the reader compares slopes, not scales.
    for j, a in enumerate(amax):
        ax_top.plot(np.arange(ny) + 0.5, CR[:, j], marker="o", ms=3.6,
                    lw=1.25, color=fs.GREY, zorder=3)
        # One label carries the unit and the rest give bare numbers, so the
        # reader learns the scale once without four repetitions of "m/s^2".
        # The deployment column (a_max=20) gets no emphasis of any kind: the
        # deployment anchor is the blue box in (a), and singling the line out
        # here would dilute this panel's single claim, that all four are flat.
        lab = (f"$a_{{\\max}}{{=}}${a:.0f}" + (" m/s$^2$" if j == 0 else ""))
        ax_top.annotate(lab, (ny - 0.5, CR[-1, j]), xytext=(3.5, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=6.8, color=fs.INK)
    ax_top.set_ylim(0.0, 90.0)
    ax_top.set_yticks([0, 30, 60, 90])
    ax_top.set_ylabel("CR (%)", fontsize=8.6)
    ax_top.tick_params(labelbottom=False, labelsize=8.0)
    ax_top.text(-0.088, 1.02, "(b)", transform=ax_top.transAxes, ha="left",
                va="bottom", fontsize=10, fontweight="bold")

    # ---- (c) a_max profile, right ----------------------------------------
    for i, g in enumerate(gam):
        ax_rgt.plot(CR[i, :], np.arange(nx) + 0.5, marker="o", ms=3.6,
                    lw=1.25, color=fs.GREY, zorder=3)
    ax_rgt.set_xlim(0.0, 90.0)
    ax_rgt.set_xticks([0, 30, 60, 90])
    ax_rgt.set_xlabel("CR (%)", fontsize=8.6)
    ax_rgt.tick_params(labelleft=False, labelsize=8.0)
    ax_rgt.text(0.055, 1.005, "(c)", transform=ax_rgt.transAxes, ha="left",
                va="bottom", fontsize=10, fontweight="bold")

    # ---- stream check at the deployment node -----------------------------
    # A second encounter stream (seed 999) repeated only six configurations,
    # with Hp varying across them; exactly one coincides with a node of this
    # surface. So the cross-stream evidence is one paired value, shown as two
    # points at that node, NOT a second surface or a difference map.
    cr_a = float(D["deploy_cr"])
    cr_b = float(D["stream_b_cr"])
    # The inset gets its OWN figure-level axes, in the empty band above panel
    # (c), rather than living inside a panel. An in-figure search over panel (c)
    # returned zero collision-free positions at any size worth drawing: (c) is a
    # narrow strip whose four traces cross most of its width, so there is no
    # interior gap. Three hand placements had appeared to work only because they
    # were checked against the wrong panel or measured on a throwaway figure.
    # Putting it outside every data panel means it cannot occlude anything, and
    # the occlusion check below now verifies that against both profile panels.
    ax_st = fig.add_axes([0.688, 0.762, 0.128, 0.150])
    ax_st.plot([0, 1], [cr_a, cr_b], color=fs.GREY, lw=1.0, zorder=2)
    ax_st.plot([0], [cr_a], marker="o", ms=5.0, color=fs.color("Stage-2"),
               mec="white", mew=0.8, ls="none", zorder=3)
    ax_st.plot([1], [cr_b], marker="D", ms=4.6, mfc="none",
               mec=fs.color("Stage-2"), mew=1.3, ls="none", zorder=3)
    ax_st.set_xlim(-0.55, 1.55)
    ax_st.set_ylim(min(cr_a, cr_b) - 4.0, max(cr_a, cr_b) + 4.0)
    ax_st.set_xticks([0, 1])
    ax_st.set_xticklabels(["A", "B"], fontsize=6.6)
    ax_st.set_yticks([round(min(cr_a, cr_b)), round(max(cr_a, cr_b))])
    ax_st.tick_params(labelsize=6.4, length=1.8)
    ax_st.set_title("streams, deployment node", fontsize=6.4,
                    color=fs.INK, pad=1.8)
    for s in ("top", "right"):
        ax_st.spines[s].set_visible(False)
    ax_st.grid(False)
    ax_st.set_facecolor(fs.PANEL_BG)

    for ax in (ax_top, ax_rgt):
        ax.grid(color=fs.GRID, lw=0.55, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---- legend for the two reference points -----------------------------
    leg = ax_top.legend(handles=[
        Line2D([], [], marker="s", ls="none", ms=8.0, mfc="none",
               mec=fs.color("Stage-2"), mew=2.0,
               label="deployment ($\\gamma{=}0.1$, $a_{\\max}{=}20$), on this "
                     "surface"),
        Line2D([], [], marker="o", ls="none", ms=8.0, mfc="none", mec=fs.INK,
               mew=1.5,
               label="weak-certificate ($\\gamma$, $a_{\\max}$) projection; "
                     "its runs used $H_p{=}8$")],
        # Measured placement (rule_9), in axes fraction. The gamma profiles fall
        # into two well-separated families -- the a_max=5 column at yfrac
        # 0.789-0.906 and the other three at 0.122-0.344 -- leaving 0.36-0.78
        # empty. The legend needs 0.29 of the height, so it goes in that gap and
        # nowhere else; the strip above the top family is only 0.08 tall.
        loc="upper left", bbox_to_anchor=(0.008, 0.760), frameon=False,
        fontsize=7.0, handlelength=1.2, handletextpad=0.6, labelspacing=0.34)

    # ---- verification -----------------------------------------------------
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # Panel (b) overlays against (b)'s curves, and the stream inset against
    # (c)'s, since the inset lives in (c). Checking the inset against (b) only,
    # as a first pass did, verifies nothing: it cannot collide with a panel it
    # is not in, so that test passes for the wrong reason.
    top_series = [CR[:, j] for j in range(nx)]
    s_top = fs.curve_samples(ax_top, np.arange(ny) + 0.5, top_series)
    # (c) is transposed -- value on x, index on y -- so figstyle.curve_samples,
    # which assumes y-versus-x, does not apply and the samples are built here.
    _tmp = []
    for i in range(ny):
        p = ax_rgt.transData.transform(
            np.column_stack([CR[i, :], np.arange(nx) + 0.5]))
        _tmp.append(p)
        t = np.linspace(0, 1, 90)[None, :, None]
        _tmp.append((p[:-1, None, :] * (1 - t)
                     + p[1:, None, :] * t).reshape(-1, 2))
    s_rgt = np.vstack(_tmp)
    # The stream inset now sits outside both data panels, so it is checked
    # against BOTH curve sets: an artist placed in figure coordinates has no
    # parent panel to constrain it, and testing it against only one panel is the
    # mistake that made an earlier placement look clear.
    # The inset must clear BOTH curve sets, since an artist positioned in figure
    # coordinates has no parent panel to bound it, and checking one panel only is
    # what made an earlier placement look clear. Its samples are therefore the
    # union of the two panels' -- listing the inset twice would make the
    # pair-overlap test compare it with itself and report a false collision.
    fs.check_overlays([("legend (b)", leg.get_window_extent(r), s_top),
                       ("stream inset", ax_st.get_tightbbox(r),
                        np.vstack([s_top, s_rgt]))])

    # Cell labels: correct cell, and no collision with the reference marker.
    # The earlier inline version checked placement only; a marker drawn at a cell
    # centre would have sat on that cell's number and passed.
    fs.check_cell_labels(ax_map, CR, r, extra_artists=[
        ("weak-configuration marker", mk_o.get_window_extent(r)),
        ("weak-configuration cross", mk_x.get_window_extent(r))])
    fs.check_clipping(fig, (ax_map, ax_top, ax_rgt), r)
    fs.check_escapes((ax_map, ax_top, ax_rgt, ax_st), legends=(leg,))
    fs.assert_registered(fs.color("Stage-2"), fs.GREY, fs.INK, fs.GRID,
                         fs.PANEL_BG)
    print("colour check: every drawn colour is registered")

    for ext in ("pdf", "png"):
        p = f"{OUTD}/fig11_landscape.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        print("saved:", p)
    print("edge ink:", fs.edge_ink(f"{OUTD}/fig11_landscape.png"))

    print("\ncaption material:")
    sr = D["span_rows"]
    print(f"  a_max 5->20 lowers CR by {sr.min():.1f}-{sr.max():.1f} pp by row; "
          f"{sr[0]:.1f} pp at the deployment gamma={gam[0]:.1f}")
    print(f"  gamma span within any column <= {float(D['span_gamma_max']):.1f} pp")
    print("  per-column gamma spans: " + ", ".join(
        f"a_max={a:.0f}: {s:.1f}pp"
        for a, s in zip(amax, D["span_gamma_per_col"])))
    print(f"  deployment {cr_a:.1f}% (stream A) vs {cr_b:.1f}% (stream B, "
          f"seed 999)")
    print(f"  weak-certificate projection cell reads "
          f"{float(D['weak_cr_projection']):.1f}% at Hp="
          f"{int(D['Hp_surface'])}, but those runs used Hp={int(D['weak_Hp'])}")


if __name__ == "__main__":
    main()
