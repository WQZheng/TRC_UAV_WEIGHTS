"""figstyle.py -- shared visual encoding for every PlanGrad-UAV results figure.

Implements the Figure_plotting plan's global rules:
  * Okabe-Ito colour-blind-safe palette, one colour per METHOD.
  * line style encodes planning family: certificate-equipped = solid,
    certificate-free = dashed.
  * Conformal-MPC uses a hollow marker (planner margin differs from the line).
  * Oracle / zero-error reference is always black.
  * fixed legend order (methods absent from a figure are simply skipped,
    relative order preserved).
  * no in-figure conclusion text, no coloured "safe band", no super-title;
    panel labels (a)-(d) only, 10 pt bold, top-left.
  * 30 m separation standard drawn as a thin red dashed line where relevant.
Colours are rendering instructions only; no colour name/value is ever drawn
as text and there is no colour legend/key inside any figure.
"""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# ---- Okabe-Ito base palette -------------------------------------------------
_OI = {
    "black":       "#000000",
    "orange":      "#E69F00",
    "skyblue":     "#56B4E9",
    "green":       "#009E73",
    "yellow":      "#F0E442",
    "blue":        "#0072B2",
    "vermillion":  "#D55E00",
    "purple":      "#CC79A7",
}

# ---- per-method visual encoding --------------------------------------------
# key: canonical method name used across all figures.
STYLE = {
    "Stage 2":            dict(color=_OI["blue"],       marker="o", ls="-",  family="cert"),
    "Stage-1b":           dict(color=_OI["green"],      marker="s", ls="-",  family="cert"),
    "Stage 1":            dict(color=_OI["skyblue"],    marker="^", ls="-",  family="cert"),
    "Constant-Velocity":  dict(color=_OI["orange"],     marker="D", ls="-",  family="cert"),
    "Conformal-MPC":      dict(color=_OI["purple"],     marker="s", ls="-",  family="cert", hollow=True),
    "Vanilla-MPC":        dict(color=_OI["vermillion"], marker="v", ls="--", family="free"),
    "Soft-IPP":           dict(color=_OI["yellow"],     marker="P", ls="--", family="free"),
    "Oracle":             dict(color=_OI["black"],      marker="*", ls="-",  family="ref"),
    # alias used in some corridor/robustness panels
    "Framework-equipped": dict(color=_OI["blue"],       marker="o", ls="-",  family="cert"),
    "ORCA-controlled":    dict(color=_OI["orange"],     marker="s", ls="-",  family="free"),
    "All aircraft":       dict(color=_OI["black"],      marker="o", ls="-",  family="ref"),
}

# fixed legend order (plan section 4)
LEGEND_ORDER = ["Stage 2", "Stage-1b", "Stage 1", "Constant-Velocity",
                "Conformal-MPC", "Vanilla-MPC", "Soft-IPP", "Oracle"]

THRESH = 30.0                    # separation standard (m)
THRESH_KW = dict(color="#D55E00", ls=":", lw=1.1, zorder=1)  # thin red dashed


def set_rc():
    """Journal-consistent typography and line weights (plan section 2)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        # unified font hierarchy (v4 layout discipline):
        #   axis title 10 / ticks 8.5 / legend 8 / default (annotations) 7.5
        "font.size": 7.5,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "legend.handlelength": 1.6, "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.35, "legend.borderaxespad": 0.6,
        "lines.linewidth": 1.7, "lines.markersize": 6.0,
        "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 600,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.constrained_layout.h_pad": 0.06,
        "figure.constrained_layout.w_pad": 0.06,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def marker_kw(name, filled=True):
    """Return plot kwargs (color/marker/ls) for a method; honour hollow arms."""
    s = STYLE[name]
    kw = dict(color=s["color"], marker=s["marker"], ls=s["ls"])
    if s.get("hollow") or not filled:
        kw.update(mfc="none", mec=s["color"], mew=1.4)
    return kw


def panel_label(ax, text, x=-0.02, y=1.04):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")


def ordered(names):
    """Filter+order a set of method names by the fixed legend order."""
    present = [n for n in LEGEND_ORDER if n in names]
    present += [n for n in names if n not in LEGEND_ORDER]  # aliases at end
    return present


def find_data(*relcandidates):
    """Return the first existing path among a list of candidates, searched
    under (1) $FIGDATA_ROOT if set, (2) the repo root inferred from this file,
    (3) each candidate as given. Lets the same script run locally and on Lab.
    """
    import os
    roots = []
    if os.environ.get("FIGDATA_ROOT"):
        roots.append(os.environ["FIGDATA_ROOT"])
    here = os.path.dirname(os.path.abspath(__file__))
    roots += [os.path.join(here, ".."),                       # local project
              "/data/lab/TRC_UAV_WEIGHTS/code",               # Lab repo/code
              "/data/lab/TRC_UAV_WEIGHTS"]                    # Lab repo root
    for c in relcandidates:
        if os.path.isabs(c) and os.path.exists(c):
            return c
        for r in roots:
            p = os.path.join(r, c)
            if os.path.exists(p):
                return p
    return None


def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for a binomial proportion; returns (lo, hi) in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / d
    return (100.0 * (c - h), 100.0 * (c + h))


# =====================================================================
# v2 visual vocabulary -- reusable helpers shared by every results figure
# so the 11 figures stay stylistically consistent. All are colour-blind
# safe (Okabe-Ito), draw only real data, and add NO in-figure conclusion
# text. KDEs use scipy.stats.gaussian_kde with a light fallback.
# =====================================================================
import numpy as _np


def _kde(x, grid, bw="scott"):
    """Gaussian KDE density on `grid`; degrades gracefully for tiny/constant x."""
    x = _np.asarray(x, float)
    x = x[_np.isfinite(x)]
    if x.size < 2 or _np.allclose(x, x[0]):
        # spike at the single value (histogram-like), avoids KDE blow-up
        d = _np.zeros_like(grid)
        if x.size:
            d[_np.argmin(_np.abs(grid - x[0]))] = 1.0
        return d
    try:
        from scipy.stats import gaussian_kde
        return gaussian_kde(x, bw_method=bw)(grid)
    except Exception:
        # manual Gaussian KDE (Silverman bandwidth)
        n = x.size
        h = 1.06 * x.std(ddof=1) * n ** (-1 / 5) or 1.0
        u = (grid[:, None] - x[None, :]) / h
        return (_np.exp(-0.5 * u * u).sum(1)) / (n * h * _np.sqrt(2 * _np.pi))


def half_violin(ax, x, y0, color, width=0.8, side="up", grid=None,
                lw=0.9, alpha=0.55, zorder=3):
    """One half-violin (density ridge) for sample `x`, baseline at y0.
    side='up' fills upward (ridgeline), 'right' fills to +x (raincloud).
    Returns the density scale used, so a caller can stack consistently."""
    x = _np.asarray(x, float); x = x[_np.isfinite(x)]
    if grid is None:
        lo, hi = _np.nanmin(x), _np.nanmax(x)
        pad = 0.08 * (hi - lo + 1e-9)
        grid = _np.linspace(lo - pad, hi + pad, 256)
    d = _kde(x, grid)
    if d.max() > 0:
        d = d / d.max() * width
    if side == "up":
        ax.fill_between(grid, y0, y0 + d, color=color, alpha=alpha,
                        lw=0, zorder=zorder)
        ax.plot(grid, y0 + d, color=color, lw=lw, zorder=zorder + 1)
    else:  # 'right' -- density along +y at horizontal position y0
        ax.fill_betweenx(grid, y0, y0 + d, color=color, alpha=alpha,
                         lw=0, zorder=zorder)
        ax.plot(y0 + d, grid, color=color, lw=lw, zorder=zorder + 1)
    return grid, d


def raincloud(ax, x, y0, color, width=0.32, jitter=0.09, box=True,
              point_ms=2.4, seed=0, zorder=3):
    """Horizontal raincloud at row y0: half-violin ('cloud', above) +
    jittered raw points ('rain', below) + median dot & IQR bar.
    Encodes the full sample, not just a summary -- Nature-style honesty."""
    x = _np.asarray(x, float); x = x[_np.isfinite(x)]
    # cloud (half violin, opening upward from y0)
    lo, hi = _np.nanmin(x), _np.nanmax(x)
    pad = 0.08 * (hi - lo + 1e-9)
    grid = _np.linspace(lo - pad, hi + pad, 256)
    d = _kde(x, grid)
    if d.max() > 0:
        d = d / d.max() * width
    ax.fill_between(grid, y0, y0 + d, color=color, alpha=0.5, lw=0,
                    zorder=zorder)
    ax.plot(grid, y0 + d, color=color, lw=0.9, zorder=zorder + 1)
    # rain (jittered raw points, below the row)
    rng = _np.random.default_rng(seed)
    yj = y0 - 0.06 - rng.uniform(0, jitter, size=x.size)
    ax.plot(x, yj, ls="none", marker="o", ms=point_ms, mfc=color,
            mec="none", alpha=0.45, zorder=zorder)
    # median + IQR
    q1, med, q3 = _np.percentile(x, [25, 50, 75])
    if box:
        ax.plot([q1, q3], [y0 - 0.02, y0 - 0.02], color="0.15", lw=2.2,
                solid_capstyle="butt", zorder=zorder + 2)
    ax.plot([med], [y0 - 0.02], marker="o", ms=4.5, mfc="white",
            mec="0.1", mew=1.1, zorder=zorder + 3)
    return med, (q1, q3)


def ridgeline(ax, samples, labels, colors, gap=0.9, width=1.35, lw=1.0):
    """Stacked density ridges (joyplot). `samples` is a list of 1-D arrays,
    drawn bottom->top; returns the y baseline of each row for tick labels.
    Overlap is controlled by width>gap. Shared x across all rows."""
    ys = []
    allx = _np.concatenate([_np.asarray(s, float) for s in samples])
    allx = allx[_np.isfinite(allx)]
    lo, hi = _np.nanmin(allx), _np.nanmax(allx)
    pad = 0.06 * (hi - lo)
    grid = _np.linspace(lo - pad, hi + pad, 320)
    for i, (s, c) in enumerate(zip(samples, colors)):
        y0 = i * gap
        ys.append(y0)
        d = _kde(s, grid)
        if d.max() > 0:
            d = d / d.max() * width
        ax.fill_between(grid, y0, y0 + d, color=c, alpha=0.72, lw=0,
                        zorder=10 + i)
        ax.plot(grid, y0 + d, color="white", lw=lw + 0.4, zorder=10 + i)
        ax.plot(grid, y0 + d, color=c, lw=lw, zorder=10 + i)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.set_ylim(-0.4 * gap, (len(samples) - 1) * gap + width + 0.25)
    return ys, (grid[0], grid[-1])


def marginal_kde(ax_m, x, color, orient="x", lw=1.2, fill=True):
    """Draw a 1-D marginal density on a slim marginal axis (for scatter
    figures). orient='x' = density along the shared x (top marginal),
    'y' = along shared y (right marginal)."""
    x = _np.asarray(x, float); x = x[_np.isfinite(x)]
    lo, hi = _np.nanmin(x), _np.nanmax(x)
    pad = 0.06 * (hi - lo + 1e-9)
    grid = _np.linspace(lo - pad, hi + pad, 256)
    d = _kde(x, grid)
    if d.max() > 0:
        d = d / d.max()
    if orient == "x":
        if fill:
            ax_m.fill_between(grid, 0, d, color=color, alpha=0.35, lw=0)
        ax_m.plot(grid, d, color=color, lw=lw)
    else:
        if fill:
            ax_m.fill_betweenx(grid, 0, d, color=color, alpha=0.35, lw=0)
        ax_m.plot(d, grid, color=color, lw=lw)


def point_range(ax, i, val, lo, hi, color, marker="o", ms=8.0, horizontal=False,
                mfc=None, mec=None, mew=0.8, capsize=3, lw=1.1, zorder=4):
    """A single point estimate with an asymmetric CI whisker at slot i.
    horizontal=True lays the whisker along x (value on x-axis, category on y)."""
    if horizontal:
        ax.plot([lo, hi], [i, i], color=color, lw=lw, solid_capstyle="butt",
                zorder=zorder)
        for e in (lo, hi):
            ax.plot([e, e], [i - 0.11, i + 0.11], color=color, lw=lw, zorder=zorder)
        ax.plot([val], [i], marker=marker, ls="none", ms=ms,
                mfc=mfc or color, mec=mec or color, mew=mew, zorder=zorder + 1)
    else:
        ax.errorbar(i, val, yerr=[[val - lo], [hi - val]], capsize=capsize,
                    elinewidth=lw, ls="none", color=color, zorder=zorder)
        ax.plot([i], [val], marker=marker, ls="none", ms=ms,
                mfc=mfc or color, mec=mec or color, mew=mew, zorder=zorder + 1)


def binary_raster(ax, M, col_colors, row_order=None, cmap_bg="#EEF1F4"):
    """Draw a binary (0/1) raster: rows = episodes, cols = methods/steps.
    A 0 cell is pale background, a 1 cell is filled with the column's colour
    (so each column keeps its method identity). `M` is (n_rows, n_cols); if
    `row_order` is given, rows are permuted by it (e.g. sorted by conflict
    pattern). Returns the row order actually used."""
    M = _np.asarray(M).astype(int)
    n, m = M.shape
    if row_order is None:
        row_order = _np.arange(n)
    Mo = M[row_order]
    # background
    ax.add_patch(plt.Rectangle((0, 0), m, n, fc=cmap_bg, ec="none", zorder=0))
    for j in range(m):
        ones = _np.where(Mo[:, j] == 1)[0]
        for r in ones:
            ax.add_patch(plt.Rectangle((j, n - 1 - r), 1, 1,
                                       fc=col_colors[j], ec="white", lw=0.3,
                                       zorder=2))
    ax.set_xlim(0, m); ax.set_ylim(0, n)
    ax.set_aspect("auto")
    return row_order


def scatter_family_hull(ax, pts, color, alpha=0.10, expand=1.0):
    """Draw a faint convex-hull polygon around a cluster of 2-D points `pts`
    (list of (x,y)) to mark a method family, without any text/legend. Falls
    back to an ellipse-free bounding pad for < 3 points."""
    P = _np.asarray(pts, float)
    if P.shape[0] < 3:
        return
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(P)
        poly = P[h.vertices]
    except Exception:
        return
    c = poly.mean(0)
    poly = c + (poly - c) * expand
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, lw=0, zorder=1)


def conflict_pattern_bars(ax_bar, ax_dot, M, col_colors, col_labels,
                          max_patterns=12):
    """UpSet-style compact view of a sparse binary matrix M (n_episodes x
    n_methods): show only the conflict PATTERNS that actually occur (which
    subset of methods conflicted together on an episode), as horizontal bars of
    episode-count (ax_bar), with a membership dot-matrix on the left (ax_dot)
    marking which methods define each pattern. The all-clear pattern is
    excluded. Rows are sorted by count (largest at top). No wasted whitespace.

    Returns list of (pattern_tuple, count) drawn, most-frequent first."""
    M = _np.asarray(M).astype(int)
    n, m = M.shape
    # count distinct non-empty patterns
    keys = {}
    for r in range(n):
        pat = tuple(M[r].tolist())
        if sum(pat) == 0:
            continue
        keys[pat] = keys.get(pat, 0) + 1
    items = sorted(keys.items(), key=lambda kv: (-kv[1], -sum(kv[0])))
    items = items[:max_patterns]
    K = len(items)
    ypos = _np.arange(K)[::-1]           # largest count at top

    # membership dot-matrix (left)
    for xi in range(m):
        ax_dot.plot([xi] * K, ypos, ls="none", marker="o", ms=6.5,
                    mfc="#E3E7EC", mec="none", zorder=1)
    for row, (pat, _) in zip(ypos, items):
        members = [xi for xi in range(m) if pat[xi] == 1]
        for xi in members:
            ax_dot.plot(xi, row, ls="none", marker="o", ms=6.5,
                        mfc=col_colors[xi], mec="none", zorder=3)
        if len(members) > 1:                     # connect co-conflict members
            ax_dot.plot([min(members), max(members)], [row, row],
                        color="0.35", lw=1.1, zorder=2)
    ax_dot.set_xlim(-0.6, m - 0.4); ax_dot.set_ylim(-0.6, K - 0.4)
    ax_dot.set_xticks(range(m))
    ax_dot.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=7)
    ax_dot.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax_dot.spines[s].set_visible(False)
    ax_dot.grid(False)

    # count bars (right); bar coloured by the "lead" (first) member method
    counts = [c for _, c in items]
    barcol = [col_colors[[xi for xi in range(m) if pat[xi] == 1][0]]
              for pat, _ in items]
    ax_bar.barh(ypos, counts, height=0.6, color=barcol, lw=0, zorder=3)
    for row, c in zip(ypos, counts):
        ax_bar.annotate(str(c), (c, row), xytext=(2, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=7)
    ax_bar.set_ylim(-0.6, K - 0.4)
    ax_bar.set_yticks([])
    ax_bar.set_xlim(0, max(counts) * 1.18)
    for s in ("left", "right", "top"):
        ax_bar.spines[s].set_visible(False)
    ax_bar.grid(axis="x", color="0.9", lw=0.5)
    return items


def ribbon(ax, x, y, e, color, lw=2.0, marker="o", ms=4.5, ls="-",
           label=None, band_alpha=0.18, zorder=3):
    """A line with a filled +-e band (mean +- SD/CI). Cleaner than error bars
    for a smooth sweep; used to differentiate dashboard panels."""
    x = _np.asarray(x, float); y = _np.asarray(y, float); e = _np.asarray(e, float)
    ax.fill_between(x, y - e, y + e, color=color, alpha=band_alpha, lw=0,
                    zorder=zorder - 1)
    ax.plot(x, y, color=color, lw=lw, marker=marker, ms=ms, ls=ls,
            label=label, zorder=zorder)
