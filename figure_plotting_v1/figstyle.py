"""figstyle.py -- the colour/typography registry for the locked v1 figure series.

WHY THIS FILE EXISTS
--------------------
Style drift has now happened four times: the serif family diverged three times
across scripts, and fig10 was drafted with Stage-1 in Vanilla-MPC's vermillion
because each plotting script hard-codes its own hex literals. The registry ends
that class of bug: from fig10 onward a script asks for an ENTITY by name and
never writes a hex literal.

RELATION TO figure_plot/figstyle.py
-----------------------------------
That file is NOT this one and must not be confused with it. It serves an older,
differently-numbered series (fig11_leadtime, fig13_corridor, ...) and twelve
scripts import it. Its assignments contradict the locked series on five
entities -- it gives Soft-IPP yellow, Conformal-MPC the purple that the locked
figures give nobody, Stage 1 sky blue, Constant-Velocity orange, and the oracle
black. Editing it would break those twelve scripts, so it is left untouched.

AUTHORITY
---------
This registry is RETROSPECTIVE. The nine locked figures are the ground truth;
every value below was read out of their sources, not chosen here. The locked
figures are NOT re-rendered to match the registry -- if a value here disagrees
with a locked figure, the registry is wrong and must be corrected.

Provenance of each value, by grep of figure_plotting_v1/:
  Stage-2        #0072B2  fig02, fig03, fig05:86, fig08:62, fig09:58
  Stage-1b       #009E73  fig02, fig03, fig08:61, fig09:57
  Stage-1        #7A7A7A  fig08:63  (prediction-space family; see GREY NOTE)
  Stage-1        #E69F00  fig06 C_S1 (arm-comparison family; see GREY NOTE)
  Oracle         #8E44AD  fig05:87 "oracle replay (new series member)"
  Const-Velocity #7A7A7A  fig02, fig03  (arm-comparison family)
  Conformal-MPC  #56B4E9  fig02, fig03
  Vanilla-MPC    #D55E00  fig02, fig03, fig06 C_VAN
  Soft-IPP       #CC79A7  fig01:47, fig03:48
  zero reference #8A8A8A  fig07 C_ZERO

GREY NOTE -- a recorded, deliberately un-reopened inconsistency
---------------------------------------------------------------
#7A7A7A carries two meanings, split by figure family:
  * arm-comparison figures (fig01/02/03/06): Constant-Velocity
  * prediction-space figures (fig08/09/10):  Stage 1
#E69F00 likewise: Fixed-Predictor in most figures, Stage-1 in fig06.
Both were settled when those figures were locked. The two families never appear
in the same figure, so no reader sees a contradiction. Reopening either would
mean redrawing three already-approved figures, which is not worth it. Recorded
here so the next author does not "discover" it and start a fix.
Consequence: ask for colours through FAMILY, which resolves the grey correctly.

Soft-IPP and the oracle never share a figure either, which is why #CC79A7 could
be spent on Soft-IPP while the oracle took #8E44AD.
"""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- colours ----
# Nine entities: seven arms, the oracle replay, and the neutral zero reference.
COLORS = {
    "Stage-2":            "#0072B2",
    "Stage-1b":           "#009E73",
    "Oracle":             "#8E44AD",
    "Conformal-MPC":      "#56B4E9",
    "Vanilla-MPC":        "#D55E00",
    "Soft-IPP":           "#CC79A7",
    "Fixed-Predictor":    "#E69F00",
    "Constant-Velocity":  "#7A7A7A",
    "zero":               "#8A8A8A",
}

# Stage 1 is family-dependent; see GREY NOTE. Never read it from COLORS.
_STAGE1_BY_FAMILY = {"prediction": "#7A7A7A", "arm": "#E69F00"}

MARKERS = {"Stage-2": "o", "Stage-1b": "s", "Stage-1": "^", "Oracle": "D",
           "Constant-Velocity": "D", "Conformal-MPC": "s",
           "Vanilla-MPC": "v", "Soft-IPP": "p", "Fixed-Predictor": "^"}

# Attribution shading, shared with fig05's raster so the darker tone means
# actuation-limited in every figure that splits conflicts by mechanism.
C_ACT = "#2F4B6E"
C_PRED = "#9EB8D4"

GREY = "#8A8A8A"          # neutral zero line / neutral regime band
INK = "#333333"           # all annotation and label text (fig05 C_INK)
GRID = "#EDEDED"          # gridlines
PANEL_BG = "#FDFDFD"      # inset panel fill
THRESH = 30.0             # separation standard (m)


def color(entity, family="prediction"):
    """Hex for `entity`. Stage 1 needs `family` ('prediction' or 'arm') because
    grey is family-split; every other entity ignores it."""
    if entity in ("Stage-1", "Stage 1", "Stage1"):
        if family not in _STAGE1_BY_FAMILY:
            raise KeyError(f"Stage-1 needs family in "
                           f"{sorted(_STAGE1_BY_FAMILY)}, got {family!r}")
        return _STAGE1_BY_FAMILY[family]
    if entity not in COLORS:
        raise KeyError(f"{entity!r} is not a registered entity; "
                       f"known: {sorted(COLORS)} plus Stage-1")
    return COLORS[entity]


def set_rc():
    """The one typography definition. Three scripts previously disagreed on the
    serif family, which is why this is not repeated per figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for a binomial proportion, in percent. z=1.96 is fixed
    here so no figure can quietly use a different critical value."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / d
    return (100.0 * (c - h), 100.0 * (c + h))


# ------------------------------------------------------- geometric checks ----
# Pixel-colour occlusion testing does not work for these figures: a curve can
# span the full panel width, so a colour probe cannot tell "hidden under the
# legend" from "same pixel row, elsewhere" (rule_6). Everything below compares
# artist bounding boxes against densely sampled curve vertices in DISPLAY
# coordinates instead.

def curve_samples(ax, xs, series, n_per_segment=90):
    """Dense display-coordinate samples along each curve in `series` (a list of
    y-arrays sharing abscissa `xs`). Vertices alone are not enough: a legend can
    sit over the middle of a long segment and touch no vertex."""
    import numpy as np
    pts = []
    for y in series:
        p = ax.transData.transform(np.column_stack([xs, y]))
        pts.append(p)
        t = np.linspace(0, 1, n_per_segment)[None, :, None]
        seg = p[:-1, None, :] * (1 - t) + p[1:, None, :] * t
        pts.append(seg.reshape(-1, 2))
    return np.vstack(pts)


def check_overlays(overlays, renderer=None):
    """Assert no overlay covers data, and no overlay covers another overlay.

    `overlays` is a list of (name, bbox, samples) triples. The second test
    exists because it caught two real collisions in fig10 that the data test
    could only report indirectly, as an inflated curve-hit count.

    Note when supplying bboxes: an inset's tightbbox extends about 0.056 panel
    heights BELOW its axes frame (y label plus tick text), and an annotation's
    window extent includes its arrow, so a note whose text is clear can still
    fail through its leader.
    """
    msgs = []
    for name, bb, pts in overlays:
        hit = ((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1)
               & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1))
        print(f"  overlay '{name}': x [{bb.x0:.0f},{bb.x1:.0f}] "
              f"y [{bb.y0:.0f},{bb.y1:.0f}]  hits={int(hit.sum())}")
        if hit.any():
            msgs.append(f"{name} covers {int(hit.sum())} curve samples")
    for i in range(len(overlays)):
        for j in range(i + 1, len(overlays)):
            n1, b1, _ = overlays[i]
            n2, b2, _ = overlays[j]
            if (b1.x0 < b2.x1 and b2.x0 < b1.x1
                    and b1.y0 < b2.y1 and b2.y0 < b1.y1):
                msgs.append(f"{n1} overlaps {n2}")
    if msgs:
        raise AssertionError("occlusion check failed:\n  " + "\n  ".join(msgs))
    print("occlusion check: no overlay covers any curve or another overlay")


def check_clipping(fig, axes, renderer):
    """Assert every text artist lies inside the canvas (rule_7). Named artists,
    not an edge-pixel count: fig10 lost a regime label and part of a note off
    the right edge, and the pixel probe could only say '105 dark pixels'."""
    W, H = fig.canvas.get_width_height()
    bad = []
    for ax in axes:
        for t in list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]:
            if not t.get_text():
                continue
            bb = t.get_window_extent(renderer)
            if bb.x1 > W or bb.x0 < 0 or bb.y1 > H or bb.y0 < 0:
                bad.append(f"{t.get_text()[:38]!r} at "
                           f"x[{bb.x0:.0f},{bb.x1:.0f}] "
                           f"y[{bb.y0:.0f},{bb.y1:.0f}] vs canvas {W}x{H}")
    if bad:
        raise AssertionError("clipping check failed:\n  " + "\n  ".join(bad))
    print("clipping check: every text artist is inside the canvas")


def check_cell_labels(ax, values, renderer, extra_artists=()):
    """Assert each cell label of a heatmap sits inside its own cell, and that no
    label collides with any other in-cell artist.

    A value attributed to the wrong configuration is the most damaging error a
    heatmap can carry and the hardest for a reader to notice -- nothing about a
    misplaced number looks wrong. Neither the overlay test nor the clipping test
    covers it: cell labels are neither overlays nor curves.

    `values` is the (ny, nx) array, indexed so that value[i, j] is drawn centred
    in the cell spanning x in [j, j+1] and y in [i, i+1] in data coordinates.
    `extra_artists` are other things drawn inside cells -- reference markers, for
    instance -- which must not sit on top of a number. Checking placement alone
    is not enough: a marker at a cell's centre lands exactly on its label.

    Returns the number of labels verified.
    """
    import numpy as np
    values = np.asarray(values, float)
    ny, nx = values.shape
    msgs = []
    boxes = {}
    for i in range(ny):
        for j in range(nx):
            want = ax.transData.transform([[j, i], [j + 1, i + 1]])
            target = f"{values[i, j]:.1f}"
            for t in ax.texts:
                if t.get_text() != target:
                    continue
                bb = t.get_window_extent(renderer)
                cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
                if not (want[0, 0] <= cx <= want[1, 0]
                        and want[0, 1] <= cy <= want[1, 1]):
                    continue                      # a like-valued label elsewhere
                if (bb.x0 < want[0, 0] or bb.x1 > want[1, 0]
                        or bb.y0 < want[0, 1] or bb.y1 > want[1, 1]):
                    msgs.append(f"cell ({i},{j}) label {target} overflows "
                                f"its cell")
                boxes[(i, j)] = bb
    if len(boxes) != ny * nx:
        missing = [(i, j) for i in range(ny) for j in range(nx)
                   if (i, j) not in boxes]
        msgs.append(f"no label found inside cells {missing}")
    for name, bb2 in extra_artists:
        for (i, j), bb1 in boxes.items():
            if (bb1.x0 < bb2.x1 and bb2.x0 < bb1.x1
                    and bb1.y0 < bb2.y1 and bb2.y0 < bb1.y1):
                msgs.append(f"{name} overlaps the label of cell ({i},{j})")
    if msgs:
        raise AssertionError("cell-label check failed:\n  " + "\n  ".join(msgs))
    print(f"cell-label check: all {ny * nx} values inside their own cell, "
          f"no collision with {len(extra_artists)} in-cell artist(s)")
    return len(boxes)


def check_escapes(axes, legends=()):
    """Assert no LaTeX escape reaches a drawn label. usetex is off, so '\\%' and
    '\\,' render as literal backslashes; fig10 shipped 'conflict rate (\\%)' onto
    the canvas and it was only caught by extracting the PDF text layer. Use a
    plain '%' and U+2009 for a thin space."""
    stray = []
    for ax in axes:
        arts = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label] \
            + list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        for t in arts:
            s = t.get_text()
            if "\\%" in s or "\\," in s:
                stray.append(repr(s[:48]))
    for lg in legends:
        for t in lg.get_texts():
            s = t.get_text()
            if "\\%" in s or "\\," in s:
                stray.append(repr(s[:48]))
    if stray:
        raise AssertionError("escape check failed: usetex is off, these would "
                            "render literally:\n  " + "\n  ".join(stray))
    print("escape check: no LaTeX escapes in any drawn label")


def edge_ink(png_path, thresh=720, depth=2):
    """Independent post-hoc confirmation: count non-white pixels in the outer
    `depth` rows/columns of the saved raster. Complements check_clipping, which
    only sees artists it was handed."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(png_path).convert("RGB"))
    nw = a.astype(int).sum(-1) < thresh
    return {"top": int(nw[:depth, :].sum()), "bottom": int(nw[-depth:, :].sum()),
            "left": int(nw[:, :depth].sum()), "right": int(nw[:, -depth:].sum())}


def assert_registered(*hexes):
    """Guard against a stray literal: every colour a figure draws with must be a
    registered value. Catches exactly the fig10 bug, where Stage-1 was given
    Vanilla-MPC's vermillion."""
    known = set(COLORS.values()) | set(_STAGE1_BY_FAMILY.values()) \
        | {C_ACT, C_PRED, GREY, INK, GRID, PANEL_BG, "#FFFFFF"}
    unknown = [h for h in hexes if h.upper() not in {k.upper() for k in known}]
    if unknown:
        raise AssertionError(
            f"unregistered colours: {unknown}. Add them to figstyle.COLORS "
            f"with provenance, or use a registered entity.")
