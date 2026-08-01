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
        "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.7, "lines.markersize": 6.0,
        "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 600,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
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
