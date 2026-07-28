"""Shared publication style for all RQ figures, matching the paper's
existing TikZ palette (figures/figure1_architecture.tex, figure4_*.tex)
so the matplotlib figures are visually consistent with the diagrams.

Palette (HTML hex from the paper):
  ink     #1A1A1A   text / axes
  navy    #1E3A8A   primary accent  (our method / Stage-2)
  orange  #E8833A   secondary accent (feedback / Stage-1 contrast)
  red     #C0524A   eVTOL / danger
  grey    #B0B5BB   rules / gridlines
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

INK = "#1A1A1A"
NAVY = "#1E3A8A"
ORANGE = "#E8833A"
RED = "#C0524A"
GREY = "#B0B5BB"
LIGHTGREY = "#F2F4F7"
GREEN = "#3E7A52"
PURPLE = "#6B4E8A"
TEAL = "#2C7A8C"

# consistent method colors/markers across every figure
METHOD_STYLE = {
    "PlanGrad (ours)":   dict(color=NAVY,   marker="o", ls="-"),
    "Conformal-MPC":     dict(color=TEAL,   marker="s", ls="--"),
    "Fixed-Predictor":   dict(color=ORANGE, marker="^", ls="-."),
    "Constant-Velocity": dict(color=GREEN,  marker="D", ls=":"),
    "Vanilla-MPC":       dict(color=RED,    marker="v", ls="--"),
    "Soft-IPP":          dict(color=PURPLE, marker="P", ls=":"),
    "Stage-1":           dict(color=ORANGE, marker="^", ls="--"),
    "Stage-2":           dict(color=NAVY,   marker="o", ls="-"),
}


def apply():
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": GREY,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": GREY,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # editable text in PDF
        "ps.fonttype": 42,
    })


def save(fig, name, outdir="/data/lab/TRC-UAV/baselines/figures_out"):
    import os
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p + ".pdf")
    fig.savefig(p + ".png", dpi=150)
    plt.close(fig)
    print(f"[saved] {p}.pdf / .png")
