#!/usr/bin/env python3
"""Figure 7 -- prediction-error structure around closest approach (CPA).

Two panels.
  (a) CPA-centred error profile: mean prediction error vs |k - k_CPA|
      (0..7), curves for Stage-1b, Stage 2, and a thin grey Stage-1 reference.
      Critical window |k-k_CPA|<=3 shaded very pale grey-blue, labelled
      "CPA-centred window" above (no "operationally critical band" text).
  (b) directional error contrast: horizontal forest plot, two rows --
      along-sight e_par (paired, Stage 2 - Stage-1b = -0.16 m
      95% CI [-0.283,-0.045]) and orthogonal |e_perp| (descriptive, hollow
      markers, no CI). Grey vertical line at x=0.

DATA PROVENANCE
  Panel (a): per-|k-k_CPA| mean error for the three learned predictors.
    Preferred source = collector fig_data/errdir_profile.npz (produced on the
    Lab at n=200, eta_w=0.3, evtol 2500-2999); if that file is absent this
    script falls back to the archived predictor-only diagnostic
    baselines/figures_gen/data/rq5_profile.json (Stage1/Stage2, 8 buckets)
    and cannot draw the Stage-1b curve -- in that case it prints a WARNING
    and the manifest records the figure as collector-dependent.
  Panel (b): v9 Table tab:errdir (lines 780-799):
    e_par mean  Stage-1b -0.18, Stage 2 -0.34 ; |e_perp| 0.68 / 2.91 ;
    paired Stage2-Stage1b = -0.16 m, 95% CI [-0.283, -0.045].
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures_generated")
os.makedirs(OUT, exist_ok=True)
NPZ = fs.find_data("fig_data/errdir_profile.npz",
                   "baselines/figures_gen/fig_data/errdir_profile.npz")

# panel (b) authoritative numbers
EPAR = {"Stage-1b": -0.18, "Stage 2": -0.34}
EPAR_SE = {"Stage-1b": 0.24, "Stage 2": 0.24}
EPERP = {"Stage-1b": 0.68, "Stage 2": 2.91}
PAIRED = (-0.16, -0.283, -0.045)   # Stage2 - Stage1b, 95% CI


def load_profile():
    """Return dict name -> (x, y) mean-error profile from the collector npz
    (the sole authoritative source; produced fresh on the Lab at n=200,
    eta_w=0.3). No stale local fallback is used."""
    if not NPZ or not os.path.exists(NPZ):
        raise SystemExit("collector errdir_profile.npz not found; run "
                         "figures_gen/collect_fig_data.py on the Lab first.")
    d = np.load(NPZ, allow_pickle=True)
    prof = {}
    for k in d.files:
        arr = np.asarray(d[k], float)
        x = np.arange(len(arr))
        label = {"Stage1": "Stage 1", "Stage-1b": "Stage-1b",
                 "Stage2": "Stage 2"}.get(k, k)
        prof[label] = (x, arr)
    return prof, "collector errdir_profile.npz (n=200, eta_w=0.3)"


def main():
    fs.set_rc()
    prof, src = load_profile()
    print("panel (a) source:", src)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1),
                               gridspec_kw=dict(width_ratios=[1.15, 1.0],
                                                wspace=0.30))

    # ---- panel (a): CPA-centred error profile ------------------------------
    a.axvspan(0, 3, color="#DCE6F1", alpha=0.7, zorder=0)   # pale grey-blue
    a.text(1.5, 0.98, "CPA-centred window", transform=a.get_xaxis_transform(),
           ha="center", va="top", fontsize=8, color="0.35")
    order = [n for n in ["Stage-1b", "Stage 2", "Stage 1"] if n in prof]
    for name in order:
        x, y = prof[name]
        if name == "Stage 1":
            a.plot(x, y, color="0.55", lw=1.0, ls="-",
                   label="Stage 1 reference", zorder=2)
        else:
            kw = fs.marker_kw(name)
            a.plot(x, y, color=kw["color"], marker=kw["marker"], ls=kw["ls"],
                   ms=4.5, label=name, zorder=3)
    a.set_xlim(0, 7); a.set_xticks(range(8))
    a.set_xlabel(r"Distance from closest-approach step, $|k-k_{\mathrm{CPA}}|$")
    a.set_ylabel("Mean prediction error (m)")
    a.legend(loc="best", frameon=False, fontsize=8)
    fs.panel_label(a, "(a)")

    # ---- panel (b): directional forest -------------------------------------
    b.axvline(0, color="0.6", lw=0.9, zorder=1)
    rows = {"Orthogonal magnitude $|e_{\\perp}|$": 1.0,
            r"Along-sight component $e_{\parallel}$": 0.0}
    # along-sight (paired, filled)
    y0 = 0.0
    for arm in ["Stage-1b", "Stage 2"]:
        kw = fs.marker_kw(arm)
        b.errorbar(EPAR[arm], y0, xerr=1.96 * EPAR_SE[arm], capsize=3,
                   elinewidth=1.1, ls="none", color=kw["color"], zorder=3)
        b.plot(EPAR[arm], y0, marker=kw["marker"], ls="none", ms=8,
               color=kw["color"], zorder=4)
        y0 += 0.0
    b.plot([EPAR["Stage-1b"], EPAR["Stage 2"]], [0, 0], color="0.5", lw=0.8,
           zorder=2)
    # orthogonal (descriptive, hollow)
    y1 = 1.0
    for arm in ["Stage-1b", "Stage 2"]:
        kw = fs.marker_kw(arm)
        b.plot(EPERP[arm], y1, marker=kw["marker"], ls="none", ms=8,
               mfc="none", mec=kw["color"], mew=1.5, zorder=4)
    b.set_yticks([0, 1])
    b.set_yticklabels([r"$e_{\parallel}$", r"$|e_{\perp}|$ (descr.)"])
    b.set_ylim(-0.6, 1.6)
    b.set_xlabel("Prediction error component (m)")
    b.set_xlim(-1.2, 3.6)
    # paired-difference annotation
    b.text(0.97, 0.02,
           f"paired $\\Delta e_\\parallel = {PAIRED[0]:.2f}$ m\n"
           f"95% CI $[{PAIRED[1]:.3f}, {PAIRED[2]:.3f}]$",
           transform=b.transAxes, ha="right", va="bottom", fontsize=7.5,
           linespacing=1.3)
    handles = [plt.Line2D([], [], color=fs.STYLE[a]["color"],
                          marker=fs.STYLE[a]["marker"], ls="none", ms=8,
                          label=a) for a in ["Stage-1b", "Stage 2"]]
    b.legend(handles=handles, loc="lower right", frameon=False, fontsize=8,
             bbox_to_anchor=(1.0, 0.16))
    fs.panel_label(b, "(b)")

    out = os.path.join(OUT, "fig07_cpa_error.pdf")
    fig.savefig(out); print("wrote", out)


if __name__ == "__main__":
    main()
