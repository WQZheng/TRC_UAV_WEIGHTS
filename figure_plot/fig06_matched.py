#!/usr/bin/env python3
"""Figure 6 -- matched comparison of displacement-only (Stage-1b) and
task-aligned (Stage 2) fine-tuning under the deployment configuration.

2x2 small multiples, Stage-1b vs Stage 2 only, identical colour/marker.
  (a) Conflict rate (%)          Wilson CI + value labels, y 5-20
  (b) Minimum separation (m)     two means, y 44-51, paired Delta annotation
  (c) Normalized control effort  mean +- SD, y 40-65, paired Delta annotation
  (d) Maximum lateral offset (m) mean +- SD, y 75-125, paired Delta annotation
Shared two-item legend, top-centre. No conclusion text inside the figure.

DATA PROVENANCE (all authoritative, v9 Table tab:stage1b, lines 703-720)
  CR 11.5 / 11.0 ; MinSep 47.8 / 47.7 ; Effort 52.9+-11.8 / 52.3+-7.6 ;
  max lateral offset 103.0+-16.8 / 103.5+-11.3.
  Paired differences (per-episode, n=200, seed 12345), from archive
  EFF_MATCHED_S1B.txt / DEV_MATCHED_S1B.txt and v9 text lines 662-665:
    MinSep  Delta = -0.04 m  95% CI [-0.35, 0.27]
    Effort  Delta = -0.55    95% CI [-1.68, 0.58]
    offset  Delta = +0.52 m  95% CI [-1.11, 2.14]
Conflict = per-episode min-sep < 30 m, n=200, eta_w=0.3.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "figures_generated")
os.makedirs(OUT, exist_ok=True)
N = 200
ARMS = ["Stage-1b", "Stage 2"]

# authoritative point estimates
CR = {"Stage-1b": 11.5, "Stage 2": 11.0}
CR_K = {"Stage-1b": 23, "Stage 2": 22}
MINSEP = {"Stage-1b": 47.8, "Stage 2": 47.7}
EFFORT = {"Stage-1b": (52.9, 11.8), "Stage 2": (52.3, 7.6)}
OFFSET = {"Stage-1b": (103.0, 16.8), "Stage 2": (103.5, 11.3)}
PAIRED = {"minsep": (-0.04, -0.35, 0.27),
          "effort": (-0.55, -1.68, 0.58),
          "offset": (+0.52, -1.11, 2.14)}


def _xpts(ax):
    ax.set_xticks([0, 1]); ax.set_xticklabels(ARMS)
    ax.set_xlim(-0.6, 1.6)


def main():
    fs.set_rc()
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.2))
    (a, b), (c, d) = axes
    xs = [0, 1]

    # (a) conflict rate with Wilson CI + labels
    for i, arm in enumerate(ARMS):
        lo, hi = fs.wilson_ci(CR_K[arm], N)
        kw = fs.marker_kw(arm)
        a.errorbar(i, CR[arm], yerr=[[CR[arm] - lo], [hi - CR[arm]]],
                   capsize=3, elinewidth=1.1, ls="none", color=kw["color"])
        a.plot(i, CR[arm], marker=kw["marker"], ls="none", ms=8,
               color=kw["color"], mfc=kw.get("mfc", kw["color"]),
               mec=kw.get("mec", kw["color"]))
        a.annotate(f"{CR[arm]:.1f}%", (i, hi), textcoords="offset points",
                   xytext=(0, 4), ha="center", fontsize=8)
    a.set_ylim(5, 20); a.set_ylabel("Conflict rate, CR (%)"); _xpts(a)
    fs.panel_label(a, "(a)")

    # (b) minimum separation
    for i, arm in enumerate(ARMS):
        kw = fs.marker_kw(arm)
        b.plot(i, MINSEP[arm], marker=kw["marker"], ls="none", ms=8,
               color=kw["color"], mfc=kw.get("mfc", kw["color"]),
               mec=kw.get("mec", kw["color"]))
    b.set_ylim(44, 51); b.set_ylabel("Minimum separation (m)"); _xpts(b)
    dm = PAIRED["minsep"]
    b.text(0.5, 0.06, f"paired $\\Delta = {dm[0]:.2f}$ m\n"
           f"95% CI $[{dm[1]:.2f}, {dm[2]:.2f}]$", transform=b.transAxes,
           ha="center", va="bottom", fontsize=8, linespacing=1.3)
    fs.panel_label(b, "(b)")

    # (c) control effort mean +- SD
    for i, arm in enumerate(ARMS):
        mu, sd = EFFORT[arm]; kw = fs.marker_kw(arm)
        c.errorbar(i, mu, yerr=sd, capsize=3, elinewidth=1.1, ls="none",
                   color=kw["color"])
        c.plot(i, mu, marker=kw["marker"], ls="none", ms=8, color=kw["color"],
               mfc=kw.get("mfc", kw["color"]), mec=kw.get("mec", kw["color"]))
    c.set_ylim(40, 65); c.set_ylabel("Normalized control effort"); _xpts(c)
    de = PAIRED["effort"]
    c.text(0.5, 0.06, f"paired $\\Delta = {de[0]:.2f}$\n"
           f"95% CI $[{de[1]:.2f}, {de[2]:.2f}]$", transform=c.transAxes,
           ha="center", va="bottom", fontsize=8, linespacing=1.3)
    fs.panel_label(c, "(c)")

    # (d) max lateral offset mean +- SD
    for i, arm in enumerate(ARMS):
        mu, sd = OFFSET[arm]; kw = fs.marker_kw(arm)
        d.errorbar(i, mu, yerr=sd, capsize=3, elinewidth=1.1, ls="none",
                   color=kw["color"])
        d.plot(i, mu, marker=kw["marker"], ls="none", ms=8, color=kw["color"],
               mfc=kw.get("mfc", kw["color"]), mec=kw.get("mec", kw["color"]))
    d.set_ylim(75, 125); d.set_ylabel("Maximum lateral offset (m)"); _xpts(d)
    do = PAIRED["offset"]
    d.text(0.5, 0.06, f"paired $\\Delta = {do[0]:+.2f}$ m\n"
           f"95% CI $[{do[1]:.2f}, {do[2]:.2f}]$", transform=d.transAxes,
           ha="center", va="bottom", fontsize=8, linespacing=1.3)
    fs.panel_label(d, "(d)")

    # shared two-item legend, top-centre
    handles = [plt.Line2D([], [], color=fs.STYLE[a]["color"],
                          marker=fs.STYLE[a]["marker"], ls="none", ms=8,
                          label=a) for a in ARMS]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.005), handletextpad=0.4,
               columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT, "fig06_matched.pdf")
    fig.savefig(out); print("wrote", out)


if __name__ == "__main__":
    main()
