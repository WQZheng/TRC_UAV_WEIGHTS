#!/usr/bin/env python3
"""Figure 12 -- matched performance under the loosened planner configuration.

Two panels.
  (a) three-stage slope chart: Stage 1 (56.5%) -> Stage-1b (56.0%) ->
      Stage 2 (28.5%). Stage 1 -> Stage-1b drawn as a thin pale-grey line
      annotated "-0.5 pp"; Stage-1b -> Stage 2 as a thick deep-blue line
      annotated "-27.5 pp". y = CR (%) 20-65. No "planner effect" text.
  (b) paired episode transitions (stacked transition / Sankey-style) for the
      matched control -> Stage 2 on the identical stream: of Stage-1b's 112
      conflict episodes, 55 resolved (teal), 57 persist (warm grey-red), 0
      newly introduced (empty frame + "0"); of 88 non-conflicts, 88 persist
      (pale grey), 0 become conflicts. Small legend bottom-right.

DATA PROVENANCE (authoritative)
  Loosened config gamma=0.4, Hp=8, a_max=10 (v9 line 952). CR 56.5 / 56.0 /
  28.5% and 55-against-0 nested discordant pairs (exact McNemar p=5.6e-17):
  v9 text lines 950-973 and archive
  Round1/05_results/robustness/p0_referee/P0_STAGE1B_LOOSE.txt
  (n=200, seed 12345). 112 = 0.560*200 Stage-1b conflicts; 88 non-conflicts.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "figures_generated")
os.makedirs(OUT, exist_ok=True)

CR = {"Stage 1": 56.5, "Stage-1b": 56.0, "Stage 2": 28.5}
N = 200
N_CONF_S1B = 112        # 0.560 * 200
N_NONCONF = 88
RESOLVED = 55
PERSIST_CONF = 57       # 112 - 55
INTRODUCED = 0

C_RESOLVED = "#009E73"    # teal (Okabe-Ito green)
C_PERSIST = "#B0736A"     # warm grey-red
C_NONCONF = "#C9CCD1"     # pale grey


def main():
    fs.set_rc()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4),
                               gridspec_kw=dict(width_ratios=[1.0, 1.15],
                                                wspace=0.32))

    # ---- panel (a): slope chart --------------------------------------------
    xs = {"Stage 1": 0, "Stage-1b": 1, "Stage 2": 2}
    a.plot([0, 1], [CR["Stage 1"], CR["Stage-1b"]], color="0.72", lw=1.4,
           zorder=2)
    a.plot([1, 2], [CR["Stage-1b"], CR["Stage 2"]], color="#0B3D66", lw=2.6,
           zorder=2)
    for name, x in xs.items():
        kw = fs.marker_kw(name)
        a.plot(x, CR[name], marker=kw["marker"], ls="none", ms=8,
               color=kw["color"], mfc=kw.get("mfc", kw["color"]),
               mec=kw.get("mec", kw["color"]), zorder=4)
        a.annotate(f"{CR[name]:.1f}%", (x, CR[name]),
                   textcoords="offset points", xytext=(0, 8), ha="center",
                   fontsize=8)
    a.text(0.5, (CR["Stage 1"] + CR["Stage-1b"]) / 2 + 2.5, "$-0.5$ pp",
           ha="center", fontsize=8, color="0.4")
    a.text(1.5, (CR["Stage-1b"] + CR["Stage 2"]) / 2 + 1.0, "$-27.5$ pp",
           ha="center", fontsize=8.5, color="#0B3D66")
    a.set_xticks([0, 1, 2]); a.set_xticklabels(["Stage 1", "Stage-1b", "Stage 2"])
    a.set_xlim(-0.4, 2.4); a.set_ylim(20, 65)
    a.set_ylabel("Conflict rate, CR (%)")
    fs.panel_label(a, "(a)")

    # ---- panel (b): paired transitions -------------------------------------
    # Two stacked bars: left = Stage-1b state, right = Stage 2 state.
    # Represent as flows via stacked rectangles + connecting bands.
    b.set_xlim(0, 3); b.set_ylim(0, N); b.axis("off")
    xL, xR, w = 0.35, 2.05, 0.6

    # left stack: 112 conflict (top), 88 non-conflict (bottom)
    b.add_patch(Rectangle((xL, N_NONCONF), w, N_CONF_S1B, fc=C_PERSIST,
                          ec="none", alpha=0.55))
    b.add_patch(Rectangle((xL, 0), w, N_NONCONF, fc=C_NONCONF, ec="none"))
    b.text(xL + w / 2, N_NONCONF + N_CONF_S1B / 2, f"{N_CONF_S1B}\nconflict",
           ha="center", va="center", fontsize=7.5, color="0.15")
    b.text(xL + w / 2, N_NONCONF / 2, f"{N_NONCONF}\nnon-conflict",
           ha="center", va="center", fontsize=7.5, color="0.25")
    b.text(xL + w / 2, N + 4, "Stage-1b", ha="center", fontsize=8.5)

    # right stack: resolved(55) + persist-conf(57) + persist-nonconf(88)
    y = 0
    b.add_patch(Rectangle((xR, y), w, N_NONCONF, fc=C_NONCONF, ec="none"))
    y += N_NONCONF
    b.add_patch(Rectangle((xR, y), w, RESOLVED, fc=C_RESOLVED, ec="none"))
    b.text(xR + w / 2, y + RESOLVED / 2, f"{RESOLVED}\nresolved", ha="center",
           va="center", fontsize=7.5, color="white")
    y += RESOLVED
    b.add_patch(Rectangle((xR, y), w, PERSIST_CONF, fc=C_PERSIST, ec="none"))
    b.text(xR + w / 2, y + PERSIST_CONF / 2, f"{PERSIST_CONF}\npersist",
           ha="center", va="center", fontsize=7.5, color="white")
    b.text(xR + w / 2, N + 4, "Stage 2", ha="center", fontsize=8.5)

    # connecting bands (simple quad fills)
    def band(y0a, y0b, h, color, alpha):
        xs = [xL + w, xR, xR, xL + w]
        ys = [y0a + h, y0b + h, y0b, y0a]
        b.fill(xs, ys, color=color, alpha=alpha, ec="none", zorder=0)
    band(N_NONCONF, N_NONCONF, RESOLVED, C_RESOLVED, 0.35)              # resolved
    band(N_NONCONF + RESOLVED, N_NONCONF + RESOLVED, PERSIST_CONF, C_PERSIST, 0.3)
    band(0, 0, N_NONCONF, C_NONCONF, 0.5)                              # non-conf

    # "0 newly introduced" empty frame
    b.add_patch(Rectangle((xR, N - 1), w, 1.0, fc="none", ec="0.4", lw=0.8,
                          ls="--"))
    b.text(xR + w + 0.05, N - 0.5, "0 newly introduced", va="center",
           fontsize=7, color="0.35")
    fs.panel_label(b, "(b)", x=0.0, y=1.0)

    handles = [plt.Line2D([], [], marker="s", ls="none", ms=8, color=c, label=l)
               for c, l in [(C_RESOLVED, "resolved"),
                            (C_PERSIST, "persistent conflict"),
                            (C_NONCONF, "persistent non-conflict")]]
    b.legend(handles=handles, loc="lower right", frameon=False, fontsize=7,
             bbox_to_anchor=(1.02, -0.02), handletextpad=0.4)

    out = os.path.join(OUT, "fig12_loose.pdf")
    fig.savefig(out, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
