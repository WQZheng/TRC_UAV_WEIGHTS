#!/usr/bin/env python3
"""Figure 10 -- robustness to plant mismatch and wind scaling.

Three panels in one full-width object.
  (a) plant-planner mismatch: horizontal dot plot, nine regimes (rows), three
      arms (Stage 1 + CBF, Stage 2 + CBF, Vanilla-MPC no-CBF); a pale grey line
      links the three points within each regime. x = CR (%) 0-100.
  (b) CR vs wind scaling eta_w, all arms with a wind sweep.
  (c) MinSep vs wind scaling eta_w, 30 m thin red dashed line.

DATA PROVENANCE (authoritative)
  Panel (a): v9 Table tab:mismatch (lines 620-628), n=200, seed 12345,
    planner nominal / plant perturbed.
  Panels (b,c): baselines/06_sim_ood/ood_results.json (n=200, seed 12345),
    etas = 0.5 / 1.0 / 1.5. NOTE: this OOD sweep carries the uniform +0.5-pt
    offset from the main table that the v9 text (lines 596-602) attributes to
    the sweep's independent wind resampling; the certificate arms are
    essentially flat and the certificate-free arms sit at 40-60%. Stage-1b has
    no wind sweep and is therefore omitted (no fabrication).
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures_generated")
os.makedirs(OUT, exist_ok=True)
OOD = fs.find_data("baselines/06_sim_ood/ood_results.json")

# panel (a): tab:mismatch. rows top->bottom per plan.
REGIMES = [
    ("Nominal",              12.5, 11.0, 41.0),
    ("Mass $-15\\%$",        10.0,  9.0, 19.5),
    ("Mass $+20\\%$",        18.5, 17.5, 75.0),
    ("Inertia $+30\\%$",     12.0, 12.0, 42.5),
    ("Thrust eff. $0.85$",   18.5, 16.5, 68.5),
    ("Delay 1 step",         22.0, 18.5, 54.0),
    ("Delay 2 steps",        34.0, 28.0, 87.5),
    ("Wind shift",           12.0, 11.0, 39.5),
    ("Combined case",        40.0, 30.0, 94.5),
]
# arm -> figstyle name for panel (a)
A_S1 = "Stage 1"; A_S2 = "Stage 2"; A_VAN = "Vanilla-MPC"

# panel (b,c) wind arm -> figstyle name mapping
WIND_MAP = {"PlanGrad (ours)": "Stage 2", "Conformal-MPC": "Conformal-MPC",
            "Fixed-Predictor": "Stage 1", "Constant-Velocity": "Constant-Velocity",
            "Vanilla-MPC": "Vanilla-MPC", "Soft-IPP": "Soft-IPP"}


def main():
    fs.set_rc()
    fig = plt.figure(figsize=(7.4, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1, 1],
                          wspace=0.42, hspace=0.5)
    axa = fig.add_subplot(gs[:, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 1])

    # ---- panel (a): mismatch dot plot --------------------------------------
    ys = np.arange(len(REGIMES))[::-1]     # first regime at top
    for y, (name, s1, s2, van) in zip(ys, REGIMES):
        axa.plot([s1, s2, van], [y, y, y], color="0.8", lw=1.0, zorder=1)
        for cr, arm in [(s1, A_S1), (s2, A_S2), (van, A_VAN)]:
            kw = fs.marker_kw(arm)
            axa.plot(cr, y, marker=kw["marker"], ls="none", ms=6.5,
                     color=kw["color"], mfc=kw.get("mfc", kw["color"]),
                     mec=kw.get("mec", kw["color"]), zorder=3)
    axa.set_yticks(ys)
    axa.set_yticklabels([r[0] for r in REGIMES])
    axa.set_xlim(0, 100); axa.set_xlabel("Conflict rate, CR (%)")
    axa.set_ylim(-0.6, len(REGIMES) - 0.4)
    fs.panel_label(axa, "(a)")
    # legend for (a) at top
    hA = [plt.Line2D([], [], color=fs.STYLE[a]["color"], marker=fs.STYLE[a]["marker"],
                     ls="none", ms=6.5, label=lab)
          for a, lab in [(A_S1, "Stage 1 + CBF"), (A_S2, "Stage 2 + CBF"),
                         (A_VAN, "Vanilla-MPC (no CBF)")]]
    axa.legend(handles=hA, loc="lower right", frameon=False, fontsize=7.5,
               handletextpad=0.3)

    # ---- panels (b,c): wind sweep ------------------------------------------
    d = json.load(open(OOD))
    etas = d["etas"]
    arms_present = []
    for raw, v in d["methods"].items():
        name = WIND_MAP.get(raw, raw)
        cr = [v["by_eta"][str(e)]["CR_%"] for e in etas]
        ms = [v["by_eta"][str(e)]["minSep_m"] for e in etas]
        kw = fs.marker_kw(name)
        axb.plot(etas, cr, marker=kw["marker"], ls=kw["ls"], color=kw["color"],
                 ms=5, mfc=kw.get("mfc", kw["color"]), mec=kw.get("mec", kw["color"]))
        axc.plot(etas, ms, marker=kw["marker"], ls=kw["ls"], color=kw["color"],
                 ms=5, mfc=kw.get("mfc", kw["color"]), mec=kw.get("mec", kw["color"]))
        arms_present.append(name)
    axb.set_ylabel("Conflict rate, CR (%)"); axb.set_ylim(0, 65)
    axb.set_xlabel(r"Wind scaling, $\eta_w$"); axb.set_xticks(etas)
    fs.panel_label(axb, "(b)")
    axc.set_ylabel("Minimum separation (m)")
    axc.axhline(fs.THRESH, **fs.THRESH_KW)
    axc.annotate("30 m", (etas[-1], fs.THRESH), textcoords="offset points",
                 xytext=(-2, 3), ha="right", fontsize=7.5, color="#D55E00")
    axc.set_xlabel(r"Wind scaling, $\eta_w$"); axc.set_xticks(etas)
    axc.set_ylim(25, 52)
    fs.panel_label(axc, "(c)")

    # shared wind legend top-centre (two rows)
    hW = [plt.Line2D([], [], color=fs.STYLE[a]["color"], marker=fs.STYLE[a]["marker"],
                     ls=fs.STYLE[a]["ls"], ms=5, label=a,
                     mfc="none" if fs.STYLE[a].get("hollow") else fs.STYLE[a]["color"])
          for a in fs.ordered(arms_present)]
    fig.legend(handles=hW, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.72, 1.02), fontsize=7.5, columnspacing=1.2,
               handletextpad=0.3)
    fig.subplots_adjust(top=0.88)
    out = os.path.join(OUT, "fig10_robustness.pdf")
    fig.savefig(out, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
