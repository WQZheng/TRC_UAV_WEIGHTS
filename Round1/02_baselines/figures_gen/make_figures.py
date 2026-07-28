"""Generate all 8 RQ figures as publication-grade PDFs, from the collected
data artifacts. Run after collect_data.py finishes. Outputs to figures_out/.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
B = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import style as st
import matplotlib.pyplot as plt

st.apply()


def jload(p):
    with open(p) as f:
        return json.load(f)


def result(folder):
    return jload(os.path.join(B, folder, "result.json"))["metrics"]


# ---- assemble main-table metrics from each baseline's result.json ----
ROWS = {
    "PlanGrad (ours)":   "00_plangrad_reference",
    "Constant-Velocity": "01_constant_velocity",
    "Vanilla-MPC":       "02_vanilla_mpc",
    "Fixed-Predictor":   "03_fixed_predictor",
    "Soft-IPP":          "04_soft_ipp",
    "Conformal-MPC":     "05_conformal_mpc",
}
M = {name: result(f) for name, f in ROWS.items()}


# =====================================================================
# FIG 1 -- ADE vs CR decoupling scatter (RQ2)
# =====================================================================
def fig1():
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for name, m in M.items():
        s = st.METHOD_STYLE[name]
        ax.scatter(m["ADE_m"], m["CR_%"], s=120, color=s["color"],
                   marker=s["marker"], edgecolor=st.INK, linewidth=0.7,
                   zorder=3, label=name)
    # annotate the two telling extremes
    ax.annotate("near-perfect ADE,\nno safety gain",
                xy=(M["Constant-Velocity"]["ADE_m"], M["Constant-Velocity"]["CR_%"]),
                xytext=(3.0, 33), fontsize=8, color=st.INK,
                arrowprops=dict(arrowstyle="->", color=st.GREY, lw=0.8))
    ax.axhspan(0, 15, color=st.NAVY, alpha=0.05, zorder=0)
    ax.text(21, 13.0, "CBF-safe band", fontsize=8, color=st.NAVY, alpha=0.8)
    ax.set_xlabel("Average displacement error, ADE (m)")
    ax.set_ylabel("Conflict rate, CR (%)")
    ax.set_title("Prediction accuracy is decoupled from operational safety")
    ax.set_ylim(0, 66)
    ax.legend(loc="center right", ncol=1)
    st.save(fig, "fig_decoupling_scatter")


# =====================================================================
# FIG 2 -- representative closed-loop rollout, Stage-1 vs Stage-2
# =====================================================================
def fig2():
    r1 = jload(os.path.join(DATA, "rollout_Fixed-Predictor.json"))
    r2 = jload(os.path.join(DATA, "rollout_PlanGrad.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))

    # (a) xy trajectories
    ax = axes[0]
    for r, lab, col in [(r1, "Stage-1 ego", st.ORANGE),
                        (r2, "Stage-2 ego (ours)", st.NAVY)]:
        e = np.array(r["ego_xy"])
        ax.plot(e[:, 0], e[:, 1], color=col, lw=2.0, label=lab,
                marker="o", ms=3)
    nb = np.array(r2["nbr_xy"])
    ax.plot(nb[:, 0], nb[:, 1], color=st.RED, lw=1.6, ls="--",
            label="neighbour", marker="x", ms=4)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("(a) Closed-loop trajectories")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")

    # (b) separation over time
    ax = axes[1]
    t1 = np.arange(len(r1["sep"])) * r1["dt"]
    t2 = np.arange(len(r2["sep"])) * r2["dt"]
    ax.plot(t1, r1["sep"], color=st.ORANGE, lw=2.0, label="Stage-1")
    ax.plot(t2, r2["sep"], color=st.NAVY, lw=2.0, label="Stage-2 (ours)")
    ax.axhline(r2["d_sep"], color=st.RED, ls=":", lw=1.5,
               label=f"separation standard ({int(r2['d_sep'])} m)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("ego-neighbour separation (m)")
    ax.set_title("(b) Separation profile")
    ax.legend(loc="best")
    fig.suptitle("Representative encounter: task alignment sharpens the avoidance",
                 fontsize=11)
    st.save(fig, "fig_rollout_compare")


# =====================================================================
# FIG 3 -- min-separation CDF across methods (RQ1)
# =====================================================================
def fig3():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    order = ["PlanGrad", "Conformal-MPC", "Fixed-Predictor",
             "Constant-Velocity", "Vanilla-MPC"]
    namemap = {"PlanGrad": "PlanGrad (ours)"}
    for tag in order:
        f = os.path.join(DATA, f"minsep_{tag}.json")
        if not os.path.exists(f):
            continue
        ms = np.sort(np.array(jload(f)["minsep"]))
        cdf = np.arange(1, len(ms) + 1) / len(ms)
        nm = namemap.get(tag, tag)
        s = st.METHOD_STYLE[nm]
        ax.plot(ms, cdf, color=s["color"], ls=s["ls"], lw=2.0, label=nm)
    ax.axvline(30, color=st.RED, ls=":", lw=1.5)
    ax.text(30.6, 0.05, "30 m standard", color=st.RED, fontsize=8, rotation=90)
    ax.set_xlabel("minimum ego-neighbour separation (m)")
    ax.set_ylabel("cumulative fraction of encounters")
    ax.set_title("Distribution of realized separation")
    ax.set_xlim(15, 75)
    ax.legend(loc="lower right")
    st.save(fig, "fig_minsep_cdf")


# =====================================================================
# FIG 4 -- prediction error vs distance-to-closest-approach (RQ5)
# =====================================================================
def fig4():
    d = jload(os.path.join(DATA, "rq5_profile.json"))
    x = np.array(d["buckets"])
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(x, d["stage1"], color=st.ORANGE, lw=2.2, marker="^",
            label="Stage-1 (ADE-trained)")
    ax.plot(x, d["stage2"], color=st.NAVY, lw=2.2, marker="o",
            label="Stage-2 (task-aligned, ours)")
    ax.axvspan(-0.3, 3.3, color=st.NAVY, alpha=0.06)
    ax.text(0.0, max(d["stage1"]) * 0.97, "operationally\ncritical band",
            fontsize=8, color=st.NAVY)
    ax.set_xlabel("|step $-$ closest-approach step|")
    ax.set_ylabel("prediction error (m)")
    ax.set_title("Where task alignment concentrates accuracy")
    ax.legend(loc="lower right")
    st.save(fig, "fig_rq5_error_profile")


# =====================================================================
# FIG 5 -- wind robustness curves (RQ4)
# =====================================================================
def fig5():
    d = jload(os.path.join(B, "06_sim_ood", "ood_results.json"))
    etas = d["etas"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for name, md in d["methods"].items():
        key = name if name in st.METHOD_STYLE else name.replace(" (ours)", "")
        s = st.METHOD_STYLE.get(name, st.METHOD_STYLE.get(key))
        crs = [md["by_eta"][str(e)]["CR_%"] for e in etas]
        sep = [md["by_eta"][str(e)]["minSep_m"] for e in etas]
        axes[0].plot(etas, crs, color=s["color"], ls=s["ls"], lw=2.0,
                     marker=s["marker"], label=name)
        axes[1].plot(etas, sep, color=s["color"], ls=s["ls"], lw=2.0,
                     marker=s["marker"], label=name)
    axes[0].set_xlabel("wind scaling $\\eta_w$")
    axes[0].set_ylabel("conflict rate, CR (%)")
    axes[0].set_title("(a) CR vs wind")
    axes[0].axvspan(0.75, 1.6, color=st.GREY, alpha=0.12)
    axes[0].text(1.05, 5, "Sim-OOD", fontsize=8, color=st.INK)
    axes[1].axhline(30, color=st.RED, ls=":", lw=1.3)
    axes[1].set_xlabel("wind scaling $\\eta_w$")
    axes[1].set_ylabel("min separation (m)")
    axes[1].set_title("(b) MinSep vs wind")
    axes[0].legend(loc="center left", fontsize=7.5)
    fig.suptitle("Safety is governed by the CBF certificate, not the wind level",
                 fontsize=11)
    st.save(fig, "fig_wind_robustness")


# =====================================================================
# FIG 6 -- planner alpha x a_max CR heatmap (RQ1/design)
# =====================================================================
def fig6():
    d = jload(os.path.join(DATA, "planner_heatmap.json"))
    cr = np.array(d["CR"])
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    im = ax.imshow(cr, cmap="YlOrRd", aspect="auto", origin="lower")
    ax.set_xticks(range(len(d["amaxs"])));
    ax.set_xticklabels([f"{a:g}" for a in d["amaxs"]])
    ax.set_yticks(range(len(d["alphas"])));
    ax.set_yticklabels([f"{a:g}" for a in d["alphas"]])
    for i in range(cr.shape[0]):
        for j in range(cr.shape[1]):
            ax.text(j, i, f"{cr[i, j]:.0f}", ha="center", va="center",
                    color=st.INK, fontsize=9)
    ax.set_xlabel("control authority $a_{\\max}$ (m/s$^2$)")
    ax.set_ylabel("CBF relaxation $\\alpha$")
    ax.set_title("Conflict rate (%) across planner settings")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("CR (%)")
    st.save(fig, "fig_planner_heatmap")


# =====================================================================
# FIG 7 -- conflict attribution stacked bars (diagnostic)
# =====================================================================
def fig7():
    d = jload(os.path.join(DATA, "attribution.json"))
    stages = ["Stage-1", "Stage-2"]
    cats = [("slack_only", "slack-active only\n(planner-limited)", st.NAVY),
            ("both", "both", st.TEAL),
            ("pred_only", "prediction-error only", st.ORANGE),
            ("neither", "neither", st.GREY)]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    bottom = np.zeros(len(stages))
    for key, lab, col in cats:
        vals = np.array([d[s][key] for s in stages], dtype=float)
        ax.bar(stages, vals, bottom=bottom, color=col, edgecolor=st.INK,
               linewidth=0.6, label=lab, width=0.55)
        bottom += vals
    for i, s in enumerate(stages):
        ax.text(i, bottom[i] + 0.3, f"{d[s]['n_conflicts']} conflicts",
                ha="center", fontsize=8, color=st.INK)
    ax.set_ylabel("number of conflicts (of 200 encounters)")
    ax.set_title("Why do conflicts happen?")
    ax.legend(loc="upper right", fontsize=7.8)
    st.save(fig, "fig_conflict_attribution")


# =====================================================================
# FIG 8 -- Energy vs CR trade-off (RQ1)
# =====================================================================
def fig8():
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for name, m in M.items():
        s = st.METHOD_STYLE[name]
        ax.scatter(m["Energy"], m["CR_%"], s=130, color=s["color"],
                   marker=s["marker"], edgecolor=st.INK, linewidth=0.7,
                   zorder=3, label=name)
    ax.axhspan(0, 15, color=st.NAVY, alpha=0.05, zorder=0)
    ax.text(20, 13.2, "CBF-safe band", fontsize=8, color=st.NAVY, alpha=0.8)
    ax.annotate("low energy but unsafe\n(weak, late maneuvers)",
                xy=(M["Vanilla-MPC"]["Energy"], M["Vanilla-MPC"]["CR_%"]),
                xytext=(22, 50), fontsize=8, color=st.INK,
                arrowprops=dict(arrowstyle="->", color=st.GREY, lw=0.8))
    ax.set_xlabel("normalized control energy")
    ax.set_ylabel("conflict rate, CR (%)")
    ax.set_title("Safety-effort trade-off")
    ax.legend(loc="center right")
    st.save(fig, "fig_energy_cr_tradeoff")


if __name__ == "__main__":
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]:
        try:
            fn(); 
        except Exception as e:
            print(f"[ERR] {fn.__name__}: {type(e).__name__}: {e}")
    print("FIGURES DONE")
