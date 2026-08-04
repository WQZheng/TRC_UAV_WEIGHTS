#!/usr/bin/env python3
"""Fig. 5 -- Residual-conflict structure and attribution.

Three panels, each answering a different question about the 22 residual
conflicts of the PlanGrad (Stage-2) arm:

  (a) Are conflicts shared across planners or arm-specific?
      Membership matrix over the four common-planner arms (exact enumeration
      of the 4 non-empty patterns; 21 of the 25 union episodes are shared by
      all four arms).

  (b) When, and how badly, is the program infeasible?
      22 x 20 raster over rollout steps. Cells are graded by the MINIMUM CBF
      relaxation the step requires: feasible / <1 m / 1-10 m / >10 m. Row
      marginal = infeasible-step count; column marginal = discrete per-step
      counts (n=22). A CPA marker sits on each row.

  (c) Would a perfect predictor have helped?
      Violation depth (30 - minSep) for Stage-2 vs an oracle-predictor replay,
      one row per episode. The zero line is the rescue line: no episode
      reaches it under either predictor.

METHOD NOTE (why the raster is solver-independent)
  Feasibility is NOT read off a solver status flag. For every (episode, step)
  we solve the always-feasible program that minimises the total CBF relaxation
  and record its magnitude; a step counts as hard-infeasible when the required
  relaxation exceeds 1 mm. The count is invariant for thresholds spanning
  1e-6 to 1e-2 m (414 cells throughout), because the distribution is bimodal:
  26 cells at ~1e-14 m (genuinely feasible) and 414 cells with median 7.6 m.
  The grey band between 1e-6 and 1e-2 m is empty. The earlier zero-slack
  status test reported 391 cells; the 23 extra cells carry relaxations of
  2.3 cm to 26.0 m (8 of them above 10 m), i.e. they were ECOS false
  negatives under 'optimal_inaccurate', not boundary jitter.

DATA (all rows aligned by episode_ids; see PROVENANCE_v2.json)
  conflict_vectors_v2.npz : four 200-dim bool vectors, derived from the
                            per-episode arrays of eval_common.evaluate_policy
                            (same rollout as the main table).
  attribution_v2.npz      : episode_ids(22), infeasible(22,20),
                            min_slack(22,20), cpa_step(22).
  oracle_v2.npz           : episode_ids(22), minsep_stage2(22),
                            minsep_oracle(22), oracle_slack_at_cpa(22),
                            oracle_conflict_200(200).

SELF-CHECKS (hard-fail; see verify() at the bottom)
  four arm vector means == table CR; discordant vs PlanGrad keyed BY ARM
  {Stage-1b:(2,1), Fixed-Predictor:(3,0), Constant-Velocity:(2,0)};
  infeasible.sum() == 414; (min_slack > 1e-3) == infeasible elementwise;
  episode_ids identical across attribution/oracle and equal to the True
  indices of the PlanGrad vector; oracle_slack_at_cpa has exactly 2 entries
  below 1e-6 (epi 117, 176); all oracle depths > 0; oracle_conflict_200 vs
  PlanGrad discordant == (2,0).

No in-figure title or conclusion text; only (a)-(d) style panel letters.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
DATA_DIR = os.environ.get(
    "FIG_DATA_DIR",
    os.path.join(_REPO, "code", "baselines", "figures_gen", "fig_data"))
OUT_DIR = os.environ.get("FIG_OUT_DIR", os.path.join(_REPO, "figures_v1"))

D_SEP = 30.0
EPS_M = 1e-3          # physical hard-infeasibility threshold (1 mm)
SLACK_EDGES = [1.0, 10.0]   # grading breakpoints: 1 m, 10 m (~1/3 d_sep)

ARMS4 = ["PlanGrad", "Stage-1b", "Fixed-Predictor", "Constant-Velocity"]
ARM_DISP = {"PlanGrad": "Stage-2 (ours)", "Stage-1b": "Stage-1b",
            "Fixed-Predictor": "Fixed-Predictor",
            "Constant-Velocity": "Constant-Velocity"}
TABLE_CR = {"PlanGrad": 11.0, "Stage-1b": 11.5,
            "Fixed-Predictor": 12.5, "Constant-Velocity": 12.0}
EXPECT_DISC = {"Stage-1b": (2, 1), "Fixed-Predictor": (3, 0),
               "Constant-Velocity": (2, 0)}

C_S2 = "#0072B2"       # Stage-2, consistent with Figs. 1-3
C_ORACLE = "#8E44AD"   # oracle replay (new series member)
C_INK = "#333333"
GRADE_COLORS = ["#FFFFFF", "#DCE6F1", "#7FA8CE", "#1F4E79"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

cv = np.load(os.path.join(DATA_DIR, "conflict_vectors_v2.npz"))
at = np.load(os.path.join(DATA_DIR, "attribution_v2.npz"))
oc = np.load(os.path.join(DATA_DIR, "oracle_v2.npz"))

EID = at["episode_ids"]
INF = at["infeasible"].astype(bool)
SLK = at["min_slack"]
CPA = at["cpa_step"]
N_EP, N_T = INF.shape


def verify():
    """Refuse to plot unless every provenance invariant holds."""
    fail = []
    for a, cr in TABLE_CR.items():
        got = 100.0 * cv[a].mean()
        if abs(got - cr) > 1e-9:
            fail.append(f"{a} vector mean {got} != table CR {cr}")
    x = cv["PlanGrad"].astype(bool)
    for a, (eb, ec) in EXPECT_DISC.items():
        y = cv[a].astype(bool)
        b, c = int((~x & y).sum()), int((x & ~y).sum())
        if (b, c) != (eb, ec):
            fail.append(f"discordant PlanGrad vs {a} = ({b},{c}) != ({eb},{ec})")
    if int(INF.sum()) != 414:
        fail.append(f"infeasible.sum() = {int(INF.sum())} != 414")
    if not ((SLK > EPS_M) == INF).all():
        fail.append("(min_slack > 1e-3) != infeasible elementwise")
    if not (np.array_equal(EID, oc["episode_ids"])
            and np.array_equal(EID, np.flatnonzero(x))):
        fail.append("episode_ids not aligned across the three files")
    zero_slack = EID[oc["oracle_slack_at_cpa"] < 1e-6]
    if list(zero_slack) != [117, 176]:
        fail.append(f"oracle zero-slack episodes {list(zero_slack)} != [117, 176]")
    if not ((D_SEP - oc["minsep_oracle"]) > 0).all():
        fail.append("some oracle depth <= 0 (an episode was rescued)")
    o200 = oc["oracle_conflict_200"].astype(bool)
    b, c = int((~x & o200).sum()), int((x & ~o200).sum())
    if (b, c) != (2, 0):
        fail.append(f"oracle_200 vs PlanGrad discordant ({b},{c}) != (2,0)")
    if INF.shape[0] != 22 or int(x.sum()) != 22:
        fail.append("raster rows != 22 != Stage-2 conflict count")
    if fail:
        raise SystemExit("PROVENANCE SELF-CHECK FAILED:\n  - "
                         + "\n  - ".join(fail))
    print("provenance self-check: all invariants hold")


verify()

# ---------------------------------------------------------------- layout
fig = plt.figure(figsize=(7.4, 8.6))
gs = fig.add_gridspec(
    3, 2, height_ratios=[0.72, 1.55, 1.20], width_ratios=[1.0, 0.30],
    left=0.145, right=0.965, top=0.975, bottom=0.062,
    hspace=0.40, wspace=0.045)

ax_a = fig.add_subplot(gs[0, :])
ax_col = fig.add_subplot(gs[1, 0])      # column marginal (discrete counts)
ax_b = None                             # raster, created below sharing x
ax_row = fig.add_subplot(gs[1, 1])      # row marginal
ax_c = fig.add_subplot(gs[2, :])


def plab(ax, s, x=-0.085, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom", ha="left")


# ============================================================ panel (a)
M = np.stack([cv[a].astype(bool) for a in ARMS4])
pats = {}
for j in range(M.shape[1]):
    key = tuple(M[:, j])
    if any(key):
        pats[key] = pats.get(key, 0) + 1
order = sorted(pats.items(), key=lambda kv: -kv[1])

nP = len(order)
xs = np.arange(nP)
row_y = {a: len(ARMS4) - 1 - i for i, a in enumerate(ARMS4)}

# membership dots
for j, (key, cnt) in enumerate(order):
    ys = [row_y[a] for a, b in zip(ARMS4, key) if b]
    ax_a.plot([j, j], [min(ys), max(ys)], color=C_INK, lw=1.4, zorder=2,
              solid_capstyle="round")
    for a, b in zip(ARMS4, key):
        ax_a.plot(j, row_y[a], "o", ms=7.0, zorder=3,
                  color=C_INK if b else "#FFFFFF",
                  markeredgecolor=C_INK if b else "#BBBBBB",
                  markeredgewidth=0.9)
    ax_a.text(j, len(ARMS4) - 0.32, str(cnt), ha="center", va="bottom",
              fontsize=9, fontweight="bold", color=C_INK)

for a in ARMS4:
    ax_a.axhline(row_y[a], color="#EEEEEE", lw=6.0, zorder=0)
ax_a.set_yticks([row_y[a] for a in ARMS4])
ax_a.set_yticklabels([ARM_DISP[a] for a in ARMS4], fontsize=8)
ax_a.set_xticks(xs)
ax_a.set_xticklabels(["all four", "Stage-2\nsafe", "Fixed-Pred.\nonly",
                      "Stage-1b\nsafe"], fontsize=7.5)
ax_a.set_xlim(-0.6, nP - 0.4)
ax_a.set_ylim(-0.6, len(ARMS4) + 0.10)
ax_a.set_ylabel("conflict pattern", labelpad=6)
ax_a.tick_params(axis="both", length=0)
for sp in ("left", "bottom"):
    ax_a.spines[sp].set_visible(False)
ax_a.text(0.995, 0.04, f"episodes with $\\geq$1 conflict: {int(M.any(0).sum())}",
          transform=ax_a.transAxes, ha="right", va="bottom", fontsize=7.5,
          color="#666666")
plab(ax_a, "(a)")

# ============================================================ panel (b)
ax_b = fig.add_subplot(gs[1, 0], sharex=None)
ax_col.remove()          # rebuilt as an inset strip above the raster
order_rows = np.argsort(-INF.sum(1), kind="stable")

grade = np.zeros_like(SLK, dtype=int)
grade[(SLK > EPS_M) & (SLK < SLACK_EDGES[0])] = 1
grade[(SLK >= SLACK_EDGES[0]) & (SLK < SLACK_EDGES[1])] = 2
grade[SLK >= SLACK_EDGES[1]] = 3

cmap = ListedColormap(GRADE_COLORS)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
ax_b.imshow(grade[order_rows], aspect="auto", cmap=cmap, norm=norm,
            interpolation="nearest", origin="upper",
            extent=[-0.5, N_T - 0.5, N_EP - 0.5, -0.5], zorder=1)

for gi, ri in enumerate(order_rows):
    ax_b.add_patch(Rectangle((CPA[ri] - 0.5, gi - 0.5), 1.0, 1.0,
                             fill=False, edgecolor="#000000", lw=1.25,
                             zorder=4))
for k in range(N_T + 1):
    ax_b.axvline(k - 0.5, color="#FFFFFF", lw=0.45, zorder=3)
for k in range(N_EP + 1):
    ax_b.axhline(k - 0.5, color="#FFFFFF", lw=0.45, zorder=3)

ax_b.set_xticks(range(0, N_T, 2))
ax_b.set_xticklabels(range(0, N_T, 2))
ax_b.set_yticks(range(N_EP))
ax_b.set_yticklabels([str(EID[i]) for i in order_rows], fontsize=6.2)
ax_b.set_xlabel("rollout step")
ax_b.set_ylabel("conflict episode (index in the 200-encounter set)", labelpad=4)
ax_b.set_xlim(-0.5, N_T - 0.5)
ax_b.set_ylim(N_EP - 0.5, -0.5)
for sp in ax_b.spines.values():
    sp.set_visible(True)
    sp.set_edgecolor("#999999")
plab(ax_b, "(b)")

leg_b = [
    Line2D([0], [0], marker="s", ls="none", ms=8, markerfacecolor=GRADE_COLORS[0],
           markeredgecolor="#999999", label="feasible"),
    Line2D([0], [0], marker="s", ls="none", ms=8, markerfacecolor=GRADE_COLORS[1],
           markeredgecolor="none", label="$<1$ m"),
    Line2D([0], [0], marker="s", ls="none", ms=8, markerfacecolor=GRADE_COLORS[2],
           markeredgecolor="none", label="$1$–$10$ m"),
    Line2D([0], [0], marker="s", ls="none", ms=8, markerfacecolor=GRADE_COLORS[3],
           markeredgecolor="none", label="$>10$ m"),
    Line2D([0], [0], marker="s", ls="none", ms=8, markerfacecolor="none",
           markeredgecolor="#000000", markeredgewidth=1.25,
           label="closest approach"),
]
ax_b.legend(handles=leg_b, loc="lower left", bbox_to_anchor=(0.0, -0.30),
            ncol=5, frameon=False, handletextpad=0.35, columnspacing=1.05,
            borderpad=0.2)
ax_b.text(1.0, 1.012, "required CBF relaxation", transform=ax_b.transAxes,
          ha="right", va="bottom", fontsize=7.5, color="#666666")

# row marginal: infeasible-step count
ax_row.barh(np.arange(N_EP), INF.sum(1)[order_rows], height=0.74,
            color="#7FA8CE", edgecolor="none", zorder=2)
ax_row.set_ylim(N_EP - 0.5, -0.5)
ax_row.set_xlim(0, N_T)
ax_row.set_yticks([])
ax_row.set_xticks([0, 10, 20])
ax_row.set_xlabel("infeasible\nsteps", fontsize=8, labelpad=2)
ax_row.axvline(INF.sum(1).mean(), color=C_INK, ls=":", lw=1.0, zorder=3)
ax_row.grid(axis="x", lw=0.4, color="#DDDDDD", alpha=0.7)
ax_row.set_axisbelow(True)

# ============================================================ panel (c)
d2 = D_SEP - oc["minsep_stage2"]
do = D_SEP - oc["minsep_oracle"]
o_ord = np.argsort(-d2, kind="stable")

yy = np.arange(N_EP)
for gi, ri in enumerate(o_ord):
    ax_c.plot([d2[ri], do[ri]], [gi, gi], color="#BBBBBB", lw=1.1, zorder=1,
              solid_capstyle="round")
ax_c.plot(d2[o_ord], yy, "o", ms=4.6, color=C_S2, zorder=3,
          markeredgecolor="white", markeredgewidth=0.5)
ax_c.plot(do[o_ord], yy, "D", ms=4.0, color=C_ORACLE, zorder=3,
          markeredgecolor="white", markeredgewidth=0.5)

ax_c.axvline(0.0, color="#C0392B", lw=1.5, zorder=2)
ax_c.text(0.30, N_EP - 1.4, "rescue line\n(no violation)", color="#C0392B",
          fontsize=7.5, ha="left", va="center", style="italic")

i_sh = int(np.argmin(do))
g_sh = int(np.flatnonzero(o_ord == i_sh)[0])
ax_c.annotate(f"{do[i_sh]:.2f} m (epi {int(EID[i_sh])})",
              xy=(do[i_sh], g_sh), xytext=(5.4, g_sh + 1.9),
              fontsize=7.2, color=C_INK, ha="left", va="center",
              arrowprops=dict(arrowstyle="-", lw=0.8, color="#888888",
                              shrinkA=0, shrinkB=2))

ax_c.set_yticks(yy)
ax_c.set_yticklabels([str(EID[i]) for i in o_ord], fontsize=6.2)
ax_c.set_ylim(N_EP - 0.5, -0.5)
ax_c.set_xlim(-0.9, 21.0)
ax_c.set_xlabel("violation depth $30-\\min$ separation (m)")
ax_c.set_ylabel("conflict episode", labelpad=4)
ax_c.grid(axis="x", lw=0.4, color="#DDDDDD", alpha=0.7)
ax_c.set_axisbelow(True)
ax_c.legend(handles=[
    Line2D([0], [0], marker="o", ls="none", ms=5.2, color=C_S2,
           label="Stage-2 (ours)"),
    Line2D([0], [0], marker="D", ls="none", ms=4.6, color=C_ORACLE,
           label="oracle-predictor replay"),
], loc="lower right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
    handletextpad=0.5, borderpad=0.5)
plab(ax_c, "(c)")

os.makedirs(OUT_DIR, exist_ok=True)
pdf = os.path.join(OUT_DIR, "fig05_attribution.pdf")
png = os.path.join(OUT_DIR, "fig05_attribution.png")
fig.savefig(pdf)
fig.savefig(png, dpi=300)
print("saved:", pdf)
print("saved:", png)

# ------------------------------------------------- reported-number summary
print(f"  (a) patterns={len(order)}  union={int(M.any(0).sum())}  "
      f"all-four={int(M.all(0).sum())}")
print(f"  (b) infeasible cells={int(INF.sum())}  mean/row="
      f"{INF.sum(1).mean():.2f}/{N_T}  contiguous-single-run rows="
      f"{sum(1 for r in INF if np.diff(np.concatenate([[0], r.astype(int), [0]])).tolist().count(1) == 1)}/{N_EP}")
print(f"      grades: feasible={int((grade == 0).sum())} <1m={int((grade == 1).sum())} "
      f"1-10m={int((grade == 2).sum())} >10m={int((grade == 3).sum())}  "
      f"max slack={SLK.max():.2f} m")
print(f"      CPA range={CPA.min()}-{CPA.max()}; all CPA steps infeasible="
      f"{bool(INF[np.arange(N_EP), CPA].all())}")
print(f"  (c) Stage-2 depth {d2.min():.2f}-{d2.max():.2f} m; "
      f"oracle {do.min():.2f}-{do.max():.2f} m; oracle deeper in "
      f"{int((do > d2).sum())}/{N_EP}; |delta| mean {np.abs(d2 - do).mean():.3f} m")
