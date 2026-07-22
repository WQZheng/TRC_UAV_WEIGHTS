"""Figure 8 (RQ5): directional decomposition of critical-region prediction
error for Stage-1, Stage-1b (domain adaptation) and Stage-2 (task-aligned).

Data from baselines/06_sim_ood/error_direction.txt (n=200, seed 12345):
  along-axis e_par (signed, m; <0 = conservative / toward ego) and
  perpendicular |e_perp| (m), critical region.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

models = ["Stage-1\n(displacement)", "Stage-1b\n(domain adapt.)",
          "Stage-2\n(task-aligned)"]
e_par = [-0.70, -0.18, -0.34]          # signed along-axis mean (m)
e_par_sem = [0.30, 0.12, 0.15]
e_perp = [8.38, 0.68, 2.91]            # perpendicular magnitude (m)
frac_neg = [53.8, 45.4, 54.6]          # % of critical errors toward ego

fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))

x = np.arange(len(models))
# Left: along-axis signed bias with SEM error bars
colors = ["#9aa0a6", "#4c78a8", "#e45756"]
ax[0].axhline(0, color="k", lw=0.8)
ax[0].bar(x, e_par, yerr=e_par_sem, capsize=4, color=colors,
          edgecolor="k", lw=0.6)
ax[0].set_xticks(x); ax[0].set_xticklabels(models, fontsize=8)
ax[0].set_ylabel("Along-axis error $e_\\parallel$ (m)", fontsize=9)
ax[0].set_title("Conservative bias (negative = toward ego)", fontsize=9)
for i, v in enumerate(e_par):
    ax[0].annotate(f"{v:+.2f}", (x[i], v), textcoords="offset points",
                   xytext=(0, -12 if v < 0 else 6), ha="center", fontsize=8)
ax[0].annotate("conservative", (0.02, 0.04), xycoords="axes fraction",
               fontsize=7.5, color="#555")

# Right: perpendicular magnitude (off-axis, ungoverned by the gradient)
ax[1].bar(x, e_perp, color=colors, edgecolor="k", lw=0.6)
ax[1].set_xticks(x); ax[1].set_xticklabels(models, fontsize=8)
ax[1].set_ylabel("Perpendicular error $|e_\\perp|$ (m)", fontsize=9)
ax[1].set_title("Off-axis error (no constraint-aligned gradient)", fontsize=9)
for i, v in enumerate(e_perp):
    ax[1].annotate(f"{v:.2f}", (x[i], v), textcoords="offset points",
                   xytext=(0, 3), ha="center", fontsize=8)

fig.tight_layout()
fig.savefig("figure8_rq5_error_direction.pdf", bbox_inches="tight")
fig.savefig("figure8_rq5_error_direction.png", dpi=150, bbox_inches="tight")
print("wrote figure8_rq5_error_direction.pdf")
