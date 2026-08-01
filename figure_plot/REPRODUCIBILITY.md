# PlanGrad-UAV — Results Figures Reproducibility Manifest

Every results figure (Fig 3–13) is produced by exactly **one standalone script**
in `figure_plot/`, sharing only `figstyle.py` (visual encoding). This manifest
records, per script, the **data files**, **model weights**, and **GitHub /
Lab locations** each one reads, so any figure can be regenerated from scratch.

## Global evaluation configuration (authoritative)

All closed-loop numbers come from the Lab evaluation stack under a single fixed
configuration (`code/baselines/common/eval_common.py`):

| constant | value | source |
|---|---|---|
| separation standard `D_SEP` | 30.0 m | eval_common.py:60 |
| wind coefficient `WIND_ETA` (η_w) | **0.3** | eval_common.py:66 |
| wind seed `WIND_SEED` | 7 | eval_common.py:65 |
| global seed `GLOBAL_SEED` | 12345 | eval_common.py:67 |
| held-out encounters `EVAL_RANGE` | 2500–2999 | eval_common.py:64 |
| episode length `T_EPISODE` | 20 steps | eval_common.py:63 |
| step `DT` | 0.2 s | eval_common.py:62 |
| deployment planner `BEST_PLANNER` | α(γ)=0.1, Hp=15, a_max=20 | eval_common.py:70 |
| GUAM data | `code/GUAM/Challenge_Problems/Data_Set_1.mat` | via `config.GUAM_MAT`; export `GUAM_MAT=…` before running |
| n per arm | 200 (multiple of batch 8) | collector / eval_common assertion |

> **η_w flag for the coauthor:** every results figure and the main STATS table
> use **η_w = 0.3** (code `WIND_ETA`, eval_common.py:66). §5.1 of the manuscript
> currently states 0.5. This is a **text/code mismatch to resolve** — the
> figures follow the code (0.3), which is what the reported CR/MinSep/Effort
> numbers were computed with. See the coauthor memo.

## Repository

- GitHub: `github.com/WQZheng/TRC_UAV_WEIGHTS` (branch `main`).
- Lab working copy: `/data/lab/TRC_UAV_WEIGHTS/`.
- Scripts live in `figure_plot/`; generated PDFs in `figures_generated/`;
  freshly-collected figure data in `code/baselines/figures_gen/fig_data/`.

## Model weights (in `code/plangrad_sim/`, symlinked to `Round1/04_weights/`)

| logical name | file | used by |
|---|---|---|
| Stage 1 (fixed predictor) | `stage1_full.pt` | Fig 4,5,7 (via collector) |
| Stage-1b (domain-adapted) | `stage1b_domainadapt.pt` | Fig 4,5,7 (via collector) |
| Stage 2 (final, deployed) | `stage2_final.pt` | Fig 4,5,7,8,9 (via collector) |
| Stage 2 (matched-budget) | `stage2_matched.pt` | (sensitivity; not a main figure) |
| loose-minsep ablation | `loose_minsep.pt` | Fig 12 numbers (archived) |
| Soft-IPP joint | `04_soft_ipp/soft_joint.pt` | **ABSENT** → Soft-IPP omitted from collector-derived panels (Fig 4 ECDF, Fig 5 uses the v9-table Soft-IPP row instead). Flagged below. |

## Data collection (the slow part)

`code/baselines/figures_gen/collect_fig_data.py` (fast OSQP path) produces, at
n=200 / η_w=0.3 / seed 12345 / evtol 2500–2999, deployment planner:

- `fig_data/minsep_effort.npz` — per-episode min-separation & control-effort
  arrays for the 7 non-Oracle arms. **Uses `FastCBFMPC` (OSQP)**, which solves
  the *identical* QP to the differentiable `CBFMPCLayer` (self-verified
  <0.5 m/s² in `fast_cbf_mpc.__main__`), ~10–50× faster. Per-arm CR reproduces
  the main table: Stage2 **11.0**, Stage-1b **11.5**, Stage1/Fixed **12.5**,
  CV **12.0** (verified this run).
- `fig_data/errdir_profile.npz` — mean |prediction error| vs |k−k_CPA| for
  Stage 1 / Stage-1b / Stage 2.
- `fig_data/planner_heatmap_n200.json` — 4×4 (γ,a_max) CR grid, Stage 2, fast
  OSQP path (produced by `run_heatmap_fast.py`; see note).
- `fig_data/infeasible_steps.npy` — per-conflict hard-infeasible step counts for
  the Fig 8 inset, dumped by the patched `plangrad_sim/zero_slack_feasibility.py`.

## Per-figure provenance

| Fig | script | data source | weights |
|---|---|---|---|
| 3 | `fig03_ade_cr.py` | v9 `tab:main-comparison` (CR/MinSep/ADE/Effort) + Cochran Q=5.000,df=3,p=0.172 (`STATS_COCHRAN`) | — (hard-coded authoritative numbers) |
| 4 | `fig04_minsep_ecdf.py` | `fig_data/minsep_effort.npz` (`<arm>__minsep`) | Stage1/1b/2 + CV + Conformal (collector) |
| 5 | `fig05_effort_cr.py` | v9 `tab:main-comparison` point estimates; Wilson CI recomputed; optional effort SE from `minsep_effort.npz` | — / collector effort |
| 6 | `fig06_matched.py` | v9 `tab:stage1b` (MinSep, Effort, max-lat-offset, paired Δ + CIs) | — (authoritative numbers) |
| 7 | `fig07_cpa_error.py` | (a) `fig_data/errdir_profile.npz`; (b) v9 `tab:errdir` (e_par, |e_perp|, paired Δ=−0.16 CI[−0.283,−0.045]) | Stage1/1b/2 (collector, panel a) |
| 8 | `fig08_attribution.py` | `P1_ORACLE_CONFLICTS.txt` (22/200, 0/22 resolved) + `ZERO_SLACK_FEAS.txt` (22/22 hard-infeasible, mean 17.77/20); inset `fig_data/infeasible_steps.npy` | stage2_final (via feasibility re-solve) |
| 9 | `fig09_heatmap.py` | `fig_data/planner_heatmap_n200.json` (γ×a_max CR) | stage2_final (collector, fast path) |
| 10 | `fig10_robustness.py` | `code/baselines/06_sim_ood/ood_results.json` (η∈{0.5,1.0,1.5}) | archived OOD sweep (all 6 arms; no Stage-1b wind sweep → omitted) |
| 11 | `fig11_leadtime.py` | v9 `tab:leadtime` (9 lead times × Stage1/1b/2/Oracle; Stage-1b None @7/10/20 s) | — (authoritative numbers) |
| 12 | `fig12_loose.py` | v9 loose-minsep block (CR 56.5/56.0/28.5; 55 resolved/0 introduced; McNemar p=5.6e-17; γ=0.4,Hp=8,a_max=10) | loose_minsep.pt (archived numbers) |
| 13 | `fig13_corridor.py` | **`PENETRATION_HIGH_CENSOR.txt` + `PENETRATION_LOW_CENSOR.txt`** (git HEAD 5f99459 `[CENSOR]` audit); panel (b) non-completion = ALL-group discard% (0→8.0% high, 0.9→10.8% low, right-censoring 0%) | — (corridor sim tables) |

Archived text tables live under
`Round1/05_results/robustness/p0_referee/`.
`ood_results.json` copies exist at `code/baselines/06_sim_ood/`,
`baselines/06_sim_ood/`, `Round1/05_results/main_and_stats/`, and
`Round1/02_baselines/06_sim_ood/` (identical).

## Path resolution

Every data-reading script calls `figstyle.find_data(...)`, which searches
`$FIGDATA_ROOT`, then the repo root inferred from the script, then
`/data/lab/TRC_UAV_WEIGHTS/code` and `/data/lab/TRC_UAV_WEIGHTS`. The same
script therefore runs unmodified locally or on the Lab.

## Known caveats / flags

1. **η_w = 0.3 vs §5.1 text (0.5).** Figures follow the code (0.3). Coauthor to
   reconcile the prose. (See memo.)
2. **Soft-IPP weight (`soft_joint.pt`) absent on the Lab.** The collector skips
   the Soft-IPP arm; Fig 4 (ECDF) therefore omits Soft-IPP, while Fig 5 shows
   the Soft-IPP point from the v9 main table (CR 53.0, Effort 17.0). If the
   weight is restored, re-run the collector to add Soft-IPP to the ECDF.
3. **Fig 9 heatmap** is produced by `run_heatmap_fast.py` (fast OSQP), not the
   in-process slow `CBFMPCLayer` path, to avoid multi-hour runtime; the QP is
   identical.
4. The **stale local `baselines/figures_gen/data/*.json` and `code_all/`** copies
   are NOT used by any figure (dead path `/data/lab/TRC-UAV/`, PlanGrad CR=12.0
   vs authoritative 11.0). Final figures use Lab data + Lab weights only.
