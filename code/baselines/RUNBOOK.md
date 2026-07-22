# RUNBOOK — reproduce all baselines from scratch

Step-by-step to reproduce every baseline result on a fresh machine, with
**all randomness eliminated** (seed 12345). Mirrors the protocol of the main
package (`code_all/RUNBOOK.md`): train on GUAM trajectories `0..2500`,
evaluate on the disjoint held-out set `2500..3000`, conflict threshold
`d_sep = 30 m`, best CBF-MPC planner `alpha=0.1 / Hp=15 / a_max=20`,
`n = 200`.

Verified hardware: single **NVIDIA RTX 4090**, Ubuntu 22.04, Python 3.10,
driver 565.57.01, CUDA 12.8.

---

## Stage 0 — paths & layout

The baselines live in `/data/lab/TRC-UAV/baselines` and import the main
package from `/data/lab/TRC-UAV/plangrad/plangrad_sim`.

The main package hard-codes the path `/data/lab/plangrad/...`, so we expose
it via a symlink (do this once):
```bash
ln -s /data/lab/TRC-UAV/plangrad /data/lab/plangrad   # if not already present
ls /data/lab/plangrad/GUAM/Challenge_Problems/Data_Set_1.mat   # must exist
```

If you keep GUAM elsewhere, instead set:
```bash
export GUAM_MAT=/abs/path/GUAM/Challenge_Problems/Data_Set_1.mat
```
GUAM is the public NASA dataset (no MATLAB needed; read via h5py):
```bash
git clone --depth 1 \
  https://github.com/nasa/Generic-Urban-Air-Mobility-GUAM.git GUAM
#   -> GUAM/Challenge_Problems/Data_Set_1.mat   (~23 MB)
```

---

## Stage 1 — environment

Dependencies (pinned to the main package's `requirements.txt`):
```bash
pip install torch==2.11.0 numpy==2.2.6 scipy==1.15.3 h5py==3.16.0 \
            cvxpy==1.7.5 cvxpylayers==0.1.9
```

### CUDA forward-compat fix (IMPORTANT on this box)
On this machine the CUDA 12.8 "compat" `libcuda` (570.124.06) shadows the
real installed driver (565.57.01) through the ldconfig cache, so plain
`python3 -c "import torch; torch.cuda.is_available()"` returns **False**
with `CUDA error 804: forward compatibility was attempted on non supported
HW`. The fix is to force-load the real system driver. This (and the seeds
and GUAM path) is bundled in **`env.sh`** — source it before every run:

```bash
source /data/lab/TRC-UAV/baselines/env.sh
python3 -c "import torch;print(torch.cuda.is_available(),torch.cuda.get_device_name(0))"
# expected:  True NVIDIA GeForce RTX 4090
```

`env.sh` contents:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.565.57.01
unset LD_LIBRARY_PATH
export GUAM_MAT=/data/lab/plangrad/GUAM/Challenge_Problems/Data_Set_1.mat
export PYTHONHASHSEED=12345
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Sanity check (optional but recommended)
```bash
source /data/lab/TRC-UAV/baselines/env.sh
cd /data/lab/plangrad/plangrad_sim && python3 test_smoke.py   # -> All smoke tests passed.
cd /data/lab/TRC-UAV/baselines/common && python3 _repro_check.py --n 24
#   confirms Stage-1 ADE ~19.8 vs Stage-2 ADE ~9.4 (ordering matches final_best)
```

---

## Stage 2 — checkpoints

These must be present in `/data/lab/plangrad/plangrad_sim/`:
- `stage1_full.pt` — Stage-1 displacement-pretrained predictor (manuscript)
- `stage2_final.pt` — Stage-2 task-aligned predictor = **PlanGrad** (manuscript)

(If you need to regenerate them, follow `code_all/RUNBOOK.md` stages 3-4.)

The Soft-IPP baseline trains its own checkpoint — see Stage 3.

---

## Stage 3 — train the Soft-IPP baseline (the only baseline that trains)

Same Stage-1 init, same TASL loss, same iters/batch/seed/data-pool as
PlanGrad's Stage-2, but the in-loop planner is the **soft** Vanilla-MPC
(no CBF). ~15-25 min on a 4090.

```bash
source /data/lab/TRC-UAV/baselines/env.sh
cd /data/lab/TRC-UAV/baselines/04_soft_ipp
python3 -u train.py --cuda --seed 12345 --iters 50 --batch 16 \
    --T 20 --Hp 15 --a_max 20 --lr 1e-4 \
    --w_coll 3.0 --w_lead 0.5 --w_ade 0.3 --n_traj_pool 2500 \
    --out soft_joint.pt
# -> saved Soft-IPP predictor -> soft_joint.pt
```
Each step prints `loss | mean_min_sep(m) | soft_coll | energy | lead | ade`.
Occasional `SKIP (solver issue: ...)` is normal (infeasible QP geometry).

> Tip: under `nohup`, add `-u` (unbuffered) or you will not see the per-step
> log until the process ends.

---

## Stage 4 — run all baselines at n = 200

The training-free / frozen baselines need no training step. Run everything
(baselines + the PlanGrad reference) with the batch script:

```bash
source /data/lab/TRC-UAV/baselines/env.sh
bash /data/lab/TRC-UAV/baselines/run_all_n200.sh
```

…or individually (each writes `result.txt` + `result.json` in its folder):
```bash
cd /data/lab/TRC-UAV/baselines/01_constant_velocity && python3 -u run.py --n 200
cd /data/lab/TRC-UAV/baselines/02_vanilla_mpc        && python3 -u run.py --n 200
cd /data/lab/TRC-UAV/baselines/03_fixed_predictor    && python3 -u run.py --n 200
cd /data/lab/TRC-UAV/baselines/04_soft_ipp           && python3 -u run.py --n 200
cd /data/lab/TRC-UAV/baselines/05_conformal_mpc      && python3 -u run.py --n 200 --delta 0.1
cd /data/lab/TRC-UAV/baselines/common                && python3 -u _ref_plangrad.py --n 200
```
Each n=200 run takes a few minutes (the differentiable CBF-MPC QP at Hp=15
dominates). Run sequentially to avoid GPU/solver contention.

### Expected results (seed 12345, n = 200) — re-verified
| Folder | CR % | minSep (m) | ADE (m) | LeadT (s) | Energy |
|--------|-----:|-----------:|--------:|----------:|-------:|
| 00_plangrad_reference (OURS) | 11.0 | 47.7 | 4.32 | 0.038 | 52.3 |
| 01_constant_velocity | 12.0 | 47.8 | 0.83 | 0.037 | 52.9 |
| 02_vanilla_mpc | 41.0 | 29.8 | 4.32 | 0.066 | 19.1 |
| 03_fixed_predictor | 12.5 | 46.3 | 20.90 | 0.039 | 51.5 |
| 04_soft_ipp | 53.0 | 29.1 | 6.45 | 0.080 | 17.0 |
| 05_conformal_mpc | 11.5 | 48.8 | 20.90 | 0.034 | 57.9 |

(Baseline 03 reproduces `final_best.py`'s Stage-1 numbers and the reference
reproduces its Stage-2 numbers, confirming the evaluator is faithful.)

> **Re-verification note.** An earlier build of this table reported
> `ADE = 10.22 m` for the Stage-2 rows (00, 02) and `CR = 60.0` for
> Vanilla-MPC (02). Both were wrong: (i) the true Stage-2 ADE is **4.32 m**
> (@30-step horizon; weighted-mean and best-mode agree to ~0.1 m, confirmed by
> `plangrad_sim/final_compare.py`) — `10.22` did not correspond to any horizon
> or mode and was a stale/mis-scaled figure; (ii) the clean single-batch re-run
> of Vanilla-MPC (w_rep=50) gives **CR 41.0**. CV, Soft-IPP and Conformal
> reproduce their prior numbers. Always regenerate the whole table in one
> `run_all_n200.sh` pass so all rows share one batch.

---

## Stage 5 — aggregate

```bash
source /data/lab/TRC-UAV/baselines/env.sh
cd /data/lab/TRC-UAV/baselines && python3 aggregate.py
# -> results.csv  and  RESULTS.md   (table + interpretation)
```

---

## Reproducibility checklist
- [x] Single seed **12345** everywhere (`seeding.set_seed`), wind seed 7.
- [x] Train range `0..2500`, eval range `2500..3000` — disjoint, no leakage.
- [x] One unified evaluator (`common/eval_common.py`) for ALL methods.
- [x] Same planner / d_sep / dt / T / n for every row.
- [x] float64 planner+dynamics; `cudnn.deterministic=True`.
- [x] All inputs are the public GUAM dataset + released checkpoints + the
      reproducibly-trained `soft_joint.pt`.

## Troubleshooting
- `CUDA available = False` / `error 804` -> you forgot `source env.sh`.
- `No such file: Data_Set_1.mat` -> fix the symlink or `export GUAM_MAT=...`.
- `ModuleNotFoundError: cvxpylayers/scipy` -> rerun the pip install (Stage 1).
- QP `SolverError` spam during Soft-IPP training -> normal; loop skips that
  mini-batch. If >50 % skipped, lower `--w_rep` or `--beta`.
- No per-step log under nohup -> use `python3 -u`.
