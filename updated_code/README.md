# `updated_code/` — Train–deploy planner-matching robustness check

This folder adds a **robustness check** requested during review of the
manuscript *"End-to-End Differentiable Prediction and Planning for Safe
Autonomous Navigation in Low-Altitude Urban Air Mobility"* (PlanGrad-UAV).
It does **not** replace anything in `code/`; it is a small, self-contained
addition that reproduces one extra experiment and ships the resulting
checkpoint.

---

## 1. Why this was added (the reviewer concern)

A reviewer pointed out a **train–deploy planner mismatch**:

- **Stage-2 training** (headline protocol) fine-tunes the predictor
  end-to-end through a *relaxed* differentiable CBF–MPC planner:
  `alpha = 0.4, Hp = 8, a_max = 10`.
- **Evaluation / deployment** (`final_best.py`, the headline numbers) uses
  a *stronger, tuned* planner: `alpha = 0.1, Hp = 15, a_max = 20`.

Because the Stage-2 end-to-end gradient therefore flows through a planner
that **differs** from the one used at test time, a reviewer can attack the
paper's central **negative** result (RQ5: "joint task-aligned training gives
no incremental operational-safety benefit") as follows:

> *"Is the absence of a safety gain real, or is it because you taught the
> predictor to accommodate planner A but examined it with planner B?"*

This is the sharpest potential attack on the core negative finding, so we
address it directly by **removing the mismatch**: we re-train Stage-2 with
the training planner set **identical** to the deployment planner
(`alpha = 0.1, Hp = 15, a_max = 20`) and re-evaluate under that same
planner. If the conclusion holds, the negative result is robust; if it
flips, that is something we must know.

---

## 2. What was changed (exact diff)

### 2.1 `train_stage2.py` — made `a_max` configurable (2-line change)

In the original `code/plangrad_sim/train_stage2.py`, the maximum control
authority `a_max` was **hard-coded to `10.0`** in the CBF–MPC layer, so it
was impossible to train Stage-2 under the deployment planner's
`a_max = 20` from the command line (`--alpha` and `--Hp` were already
exposed, but `--a_max` was not).

**Change 1** — pass `a_max` through instead of hard-coding it
(around line 105):

```diff
-        mpc = CBFMPCLayer(n_neighbors=1, horizon=args.Hp, dt=args.dt,
-                          d_sep=D_SEP, alpha=args.alpha, a_max=10.0)
+        mpc = CBFMPCLayer(n_neighbors=1, horizon=args.Hp, dt=args.dt,
+                          d_sep=D_SEP, alpha=args.alpha, a_max=args.a_max)
```

**Change 2** — add the corresponding CLI argument (right after `--alpha`):

```diff
     ap.add_argument("--alpha", type=float, default=0.4)
+    ap.add_argument("--a_max", type=float, default=10.0)
```

The default is `10.0`, so **the original headline behaviour is unchanged**
when `--a_max` is not supplied. Nothing else in the training logic,
losses, or hyperparameters was touched.

### 2.2 `eval_matched.py` — new evaluation script (new file)

A thin wrapper around the same evaluation protocol as
`code/plangrad_sim/final_best.py` (deployment planner
`alpha = 0.1, Hp = 15, a_max = 20`, `n = 200` held-out encounters
2500–3000, seed 12345, `d_sep = 30 m`). It loads `stage1_full.pt` and an
arbitrary Stage-2 checkpoint passed on the command line, and prints
CR / MinSep / ADE for both plus the Stage-2 − Stage-1 delta. It exists so
the matched-config checkpoint can be scored under the *identical* protocol
as the headline table.

---

## 3. What was generated

### `stage2_matched.pt`

The Stage-2 predictor **re-trained under the matched deployment planner**
(`alpha = 0.1, Hp = 15, a_max = 20`), everything else identical to the
headline Stage-2 run: initialised from `stage1_full.pt`, seed `12345`,
2500-trajectory encounter pool, 70 iterations, `batch = 16`, `T = 20`,
`lr = 1e-4`, `--w_coll 3.0 --w_lead 0.5 --w_ade 0.3`.

Reproduce with (from `code/plangrad_sim/`, with this folder's
`train_stage2.py` on the path and `GUAM_MAT` set):

```bash
python3 train_stage2.py --cuda --seed 12345 \
    --stage1 stage1_full.pt --out stage2_matched.pt \
    --iters 70 --batch 16 --T 20 --Hp 15 --alpha 0.1 --a_max 20 --lr 1e-4 \
    --w_coll 3.0 --w_lead 0.5 --w_ade 0.3 --n_traj_pool 2500
```

Then evaluate:

```bash
python3 eval_matched.py stage2_matched.pt
```

---

## 4. Result (the point of the whole exercise)

Evaluated under the **deployment** planner (`alpha = 0.1, Hp = 15,
a_max = 20`), `n = 200`, seed 12345:

| Stage-2 training planner              | ΔCR (pts) | ΔMinSep (m) | ΔADE (m) |
|---------------------------------------|----------:|------------:|---------:|
| Relaxed `α0.4,Hp8,amax10` (headline)  |     −1.5  |       +1.4  |   −16.6  |
| **Matched `α0.1,Hp15,amax20`**        | **+1.0**  |     −1.5    | **−11.9** |

Absolute values under the matched training planner:

| model            | CR (%) | MinSep (m) | ADE (m) |
|------------------|-------:|-----------:|--------:|
| Stage-1          |  12.5  |    46.3    |  20.90  |
| Stage-2 (matched)|  13.5  |    44.9    |   9.05  |

**Conclusion — the negative result is robust.** Even when Stage-2 is
trained through the very planner it is later evaluated on, the conflict
rate relative to the frozen Stage-1 predictor moves by only +1.0 pt
(12.5% → 13.5%) — the same tiny magnitude as, and opposite in sign to, the
−1.5 pt seen under the mismatched headline protocol. Both differences are
within the sampling noise of a binary per-episode conflict indicator at
n = 200. The hypothesis that matching the planners would surface a hidden
safety gain is thus **falsified by measurement**, not merely argued away.
Meanwhile the prediction-quality benefit of task alignment survives
intact: ADE still drops from 20.90 m to 9.05 m (−57%). This directly
supports the paper's decoupling claim: task alignment buys much better
prediction, and that accuracy does **not** translate into a lower conflict
rate because the residual conflicts are actuation-limited, not
prediction-limited.

The single fixed seed (12345) here establishes the direction of the
effect; a multi-seed replication is intended for the appendix.

---

## 5. File manifest

| file                | type      | description |
|---------------------|-----------|-------------|
| `train_stage2.py`   | modified  | Stage-2 trainer with the new `--a_max` CLI arg (default 10.0 → headline behaviour unchanged). |
| `eval_matched.py`   | new       | Scores Stage-1 vs a given Stage-2 checkpoint under the deployment planner (α0.1/Hp15/amax20), n=200, seed 12345 — same protocol as `final_best.py`. |
| `stage2_matched.pt` | generated | Stage-2 predictor re-trained under the matched deployment planner; the checkpoint behind Section 4. |
| `README.md`         | doc       | This file. |

> Note: `train_stage2.py` here is the **patched** trainer. The original,
> unmodified trainer remains untouched at
> `code/plangrad_sim/train_stage2.py`.
