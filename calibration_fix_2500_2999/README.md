# Calibration-set data-integrity fix (Conformal-MPC baseline)

This folder documents a fix to the **split-conformal calibration set** used by
the Conformal-MPC baseline, and archives the re-run results under the corrected
calibration. No main-line PlanGrad result, and no trained model weight, is
affected — Conformal-MPC is a post-hoc calibration of the *frozen* Stage-1
predictor, so this fix produces **no new model checkpoint**, only updated
baseline result artifacts.

---

## 1. The bug

Split conformal prediction requires the calibration set to be **disjoint from
the training set** (otherwise the frozen predictor has already seen the
calibration trajectories, its nonconformity scores are optimistically small,
the radius `r_conf` is too small, and the finite-sample coverage guarantee is
void).

- Stage-1 training used GUAM trajectories **0–2499**
  (`train_stage1.py`, `traj_indices=range(2500)`).
- The Conformal-MPC calibration set was **`range(2000, 2500)`**
  (`conformal.py`, old `CALIB_RANGE`).

These **overlap**: trajectories 2000–2499 were used for *both* training and
calibration. `GUAMWindowDataset` (training) and `GUAMEncounters` (calibration)
address the same underlying GUAM trajectories by the same integer index, so the
overlap is real, not nominal. The old `conformal.py` even carried a self-
contradictory comment claiming the range was *"disjoint from train(0-2500
used)"* while sitting inside that very range.

## 2. The fix (and why this particular fix)

Two separations must hold for split-conformal validity, with different
consequences if broken:

- **(a) calibration ∩ training = ∅** — was violated; MUST fix.
- **(b) calibration and evaluation exchangeable** — does NOT require disjoint
  trajectory *pools*; independent draws from the same generating distribution
  are the textbook setting.

We therefore calibrate on the **held-out pool 2500–2999** (the same pool the
evaluation episodes come from) but with a **dedicated calibration seed
`CALIB_SEED = 777`**, independent of the evaluation seed (12345). The
`GUAMEncounters` generator is programmatic (pool + seed → sampled/rotated/
translated encounters), so a different seed yields an independent draw with no
shared episode.

This satisfies all three hard constraints at once:

1. `calibration ∩ training = ∅`  →  2500–2999 vs 0–2499 ✓ (fixes bug (a))
2. `calibration ⫫ evaluation`   →  same pool, independent seeds, no shared
   episode ✓ (exchangeability (b))
3. **evaluation stream byte-for-byte unchanged** (still pool 2500–2999, seed
   12345) → the paired **McNemar / Wilson-CI** framework across all arms is
   preserved. This is why we did NOT move the evaluation to a different pool
   (e.g. 2600–3000): that would have broken the per-episode pairing that the
   paired significance tests require.

Changed exactly one file: `conformal.py`
(`CALIB_RANGE = range(2500, 3000)`, `CALIB_SEED = 777`, calibration RNG forced
to `CALIB_SEED` regardless of caller, docstring/comment rewritten + changelog).
All five consumers of `conformal_radius` (05_conformal_mpc, 06_sim_ood,
figures_gen/collect_data{,_extra}, common/stats_tests) pick up the new
calibration automatically.

## 3. Result — direction sanity check passes, conclusions unchanged

| quantity                       | OLD (contaminated) | NEW (clean) |
|--------------------------------|-------------------:|------------:|
| calibration range             | 2000–2499          | 2500–2999   |
| calibration seed              | 12345              | 777         |
| **r_conf (m)**                | **18.57**          | **18.93**   |
| mean nonconformity score (m)  | 10.61              | 11.20       |
| Conformal-MPC CR (%)          | 11.5               | 11.5        |
| Conformal-MPC minSep (m)      | 48.78              | 48.80       |
| Conformal-MPC ADE (m)         | 20.90              | 20.90       |
| Conformal-MPC Energy          | 57.93              | 58.13       |

As expected under a clean (unseen) calibration set, the nonconformity scores
grow and **`r_conf` increases** (18.57 → 18.93 m), i.e. the honest calibrated
buffer is slightly larger — confirming the predictor is indeed less accurate on
trajectories it has not seen. The change (+0.36 m) is small relative to the
30 m separation standard, so the discrete operational metrics are essentially
unchanged (CR identical at 11.5%, minSep +0.02 m). The paired tests
(`STATS_CLEAN.txt`) are unchanged: Conformal-MPC CR 11.5% (Wilson [7.8, 16.7]),
McNemar vs PlanGrad p = 1.000 (n.s.). The OOD sweep (`ood_results_CLEAN.*`)
holds `r_conf = 18.93 m` fixed across wind levels; Conformal-MPC stays at
CR 11.5% / minSep 48.8 m at all three η_w.

**Bottom line:** the qualitative conclusion is direction-stable — the fix
requires no new explanatory prose beyond the corrected data-split description.

## 4. File manifest

| file | description |
|------|-------------|
| `conformal_FIXED.py` | the corrected calibration module (also live at `code/baselines/05_conformal_mpc/conformal.py`). |
| `conformal_mpc_result_CLEAN.{txt,json}` | Conformal-MPC main arm, clean calibration, n=200, seed 12345. |
| `ood_results_CLEAN.{txt,json}` | RQ4 Sim-OOD wind sweep, all arms, clean calibration. |
| `STATS_CLEAN.txt` | paired McNemar + Wilson CI, clean calibration. |
| `conformal_mpc_result_OLD_contaminated.txt` | the previous (contaminated) Conformal-MPC result, kept for reference. |

## 5. Manuscript edits implied (to be applied to the .tex, tracked separately)

- §4.2: "disjoint from both the training and the evaluation data" →
  "disjoint from the training data and generated independently from the
  held-out pool (dedicated calibration seed)".
- §5 protocol: state the three data segments explicitly — training 0–2499,
  calibration 2500–2999 (seed 777, independent), evaluation 2500–2999
  (seed 12345) — with the exchangeability argument.
- Conformal-MPC numbers in the main table / STATS / OOD table: r_conf
  18.57 → 18.93 m; operational metrics unchanged.
