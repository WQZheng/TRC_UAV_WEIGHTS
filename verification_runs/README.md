# verification_runs/ — planner-selection validation + conflict-attribution proof

Two verification experiments requested during review, both diagnostic only
(no new model weights): they use the frozen, already-trained checkpoints.

---

## 1. Independent-stream validation of the planner-config selection

**Why.** The "best" planner configuration (alpha=0.1, Hp=15, a_max=20) was
originally selected by a 6-point sweep evaluated on the SAME held-out stream
(pool 2500-2999, seed 12345, n=200) used for the final reported numbers. That
is hyperparameter selection on the test stream, so the absolute CBF-arm
numbers carry an optimistic bias and the "best planner" label needs support.

**What.** `scan_planner_indep.py` re-runs the identical 6-config coarse grid
on the SAME pool but with an INDEPENDENT seed (999), so config ranking is
validated on a stream disjoint (by seed) from the one it was observed on.
Every arm still shares one config, so inter-arm comparisons are unaffected;
only the absolute level and the selection legitimacy are at issue.

**Result** (`PLANNER_SCAN_INDEP_seed999.txt`):

| config                   | CR% (indep seed 999) | minSep (m) |
|--------------------------|---------------------:|-----------:|
| alpha0.4 Hp8  amax10     | 26.0 | 34.4 |
| alpha0.2 Hp8  amax10     | 26.0 | 35.7 |
| alpha0.1 Hp8  amax10     | 26.0 | 36.0 |
| alpha0.2 Hp12 amax10     | 28.0 | 34.0 |
| alpha0.2 Hp12 amax16     | 18.0 | 42.2 |
| **alpha0.1 Hp15 amax20** | **12.0** | **48.5** |

The ranking reproduces: `alpha0.1 Hp15 amax20` is again the unique best
config and the only one reaching the 12% level. Absolute mid-grid CR values
shift with the stream (which itself demonstrates the stream-dependence of
absolute numbers), but the SELECTION is stream-robust. This upgrades the §5
write-up from "disclose a caveat" to "selection validated on an independent
stream (same ranking, same winner); all arms share the config, so inter-arm
comparisons are unaffected."

## 2. Zero-slack feasibility re-solve — "actuation-limited" attribution proof

**Why.** §6 attributes residual conflicts to being "actuation-limited",
evidenced by the CBF-MPC slack variable eps being active (non-zero) at the
conflict. But a non-zero slack only shows the solver CHOSE to relax the
constraint for cost reasons; it does not prove the hard problem was
INFEASIBLE. Without an infeasibility test the correct term is only the weaker
"slack-associated".

**What.** `zero_slack_feasibility.py` re-solves, at every step of every
conflict episode, the SAME double-integrator CBF-MPC QP but with the slack
REMOVED (eps == 0), as a pure feasibility problem, with the neighbour
prediction held fixed (detached). Best planner, n=200, seed 12345.
  * eps=0 infeasible at a conflict step -> actuation-limited (justified).
  * eps=0 feasible everywhere            -> slack-associated only.

**Result** (`ZERO_SLACK_FEAS.txt`):

| quantity | value |
|----------|------:|
| total conflict episodes | 22 |
| with >=1 eps=0-INFEASIBLE step (actuation-limited) | **22 (100%)** |
| eps=0 feasible at every step (slack-associated only) | 0 |
| infeasible steps per conflict (mean / max) | 17.77 / 20 |

Every conflict has an eps=0-infeasible step, and typically almost the entire
20-step window is infeasible (mean 17.77). The strong term
**"actuation-limited"** is therefore justified: these conflicts are not the
solver electing to use slack for cost, they are episodes in which the hard
separation constraint is genuinely infeasible under the actuation envelope.
Evidence upgrades from weak (slack non-zero) to strong (eps=0 infeasible).

## 3. File manifest

| file | description |
|------|-------------|
| `scan_planner_indep.py` | independent-stream (seed 999) 6-config planner sweep. |
| `PLANNER_SCAN_INDEP_seed999.txt` | its result — ranking reproduces, winner unchanged. |
| `zero_slack_feasibility.py` | eps=0 feasibility re-solve over conflict episodes. |
| `ZERO_SLACK_FEAS.txt` | its result — 22/22 conflicts actuation-limited. |

Both scripts also live under `code/plangrad_sim/`; copied here for a
self-contained verification bundle. No model weight is produced or changed.

## 4. Manuscript touch-points (applied to .tex separately)

- §5 protocol: state that the planner config was chosen by a coarse
  6-point sweep and validated on an independent stream (seed 999) that
  reproduces the same ranking and winner; all arms share the config.
- §6 attribution: keep "actuation-limited"; cite the zero-slack re-solve
  (22/22 conflicts eps=0-infeasible) as the justification, replacing the
  weaker slack-active argument.
