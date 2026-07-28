# FINAL2 paired McNemar (loose-planner Stage-1 vs Stage-2)

Paired significance test for the loose-configuration contrast reported in
Section 5.5 ("The boundaries of the findings" / weakened-constraint arm).

## What this shows

Under the **loose training/planner configuration** (CBF-MPC
`alpha=0.4`, `a_max=10`, `Hp=8`; `T=20`, `dt=0.2`, `eta_w=0.3`,
`d_sep=30`), where the certificate no longer saturates achievable safety,
fine-tuning the predictor through the planner (Stage-2) reduces conflicts
relative to the frozen Stage-1 predictor. Both arms are evaluated on the
**identical held-out episodes** (`range(2500,3000)`, seed 12345, n=200),
so the comparison is exactly paired and admits a McNemar test.

## Result (`MCNEMAR_FINAL2.txt`)

```
Stage-1 CR = 56.5%   Stage-2 CR = 28.5%   delta = -28.0 pts

Paired 2x2 (rows S1, cols S2):
                 S2 conflict   S2 safe
  S1 conflict          57          56
  S1 safe               0          87

discordant: b (S1-conflict/S2-safe) = 56,  c (S1-safe/S2-conflict) = 0
McNemar exact two-sided p = 2.776e-17
```

The improvement is **strictly nested**: Stage-2 resolves 56 conflicts
that Stage-1 incurs and introduces none in the other direction (c = 0).
The marginal conflict rates (56.5% / 28.5%) reproduce `FINAL2.txt`
exactly, confirming the same-stream rollout.

This closes the two-sided bracket around the negative result:
- **gain appears** (loose certificate): b=56, c=0, p = 2.8e-17 (this file)
- **gain vanishes** (strong deployment certificate): Stage-1b vs Stage-2
  McNemar p = n.s. (see `../../main_and_stats/STATS.txt` and the
  matched-pair analysis).

## Reproduce

From a checkout where `code/plangrad_sim/` has the weights
(`stage1_full.pt`, `stage2_final.pt`) and the GUAM data reachable via
`config.GUAM_MAT`:

```
cd code/plangrad_sim
python3 mcnemar_final2.py 200      # writes MCNEMAR_FINAL2.txt
```

The script does not modify `final_compare.py` or `FINAL2.txt`; it
re-runs the identical loose-config rollout and collects the per-episode
minimum separation so the two arms can be paired episode-by-episode.

## Files
- `mcnemar_final2.py`   — the paired-test script (self-contained)
- `MCNEMAR_FINAL2.txt`  — the run output (2x2 table + exact p)
- `FINAL2.txt`          — the original marginal-CR summary (for reference)
