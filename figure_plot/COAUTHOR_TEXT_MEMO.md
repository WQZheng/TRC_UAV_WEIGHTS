# Memo to coauthor — text fixes for v9 (our side = data/scripts/figures only)

We did **not** edit any `.tex`. The items below are text changes for you to make
in the authoritative manuscript; each is cross-checked against the Lab code and
the archived gold files with locations.

## 1. η_w mismatch — MUST fix (affects every reported number)

- **Text (v9, 05_experiments_results_v9.tex:93–94):** "…scaling factor η_w,
  held at a nominal **0.5** for training **and in-distribution evaluation** and
  raised in the robustness analyses."
- **Code:** in-distribution evaluation uses **η_w = 0.3**
  (`code/baselines/common/eval_common.py:66`, `WIND_ETA = 0.3`). Every
  main-table CR/MinSep/ADE/Effort and every results figure was computed at
  η_w = 0.3 (verified this session: the collector reproduces Stage2 CR=11.0,
  Stage-1b 11.5, Stage1 12.5, CV 12.0 at η_w=0.3).
- **Action:** change the in-distribution evaluation value in §5.1 to 0.3, or
  clarify that 0.5 is the *training* wind and 0.3 the *evaluation* wind. As
  written, "0.5 for … in-distribution evaluation" contradicts the code that
  produced the reported results. Do NOT restate 0.5 for evaluation.

## 2. Fig 13 / §5.5 corridor non-completion — now fully supported

- Panel (b) of Fig 13 previously could not show a non-completion share (the SD
  tables reported only completed counts). The new **`[CENSOR]` audit** (git HEAD
  `5f99459`) adds `completion%`/`discard%` and a right-censoring check to
  `PENETRATION_HIGH_CENSOR.txt` / `PENETRATION_LOW_CENSOR.txt`.
- Fig 13(b) now plots the **ALL-group discard%** directly: high demand
  0→8.0%, low demand 0.9→10.8% (peak at p=75), which **matches v9:1037–1039**
  verbatim. Right-censoring share = 0% at every p (no aircraft left in corridor
  at sim end), and every discard is a lateral exit (timeout=0), consistent with
  v9:1042–1047. **No text change required** — the figure now agrees with the
  text; just confirm you are pointing Fig 13 at the CENSOR tables.

## 3. Reviewer items carried over (confirm resolved in v9)

These were flagged in earlier rounds; verify the v9 prose still covers them
(figures now encode the evidence):

- **(一) Certificate vs certificate-free separation.** Figs 4/5 show the two
  clusters (CR≈11–12% at effort≈52 vs CR 41/53% at effort 17–19). Line encoding:
  certificate-equipped solid, certificate-free dashed.
- **(二) Stage-1b matched-budget parity.** Fig 6 shows MinSep 47.8/47.7, Effort
  52.9/52.3, max-lat-offset 103.0/103.5, all paired CIs spanning zero
  (v9 `tab:stage1b`) — i.e. the domain-adaptation gain is not a budget artefact.
- **(四) Residual-conflict attribution.** Fig 8 tree: 22/200 conflicts, zero-slack
  re-solve 22/22 hard-infeasible, zero-error replay 0/22 resolved — residual
  conflicts are actuation-limited, not predictor-limited
  (`P1_ORACLE_CONFLICTS.txt`, `ZERO_SLACK_FEAS.txt`).
- **(五) Lead-time degradation.** Fig 11 (v9 `tab:leadtime`): Stage-1b is
  undefined ("None") at 7/10/20 s lead — state this explicitly in the caption so
  the missing markers are not read as zeros.
- **(七) Loose-margin ablation.** Fig 12: CR 56.5/56.0/28.5, 55 resolved / 0
  introduced, McNemar p=5.6e-17 (γ=0.4, Hp=8, a_max=10).
- **(八) Robustness / OOD wind sweep.** Fig 10: η∈{0.5,1.0,1.5}; note the OOD
  sweep uses a separate +0.5 protocol offset (v9:596–602) so PlanGrad CR reads
  flat 12.0 there vs 11.0 in the main table — keep the explanatory sentence.

## 4. Fig 3 vs Cochran

Fig 3(b) reports Cochran Q=5.000, df=3, p=0.172 with Holm-adjusted pairwise
p=1.00 (`STATS_COCHRAN`) — i.e. the four certificate arms are statistically
indistinguishable in CR. Ensure the §5.2 text does not over-claim a CR ranking
among them.

## 6. Fig 3(c) reveals conflicts are difficulty-driven, not method-driven

New panel Fig 3(c) shows the conflict *co-occurrence patterns* across the four
common-planner arms (`conflict_vectors_q2.npz`, n=200, verified this session):

- **25/200** episodes have ≥1 arm in conflict; per-arm conflict counts are
  22/23/25/24 (= main-table CR 11.0/11.5/12.5/12.0%).
- Of those 25 episodes, **21 are all-four-arms-conflict** (every arm conflicts on
  the *same* episode); only 4 episodes distinguish the methods.

**Interpretation (worth one sentence in §5.2):** residual conflicts are driven
almost entirely by **episode difficulty** — a small set of intrinsically hard
encounters where every planner conflicts — rather than by planner-specific
weakness. This directly reinforces the §5.5 / Fig 8 attribution that residual
conflicts are actuation-limited, not predictor-limited. No text change is
required, but stating this makes Fig 3(c) self-explanatory and preempts the
"why do all methods look similar" reviewer question.

## 5. Soft-IPP weight missing on Lab

`04_soft_ipp/soft_joint.pt` is absent, so the collector omits Soft-IPP from the
Fig 4 ECDF (Fig 5 still shows its main-table point). If Soft-IPP must appear in
the ECDF, restore the weight and we re-run the collector. Not a text issue, but
flag if a reviewer asks why Soft-IPP is missing from one panel.
