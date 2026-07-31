# Referee experiments (loose-config Stage-1b control, episode-level e_par, oracle attribution)

Three decisive evaluation-only experiments run to close referee points #10,
#7b and #4 before the §5.3 / §5.4 / §5.5 wording can be frozen. Nothing retrained.
Main model remains `stage2_final.pt`. Seed 12345, n=200, held-out encounters
range(2500,3000). Loose config: CBF-MPC alpha=0.4, a_max=10, Hp=8, T=20,
dt=0.2, eta_w=0.3, d_sep=30, batch=8.

## P0-1 — Stage-1b under the loose certificate (referee #10)

Motivation: FINAL2 only compared Stage-1 vs Stage-2 under the loose planner,
confounding "encounter data adaptation" with "planner-informative task signal".
Stage-1b (domain-adapted predictor, NO planner term in its loss) is the missing
control that isolates the two. Evaluation-only: Stage-1b's loss has no planner,
so no retrain is needed — just roll it out under the loose config.

Command:
    cd code/plangrad_sim
    export GUAM_MAT=$PWD/../GUAM/Challenge_Problems/Data_Set_1.mat
    python3 p0_stage1b_loose.py     # -> P0_STAGE1B_LOOSE.txt, loose_minsep.pt

Result (three points, loose config):
    Stage-1  (no adapt, no signal)  CR = 56.5 %
    Stage-1b (adapt,   no signal)   CR = 56.0 %
    Stage-2  (adapt +  signal)      CR = 28.5 %
    data-adaptation alone (S1->S1b) = -0.5 pts  (2 % of the total gap)
    planner-info signal (S1b->S2)   = -27.5 pts (98 % of the total gap)
    loose S1b-vs-S2 paired: b=55, c=0 (strictly nested), McNemar p = 5.55e-17

S1/S2 reproduce FINAL2 exactly (56.5 / 28.5; cf. MCNEMAR_FINAL2.txt b=56,c=0,
p=2.78e-17). VERDICT: adaptation alone does NOT close the gap; the
planner-informative signal is what regains leverage under the weak constraint.
§5.5 "prediction signal regains leverage" STANDS and now rests on a
confound-separated three-point control (the data-adaptation alternative
explanation is falsified at 0.5 pt).

## P0-2 — episode-level directional error + direct paired test (referee #7b)

Motivation: diag_error_direction.py pooled ~7 correlated critical-window steps
per episode into ~1400 "samples" and computed SEM=sd/sqrt(1400), understating
the SEM ~sqrt(7)x. #7b: recompute at EPISODE level (n=200 clusters) and add the
never-done direct paired test e_par(S2)-e_par(S1b). Critical window
|idx-CPA|<=3 is model-independent, so the three arms are episode-aligned and
legitimately paired.

Command:
    python3 p0_errdir_episode.py    # -> P0_ERRDIR_EPISODE.txt

Result (episode-level, n_epi=200):
    single-arm e_par mean (SEM):  S1 -0.70 (0.49)  S1b -0.18 (0.24)  S2 -0.34 (0.24)
      -> ALL |mean| < 2*SEM: no single arm is individually significant after
         clustering (S2 t=-1.45). The old step-level "significant S2 bias"
         does NOT survive clustering — #7b is correct.
    DIRECT PAIRED:
      S2 - S1b = -0.164 m  paired-t p=0.0074  Wilcoxon p=0.0011  -> SIGNIFICANT
      S2 - S1  = +0.359 m  paired-t p=0.40    Wilcoxon p=0.56    -> n.s.

VERDICT for §5.4: do NOT claim "S2 shows a significant conservative bias"
(single-arm significance is gone). Instead state the CONTROLLED contrast that
IS significant: relative to the domain-adaptation control S1b, S2's
critical-window error is biased toward the ego by -0.16 m (paired-t p=0.007,
Wilcoxon p=0.001, n=200 episodes); single-arm episode means are directionally
consistent but not individually significant after clustering. The fingerprint
is thus an UPGRADED paired inference, not a downgrade to descriptive — and it is
only defensible against S1b (same data, task signal removed), NOT against S1
(where |e_par| is actually larger and the paired diff is +0.36 m, n.s.),
consistent with the P0-1 logic that the task signal must be isolated with S1b.

## P1 — oracle-prediction re-run of the deploy-config conflict episodes (referee #4)

Motivation: "22/22 infeasible" (all main-table PlanGrad conflicts had the CBF QP
relax its separation constraint) shows the PLANNER ran out of avoidance
authority, but does not by itself prove a better PREDICTOR could not have helped
-- that is a claim about the realized prediction stream, not prediction quality
in general. The referee's response tried to close this with the leadtime 2s row
(Oracle=Stage-2=11.0%), but that row is measured under a fixed-CPA controlled
geometry that is NOT the main-table 22-episode geometry -- cross-setting
borrowing, not a substitute. The clean test: replay the SAME 22 episodes with an
ORACLE predictor (true future neighbour window) feeding the SAME planner and SAME
plant.

Deploy / main-table config (reproduces BEST.txt CR=11.0%): CBF-MPC alpha=0.1,
Hp=15, a_max=20, d_sep=30, T=20, dt=0.2, eta_w=0.3, n=200, seed=12345.
Implementation: OraclePolicy subclasses SafePolicy and overrides only the
prediction step to return nf[:, :, t+1:t+1+Hp+1, :] (the true absolute future
neighbour positions); mpc.solve and the inner control law are byte-identical to
the deploy planner.

Command:
    python3 p1_oracle_conflicts.py     # -> P1_ORACLE_CONFLICTS.txt

Result:
    Stage-2 (real)   CR = 11.0 % (22/200)   [reproduces main table]
    Oracle (perfect) CR = 12.0 % (24/200)   [a perfect predictor is NO better;
                                             it even breaks 2 marginal episodes]
    3-way attribution of the 22 Stage-2 conflicts:
      (A) actuation-limited / hard-infeasible under any prediction = 22
      (B) prediction-induced (oracle avoids)                       =  0
    CBF QP relaxed its separation constraint (slack>0) at closest approach in
    20/22 episodes; hardest case epi 85: minSep 10.9 m, oracle slack 5.29.

VERDICT for §5.3 / #4: every one of the 22 conflicts ALSO occurs under a perfect
predictor on the identical geometry, so "hard-infeasible under the realized
prediction stream" is now episode-level fact, not inference. Write: replaying all
22 conflict episodes with an oracle predictor under the identical planner and
plant leaves every one in conflict (oracle CR 12.0% >= Stage-2 11.0%); the
binding limit is the planner's avoidance authority (QP slack active at CPA in
20/22), not prediction quality. The leadtime 2s Oracle=Stage-2 row is a
population-level CORROBORATION under a fixed-CPA setting, explicitly NOT a
substitute for this episode-level test. Soften "no improvement of the predictor
could have changed that" -> "hard-infeasible under the realized prediction
stream".

## Second-pass checklist experiments (TODO-1BH / COV / 1BP / DEV)

All deploy/main-table config (CBF-MPC alpha=0.1, Hp=15, a_max=20, d_sep=30,
T=20, dt=0.2, eta_w=0.3, n=200, seed=12345, pool 2500-3000), evaluation-only.

### [TODO-1BH] Stage-1b detection-horizon at 1s/2s (was missing from LEADTIME_S1B)
LEADTIME_S1B.txt only had 3-6s (all 0.0%); the informative rows are 1-2s.
    python3 eval_leadtime.py --stage1 stage1_full.pt \
        --stage2 stage1b_domainadapt.pt --horizons 1,2 --out LEADTIME_S1B_1_2.txt
Result (S2 column = Stage-1b here):
    1s: S1 86.0 / S1b 83.0 / Oracle 82.0
    2s: S1 13.0 / S1b 11.5 / Oracle 11.0
With main LEADTIME.txt (S2 1s=82.5, 2s=11.0) the full 2s matched-control row is
S1 13.0 / S1b 11.5 / S2 11.0 / Oracle 11.0: at the tactical horizon the matched
control and the task-aligned predictor are both actuation-bound and sit on the
oracle. VERDICT: the matched-control leadtime claim is SUPPORTED -> add the
Stage-1b row to the leadtime table; do NOT delete the claim.

### [TODO-COV] empirical pooled coverage on the evaluation stream
    python3 cov_validate.py                 # -> COVERAGE_VALID.txt
Target coverage 90% (delta=0.1), calibrated r_conf=18.93 m (n_calib=3000,
independent seed 777). Empirical pooled coverage on the eval stream (seed 12345,
3000 scores) = 90.8% (p90 score 18.65 m ~ r_conf). VERDICT: calibration VALID on
held-out; report the number in the results chapter (no need to defer to appendix).

### [TODO-1BP] deploy-config paired McNemar Stage-1b vs Stage-2 (from per-episode)
### [TODO-DEV] route (cross-track) deviation of the matched pair
    python3 p2_mcnemar_dev.py               # -> P2_MCNEMAR_DEV.txt
Deploy CR: S1 12.5 / S1b 11.5 / S2 11.0 / S2(matched) 13.5.
1BP paired (per-episode, NOT inferred from net CR):
  S1b vs S2: discordant b=2 c=1, McNemar p=1.0; paired min-sep diff
    (S2 - S1b) = -0.04 m, 95% CI [-0.35, +0.27] (includes 0)
    -> Stage-1b and Stage-2 statistically indistinguishable under deploy; direct
       paired evidence for the decoupling / negative joint-training result.
  S1 vs S2: b=3 c=0, McNemar p=0.25; paired min-sep diff +1.41 m [+0.59,+2.22].
DEV route cross-track deviation (metres, mean +/- SD over 200 episodes):
  per-episode MAX xtrack: S1 99.65+/-24.32, S2 103.53+/-11.27,
    S2(matched) 115.01+/-11.15
  matched-pair contrast S2(final) - S2(matched) on MAX xtrack = -11.47 m,
    95% CI [-13.43, -9.51] (excludes 0) -> Stage-2 does NOT buy behaviour with a
    larger lateral excursion; it is SMALLER than the matched control's. The
    "yaw-to-buy-performance" concern is refuted.
  S2(final) - S1 on MAX xtrack = +3.88 m [+1.26,+6.50] (S1 xtrack SD 24.3 m
    reflects the OOD predictor's erratic manoeuvres).

## Mixed-traffic penetration analogy references (verified CrossRef+OpenAlex)
For §07 mixed-traffic analogy: peng2025enhancing (2025), wang2024energy (2024),
hou2023evaluating (2023), mohammed2023vehicle (2023). NOTE the penetration data
(PENETRATION.txt) show unequipped CR is NOT monotone in p (50.3->59.7->51.7->53.7
high demand), so any "positive/monotone externality" wording is a FACTUAL error
and must be removed; report the penetration study descriptively (#11).

## Third-pass fixes (referee回执: DEV corrected, penetration SD, recap numbers)

### [DEV corrected] the earlier DEV compared the WRONG control -- fixed here
The referee打回: "matched control" = Stage-1b throughout the paper, NOT
stage2_matched.pt. The earlier P2 run reported S2(final) vs S2(matched); that
belongs in the appendix and must not carry the name "matched control" into 5.4.
Corrected run (dev_matched_s1b.py, deploy config, same 200 episodes):
  Stage-1b (matched control)  MAX lateral offset = 103.02 +/- 16.84 m
  Stage-2  (final)            MAX lateral offset = 103.53 +/- 11.27 m
  matched-pair S2(final) - S1b = +0.52 m, 95% CI [-1.11, +2.14] (includes 0)
  -> Stage-2 does NOT buy behaviour with a larger lateral excursion than the
     matched control Stage-1b; the "yaw-to-buy-performance" concern is refuted.
  METRIC NAME: this is the per-episode MAXIMUM lateral offset, NOT Phi_dev (the
  training functional is time-AVERAGED); label the table entry accordingly.
  tab:stage1b deviation row: S1b 103.02+/-16.84 m, S2 103.53+/-11.27 m.
The old S2(final)-S2(matched) = -11.47 m contrast is a valid APPENDIX robustness
observation (config-matched retrain uses a LARGER lateral excursion), but under
its own name, not "matched control".

### [TODO-PEN] penetration re-run with Thr/Delay SD (run_penetration_sd.py)
Original PENETRATION*.txt recorded SD only for CR; agg() now also emits
throughput/delay SD. CR values reproduce the originals exactly.
LOW demand (arrival=0.06, reps=6, horizon=400, warmup=100, K=3, seed=12345):
  ALL Thr/Delay by p: 0%=18.67+/-4.27,2.9+/-0.9; 25%=18.50+/-1.71,2.3+/-0.5;
  50%=18.00+/-2.83,3.1+/-0.6; 75%=15.17+/-2.34,1.6+/-0.9; 100%=16.83+/-5.08,1.4+/-0.5.
  (HIGH demand arrival=0.16 rerun in progress; PENETRATION_SD.txt appended on完成.)
Protocol params (both regimes): single bidirectional corridor, two Poisson
arrival streams, fixed demand across the sweep, equipped=SafePolicy /
unequipped=ORCA through identical 6-DOF dynamics; K=3 nearest neighbours;
pass=reach far end; delay=realised-freeflow travel time; throughput=passes/min;
conflict=any pair below d_sep at any tick; reps=6; warmup=100; horizon=400 steps.

### Recap numbers to the referee
  1. S1b leadtime: 1s = 83.0% (2s = 11.5%); 7/10/20s not run (table dashes ok).
  2. COVERAGE n_eval = 3000 pooled evaluation scores (calibration n=3000, indep).
  3. penetration Thr/Delay SD: see above (LOW done; HIGH appended on完成).
  4. S1b MAX lateral offset = 103.02 +/- 16.84 m (see DEV corrected).

## [TODO-Q2] omnibus test on the CORRECT arm set -- "among the arms" LICENSED
The global claim is about the four arms that share ONE deploy planner:
{PlanGrad, Stage-1b, Fixed-Predictor, Constant-Velocity}. The first Cochran run
[TODO-Q] used the WRONG set (it included Conformal-MPC, which changes the planner
threshold d_sep+r_conf and was removed from the fixed-planner ranking, and it
OMITTED Stage-1b). stats_tests_cochran_q2.py fixes the set: it reuses the
PG/Fixed/CV per-episode vectors from conflict_vectors.npz and recomputes the
Stage-1b vector with the deploy rollout of p2_mcnemar_dev.py verbatim
(stage1b_domainadapt.pt, Hp=15/d_sep=30/T=20/dt=0.2/eta_w=0.3/batch=8), which
reproduces the P2 result Stage-1b CR = 11.5% (23/200) exactly.
  CR: PlanGrad 11.0 (22/200) / Stage-1b 11.5 (23/200) / Fixed 12.5 (25/200) /
      Constant-Velocity 12.0 (24/200).
  Cochran's Q (omnibus, H0 = four common-planner arms equal): Q=5.000, df=3,
    p=0.172 (n.s.).
  All 6 exact McNemar edges, Holm-Bonferroni: EVERY edge n.s. (Holm=1.000),
    including the two edges the claim needs and Q1 lacked: Stage-1b-Fixed
    (disc 2/0, p=0.500) and Stage-1b-CV (disc 1/0, p=1.000).
  -> omnibus null NOT rejected + all corrected pairwise edges n.s. => the four
     COMMON-PLANNER arms are JOINTLY indistinguishable on conflict rate; the
     global "among the arms ... statistically indistinguishable" wording is
     LICENSED and may be restored from the narrowed "prespecified contrasts".
Per-episode 0/1 vectors: conflict_vectors_q2.npz (correct set). Files:
STATS_COCHRAN_Q2.txt, stats_tests_cochran_q2.py, conflict_vectors_q2.npz.

## [TODO-Q] (SUPERSEDED by Q2 -- kept for the secondary Conformal statement)
The Conformal-inclusive run {PG, Conformal, Fixed, CV} (Q=4.286, df=3, p=0.232,
all six Holm edges n.s.) does NOT test the fixed-planner claim (wrong arm set),
but it separately LICENSES a secondary statement: even with the margin baseline
(Conformal-MPC) included, ALL certificate-equipped arms are jointly
indistinguishable on conflict rate. Use this as a secondary sentence only, not
for the primary "among the (common-planner) arms" claim.
The RQ1 claim "no CBF-equipped method is best AMONG THE ARMS" is a GLOBAL,
all-arms statement; STATS.txt only ran three PAIRWISE McNemar edges (PlanGrad vs
each other arm) and never an omnibus test or the three remaining edges.
stats_tests_cochran.py reuses the STATS.txt harness verbatim (same held-out
encounters, seed 12345, n=200, deploy planner alpha=0.1/Hp=15/a_max=20,
conflict = per-episode min-sep < 30 m; CR reproduces STATS.txt exactly:
PlanGrad 11.0 / Conformal 11.5 / Fixed 12.5 / CV 12.0) and adds:
  Cochran's Q (omnibus, H0 = all four arms equal): Q=4.2857, df=3, p=0.232 (n.s.)
  All 6 exact McNemar edges, Holm-Bonferroni: EVERY edge n.s. (Holm=1.000),
    including the three never-tested edges Conformal-Fixed (p=0.500),
    Conformal-CV (p=1.000), Fixed-CV (p=1.000).
  -> omnibus null NOT rejected + all corrected pairwise edges n.s. => the four
     CBF-equipped arms are JOINTLY indistinguishable on conflict rate. The global
     "among the arms ... statistically indistinguishable" wording is LICENSED and
     may be restored from the narrowed "three prespecified contrasts" fallback.
Per-episode 0/1 vectors serialised to conflict_vectors.npz (reproducible without
re-running rollouts). Files: STATS_COCHRAN.txt, stats_tests_cochran.py,
conflict_vectors.npz.

## Fourth-pass fixes (referee: effort, discard, ORCA clearance, seed dimension)

### [TODO-EFF] matched-pair control effort -- fourth run-time quantity, no increment
The "no detected increment" enumeration must cover all four run-time quantities
(conflict, separation, EFFORT, yaw). Effort was the missing one. eff_matched_s1b.py
recomputes the manuscript's normalised control energy per episode
(sum_t [((u0-weight)/weight)^2 + ||u_{1:4}/max_body_moment||^2], identical to
eval_common) on the matched pair (Stage-1b vs Stage-2(final)), deploy config,
same held-out encounters (paired):
  Stage-1b (matched control)  effort = 52.8983 +/- 11.8061  (mean +/- SD, n=200)
  Stage-2  (final)            effort = 52.3487 +/-  7.5702
  paired diff S2 - S1b = -0.5496, SE=0.5745, 95% CI [-1.6824, +0.5832] (includes 0)
  -> Stage-2 draws NO measurable extra control effort than the matched control.
All four run-time quantities now show a paired contrast whose CI includes 0:
  conflict (Cochran Q p=0.172), separation (-0.04 m [-0.35,+0.27]),
  effort (-0.55 [-1.68,+0.58]), yaw/max-lateral-offset (+0.52 m [-1.11,+2.14]).

### [TODO-DISC] penetration completion / discard rate per p (survivor-bias audit)
run_penetration_disc.py adds completion=passed/n and discard=1-completion to the
existing sweep (CR/Thr/Delay reproduce the SD run point-by-point). An agent is
discarded if it leaves the corridor (|y|>3*halfwidth) or times out
(travel_steps>4*free_flow_steps); discarded agents ARE in the CR denominator but
NOT in throughput/delay (which condition on passing) -- hence the survivor-bias
caveat this quantifies.
LOW demand (arrival=0.06): completion by p = 99.1 / 97.4 / 91.5 / 89.2 / 90.2 %
  (discard 0.9 / 2.6 / 8.5 / 10.8 / 9.8 %). Discard rises with p but peaks ~11%.
  (HIGH demand arrival=0.16 rerun in progress; PENETRATION_DISC.txt appended on完成.)

### [CHECK-ORCA] pairwise clearance = 30 m (combined-disc), NOT 60 m
orca_baseline.py: self.radius = d_sep/2 = 15 m per agent, and the collision test
uses comb_r = 2*radius = d_sep = 30 m (lines 41/57/60/65/154). So the ORCA
minimum PAIRWISE separation (centre-to-centre) is 30 m -- exactly the same
clearance the CBF-MPC arm enforces (d_sep=30). The "same pairwise clearance"
sentence STANDS; radius=15 is the per-agent disc, not the pairwise clearance.

### [CHECK-P] seed formula: p is a FRACTION (0-1), not a percent
run_penetration_sd.py L57: ps = [int(x)/100.0 for x in args.ps.split(",")], so
p in {0.0,0.25,0.50,0.75,1.0}; the seed L78 = base + 1000*rep + int(p*97), giving
floor(97 p) in {0,24,48,72,97}. If p were read as a percent (25), floor(97*25)
=2425 -- two orders of magnitude off. Reproducibility text MUST state p in [0,1].

Files: EFF_MATCHED_S1B.txt, eff_matched_s1b.py, PENETRATION_LOW_DISC.txt,
PENETRATION_DISC.txt (high, on完成), run_penetration_disc.py.

## Version discipline
Authoritative source = 05_experiments_results_v3.tex (holds all E/F/P batches).
v4 is a divergent fork (it reverted the TASL weight placeholders) and is to be
ABANDONED; do NOT edit v4. All numbers above land in v3, then v4 is diffed for
any v4-only content and retired.

Files: P0_STAGE1B_LOOSE.txt, P0_ERRDIR_EPISODE.txt, P1_ORACLE_CONFLICTS.txt,
LEADTIME_S1B_1_2.txt, COVERAGE_VALID.txt, P2_MCNEMAR_DEV.txt,
DEV_MATCHED_S1B.txt, PENETRATION_LOW_SD.txt, PENETRATION_SD.txt (high, on完成),
loose_minsep.pt, p0_stage1b_loose.py, p0_errdir_episode.py,
p1_oracle_conflicts.py, cov_validate.py, p2_mcnemar_dev.py, dev_matched_s1b.py,
run_penetration_sd.py, refs_mixedtraffic.bib.
