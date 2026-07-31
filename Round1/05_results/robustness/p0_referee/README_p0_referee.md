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

Files: P0_STAGE1B_LOOSE.txt, P0_ERRDIR_EPISODE.txt, P1_ORACLE_CONFLICTS.txt,
loose_minsep.pt (per-episode min-sep S1/S1b/S2 under loose config),
p0_stage1b_loose.py, p0_errdir_episode.py, p1_oracle_conflicts.py.
