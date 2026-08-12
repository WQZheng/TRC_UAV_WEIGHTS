#!/usr/bin/env python3
"""The fourth cell: deployment-config-trained Stage-2 under the WEAK planner.

WHY THIS EXPERIMENT EXISTS
  Section 5.4 reports that under the weaker planning configuration
  (gamma, H_p, a_max) = (0.4, 8, 10) Stage-1b conflicts in 56.0% of encounters
  while Stage-2 conflicts in 28.5%, with a strictly nested episode structure
  (55 resolved, 0 introduced). The manuscript reads this as: when the planning
  layer protects less, the task-aligned objective regains operational leverage.

  That configuration is ALSO, exactly, the planner Stage-2 was trained under.
  Verified in source, not inferred:
      train_stage2.py:139   --Hp    default=8
      train_stage2.py:142   --alpha default=0.4
      train_stage2.py:105   a_max=10.0
  So the published contrast confounds two explanations that the current data
  cannot separate:
      A  weak planning regime restores task-aligned leverage
      B  Stage-2 only looks good when the evaluation planner equals its
         training planner (a train-test matching effect)
  Both predict exactly what was observed. The manuscript asserts A.

WHAT THIS SCRIPT DOES
  It fills the empty cell of the 2x2. Three cells already exist:

                              deployment eval     weak eval
      trained @ weak/training      11.0             28.5
      trained @ deployment         13.5              ?

  The '?' discriminates. Nothing is retrained: the deployment-config Stage-2
  checkpoint (stage2_matched.pt, the run behind the published 13.5%) is simply
  evaluated under the weak planner on the identical episode-aligned encounter
  stream used for the published weak-config numbers.

HOW TO READ THE RESULT
  If the deployment-trained checkpoint also lands far below Stage-1b's 56.0%
  (order ~28%), then leverage does not require train-test planner matching and
  explanation A survives, considerably strengthened.
  If it lands near 56.0%, then the published 28.5% was substantially a matching
  effect, and the central positive claim of the paper must be restated as
  "the task-aligned benefit depends on train-deployment planner matching".
  An intermediate value means both mechanisms contribute, and the claim has to
  be split accordingly. The script does not decide the wording; it prints the
  contrast and the paired structure and leaves attribution to the manuscript.

METHOD REUSE
  per_episode_minsep() from mcnemar_final2 is the exact rollout that produced
  the published weak-config MinSep vectors (alpha=0.4, a_max=10, Hp=8, T=20,
  dt=0.2, eta_w=0.3, wind seed 7, d_sep=30, encounters range(2500,3000),
  seed 12345, batch 8). It is imported and called unmodified, so the new cell is
  episode-aligned with the existing three arms and paired tests are legitimate.
  A conflict is DERIVED as min separation < 30 m, never re-declared.

CONTROLS
  The script re-evaluates Stage-1b and the primary Stage-2 through the same path
  and asserts it reproduces the published 56.0% / 28.5% and the published 2x2
  (57, 55, 0, 88) before reporting anything new. If those fail, the rollout path
  has drifted and no new number is trustworthy, so it refuses to write output.
"""
import os
import sys

import numpy as np
import torch

SIM = "/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim"
W = "/data/lab/TRC_UAV_WEIGHTS/Round1/04_weights"
OUT_DIR = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
TXT = "/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim/WEAK_2X2_MATCH.txt"

sys.path.insert(0, SIM)
os.chdir(SIM)                      # checkpoints are resolved relative to here
from mcnemar_final2 import (per_episode_minsep, mcnemar_exact_two_sided,  # noqa
                            D_SEP)

# Published weak-configuration values (Draft_v8 5.4 / fig12), which the
# control arms must reproduce exactly through this same code path.
PUB_WEAK_CR = dict(stage1b=56.0, stage2_primary=28.5)
PUB_WEAK_2X2 = (57, 55, 0, 88)     # rows Stage-1b, cols primary Stage-2
PUB_DEPLOY_MATCHED_CR = 13.5       # stage2_matched under the deployment planner

CKPT = {
    "stage1b":        "stage1b_domainadapt.pt",
    "stage2_primary": "stage2_final.pt",
    "stage2_deploy":  "stage2_matched.pt",
}


def paired(cx, cy):
    return (int((cx & cy).sum()), int((cx & ~cy).sum()),
            int((~cx & cy).sum()), int((~cx & ~cy).sum()))


def main():
    lines = []

    def w(s=""):
        print(s, flush=True)
        lines.append(s)

    w("FOURTH CELL: deployment-trained Stage-2 under the WEAK planner")
    w("  weak planner gamma=0.4 H_p=8 a_max=10 (== the Stage-2 TRAINING planner)")
    w("  n=200 held-out 2500-2999, seed 12345, eta_w=0.3, wind seed 7,")
    w("  d_sep=30 m, T=20, dt=0.2. Conflict := min separation < 30 m.")
    w("  Nothing is retrained; checkpoints are only re-evaluated.")
    w("=" * 74)

    ms, cf = {}, {}
    for name, fn in CKPT.items():
        path = os.path.join(W, fn)
        assert os.path.exists(path), f"missing checkpoint {path}"
        v = per_episode_minsep(path)
        ms[name] = np.asarray(v, dtype=float)
        cf[name] = ms[name] < D_SEP
        w(f"  {name:16s} ({fn:26s})  CR = "
          f"{100.0 * cf[name].mean():5.1f}%   "
          f"MinSep = {ms[name].mean():6.2f} m")

    # ---- controls: the published weak-config numbers must come back ---------
    w()
    for k, want in PUB_WEAK_CR.items():
        got = 100.0 * cf[k].mean()
        if abs(got - want) > 1e-9:
            raise AssertionError(
                f"control arm {k} gives {got:.2f}% but the manuscript reports "
                f"{want}%; the weak-config rollout path has drifted, so the "
                f"new cell cannot be trusted")
    got2x2 = paired(cf["stage1b"], cf["stage2_primary"])
    if got2x2 != PUB_WEAK_2X2:
        raise AssertionError(
            f"published weak 2x2 not reproduced: {got2x2} != {PUB_WEAK_2X2}")
    w("  control: Stage-1b 56.0% and primary Stage-2 28.5% reproduced exactly,")
    w(f"  together with the published paired 2x2 {PUB_WEAK_2X2}")

    # ---- the new contrast ---------------------------------------------------
    cr_1b = 100.0 * cf["stage1b"].mean()
    cr_pr = 100.0 * cf["stage2_primary"].mean()
    cr_dp = 100.0 * cf["stage2_deploy"].mean()

    w()
    w("THE 2x2 (conflict rate, %)")
    w(f"  {'Stage-2 trained at':<24s} {'deployment eval':>16s} {'weak eval':>11s}")
    w(f"  {'weak / training cfg':<24s} {11.0:>15.1f} {cr_pr:>11.1f}")
    w(f"  {'deployment cfg':<24s} {PUB_DEPLOY_MATCHED_CR:>15.1f} {cr_dp:>11.1f}")
    w(f"  {'Stage-1b reference':<24s} {11.5:>15.1f} {cr_1b:>11.1f}")

    a, b, c, d = paired(cf["stage1b"], cf["stage2_deploy"])
    p = mcnemar_exact_two_sided(b, c)
    w()
    w("PAIRED: Stage-1b vs deployment-trained Stage-2, weak config")
    w(f"  both conflict {a}, only Stage-1b {b}, only Stage-2 {c}, neither {d}")
    w(f"  risk difference {cr_dp - cr_1b:+.1f} pp, exact McNemar p = {p:.3e}")
    w(f"  nested (Stage-2 introduces no new conflict)? {'yes' if c == 0 else 'no'}")

    a2, b2, c2, d2 = paired(cf["stage2_primary"], cf["stage2_deploy"])
    p2 = mcnemar_exact_two_sided(b2, c2)
    w()
    w("PAIRED: primary vs deployment-trained Stage-2, weak config")
    w(f"  discordant {b2} / {c2}, risk difference {cr_dp - cr_pr:+.1f} pp, "
      f"exact McNemar p = {p2:.3e}")

    # ---- what it means -----------------------------------------------------
    span = cr_1b - cr_pr                      # 27.5 pp, the published effect
    recovered = (cr_1b - cr_dp) / span if span else float("nan")
    w()
    w(f"  published weak-config effect (Stage-1b -> primary Stage-2): "
      f"{span:.1f} pp")
    w(f"  effect retained by the deployment-trained checkpoint: "
      f"{cr_1b - cr_dp:.1f} pp ({100.0 * recovered:.0f}% of it)")
    w()
    if recovered < 0.0:
        w("  READ: the deployment-trained checkpoint is WORSE than Stage-1b")
        w("  under the weak planner, so the published 28.5% cannot be read as")
        w("  'weak planning restores task-aligned leverage' in general. The")
        w("  effect is specific to the checkpoint whose training planner equals")
        w("  the evaluation planner, i.e. a train-test MATCHING effect. The")
        w("  central positive claim of 5.4 must be restated accordingly.")
    elif recovered >= 0.7:
        w("  READ: the deployment-trained checkpoint retains most of the effect,")
        w("  so leverage does NOT require the evaluation planner to match the")
        w("  training planner. Explanation A (weak planning regime restores")
        w("  task-aligned leverage) survives and is strengthened; the coincidence")
        w("  of the weak and training configurations is not what drives 5.4.")
    elif recovered <= 0.3:
        w("  READ: the deployment-trained checkpoint retains little of the")
        w("  effect, so the published 28.5% was substantially a train-test")
        w("  planner MATCHING effect. The central positive claim must be")
        w("  restated: the task-aligned benefit depends on train-deployment")
        w("  planner matching, not on weak planning protection alone.")
    else:
        w("  READ: the effect is partially retained. Both mechanisms contribute")
        w("  and neither explanation alone is adequate; 5.4 must report this")
        w("  cell and split the claim rather than asserting either A or B.")

    out = os.path.join(OUT_DIR, "weak_2x2_v2.npz")
    np.savez(out,
             cr_weak_stage1b=np.array([cr_1b]),
             cr_weak_stage2_primary=np.array([cr_pr]),
             cr_weak_stage2_deploy=np.array([cr_dp]),
             cr_deploy_stage2_matched=np.array([PUB_DEPLOY_MATCHED_CR]),
             minsep_stage1b=ms["stage1b"],
             minsep_stage2_primary=ms["stage2_primary"],
             minsep_stage2_deploy=ms["stage2_deploy"],
             conflict_stage1b=cf["stage1b"],
             conflict_stage2_primary=cf["stage2_primary"],
             conflict_stage2_deploy=cf["stage2_deploy"],
             paired_1b_vs_deploy=np.array([a, b, c, d]),
             p_1b_vs_deploy=np.array([p]),
             paired_primary_vs_deploy=np.array([a2, b2, c2, d2]),
             fraction_effect_retained=np.array([recovered]),
             weak_cfg=np.array([0.4, 8, 10.0]))
    w()
    w(f"wrote {out}")
    with open(TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {TXT}")


if __name__ == "__main__":
    main()
