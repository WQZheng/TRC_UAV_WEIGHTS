"""P0-1: Stage-1b under the LOOSE (FINAL2) planner config.

Referee point #10: FINAL2 only compared Stage-1 vs Stage-2 under the loose
certificate, confounding "data adaptation" with "planner-informative signal".
Stage-1b (domain-adapted predictor, NO planner term in its loss) is the missing
control. We reuse the EXACT loose-config rollout from mcnemar_final2 and evaluate
all three checkpoints on the identical held-out encounter stream so the three
points S1 / S1b / S2 are episode-aligned.

Loose config (reproduces FINAL2): CBF-MPC alpha=0.4, a_max=10, Hp=8, T=20,
dt=0.2, eta_w=0.3, d_sep=30, batch=8, encounters range(2500,3000), seed=12345.

Decision rule (the fork, computed NOT assumed):
  * If S1b CR ~ 50% (near S1 56.5%)  -> data adaptation alone does NOT close the
    gap; the planner-informative signal (S1->S2) is what regains leverage under
    the weak constraint. 5.5 "prediction signal regains leverage" stands.
  * If S1b CR ~ 28% (near S2 28.5%)  -> data adaptation alone already recovers
    most of the gain; 5.5 must be rewritten as "encounter adaptation regains
    leverage", NOT the joint planner signal.

Also emits the loose-config S1b-vs-S2 paired 2x2 + exact McNemar, and saves
per-episode min-sep for all three arms to loose_minsep.pt (reused by P0-2 gate).
Writes P0_STAGE1B_LOOSE.txt. Retrains nothing.
"""
import torch
from math import comb
from mcnemar_final2 import per_episode_minsep, mcnemar_exact_two_sided, N, D_SEP

OUT = "P0_STAGE1B_LOOSE.txt"


def w(line):
    with open(OUT, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def paired(cx, cy):
    a = int((cx & cy).sum())
    b = int((cx & ~cy).sum())
    c = int((~cx & cy).sum())
    d = int((~cx & ~cy).sum())
    return a, b, c, d


if __name__ == "__main__":
    open(OUT, "w").close()
    w("P0-1  Stage-1b under LOOSE config (referee #10 control)")
    w("  CBF-MPC alpha=0.4 a_max=10 Hp=8, T=20 dt=0.2 eta_w=0.3 d_sep=30,")
    w("  held-out 2500-3000, seed=12345, n=%d" % N)
    w("")

    s1 = per_episode_minsep("stage1_full.pt")
    s1b = per_episode_minsep("stage1b_domainadapt.pt")
    s2 = per_episode_minsep("stage2_final.pt")
    assert len(s1) == len(s1b) == len(s2) == N

    torch.save({"s1": s1, "s1b": s1b, "s2": s2, "d_sep": D_SEP, "n": N},
               "loose_minsep.pt")

    c1 = s1 < D_SEP
    c1b = s1b < D_SEP
    c2 = s2 < D_SEP
    cr1 = 100.0 * c1.float().mean().item()
    cr1b = 100.0 * c1b.float().mean().item()
    cr2 = 100.0 * c2.float().mean().item()

    w("THREE-POINT CR (loose config):")
    w("  Stage-1  (no adapt, no signal) CR = %.1f%%" % cr1)
    w("  Stage-1b (adapt,   no signal)  CR = %.1f%%" % cr1b)
    w("  Stage-2  (adapt +  signal)     CR = %.1f%%" % cr2)
    w("")
    w("  gap S1 -> S1b (data adaptation)      = %+.1f pts" % (cr1b - cr1))
    w("  gap S1b -> S2 (planner-info signal)  = %+.1f pts" % (cr2 - cr1b))
    w("  total S1 -> S2                       = %+.1f pts" % (cr2 - cr1))
    frac = (cr1 - cr1b) / (cr1 - cr2) * 100.0 if (cr1 - cr2) != 0 else float("nan")
    w("  share of total gap explained by adaptation alone = %.0f%%" % frac)
    w("")

    # loose-config S1b vs S2 paired McNemar
    a, b, c, d = paired(c1b, c2)
    p = mcnemar_exact_two_sided(b, c)
    w("Paired 2x2 loose (rows S1b, cols S2):")
    w("                 S2 conflict   S2 safe")
    w("  S1b conflict       %4d        %4d" % (a, b))
    w("  S1b safe           %4d        %4d" % (c, d))
    w("discordant: b(S1b-conf/S2-safe)=%d  c(S1b-safe/S2-conf)=%d" % (b, c))
    w("McNemar exact two-sided p(S1b vs S2, loose) = %.3e" % p)
    w("")

    # interpretation guard (printed, not decided by me)
    if abs(cr1b - cr1) <= abs(cr1b - cr2):
        w("READ: S1b closer to S1  -> adaptation alone does NOT close gap;")
        w("      planner-informative signal regains leverage. 5.5 STANDS.")
    else:
        w("READ: S1b closer to S2  -> adaptation alone recovers most gain;")
        w("      REWRITE 5.5 as 'encounter adaptation regains leverage'.")
