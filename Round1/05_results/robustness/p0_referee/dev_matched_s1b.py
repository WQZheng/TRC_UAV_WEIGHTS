"""[TODO-DEV corrected] Route (maximum lateral offset) of the TRUE matched pair.

CORRECTION to the earlier P2 run: the referee's question is "does Stage-2 buy
its behaviour with a larger lateral excursion than the MATCHED CONTROL", and the
matched control is Stage-1b (domain adaptation, no planner term) throughout the
paper -- NOT stage2_matched.pt (the appendix config-matched retrained Stage-2).
The earlier run compared S2(final) vs S2(matched); that contrast belongs in the
appendix robustness note and must not carry the name "matched control" into 5.4.
Here we compute the matched-pair contrast Stage-2(final) vs Stage-1b on the
IDENTICAL 200-episode deploy stream so tab:stage1b's deviation row can be filled.

Metric note: we report the per-episode MAXIMUM lateral offset (perpendicular
distance of the realized position from the straight-line initial reference). This
is NOT the training functional Phi_dev, which is a time-AVERAGED deviation; label
the table entry "maximum lateral offset", not Phi_dev.

Deploy config: CBF-MPC alpha=0.1, Hp=15, a_max=20, d_sep=30, T=20, dt=0.2,
eta_w=0.3, n=200, seed=12345, pool 2500-3000. Evaluation-only; main model stays
stage2_final.pt. Reuses p2_mcnemar_dev.rollout unchanged (byte-identical rollout).
Writes DEV_MATCHED_S1B.txt.
"""
from __future__ import annotations
from p2_mcnemar_dev import rollout, paired_diff_ci, N

OUT = "DEV_MATCHED_S1B.txt"


def w(s):
    with open(OUT, "a") as f:
        f.write(s + "\n")
    print(s, flush=True)


def ms(v):
    return float(v.mean()), float(v.std(unbiased=True))


if __name__ == "__main__":
    open(OUT, "w").close()
    w("[TODO-DEV corrected] maximum lateral offset, TRUE matched pair")
    w("  matched control = Stage-1b (domain adaptation); deploy config,")
    w("  n=%d seed=12345; metric = per-episode MAX lateral offset (NOT Phi_dev)" % N)
    w("")

    sep1b, xm1b, xa1b = rollout("stage1b_domainadapt.pt")
    sep2,  xm2,  xa2  = rollout("stage2_final.pt")

    for nm, vmax, vmean in [("Stage-1b (matched control)", xm1b, xa1b),
                            ("Stage-2 (final)", xm2, xa2)]:
        mmx, smx = ms(vmax); mmn, smn = ms(vmean)
        w("  %-27s  MAX lateral offset = %6.2f +/- %5.2f m   "
          "MEAN lateral offset = %5.2f +/- %5.2f m"
          % (nm, mmx, smx, mmn, smn))
    w("")

    md, (lo, hi) = paired_diff_ci(xm2, xm1b)   # S2(final) - S1b, per-episode
    w("MATCHED-PAIR contrast (the referee's question):")
    w("  Stage-2(final) - Stage-1b on MAX lateral offset = %+.2f m  95%% CI [%+.2f, %+.2f]"
      % (md, lo, hi))
    if lo <= 0 <= hi:
        w("  -> CI includes 0: Stage-2 does NOT buy behaviour with a larger")
        w("     lateral excursion than the matched control Stage-1b.")
    elif md > 0:
        w("  -> CI excludes 0, positive: Stage-2 DOES use a larger lateral")
        w("     excursion than the matched control (report honestly).")
    else:
        w("  -> CI excludes 0, negative: Stage-2 uses a SMALLER lateral excursion")
        w("     than the matched control.")
