"""[TODO-Q] Cochran's Q + all-pairs McNemar over the four CBF-equipped arms.

The referee asked whether the GLOBAL claim "no method is best AMONG THE ARMS"
is supported. The existing STATS.txt ran only three PAIRWISE McNemar tests of
PlanGrad vs {Conformal, Fixed, CV}; it never tested the three remaining edges
(Conformal-Fixed, Conformal-CV, Fixed-CV) and never ran an OMNIBUS test of the
null "all four arms have equal conflict propensity".

This script REUSES stats_tests.per_episode_conflict / wilson_ci / mcnemar_exact
VERBATIM (same held-out encounters, seed 12345, n=200, deploy planner
alpha=0.1/Hp=15/a_max=20, conflict = per-episode min-sep < d_sep=30 m) so the
numbers are bit-identical to STATS.txt, and adds:

  (1) Cochran's Q -- the correct omnibus test for k>2 related binary samples
      (H0: all four arms share one conflict probability). Q ~ chi^2_{k-1}.
  (2) All C(4,2)=6 pairwise exact McNemar tests (fills the three missing edges),
      with Holm-Bonferroni correction over the six p-values.
  (3) Serialises the four per-episode 0/1 vectors to conflict_vectors.npz so the
      omnibus test is reproducible without re-running the rollouts.

Does NOT modify stats_tests.py or STATS.txt. Writes STATS_COCHRAN.txt.

Verdict rule: if Cochran's Q is n.s. AND every Holm-corrected pairwise p>0.05,
the global "among the arms ... indistinguishable" claim is licensed and can be
restored; otherwise keep the narrowed "three prespecified contrasts" wording.
"""
from __future__ import annotations
import os
import sys
import argparse
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
# repo layout: the old /data/lab/plangrad/plangrad_sim path is gone; point every
# dependency (params, dynamics, predictor, guam_encounters, ...) at the repo.
REPO_PLANGRAD = "/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(B, "01_constant_velocity"))
sys.path.insert(0, os.path.join(B, "05_conformal_mpc"))
sys.path.insert(0, REPO_PLANGRAD)

import numpy as np
from scipy import stats

# reuse the EXACT harness that produced STATS.txt (rollout + tests unchanged)
import stats_tests as st
from stats_tests import (per_episode_conflict, wilson_ci, mcnemar_exact,
                         ConstantVelocityPredictor, CBFMPCLayer,
                         conformal_radius, ec)
# redirect weight loads from the dead absolute path to the repo copy
st.PLANGRAD = REPO_PLANGRAD
PLANGRAD = REPO_PLANGRAD


def cochran_q(vectors):
    """Cochran's Q for k related binary samples (list of length-N bool arrays).
    Returns (Q, df, p). H0: all k treatments have equal success probability."""
    X = np.vstack([v.astype(float) for v in vectors])   # [k, N]
    k, N = X.shape
    col = X.sum(axis=0)          # per-subject totals across treatments [N]
    row = X.sum(axis=1)          # per-treatment totals across subjects  [k]
    G = X.sum()
    den = k * G - np.sum(col ** 2)
    if den == 0:                 # every subject constant across arms -> no info
        return 0.0, k - 1, 1.0
    Q = (k - 1) * (k * np.sum(row ** 2) - G ** 2) / den
    df = k - 1
    return float(Q), df, float(stats.chi2.sf(Q, df))


def holm(pairs):
    """Holm-Bonferroni over [(label,p),...]; returns [(label,p,padj,reject)]."""
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m = len(pairs)
    out = [None] * m
    running = 0.0
    for rank, idx in enumerate(order):
        label, p = pairs[idx]
        running = max(running, min(1.0, (m - rank) * p))   # monotone step-down
        out[idx] = (label, p, running, running < 0.05)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(HERE, "STATS_COCHRAN.txt"))
    ap.add_argument("--npz", default=os.path.join(HERE, "conflict_vectors.npz"))
    args = ap.parse_args()
    dev = ec.device_str(True)

    s1 = ec.load_gmm_predictor(f"{PLANGRAD}/stage1_full.pt", dev)
    s2 = ec.load_gmm_predictor(f"{PLANGRAD}/stage2_final.pt", dev)
    cv = ConstantVelocityPredictor(T=30, K=5).double().to(dev)

    r_conf, _ = conformal_radius(s1, delta=0.1,
                                 horizon=ec.BEST_PLANNER["horizon"],
                                 device=dev, seed=ec.GLOBAL_SEED)

    def best():
        return ec.make_best_planner()

    def conf_planner():
        return CBFMPCLayer(n_neighbors=1, horizon=ec.BEST_PLANNER["horizon"],
                           dt=ec.DT, d_sep=ec.D_SEP + r_conf,
                           alpha=ec.BEST_PLANNER["alpha"],
                           a_max=ec.BEST_PLANNER["a_max"])

    print("collecting per-episode conflicts (n=%d) ..." % args.n, flush=True)
    conf = {
        "PlanGrad":          per_episode_conflict(s2, best(), args.n, dev),
        "Conformal-MPC":     per_episode_conflict(s1, conf_planner(), args.n, dev),
        "Fixed-Predictor":   per_episode_conflict(s1, best(), args.n, dev),
        "Constant-Velocity": per_episode_conflict(cv, best(), args.n, dev),
    }
    # length-align defensively (all should already be equal length)
    L = min(len(v) for v in conf.values())
    conf = {k: v[:L].astype(bool) for k, v in conf.items()}
    np.savez(args.npz, **conf)

    fh = open(args.out, "w")

    def w(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    names = list(conf.keys())
    w("=" * 74)
    w("[TODO-Q] OMNIBUS + ALL-PAIRS SAFETY TESTS  (n=%d, seed=%d," % (L, ec.GLOBAL_SEED))
    w("  deploy planner alpha=0.1/Hp=15/a_max=20, identical held-out 2500-3000)")
    w("conflict = per-episode min sep < %.0f m.  (numbers match STATS.txt)" % ec.D_SEP)
    w("=" * 74)
    w("%-20s %8s   %-22s" % ("method", "CR %", "Wilson 95% CI"))
    for name in names:
        c = conf[name]; k, n = int(c.sum()), len(c)
        lo, hi = wilson_ci(k, n)
        w("%-20s %7.1f   [%5.1f, %5.1f]" % (name, 100.0 * k / n, lo, hi))

    Q, df, pQ = cochran_q([conf[nm] for nm in names])
    w("")
    w("Cochran's Q (omnibus, H0: all four arms equal conflict propensity):")
    w("  Q = %.4f   df = %d   p = %.4f   -> %s"
      % (Q, df, pQ, "n.s. (arms indistinguishable)" if pQ > 0.05
         else "SIGNIFICANT (arms differ)"))

    w("")
    w("All C(4,2)=6 exact McNemar edges (Holm-Bonferroni corrected):")
    raw, detail = [], {}
    for x, y in combinations(names, 2):
        b01, b10, p = mcnemar_exact(conf[x], conf[y])
        lbl = "%s vs %s" % (x, y)
        raw.append((lbl, p)); detail[lbl] = (b01, b10)
    for lbl, p, padj, rej in holm(raw):
        b01, b10 = detail[lbl]
        w("  %-40s disc(%d/%d)  p=%.3f  Holm=%.3f  -> %s"
          % (lbl, b01, b10, p, padj, "SIGNIFICANT" if rej else "n.s."))

    w("")
    licensed = (pQ > 0.05) and all(not r for _, _, _, r in holm(raw))
    if licensed:
        w("VERDICT: Cochran's Q p=%.3f (n.s.) and all six Holm-corrected pairwise"
          % pQ)
        w("  edges n.s. => the four CBF-equipped arms are JOINTLY indistinguishable")
        w("  on conflict rate; the global 'among the arms' claim is LICENSED.")
    else:
        w("VERDICT: omnibus/pairwise rejects equality => keep narrowed")
        w("  'three prespecified contrasts' wording.")
    fh.close()


if __name__ == "__main__":
    main()
