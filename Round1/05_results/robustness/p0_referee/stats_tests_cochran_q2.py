"""[TODO-Q2] Cochran's Q + all-pairs McNemar over the CORRECT four common-planner
arms {PlanGrad, Stage-1b, Fixed-Predictor, Constant-Velocity}.

Correction to [TODO-Q]: the global "no method is best among the arms" claim is
about the four arms that share ONE deploy planner (alpha=0.1/Hp=15/a_max=20).
Conformal-MPC changes the planner threshold (d_sep + r_conf) and was explicitly
removed from the fixed-planner ranking, so it does NOT belong in this set; the
arm that DOES belong -- Stage-1b (domain-adaptation only) -- was missing from
the first Cochran run. That first run therefore tested two irrelevant edges
(Conformal-Fixed, Conformal-CV) and omitted the two edges the claim needs
(CV-Stage1b, Fixed-Stage1b). This script fixes the arm set.

Reuses:
  * PlanGrad / Fixed-Predictor / Constant-Velocity per-episode 0/1 vectors from
    conflict_vectors.npz (already collected, deploy planner, n=200, seed 12345);
  * Stage-1b per-episode conflict recomputed here from stage1b_domainadapt.pt
    with the IDENTICAL deploy rollout used by p2_mcnemar_dev.py (Hp=15, d_sep=30,
    T=20, dt=0.2, eta_w=0.3, batch=8) so conflict = (per-episode min-sep < 30 m);
    this reproduces the P2 result S1b CR = 11.5% (23/200).

Adds Cochran's Q (omnibus, H0: four arms equal), all C(4,2)=6 exact McNemar
edges with Holm-Bonferroni, and writes STATS_COCHRAN_Q2.txt + saves the correct
four-arm vectors to conflict_vectors_q2.npz.

Verdict rule: Q n.s. AND all six Holm-corrected pairwise p>0.05 => the global
"among the (common-planner) arms ... indistinguishable" claim is licensed.
"""
from __future__ import annotations
import os
import sys
import argparse
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from scipy import stats

# reuse the verified stats helpers (do NOT re-collect PG/Fixed/CV)
from stats_tests_cochran import cochran_q, holm

# ---- Stage-1b deploy rollout: replicate p2_mcnemar_dev.rollout() exactly ----
PLANGRAD = "/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim"
sys.path.insert(0, PLANGRAD)
import torch

DTYPE = torch.float64
DEV = "cpu"
DSEP = 30.0
DT = 0.2
T = 20
HP = 15
ETA_W = 0.3
BATCH = 8
SEED = 12345
N = 200


def mcnemar_exact(a, b):
    """Exact two-sided McNemar on paired binary vectors (conflict=1)."""
    import math
    a = a.astype(bool); b = b.astype(bool)
    b01 = int((~a & b).sum()); b10 = int((a & ~b).sum())
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) * 2
    return b01, b10, min(1.0, p)


@torch.no_grad()
def stage1b_conflict():
    """Per-episode conflict bool[N] for Stage-1b under the deploy planner.
    Mirrors p2_mcnemar_dev.rollout()'s min-sep computation."""
    from predictor import GMMTrajectoryPredictor
    from cbf_mpc import CBFMPCLayer
    from safe_policy import SafePolicy
    from params import DEFAULT_PARAMS
    from dynamics import EVTOLDynamics
    from wind import UrbanWindField
    from guam_encounters import GUAMEncounters
    from seeding import set_seed

    set_seed(SEED)
    net = GMMTrajectoryPredictor(T=30, K=5).double().to(DEV)
    net.load_state_dict(torch.load(f"{PLANGRAD}/stage1b_domainadapt.pt",
                                   map_location=DEV))
    net.eval()
    planner = CBFMPCLayer(n_neighbors=1, horizon=HP, dt=DT, d_sep=DSEP,
                          alpha=0.1, a_max=20.0)
    policy = SafePolicy(net, planner)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=DEV)
    wind = UrbanWindField(eta_w=ETA_W, dtype=DTYPE, device=DEV, seed=7)
    gen = GUAMEncounters(os.environ["GUAM_MAT"], range(2500, 3000), seed=SEED)
    weight = DEFAULT_PARAMS.weight
    mins = []
    for _ in range(max(1, N // BATCH)):
        x0, nh, nf, _ref, _ = gen.sample(BATCH, T, DEV)
        x = x0
        min_sep = torch.full((BATCH,), 1e6, dtype=DTYPE, device=DEV)
        for t in range(T):
            p0, v0 = x[:, 0:3], x[:, 3:6]
            tt = torch.arange(HP + 1, dtype=DTYPE, device=DEV) * DT
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            try:
                u, _ = policy(x, nh, nf[:, :, t, :], p_ref)
            except Exception:
                u = torch.zeros(BATCH, 4, dtype=DTYPE, device=DEV); u[:, 0] = weight
            x = dyn.step(x, u, wind.sample(p0), DT)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            min_sep = torch.minimum(min_sep, d)
        mins.append(min_sep.cpu())
    ms = torch.cat(mins)[:N]
    return (ms < DSEP).numpy().astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_in", default=os.path.join(HERE, "conflict_vectors.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "STATS_COCHRAN_Q2.txt"))
    ap.add_argument("--npz_out", default=os.path.join(HERE, "conflict_vectors_q2.npz"))
    args = ap.parse_args()

    prev = np.load(args.npz_in)
    print("recomputing Stage-1b deploy conflict vector ...", flush=True)
    s1b = stage1b_conflict()

    # CORRECT arm set: the four that share ONE deploy planner
    conf = {
        "PlanGrad":          prev["PlanGrad"].astype(bool),
        "Stage-1b":          s1b,
        "Fixed-Predictor":   prev["Fixed-Predictor"].astype(bool),
        "Constant-Velocity": prev["Constant-Velocity"].astype(bool),
    }
    L = min(len(v) for v in conf.values())
    conf = {k: v[:L] for k, v in conf.items()}
    np.savez(args.npz_out, **conf)

    fh = open(args.out, "w")

    def w(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    names = list(conf.keys())
    w("=" * 74)
    w("[TODO-Q2] OMNIBUS + ALL-PAIRS SAFETY TESTS -- CORRECT common-planner arm set")
    w("  {PlanGrad, Stage-1b, Fixed-Predictor, Constant-Velocity}  (Conformal")
    w("  EXCLUDED: it changes the planner threshold, not in the fixed-planner set)")
    w("  n=%d, seed=%d, deploy planner alpha=0.1/Hp=15/a_max=20, held-out 2500-3000"
      % (L, SEED))
    w("  conflict = per-episode min sep < %.0f m." % DSEP)
    w("=" * 74)
    w("%-20s %8s   %-14s" % ("method", "CR %", "(k/n)"))
    for name in names:
        c = conf[name]; k, n = int(c.sum()), len(c)
        w("%-20s %7.1f   (%d/%d)" % (name, 100.0 * k / n, k, n))

    Q, df, pQ = cochran_q([conf[nm] for nm in names])
    w("")
    w("Cochran's Q (omnibus, H0: all four common-planner arms equal):")
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
    holm_res = holm(raw)
    for lbl, p, padj, rej in holm_res:
        b01, b10 = detail[lbl]
        w("  %-40s disc(%d/%d)  p=%.3f  Holm=%.3f  -> %s"
          % (lbl, b01, b10, p, padj, "SIGNIFICANT" if rej else "n.s."))

    w("")
    licensed = (pQ > 0.05) and all(not r for _, _, _, r in holm_res)
    if licensed:
        w("VERDICT: Cochran's Q p=%.3f (n.s.) and all six Holm-corrected pairwise" % pQ)
        w("  edges n.s. => the four COMMON-PLANNER arms are JOINTLY indistinguishable")
        w("  on conflict rate; the global 'among the arms' claim is LICENSED.")
        w("  (The earlier Conformal-inclusive run separately licenses the secondary")
        w("   statement that ALL certificate-equipped arms, margin baseline included,")
        w("   are jointly indistinguishable.)")
    else:
        w("VERDICT: omnibus/pairwise rejects equality => keep narrowed")
        w("  'three prespecified contrasts' wording.")
    fh.close()


if __name__ == "__main__":
    main()
