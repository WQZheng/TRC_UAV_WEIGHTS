#!/usr/bin/env python3
"""Export matched_pair_v2.npz for the redesigned Fig 6 (Stage-1b vs Stage-2
matched-operational-differences forest).

PROVENANCE / ZERO-FABRICATION CONTRACT
  * The per-episode MAX lateral offset is produced by the *byte-identical*
    authoritative rollout `p2_mcnemar_dev.rollout` (the same code that wrote
    DEV_MATCHED_S1B.txt). We do NOT re-implement the cross-track geometry; we
    import and call the canonical function so the blood-line is clean.
  * The five matched-regime paired conflict vectors are copied verbatim from the
    authoritative stage1b_mismatch_v2.npz (already used for the main table).
  * nominal min-sep / effort per-episode arrays come from the authoritative
    baseline per_episode.npz sidecars (which reproduce the main table exactly).

SELF-CHECKS (any failure -> hard exit, nothing is written)
  1. five-regime discordant (b,c) equal the archived table, nominal == (2,1);
  2. effort paired CI reproduces EFF_MATCHED_S1B.txt : -0.55 [-1.68, +0.58];
  3. max-offset paired CI reproduces DEV_MATCHED_S1B.txt: +0.52 [-1.11, +2.14]
     (this quantity is from the old-solver-era txt and is NOT trusted until it
      recomputes exactly; mismatch stops the pipeline and reports).

Run from code/plangrad_sim (deps + p2_mcnemar_dev live there):
  export GUAM_MAT=.../GUAM/Challenge_Problems/Data_Set_1.mat
  python3 export_matched_pair_v2.py
"""
from __future__ import annotations
import os, sys
import numpy as np

ROOT = "/data/lab/TRC_UAV_WEIGHTS"
WEIGHTS = os.path.join(ROOT, "Round1/04_weights")
MISMATCH = os.path.join(ROOT, "code/baselines/figures_gen/fig_data/"
                        "stage1b_mismatch_v2.npz")
PEREP_S1B = os.path.join(ROOT, "code/baselines/07_stage1b/per_episode.npz")
PEREP_S2 = os.path.join(ROOT, "code/baselines/00_plangrad_reference/"
                        "per_episode.npz")
OUT = os.path.join(ROOT, "code/baselines/figures_gen/fig_data/"
                   "matched_pair_v2.npz")

# authoritative reproduction targets (from the referee txts)
TGT_EFFORT = (-0.5496, -1.6824, +0.5832)     # EFF_MATCHED_S1B.txt
TGT_OFFSET = (+0.52, -1.11, +2.14)            # DEV_MATCHED_S1B.txt (2 d.p.)
TGT_DISCORDANT = {                            # stage1b_mismatch_v2.npz table
    "nominal": (2, 1), "mass +20%": (1, 2), "thrust eff 0.85": (2, 3),
    "actuator delay 2": (5, 1), "combined": (2, 0),
}


def fail(msg):
    print("\n*** SELF-CHECK FAILED -- nothing written ***\n" + msg)
    sys.exit(1)


def paired_diff_ci_np(a, b, z=1.96):
    """Normal-approx paired-difference CI, identical to
    p2_mcnemar_dev.paired_diff_ci (z=1.96, unbiased SD). Used for the
    max-offset check (DEV txt reused this z=1.96 function)."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    n = diff.size
    m = float(diff.mean()); sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(n)
    return m, (m - z * se, m + z * se)


def paired_diff_ci_t(a, b):
    """Paired-difference CI using Student-t (df=n-1). EFF_MATCHED_S1B.txt was
    written with t_crit(0.975, df=199)=1.972, NOT z=1.96, so effort must be
    checked against the t interval to reproduce that txt exactly."""
    from math import sqrt
    try:
        from scipy.stats import t as _t
        diff = np.asarray(a, float) - np.asarray(b, float)
        n = diff.size
        tc = float(_t.ppf(0.975, n - 1))
    except Exception:
        tc = 1.972
        diff = np.asarray(a, float) - np.asarray(b, float)
        n = diff.size
    m = float(diff.mean()); sd = float(diff.std(ddof=1))
    se = sd / sqrt(n)
    return m, (m - tc * se, m + tc * se), tc


def main():
    import torch
    from p2_mcnemar_dev import rollout, paired_diff_ci  # authoritative

    # ---------- (A) max lateral offset via authoritative rollout ----------
    # cache the (slow, ~8 min) rollout so a self-check tweak does not re-run it.
    CACHE = os.path.join(os.path.dirname(OUT), "_maxoffset_cache.npz")
    if os.path.exists(CACHE):
        c = np.load(CACHE)
        xm1b, xm2 = c["xm1b"], c["xm2"]
        print("[1/4] loaded cached authoritative max-offset arrays")
    else:
        print("[1/4] running authoritative p2_mcnemar_dev.rollout "
              "(Stage-1b, Stage-2)")
        sep1b, xm1b, xa1b = rollout(os.path.join(WEIGHTS,
                                                 "stage1b_domainadapt.pt"))
        sep2, xm2, xa2 = rollout(os.path.join(WEIGHTS, "stage2_final.pt"))
        xm1b = xm1b.double().cpu().numpy(); xm2 = xm2.double().cpu().numpy()
        np.savez(CACHE, xm1b=xm1b, xm2=xm2)
        print("    cached ->", CACHE)

    # reproduce the offset CI with the canonical function AND our np copy
    md_c, (lo_c, hi_c) = paired_diff_ci(torch.tensor(xm2), torch.tensor(xm1b))
    md_n, (lo_n, hi_n) = paired_diff_ci_np(xm2, xm1b)
    print(f"    canonical  paired max-offset (S2-S1b) = {md_c:+.4f} "
          f"[{lo_c:+.4f}, {hi_c:+.4f}]")
    print(f"    numpy copy paired max-offset (S2-S1b) = {md_n:+.4f} "
          f"[{lo_n:+.4f}, {hi_n:+.4f}]")
    print(f"    S1b MAX offset = {xm1b.mean():.2f} +/- {xm1b.std(ddof=1):.2f}"
          f" | S2 MAX offset = {xm2.mean():.2f} +/- {xm2.std(ddof=1):.2f}")

    # ---------- (B) nominal min-sep / effort from authoritative sidecars ---
    d1b = np.load(PEREP_S1B); d2 = np.load(PEREP_S2)
    ms1b = d1b["minsep_per_ep"]; ef1b = d1b["effort_per_ep"]
    ms2 = d2["minsep_per_ep"]; ef2 = d2["effort_per_ep"]

    # ---------- (C) five-regime paired conflict vectors (verbatim copy) ----
    dm = np.load(MISMATCH, allow_pickle=True)
    regimes = [str(r) for r in dm["regimes"]]

    # =================== SELF-CHECKS ===================
    print("[2/4] self-check 1: five-regime discordant (b,c)")
    for r in regimes:
        b, c = [int(v) for v in dm[f"discordant__{r}"]]
        # recompute from the conflict vectors too, to be safe
        c1b = dm[f"conflict__{r}__stage1b"].astype(bool)
        c2 = dm[f"conflict__{r}__stage2"].astype(bool)
        bb = int((c1b & ~c2).sum()); cc = int((~c1b & c2).sum())
        exp = TGT_DISCORDANT.get(r)
        ok = (b, c) == exp and (bb, cc) == exp
        print(f"    {r:18s} stored=({b},{c}) recomputed=({bb},{cc}) "
              f"expected={exp}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            fail(f"regime {r}: discordant mismatch")

    print("[3/4] self-check 2: effort paired CI vs EFF txt (Student-t, df=199)")
    md_e, (lo_e, hi_e), tc_e = paired_diff_ci_t(ef2, ef1b)
    print(f"    effort paired (S2-S1b) = {md_e:+.4f} [{lo_e:+.4f}, {hi_e:+.4f}]"
          f"  (t_crit={tc_e:.3f})  target {TGT_EFFORT}")
    if not (abs(md_e - TGT_EFFORT[0]) < 5e-3 and abs(lo_e - TGT_EFFORT[1]) < 5e-3
            and abs(hi_e - TGT_EFFORT[2]) < 5e-3):
        fail("effort paired CI does not reproduce EFF_MATCHED_S1B.txt")
    # also confirm the per-arm effort means/SD match the txt
    for nm, arr, tgt in [("S1b", ef1b, (52.8983, 11.8061)),
                         ("S2", ef2, (52.3487, 7.5702))]:
        if not (abs(arr.mean() - tgt[0]) < 5e-3
                and abs(arr.std(ddof=1) - tgt[1]) < 5e-3):
            fail(f"{nm} effort mean/SD does not match EFF txt")

    print("[4/4] self-check 3: max-offset paired CI vs DEV txt (STRICT)")
    if not (abs(round(md_n, 2) - TGT_OFFSET[0]) < 1e-9
            and abs(round(lo_n, 2) - TGT_OFFSET[1]) < 1e-9
            and abs(round(hi_n, 2) - TGT_OFFSET[2]) < 1e-9):
        fail(f"max-offset CI recomputed {md_n:+.2f} [{lo_n:+.2f}, {hi_n:+.2f}] "
             f"!= DEV txt {TGT_OFFSET}. This quantity comes from an old-solver "
             f"txt; refusing to trust it. STOP and report.")
    print("    max-offset CI reproduces DEV txt exactly -> blood-line clean.")

    # =================== WRITE ===================
    payload = {
        "regimes": np.array(regimes),
        # nominal continuous per-episode arrays (authoritative sidecars)
        "nominal_minsep_stage1b": ms1b, "nominal_minsep_stage2": ms2,
        "nominal_effort_stage1b": ef1b, "nominal_effort_stage2": ef2,
        # max lateral offset per-episode (authoritative rollout)
        "nominal_maxoffset_stage1b": xm1b, "nominal_maxoffset_stage2": xm2,
        # precomputed contrasts (for the figure caption / labels)
        "ci_effort": np.array([md_e, lo_e, hi_e]),
        "ci_maxoffset": np.array([md_n, lo_n, hi_n]),
    }
    for r in regimes:
        payload[f"conflict__{r}__stage1b"] = dm[f"conflict__{r}__stage1b"]
        payload[f"conflict__{r}__stage2"] = dm[f"conflict__{r}__stage2"]
        payload[f"discordant__{r}"] = dm[f"discordant__{r}"]
    np.savez(OUT, **payload)
    print("\nAll self-checks passed. wrote", OUT)
    print("keys:", sorted(payload.keys()))


if __name__ == "__main__":
    main()
