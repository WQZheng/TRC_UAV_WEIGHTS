#!/usr/bin/env python3
"""Export leadtime_v2.npz: the detection-horizon sweep for FOUR named arms.

Why this rerun exists
---------------------
tab:leadtime was assembled from three text files whose column HEADERS lie.
eval_leadtime.py only has --stage1 and --stage2 positions and always prints
"S1 CR%" / "S2 CR%", so LEADTIME_S1B*.txt carry Stage-1b results under a header
that says "S2" (README_p0_referee.md line 122 records
"--stage2 stage1b_domainadapt.pt"). Worse, the two files disagree about the arm
they nominally share: LEADTIME_S1B.txt's "S1" column reads 0.0/0.0/0.5/0.5 over
3-6 s while LEADTIME_FINE.txt's "S1" column reads 7.0/10.0/12.0 over 4-6 s for
what should be the same stage1_full.pt at the same seed, n and planner
configuration -- an order of magnitude apart.

That ambiguity is not cosmetic. If LEADTIME_S1B.txt actually ran Stage-1b in
BOTH positions, then "the matched control shows no hump" rests on one column
instead of two, and that claim is what separates the Stage-1 hump from a purely
geometric explanation. So all four arms are rerun here into one file with
explicit key names, and the contradictory interval is asserted directly.

Arms are keyed by name, never by position: cr__Stage1, cr__Stage-1b, cr__Stage2,
cr__Oracle. Per-episode conflict vectors are kept so the attribution can be
recomputed without rerunning.

Products
--------
leadtime_v2.npz
  horizons_s            lead times in seconds
  cr__<arm>             conflict rate (%) per horizon, arm in the four above
  conflict__<arm>       per-episode boolean conflict vector, [n_horizons, n]
  act_limited           Stage-2 conflicts the oracle also has, per horizon
  pred_limited          Stage-2 conflicts the oracle avoids, per horizon
  n_s2_conflicts        Stage-2 conflict count per horizon
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch

ROOT = "/data/lab/TRC_UAV_WEIGHTS"
sys.path.insert(0, f"{ROOT}/code/plangrad_sim")
sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")

# Authoritative rollout and encounter generator; NOT reimplemented here.
from eval_leadtime import (LeadTimeEncounters, rollout_condition, load_pred,
                          DT, HP, GUAM_MAT, set_seed)          # noqa: E402

OUT = f"{ROOT}/code/baselines/figures_gen/fig_data/leadtime_v2.npz"
RAW = f"{ROOT}/code/baselines/figures_gen/fig_data/_leadtime_raw_cache.npz"
# The canonical weights directory, the same one eval_common.PLANGRAD_DIR names.
# Verified identical by md5 to TRC_UAV_WEIGHTS/plangrad_sim and
# Round1/04_weights, so the choice of copy cannot change a result.
WDIR = os.environ.get("PLANGRAD_WEIGHTS", "/data/lab/plangrad/plangrad_sim")

N = int(os.environ.get("LT_N", "200"))
SEED = 12345
HORIZONS = [float(x) for x in os.environ.get(
    "LT_HORIZONS", "1,2,3,4,5,6,7,10,20").split(",")]
# A smoke run (LT_N small, or a subset of horizons) may not reproduce the
# published percentages, so the table assertions only bind the full protocol.
FULL = (N == 200 and len(HORIZONS) == 9)

ARMS = [("Stage1", "stage1_full.pt"),
        ("Stage-1b", "stage1b_domainadapt.pt"),
        ("Stage2", "stage2_final.pt")]

# tab:leadtime as published (lines 937-945). None = dash, not yet evaluated.
TAB = {
    1.0:  {"Stage1": 86.0, "Stage-1b": 83.0, "Stage2": 82.5, "Oracle": 82.0},
    2.0:  {"Stage1": 13.0, "Stage-1b": 11.5, "Stage2": 11.0, "Oracle": 11.0},
    3.0:  {"Stage1":  1.5, "Stage-1b":  0.0, "Stage2":  0.0, "Oracle":  0.0},
    4.0:  {"Stage1":  7.0, "Stage-1b":  0.0, "Stage2":  0.0, "Oracle":  0.0},
    # 5 s and 6 s Stage-1b were published as 0.0 from LEADTIME_S1B.txt. That
    # file ran Stage-1b in BOTH predictor positions -- its "S1" column reads
    # 0.0/0.0/0.5/0.5 over 3-6 s while the authoritative Stage-1 is
    # 7.0/10.0/12.0 -- so it held two independent Stage-1b streams and the
    # manuscript took the 0.0 one. The named rerun gives 0.5 at both horizons
    # (one episode in 200). The rerun is authoritative because all four arms
    # come from one file keyed by name, so the table is corrected to 0.5 and
    # the previously published 0.0 is recorded as coming from a retired source.
    5.0:  {"Stage1": 10.0, "Stage-1b":  0.5, "Stage2":  0.0, "Oracle":  0.0},
    6.0:  {"Stage1": 12.0, "Stage-1b":  0.5, "Stage2":  0.0, "Oracle":  0.0},
    7.0:  {"Stage1":  8.5, "Stage-1b": None, "Stage2":  0.0, "Oracle":  0.0},
    10.0: {"Stage1":  0.5, "Stage-1b": None, "Stage2":  0.0, "Oracle":  0.0},
    20.0: {"Stage1":  0.0, "Stage-1b": None, "Stage2":  0.0, "Oracle":  0.0},
}
# What the manuscript currently prints, so the correction is explicit and a
# later reader can see exactly which cells moved and why.
TAB_AS_PUBLISHED = {5.0: {"Stage-1b": 0.0}, 6.0: {"Stage-1b": 0.0}}
# The disputed interval, asserted on its own (requirement 1).
FINE_S1 = {4.0: 7.0, 5.0: 10.0, 6.0: 12.0}
S1B_FILE_S1COL = {3.0: 0.0, 4.0: 0.0, 5.0: 0.5, 6.0: 0.5}
# attribution rows that exist in LEADTIME.txt
TAB_ATTR = {1.0: (165, 98.0, 2.0), 2.0: (22, 95.0, 5.0)}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  n={N}  seed={SEED}  horizons={HORIZONS}", flush=True)
    nets = {}
    for tag, w in ARMS:
        p = f"{WDIR}/{w}"
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing weights {p}")
        nets[tag] = load_pred(p, dev)
        print(f"  loaded {tag:9s} <- {w}", flush=True)

    names = [a[0] for a in ARMS] + ["Oracle"]
    cr = {k: [] for k in names}
    conf = {k: [] for k in names}
    act_l, pred_l, n_s2 = [], [], []

    for h in HORIZONS:
        t_cpa = int(round(h / DT))
        T = t_cpa + HP + 2

        def mk():
            return LeadTimeEncounters(GUAM_MAT, range(2500, 3000), seed=SEED,
                                      t_cpa=t_cpa)

        row = {}
        for tag, _w in ARMS:
            set_seed(SEED)
            c = rollout_condition(mk(), "stage", dev, N, T, pred=nets[tag])
            row[tag] = c
        set_seed(SEED)
        row["Oracle"] = rollout_condition(mk(), "oracle", dev, N, T)

        for k in names:
            cr[k].append(100.0 * row[k].mean())
            conf[k].append(row[k].astype(bool))

        c2, co = row["Stage2"], row["Oracle"]
        a = int((c2 & co).sum())
        p = int((c2 & ~co).sum())
        act_l.append(a)
        pred_l.append(p)
        n_s2.append(int(c2.sum()))
        print(f"  {h:5.1f}s  " + "  ".join(
            f"{k}={100.0*row[k].mean():5.1f}" for k in names)
            + f"   | of {int(c2.sum())} S2-conflicts: act={a} pred={p}",
            flush=True)

    out = {"horizons_s": np.array(HORIZONS),
           "act_limited": np.array(act_l),
           "pred_limited": np.array(pred_l),
           "n_s2_conflicts": np.array(n_s2)}
    for k in names:
        out[f"cr__{k}"] = np.array(cr[k])
        out[f"conflict__{k}"] = np.array(conf[k])

    # The rollouts cost about 1.6 h, so the RAW result is cached unconditionally
    # BEFORE the assertions run. The assertions still gate the published npz --
    # a failure must stop the figure -- but a second disagreement no longer
    # costs another full sweep to re-examine. This cache is not a figure input.
    np.savez_compressed(RAW, **out)
    print(f"\nraw sweep cached (pre-assertion) -> {RAW}", flush=True)

    # ---------------------------- self-checks ------------------------------
    errs = []
    warns = []

    def chk(ok, msg, hard=True):
        print(("  OK   " if ok else "  FAIL ") + msg)
        if not ok:
            (errs if hard else warns).append(msg)

    if not FULL:
        print(f"\nSMOKE RUN (n={N}, horizons={HORIZONS}): table assertions are "
              f"skipped and nothing is written. Rerun with the full protocol.")
        for i, h in enumerate(HORIZONS):
            print(f"  {h:5.1f}s  " + "  ".join(
                f"{k}={out[f'cr__{k}'][i]:5.1f}" for k in names)
                + f"   (table: " + " ".join(
                    f"{k}={TAB[h][k]}" for k in names) + ")")
        return

    print("\nself-check 1: every published cell of tab:leadtime")
    for i, h in enumerate(HORIZONS):
        for k in names:
            exp = TAB[h][k]
            got = out[f"cr__{k}"][i]
            if exp is None:
                print(f"  ---- {h:4.1f}s {k:9s} was a dash; now {got:5.1f}")
                continue
            note = ""
            if h in TAB_AS_PUBLISHED and k in TAB_AS_PUBLISHED[h]:
                note = (f"  [corrected: manuscript prints "
                        f"{TAB_AS_PUBLISHED[h][k]:.1f}]")
            chk(abs(got - exp) < 0.051,
                f"{h:4.1f}s {k:9s} {got:5.1f} vs table {exp:5.1f}{note}")

    print("\nself-check 2: the disputed 4-6 s Stage-1 interval")
    for h, v in FINE_S1.items():
        got = out["cr__Stage1"][HORIZONS.index(h)]
        chk(abs(got - v) < 0.051,
            f"{h:4.1f}s Stage1 {got:5.1f} vs LEADTIME_FINE {v:5.1f}")
    print("  for the record, LEADTIME_S1B.txt's 'S1' column over 3-6 s was "
          + ", ".join(f"{k:.0f}s={v}" for k, v in S1B_FILE_S1COL.items()))
    matches_fine = all(
        abs(out["cr__Stage1"][HORIZONS.index(h)] - v) < 0.051
        for h, v in FINE_S1.items())
    matches_s1b = all(
        abs(out["cr__Stage1"][HORIZONS.index(h)] - v) < 0.051
        for h, v in S1B_FILE_S1COL.items() if h in FINE_S1 or h == 3.0)
    print(f"  VERDICT: reproduces LEADTIME_FINE = {matches_fine}; "
          f"reproduces LEADTIME_S1B 'S1' column = {matches_s1b}")
    if matches_s1b and not matches_fine:
        raise SystemExit(
            "STOP AND REPORT: the rerun reproduces LEADTIME_S1B.txt's S1 column "
            "rather than LEADTIME_FINE.txt. That would mean the hump's own data "
            "basis is in question, which reaches past this figure. Nothing "
            "written; escalate before proceeding.")

    print("\nself-check 3: the hump is non-monotone and Stage-1b is flat")
    s1 = out["cr__Stage1"]
    i3, i6, i10 = (HORIZONS.index(x) for x in (3.0, 6.0, 10.0))
    chk(s1[i6] > s1[i3], f"Stage1 rises 3->6 s ({s1[i3]:.1f} -> {s1[i6]:.1f})")
    chk(s1[i10] < s1[i6], f"Stage1 falls 6->10 s ({s1[i6]:.1f} -> "
                          f"{s1[i10]:.1f})")
    for k in ("Stage2", "Oracle"):
        tail = out[f"cr__{k}"][i3:]
        chk(float(np.max(tail)) < 0.051,
            f"{k} is zero from 3 s on (max {np.max(tail):.2f})")
    s1b_tail = out["cr__Stage-1b"][i3:]
    chk(float(np.max(s1b_tail)) < 1.01,
        f"Stage-1b shows no hump from 3 s on (max {np.max(s1b_tail):.2f})")

    print("\nself-check 4: attribution rows that LEADTIME.txt published")
    for h, (nc, fa, fp) in TAB_ATTR.items():
        i = HORIZONS.index(h)
        chk(abs(out["n_s2_conflicts"][i] - nc) <= 1,
            f"{h:.0f}s Stage-2 conflicts {out['n_s2_conflicts'][i]} vs {nc}")
        tot = max(1, out["n_s2_conflicts"][i])
        ga = 100.0 * out["act_limited"][i] / tot
        chk(abs(ga - fa) < 1.6, f"{h:.0f}s actuation-limited {ga:.0f}% vs {fa}%")
    print("  attribution is undefined where there are no Stage-2 conflicts:")
    for i, h in enumerate(HORIZONS):
        if out["n_s2_conflicts"][i] == 0:
            print(f"    {h:5.1f}s : 0 conflicts")

    if errs:
        raise AssertionError(f"{len(errs)} self-check(s) failed; nothing "
                             f"written:\n  " + "\n  ".join(errs))
    np.savez_compressed(OUT, **out)
    print("\nall self-checks passed. wrote", OUT)
    if warns:
        print(f"({len(warns)} soft warning(s))")


if __name__ == "__main__":
    main()
