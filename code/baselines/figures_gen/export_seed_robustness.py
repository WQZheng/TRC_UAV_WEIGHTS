#!/usr/bin/env python3
"""Export per-seed Stage-2 training-stochasticity results for Section 5.4.

WHY THIS SCRIPT EXISTS
  RUNBOOK.md:336 reports "CR = 11.4 +/- 0.4%" and "ADE = 6.3 +/- 1.4 m" over
  three Stage-2 training seeds, but the underlying per-seed numbers were never
  written to disk: train_seeds.log holds only the training curves (whose `ade`
  column is a normalised training quantity, not metres), and no evaluation
  output for stage2_seed{1,2,3}.pt exists anywhere in the repository. The three
  checkpoints do exist and are distinct (md5 f44b69a7 / 4215fda0 / 2aca6b73),
  so the evaluation is re-run here and the per-seed values retained.

  Whether the reported +/- was an SD or an SEM could only be inferred
  arithmetically before this run (only an SD admits a solution for
  6.3 +/- 1.4 given a worst seed of 7.93 m and n=3; the SEM would be about
  0.82). An inference is not a measurement, so both are computed and printed
  here and the ambiguity is resolved by the data rather than by reconstruction.

WHAT IS EVALUATED
  Each seed's checkpoint under the deployment planner, through the same
  authoritative evaluator as the headline table:
      eval_common.evaluate_policy, gamma=0.1 H_p=15 a_max=20 d_sep=30,
      n=200 held-out encounters 2500-2999, seed 12345, eta_w=0.3, gust_std=3.0.
  Note gamma is the manuscript's name for what the code calls alpha; the
  manuscript reserves alpha for the predictor mixing weight alpha_{i,m}.

  evaluate_policy guarantees CR_% == 100*mean(minsep_per_ep < d_sep), so the
  200-dim conflict vector is derived from the same per-episode array that
  produced the scalar, not recomputed by a parallel route. That identity is
  asserted per seed.

  stage2_final.pt is evaluated on the same path as a control: it must reproduce
  the published 11.0% (22/200). If it does not, the pipeline is not the one that
  produced the manuscript and no per-seed number from it can be trusted.

EXACT McNEMAR
  Each Stage-2 seed is compared against a FIXED Stage-1b comparator
  (stage1b_domainadapt.pt) on the same 200 episodes, paired per episode, using
  the exact two-sided binomial test on the discordant pairs -- not the
  chi-square approximation, since the discordant counts here are single digits.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/data/lab/TRC_UAV_WEIGHTS/code/baselines/common")
sys.path.insert(0, "/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim")
import eval_common as ec

W_DIR = "/data/lab/TRC_UAV_WEIGHTS/Round1/04_weights"
OUT_DIR = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
TXT = "/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim/SEED_ROBUSTNESS.txt"

SEEDS = [("seed1", "stage2_seed1.pt"),
         ("seed2", "stage2_seed2.pt"),
         ("seed3", "stage2_seed3.pt")]
CONTROL = ("stage2_final", "stage2_final.pt")
COMPARATOR = ("Stage-1b", "stage1b_domainadapt.pt")

# Published values the control must reproduce (P2_MCNEMAR_DEV.txt, deploy
# config). If these break, nothing else in this file means anything.
PUB_FINAL_CR = 11.0
PUB_FINAL_K = 22
PUB_S1B_CR = 11.5
PUB_S1B_K = 23
# RUNBOOK.md:336, the aggregate this script exists to decompose.
RUNBOOK_CR = (11.4, 0.4)
RUNBOOK_ADE = (6.3, 1.4)


def mcnemar_exact_two_sided(b, c):
    """Exact two-sided binomial test on discordant pairs. Identical in form to
    p2_mcnemar_dev.py so the numbers are comparable to the published table."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def evaluate(name, weights, dev):
    pred = ec.load_gmm_predictor(os.path.join(W_DIR, weights), dev)
    planner = ec.make_best_planner()
    m = ec.evaluate_policy(pred, planner, n=200, device=dev)
    ms = np.asarray(m["minsep_per_ep"], float)
    conf = ms < ec.D_SEP
    # The scalar and the vector must be the same measurement.
    assert ms.size == 200, f"{name}: {ms.size} episodes"
    assert abs(100.0 * conf.mean() - m["CR_%"]) < 1e-9, (
        f"{name}: conflict vector mean {100 * conf.mean()} != CR {m['CR_%']}")
    return m, ms, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="seed_robustness_v2.npz")
    args = ap.parse_args()

    dev = ec.device_str(True)
    lines = []

    def w(s=""):
        print(s, flush=True)
        lines.append(s)

    store = {}
    res = {}

    w("STAGE-2 TRAINING-STOCHASTICITY ROBUSTNESS (deploy config)")
    w("  gamma=0.1 Hp=15 a_max=20 d_sep=30, n=200 held-out 2500-2999,")
    w("  eval seed=12345, eta_w=0.3, gust_std=3.0. gamma is the manuscript's")
    w("  name for the code's alpha (CBFMPCLayer(alpha=...)).")
    w("  CR is per-episode: an episode counts as a conflict if min separation")
    w("  fell below 30 m at any tick. Effort = mean normalised control energy.")
    w("=" * 78)
    w()

    # ---- control first: if this does not reproduce, stop ---------------------
    for tag, wt in [CONTROL, COMPARATOR]:
        m, ms, conf = evaluate(tag, wt, dev)
        res[tag] = (m, ms, conf)
        store[f"minsep__{tag}"] = ms
        store[f"conflict__{tag}"] = conf
        w(f"  {tag:14s} CR={m['CR_%']:5.2f}%  ({int(conf.sum())}/200)  "
          f"ADE={m['ADE_m']:6.3f} m  MinSep={m['minSep_m']:6.2f} m  "
          f"Effort={m['Energy']:7.3f}")

    cr_f = res["stage2_final"][0]["CR_%"]
    k_f = int(res["stage2_final"][2].sum())
    cr_s = res["Stage-1b"][0]["CR_%"]
    k_s = int(res["Stage-1b"][2].sum())
    assert abs(cr_f - PUB_FINAL_CR) < 0.01 and k_f == PUB_FINAL_K, (
        f"stage2_final gives {cr_f}% ({k_f}/200), published {PUB_FINAL_CR}% "
        f"({PUB_FINAL_K}/200); this is not the pipeline that produced the "
        f"manuscript and no per-seed number from it can be trusted")
    assert abs(cr_s - PUB_S1B_CR) < 0.01 and k_s == PUB_S1B_K, (
        f"Stage-1b gives {cr_s}% ({k_s}/200), published {PUB_S1B_CR}% "
        f"({PUB_S1B_K}/200)")
    w()
    w("  control reproduces the published deploy-config values exactly: "
      f"Stage-2 {cr_f:.1f}% ({k_f}/200), Stage-1b {cr_s:.1f}% ({k_s}/200)")
    w()

    # ---- the three training seeds -------------------------------------------
    w("PER-SEED RESULTS")
    w(f"  {'seed':6s} {'CR %':>7s} {'conflicts':>10s} {'ADE m':>8s} "
      f"{'MinSep m':>9s} {'Effort':>9s}")
    for tag, wt in SEEDS:
        m, ms, conf = evaluate(tag, wt, dev)
        res[tag] = (m, ms, conf)
        store[f"minsep__{tag}"] = ms
        store[f"conflict__{tag}"] = conf
        w(f"  {tag:6s} {m['CR_%']:7.2f} {int(conf.sum()):10d} "
          f"{m['ADE_m']:8.3f} {m['minSep_m']:9.2f} {m['Energy']:9.3f}")

    crs = np.array([res[t][0]["CR_%"] for t, _ in SEEDS])
    ades = np.array([res[t][0]["ADE_m"] for t, _ in SEEDS])
    mss = np.array([res[t][0]["minSep_m"] for t, _ in SEEDS])
    efforts = np.array([res[t][0]["Energy"] for t, _ in SEEDS])

    def spread(v, label, unit=""):
        sd = v.std(ddof=1)
        sem = sd / math.sqrt(v.size)
        w(f"  {label:10s} mean {v.mean():7.3f}{unit}   "
          f"SD {sd:6.3f}   SEM {sem:6.3f}   "
          f"min {v.min():.3f}  max {v.max():.3f}")
        return sd, sem

    w()
    w("SPREAD ACROSS THE THREE TRAINING SEEDS (n=3, ddof=1)")
    sd_cr, sem_cr = spread(crs, "CR %")
    sd_ade, sem_ade = spread(ades, "ADE m")
    spread(mss, "MinSep m")
    spread(efforts, "Effort")

    w()
    w("WHICH CONVENTION DID RUNBOOK.md:336 USE?")
    for label, got_mean, sd, sem, (pm, pv) in [
            ("CR", crs.mean(), sd_cr, sem_cr, RUNBOOK_CR),
            ("ADE", ades.mean(), sd_ade, sem_ade, RUNBOOK_ADE)]:
        d_sd = abs(sd - pv)
        d_sem = abs(sem - pv)
        verdict = "SD" if d_sd < d_sem else "SEM"
        w(f"  {label:4s} reported {pm} +/- {pv}; measured mean "
          f"{got_mean:.3f}, SD {sd:.3f}, SEM {sem:.3f} "
          f"-> the reported spread is the {verdict} "
          f"(|diff| SD {d_sd:.3f} vs SEM {d_sem:.3f})")

    # ---- exact McNemar, each seed against the FIXED Stage-1b ----------------
    w()
    w("EXACT McNEMAR, each Stage-2 seed vs the fixed Stage-1b comparator")
    w("  paired per episode on the same 200 encounters; exact two-sided")
    w("  binomial on discordant pairs (the counts are single digits, so the")
    w("  chi-square approximation is not used)")
    w(f"  {'seed':6s} {'both':>5s} {'S1b only':>9s} {'seed only':>10s} "
      f"{'neither':>8s} {'b':>3s} {'c':>3s} {'net pp':>7s} {'p':>8s}")
    s1b_conf = res["Stage-1b"][2]
    mc = {}
    for tag, _ in SEEDS:
        cf = res[tag][2]
        both = int((s1b_conf & cf).sum())
        b = int((s1b_conf & ~cf).sum())     # S1b conflict, seed safe
        c = int((~s1b_conf & cf).sum())     # S1b safe, seed conflict
        neither = int((~s1b_conf & ~cf).sum())
        assert both + b + c + neither == 200
        p = mcnemar_exact_two_sided(b, c)
        net = 100.0 * (cf.mean() - s1b_conf.mean())
        mc[tag] = dict(both=both, b=b, c=c, neither=neither, p=p, net_pp=net)
        w(f"  {tag:6s} {both:5d} {b:9d} {c:10d} {neither:8d} "
          f"{b:3d} {c:3d} {net:+7.1f} {p:8.4f}")
        store[f"mcnemar__{tag}"] = np.array([both, b, c, neither, p], float)

    w()
    w("  Every comparison here is against the SAME Stage-1b vector, so the")
    w("  three tests are not independent of one another; they answer 'is this")
    w("  seed distinguishable from the matched control', not 'do the seeds")
    w("  differ among themselves'.")

    # pooled: does ANY seed separate from the comparator
    w()
    ps = [mc[t]["p"] for t, _ in SEEDS]
    w(f"  smallest p across the three seeds: {min(ps):.4f}"
      + ("  -> no seed is distinguishable from the matched control at 0.05"
         if min(ps) >= 0.05 else
         "  -> at least one seed separates from the matched control"))

    store["seed_names"] = np.array([t for t, _ in SEEDS])
    store["cr"] = crs
    store["ade"] = ades
    store["minsep"] = mss
    store["effort"] = efforts
    store["d_sep"] = np.array([ec.D_SEP])
    store["planner"] = np.array(["gamma=0.1 Hp=15 a_max=20 d_sep=30"])

    out = os.path.join(OUT_DIR, args.out)
    np.savez(out, **store)
    w()
    w(f"wrote {out}")
    w(f"  per-seed 200-dim conflict vectors: "
      + ", ".join(f"conflict__{t}" for t, _ in SEEDS)
      + f", conflict__stage2_final, conflict__Stage-1b")

    with open(TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {TXT}")

    with open(os.path.join(OUT_DIR, "seed_robustness_v2.json"), "w") as f:
        json.dump({
            "planner": "gamma=0.1 Hp=15 a_max=20 d_sep=30",
            "n": 200, "eval_seed": 12345,
            "per_seed": {t: {"CR_%": float(res[t][0]["CR_%"]),
                             "conflicts": int(res[t][2].sum()),
                             "ADE_m": float(res[t][0]["ADE_m"]),
                             "minSep_m": float(res[t][0]["minSep_m"]),
                             "Effort": float(res[t][0]["Energy"]),
                             "mcnemar_vs_Stage1b": mc[t]}
                         for t, _ in SEEDS},
            "control": {"stage2_final_CR_%": float(cr_f),
                        "stage2_final_conflicts": k_f,
                        "Stage-1b_CR_%": float(cr_s),
                        "Stage-1b_conflicts": k_s},
            "spread": {"CR_mean": float(crs.mean()),
                       "CR_SD": float(crs.std(ddof=1)),
                       "CR_SEM": float(crs.std(ddof=1) / math.sqrt(3)),
                       "ADE_mean": float(ades.mean()),
                       "ADE_SD": float(ades.std(ddof=1)),
                       "ADE_SEM": float(ades.std(ddof=1) / math.sqrt(3))},
        }, f, indent=2)
    print("wrote seed_robustness_v2.json")


if __name__ == "__main__":
    main()
