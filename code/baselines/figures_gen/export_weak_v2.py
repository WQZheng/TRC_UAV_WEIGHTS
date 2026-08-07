#!/usr/bin/env python3
"""Export the weak-certificate recovery data for Figure 12.

The figure contrasts ONE comparison -- Stage-1b vs Stage-2 -- across TWO
certificate configurations. That is the whole point: the task-aligned signal is
undetectable under the deployment certificate and large under the weak one, so
the quantity that must be held fixed is the contrast, and the quantity that
must vary is the certificate.

  strong / deployment   gamma=0.1  H_p=15  a_max=20   11.5% -> 11.0%   -0.5 pp
  weak / loosened       gamma=0.4  H_p=8   a_max=10   56.0% -> 28.5%  -27.5 pp

TWO DIFFERENT -0.5 pp NUMBERS EXIST AND MUST NOT BE CONFLATED.
  (i)  strong-certificate S1b -> S2  = -0.5 pp  (b=2, c=1, p=1.0)
       the signal does nothing when the certificate is strong.  <-- the figure
  (ii) weak-certificate   S1  -> S1b = -0.5 pp  (b=24, c=23, p=1.0)
       data adaptation alone does nothing when the certificate is weak.
Both are -0.5 pp, both have p=1.0, and they mean unrelated things. Quantity (i)
is the strong arm of this figure. Quantity (ii) is exported only so the caption
can state the contrast, and is named span_adaptation_only to keep it apart.

PROVENANCE
  weak side    Round1/05_results/robustness/p0_referee/loose_minsep.pt
               per-episode MinSep for all three arms on one episode-aligned
               stream. CBF-MPC alpha=0.4 a_max=10 Hp=8, T=20 dt=0.2
               eta_w=0.3 d_sep=30, held-out 2500-3000, seed 12345, n=200.
               Published table: P0_STAGE1B_LOOSE.txt.
  strong side  code/baselines/figures_gen/fig_data/conflict_vectors_v2.npz
               per-episode conflict booleans, deployment config
               alpha=0.1 Hp=15 a_max=20 d_sep=30, n=200 seed 12345.
               Published table: P2_MCNEMAR_DEV.txt.
               conflict_vectors_q2.npz under Round1/05_results/robustness/
               p0_referee is byte-identical to this file (md5 8d43503cc757),
               so it is a second copy of the same content, NOT an earlier
               pipeline. An earlier draft of this header called it superseded;
               that was wrong and is corrected here. The distinction matters
               because "superseded" would imply a reader must avoid q2, whereas
               either path yields the same vectors.

  alpha (code) == gamma (manuscript). Same quantity, different name; the
  manuscript reserves alpha for the predictor mixing weight alpha_{i,m}.

  The weak configuration is the point marked with a circled cross on the
  Figure 11 landscape. Its H_p=8 puts it off that H_p=15 surface, which is
  exactly why its results need their own figure rather than a cell value.
"""
import os
import numpy as np
import torch
from scipy.stats import binomtest

ROOT = "/data/lab/TRC_UAV_WEIGHTS"
OUT_DIR = os.environ.get(
    "FIG_DATA_DIR", os.path.join(ROOT, "code/baselines/figures_gen/fig_data"))
LOOSE_PT = os.path.join(
    ROOT, "Round1/05_results/robustness/p0_referee/loose_minsep.pt")
STRONG_NPZ = os.path.join(OUT_DIR, "conflict_vectors_v2.npz")

# Published values these exports must reproduce, from the archived text files.
PUB_WEAK_CR = dict(stage1=56.5, stage1b=56.0, stage2=28.5)
PUB_WEAK_2X2 = (57, 55, 0, 88)          # a, b, c, d with rows = Stage-1b
PUB_WEAK_P = 5.551e-17
PUB_STRONG_CR = dict(stage1b=11.5, stage2=11.0)
PUB_STRONG_2X2 = (21, 2, 1, 176)
PUB_STRONG_P = 1.0

WEAK_CFG = dict(gamma=0.4, Hp=8, a_max=10.0)
STRONG_CFG = dict(gamma=0.1, Hp=15, a_max=20.0)


def wilson(k, n, z=1.96):
    """Wilson score interval in percent. Never returns a bound outside [0,100]."""
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100.0 * (c - h), 100.0 * (c + h)


def paired(x, y):
    """2x2 with rows = x. Returns a (both), b (x only), c (y only), d (neither)."""
    return (int((x & y).sum()), int((x & ~y).sum()),
            int((~x & y).sum()), int((~x & ~y).sum()))


def mcnemar_exact(b, c):
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)


def main():
    # ---- weak side: recompute everything from per-episode MinSep ------------
    raw = torch.load(LOOSE_PT, weights_only=False)
    d_sep = float(raw["d_sep"])
    n_weak = int(raw["n"])
    ms1 = np.asarray(raw["s1"], dtype=float)
    ms1b = np.asarray(raw["s1b"], dtype=float)
    ms2 = np.asarray(raw["s2"], dtype=float)
    assert ms1.shape == ms1b.shape == ms2.shape == (n_weak,)
    assert d_sep == 30.0, f"separation standard moved: {d_sep}"

    # A conflict IS a separation below the standard. Derive, never re-declare.
    w1, w1b, w2 = ms1 < d_sep, ms1b < d_sep, ms2 < d_sep

    cr_w = {k: 100.0 * v.mean() for k, v in
            (("stage1", w1), ("stage1b", w1b), ("stage2", w2))}
    for k, want in PUB_WEAK_CR.items():
        assert abs(cr_w[k] - want) < 1e-9, f"weak {k}: {cr_w[k]} != {want}"

    a_w, b_w, c_w, d_w = paired(w1b, w2)
    assert (a_w, b_w, c_w, d_w) == PUB_WEAK_2X2, (a_w, b_w, c_w, d_w)
    p_w = mcnemar_exact(b_w, c_w)
    assert abs(p_w - PUB_WEAK_P) < 1e-19, p_w

    # The crown claim: Stage-2's conflict set is a PROPER SUBSET of Stage-1b's.
    # This is what licenses "resolves 55, introduces none". If it ever fails the
    # figure must not be drawn, because c=0 would no longer be structural.
    assert int((w2 & ~w1b).sum()) == 0, "Stage-2 conflicts are NOT nested in Stage-1b"
    assert c_w == 0
    assert b_w > 0

    # Adaptation alone, weak config. This is the OTHER -0.5 pp. Exported under a
    # name that cannot be mistaken for the strong-certificate contrast.
    a_ad, b_ad, c_ad, d_ad = paired(w1, w1b)
    p_ad = mcnemar_exact(b_ad, c_ad)
    assert (b_ad, c_ad) == (24, 23), (b_ad, c_ad)
    # Adaptation is emphatically NOT nested -- it reshuffles both ways. That
    # asymmetry is the mechanism behind "noise, not structure".
    assert b_ad > 0 and c_ad > 0, "adaptation unexpectedly nested"

    # ---- strong side: per-episode conflict booleans -------------------------
    sv = np.load(STRONG_NPZ, allow_pickle=True)
    s1b, s2 = sv["Stage-1b"], sv["PlanGrad"]
    n_strong = int(s2.size)
    assert s1b.dtype == bool and s2.dtype == bool
    assert n_strong == n_weak == 200, (n_strong, n_weak)

    cr_s = dict(stage1b=100.0 * s1b.mean(), stage2=100.0 * s2.mean())
    for k, want in PUB_STRONG_CR.items():
        assert abs(cr_s[k] - want) < 1e-9, f"strong {k}: {cr_s[k]} != {want}"

    a_s, b_s, c_s, d_s = paired(s1b, s2)
    assert (a_s, b_s, c_s, d_s) == PUB_STRONG_2X2, (a_s, b_s, c_s, d_s)
    p_s = mcnemar_exact(b_s, c_s)
    assert abs(p_s - PUB_STRONG_P) < 1e-12, p_s
    # Strong side is NOT nested either -- one episode goes the other way. So
    # "introduces none" is a weak-certificate fact, not a Stage-2 property.
    assert c_s > 0, "strong side unexpectedly nested; caption wording would break"

    # ---- the two effects the figure compares -------------------------------
    delta_strong = cr_s["stage2"] - cr_s["stage1b"]          # -0.5
    delta_weak = cr_w["stage2"] - cr_w["stage1b"]            # -27.5
    span_adaptation_only = cr_w["stage1b"] - cr_w["stage1"]  # -0.5, unrelated
    assert abs(delta_strong - (-0.5)) < 1e-9
    assert abs(delta_weak - (-27.5)) < 1e-9
    assert abs(span_adaptation_only - (-0.5)) < 1e-9
    # The coincidence is real and is the reason for the naming discipline.
    assert abs(delta_strong - span_adaptation_only) < 1e-9, \
        "the two -0.5 pp values are expected to coincide numerically"
    # The effect ratio the figure exists to show.
    assert abs(delta_weak) > 50 * abs(delta_strong)

    # Baseline headroom: the strong certificate leaves almost no room for the
    # signal to act in. This is shown by the connector geometry, not asserted
    # as a mechanism.
    baseline_gap = cr_w["stage1b"] - cr_s["stage1b"]         # 44.5 pp
    assert abs(baseline_gap - 44.5) < 1e-9

    # ---- MinSep, weak config only ------------------------------------------
    # Strong-side MinSep is deliberately absent: that null is carried by the
    # matched-pair figure and importing it here would add a third data source.
    ms_mean = np.array([ms1.mean(), ms1b.mean(), ms2.mean()])
    ms_sd = np.array([ms1.std(ddof=1), ms1b.std(ddof=1), ms2.std(ddof=1)])
    ms_se = ms_sd / np.sqrt(n_weak)
    ms_lo, ms_hi = ms_mean - 1.96 * ms_se, ms_mean + 1.96 * ms_se
    # Paired shift, Stage-1b -> Stage-2.
    dif = ms2 - ms1b
    dm = dif.mean()
    dse = dif.std(ddof=1) / np.sqrt(n_weak)
    assert abs(dm - 6.137) < 5e-4, dm
    # The matched control sits BELOW the separation standard on average while
    # Stage-2 sits above it. Both facts are visible in the panel.
    assert ms_mean[1] < d_sep < ms_mean[2], (ms_mean[1], ms_mean[2])

    wl_w = np.array([wilson(int(v.sum()), n_weak) for v in (w1, w1b, w2)])
    wl_s = np.array([wilson(int(v.sum()), n_strong) for v in (s1b, s2)])

    out = os.path.join(OUT_DIR, "weak_recovery_v2.npz")
    np.savez(
        out,
        # panel (a): two paired connectors on one CR axis
        cr_strong=np.array([cr_s["stage1b"], cr_s["stage2"]]),
        cr_weak=np.array([cr_w["stage1b"], cr_w["stage2"]]),
        wilson_strong=wl_s,
        wilson_weak=wl_w[1:],
        cr_weak_stage1=cr_w["stage1"],
        wilson_weak_stage1=wl_w[0],
        table_strong=np.array([a_s, b_s, c_s, d_s]),
        table_weak=np.array([a_w, b_w, c_w, d_w]),
        p_strong=p_s,
        p_weak=p_w,
        delta_strong=delta_strong,
        delta_weak=delta_weak,
        baseline_gap=baseline_gap,
        # adaptation-only contrast, weak config: caption material only
        table_adaptation=np.array([a_ad, b_ad, c_ad, d_ad]),
        p_adaptation=p_ad,
        span_adaptation_only=span_adaptation_only,
        # panel (c): MinSep, weak config
        minsep_mean=ms_mean,
        minsep_sd=ms_sd,
        minsep_lo=ms_lo,
        minsep_hi=ms_hi,
        minsep_paired_delta=dm,
        minsep_paired_lo=dm - 1.96 * dse,
        minsep_paired_hi=dm + 1.96 * dse,
        d_sep=d_sep,
        # config and bookkeeping
        arms=np.array(["Stage-1", "Stage-1b", "Stage-2"]),
        n=n_weak,
        gamma_weak=WEAK_CFG["gamma"], Hp_weak=WEAK_CFG["Hp"],
        amax_weak=WEAK_CFG["a_max"],
        gamma_strong=STRONG_CFG["gamma"], Hp_strong=STRONG_CFG["Hp"],
        amax_strong=STRONG_CFG["a_max"],
    )

    print(f"wrote {out}")
    print(f"  strong  gamma={STRONG_CFG['gamma']} Hp={STRONG_CFG['Hp']} "
          f"a_max={STRONG_CFG['a_max']}  "
          f"{cr_s['stage1b']:.1f}% -> {cr_s['stage2']:.1f}%  "
          f"({delta_strong:+.1f} pp)  b={b_s} c={c_s} p={p_s:.3g}")
    print(f"  weak    gamma={WEAK_CFG['gamma']} Hp={WEAK_CFG['Hp']} "
          f"a_max={WEAK_CFG['a_max']}  "
          f"{cr_w['stage1b']:.1f}% -> {cr_w['stage2']:.1f}%  "
          f"({delta_weak:+.1f} pp)  b={b_w} c={c_w} p={p_w:.3g}")
    print(f"  nesting verified: Stage-2 conflicts are a proper subset of "
          f"Stage-1b's (c=0, b={b_w})")
    print(f"  baseline headroom at Stage-1b: {baseline_gap:.1f} pp "
          f"({cr_w['stage1b']:.1f}% weak vs {cr_s['stage1b']:.1f}% strong)")
    print(f"  adaptation alone (weak, caption only): {span_adaptation_only:+.1f} pp "
          f"b={b_ad} c={c_ad} p={p_ad:.3g}, {b_ad + c_ad} episodes flipped")
    print(f"  MinSep weak: " + ", ".join(
        f"{a}={m:.2f}" for a, m in zip(["S1", "S1b", "S2"], ms_mean)) +
        f"; paired S2-S1b {dm:+.3f} m [{dm - 1.96 * dse:+.3f},"
        f"{dm + 1.96 * dse:+.3f}], standard {d_sep:.0f} m")


if __name__ == "__main__":
    main()
