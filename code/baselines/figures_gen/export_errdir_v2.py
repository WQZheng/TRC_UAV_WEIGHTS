#!/usr/bin/env python3
"""Export errdir_v2.npz: the CPA error profile and the per-episode e_parallel
arrays, both from the authoritative geometry.

Why the old file is being replaced
----------------------------------
figures_gen/collect_fig_data.py::errdir_profile() produced a "profile" that
cannot reproduce tab:rq5-profile by any subsetting: its Stage-1b buckets span
1.82-2.14 m while the table's critical value is 1.04 m. Three substantive
deviations from the authoritative geometry explain it.

  1. CPA anchor. The old code anchored on the ego's INITIAL position
     (ego0 = x0[:,0:3], a fixed point), while diag_error_direction.measure()
     anchors on the ego REFERENCE TRAJECTORY (ref[:,1:h+1,:]), which moves.
     Different CPA steps put the same physical step in different buckets.
  2. Frame and time alignment. The old code differenced in the recentred /
     scaled space and multiplied by SCALE, indexing nfut[:,0,:h]. The
     authoritative code converts to the absolute frame
     (pred_abs = mean_pred*SCALE + nei_origin) and compares against
     nf[:,0,1:h+1,:] -- one step offset, and a different array.
  3. Episode horizon. The old code sampled T=30, the authoritative path
     T = T_EPISODE = 20, which changes the reference trajectory and hence CPA.

So the profile is recomputed here with the SAME geometry as the table, and the
self-check binds them: the step-count-weighted mean of the buckets at
|k-kCPA| <= 3 must equal the table's critical value, and the buckets beyond it
the inert value. Without that identity the profile has no standing.

Products
--------
errdir_v2.npz
  profile__<arm>        mean |error| per |k - kCPA| bucket, k = 0..MAXD
  counts__<arm>         number of contributing steps per bucket
  epi_par__<arm>        per-episode critical-window mean e_parallel (n=200)
  epi_fneg__<arm>       per-episode toward-ego fraction (n=200)
  epi_ade__<arm>        per-episode critical-window mean |error| (n=200)
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch

ROOT = "/data/lab/TRC_UAV_WEIGHTS"
sys.path.insert(0, f"{ROOT}/code/baselines/common")
sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")

import eval_common as ec                                    # noqa: E402
from guam_encounters import GUAMEncounters                    # noqa: E402
from predictor import GMMTrajectoryPredictor                  # noqa: E402

OUT = f"{ROOT}/code/baselines/figures_gen/fig_data/errdir_v2.npz"
DTYPE = torch.float64
SCALE = 100.0
MAXD = 7
CRIT = 3                     # critical window: |k - kCPA| <= 3
N = 200
T = ec.T_EPISODE             # 20, as in the authoritative path

ARMS = [("Stage1", "stage1_full.pt"),
        ("Stage-1b", "stage1b_domainadapt.pt"),
        ("Stage2", "stage2_final.pt")]

# tab:rq5-profile -- the identity the profile must satisfy
TAB = {"Stage1": (13.53, 23.14), "Stage-1b": (1.04, 2.08),
       "Stage2": (4.07, 4.40)}
# P0_ERRDIR_EPISODE.txt -- episode-level means and SEMs
TAB_EPI = {"Stage1": (-0.699, 0.493, 0.538),
           "Stage-1b": (-0.176, 0.239, 0.454),
           "Stage2": (-0.340, 0.235, 0.546)}
TAB_PAIRED = (-0.164, 0.007399, 0.001137)   # S2-S1b: mean, t p, Wilcoxon p


def load(weights):
    net = GMMTrajectoryPredictor(T=30, K=5).double().to(ec.device_str(True))
    net.load_state_dict(torch.load(f"{ec.PLANGRAD_DIR}/{weights}",
                                   map_location=ec.device_str(True)))
    net.eval()
    return net


@torch.no_grad()
def measure(net, device):
    """Profile buckets + per-episode arrays, using the geometry of
    diag_error_direction.measure() / p0_errdir_episode.per_episode()."""
    ec.set_seed(ec.GLOBAL_SEED)
    gen = GUAMEncounters(ec.GUAM_MAT, ec.EVAL_RANGE, seed=ec.GLOBAL_SEED)
    sums = np.zeros(MAXD + 1)
    cnts = np.zeros(MAXD + 1)
    crit_sum = crit_cnt = inert_sum = inert_cnt = 0.0
    epi_par, epi_fneg, epi_ade = [], [], []
    batch = 8
    for _ in range(N // batch):
        x0, nh, nf, ref, nfut = gen.sample(batch, T, device)
        out = net(nh.reshape(batch, 25, 3))
        mean_pred = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)
        h = min(30, nfut.shape[2])

        # absolute frame, exactly as the authoritative path
        nei_origin = nf[:, 0, 0, :].unsqueeze(1)
        pred_abs = mean_pred[:, :h] * SCALE + nei_origin
        true_abs = nf[:, 0, 1:h + 1, :]
        ego_abs = ref[:, 1:h + 1, :]
        err_vec = pred_abs - true_abs
        axis = true_abs - ego_abs
        axis_n = axis / torch.linalg.norm(
            axis, dim=-1, keepdim=True).clamp_min(1e-6)
        e_par = (err_vec * axis_n).sum(-1)
        ade = torch.linalg.norm(err_vec, dim=-1)

        # CPA from the ego REFERENCE trajectory, not the ego start point
        d = torch.linalg.norm(ego_abs - true_abs, dim=-1)
        ca = d.argmin(dim=1)
        idx = torch.arange(h, device=device).unsqueeze(0)
        dd_all = (idx - ca.unsqueeze(1)).abs()
        crit = dd_all <= CRIT

        for b in range(batch):
            for k in range(h):
                dd = int(dd_all[b, k].item())
                v = float(ade[b, k].item())
                if dd <= MAXD:
                    sums[dd] += v
                    cnts[dd] += 1
                if dd <= CRIT:
                    crit_sum += v
                    crit_cnt += 1
                else:
                    inert_sum += v
                    inert_cnt += 1
            cm = crit[b]
            vals = e_par[b, cm]
            if vals.numel() == 0:
                continue
            epi_par.append(float(vals.mean().cpu()))
            epi_fneg.append(float((vals < 0).double().mean().cpu()))
            epi_ade.append(float(ade[b, cm].mean().cpu()))

    return (sums / np.clip(cnts, 1, None), cnts,
            crit_sum / max(crit_cnt, 1), inert_sum / max(inert_cnt, 1),
            np.array(epi_par), np.array(epi_fneg), np.array(epi_ade))


def main():
    dev = ec.device_str(True)
    out = {}
    crit_inert = {}
    for tag, w in ARMS:
        prof, cnts, cm, im, ep, ef, ea = measure(load(w), dev)
        out[f"profile__{tag}"] = prof
        out[f"counts__{tag}"] = cnts
        out[f"epi_par__{tag}"] = ep
        out[f"epi_fneg__{tag}"] = ef
        out[f"epi_ade__{tag}"] = ea
        crit_inert[tag] = (cm, im)
        print(f"[{tag}] profile = " + " ".join(f"{v:.3f}" for v in prof))
        print(f"        counts  = " + " ".join(f"{int(v)}" for v in cnts))
        print(f"        critical={cm:.4f} (tab {TAB[tag][0]})  "
              f"inert={im:.4f} (tab {TAB[tag][1]})")
        print(f"        epi e_par mean={ep.mean():+.4f} "
              f"SEM={ep.std(ddof=1)/np.sqrt(ep.size):.4f} "
              f"fneg={ef.mean():.4f}  n={ep.size}")

    # ---------------- self-checks ----------------
    errs = []

    def chk(ok, msg):
        print(("  OK   " if ok else "  FAIL ") + msg)
        if not ok:
            errs.append(msg)

    print("\nself-check 1: bucket-weighted critical/inert == tab:rq5-profile")
    for tag, _w in ARMS:
        prof = out[f"profile__{tag}"]
        cnts = out[f"counts__{tag}"]
        wc = float((prof[:CRIT + 1] * cnts[:CRIT + 1]).sum()
                   / cnts[:CRIT + 1].sum())
        chk(abs(wc - TAB[tag][0]) < 5e-3,
            f"{tag} critical: buckets 0-3 weighted = {wc:.4f}, "
            f"table {TAB[tag][0]}")
        chk(abs(crit_inert[tag][1] - TAB[tag][1]) < 5e-3,
            f"{tag} inert: {crit_inert[tag][1]:.4f}, table {TAB[tag][1]}")

    print("self-check 2: episode-level means / SEMs == P0_ERRDIR_EPISODE.txt")
    for tag, _w in ARMS:
        ep = out[f"epi_par__{tag}"]
        ef = out[f"epi_fneg__{tag}"]
        m, s, f = TAB_EPI[tag]
        chk(ep.size == 200, f"{tag} n_epi = {ep.size}")
        chk(abs(ep.mean() - m) < 5e-4, f"{tag} mean {ep.mean():+.4f} vs {m}")
        chk(abs(ep.std(ddof=1) / np.sqrt(ep.size) - s) < 5e-4,
            f"{tag} SEM {ep.std(ddof=1)/np.sqrt(ep.size):.4f} vs {s}")
        chk(abs(ef.mean() - f) < 5e-4, f"{tag} fneg {ef.mean():.4f} vs {f}")

    print("self-check 3: paired S2-S1b reproduces the reported test")
    from scipy import stats
    d = out["epi_par__Stage2"] - out["epi_par__Stage-1b"]
    tp = stats.ttest_rel(out["epi_par__Stage2"],
                         out["epi_par__Stage-1b"]).pvalue
    wp = stats.wilcoxon(out["epi_par__Stage2"],
                        out["epi_par__Stage-1b"]).pvalue
    chk(abs(d.mean() - TAB_PAIRED[0]) < 5e-4,
        f"paired mean {d.mean():+.4f} vs {TAB_PAIRED[0]}")
    chk(abs(tp - TAB_PAIRED[1]) < 5e-5, f"paired t p {tp:.6f} vs {TAB_PAIRED[1]}")
    chk(abs(wp - TAB_PAIRED[2]) < 5e-5, f"Wilcoxon p {wp:.6f} vs {TAB_PAIRED[2]}")
    se = d.std(ddof=1) / np.sqrt(d.size)
    tc = stats.t.ppf(0.975, d.size - 1)
    print(f"       paired CI = [{d.mean()-tc*se:+.4f}, {d.mean()+tc*se:+.4f}]"
          f"   manuscript [-0.283, -0.045]")

    print("self-check 4: Stage-2 spike audit (old file had 12.638 at bucket 1)")
    p2 = out["profile__Stage2"]
    med = float(np.median(p2))
    spikes = [(i, v) for i, v in enumerate(p2) if v > 3 * med]
    if spikes:
        print(f"       buckets exceeding 3x median ({med:.3f}): {spikes}")
    else:
        print(f"       none; profile is smooth about its median {med:.3f}. "
              f"The old 12.638 was an artefact of the deprecated geometry.")

    if errs:
        raise AssertionError(f"{len(errs)} self-check(s) failed; nothing "
                             f"written:\n  " + "\n  ".join(errs))

    out["buckets"] = np.arange(MAXD + 1)
    out["crit_window"] = np.array([CRIT])
    np.savez_compressed(OUT, **out)
    print("\nall self-checks passed. wrote", OUT)


if __name__ == "__main__":
    main()
