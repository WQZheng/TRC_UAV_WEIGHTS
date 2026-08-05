"""Export the planner regime surface (fig11) to landscape_v2.npz.

The surface itself was already computed and lives in planner_heatmap_n200.json.
This script does not recompute it -- it validates it, resolves the gamma/alpha
naming, attaches the independent-stream comparison that is actually admissible,
and records which reference points may be drawn on the surface and which may not.

Nothing here is wrapped in try/except: a wrong path must crash (rule_2).
"""
import json
import os
import sys

import numpy as np

DATA = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
SRC = f"{DATA}/planner_heatmap_n200.json"
OUT = f"{DATA}/landscape_v2.npz"

# The independent-stream sweep, transcribed from
# code/plangrad_sim/PLANNER_SCAN_INDEP_seed999.txt (md5 353933e7..., identical
# in all three copies). Six configurations, not a grid: alpha, Hp and a_max all
# vary together, so only the row whose Hp equals the surface's Hp is comparable.
INDEP = [
    # (label,                  gamma, Hp, a_max, CR%,  minSep)
    ("alpha0.4 Hp8  amax10",   0.4,   8,  10.0, 26.0, 34.4),
    ("alpha0.2 Hp8  amax10",   0.2,   8,  10.0, 26.0, 35.7),
    ("alpha0.1 Hp8  amax10",   0.1,   8,  10.0, 26.0, 36.0),
    ("alpha0.2 Hp12 amax10",   0.2,  12,  10.0, 28.0, 34.0),
    ("alpha0.2 Hp12 amax16",   0.2,  12,  16.0, 18.0, 42.2),
    ("alpha0.1 Hp15 amax20",   0.1,  15,  20.0, 12.0, 48.5),
]

# Training / weak-certificate configuration. These are THE SAME POINT: the weak
# experiments reuse the configuration the Stage-2 predictor was fine-tuned under.
# Evidence for Hp=8 is three independent documents agreeing, not a runtime echo:
#   train_stage2.py:139   argparse default --Hp 8
#   train_stage2.py:15    docstring "Verified config ... --T 20 --Hp 8"
#   README_p0_referee.md:6 "Loose config: CBF-MPC alpha=0.4, a_max=10, Hp=8"
# stage2_train.log does not print hyperparameters, so this is documentary
# agreement rather than direct observation (rule_3), and is labelled as such.
TRAIN_WEAK = dict(gamma=0.4, Hp=8, a_max=10.0)
DEPLOY = dict(gamma=0.1, Hp=15, a_max=20.0)


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC}")
    with open(SRC) as fh:
        J = json.load(fh)

    gam = np.asarray(J["gammas"], float)       # rows
    amax = np.asarray(J["amaxs"], float)       # cols
    CR = np.asarray(J["CR"], float)            # CR[i_gamma, j_amax]
    Hp_surface = int(J["Hp"])

    errs = []

    # --- the gamma/alpha identity, asserted rather than assumed --------------
    # The manuscript calls the CBF class-K coefficient gamma (definition at
    # 05_experiments_results_v9.tex:799-800, h_k >= (1-gamma) h_{k-1},
    # gamma in (0,1]). The code calls the same argument alpha
    # (CBFMPCLayer(alpha=...)), which is the usual CBF-literature name; the
    # manuscript renamed it because alpha_{i,m} is already the predictor's
    # mixture weight (tex:670,701). Same quantity, two names. The json stores it
    # under "gammas" while scan_planner_indep.py passes it as alpha, so the two
    # must agree elementwise or the surface is mislabelled.
    scan_alphas = sorted({g for _l, g, _h, _a, _c, _s in INDEP})
    for a in scan_alphas:
        if not np.any(np.isclose(gam, a)):
            errs.append(f"scan alpha={a} absent from json gammas {gam.tolist()}")
    if not np.all(np.diff(gam) > 0):
        errs.append("gammas not ascending")
    if not np.all((gam > 0.0) & (gam <= 1.0)):
        errs.append("gamma outside the manuscript's (0,1] domain")

    # --- shape and provenance -----------------------------------------------
    if CR.shape != (gam.size, amax.size):
        errs.append(f"CR shape {CR.shape} != ({gam.size},{amax.size})")
    if J["predictor"] != "stage2_final.pt":
        errs.append(f"unexpected predictor {J['predictor']}")
    if int(J["n"]) != 200:
        errs.append(f"n={J['n']} != 200")
    if Hp_surface != 15:
        errs.append(f"surface Hp={Hp_surface} != 15")

    # --- the deployment point must be ON the surface -------------------------
    di = int(np.argmin(np.abs(gam - DEPLOY["gamma"])))
    dj = int(np.argmin(np.abs(amax - DEPLOY["a_max"])))
    if not (np.isclose(gam[di], DEPLOY["gamma"])
            and np.isclose(amax[dj], DEPLOY["a_max"])):
        errs.append("deployment point is not a grid node")
    if Hp_surface != DEPLOY["Hp"]:
        errs.append("deployment Hp differs from the surface Hp; it would not "
                    "be drawable as a cell")
    cr_deploy = float(CR[di, dj])

    # --- the training/weak point must be OFF the surface ---------------------
    # Asserted in the failing direction on purpose: if this ever stops being
    # true the figure's annotation becomes wrong, and it must break loudly.
    if TRAIN_WEAK["Hp"] == Hp_surface:
        errs.append("training/weak Hp now equals the surface Hp; the "
                    "projection-only annotation is no longer correct")
    wi = int(np.argmin(np.abs(gam - TRAIN_WEAK["gamma"])))
    wj = int(np.argmin(np.abs(amax - TRAIN_WEAK["a_max"])))
    cr_weak_projection = float(CR[wi, wj])

    # --- the independent stream: exactly one comparable row ------------------
    comparable = [r for r in INDEP
                  if r[2] == Hp_surface
                  and np.any(np.isclose(gam, r[1]))
                  and np.any(np.isclose(amax, r[3]))]
    if len(comparable) != 1:
        errs.append(f"expected exactly 1 stream-comparable config, "
                    f"got {len(comparable)}")
    if comparable:
        lab, g, hp, am, cr_b, _sep = comparable[0]
        if not (np.isclose(g, DEPLOY["gamma"])
                and np.isclose(am, DEPLOY["a_max"])):
            errs.append("the comparable stream row is not the deployment point")
        cr_stream_b = float(cr_b)
    else:
        cr_stream_b = float("nan")

    # --- the structural claim the figure is built on -------------------------
    # Range along a_max versus range along gamma. If gamma ever stopped being
    # nearly inert, the top profile panel would be misdesigned.
    # Per-row spans are what the caption quotes: for a fixed gamma, how far does
    # raising a_max from 5 to 20 move the conflict rate. This is the directly
    # interpretable quantity. Two other spans were computed earlier and both are
    # awkward: the difference of column maxima (69.5 pp) mixes rows, and the
    # global max-min (70.5 pp) happens to coincide with the deployment row only by
    # accident. Only per-row spans are exported for the caption.
    span_rows = CR[:, 0] - CR[:, -1]
    span_amax = float(CR.max(axis=0).max() - CR.max(axis=0).min())
    span_gamma_per_col = CR.max(axis=0) - CR.min(axis=0)
    span_gamma_max = float(span_gamma_per_col.max())
    if span_gamma_max >= span_amax:
        errs.append("gamma is not inert relative to a_max; the profile panels "
                    "would need redesigning")

    if errs:
        raise AssertionError("landscape export self-check failed:\n  "
                             + "\n  ".join(errs))

    np.savez(
        OUT,
        gammas=gam, amaxs=amax, CR=CR,
        Hp_surface=np.array(Hp_surface),
        n_per_cell=np.array(int(J["n"])),
        seed=np.array(int(J["seed"])),
        eta_w=np.array(float(J["eta_w"])),
        eval_range=np.asarray(J["eval_range"], int),
        predictor=np.array(J["predictor"]),
        deploy_ij=np.array([di, dj]),
        deploy_cr=np.array(cr_deploy),
        weak_ij=np.array([wi, wj]),
        weak_cr_projection=np.array(cr_weak_projection),
        weak_Hp=np.array(TRAIN_WEAK["Hp"]),
        stream_b_cr=np.array(cr_stream_b),
        stream_b_seed=np.array(999),
        indep_labels=np.array([r[0] for r in INDEP]),
        indep_gamma=np.array([r[1] for r in INDEP]),
        indep_Hp=np.array([r[2] for r in INDEP]),
        indep_amax=np.array([r[3] for r in INDEP]),
        indep_cr=np.array([r[4] for r in INDEP]),
        indep_minsep=np.array([r[5] for r in INDEP]),
        span_amax=np.array(span_amax),
        span_gamma_max=np.array(span_gamma_max),
        span_gamma_per_col=span_gamma_per_col,
        span_rows=span_rows,
    )

    print(f"wrote {OUT}")
    print(f"  surface: {gam.size}x{amax.size}, Hp={Hp_surface}, "
          f"n={J['n']}/cell, predictor={J['predictor']}, seed={J['seed']}")
    print(f"  gamma == code alpha: verified for {scan_alphas}")
    print(f"  deployment (gamma={DEPLOY['gamma']}, a_max={DEPLOY['a_max']:.0f}) "
          f"on surface, CR={cr_deploy:.1f}%")
    print(f"  stream B at the same node: CR={cr_stream_b:.1f}% (seed 999)")
    print(f"  training/weak (gamma={TRAIN_WEAK['gamma']}, "
          f"a_max={TRAIN_WEAK['a_max']:.0f}, Hp={TRAIN_WEAK['Hp']}) OFF surface; "
          f"cell there reads {cr_weak_projection:.1f}% under Hp={Hp_surface}")
    print(f"  per-row a_max spans (5 -> 20): "
          + ", ".join(f"gamma={g:.1f}: {v:.1f}pp"
                      for g, v in zip(gam, span_rows)))
    print(f"  deployment row (gamma={gam[0]:.1f}): {span_rows[0]:.1f} pp; "
          f"range across rows {span_rows.min():.1f}-{span_rows.max():.1f} pp")
    print(f"  gamma span <= {span_gamma_max:.1f} pp -> gamma inert")
    print("  per-column gamma spans: "
          + ", ".join(f"a_max={a:.0f}: {s:.1f}pp"
                      for a, s in zip(amax, span_gamma_per_col)))


if __name__ == "__main__":
    main()
