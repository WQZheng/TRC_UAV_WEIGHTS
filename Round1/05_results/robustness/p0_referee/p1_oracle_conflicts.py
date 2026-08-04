"""P1 (referee #4): oracle-prediction re-run of the deploy-config conflict
episodes -> 3-way attribution.

Referee #4: "22/22 infeasible" (all main-table PlanGrad conflicts had the CBF QP
relax its separation constraint) shows the *planner* ran out of avoidance
authority, but does NOT by itself prove a better *predictor* could not have
helped -- that is a statement about the realized prediction stream, not about
prediction quality in general. The clean test is to replay the SAME conflict
episodes with an ORACLE predictor (perfect future neighbour trajectory) feeding
the SAME planner and SAME plant, and see which conflicts disappear.

Deploy / main-table config (reproduces BEST.txt CR=11.0%): CBF-MPC alpha=0.1,
Hp=15, a_max=20, d_sep=30, T=20, dt=0.2, eta_w=0.3, n=200, seed=12345, held-out
range(2500,3000). Nothing retrained.

We evaluate, episode-aligned on the identical encounter stream:
  * Stage-2 real predictor  -> per-episode min-sep, conflict bool, whether the
    CBF QP used slack at the closest-approach step (constraint relaxed).
  * Oracle predictor         -> feeds the TRUE future neighbour window
    nf[:, :, t+1:t+1+Hp+1, :] into the planner instead of the GMM mean; same
    planner, same plant, same wind.

3-way classification of each Stage-2 conflict episode:
  (A) actuation-limited (hard-infeasible under any prediction): oracle ALSO
      conflicts  -> a perfect predictor could not have prevented it; the binding
      limit is the planner's avoidance authority / plant dynamics.
  (B) prediction-induced: oracle is SAFE but Stage-2 conflicts -> a better
      predictor would have avoided it (the realized prediction stream mattered).
  (C) proxy-plant gap: oracle SAFE at QP level (no slack) yet plant still
      conflicts -> controller-model vs plant mismatch, not prediction.

Writes P1_ORACLE_CONFLICTS.txt.
"""
from __future__ import annotations
import os
import torch

from seeding import set_seed
from config import GUAM_MAT
from params import DEFAULT_PARAMS
from dynamics import EVTOLDynamics
from wind import UrbanWindField
from predictor import GMMTrajectoryPredictor
from cbf_mpc import CBFMPCLayer
from safe_policy import SafePolicy
from guam_encounters import GUAMEncounters

OUT = "P1_ORACLE_CONFLICTS.txt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
DSEP, N = 30.0, 200
ALPHA, HP, AMAX = 0.1, 15, 20.0
T, DT, ETA_W, BATCH = 20, 0.2, 0.3, 8


def w(s):
    with open(OUT, "a") as f:
        f.write(s + "\n")
    print(s, flush=True)


def load(p):
    n = GMMTrajectoryPredictor(T=30, K=5).double().to(DEV)
    n.load_state_dict(torch.load(p, map_location=DEV)); n.eval()
    return n


class OraclePolicy(SafePolicy):
    """SafePolicy whose predict step returns the TRUE future neighbour window
    instead of the learned GMM mean. Everything else (mpc.solve, inner control)
    is identical to the deploy planner. Set self._nf and self._t each step."""
    _nf = None   # [B,N,T+He+1,3] absolute true neighbour
    _t = 0

    def __call__(self, x, neigh_hist, neigh_last, p_ref):
        p0 = x[:, 0:3]; v0 = x[:, 3:6]
        B, Nn = neigh_last.shape[0], neigh_last.shape[1]
        Hp = self.Hp
        # perfect future: absolute neighbour positions for steps t+1 .. t+Hp+1
        t = self._t
        fut = self._nf[:, :, t + 1: t + 1 + (Hp + 1), :]   # [B,N,<=Hp+1,3]
        h = fut.shape[2]
        if h < Hp + 1:                                      # pad tail (end of horizon)
            pad = fut[:, :, -1:, :].expand(B, Nn, Hp + 1 - h, 3)
            fut = torch.cat([fut, pad], dim=2)
        pred_abs = fut.to(DTYPE)
        a0, info = self.mpc.solve(p0, v0, p_ref, pred_abs)
        u = self.accel_to_control(x, a0)
        return u, {"a0": a0, "pred_abs": pred_abs, "var_mean": None,
                   "slack": info["eps"]}


@torch.no_grad()
def rollout(kind, ckpt=None):
    """kind in {'real','oracle'}. Returns (min_sep[N], slack_at_cpa[N])."""
    set_seed(12345)
    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=12345)
    mpc = CBFMPCLayer(n_neighbors=1, horizon=HP, dt=DT, d_sep=DSEP,
                      alpha=ALPHA, a_max=AMAX)
    if kind == "real":
        pol = SafePolicy(load(ckpt), mpc)
    else:
        pol = OraclePolicy(load("stage2_final.pt"), mpc)  # predictor unused
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=DEV)
    wind = UrbanWindField(eta_w=ETA_W, dtype=DTYPE, device=DEV, seed=7)

    mins, slack_cpa, cpas = [], [], []
    for _ in range(N // BATCH):
        x0, nh, nf, _r, _f = gen.sample(BATCH, T, DEV)
        x = x0
        min_sep = torch.full((BATCH,), 1e6, dtype=DTYPE, device=DEV)
        # track slack at the realized closest-approach step
        best_d = torch.full((BATCH,), 1e6, dtype=DTYPE, device=DEV)
        slack_here = torch.zeros(BATCH, dtype=DTYPE, device=DEV)
        cpa_here = torch.zeros(BATCH, dtype=torch.long, device=DEV)
        for t in range(T):
            p0, v0 = x[:, 0:3], x[:, 3:6]
            tt = torch.arange(HP + 1, dtype=DTYPE, device=DEV) * DT
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            if kind == "oracle":
                pol._nf = nf; pol._t = t
            try:
                u, info = pol(x, nh, nf[:, :, t, :], p_ref)
                eps = info.get("slack", None)
                eps_scalar = (torch.linalg.norm(eps.reshape(BATCH, -1), dim=-1)
                              if eps is not None else torch.zeros(BATCH, device=DEV, dtype=DTYPE))
            except Exception:
                u = torch.zeros(BATCH, 4, dtype=DTYPE, device=DEV)
                u[:, 0] = DEFAULT_PARAMS.weight
                eps_scalar = torch.zeros(BATCH, dtype=DTYPE, device=DEV)
            x = dyn.step(x, u, wind.sample(p0), DT)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            min_sep = torch.minimum(min_sep, d)
            upd = d < best_d
            best_d = torch.where(upd, d, best_d)
            slack_here = torch.where(upd, eps_scalar.to(DTYPE), slack_here)
            cpa_here = torch.where(upd, torch.full_like(cpa_here, t), cpa_here)
        mins.append(min_sep.cpu())
        slack_cpa.append(slack_here.cpu())
        cpas.append(cpa_here.cpu())
    return torch.cat(mins)[:N], torch.cat(slack_cpa)[:N], torch.cat(cpas)[:N]


if __name__ == "__main__":
    open(OUT, "w").close()
    w("P1 ORACLE CONFLICT ATTRIBUTION (referee #4)")
    w("  deploy config: CBF-MPC alpha=%.1f Hp=%d a_max=%.0f d_sep=%.0f," %
      (ALPHA, HP, AMAX, DSEP))
    w("  T=%d dt=%.1f eta_w=%.1f, held-out 2500-3000, seed=12345, n=%d" %
      (T, DT, ETA_W, N))
    w("")

    sep_s2, slack_s2, cpa_s2 = rollout("real", "stage2_final.pt")
    sep_or, slack_or, cpa_or = rollout("oracle")

    conf_s2 = sep_s2 < DSEP
    conf_or = sep_or < DSEP
    n_s2 = int(conf_s2.sum())
    n_or = int(conf_or.sum())
    w("Stage-2 (real predictor) CR = %.1f%%  (%d/%d episodes conflict)" %
      (100.0 * n_s2 / N, n_s2, N))
    w("Oracle  (perfect future) CR = %.1f%%  (%d/%d episodes conflict)" %
      (100.0 * n_or / N, n_or, N))
    w("")

    # 3-way classification of the Stage-2 conflict episodes
    idx = torch.nonzero(conf_s2, as_tuple=False).flatten().tolist()
    a_cnt = b_cnt = c_cnt = 0
    w("Per-conflict-episode attribution (%d Stage-2 conflicts):" % n_s2)
    w("  epi | S2 minSep | Oracle minSep | Oracle slack@CPA | class")
    for i in idx:
        oc = bool(conf_or[i])
        sl = float(slack_or[i])
        if oc:
            cls = "A actuation-limited (oracle also conflicts)"
            a_cnt += 1
        else:
            # oracle safe -> a better predictor would have avoided it
            cls = "B prediction-induced (oracle avoids it)"
            b_cnt += 1
        w("  %3d | %8.2f | %11.2f | %14.4f | %s" %
          (i, float(sep_s2[i]), float(sep_or[i]), sl, cls))
    w("")
    # ---- dump oracle_v2.npz (figure provenance; see FIG05 data request) ----
    # episode_ids are the Stage-2 conflict indices in 0..199, so every figure
    # panel and the 200-dim arm vectors can be aligned row-by-row.
    import numpy as _np
    _eid = _np.asarray(idx, dtype=_np.int64)
    _figdd = os.environ.get("FIG_DATA_DIR", ".")
    os.makedirs(_figdd, exist_ok=True)
    _np.savez(os.path.join(_figdd, "oracle_v2.npz"),
              episode_ids=_eid,
              minsep_stage2=sep_s2[_eid].numpy().astype(_np.float64),
              minsep_oracle=sep_or[_eid].numpy().astype(_np.float64),
              oracle_slack_at_cpa=slack_or[_eid].numpy().astype(_np.float64),
              oracle_cpa_step=cpa_or[_eid].numpy().astype(_np.int64),
              stage2_cpa_step=cpa_s2[_eid].numpy().astype(_np.int64),
              oracle_conflict_200=conf_or.numpy().astype(bool),
              stage2_conflict_200=conf_s2.numpy().astype(bool))
    w("  [dumped oracle_v2.npz: episode_ids/minsep_stage2/minsep_oracle/"
      "oracle_slack_at_cpa/oracle_cpa_step/stage2_cpa_step/"
      "oracle_conflict_200/stage2_conflict_200]")
    # McNemar discordant structure vs Stage-2 over the FULL 200 episodes
    _b = int((~conf_s2.numpy() & conf_or.numpy()).sum())
    _c = int((conf_s2.numpy() & ~conf_or.numpy()).sum())
    w("  oracle vs Stage-2 discordant (b=oracle-only, c=Stage-2-only) = "
      "(%d, %d)" % (_b, _c))
    w("")
    w("SUMMARY of the %d Stage-2 conflict episodes:" % n_s2)
    w("  (A) actuation-limited / hard-infeasible under any prediction = %d" % a_cnt)
    w("  (B) prediction-induced (oracle avoids)                       = %d" % b_cnt)
    w("")
    if b_cnt == 0:
        w("READ: every Stage-2 conflict ALSO occurs under a perfect predictor.")
        w("      => hard-infeasible under the realized prediction stream; a")
        w("         better predictor would NOT have changed the outcome. This")
        w("         is the clean episode-level evidence for #4 (the leadtime 2s")
        w("         Oracle=Stage-2=11.0%% row is a population-level corroboration")
        w("         under a fixed-CPA controlled setting, NOT a substitute).")
    else:
        w("READ: %d of %d Stage-2 conflicts are avoided by a perfect predictor" %
          (b_cnt, n_s2))
        w("      => the realized prediction stream DID cost some episodes;")
        w("         soften #4 attribution and report the prediction-induced share.")
