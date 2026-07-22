"""Detection-horizon (lead-time) sweep  --  referee question:

  "Is 'residual conflicts are actuation-limited, prediction has no leverage'
   a PROPERTY of low-altitude operation, or an ARTIFACT of an encounter
   generator that makes conflicts detectable only ~1 s before closest
   approach (CPA), turning every encounter into a pure maneuver-authority
   race that no predictor could win?"

Design
------
The medium-difficulty GUAM encounter places closest approach at step
``T//2`` (2.0 s at dt=0.2) into the rollout. We generalise this: a single
parameter ``t_cpa`` (in steps) sets how far AHEAD of CPA the episode
begins, i.e. the *detection / actionable horizon*. Everything else in the
encounter geometry (per-encounter offset distribution, neighbour speed,
yaw, the real GUAM segments, the planner, a_max, seed) is held fixed, so
the ONLY thing that changes across the sweep is how early the conflict is
observable.

For each horizon h in {3,6,10,20} s we roll out the SAME held-out
encounters under the tuned CBF-MPC planner (alpha=0.1, Hp=15, a_max=20)
with three neighbour-prediction conditions:

  * Stage-1   : displacement-trained predictor        (frozen)
  * Stage-2   : task-aligned predictor  = PlanGrad
  * Oracle    : the TRUE neighbour future is fed to the planner
                (an upper bound: the best any predictor could possibly do)

Conflict attribution (per encounter)
------------------------------------
  actuation-limited  : the ORACLE also conflicts -> no predictor, however
                       perfect, could have avoided it with the available
                       control authority at this horizon. Physics-bound.
  prediction-limited : the oracle avoids it but Stage-2 does not -> better
                       prediction WOULD have helped. Prediction has leverage.

If actuation-limited dominates at short horizon but prediction-limited
grows at long horizon, the "authority-governed" claim is horizon-specific:
under tactical last-second de-confliction safety is maneuver-bound, but
with earlier detection prediction accuracy regains leverage -- a stronger,
more operationally useful conclusion (it quantifies the value of detection
lead time to an air-traffic operator).

Eval-only; no gradients; uses FastCBFMPC (verified equivalent to the
differentiable layer). Writes LEADTIME.txt.

Usage:
    export GUAM_MAT=/path/Data_Set_1.mat
    python3 eval_leadtime.py --n 200 --seed 12345 \
        --stage1 stage1_full.pt --stage2 stage2_final.pt
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from seeding import set_seed
from config import GUAM_MAT
from params import DEFAULT_PARAMS
from dynamics import EVTOLDynamics
from wind import UrbanWindField
from predictor import GMMTrajectoryPredictor
from guam_encounters import GUAMEncounters
from fast_cbf_mpc import FastCBFMPC

DTYPE = torch.float64
SCALE = 100.0
D_SEP = 30.0
DT = 0.2
HP = 15
AMAX = 20.0
ALPHA = 0.1


class LeadTimeEncounters(GUAMEncounters):
    """GUAMEncounters variant that places closest approach at a configurable
    step ``t_cpa`` (instead of the hard-coded T//2), so the actionable
    detection horizon before CPA can be swept. All other geometry identical.
    """
    def __init__(self, *a, t_cpa: int = 10, **k):
        self.t_cpa = int(t_cpa)
        super().__init__(*a, **k)

    def sample(self, B, T, device):
        L = self.L
        He = self.extra_horizon
        # ensure the neighbour window is long enough to reach t_cpa + buffer
        need = L + T + He + 1
        x0 = torch.zeros(B, 12, dtype=DTYPE)
        neigh_hist = torch.zeros(B, 1, L, 3, dtype=DTYPE)
        neigh_full = torch.zeros(B, 1, T + He + 1, 3, dtype=DTYPE)
        ego_ref = torch.zeros(B, T + He + 1, 3, dtype=DTYPE)
        neigh_fut = torch.zeros(B, 1, 30, 3, dtype=DTYPE)

        cpa = min(self.t_cpa, T)                    # CPA step within rollout
        for b in range(B):
            ego = self._rand_window(need)
            nei = self._rand_window(need)

            ego_now = ego[L - 1]
            ego_path = ego[L - 1:]
            ego_ref[b, :, :] = torch.tensor(ego_path[:T + He + 1])
            x0[b, 0:3] = torch.tensor(ego_now)
            v0 = (ego[L - 1] - ego[L - 2]) / self.dt
            x0[b, 3:6] = torch.tensor(v0)

            nei_now = nei[L - 1]
            nei_rel = (nei - nei_now) * self.neigh_speed_scale
            theta = self.rng.uniform(0.0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            nei_rot = nei_rel @ Rz.T

            mid_idx = min(cpa, ego_path.shape[0] - 1)
            mid = ego_path[mid_idx]
            nei_mid_idx = L - 1 + cpa
            nei_mid_rel = nei_rot[nei_mid_idx] - nei_rot[L - 1]
            off_mag = self.rng.uniform(*self.approach_offset)
            off_dir = self.rng.normal(size=3); off_dir[2] = 0.0
            off_dir = off_dir / (np.linalg.norm(off_dir) + 1e-9)
            offset = mid - nei_mid_rel + off_dir * off_mag
            offset[2] = 0.0
            nei_abs = nei_rot - nei_rot[L - 1] + offset

            neigh_full[b, 0, :, :] = torch.tensor(
                nei_abs[L - 1:L - 1 + T + He + 1])
            hist = nei_abs[:L] - nei_abs[L - 1]
            neigh_hist[b, 0, :, :] = torch.tensor(hist / SCALE)
            fut = (nei_abs[L:L + 30] - nei_abs[L - 1]) / SCALE
            neigh_fut[b, 0, :fut.shape[0], :] = torch.tensor(fut)

        return (x0.to(device), neigh_hist.to(device), neigh_full.to(device),
                ego_ref.to(device), neigh_fut.to(device))


def load_pred(path, dev):
    net = GMMTrajectoryPredictor(T=30, K=5).double().to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()
    return net


@torch.no_grad()
def predict_meantraj(pred, neigh_hist_b, last_pos_b):
    """Predictor mean future in ABSOLUTE coords -> (Hp+1,3) numpy for one agent."""
    out = pred(neigh_hist_b.reshape(1, 25, 3))
    mean = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)[0]     # (30,3) scaled/rel
    h = HP + 1
    traj = mean[:h] * SCALE + last_pos_b.reshape(1, 3)
    if traj.shape[0] < h:                     # pad by holding last
        pad = traj[-1:].repeat(h - traj.shape[0], 1)
        traj = torch.cat([traj, pad], 0)
    return traj.cpu().numpy()


@torch.no_grad()
def rollout_condition(gen, cond, dev, n, T, pred=None):
    """Run closed loop for one prediction condition.
    cond in {'stage','oracle'}; pred supplied for 'stage'.
    Returns boolean array conflict[n] (min separation < D_SEP)."""
    planner = FastCBFMPC(n_neighbors=1, horizon=HP, dt=DT, d_sep=D_SEP,
                         a_max=AMAX, alpha=ALPHA)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=dev)
    wind = UrbanWindField(eta_w=0.3, dtype=DTYPE, device=dev, seed=7)
    weight = DEFAULT_PARAMS.weight
    conflicts = []
    B = 8
    for _ in range(n // B):
        x0, nh, nf, _ref, _ = gen.sample(B, T, dev)
        x = x0
        min_sep = torch.full((B,), 1e6, dtype=DTYPE, device=dev)
        for t in range(T):
            p0 = x[:, 0:3]; v0 = x[:, 3:6]
            tt = torch.arange(HP + 1, dtype=DTYPE, device=dev) * DT
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            u = torch.zeros(B, 4, dtype=DTYPE, device=dev); u[:, 0] = weight
            for b in range(B):
                # neighbour prediction (absolute) fed to the planner
                if cond == "oracle":
                    # true future neighbour positions over the horizon
                    idx = [min(t + k, nf.shape[2] - 1) for k in range(HP + 1)]
                    npred = nf[b, 0, idx, :].cpu().numpy()[None]     # (1,Hp+1,3)
                else:
                    npred = predict_meantraj(pred, nh[b, 0], nf[b, 0, t, :])[None]
                a0 = planner.solve_np(p0[b].cpu().numpy(), v0[b].cpu().numpy(),
                                      p_ref[b].cpu().numpy(), npred)
                if a0 is None:
                    continue
                a0 = torch.tensor(a0, dtype=DTYPE, device=dev)
                # map accel -> control via the same clamped inner law as SafePolicy
                m = DEFAULT_PARAMS.mass; g = DEFAULT_PARAMS.g
                f_des = m * a0.clone(); f_des[2] += m * g
                thrust = torch.linalg.norm(f_des).clamp_min(1.0)
                ax = f_des[0] / thrust; ay = f_des[1] / thrust
                tilt = 0.45
                roll = torch.clamp(-ay, -tilt, tilt)
                pitch = torch.clamp(ax, -tilt, tilt)
                eta = x[b, 6:9]; om = x[b, 9:12]
                I = torch.tensor(DEFAULT_PARAMS.inertia_diag, dtype=DTYPE, device=dev)
                mom = torch.stack([(2.0 * (roll - eta[0]) - 1.5 * om[0]) * I[0],
                                   (2.0 * (pitch - eta[1]) - 1.5 * om[1]) * I[1],
                                   (-1.5 * om[2]) * I[2]])
                mmax = DEFAULT_PARAMS.max_body_moment
                mom = torch.clamp(mom, -mmax, mmax)
                u[b] = torch.cat([torch.linalg.norm(f_des).reshape(1), mom])
            x = dyn.step(x, u, wind.sample(p0), DT)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            min_sep = torch.minimum(min_sep, d)
        conflicts.append((min_sep < D_SEP).cpu().numpy())
    return np.concatenate(conflicts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--stage1", default="stage1_full.pt")
    ap.add_argument("--stage2", default="stage2_final.pt")
    ap.add_argument("--horizons", default="3,6,10,20",
                    help="detection horizons in seconds (CPA lead time)")
    ap.add_argument("--out", default="LEADTIME.txt")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s1 = load_pred(args.stage1, dev)
    s2 = load_pred(args.stage2, dev)
    horizons_s = [float(x) for x in args.horizons.split(",")]

    fh = open(args.out, "w")

    def w(s=""):
        print(s, flush=True)
        fh.write(s + "\n"); fh.flush()

    w("=" * 78)
    w("DETECTION-HORIZON (LEAD-TIME) SWEEP")
    w("Tuned CBF-MPC (alpha=%.1f Hp=%d a_max=%.0f d_sep=%.0f), n=%d seed=%d"
      % (ALPHA, HP, AMAX, D_SEP, args.n, args.seed))
    w("CPA placed at t_cpa = horizon/dt steps; geometry otherwise fixed.")
    w("Conflict = min separation < %.0f m. Oracle = true neighbour future." % D_SEP)
    w("=" * 78)
    w("%6s | %8s %8s %8s | %s" %
      ("horiz", "S1 CR%", "S2 CR%", "Oracle%", "attribution of S2 conflicts"))

    for h_s in horizons_s:
        t_cpa = int(round(h_s / DT))
        T = t_cpa + HP + 2          # roll out past CPA so the miss is realised
        def mk():
            return LeadTimeEncounters(GUAM_MAT, range(2500, 3000), seed=args.seed,
                                      t_cpa=t_cpa)
        set_seed(args.seed); c1 = rollout_condition(mk(), "stage", dev, args.n, T, pred=s1)
        set_seed(args.seed); c2 = rollout_condition(mk(), "stage", dev, args.n, T, pred=s2)
        set_seed(args.seed); co = rollout_condition(mk(), "oracle", dev, args.n, T)

        n = len(c2)
        cr1, cr2, cro = 100.0 * c1.mean(), 100.0 * c2.mean(), 100.0 * co.mean()
        # attribution among Stage-2 conflicts
        s2_conf = c2.sum()
        act_lim = int((c2 & co).sum())               # oracle also conflicts
        pred_lim = int((c2 & ~co).sum())             # oracle avoids, S2 does not
        frac_act = 100.0 * act_lim / max(1, s2_conf)
        frac_pred = 100.0 * pred_lim / max(1, s2_conf)
        w("%5.0fs | %8.1f %8.1f %8.1f | of %d S2-conflicts: %.0f%% actuation-limited, "
          "%.0f%% prediction-limited"
          % (h_s, cr1, cr2, cro, int(s2_conf), frac_act, frac_pred))

    w("=" * 78)
    w("READING GUIDE")
    w(" - Oracle% is the best achievable CR at each horizon (perfect prediction).")
    w(" - If Oracle% ~ S2% at short horizon, prediction has NO leverage there")
    w("   (authority-bound). If Oracle% << S2% at long horizon, better prediction")
    w("   WOULD reduce conflicts -> prediction regains leverage with lead time.")
    w(" - 'prediction-limited' fraction = share of PlanGrad conflicts an oracle")
    w("   predictor would have avoided at that detection horizon.")
    w("=" * 78)
    fh.close()


if __name__ == "__main__":
    main()
