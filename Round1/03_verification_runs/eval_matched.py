"""Robustness check eval: evaluate an arbitrary Stage-2 checkpoint against
Stage-1 under the BEST/deploy planner config (alpha=0.1, Hp=15, a_max=20),
n=200, seed 12345 -- identical protocol to final_best.py."""
import sys, torch
from seeding import set_seed
from config import GUAM_MAT
from params import DEFAULT_PARAMS
from dynamics import EVTOLDynamics
from wind import UrbanWindField
from predictor import GMMTrajectoryPredictor
from cbf_mpc import CBFMPCLayer
from safe_policy import SafePolicy
from guam_encounters import GUAMEncounters

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DSEP, N = 30.0, 200
ALPHA, HP, AMAX = 0.1, 15, 20.0

def load(p):
    n = GMMTrajectoryPredictor(T=30, K=5).double().to(DEV)
    n.load_state_dict(torch.load(p, map_location=DEV)); n.eval(); return n

@torch.no_grad()
def evaluate(pred, T=20):
    mpc = CBFMPCLayer(n_neighbors=1, horizon=HP, dt=0.2, d_sep=DSEP, alpha=ALPHA, a_max=AMAX)
    pol = SafePolicy(pred, mpc)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=torch.float64, device=DEV)
    wind = UrbanWindField(eta_w=0.3, dtype=torch.float64, device=DEV, seed=7)
    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=12345)
    n_coll = tot = 0; sep_sum = ade_sum = 0.0; SCALE = 100.0
    for _ in range(N // 8):
        x0, nh, nf, _r, nfut = gen.sample(8, T, DEV)
        out = pred(nh.reshape(8, 25, 3))
        mean_traj = (out["alpha"].unsqueeze(-1) * out["mu"]).sum(2)
        h = min(30, nfut.shape[2])
        ade_sum += torch.linalg.norm(mean_traj[:, :h] - nfut[:, 0, :h, :], dim=-1).mean().item() * SCALE * 8
        x = x0; min_sep = torch.full((8,), 1e6, dtype=torch.float64, device=DEV)
        for t in range(T):
            p0, v0 = x[:, 0:3], x[:, 3:6]
            tt = torch.arange(HP + 1, dtype=torch.float64, device=DEV) * 0.2
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            try:
                u, _ = pol(x, nh, nf[:, :, t, :], p_ref)
            except Exception:
                u = torch.zeros(8, 4, dtype=torch.float64, device=DEV); u[:, 0] = DEFAULT_PARAMS.weight
            x = dyn.step(x, u, wind.sample(p0), 0.2)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            min_sep = torch.minimum(min_sep, d)
        n_coll += int((min_sep < DSEP).sum()); sep_sum += float(min_sep.sum()); tot += 8
    return 100.0 * n_coll / tot, sep_sum / tot, ade_sum / tot

if __name__ == "__main__":
    s2_path = sys.argv[1] if len(sys.argv) > 1 else "stage2_matched.pt"
    set_seed(12345)
    print("MATCHED-CONFIG robustness eval: planner a=%.1f Hp=%d amax=%.0f, n=%d, seed 12345"%(ALPHA,HP,AMAX,N))
    cr1, sep1, ade1 = evaluate(load("stage1_full.pt"))
    print("Stage-1            CR=%.1f%%  minSep=%.1f m  ADE=%.2f m" % (cr1, sep1, ade1))
    cr2, sep2, ade2 = evaluate(load(s2_path))
    print("Stage-2(matched)   CR=%.1f%%  minSep=%.1f m  ADE=%.2f m" % (cr2, sep2, ade2))
    print("delta              CR %+.1f pts  minSep %+.1f m  ADE %+.2f m" % (cr2-cr1, sep2-sep1, ade2-ade1))
