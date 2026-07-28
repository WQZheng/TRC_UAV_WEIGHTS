"""Why do conflicts happen? Decompose on held-out (2500-3000).

Records whether the CBF QP used slack (constraint relaxed = physically
hard) and the predicted-vs-true neighbour error. Result on this setup:
~all conflicts are slack-active with small prediction error => the
bottleneck is the PLANNER's avoidance authority, not the predictor.
Writes DIAG.txt.
"""
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

OUT = "DIAG.txt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DSEP = 30.0
set_seed(12345)


def load(p):
    n = GMMTrajectoryPredictor(T=30, K=5).double().to(DEV)
    n.load_state_dict(torch.load(p, map_location=DEV))
    n.eval()
    return n


@torch.no_grad()
def run(ckpt, N=160, T=20, Hp=8):
    pred = load(ckpt)
    mpc = CBFMPCLayer(n_neighbors=1, horizon=Hp, dt=0.2, d_sep=DSEP,
                      alpha=0.4, a_max=10.0)
    pol = SafePolicy(pred, mpc)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=torch.float64, device=DEV)
    wind = UrbanWindField(eta_w=0.3, dtype=torch.float64, device=DEV, seed=7)
    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=12345)
    n_coll = coll_slack = coll_pred = tot = 0
    slack_sum = perr_sum = 0.0
    for _ in range(N // 8):
        x0, nh, nf, _r, _f = gen.sample(8, T, DEV)
        x = x0
        min_sep = torch.full((8,), 1e6, dtype=torch.float64, device=DEV)
        max_slack = torch.zeros(8, dtype=torch.float64, device=DEV)
        max_perr = torch.zeros(8, dtype=torch.float64, device=DEV)
        for t in range(T):
            p0, v0 = x[:, 0:3], x[:, 3:6]
            tt = torch.arange(Hp + 1, dtype=torch.float64, device=DEV) * 0.2
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            u, aux = pol(x, nh, nf[:, :, t, :], p_ref)
            pred_next = aux["pred_abs"][:, 0, 1, :]
            true_next = nf[:, 0, min(t + 1, nf.shape[2] - 1), :]
            max_perr = torch.maximum(
                max_perr, torch.linalg.norm(pred_next - true_next, dim=-1))
            max_slack = torch.maximum(
                max_slack, aux["slack"].reshape(8, -1).max(dim=1).values)
            x = dyn.step(x, u, wind.sample(p0), 0.2)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            min_sep = torch.minimum(min_sep, d)
        coll = min_sep < DSEP
        n_coll += int(coll.sum())
        tot += 8
        slack_sum += float(max_slack.sum())
        perr_sum += float(max_perr.sum())
        coll_slack += int((coll & (max_slack > 1.0)).sum())
        coll_pred += int((coll & (max_perr > 20.0)).sum())
    line = ("%-16s CR=%.1f%% | conflicts: slack-active=%d predErr>20m=%d "
            "| avg maxslack=%.2f avg maxPredErr=%.1fm\n"
            % (ckpt, 100.0 * n_coll / tot, coll_slack, coll_pred,
               slack_sum / tot, perr_sum / tot))
    with open(OUT, "a") as f:
        f.write(line)
    print(line, end="", flush=True)


if __name__ == "__main__":
    open(OUT, "w").close()
    run("stage1_full.pt")
    run("stage2_final.pt")
