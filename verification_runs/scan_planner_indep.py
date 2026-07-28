"""Independent-stream validation of the planner-config selection.

Same held-out pool (2500-3000) and same 6-config coarse grid as
scan_planner.py, but drawn with an INDEPENDENT seed (999) so the
configuration ranking is validated on a stream disjoint (by seed) from the
one on which it was originally observed. If the ranking reproduces, the
selection is validated rather than merely disclosed.
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

OUT = "PLANNER_SCAN_INDEP_seed999.txt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DSEP = 30.0
N = 200
CKPT = "stage2_final.pt"
VAL_SEED = 999


def load(p):
    n = GMMTrajectoryPredictor(T=30, K=5).double().to(DEV)
    n.load_state_dict(torch.load(p, map_location=DEV))
    n.eval()
    return n


@torch.no_grad()
def evaluate(pred, alpha, Hp, a_max, T=20):
    mpc = CBFMPCLayer(n_neighbors=1, horizon=Hp, dt=0.2, d_sep=DSEP,
                      alpha=alpha, a_max=a_max)
    pol = SafePolicy(pred, mpc)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=torch.float64, device=DEV)
    wind = UrbanWindField(eta_w=0.3, dtype=torch.float64, device=DEV, seed=7)
    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=VAL_SEED)
    n_coll = tot = 0
    sep_sum = 0.0
    for _ in range(N // 8):
        x0, nh, nf, _r, _f = gen.sample(8, T, DEV)
        x = x0
        min_sep = torch.full((8,), 1e6, dtype=torch.float64, device=DEV)
        for t in range(T):
            p0, v0 = x[:, 0:3], x[:, 3:6]
            tt = torch.arange(Hp + 1, dtype=torch.float64, device=DEV) * 0.2
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            try:
                u, _ = pol(x, nh, nf[:, :, t, :], p_ref)
            except Exception:
                u = torch.zeros(8, 4, dtype=torch.float64, device=DEV)
                u[:, 0] = DEFAULT_PARAMS.weight
            x = dyn.step(x, u, wind.sample(p0), 0.2)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            min_sep = torch.minimum(min_sep, d)
        n_coll += int((min_sep < DSEP).sum())
        sep_sum += float(min_sep.sum())
        tot += 8
    return 100.0 * n_coll / tot, sep_sum / tot


def main():
    set_seed(VAL_SEED)
    pred = load(CKPT)
    with open(OUT, "w") as f:
        f.write("INDEPENDENT-STREAM planner sweep on %s, held-out 2500-3000, "
                "n=%d, VAL_SEED=%d\n\n" % (CKPT, N, VAL_SEED))
        f.write("%-28s %8s %10s\n" % ("config", "CR%", "minSep"))
    configs = [
        ("alpha0.4 Hp8  amax10", 0.4, 8, 10.0),
        ("alpha0.2 Hp8  amax10", 0.2, 8, 10.0),
        ("alpha0.1 Hp8  amax10", 0.1, 8, 10.0),
        ("alpha0.2 Hp12 amax10", 0.2, 12, 10.0),
        ("alpha0.2 Hp12 amax16", 0.2, 12, 16.0),
        ("alpha0.1 Hp15 amax20", 0.1, 15, 20.0),
    ]
    for name, a, hp, am in configs:
        set_seed(VAL_SEED)
        cr, sep = evaluate(pred, a, hp, am)
        line = "%-28s %8.1f %10.1f\n" % (name, cr, sep)
        with open(OUT, "a") as f:
            f.write(line)
        print(line, end="", flush=True)


if __name__ == "__main__":
    main()
