"""Paired McNemar test for the FINAL2 (loose-planner) Stage-1 vs Stage-2
comparison.

Reproduces EXACTLY the rollout of eval_stage1_vs_stage2.evaluate (loose
config: CBF-MPC alpha=0.4, a_max=10, Hp=8, T=20, dt=0.2, eta_w=0.3,
d_sep=30, batch=8), on the identical held-out encounter stream
(range(2500,3000), seed 12345), but COLLECTS the per-episode min-sep so
that Stage-1 and Stage-2 are aligned episode-by-episode. Both models are
evaluated on the SAME encounters (set_seed(12345) + fresh generator each
time), so the pairing is exact.

Emits, for conflict = (min_sep < 30 m):
  n, marginal CR each arm, the 2x2 paired table, discordant pair counts
  b (S1-conflict & S2-safe) and c (S1-safe & S2-conflict), and the
  exact two-sided McNemar p-value (binomial on the discordant pairs).

Does NOT modify final_compare.py or FINAL2.txt. Writes MCNEMAR_FINAL2.txt.
"""
import sys
import torch
from math import comb

from seeding import set_seed
from config import GUAM_MAT
from params import DEFAULT_PARAMS
from predictor import GMMTrajectoryPredictor
from guam_encounters import GUAMEncounters
from cbf_mpc import CBFMPCLayer
from safe_policy import SafePolicy
from dynamics import EVTOLDynamics
from wind import UrbanWindField

DTYPE = torch.float64
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D_SEP = 30.0
OUT = "MCNEMAR_FINAL2.txt"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
T, Hp, dt, eta_w, batch = 20, 8, 0.2, 0.3, 8


def w(line):
    with open(OUT, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load(p):
    n = GMMTrajectoryPredictor(T=30, K=5).double().to(DEV)
    n.load_state_dict(torch.load(p, map_location=DEV))
    n.eval()
    return n


@torch.no_grad()
def per_episode_minsep(ckpt):
    """Exact copy of evaluate()'s loose-config rollout, but returns the
    per-episode min separation (metres) as a 1-D tensor of length N."""
    set_seed(12345)
    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=12345)
    predictor = load(ckpt)
    mpc = CBFMPCLayer(n_neighbors=1, horizon=Hp, dt=dt, d_sep=D_SEP,
                      alpha=0.4, a_max=10.0)
    policy = SafePolicy(predictor, mpc)
    dyn = EVTOLDynamics(DEFAULT_PARAMS, dtype=DTYPE, device=DEV)
    wind = UrbanWindField(eta_w=eta_w, dtype=DTYPE, device=DEV, seed=7)

    mins = []
    for _ in range(max(1, N // batch)):
        x0, nh, nf, _ref, _nfut = gen.sample(batch, T, DEV)
        x = x0
        min_sep = torch.full((batch,), 1e6, dtype=DTYPE, device=DEV)
        for t in range(T):
            p0 = x[:, 0:3]; v0 = x[:, 3:6]
            tt = torch.arange(Hp + 1, dtype=DTYPE, device=DEV) * dt
            p_ref = p0.unsqueeze(1) + v0.unsqueeze(1) * tt.view(1, -1, 1)
            u, _ = policy(x, nh, nf[:, :, t, :], p_ref)
            x = dyn.step(x, u, wind.sample(p0), dt)
            d = torch.linalg.norm(x[:, 0:3] - nf[:, 0, t + 1, :], dim=-1)
            min_sep = torch.minimum(min_sep, d)
        mins.append(min_sep.cpu())
    return torch.cat(mins)[:N]


def mcnemar_exact_two_sided(b, c):
    """Exact two-sided McNemar via the binomial on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial p at prob 0.5
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


if __name__ == "__main__":
    open(OUT, "w").close()
    w("MCNEMAR FINAL2 (loose planner: CBF-MPC alpha=0.4 a_max=10 Hp=8,")
    w("  T=20 dt=0.2 eta_w=0.3 d_sep=30; held-out 2500-3000, seed=12345, n=%d)" % N)

    s1 = per_episode_minsep("stage1_full.pt")
    s2 = per_episode_minsep("stage2_final.pt")
    assert len(s1) == len(s2) == N, (len(s1), len(s2), N)

    c1 = (s1 < D_SEP)  # Stage-1 conflict per episode (bool)
    c2 = (s2 < D_SEP)  # Stage-2 conflict per episode (bool)

    cr1 = 100.0 * c1.float().mean().item()
    cr2 = 100.0 * c2.float().mean().item()

    # 2x2 paired table
    a = int((c1 & c2).sum())            # both conflict
    b = int((c1 & ~c2).sum())           # S1 conflict, S2 safe  (S2 fixed it)
    c = int((~c1 & c2).sum())           # S1 safe, S2 conflict  (S2 broke it)
    d = int((~c1 & ~c2).sum())          # both safe

    p = mcnemar_exact_two_sided(b, c)

    w("")
    w("Stage-1 CR = %.1f%%   Stage-2 CR = %.1f%%   delta = %+.1f pts"
      % (cr1, cr2, cr2 - cr1))
    w("")
    w("Paired 2x2 (rows S1, cols S2):")
    w("                 S2 conflict   S2 safe")
    w("  S1 conflict        %4d        %4d" % (a, b))
    w("  S1 safe            %4d        %4d" % (c, d))
    w("")
    w("discordant: b (S1-conflict/S2-safe) = %d,  c (S1-safe/S2-conflict) = %d"
      % (b, c))
    w("McNemar exact two-sided p = %.3e" % p)
    if p < 1e-3:
        w("  -> p < 0.001: gain under the loose certificate is significant.")
    else:
        w("  -> p = %.4f" % p)
