"""Read-only deep probe of how stage2_final allocates predictive variance,
vs stage1_full, on the SAME held-out GUAM encounters (n=200, seed 12345).

Goal: determine, honestly, whether a variance-reallocation mechanism exists
for the checkpoint actually used in the paper (stage2_final.pt), using
several complementary lenses instead of a single +/-3-step binary window:

  (1) variance profiled by distance-to-closest-approach buckets
  (2) NORMALISED allocation share: fraction of total predictive variance
      placed on critical steps (controls for any global scale change in
      variance between Stage-1 and Stage-2)
  (3) prediction ERROR (not just variance) on critical vs inert steps
      -- the operationally meaningful quantity is whether the predictor is
      more ACCURATE where it matters, which is what the planner consumes.

Nothing here is written to disk; it only prints. No checkpoints other than
stage1_full.pt and stage2_final.pt are touched.
"""
import sys
import torch

sys.path.insert(0, "/data/lab/plangrad/plangrad_sim")
from config import GUAM_MAT
from predictor import GMMTrajectoryPredictor
from guam_encounters import GUAMEncounters
from seeding import set_seed

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 100.0
set_seed(12345)


def load(p):
    n = GMMTrajectoryPredictor(T=30, K=5).double().to(DEV)
    n.load_state_dict(torch.load(f"/data/lab/plangrad/plangrad_sim/{p}",
                                 map_location=DEV))
    n.eval()
    return n


@torch.no_grad()
def probe(net, n=200, T=20, win=3):
    gen = GUAMEncounters(GUAM_MAT, range(2500, 3000), seed=12345)
    # accumulators
    crit_var = inert_var = 0.0
    crit_cnt = inert_cnt = 0
    crit_err = inert_err = 0.0
    tot_var = 0.0
    # distance-to-CA bucketed variance (buckets of step-distance 0..6+)
    bucket_var = torch.zeros(8)
    bucket_cnt = torch.zeros(8)
    b = 16
    for _ in range(max(1, n // b)):
        x0, nh, nf, ref, nfut = gen.sample(b, T, DEV)
        out = net(nh.reshape(b, 25, 3))
        alpha, mu = out["alpha"], out["mu"]
        var = torch.exp(2.0 * out["log_sigma"])
        tr = (alpha.unsqueeze(-1) * var).sum(2).sum(-1)          # [b,30]
        mean_pred = (alpha.unsqueeze(-1) * mu).sum(2)            # [b,30,3]
        h = min(30, nfut.shape[2])
        gt = nfut[:, 0, :h, :]
        err = torch.linalg.norm(mean_pred[:, :h] - gt, dim=-1) * SCALE  # [b,h]
        tr_h = tr[:, :h] * SCALE * SCALE

        d = torch.linalg.norm(ref[:, 1:h + 1, :] - nf[:, 0, 1:h + 1, :],
                              dim=-1)
        ca = d.argmin(dim=1)                                      # [b]
        idx = torch.arange(h, device=DEV).unsqueeze(0)
        dist_to_ca = (idx - ca.unsqueeze(1)).abs()               # [b,h]
        crit_mask = dist_to_ca <= win
        inert_mask = ~crit_mask

        crit_var += (tr_h * crit_mask).sum().item()
        inert_var += (tr_h * inert_mask).sum().item()
        crit_cnt += int(crit_mask.sum())
        inert_cnt += int(inert_mask.sum())
        crit_err += (err * crit_mask).sum().item()
        inert_err += (err * inert_mask).sum().item()
        tot_var += tr_h.sum().item()

        dca = dist_to_ca.clamp(max=7).cpu()
        for bk in range(8):
            m = (dca == bk)
            bucket_var[bk] += (tr_h.cpu() * m).sum().item()
            bucket_cnt[bk] += int(m.sum())

    cv = crit_var / max(crit_cnt, 1)
    iv = inert_var / max(inert_cnt, 1)
    ce = crit_err / max(crit_cnt, 1)
    ie = inert_err / max(inert_cnt, 1)
    share = crit_var / max(tot_var, 1e-9)
    prof = (bucket_var / bucket_cnt.clamp(min=1)).tolist()
    return {"var_crit": cv, "var_inert": iv, "ratio": cv / max(iv, 1e-9),
            "crit_share": share, "err_crit": ce, "err_inert": ie,
            "err_ratio": ce / max(ie, 1e-9), "profile": prof}


def main():
    s1 = probe(load("stage1_full.pt"))
    s2 = probe(load("stage2_final.pt"))
    print(f"{'metric':<14}{'Stage-1':>12}{'Stage-2':>12}")
    for k in ["var_crit", "var_inert", "ratio", "crit_share",
              "err_crit", "err_inert", "err_ratio"]:
        print(f"{k:<14}{s1[k]:>12.4f}{s2[k]:>12.4f}")
    print("\nvariance profile by |step - closest-approach| (bucket 0=at CA):")
    print("bucket:   " + "".join(f"{i:>9d}" for i in range(8)))
    print("Stage-1:  " + "".join(f"{v:>9.1f}" for v in s1["profile"]))
    print("Stage-2:  " + "".join(f"{v:>9.1f}" for v in s2["profile"]))
    print("\nInterpretation hooks:")
    print(f"  critical-variance SHARE: {s1['crit_share']:.3f} -> "
          f"{s2['crit_share']:.3f}  (down = concentrated away from critical)")
    print(f"  critical PRED-ERROR:     {s1['err_crit']:.2f}m -> "
          f"{s2['err_crit']:.2f}m  (down = more accurate where it matters)")
    print(f"  err_crit/err_inert:      {s1['err_ratio']:.3f} -> "
          f"{s2['err_ratio']:.3f}  (down = accuracy concentrated at critical)")


if __name__ == "__main__":
    main()
