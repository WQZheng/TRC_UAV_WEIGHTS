"""End-to-end safe policy: predictor -> CBF-MPC -> 6-DOF control.

Wires the three building blocks into a single differentiable policy that
the Stage-2 task-aligned loss back-propagates through (Section 4, Stage 2):
  1. f_theta predicts neighbour future positions (alpha-weighted mean
     fed to the planner; variance available for margin inflation).
  2. CBFMPCLayer computes a commanded acceleration that tracks the ego
     reference while satisfying CBF separation.
  3. The acceleration is mapped to a 6-DOF control (collective thrust +
     body moments) by a clamped feedback-linearising inner law.
Every stage is differentiable, so gradients of an operational loss on
the realised trajectory flow all the way back into theta.
"""
from __future__ import annotations
import torch

from predictor import GMMTrajectoryPredictor
from cbf_mpc import CBFMPCLayer
from params import DEFAULT_PARAMS


class SafePolicy:
    def __init__(self, predictor: GMMTrajectoryPredictor,
                 cbf_layer: CBFMPCLayer, params=DEFAULT_PARAMS,
                 pos_scale: float = 100.0, kp: float = 2.0, kd: float = 1.5):
        self.f = predictor
        self.mpc = cbf_layer
        self.p = params
        self.pos_scale = pos_scale
        self.kp = kp
        self.kd = kd
        self.Hp = cbf_layer.Hp

    def predict_neighbours(self, neigh_hist, last_pos):
        """neigh_hist [B,N,L,3], last_pos [B,N,3] -> pred_abs [B,N,Hp+1,3],
        var_mean [B,N]."""
        B, N, L, _ = neigh_hist.shape
        out = self.f(neigh_hist.reshape(B * N, L, 3))
        alpha = out["alpha"]
        mu = out["mu"]
        var = torch.exp(2.0 * out["log_sigma"])
        mean_traj = (alpha.unsqueeze(-1) * mu).sum(2)
        var_tr = (alpha.unsqueeze(-1) * var).sum(2).sum(-1).mean(-1)

        T = mean_traj.shape[1]
        Hp = self.Hp
        horizon = min(Hp, T)
        traj = mean_traj[:, :horizon, :] * self.pos_scale
        traj = traj.reshape(B, N, horizon, 3)
        cur = last_pos.unsqueeze(2)
        pred_abs = cur + traj
        if horizon < Hp + 1:
            pad = pred_abs[:, :, -1:, :].expand(B, N, Hp + 1 - horizon, 3)
            pred_abs = torch.cat([pred_abs, pad], dim=2)
        return pred_abs, var_tr.reshape(B, N)

    def accel_to_control(self, x, a_cmd):
        """Map commanded inertial acceleration -> (thrust, moments) [B,4].
        Tilt targets and moments clamped to the physical envelope."""
        m = self.p.mass
        g = self.p.g
        f_des = m * a_cmd.clone()
        f_des[:, 2] = f_des[:, 2] + m * g
        thrust = torch.linalg.norm(f_des, dim=-1)

        fmag = thrust.clamp_min(1.0)
        ax = f_des[:, 0] / fmag
        ay = f_des[:, 1] / fmag
        tilt_max = 0.45
        roll_des = torch.clamp(-ay, -tilt_max, tilt_max)
        pitch_des = torch.clamp(ax, -tilt_max, tilt_max)
        eta = x[:, 6:9]
        omega = x[:, 9:12]
        ang_acc_r = self.kp * (roll_des - eta[:, 0]) - self.kd * omega[:, 0]
        ang_acc_p = self.kp * (pitch_des - eta[:, 1]) - self.kd * omega[:, 1]
        ang_acc_y = -self.kd * omega[:, 2]
        I = torch.tensor(self.p.inertia_diag, dtype=x.dtype, device=x.device)
        moments = torch.stack([ang_acc_r * I[0], ang_acc_p * I[1],
                               ang_acc_y * I[2]], dim=-1)
        mmax = self.p.max_body_moment
        moments = torch.clamp(moments, -mmax, mmax)
        return torch.cat([thrust.unsqueeze(-1), moments], dim=-1)

    def __call__(self, x, neigh_hist, neigh_last, p_ref):
        p0 = x[:, 0:3]
        v0 = x[:, 3:6]
        pred_abs, var_mean = self.predict_neighbours(neigh_hist, neigh_last)
        a0, info = self.mpc.solve(p0, v0, p_ref, pred_abs)
        u = self.accel_to_control(x, a0)
        return u, {"a0": a0, "pred_abs": pred_abs, "var_mean": var_mean,
                   "slack": info["eps"]}
