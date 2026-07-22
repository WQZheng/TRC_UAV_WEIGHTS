"""Differentiable 6-DOF eVTOL rigid-body dynamics (PyTorch).

Implements the discretized dynamics x_{t+1} = g(x_t, u_t, w_t) of the
manuscript (Section 3.1), with a fourth-order Runge-Kutta integrator.
Everything is written in torch so that gradients flow through g.

State x in R^12 (batched as [B, 12]):
    [0:3]   p  = inertial position (x, y, z)            [m]   (z up)
    [3:6]   v  = inertial velocity                      [m/s]
    [6:9]   eta = Euler attitude (roll, pitch, yaw)     [rad]
    [9:12]  omega = body angular rate (p, q, r)         [rad/s]

Control u in R^4 (batched as [B, 4]):
    [0]   collective thrust magnitude along body z      [N]
    [1:4] body-axis moments (L, M, N)                   [N m]

Wind w in R^3 (batched as [B, 3]): inertial wind velocity [m/s].

Lumped-actuator model: the 9 physical rotors are abstracted into a
single collective thrust + 3 body moments. Mass and inertia come from
the real NASA SACD configuration (see params.py).
"""
from __future__ import annotations
import torch
from params import LiftCruiseParams, DEFAULT_PARAMS


def euler_to_rotmat(eta: torch.Tensor) -> torch.Tensor:
    """Body->inertial rotation matrix from ZYX Euler angles. [B,3]->[B,3,3]."""
    phi, theta, psi = eta[:, 0], eta[:, 1], eta[:, 2]
    cphi, sphi = torch.cos(phi), torch.sin(phi)
    cth, sth = torch.cos(theta), torch.sin(theta)
    cpsi, spsi = torch.cos(psi), torch.sin(psi)

    R = torch.zeros(eta.shape[0], 3, 3, dtype=eta.dtype, device=eta.device)
    R[:, 0, 0] = cpsi * cth
    R[:, 0, 1] = cpsi * sth * sphi - spsi * cphi
    R[:, 0, 2] = cpsi * sth * cphi + spsi * sphi
    R[:, 1, 0] = spsi * cth
    R[:, 1, 1] = spsi * sth * sphi + cpsi * cphi
    R[:, 1, 2] = spsi * sth * cphi - cpsi * sphi
    R[:, 2, 0] = -sth
    R[:, 2, 1] = cth * sphi
    R[:, 2, 2] = cth * cphi
    return R


def euler_rate_matrix(eta: torch.Tensor) -> torch.Tensor:
    """Map body rates -> Euler-angle rates: eta_dot = T(eta) omega."""
    phi, theta = eta[:, 0], eta[:, 1]
    cphi, sphi = torch.cos(phi), torch.sin(phi)
    cth = torch.cos(theta)
    tth = torch.tan(theta)
    cth = torch.where(cth.abs() < 1e-4, torch.full_like(cth, 1e-4), cth)

    T = torch.zeros(eta.shape[0], 3, 3, dtype=eta.dtype, device=eta.device)
    T[:, 0, 0] = 1.0
    T[:, 0, 1] = sphi * tth
    T[:, 0, 2] = cphi * tth
    T[:, 1, 1] = cphi
    T[:, 1, 2] = -sphi
    T[:, 2, 1] = sphi / cth
    T[:, 2, 2] = cphi / cth
    return T


class EVTOLDynamics:
    """Continuous-time 6-DOF dynamics + RK4 discretization (differentiable)."""

    def __init__(self, params: LiftCruiseParams = DEFAULT_PARAMS,
                 device: str = "cpu", dtype: torch.dtype = torch.float64):
        self.p = params
        self.device = device
        self.dtype = dtype
        self.mass = torch.tensor(params.mass, dtype=dtype, device=device)
        self.I = torch.tensor(params.inertia_diag, dtype=dtype, device=device)
        self.g = torch.tensor(params.g, dtype=dtype, device=device)

    def deriv(self, x, u, w):
        """Continuous-time state derivative xdot = f(x, u, w). [B,12]."""
        v = x[:, 3:6]
        eta = x[:, 6:9]
        omega = x[:, 9:12]

        R = euler_to_rotmat(eta)
        thrust = u[:, 0]
        f_body = torch.zeros_like(v)
        f_body[:, 2] = thrust
        f_inertial = torch.bmm(R, f_body.unsqueeze(-1)).squeeze(-1)

        v_rel = v - w
        drag_coeff = 0.5 * self.p.rho * self.p.S * 0.3
        f_drag = -drag_coeff * v_rel * v_rel.abs()

        gravity = torch.zeros_like(v)
        gravity[:, 2] = -self.mass * self.g
        v_dot = (f_inertial + f_drag + gravity) / self.mass

        T = euler_rate_matrix(eta)
        eta_dot = torch.bmm(T, omega.unsqueeze(-1)).squeeze(-1)

        moments = u[:, 1:4]
        Iw = self.I.unsqueeze(0) * omega
        gyro = torch.cross(omega, Iw, dim=1)
        omega_dot = (moments - gyro) / self.I.unsqueeze(0)

        return torch.cat([v, v_dot, eta_dot, omega_dot], dim=1)

    def step(self, x, u, w, dt):
        """One RK4 step: returns x_{t+1}. Differentiable in x, u, w."""
        k1 = self.deriv(x, u, w)
        k2 = self.deriv(x + 0.5 * dt * k1, u, w)
        k3 = self.deriv(x + 0.5 * dt * k2, u, w)
        k4 = self.deriv(x + dt * k3, u, w)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
