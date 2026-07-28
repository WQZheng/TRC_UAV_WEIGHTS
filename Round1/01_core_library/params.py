"""Lift+Cruise eVTOL physical parameters.

All values are taken from the publicly released NASA SACD Lift+Cruise
reference configuration as encoded in the open-source GUAM simulation
(vehicles/Lift+Cruise/setup/LpC_model_parameters.m). The original GUAM
values are in US customary units (slug, ft); we convert to SI here and
keep the raw imperial values in comments for traceability.

References:
  Acheson, Cook, Simmons et al., "Generic Urban Air Mobility (GUAM)
  Simulation v1.1", NASA LaRC, 2024.
  NASA SACD UAM reference configurations: https://sacd.larc.nasa.gov/uam-refs/
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# --- unit conversions ---
SLUG_TO_KG = 14.593902937
FT_TO_M = 0.3048
SLUGFT2_TO_KGM2 = SLUG_TO_KG * FT_TO_M * FT_TO_M  # = 1.355817948
G0 = 9.80665  # m/s^2


@dataclass
class LiftCruiseParams:
    """Mass, inertia, and geometry of the NASA Lift+Cruise eVTOL (SI)."""

    # --- mass [kg] ---  GUAM: Model.mass = 181.789249 slug
    mass: float = 181.789249 * SLUG_TO_KG  # ~ 2652.9 kg

    # --- diagonal inertia [kg m^2] ---
    # GUAM: Model.I = diag([13051.74318, 16660.75897, 24735.13582]) slug-ft^2
    Ixx: float = 13051.74318 * SLUGFT2_TO_KGM2
    Iyy: float = 16660.75897 * SLUGFT2_TO_KGM2
    Izz: float = 24735.13582 * SLUGFT2_TO_KGM2

    # --- aerodynamic reference geometry ---
    S: float = 186.0 * FT_TO_M * FT_TO_M      # wing reference area [m^2]
    cbar: float = 3.18 * FT_TO_M              # mean aerodynamic chord [m]
    b: float = 47.5 * FT_TO_M                 # wingspan [m]

    n_engines: int = 9                        # 8 lift + 1 pusher

    rho: float = 1.225          # air density at low altitude [kg/m^3]
    g: float = G0               # gravity [m/s^2]

    # actuator envelope (for the lumped control model)
    max_thrust_ratio: float = 2.0
    max_body_moment: float = 4.0e4            # N m, per-axis bound
    max_tilt_rate: float = np.deg2rad(60.0)   # rad/s

    @property
    def inertia_diag(self) -> np.ndarray:
        return np.array([self.Ixx, self.Iyy, self.Izz], dtype=np.float64)

    @property
    def weight(self) -> float:
        return self.mass * self.g

    def summary(self) -> str:
        return (
            f"Lift+Cruise (SI): mass={self.mass:.1f} kg, "
            f"I=diag({self.Ixx:.0f},{self.Iyy:.0f},{self.Izz:.0f}) kg m^2, "
            f"S={self.S:.2f} m^2, b={self.b:.2f} m, cbar={self.cbar:.2f} m, "
            f"weight={self.weight:.0f} N, n_engines={self.n_engines}"
        )


DEFAULT_PARAMS = LiftCruiseParams()

if __name__ == "__main__":
    print(DEFAULT_PARAMS.summary())
