"""Run the integrated PocketQube truth environment."""

import numpy as np

from pocketadcs_sim.dynamics import InitialState, SimulationConfig, SpacecraftConfig
from pocketadcs_sim.truth_environment import TruthEnvironmentConfig, run_truth_environment

history = run_truth_environment(
    SpacecraftConfig(
        mass_kg=0.25,
        inertia_kg_m2=np.diag([8.0e-4, 9.0e-4, 7.0e-4]),
    ),
    InitialState(
        position_m=np.array([6_878_137.0, 0.0, 0.0]),
        velocity_m_s=np.array([0.0, 7_612.6, 0.0]),
        mrp_body_from_inertial=np.array([0.1, -0.2, 0.05]),
        angular_velocity_body_rad_s=np.array([0.01, -0.02, 0.03]),
    ),
    SimulationConfig(duration_s=10.0, step_s=1.0),
    TruthEnvironmentConfig(
        epoch_utc="2025-01-01T00:00:00",
        gravity_degree=20,
    ),
)

print(f"Recorded {history.time_s.size} truth samples")
print(f"Final geodetic position [deg, deg, m]: "
      f"{history.geodetic_latitude_deg[-1]:.6f}, "
      f"{history.geodetic_longitude_deg[-1]:.6f}, "
      f"{history.geodetic_height_m[-1]:.1f}")
print(f"Final illumination fraction: {history.illumination_fraction[-1]:.6f}")
print(f"Final body-frame Sun direction: {history.sun_direction_body[-1]}")
print(f"Final body-frame magnetic field [T]: {history.magnetic_field_body_t[-1]}")
print(
    "Final gravity-gradient torque [N m]: "
    f"{history.gravity_gradient_torque_body_nm[-1]}"
)
