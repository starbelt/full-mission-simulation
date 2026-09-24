"""Run a short Basilisk PocketQube truth simulation."""

import numpy as np

from pocketadcs_sim.dynamics import (
    InitialState,
    SimulationConfig,
    SpacecraftConfig,
    run_truth_simulation,
)

vehicle = SpacecraftConfig(
    mass_kg=0.25,
    inertia_kg_m2=np.diag([8.0e-4, 9.0e-4, 7.0e-4]),
)
initial = InitialState(
    position_m=np.array([6_878_137.0, 0.0, 0.0]),
    velocity_m_s=np.array([0.0, 7_612.6, 0.0]),
    mrp_body_from_inertial=np.array([0.1, -0.2, 0.05]),
    angular_velocity_body_rad_s=np.array([0.01, -0.02, 0.03]),
)
history = run_truth_simulation(
    vehicle,
    initial,
    SimulationConfig(duration_s=60.0, step_s=0.1),
)

print(f"Recorded {history.time_s.size} states")
print(f"Final position [m]: {history.position_m[-1]}")
print(f"Final velocity [m/s]: {history.velocity_m_s[-1]}")
print(f"Final MRP B/N: {history.mrp_body_from_inertial[-1]}")
print(f"Final body rate [rad/s]: {history.angular_velocity_body_rad_s[-1]}")
