import numpy as np

from pocketadcs_sim.dynamics import (
    InitialState,
    SimulationConfig,
    SpacecraftConfig,
    run_truth_simulation,
)


def test_basilisk_truth_simulation_exposes_six_dof_history() -> None:
    spacecraft = SpacecraftConfig(
        mass_kg=0.25,
        inertia_kg_m2=np.diag([8.0e-4, 8.0e-4, 8.0e-4]),
    )
    initial = InitialState(
        position_m=np.array([6_878_137.0, 0.0, 0.0]),
        velocity_m_s=np.array([0.0, 7_612.6, 0.0]),
        mrp_body_from_inertial=np.array([0.1, -0.2, 0.05]),
        angular_velocity_body_rad_s=np.array([0.01, -0.02, 0.03]),
    )

    history = run_truth_simulation(
        spacecraft,
        initial,
        SimulationConfig(duration_s=2.0, step_s=0.1),
    )

    assert history.time_s.shape == (21,)
    assert history.position_m.shape == (21, 3)
    assert history.velocity_m_s.shape == (21, 3)
    assert history.mrp_body_from_inertial.shape == (21, 3)
    assert history.angular_velocity_body_rad_s.shape == (21, 3)
    np.testing.assert_allclose(history.position_m[0], initial.position_m)
    np.testing.assert_allclose(
        history.angular_velocity_body_rad_s,
        np.broadcast_to(initial.angular_velocity_body_rad_s, (21, 3)),
        atol=1e-12,
    )
    assert history.position_m[-1, 1] > 15_000.0
    assert np.isfinite(history.mrp_body_from_inertial).all()
