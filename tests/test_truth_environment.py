import numpy as np

from pocketadcs_sim.dynamics import InitialState, SimulationConfig, SpacecraftConfig
from pocketadcs_sim.truth_environment import TruthEnvironmentConfig, run_truth_environment


def test_truth_environment_records_orbit_attitude_sun_eclipse_and_magnetic_field() -> None:
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

    history = run_truth_environment(
        vehicle,
        initial,
        SimulationConfig(duration_s=1.0, step_s=0.5),
        TruthEnvironmentConfig(epoch_utc="2025-01-01T00:00:00", gravity_degree=4),
    )

    assert history.time_s.shape == (3,)
    assert history.position_gcrs_m.shape == (3, 3)
    assert history.position_itrs_m.shape == (3, 3)
    assert history.sun_position_gcrs_m.shape == (3, 3)
    assert history.sun_direction_gcrs.shape == (3, 3)
    assert history.sun_direction_body.shape == (3, 3)
    assert history.magnetic_field_itrs_t.shape == (3, 3)
    assert history.magnetic_field_gcrs_t.shape == (3, 3)
    assert history.magnetic_field_body_t.shape == (3, 3)
    assert history.gravity_gradient_torque_body_nm.shape == (3, 3)
    assert np.all((history.illumination_fraction >= 0.0) & (history.illumination_fraction <= 1.0))
    assert np.all(np.linalg.norm(history.magnetic_field_itrs_t, axis=1) > 1.0e-6)
    assert np.all(np.linalg.norm(history.magnetic_field_itrs_t, axis=1) < 1.0e-3)
    np.testing.assert_allclose(
        np.linalg.norm(history.magnetic_field_itrs_t, axis=1),
        np.linalg.norm(history.magnetic_field_gcrs_t, axis=1),
        rtol=1e-10,
    )
    np.testing.assert_allclose(np.linalg.norm(history.sun_direction_gcrs, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(history.sun_direction_body, axis=1), 1.0)
    np.testing.assert_allclose(
        np.linalg.norm(history.magnetic_field_gcrs_t, axis=1),
        np.linalg.norm(history.magnetic_field_body_t, axis=1),
        rtol=1e-12,
    )
