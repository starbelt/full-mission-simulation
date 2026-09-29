"""Tests for the ideal sensor models and the readings layer."""

import numpy as np

from pocketadcs_sim.dynamics import InitialState, SimulationConfig, SpacecraftConfig
from pocketadcs_sim.sensors import (
    GyroscopeModel,
    MagnetometerModel,
    SunSensorModel,
    collect_readings,
    sample_history,
)
from pocketadcs_sim.truth_environment import TruthEnvironmentConfig, run_truth_environment


def _short_truth():
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
    return run_truth_environment(
        vehicle,
        initial,
        SimulationConfig(duration_s=2.0, step_s=0.5),
        TruthEnvironmentConfig(epoch_utc="2025-01-01T00:00:00", gravity_degree=4),
    )


def test_measure_matches_truth_at_sample_indices():
    truth = _short_truth()
    gyro = GyroscopeModel(sample_rate_hz=10.0)
    indices = sample_history(truth, gyro.sample_rate_hz)
    sampled = gyro.measure(truth)
    assert sampled.shape[1] == 3
    assert sampled.shape[0] == len(indices)
    assert np.allclose(sampled, truth.angular_velocity_body_rad_s[indices])

def test_all_sensors_sample_and_report_three_axes():
    truth = _short_truth()
    sensors = {
        "gyro": GyroscopeModel(sample_rate_hz=20.0),
        "magnetometer": MagnetometerModel(sample_rate_hz=10.0),
        "sun": SunSensorModel(sample_rate_hz=5.0),
    }
    readings = collect_readings(truth, sensors)
    for name, expected in [
        ("gyro", truth.angular_velocity_body_rad_s),
        ("magnetometer", truth.magnetic_field_body_t),
        ("sun", truth.sun_direction_body),
    ]:
        reading = readings[name]
        assert reading.values.shape[1] == 3
        assert len(reading.values) == len(sample_history(truth, reading.sample_rate_hz))
        assert np.allclose(
            reading.values,
            expected[sample_history(truth, reading.sample_rate_hz)],
        )

def test_sun_direction_is_unit_vector_and_eclipse_is_separate():
    truth = _short_truth()
    sun = SunSensorModel(sample_rate_hz=10.0).measure(truth)
    assert np.allclose(np.linalg.norm(sun, axis=1), 1.0)
    assert truth.illumination_fraction.min() >= 0.0

def test_zero_rate_rejected():
    try:
        GyroscopeModel(sample_rate_hz=0.0)
    except ValueError:
        return
    raise AssertionError("zero sample rate should raise")
