"""Tests for the TRIAD attitude estimator (reference impl: ahrs package)."""

import numpy as np

from pocketadcs_sim.attitude import TriadEstimator
from pocketadcs_sim.attitude.triad import _angle_error_deg
from pocketadcs_sim.dynamics import InitialState, SimulationConfig, SpacecraftConfig
from pocketadcs_sim.truth_environment import (
    TruthEnvironmentConfig,
    run_truth_environment,
)


def _truth():
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
        SimulationConfig(duration_s=5.0, step_s=0.1),
        TruthEnvironmentConfig(epoch_utc="2025-01-01T00:00:00", gravity_degree=4),
    )


def test_triad_recovers_truth_attitude():
    truth = _truth()
    result = TriadEstimator(sample_rate_hz=1.0).estimate(truth)
    assert len(result["time_s"]) >= 3
    assert result["error_deg"].max() < 1e-3


def test_triad_dcm_is_proper_rotation():
    truth = _truth()
    result = TriadEstimator(sample_rate_hz=1.0).estimate(truth)
    dcm = result["dcm"][0]
    assert np.abs(dcm.T @ dcm - np.eye(3)).max() < 1e-12
    assert np.linalg.det(dcm) > 0


def test_angle_error_deg_identity_is_zero():
    identity = np.eye(3)
    assert _angle_error_deg(identity, identity) == 0.0


def test_degenerate_parallel_vectors_flagged():
    # TRIAD's reference triad is undefined when the two vectors are parallel;
    # the reference implementation (ahrs) emits a zero cross-product triad.
    import warnings

    from ahrs.filters import TRIAD
    triad = TRIAD()
    triad.v1 = np.array([1.0, 0.0, 0.0])
    triad.v2 = np.array([1.0, 0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            a = triad.estimate(w1=np.array([0.0, 1.0, 0.0]), w2=np.array([1.0, 0.0, 0.0]))
        except (ZeroDivisionError, ValueError):
            return
    assert not np.all(np.isfinite(a))
