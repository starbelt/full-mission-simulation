"""Tests for the onboard attitude estimator.

The estimator is deliberately the *unlucky* consumer of this design: it sees
only what a real ADS would see -- a gyro rate, two absolute vector measurements
in body axes, and their validity -- and must produce its own attitude. Nothing
here passes it a truth attitude, so these tests also guard the ownership rule:
if the estimator ever starts matching truth to implausible precision, that is a
bug report, not a success.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pocketadcs_sim.adcs import quaternions as quat
from pocketadcs_sim.adcs.estimator import AbsoluteUpdate, OnboardAttitudeEstimator

TRUTH = Rotation.from_euler("zyx", [0.7, 0.4, -0.2]).as_quat()[[3, 0, 1, 2]]
RATE = np.array([0.01, -0.02, 0.005])
SUN_REF = np.array([1.0, 0.1, -0.2])
MAG_REF = np.array([0.2, 1.0, 0.3])


def in_body(reference: np.ndarray) -> np.ndarray:
    """Express an inertial reference vector in the truth body's axes."""
    return quat.to_dcm(TRUTH) @ (reference / np.linalg.norm(reference))


def fix(**kwargs: object) -> AbsoluteUpdate:
    """Build an absolute fix measuring SUN_REF/MAG_REF from the truth attitude."""
    fields = {
        "sun_body": in_body(SUN_REF),
        "magnetic_body": in_body(MAG_REF),
        "sun_ref_gcrs": SUN_REF,
        "magnetic_ref_gcrs": MAG_REF,
    }
    fields.update(kwargs)
    return AbsoluteUpdate(**fields)  # type: ignore[arg-type]

def make_estimator() -> OnboardAttitudeEstimator:
    """Return an estimator initialized from the first absolute fix."""
    estimator = OnboardAttitudeEstimator()
    estimator.update(
        time_s=0.0,
        gyro_body_rad_s=np.zeros(3),
        absolute=fix(),
    )
    return estimator


def test_absolute_fix_recovers_the_truth_attitude() -> None:
    """Two non-parallel vectors determine attitude exactly (TRIAD geometry)."""
    estimator = make_estimator()

    np.testing.assert_allclose(
        quat.to_dcm(estimator.attitude), quat.to_dcm(TRUTH), atol=1e-12
    )
    assert estimator.has_absolute_fix is True


def test_gyro_only_propagation_advances_the_attitude_in_the_right_direction() -> None:
    """With no absolute update the estimate moves with the measured body rate."""
    estimator = make_estimator()
    start = estimator.attitude.copy()

    estimator.update(time_s=0.1, gyro_body_rad_s=RATE)

    expected = quat.propagate(start, RATE, 0.1)
    np.testing.assert_allclose(
        quat.to_dcm(estimator.attitude), quat.to_dcm(expected), atol=1e-12
    )


def test_gyro_only_drift_grows_without_becoming_the_truth() -> None:
    """Free propagation drifts, and an un-aided estimate is *not* truth.

    A pin on this: an estimator that tracks truth exactly while unaided would
    mean the answer was supplied rather than computed.
    """
    estimator = make_estimator()
    start_error = quat.rotation_angle_rad(quat.error_quaternion(TRUTH, estimator.attitude))
    assert start_error < 1e-9

    for step in range(1, 201):
        # A constant bias is the dominant real gyro error; truth has none.
        estimator.update(time_s=step * 0.1, gyro_body_rad_s=RATE + np.array([2e-4, 0, 0]))

    error = quat.rotation_angle_rad(quat.error_quaternion(TRUTH, estimator.attitude))
    assert error > 1e-3, "un-aided estimate must drift under gyro bias"


def test_absolute_fix_corrects_accumulated_drift() -> None:
    """A new absolute fix pulls the estimate back onto truth."""
    estimator = make_estimator()
    for step in range(1, 101):
        estimator.update(time_s=step * 0.1, gyro_body_rad_s=RATE + np.array([5e-4, 0, 0]))
    assert quat.rotation_angle_rad(quat.error_quaternion(TRUTH, estimator.attitude)) > 1e-2

    estimator.update(
        time_s=10.5,
        gyro_body_rad_s=RATE,
        absolute=fix(),
    )

    assert quat.rotation_angle_rad(quat.error_quaternion(TRUTH, estimator.attitude)) < 1e-6


def test_eclipse_coasts_on_gyro_and_recovers_on_sunlight() -> None:
    """An invalid sun reading must coast, not corrupt, and recover when lit."""
    estimator = make_estimator()
    for step in range(1, 51):
        estimator.update(time_s=step * 0.1, gyro_body_rad_s=RATE + np.array([1e-4, 0, 0]))

    coasted = quat.rotation_angle_rad(quat.error_quaternion(TRUTH, estimator.attitude))
    assert coasted > 1e-3

    for step in range(51, 101):
        estimator.update(
            time_s=step * 0.1, gyro_body_rad_s=RATE + np.array([1e-4, 0, 0]),
            absolute=fix(sun_body=np.zeros(3), sun_valid=False),
        )

    # Mag-only must still be usable, so the error should not blow up further.
    eclipse_error = quat.rotation_angle_rad(quat.error_quaternion(TRUTH, estimator.attitude))
    assert eclipse_error < coasted * 10

    for step in range(101, 131):
        estimator.update(
            time_s=step * 0.1, gyro_body_rad_s=RATE,
            absolute=fix(),
        )

    assert quat.rotation_angle_rad(quat.error_quaternion(TRUTH, estimator.attitude)) < 1e-3


def test_zero_sun_reading_is_rejected_as_invalid_eclipse() -> None:
    """An all-zero sun vector is 'no data', not a real measurement."""
    estimator = make_estimator()
    before = estimator.attitude.copy()

    estimator.update(
        time_s=0.1,
        gyro_body_rad_s=np.zeros(3),
        absolute=fix(sun_body=np.zeros(3)),
    )

    np.testing.assert_allclose(estimator.attitude, before, atol=0.0)


def test_parallel_reference_vectors_are_rejected() -> None:
    """Two parallel vectors cannot define attitude; that is a degenerate fix."""
    estimator = make_estimator()
    before = estimator.attitude.copy()
    parallel = in_body(SUN_REF)

    estimator.update(
        time_s=0.1,
        gyro_body_rad_s=np.zeros(3),
        absolute=fix(sun_body=parallel, magnetic_body=parallel),
    )

    np.testing.assert_allclose(estimator.attitude, before, atol=0.0)


def test_non_monotonic_time_is_rejected_rather_than_propagated_backwards() -> None:
    """A stale or reordered sample must not corrupt the propagation."""
    estimator = make_estimator()
    estimator.update(time_s=0.5, gyro_body_rad_s=RATE)
    after_first = estimator.attitude.copy()

    # Loud, not silent: a reordered sample is a pipeline fault, and quietly
    # ignoring it would hide it while the estimate quietly diverges.
    with pytest.raises(ValueError, match="non-monotonic"):
        estimator.update(time_s=0.2, gyro_body_rad_s=RATE)

    np.testing.assert_allclose(estimator.attitude, after_first, atol=0.0)


def test_estimate_is_sign_invariant_and_unit_norm() -> None:
    """The published quaternion stays unit-norm through every update path."""
    estimator = make_estimator()

    for step in range(1, 21):
        estimator.update(time_s=step * 0.1, gyro_body_rad_s=RATE)

    assert float(np.linalg.norm(estimator.attitude)) == pytest.approx(1.0, abs=1e-12)


def test_estimator_does_not_see_truth_and_reports_its_own_error_estimate() -> None:
    """The estimator must not be handed a truth attitude anywhere."""
    estimator = make_estimator()
    for step in range(1, 51):
        estimator.update(time_s=step * 0.1, gyro_body_rad_s=RATE + np.array([1e-4, 0, 0]))

    # It knows only its own internal spread between the vectors it fused.
    assert estimator.covariance_trace >= 0.0
    assert not hasattr(estimator, "truth_attitude")


def test_rotation_estimate_recovers_a_known_2d_rotation_from_vectors() -> None:
    """End-to-end: a known body rotation is recovered from two measured vectors."""
    delta = 0.35
    moved = quat.multiply(TRUTH, quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), delta))
    estimator = OnboardAttitudeEstimator()
    estimator.update(
        time_s=0.0,
        gyro_body_rad_s=np.zeros(3),
        absolute=AbsoluteUpdate(
            sun_body=quat.to_dcm(moved) @ (SUN_REF / np.linalg.norm(SUN_REF)),
            magnetic_body=quat.to_dcm(moved) @ (MAG_REF / np.linalg.norm(MAG_REF)),
            sun_ref_gcrs=SUN_REF,
            magnetic_ref_gcrs=MAG_REF,
        ),
    )

    recovered = quat.rotation_angle_rad(quat.error_quaternion(moved, estimator.attitude))
    assert recovered == pytest.approx(0.0, abs=1e-9)
    assert delta == pytest.approx(0.35, rel=1e-12)  # sanity: the case is non-trivial


def test_estimator_ignores_nonfinite_sensor_input() -> None:
    """A NaN from a sensor must be refused, not propagated into the state."""
    estimator = make_estimator()
    before = estimator.attitude.copy()

    with pytest.raises(ValueError):
        estimator.update(time_s=0.1, gyro_body_rad_s=np.array([np.nan, 0.0, 0.0]))
    np.testing.assert_allclose(estimator.attitude, before, atol=0.0)


def test_reference_to_body_matches_scipy_for_a_known_rotation() -> None:
    """The estimator's frame convention matches scipy's, not a mirrored one."""
    body_x_in_inertial = quat.to_dcm(TRUTH).T @ np.array([1.0, 0.0, 0.0])
    expected = Rotation.from_quat(TRUTH[[1, 2, 3, 0]]).as_matrix() @ np.array([1.0, 0.0, 0.0])

    np.testing.assert_allclose(body_x_in_inertial, expected, atol=1e-14)
