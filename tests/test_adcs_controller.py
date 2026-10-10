"""Tests for the quaternion attitude controller.

The controller's contract is narrow and worth stating: it sees the *estimated*
attitude and *estimated* rate, produces a body-frame torque, and never sees
truth. Saturation and the reaction-wheel allocation boundary are pinned here
because both are places where a mirrored sign still "works" in a self-consistent
test.
"""

from __future__ import annotations

import numpy as np
import pytest

from pocketadcs_sim.adcs import quaternions as quat
from pocketadcs_sim.adcs.controller import AttitudeController


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return quat.rotation_angle_rad(quat.error_quaternion(b, a)) * 180.0 / np.pi


def test_zero_error_produces_zero_torque() -> None:
    """On the commanded attitude with zero rate, the controller does nothing."""
    controller = AttitudeController()

    torque = controller.compute(
        estimated_attitude=[1.0, 0.0, 0.0, 0.0],
        estimated_rate_rad_s=np.zeros(3),
        desired_attitude=[1.0, 0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(torque, np.zeros(3), atol=1e-18)


def test_attitude_error_drives_torque_in_the_reducing_direction() -> None:
    """A +Z error must produce a +Z torque to rotate the body toward the target.

    This is the sign that a mirrored controller gets wrong: it still settles,
    just the wrong way, so only a test that names the direction can catch it.
    """
    controller = AttitudeController()
    desired = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.2)
    actual = np.array([1.0, 0.0, 0.0, 0.0])  # 0.2 rad short about +Z

    torque = controller.compute(
        estimated_attitude=actual,
        estimated_rate_rad_s=np.zeros(3),
        desired_attitude=desired,
    )

    assert torque[2] > 0.0
    assert abs(torque[0]) < 1e-12 and abs(torque[1]) < 1e-12


def test_rate_error_damps_the_body_rate() -> None:
    """Excess body rate is opposed by the controller."""
    controller = AttitudeController()

    torque = controller.compute(
        estimated_attitude=[1.0, 0.0, 0.0, 0.0],
        estimated_rate_rad_s=np.array([0.0, 0.0, 0.05]),
        desired_attitude=[1.0, 0.0, 0.0, 0.0],
    )

    assert torque[2] < 0.0


def test_controller_produces_no_momentum_exchange_when_tracking() -> None:
    """A settled vehicle commanded to hold attitude applies no sustained torque."""
    controller = AttitudeController()
    attitude = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.3)

    torque = controller.compute(
        estimated_attitude=attitude,
        estimated_rate_rad_s=np.zeros(3),
        desired_attitude=attitude,
    )

    np.testing.assert_allclose(torque, np.zeros(3), atol=1e-15)


def test_torque_is_clipped_to_the_momentum_wheel_envelope() -> None:
    """Three wheels cannot null a torque purely along their common axis.

    This is the allocation boundary: ``tau = [0,0,T]`` needs a reaction
    torque of zero, which a perfect triad must produce exactly.
    """
    controller = AttitudeController()
    axes = np.eye(3)
    torque = controller.compute(
        estimated_attitude=[1.0, 0.0, 0.0, 0.0],
        estimated_rate_rad_s=np.array([0.0, 0.0, 1.0]),
        desired_attitude=[1.0, 0.0, 0.0, 0.0],
        wheel_axes=axes,
    )

    assert np.all(np.isfinite(torque))
    assert np.max(np.abs(torque)) <= controller.max_torque_nm


def test_large_error_does_not_produce_an_unbounded_torque() -> None:
    """A 180 deg error must stay inside the torque limit."""
    controller = AttitudeController()
    actual = [1.0, 0.0, 0.0, 0.0]
    desired = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi)

    torque = controller.compute(
        estimated_attitude=actual,
        estimated_rate_rad_s=np.zeros(3),
        desired_attitude=desired,
    )

    assert np.max(np.abs(torque)) <= controller.max_torque_nm


def test_gains_are_exposed_for_results_and_are_positive() -> None:
    """The gains must appear in the output record; a hidden gain is unreproducible."""
    controller = AttitudeController()
    record = controller.to_dict()

    assert record["proportional_gain_rad_per_nm"] > 0.0
    assert record["derivative_gain_nm_s_per_rad"] > 0.0
    assert record["proportional_gain_rad_per_nm"] == controller.proportional_gain


def test_invalid_inputs_are_rejected() -> None:
    """Bad state must raise rather than propagate into a torque command."""
    controller = AttitudeController()

    with pytest.raises(ValueError):
        controller.compute(
            estimated_attitude=np.array([1.0, 0.0, 0.0]),
            estimated_rate_rad_s=np.zeros(3),
            desired_attitude=[1.0, 0.0, 0.0, 0.0],
        )
    with pytest.raises(ValueError):
        controller.compute(
            estimated_attitude=[1.0, 0.0, 0.0, 0.0],
            estimated_rate_rad_s=np.array([np.nan, 0.0, 0.0]),
            desired_attitude=[1.0, 0.0, 0.0, 0.0],
        )


def test_feedback_gain_sign_makes_the_closed_loop_stable() -> None:
    """A single-axis case must converge, which pins the loop sign end-to-end.

    Integrates the plant against the controller with no measurement noise:
    if the feedback sign were mirrored the body would run away instead.
    """
    from pocketadcs_sim.adcs.plant import PlantConfig, TruthPlant

    controller = AttitudeController()
    desired = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.3)
    plant = TruthPlant(
        config=PlantConfig(hub_inertia_kg_m2=(0.02, 0.02, 0.02)),
        initial_attitude=[1.0, 0.0, 0.0, 0.0],
    )
    estimate = np.array([1.0, 0.0, 0.0, 0.0])

    for _ in range(20000):
        body_torque = controller.compute(
            estimated_attitude=estimate,
            estimated_rate_rad_s=plant.body_rate,
            desired_attitude=desired,
        )
        # The plant integrates wheel torque; spinning rotor i up applies -u_i to
        # the hub, so a desired body torque t is commanded as u = -t.
        plant.step(0.001, -body_torque)
        estimate = quat.propagate(estimate, plant.body_rate, 0.001)

    error = quat.rotation_angle_rad(quat.error_quaternion(plant.attitude, desired))
    assert error < 1e-3, f"closed loop did not converge, error {error} rad"


def test_controller_records_the_attitude_it_was_actually_handed() -> None:
    """The controller must retain the argument it received, not a copy of intent.

    The closed loop's truth cross-check reads this back, so it has to be the
    value that actually arrived rather than whatever the caller believes it
    sent. A controller that recomputed or defaulted the attitude would make the
    cross-check report a number that was never used to compute a torque, which
    is the same defect as a hardcoded flag: a measurement that cannot detect the
    thing it claims to measure.
    """
    controller = AttitudeController(proportional_gain=0.35, derivative_gain=0.15)

    sent = quat.from_axis_angle(np.array([0.3, 0.8, 0.5]), 0.7)
    controller.compute(
        estimated_attitude=sent,
        estimated_rate_rad_s=np.zeros(3),
        desired_attitude=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    assert controller.last_estimated_attitude is not None
    recorded = np.asarray(controller.last_estimated_attitude, dtype=float)
    assert _angle_deg(np.asarray(sent, dtype=float), recorded) == pytest.approx(
        0.0, abs=1e-15
    )

    # A second call overwrites it: it is "what this call was given", not a
    # cumulative log that would silently report a stale value. Compare the live
    # attribute -- `recorded` is a copy taken before the second call, so
    # comparing *it* to `sent` is zero by construction and asserts nothing.
    other = quat.from_axis_angle(np.array([1.0, 0.0, 0.0]), -0.4)
    controller.compute(
        estimated_attitude=other,
        estimated_rate_rad_s=np.zeros(3),
        desired_attitude=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    live = np.asarray(controller.last_estimated_attitude, dtype=float)
    assert _angle_deg(np.asarray(other, dtype=float), live) == pytest.approx(
        0.0, abs=1e-15
    )
    assert _angle_deg(np.asarray(sent, dtype=float), live) > 1.0


def test_recorded_attitude_is_not_writable_through_the_public_attribute() -> None:
    """The record must be a copy: mutating the caller's array cannot rewrite it.

    ``compute`` receives a numpy array by reference. If the recorded value were
    the same object, a caller that reuses one buffer across steps would rewrite
    the record after the fact and the cross-check would score a value that was
    never used.
    """
    controller = AttitudeController()
    handed = quat.from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.9)

    controller.compute(
        estimated_attitude=handed,
        estimated_rate_rad_s=np.zeros(3),
        desired_attitude=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    recorded = np.array(controller.last_estimated_attitude, dtype=float, copy=True)

    handed[0] = -handed[0]
    handed[1:] *= -1.0

    assert np.allclose(
        np.asarray(controller.last_estimated_attitude, dtype=float), recorded
    )
