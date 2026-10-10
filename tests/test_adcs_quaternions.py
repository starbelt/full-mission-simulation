"""Behaviour tests for the closed-loop ADCS quaternion module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pocketadcs_sim.adcs import quaternions as q

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def test_known_axis_propagation_matches_closed_form_rotation() -> None:
    """A constant body rate about +Z for 1 rad yields exactly the 1 rad rotation."""
    rate = np.array([0.0, 0.0, 1.0])

    result = q.propagate(np.array([1.0, 0.0, 0.0, 0.0]), rate, 1.0)

    np.testing.assert_allclose(result, q.from_axis_angle(Z, 1.0), atol=1e-15)


def test_propagation_is_exact_so_splitting_one_rate_changes_nothing() -> None:
    """Splitting a constant rate into segments equals one segment of the summed time.

    This is the defining property of the exact (exponential) solution; a
    first-order truncation leaves an O(dt^2) seam at every split.
    """
    rate = np.array([0.1, -0.2, 0.3])
    start = q.from_axis_angle(np.array([1.0, 2.0, -0.5]), 0.2)

    single = q.propagate(start, rate, 1.0)
    split = q.propagate(q.propagate(q.propagate(start, rate, 0.25), rate, 0.25), rate, 0.5)

    np.testing.assert_allclose(split, single, atol=1e-14)


def test_propagation_composes_correctly_from_a_nonzero_attitude() -> None:
    """A body rate steps the attitude in body axes, wherever you start.

    Starting from the identity cannot distinguish right- from left-multiplication,
    so this starts from a tilted attitude where the two differ. For a body rate
    the exact result is ``C_new = R_axis(omega dt) C_old`` -- the increment is
    expressed in the *current* body axes, so it left-multiplies the DCM.
    """
    from scipy.spatial.transform import Rotation

    attitude = Rotation.from_euler("zyx", [0.6, 0.4, 0.9]).as_quat()[[3, 0, 1, 2]]
    rate = np.array([0.3, -0.5, 0.2])
    dt = 0.25

    propagated = q.propagate(attitude, rate, dt)

    # C_BN maps inertial to body components, so a body rotation by +theta about
    # +Z gives C = Rz(-theta): C_dot = -[omega]x C, i.e. C(t) = R(-omega dt) C0.
    expected = Rotation.from_rotvec(-rate * dt).as_matrix() @ q.to_dcm(attitude)
    np.testing.assert_allclose(q.to_dcm(propagated), expected, atol=1e-13)


def test_propagation_direction_is_body_rate_not_inertial_rate() -> None:
    """A +body-Z rate must tilt the body +X axis toward inertial +Y, not -Y.

    Constructed from the definition of a body rate rather than from the
    implementation: after rotating the body about +Z by angle ``t``, a body
    vector that pointed along inertial +X now has inertial components
    ``(cos t, sin t, 0)``, so ``C_BN @ e_x == (cos t, sin t, 0)``. Mirroring the
    sign is the classic way to make a closed loop diverge, and it is invisible
    in any self-consistent round-trip test.
    """
    rate = np.array([0.0, 0.0, 1.0])
    angle = 0.9

    attitude = q.propagate(np.array([1.0, 0.0, 0.0, 0.0]), rate, angle)

    body_x_in_inertial = q.to_dcm(attitude).T @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        body_x_in_inertial, [np.cos(angle), np.sin(angle), 0.0], atol=1e-14
    )


def test_propagation_with_zero_rate_holds_attitude() -> None:
    """A zero body rate leaves the attitude untouched."""
    start = q.from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.4)

    np.testing.assert_allclose(q.propagate(start, np.zeros(3), 0.5), start, atol=1e-16)


def test_noncommuting_rate_changes_depend_on_order() -> None:
    """Rate order matters: propagating +X then +Y differs from +Y then +X."""
    start = np.array([1.0, 0.0, 0.0, 0.0])

    xy = q.propagate(
        q.propagate(start, np.array([1.0, 0.0, 0.0]), 1.0), np.array([0.0, 1.0, 0.0]), 1.0
    )
    yx = q.propagate(
        q.propagate(start, np.array([0.0, 1.0, 0.0]), 1.0), np.array([1.0, 0.0, 0.0]), 1.0
    )

    angle_between = 2.0 * np.arccos(abs(float(xy @ yx)))
    assert angle_between > 0.1
    # C_BN is the transpose of the active rotation matrix scipy builds.
    np.testing.assert_allclose(
        q.to_dcm(xy), Rotation.from_quat(xy[[1, 2, 3, 0]]).as_matrix().T, atol=1e-12
    )


def test_quaternion_is_sign_invariant_everywhere() -> None:
    """q and -q give identical DCMs, MRPs, error quaternions and angles."""
    attitude = q.from_axis_angle(np.array([1.0, 2.0, -0.5]), 0.7)
    negated = -attitude
    rate = np.array([0.01, -0.02, 0.03])

    np.testing.assert_allclose(q.to_dcm(attitude), q.to_dcm(negated), atol=0.0)
    np.testing.assert_allclose(q.to_mrp(attitude), q.to_mrp(negated), atol=0.0)
    np.testing.assert_allclose(
        q.propagate(attitude, rate, 0.1), -q.propagate(negated, rate, 0.1), atol=0.0
    )
    np.testing.assert_allclose(
        q.error_quaternion(attitude, q.from_axis_angle(rate, 0.05)),
        q.error_quaternion(negated, -q.from_axis_angle(rate, 0.05)),
        atol=0.0,
    )
    assert q.rotation_angle_rad(attitude) == q.rotation_angle_rad(negated)


def test_dcm_round_trip_through_scipy_mrp_convention() -> None:
    """to_dcm reproduces the repository convention Rotation.from_mrp(sigma).inv()."""
    sigma = np.array([0.3, -0.4, 0.2])
    attitude = q.from_mrp(sigma)

    expected = Rotation.from_mrp(sigma).inv().as_matrix()
    np.testing.assert_allclose(q.to_dcm(attitude), expected, atol=1e-14)
    np.testing.assert_allclose(
        q.from_dcm(Rotation.from_mrp(sigma).inv().as_matrix()), attitude, atol=1e-12
    )


def test_dcm_round_trip_covers_large_angles() -> None:
    """from_dcm/to_dcm round-trips every rotation angle, 180 deg included.

    ``trace == -1`` (a 180 deg rotation) is the worst-conditioned case in the
    matrix-to-quaternion map -- every trace-based formula has a zero divisor
    there -- so it gets its own explicit tolerance rather than inheriting the
    1e-9 used elsewhere. 1e-5 rad is still ~2e-4 degrees.
    """
    axis = np.array([0.3, -0.5, 0.81])
    for angle in (0.0, 0.5, np.pi / 2, 2.0, np.pi - 1e-6, np.pi):
        attitude = q.from_axis_angle(axis, angle)
        dcm = q.to_dcm(attitude)
        tolerance = 1e-5 if angle > np.pi - 1e-3 else 1e-9

        np.testing.assert_allclose(q.to_dcm(q.from_dcm(dcm)), dcm, atol=tolerance)
        assert q.rotation_angle_rad(q.from_dcm(dcm)) == pytest.approx(angle, abs=tolerance)


def test_mrp_round_trip_including_antipodal_rotation() -> None:
    """to_mrp/from_mrp round-trips a 180 deg rotation, where the map is singular."""
    attitude = q.from_axis_angle(Z, np.pi)

    # |sigma| == 1 sits exactly on the singularity, so compare rotations rather
    # than quaternion values (which are only defined up to sign).
    np.testing.assert_allclose(
        q.to_dcm(q.from_mrp(q.to_mrp(attitude))), q.to_dcm(attitude), atol=1e-9
    )
    np.testing.assert_allclose(q.to_mrp(attitude), q.to_mrp(-attitude), atol=1e-12)


def test_to_mrp_agrees_with_scipy_active_convention() -> None:
    """to_mrp matches scipy's as_mrp for the same rotation, sign-invariant."""
    for attitude in (
        q.from_axis_angle(Z, 0.0),
        q.from_axis_angle(np.array([0.3, -0.5, 0.81]), 0.9),
        q.from_axis_angle(np.array([1.0, 0.0, 0.0]), 3.0),
    ):
        np.testing.assert_allclose(
            q.to_mrp(attitude),
            Rotation.from_quat(attitude[[1, 2, 3, 0]]).as_mrp(),
            atol=1e-14,
        )


def test_error_quaternion_reports_shortest_rotation() -> None:
    """The error vector points along the rotation still needed, with |angle| <= 180."""
    desired = q.from_axis_angle(Y, 0.3)
    actual = q.from_axis_angle(Y, 0.3 - 0.1)

    error = q.error_quaternion(desired, actual)

    assert error[0] > 0.0
    np.testing.assert_allclose(q.to_dcm(error), q.to_dcm(q.from_axis_angle(Y, 0.1)), atol=1e-12)
    assert q.rotation_angle_rad(error) == pytest.approx(0.1, abs=1e-12)


def test_error_quaternion_of_a_far_attitude_stays_short() -> None:
    """A 350 deg error is reported as the 10 deg rotation the other way."""
    desired = q.from_axis_angle(Z, 0.0)
    actual = q.from_axis_angle(Z, np.radians(350.0))

    assert q.rotation_angle_rad(q.error_quaternion(desired, actual)) == pytest.approx(
        np.radians(10.0), abs=1e-9
    )


def test_multiply_composes_rotations_in_application_order() -> None:
    """left (x) right applies right first, matching matrix product C(left) @ C(right)."""
    a = q.from_axis_angle(X, 0.4)
    b = q.from_axis_angle(Z, -0.9)

    # C_BN is the transpose of the active rotation matrix, so the composition
    # order of to_dcm reverses relative to the quaternion product.
    np.testing.assert_allclose(
        q.to_dcm(q.multiply(a, b)), q.to_dcm(b) @ q.to_dcm(a), atol=1e-14
    )
    np.testing.assert_allclose(
        q.to_dcm(q.multiply(a, b)),
        Rotation.from_matrix(q.to_dcm(b) @ q.to_dcm(a)).as_matrix(),
        atol=1e-14,
    )
    np.testing.assert_allclose(q.multiply(a, q.conjugate(a)), [1.0, 0.0, 0.0, 0.0], atol=1e-15)


def test_propagation_rejects_invalid_dt_and_rate() -> None:
    """Non-positive or non-finite dt, and non-finite rates, are refused."""
    start = np.array([1.0, 0.0, 0.0, 0.0])
    for bad_dt in (0.0, -0.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            q.propagate(start, np.zeros(3), bad_dt)
    with pytest.raises(ValueError):
        q.propagate(start, np.array([np.nan, 0.0, 0.0]), 0.1)
    with pytest.raises(ValueError):
        q.as_scalar_first(np.zeros(4))


def test_repeated_propagation_stays_on_the_unit_sphere() -> None:
    """10000 gyro steps do not accumulate norm drift."""
    attitude = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(10_000):
        attitude = q.propagate(attitude, np.array([0.01, 0.02, -0.005]), 0.01)

    assert float(np.linalg.norm(attitude)) == pytest.approx(1.0, abs=1e-12)
