"""Tests for the reaction-wheel board stand-in.

The board owns the commanded wheel torque and its own continuous rotor state.
These tests pin the three behaviours the card calls out: a commanded
acceleration cap, speed saturation, and continuity of wheel state across a
command timeout (no reset).
"""

from __future__ import annotations

import numpy as np
import pytest

from pocketadcs_sim.adcs.wheel_board import CommandTimeout, WheelBoard


def make_board(**kwargs: object) -> WheelBoard:
    """Return a board on the default three-wheel set."""
    return WheelBoard(**kwargs)  # type: ignore[arg-type]


def test_uncommanded_wheels_stay_at_rest() -> None:
    """No command means no torque, and the rotors do not drift."""
    board = make_board()

    applied = board.update(np.zeros(3), dt_s=0.01)

    np.testing.assert_allclose(applied, np.zeros(3), atol=0.0)
    np.testing.assert_allclose(board.wheel_speed_rad_s, np.zeros(3), atol=0.0)


def test_commanded_torque_below_the_cap_is_passed_through_unchanged() -> None:
    """Inside its limits the board is transparent, not lossy."""
    board = make_board()
    torque = np.array([1e-4, -1e-4, 5e-5])

    applied = board.update(torque, dt_s=0.01)

    np.testing.assert_allclose(applied, torque, atol=0.0)
    assert not board.saturated


def test_acceleration_cap_limits_the_torque_for_the_step() -> None:
    """The commanded acceleration cap bounds u/J, hence the torque per step."""
    cap = 20.0
    board = make_board(acceleration_cap_rad_s2=cap)
    dt = 0.01
    huge = np.array([0.0028, 0.0, 0.0])  # far beyond the cap

    applied = board.update(huge, dt_s=dt)

    inertia = 3.27e-6
    np.testing.assert_allclose(applied[0], cap * inertia, rtol=1e-12)
    np.testing.assert_allclose(applied[1:], np.zeros(2), atol=0.0)
    assert board.saturated


def test_acceleration_cap_scales_the_wheel_step_not_the_momentum() -> None:
    """A capped step advances Omega by exactly cap * dt, not by more."""
    cap = 20.0
    board = make_board(acceleration_cap_rad_s2=cap)

    board.update(np.array([0.0028, 0.0, 0.0]), dt_s=0.25)

    np.testing.assert_allclose(board.wheel_speed_rad_s[0], cap * 0.25, rtol=1e-12)


def test_speed_limit_blocks_further_acceleration_at_the_bound() -> None:
    """At the speed bound the board refuses torque that would push past it."""
    board = make_board()
    board.set_wheel_speed(np.array([board.max_speed_rad_s, 0.0, 0.0]))

    applied = board.update(np.array([1e-4, 0.0, 0.0]), dt_s=0.01)

    assert applied[0] == 0.0
    assert board.wheel_speed_rad_s[0] == pytest.approx(board.max_speed_rad_s, rel=1e-12)
    assert board.saturated


def test_speed_limit_still_allows_spinning_down() -> None:
    """A wheel at its bound must still be able to decelerate (braking is allowed)."""
    board = make_board()
    board.set_wheel_speed(np.array([board.max_speed_rad_s, 0.0, 0.0]))

    applied = board.update(np.array([-1e-4, 0.0, 0.0]), dt_s=0.01)

    assert applied[0] < 0.0
    assert board.wheel_speed_rad_s[0] < board.max_speed_rad_s


def test_saturation_is_reported_per_wheel() -> None:
    """Reporting says which wheel saturated, not merely that something did."""
    board = make_board(acceleration_cap_rad_s2=20.0)
    # The cap is 20 * 3.27e-6 = 6.54e-5 N.m, so the middle wheel must stay
    # under it to be the unsaturated case.
    board.update(np.array([0.0028, 1e-5, 0.0028]), dt_s=0.01)

    np.testing.assert_array_equal(board.saturated_per_wheel, [True, False, True])
    assert board.saturated


def test_command_timeout_keeps_wheel_state_and_zeroes_torque() -> None:
    """A command timeout must not reset rotor speed; the wheels coast."""
    board = make_board()
    board.update(np.array([1e-4, -2e-4, 3e-4]), dt_s=0.5)
    speed_before = board.wheel_speed_rad_s.copy()

    applied = board.update(None, dt_s=0.5)

    np.testing.assert_allclose(board.wheel_speed_rad_s, speed_before, atol=0.0)
    np.testing.assert_allclose(applied, np.zeros(3), atol=0.0)
    assert board.timed_out is True
    assert board.saturated is False


def test_recovery_after_timeout_resumes_from_the_preserved_state() -> None:
    """The board picks up from the state it kept, with no discontinuity."""
    board = make_board()
    board.update(np.array([1e-4, 0.0, 0.0]), dt_s=0.5)
    speed_before = board.wheel_speed_rad_s.copy()

    board.update(None, dt_s=0.5)
    board.update(np.array([1e-4, 0.0, 0.0]), dt_s=0.5)

    expected = speed_before[0] + 1e-4 / 3.27e-6 * 0.5
    assert board.wheel_speed_rad_s[0] == pytest.approx(expected, rel=1e-12)
    assert board.timed_out is False


def test_commanded_and_applied_are_reported_separately() -> None:
    """The commanded torque must be distinguishable from what was applied.

    A stand-in that silently clips is a stand-in that cannot be debugged, and a
    downstream controller cannot tell a limit from an actuator fault.
    """
    board = make_board(acceleration_cap_rad_s2=20.0)
    commanded = np.array([0.0028, 0.0, 0.0])

    applied = board.update(commanded, dt_s=0.01)

    np.testing.assert_allclose(board.commanded_torque_nm, commanded, atol=0.0)
    np.testing.assert_allclose(board.applied_torque_nm, applied, atol=0.0)
    assert board.commanded_torque_nm[0] != board.applied_torque_nm[0]


def test_reverse_command_after_saturation_is_not_clipped() -> None:
    """The cap is a bound on this step's acceleration, not a latched limit."""
    cap = 20.0
    board = make_board(acceleration_cap_rad_s2=cap)
    board.update(np.array([0.0028, 0.0, 0.0]), dt_s=1.0)

    applied = board.update(np.array([-0.0028, 0.0, 0.0]), dt_s=1.0)

    np.testing.assert_allclose(applied[0], -cap * 3.27e-6, rtol=1e-12)
    np.testing.assert_allclose(board.wheel_speed_rad_s[0], 0.0, atol=1e-9)


def test_invalid_inputs_are_rejected() -> None:
    """Bad commands raise rather than producing silently wrong wheel state."""
    board = make_board()

    with pytest.raises(ValueError):
        board.update(np.array([np.nan, 0.0, 0.0]), dt_s=0.01)
    with pytest.raises(ValueError):
        board.update(np.zeros(3), dt_s=0.0)
    with pytest.raises(ValueError):
        board.update(np.zeros(3), dt_s=-0.01)
    with pytest.raises(ValueError):
        board.update(np.zeros(2), dt_s=0.01)


def test_command_timeout_error_is_available_as_an_exception() -> None:
    """The timeout type exists so callers can distinguish it from a zero command."""
    assert issubclass(CommandTimeout, Exception)


def test_board_exposes_its_configuration_for_results() -> None:
    """The record must show the cap is an engineering parameter, and the source."""
    record = make_board(acceleration_cap_rad_s2=25.0).to_dict()

    assert record["acceleration_cap_rad_s2"] == 25.0
    assert record["acceleration_cap_is_datasheet_claim"] is False
    assert "TGR 29-12" in str(record["source"])
    assert record["commutation"] == "perfect"
    assert len(record["axes"]) == 3
