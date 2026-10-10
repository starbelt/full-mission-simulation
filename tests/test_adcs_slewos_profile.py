"""Independent checks on the reference profile that ``angle_at`` produces.

The rest of the adapter tests are about *reading* a plan. This file is about the
trajectory that reading it produces, and it deliberately avoids re-deriving the
implementation's own closed form -- a transcription of the code's expression
passes on the code even when the expression is wrong, which is exactly how the
wrong-denominator defect survived two review rounds.

Two independent references are used instead:

* **Numeric integration.** The profile's velocity is a piecewise-linear ramp
  that starts and ends at zero. Integrating it on a fine grid and comparing
  against ``angle_at`` checks the position against an entirely different
  algorithm than the analytic one under test.
* **The planner's own ``alpha``.** A symmetric trapezoid's peak velocity is
  ``alpha * accel_s`` and its covered distance is ``v_peak * V`` with
  ``V = T - (accel + decel) / 2``. Differencing ``angle_at`` therefore yields
  the acceleration the planner actually commanded. Where the sibling SlewOS repo
  is present, that is compared against ``physics.trapezoid_timing`` itself.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from pocketadcs_sim.adcs.slewos_adapter import (
    SLEWOS_DEFAULT_PATH,
    AdapterError,
    ScheduleSegment,
)

#: Angular acceleration the planner used, rad/s^2. Matches FA01's study
#: specification default; only used to build a physically consistent segment.
ALPHA_RAD_S2 = 2.0


def planner_alpha_for(delta_rad: float, duration_s: float) -> float:
    """Return the acceleration implied by the planner's timing equation.

    Independent of the adapter: this inverts FA01's ``trapezoid_timing``
    discriminant for ``alpha``, so the test does not need to import the sibling
    package.

    Given an acceleration ramp ``a`` measured out of a trapezoid of total time
    ``T``, the timing equation ``T - 2a = sqrt(T^2 - 4|delta|/alpha)`` gives
    ``alpha = 4|delta| / (T^2 - (T - 2a)^2)``. Written this way the cancellation
    is in the *given* ramp time rather than in a subtraction of nearly equal
    roots, so it stays accurate for the long-coast profiles FA01 really emits.
    """
    accel_s = 0.5 * (duration_s - np.sqrt(duration_s**2 - 4.0 * abs(delta_rad) / ALPHA_RAD_S2))
    return 4.0 * abs(delta_rad) / (duration_s**2 - (duration_s - 2.0 * accel_s) ** 2)


def make_segment(
    delta_rad: float, duration_s: float, *, coast_s: float | None = None
) -> ScheduleSegment:
    """Build a physically consistent trapezoidal segment of ``delta_rad``.

    ``coast_s`` sets the coast explicitly; pass 0 for the triangular profile the
    planner produces when a move is exactly duration-limited. Left as ``None``,
    the coast absorbs whatever the planner's timing equation leaves over, which
    is how every real FA01 profile is built.
    """
    if coast_s is None:
        discriminant = duration_s**2 - 4.0 * abs(delta_rad) / ALPHA_RAD_S2
        accel_s = 0.5 * (duration_s - np.sqrt(discriminant))
        coast_s = max(duration_s - 2.0 * accel_s, 0.0)
    else:
        accel_s = 0.5 * (duration_s - coast_s)
    decel_s = accel_s
    return ScheduleSegment(
        objective_id=0,
        start_angle_rad=0.3,
        target_angle_rad=0.3 + delta_rad,
        duration_s=duration_s,
        start_time_s=0.0,
        profile_s=(accel_s, coast_s, decel_s),
    )


def reference_velocity(segment: ScheduleSegment, times: np.ndarray) -> np.ndarray:
    """Piecewise-linear velocity of the planner's profile, in rad/s.

    Signed by the direction of the move, so a negative-angle slew integrates
    downward rather than being compared against an upward ramp.
    """
    delta = segment.target_angle_rad - segment.start_angle_rad
    direction = 1.0 if delta >= 0.0 else -1.0
    accel, coast, _ = segment.profile_s
    peak = ALPHA_RAD_S2 * accel
    velocity = np.empty_like(times)
    accel_phase = times <= accel
    coast_phase = (times > accel) & (times <= accel + coast)
    decel_phase = times > accel + coast
    velocity[accel_phase] = ALPHA_RAD_S2 * times[accel_phase]
    velocity[coast_phase] = peak
    velocity[decel_phase] = peak - ALPHA_RAD_S2 * (times[decel_phase] - accel - coast)
    return direction * velocity


def integrated_angle(segment: ScheduleSegment, t_s: float, *, steps: int = 200_000) -> float:
    """Angle at ``t_s`` from numerically integrating the reference velocity."""
    grid = np.linspace(0.0, float(t_s), steps)
    velocity = reference_velocity(segment, grid)
    return float(segment.start_angle_rad + np.trapezoid(velocity, grid))


@pytest.mark.parametrize(
    ("delta_rad", "duration_s", "coast_s"),
    [
        (0.5, 1.0, None),  # triangular: exactly the planner's minimum duration
        (0.5, 2.0, None),  # trapezoid with a real coast
        (1.2, 3.7, None),  # longer, less symmetric in the numbers
        (0.5, 1.0, 0.0),  # triangular, forced explicitly
        (-0.75, 2.5, None),  # negative move
    ],
)
def test_angle_matches_numeric_integration_of_the_profile(
    delta_rad: float, duration_s: float, coast_s: float | None
) -> None:
    """``angle_at`` agrees with an independent numerical integration."""
    segment = make_segment(delta_rad, duration_s, coast_s=coast_s)

    for local in (0.25, 0.5, 0.75):
        t_s = local * segment.duration_s
        assert segment.angle_at(t_s) == pytest.approx(
            integrated_angle(segment, t_s), abs=1e-7
        )


def test_asymmetric_profile_uses_accel_time_on_the_accel_side() -> None:
    """A profile with ``accel != decel`` is evaluated with the right one.

    FA01 only ever emits symmetric profiles, so no symmetric fixture can tell an
    ``accel`` in the acceleration branch from a ``decel`` written there by
    mistake. The dataclass accepts any split, so the branch is pinned with an
    asymmetric one: swapping the two in the acceleration branch leaves every
    symmetric case untouched and moves only this.
    """
    segment = ScheduleSegment(
        objective_id=0,
        start_angle_rad=0.0,
        target_angle_rad=1.0,
        duration_s=9.0,
        start_time_s=0.0,
        profile_s=(1.0, 4.0, 4.0),
    )

    # Halfway through the acceleration ramp the profile is at half its peak
    # velocity, so a fraction of v_peak * (T - (accel + decel) / 2) is covered.
    effective_s = 9.0 - 0.5 * (1.0 + 4.0)
    assert segment.angle_at(0.5) == pytest.approx(
        0.5 * 0.5 * 0.5 / (1.0 * effective_s), abs=1e-12
    )
    # Using decel (4.0 s) there instead would put the angle 4x lower.
    assert segment.angle_at(0.5) > 0.5 * 0.5 * 0.5 / (4.0 * effective_s)

    # The deceleration ramp is the mirror image, and is measured with decel.
    assert segment.angle_at(8.5) == pytest.approx(
        1.0 - 0.5 * 0.5 * 0.5 / (4.0 * effective_s), abs=1e-12
    )


@pytest.mark.parametrize("coast_s", [None, 0.0])
def test_profile_is_continuous_at_the_coast_to_decel_junction(coast_s: float | None) -> None:
    """No commanded-angle step at the junction; the profile is rest-to-rest."""
    segment = make_segment(0.5, 2.0, coast_s=coast_s)
    accel, coast, _ = segment.profile_s
    junction = accel + coast

    before = segment.angle_at(np.nextafter(junction, 0.0))
    at = segment.angle_at(junction)

    assert abs(at - before) < 1e-12
    # And the angle is still moving there, so this is not a trivially flat seam.
    assert segment.angle_at(junction + 0.5) > at


@pytest.mark.parametrize("duration_s", [1.0, 2.0, 3.7])
def test_profile_starts_at_start_and_ends_exactly_at_target(duration_s: float) -> None:
    """Endpoints are exact, not merely close."""
    segment = make_segment(0.5, duration_s)

    assert segment.angle_at(segment.start_time_s) == segment.start_angle_rad
    assert segment.angle_at(segment.duration_s) == segment.target_angle_rad


@pytest.mark.parametrize("duration_s", [1.0, 2.0, 3.7])
def test_implied_acceleration_is_the_planner_alpha(duration_s: float) -> None:
    """The angular acceleration of ``angle_at`` is the planner's ``alpha``.

    The coast phase is a constant velocity, so it pins nothing; the acceleration
    is read off the acceleration ramp instead, where a wrongly normalised
    profile immediately shows a different value.
    """
    segment = make_segment(0.5, duration_s)

    grid = np.linspace(0.0, segment.duration_s, 200_001)
    angles = np.array([segment.angle_at(float(t)) for t in grid])
    dt = float(grid[1] - grid[0])
    accel_ramp, coast, _ = segment.profile_s
    # Second differences in the acceleration ramp: the first difference of angle
    # is the velocity ramp (alpha * t), so the second difference recovers alpha.
    # A wrongly normalised profile shows a different constant here immediately.
    # Stay two samples clear of the ramp/coast junction too: a central
    # difference straddling it mixes the ramp with the constant-velocity phase.
    indices = np.flatnonzero((grid > 2.0 * dt) & (grid < accel_ramp - 2.0 * dt))
    assert indices.size > 10, "expected a non-empty ramp phase to difference"
    velocity = (angles[1:] - angles[:-1]) / dt
    implied = (velocity[indices + 1] - velocity[indices - 1]) / (2.0 * dt)

    assert implied == pytest.approx(ALPHA_RAD_S2, rel=1e-3)
    # rel=1e-4: second differences of a piecewise-quadratic carry the profile's
    # own floating-point noise, which is ~1e-6 relative at double precision.
    assert implied == pytest.approx(planner_alpha_for(0.5, duration_s), rel=1e-4)
    # The deceleration ramp carries the same magnitude, opposite sign. Stay two
    # samples clear of the end, where angle_at clamps to the target angle.
    decel = np.flatnonzero(
        (grid > accel_ramp + coast + 2.0 * dt) & (grid < segment.duration_s - 2.0 * dt)
    )
    assert decel.size > 10
    assert (velocity[decel + 1] - velocity[decel - 1]) / (2.0 * dt) == pytest.approx(
        -ALPHA_RAD_S2, rel=1e-3
    )


def test_triangular_profile_reaches_half_the_move_at_its_midpoint() -> None:
    """With no coast, half the distance is covered by half the duration.

    This is the profile FA01 actually produces when a move is duration-limited,
    and it is where normalising by ``T`` instead of ``V`` is worst.
    """
    segment = make_segment(0.5, 1.0, coast_s=0.0)

    assert segment.angle_at(0.5 * segment.duration_s) == pytest.approx(
        segment.start_angle_rad + 0.25, abs=1e-9
    )


def test_degenerate_profile_with_no_effective_time_is_refused() -> None:
    """A profile with no time to move in raises instead of dividing by zero."""
    segment = ScheduleSegment(
        objective_id=0,
        start_angle_rad=0.0,
        target_angle_rad=1.0,
        duration_s=0.4,
        start_time_s=0.0,
        profile_s=(0.4, 0.0, 0.4),
    )

    with pytest.raises(AdapterError, match="effective time|zero"):
        segment.angle_at(0.2 * segment.duration_s)


def test_real_a2_schedule_reconstructed_profile_matches_planner_commanded_trapezoid() -> None:
    """Every real segment's implied acceleration is the planner's ``alpha``.

    Reads an actual FA01 result from the sibling repo and checks each segment
    of the plan, so the defect this file guards against cannot hide in the one
    profile the fixtures happen to use.
    """
    pytest.importorskip("numpy")
    if not SLEWOS_DEFAULT_PATH.exists():
        pytest.skip("sibling SlewOS repo not present")

    studies = SLEWOS_DEFAULT_PATH / "src" / "simple_model" / "studies"
    if str(studies) not in sys.path:
        sys.path.insert(0, str(studies))
    physics = pytest.importorskip(
        "fa01.physics", reason="sibling FA01 package not importable"
    )

    import glob
    import json
    from pathlib import Path

    from pocketadcs_sim.adcs.slewos_adapter import load_schedule

    candidates = sorted(
        glob.glob(str(SLEWOS_DEFAULT_PATH / "results" / "fa01" / "fa01_a*" / "raw.jsonl"))
    )
    if not candidates:
        pytest.skip("no FA01 study A results found in the sibling repo")
    results = Path(candidates[-1]).parent
    manifest = json.loads(
        (results / "n2" / "manifest.json").read_text(encoding="utf-8")
    )
    case_id = manifest["cases"][0]["case_id"]
    first = json.loads((results / "raw.jsonl").read_text(encoding="utf-8").splitlines()[0])
    if not first.get("feasible", True):
        pytest.skip("first planner record is infeasible")

    schedule = load_schedule(results, case_id=case_id, method=first["method"])

    def ramp_velocity(
        times: np.ndarray, accel: float, coast: float, alpha: float, direction: float
    ) -> np.ndarray:
        """Velocity of a real segment's profile; resolvable at ~0.6 ms ramps.

        Signed by the direction of the move: FA01 orders objectives whichever way
        the angles require, so half the real segments slew downward.
        """
        peak = alpha * accel
        velocity = np.empty_like(times)
        rising = times <= accel
        cruising = (times > accel) & (times <= accel + coast)
        falling = times > accel + coast
        velocity[rising] = alpha * times[rising]
        velocity[cruising] = peak
        velocity[falling] = peak - alpha * (times[falling] - accel - coast)
        return direction * velocity

    checked = 0
    for segment in schedule.segments:
        delta = segment.target_angle_rad - segment.start_angle_rad
        if abs(delta) < 1e-6:
            continue
        accel, coast, _ = segment.profile_s
        if accel <= 0.0:
            continue
        # Recover the acceleration the profile carries, then hand it back to the
        # planner's own timing function. Agreement means angle_at reproduces the
        # trapezoid FA01 actually commanded, using FA01's own code to say so.
        alpha = 4.0 * abs(delta) / (
            segment.duration_s**2 - (segment.duration_s - 2.0 * accel) ** 2
        )
        assert physics.trapezoid_timing(delta, segment.duration_s, alpha) == pytest.approx(
            segment.profile_s, rel=1e-9
        )
        # Compare against a numerical integration of the velocity ramp. Two
        # windows: one resolving the millisecond-long acceleration ramp, one
        # spanning the coast. A single uniform grid over 555 s cannot resolve a
        # 0.6 ms ramp, which is exactly the profile shape FA01 really produces.
        # angle_at takes absolute time, so the local windows below are shifted by
        # the segment's own start time -- the second segment of a real plan does
        # not start at t = 0.
        t0 = segment.start_time_s
        for local in (
            np.linspace(0.0, min(3.0 * accel, segment.duration_s), 20_001),
            np.linspace(3.0 * accel, segment.duration_s, 20_001),
        ):
            grid = t0 + local
            direction = 1.0 if delta >= 0.0 else -1.0
            velocity = ramp_velocity(local, accel, coast, alpha, direction)
            # Each window is integrated from its own left edge, then shifted onto
            # that edge's commanded angle.
            integrated = segment.angle_at(float(grid[0])) + np.concatenate(
                (
                    [0.0],
                    np.cumsum(0.5 * (velocity[:-1] + velocity[1:]) * np.diff(grid)),
                )
            )
            angles = np.array([segment.angle_at(float(t)) for t in grid])
            # 1e-5 rad, not tighter, and the last sample is excluded: raw.jsonl
            # stores profiles_s rounded to five decimals, so the acceleration
            # recoverable from the file differs from the planner's own by ~1e-5
            # relative, and integrating that over a 2000 s coast accumulates
            # ~1.5e-5 rad. The endpoint is checked exactly below instead. That
            # rounding is an input-precision limit, not slack in the profile:
            # the synthetic cases above carry the exact discrimination.
            assert angles[:-1] == pytest.approx(integrated[:-1], abs=1e-5)
        # Endpoints exact, and no step at the coast/decel junction.
        assert segment.angle_at(segment.start_time_s) == segment.start_angle_rad
        assert segment.angle_at(segment.end_time_s) == segment.target_angle_rad
        junction = segment.start_time_s + accel + coast
        assert abs(
            segment.angle_at(junction) - segment.angle_at(np.nextafter(junction, 0.0))
        ) < 1e-12
        checked += 1

    assert checked > 0, "expected at least one real segment to check"
