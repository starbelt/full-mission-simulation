"""The closed-loop HIL harness's scoring layer, tested off-board.

The scoring code decides whether a run passes, and the exit code IS the verdict.
A check that is too weak is worse than no check, so each one here is written to
fail on the specific false PASS it exists to prevent -- and each of those false
PASSes has actually happened in this project.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.hil_closed_loop import (
    EPOCH_START_UNIX_MS,
    ProtocolError,
    format_xs2,
    overshoot_deg,
    parse_y,
    plan_lines,
    quaternion_error_deg,
    reference_profile_lines,
    settling_time,
)
from pocketadcs_sim.adcs import quaternions as quat


def y_line(**overrides: float | int) -> str:
    """A well-formed Y reply, with individual fields overridable."""
    fields: list[float] = [
        0.0,      # status
        1.0,      # seq
        1.0,      # has_fix
        1.0, 0.0, 0.0, 0.0,   # estimate (identity)
        1.0, 0.0, 0.0, 0.0,   # control input (identity)
        0.0, 0.001, 0.002,     # body torque
        -0.001, -0.002, 0.0,   # wheel torque
        0.0,      # saturated
        0.0,      # stale
        0.0,      # reject
        100.0, 200.0, 50.0, 30.0,  # stage cycles
    ]
    for index, value in overrides.items():
        fields[int(index[1:])] = value
    return "Y " + " ".join(repr(float(v)) for v in fields)


class TestFrameFormatting:
    def test_frame_carries_absolute_epoch_and_no_truth_field(self) -> None:
        """The epoch must be absolute, or the onboard ephemeris is undefined.

        Seconds-since-start forces a mission-epoch constant to stay in sync on
        two sides, and a disagreement between them is a plausible wrong
        attitude rather than a parse error.
        """
        line = format_xs2(7, 12.5, [0, 0, 0.1], [0.2, 1, 0.3], [1, 0, 0], [6.9e6, 0, 0])
        values = [float(t) for t in line.split()[1:]]
        assert len(values) == 15, f"XS2 must be 15 numbers, got {len(values)}"
        assert values[1] == EPOCH_START_UNIX_MS + 12500
        assert values[2] == 12.5

    def test_frame_has_no_field_a_vehicle_could_not_know(self) -> None:
        """The ownership test: grep the serializer for smuggled truth.

        Truth attitude, truth rate, or a precomputed reference vector would make
        the host the ADS and every accuracy number a measurement of the host.
        """
        forbidden = ("truth", "mrp", "ref_sun", "ref_mag", "reference", "attitude")
        with open("examples/hil_closed_loop.py", encoding="utf-8") as fh:
            source = fh.read()
        serializer = source[source.index("def format_xs2"):source.index("def plan_lines")]
        for token in forbidden:
            assert token not in serializer, f"XS2 serializer mentions {token!r}"

    def test_all_zero_sun_is_representable_as_eclipse(self) -> None:
        """Zero sun must mean "no measurement", not "pointing nowhere".

        A zero vector is a real reading in other sensors; here it is the only
        way to say the sun is occluded, and the device tests for exactly this.
        """
        line = format_xs2(1, 0.0, [0, 0, 0], [0, 1, 0], [0, 0, 0], [6.9e6, 0, 0])
        sun = [float(t) for t in line.split()[1:]][9:12]
        assert sun == [0.0, 0.0, 0.0]

    def test_plan_frame_counts_segments_before_the_end_marker(self) -> None:
        lines = plan_lines(0.0, [(0.0, 0.5, 100.0, 0, 0, 100.0, 0.0)])
        assert lines[0] == "XL 0.0 1"
        assert len(lines) == 3 and lines[-1] == "XE"

    def test_reference_profile_declares_zero_segments(self) -> None:
        """A profile plan declares no segments, and the device must accept that.

        XL used to refuse a count below 1, which made this sequence -- the only
        way to reach XR and therefore the 3D pointing path -- unrunnable.
        """
        lines = reference_profile_lines([1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0], 60.0)
        assert lines[0] == "XL 0 0", "a profile plan declares no segments"
        assert len(lines[1].split()) == 10, "XR is qStart[4] qTarget[4] total_s + tag"
        assert lines[-1] == "XE"

    def test_reference_profile_rejects_a_mismatched_quaternion_count(self) -> None:
        """Fail in the host rather than transmitting a frame the device refuses."""
        with pytest.raises(ValueError):
            reference_profile_lines([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 60.0)


class TestReplyParsing:
    def test_well_formed_reply_parses(self) -> None:
        reply = parse_y(y_line())
        assert reply.status == 0 and reply.seq == 1 and reply.has_fix
        assert reply.estimate == pytest.approx([1, 0, 0, 0])
        assert reply.cycles.tolist() == [100.0, 200.0, 50.0, 30.0]

    def test_wrong_field_count_is_rejected(self) -> None:
        with pytest.raises(ProtocolError):
            parse_y("Y 1 2 3")

    def test_wrong_tag_is_rejected(self) -> None:
        with pytest.raises(ProtocolError):
            parse_y(y_line().replace("Y ", "R ", 1))

    def test_non_finite_field_is_rejected(self) -> None:
        """A NaN silently compares false against every tolerance.

        Without this check, a firmware that answered ``nan`` for the quaternion
        would sail through an error assertion because ``nan <= tol`` is False in
        a way a bare ``assert`` reads as a pass.
        """
        with pytest.raises(ProtocolError):
            parse_y(y_line(f3=float("nan")))

    def test_non_numeric_field_is_rejected(self) -> None:
        with pytest.raises(ProtocolError):
            parse_y(y_line().replace("1.0", "abc", 1))


class TestQuaternionError:
    def test_identical_attitudes_give_exactly_zero(self) -> None:
        q = quat.from_axis_angle([0.3, 0.8, 0.5], 0.7)
        assert quaternion_error_deg(q, q) == pytest.approx(0.0, abs=1e-9)

    def test_quarter_turn_measures_ninety_degrees(self) -> None:
        """A known rotation must measure itself, catching a half-angle error.

        acos(|dot|) reports 45 deg for a 90 deg rotation, which halves every
        error and would pass a 2 deg tolerance with a 3 deg error.
        """
        q = quat.from_axis_angle([0.0, 0.0, 1.0], np.pi / 2.0)
        assert quaternion_error_deg(q, [1.0, 0, 0, 0]) == pytest.approx(90.0, abs=1e-6)

    def test_sign_flip_of_the_same_rotation_is_zero(self) -> None:
        q = quat.from_axis_angle([0.0, 1.0, 0.0], 0.3)
        assert quaternion_error_deg(q, -np.asarray(q)) == pytest.approx(0.0, abs=1e-9)

    def test_slightly_nonunit_quaternion_does_not_read_as_zero(self) -> None:
        """A dot just over 1 clips to exactly 0 deg and hides the error.

        This is the "mean 0.004 deg hiding behind a median of 0.000000" case:
        divide by both norms and clamp, rather than clamping alone.
        """
        q = np.array([1.0, 0.0, 0.0, 0.0]) * (1.0 + 1e-8)
        truth = quat.from_axis_angle([0.0, 0.0, 1.0], 1e-3)
        assert quaternion_error_deg(q, truth) > 1e-4

    def test_zero_quaternion_is_rejected_not_treated_as_perfect(self) -> None:
        """The all-zero quaternion was a real regression: valid=1, norm 0.

        ``dcmToQuaternion`` emitted it via ``signf(0) == 0`` on an exact 180 deg
        attitude. Treating it as zero error would be a perfect score for a
        device that returned nothing.
        """
        with pytest.raises(ProtocolError):
            quaternion_error_deg([0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])


class TestSettlingAndOvershoot:
    def test_first_crossing_alone_does_not_count_as_settled(self) -> None:
        """Touching tolerance once and then diverging has NOT settled.

        A first-crossing metric reports 1.0 s here; the honest answer is None.
        """
        errors = [50.0, 5.0, 0.5, 5.0, 40.0, 30.0]
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert settling_time(errors, times, 1.0) is None

    def test_first_sustained_entry_is_the_settling_time(self) -> None:
        errors = [50.0, 5.0, 0.5, 0.4, 0.6]
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        assert settling_time(errors, times, 1.0) == 2.0

    def test_overrun_past_the_target_is_reported(self) -> None:
        errors = [10.0, 2.0, 1.0, 4.0, 2.0]
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        # Only excursions after the command reaches the target count; the 10 deg
        # initial error is not an overshoot.
        assert overshoot_deg(errors, times, target_s=2.0) == 4.0

    def test_no_post_target_samples_gives_zero_overshoot(self) -> None:
        assert overshoot_deg([10.0, 5.0], [0.0, 1.0], target_s=99.0) == 0.0
