"""The HIL frame protocol: the host must not do the attitude determination.

The attitude determination runs on the ESP32. The host's job is to simulate
the orbit and the sensor hardware, transmit what the vehicle legitimately
knows about itself, and score the reply against ground truth it never sends.

This fixture pins the two halves of that separation:

  * the wire frame carries position and time, NOT the inertial reference
    vectors -- refSun/refMag in the frame would mean the host computed them,
    and the firmware was echoing a rotation matrix back;
  * ground truth is present on the host side (for scoring) and absent from
    the serialized line.
"""

from __future__ import annotations

import math
import re

import pytest

from examples.hil_stream import (
    FLAG_VALID,
    Frame,
)

#: A representative S2 line: type, seq, epoch_ms, gyro[3], mag[3], sun[3], pos[3].
S2_RE = re.compile(
    r"^S2 (?P<seq>\d+) (?P<epoch>\d+) "
    r"(?P<gyro>-?\d+\.?\d*) (?P<gyro2>-?\d+\.?\d*) (?P<gyro3>-?\d+\.?\d*) "
    r"(?P<mag>-?\d+\.?\d*) (?P<mag2>-?\d+\.?\d*) (?P<mag3>-?\d+\.?\d*) "
    r"(?P<sun>-?\d+\.?\d*) (?P<sun2>-?\d+\.?\d*) (?P<sun3>-?\d+\.?\d*) "
    r"(?P<x>-?\d+\.?\d*) (?P<y>-?\d+\.?\d*) (?P<z>-?\d+\.?\d*)$"
)


def _frame(**overrides) -> Frame:
    base = dict(
        seq=7,
        time_s=12.5,
        gyro=(0.01, -0.02, 0.03),
        mag=(10.0, 20.0, 30.0),
        sun=(0.0, 0.0, 1.0),
        pos_gcrs_m=(6_878_137.0, 1000.0, 2000.0),
        epoch_ms=1_735_689_600_000,
        truth_mrp=(0.1, -0.2, 0.05),
        sun_ref=(0.0, 0.0, 1.0),
        mag_ref=(1.0, 0.0, 0.0),
    )
    base.update(overrides)
    return Frame(**base)


class TestWireFormat:
    def test_line_is_s2(self):
        assert _frame().to_line().startswith("S2 ")

    def test_line_matches_the_s2_grammar(self):
        assert S2_RE.match(_frame().to_line())

    def test_frame_carries_position(self):
        """The firmware's ephemeris needs to know where the vehicle is."""
        match = S2_RE.match(_frame().to_line())
        assert match is not None
        assert float(match["x"]) == pytest.approx(6_878_137.0, abs=1e-3)
        assert float(match["y"]) == pytest.approx(1000.0, abs=1e-3)
        assert float(match["z"]) == pytest.approx(2000.0, abs=1e-3)

    def test_frame_carries_absolute_epoch(self):
        """An absolute date, so the firmware needs no shared magic epoch."""
        match = S2_RE.match(_frame().to_line())
        assert match is not None
        assert int(match["epoch"]) == 1_735_689_600_000

    def test_frame_does_not_carry_reference_vectors(self):
        """THE load-bearing property.

        refSun/refMag in the frame means the host performed the attitude
        determination. The 16-value v1 frame is gone; v2 sends position
        instead, and the firmware derives the references onboard.
        """
        line = _frame().to_line()
        # v1 sent: type, seq, time, then 5 vectors (gyro/mag/sun/refSun/refMag)
        # = 2 scalars + 15 components. v2 sends type, seq, epoch_ms, then 4
        # vectors (gyro/mag/sun/pos) = 3 scalars + 12 components + type = 15.
        fields = line.split()
        assert len(fields) == 15, f"expected type + 14 fields, got {len(fields)}"
        # Positionally: the last triple is position, not a reference vector.
        values = [float(f) for f in fields[2:]]
        assert values[-3:] == pytest.approx([6_878_137.0, 1000.0, 2000.0], abs=1e-3)
        # And the epoch is an integer milliseconds value, not seconds-since-start.
        assert int(fields[2]) == 1_735_689_600_000


class TestGroundTruthStaysHostSide:
    def test_truth_is_not_in_the_line(self):
        """Ground truth exists for scoring and must never be transmitted."""
        frame = _frame()
        line = frame.to_line()
        for value in frame.truth_mrp:
            assert f"{value:.4f}" not in line

    def test_reference_vectors_are_not_in_the_line(self):
        frame = _frame()
        line = frame.to_line()
        # The inertial references are the *answer*; only the measured body-frame
        # vectors belong on the wire.
        assert line.count("1.000000") == 0

    def test_frame_still_exposes_truth_for_scoring(self):
        """Scoring needs it, so it must survive on the host side."""
        frame = _frame()
        assert len(frame.truth_mrp) == 3
        assert math.isfinite(frame.angle_sun_mag_deg())
        assert len(frame.truth_quaternion()) == 4

    def test_reference_vectors_are_still_retained_host_side(self):
        """Kept for host-side comparison, so we can quantify model error."""
        frame = _frame()
        assert len(frame.sun_ref) == 3
        assert len(frame.mag_ref) == 3


class TestSequenceAndFlagContract:
    def test_seq_is_carried(self):
        assert S2_RE.match(_frame(seq=1234).to_line()).group("seq") == "1234"

    def test_valid_flag_is_defined(self):
        """The harness gates on FLAG_VALID; the firmware sets it on success."""
        assert isinstance(FLAG_VALID, int)
