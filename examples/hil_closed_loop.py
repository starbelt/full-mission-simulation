"""Closed-loop HIL: the HOST is the stand-in, the DEVICE is the ADCS.

This harness scores the loop that runs on the ESP32-S3. It deliberately does NOT
reproduce the closed loop on the host: a host-side "expected" attitude would be a
second ADS, and comparing the device against it would measure neither.

What the host owns (a stand-in for reality):
  * the orbit and the environment,
  * sensor hardware errors (bias, noise, quantization, dropouts),
  * the PLANT -- the spacecraft's rotational dynamics. Only the host knows this;
    it is not simulated on the device,
  * the SCORING: attitude error against withheld truth.

What the device owns (the thing under test):
  * the inertial reference vectors, from its own position and clock,
  * the TRIAD attitude solution,
  * propagation and rejection logic,
  * the control law and wheel allocation,
  * the safety logic (stale gap, no-fix, over-long line).

There is deliberately no field for truth attitude, truth rate, or a
precomputed reference vector. The protocol has no room for them, which is the
point: see ``hil_firmware_truth_ownership``.

Measured numbers from this harness:
  * attitude error vs withheld truth, in degrees,
  * settling time to the commanded final target,
  * overshoot past the target,
  * maximum wheel speed actually commanded, vs the rotor bound,
  * EXACTLY-ONCE reply accounting, and every refusal with its reason.

NOT measured here, and deliberately absent:
  * stage execution time. The device reports cycle counts per stage; this
    harness passes them through to the caller unconverted. A rate ceiling
    derived from a host write-to-reply round trip is TRANSPORT latency and is
    not an algorithm number.

Run::

    PYTHONPATH=src .venv/Scripts/python.exe examples/hil_closed_loop.py --port COM5
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

import numpy as np

EXIT_OK = 0
EXIT_SETUP_FAILURE = 1
EXIT_MISMATCH = 2

#: Mission epoch as Unix milliseconds (2025-01-01T00:00:00Z). Absolute, not
#: seconds-since-start: the onboard ephemeris needs a real date, and the same
#: constant cannot drift between the two sides.
EPOCH_START_UNIX_MS = 1735689600000

# Protocol status codes, mirroring esp32/src/adcs_protocol.h.
STATUS_OK = 0
STATUS_BAD_ARGUMENT = 1
STATUS_TIME_REGRESSION = 2
STATUS_CAPACITY = 3
STATUS_NOT_CONFIGURED = 4

#: Stage order in the Y reply's cycle counts. Ephemeris first: it is the model,
#: not the algorithm under test, and reporting it separately keeps the ADS
#: number honest.
STAGE_NAMES = ("ephemeris", "estimator", "controller", "allocation")


class ProtocolError(RuntimeError):
    """The device said no. Raised for anything that fails a check."""


def format_xs2(seq: int, t_s: float, gyro: Sequence[float], magnetic: Sequence[float],
               sun: Sequence[float], pos_gcrs_m: Sequence[float]) -> str:
    """One closed-loop sample frame.

    ``sun`` of all zeros means "no sun measurement" (eclipse). It is not a real
    direction and the device treats it as absent, so the harness must be able to
    say it.
    """
    fields = [seq, EPOCH_START_UNIX_MS + int(round(t_s * 1000.0)), t_s]
    fields.extend(float(v) for v in gyro)
    fields.extend(float(v) for v in magnetic)
    fields.extend(float(v) for v in sun)
    fields.extend(float(v) for v in pos_gcrs_m)
    return "XS2 " + " ".join(repr(float(v)) for v in fields)


def plan_lines(initial_angle_rad: float,
               segments: Iterable[Sequence[float]]) -> list[str]:
    """``XL`` / ``XS`` / ``XE`` for a piecewise-linear angle schedule.

    Each segment is ``(objective_id, start_angle_rad, target_angle_rad,
    duration_s, accel_s, coast_s, decel_s)`` -- the firmware's own ``XS`` order.
    Segments are contiguous by construction: the device assigns each segment's
    start time from the durations before it, which is also what makes a
    mis-ordered plan detectable at ``XE``.

    A declared count of 0 opens a reference-profile transaction instead; see
    :func:`reference_profile_lines`.
    """
    segments = list(segments)
    lines = [f"XL {initial_angle_rad!r} {len(segments)}"]
    for segment in segments:
        if len(segment) != 7:
            raise ValueError(f"a plan segment has {len(segment)} fields, expected 7")
        lines.append("XS " + " ".join(repr(float(v)) for v in segment))
    lines.append("XE")
    return lines


def reference_profile_lines(start: Sequence[float], target: Sequence[float],
                            total_s: float) -> list[str]:
    """``XL 0 0`` + ``XR`` + ``XE``, the path used for 3D pointing.

    The device holds no attitude knowledge of its own here: the host supplies a
    reference profile, the device times and interpolates it and controls to it.
    That is the planner interface, not an injected truth -- and it is recorded as
    an externally supplied profile, because the implemented FA01 engine is
    single-axis rest-to-rest and this is not planner output.
    """
    quaternions = [*start, *target]
    if len(quaternions) != 8:
        raise ValueError(f"start and target must be 4 numbers each, got "
                         f"{len(quaternions)}")
    return ["XL 0 0", "XR " + " ".join(repr(float(v)) for v in
                                       (*quaternions, float(total_s))), "XE"]


@dataclass(frozen=True, slots=True)
class YReply:
    """A parsed ``Y`` reply. Field positions are fixed by the wire format."""

    status: int
    seq: int
    has_fix: bool
    estimate: np.ndarray          # the attitude the CONTROLLER used
    control_input: np.ndarray     # must equal estimate; a mismatch means truth leaked
    body_torque: np.ndarray
    wheel_torque: np.ndarray
    saturated: bool
    stale: bool
    reject: int
    cycles: np.ndarray            # raw device counts, unconverted
    service_cycles: int | None = None  # optional target-only complete dispatch span

    def cycle_times_us(self, counter_hz: int) -> dict[str, object]:
        """Convert device counts using validated XI frequency, never RTT or stage sum.

        Callers must establish device provenance; native callback counts are
        test-only and must not be promoted to hardware performance.
        """
        if (not math.isfinite(counter_hz) or counter_hz != int(counter_hz)
                or not 0 < counter_hz <= 0xFFFFFFFF):
            raise ProtocolError("invalid device counter frequency")
        return {
            "stages": {name: int(cycles) * 1e6 / counter_hz
                       for name, cycles in zip(STAGE_NAMES, self.cycles, strict=True)},
            "service": None if self.service_cycles is None
                       else self.service_cycles * 1e6 / counter_hz,
        }

    def __post_init__(self) -> None:
        for name in ("estimate", "control_input", "body_torque", "wheel_torque"):
            value = getattr(self, name)
            if value.shape != (4,) and value.shape != (3,):
                raise ProtocolError(f"{name} has shape {value.shape}, expected (4,) or (3,)")


def parse_y(line: str) -> YReply:
    """Parse a ``Y`` reply, rejecting anything malformed rather than guessing."""
    tokens = line.split()
    if not tokens or tokens[0] != "Y":
        raise ProtocolError(f"not a Y reply: {line!r}")
    if len(tokens) not in (25, 26):
        # Original 24 numbers, optionally followed by target service_cycles.
        raise ProtocolError(f"Y reply has {len(tokens)} fields, expected 25 or 26: {line!r}")
    try:
        values = [float(t) for t in tokens[1:]]
    except ValueError as exc:
        raise ProtocolError(f"Y reply has a non-numeric field: {line!r}") from exc
    if not all(math.isfinite(v) for v in values):
        raise ProtocolError(f"Y reply has a non-finite field: {line!r}")
    for index in (0, 1, 2, 17, 18, 19, *range(20, len(values))):
        if values[index] != int(values[index]):
            raise ProtocolError("Y reply has a fractional integer field")
    if any(values[index] not in (0, 1) for index in (2, 17, 18)):
        raise ProtocolError("Y reply has an invalid boolean field")
    # Validate exact decimal cycle tokens before float rounding can turn a
    # fractional/over-limit count into a valid-looking uint32 boundary value.
    try:
        cycle_values = [Decimal(token) for token in tokens[21:]]
    except InvalidOperation as exc:
        raise ProtocolError("Y reply has an invalid decimal cycle count") from exc
    if any(not value.is_finite() or not 0 <= value <= 0xFFFFFFFF
           or value != value.to_integral_value() for value in cycle_values):
        raise ProtocolError("Y reply cycle counts must be uint32")
    for first in (3, 7):
        norm = np.linalg.norm(values[first:first + 4])
        if not 0.9 <= norm <= 1.1:
            raise ProtocolError("Y reply has a wildly nonunit quaternion")
    return YReply(
        status=int(values[0]),
        seq=int(values[1]),
        has_fix=bool(values[2]),
        estimate=np.array(values[3:7]),
        control_input=np.array(values[7:11]),
        body_torque=np.array(values[11:14]),
        wheel_torque=np.array(values[14:17]),
        saturated=bool(values[17]),
        stale=bool(values[18]),
        reject=int(values[19]),
        cycles=np.array(values[20:24]),
        service_cycles=int(values[24]) if len(values) == 25 else None,
    )


def quaternion_error_deg(estimate: Sequence[float], truth: Sequence[float]) -> float:
    """Rotation angle between two quaternions, in degrees.

    ``2*acos(|dot|)``, not ``acos(|dot|)``. The half-angle halves every reported
    error, which is how a 3 deg error passes a 2 deg tolerance. Both are
    normalised first: a dot of 1.00000002 clips to 0 deg and hides real error
    behind a median of 0.000000.
    """
    a = np.asarray(estimate, dtype=float)
    b = np.asarray(truth, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        raise ProtocolError("a zero quaternion is not an attitude")
    dot = abs(float(np.dot(a / na, b / nb)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


@dataclass(slots=True)
class LoopScore:
    """Everything measured about one closed-loop run."""

    max_error_deg: float = 0.0
    final_error_deg: float = 0.0
    settling_time_s: float | None = None
    overshoot_deg: float = 0.0
    max_wheel_speed_rad_s: float = 0.0
    saturated_steps: int = 0
    max_stage_cycles: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "max_error_deg": self.max_error_deg,
            "final_error_deg": self.final_error_deg,
            "settling_time_s": self.settling_time_s,
            "overshoot_deg": self.overshoot_deg,
            "max_wheel_speed_rad_s": self.max_wheel_speed_rad_s,
            "saturated_steps": self.saturated_steps,
            # Raw device counts. Converting these to microseconds needs the real
            # core clock, which this harness has not measured.
            "max_stage_cycles": dict(self.max_stage_cycles),
        }


def settling_time(errors_deg: Sequence[float], times_s: Sequence[float],
                  tolerance_deg: float) -> float | None:
    """First time the error stays within tolerance for the rest of the run.

    Not "first crossing": a loop that touches tolerance once and then diverges
    has not settled, and a first-crossing metric reports it as settled.
    """
    for index in range(len(errors_deg)):
        if all(e <= tolerance_deg for e in errors_deg[index:]):
            return float(times_s[index])
    return None


def overshoot_deg(errors_deg: Sequence[float], times_s: Sequence[float],
                  target_s: float) -> float:
    """Peak error excursion, in degrees, after the command reaches the target."""
    tail = [e for e, t in zip(errors_deg, times_s, strict=False) if t >= target_s]
    return max(tail) if tail else 0.0
