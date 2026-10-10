"""Stream ideal sensor readings from the truth simulation to an attached ESP32.

Protocol (ASCII, newline terminated, single serial line per sample):

    S <seq> <time_s> <gx> <gy> <gz> <mx> <my> <mz> <sx> <sy> <sz>
      <rx> <ry> <rz> <hx> <hy> <hz>

gx..gz gyro; mx..mz body MAG; sx..sy sz body Sun; rx..rz reference Sun unit
vector and hx..hz reference MAG unit vector (both normalized, GCRS).

The firmware echoes each frame back as:

    R <seq> <angle_deg> <rate_mag> <mag_norm> <sun_norm> <flags> <qw qx qy qz>

The harness verifies the firmware's TRIAD quaternion against truth attitude
via chiaverini (from the same reference implementation the MCU ports).

Acceptance verdict (the process exit code IS the verdict):

    0  PASS  — every expected sequence answered exactly once with a well-formed
               reply, no firmware error flags, all values finite, every
               reported rate/mag/sun magnitude matches the ORIGINAL vector the
               harness sent (a mag_norm of 1.0 is the documented in-place
               normalization regression), and the max angle / attitude errors
               within --angle-tol / --quat-tol.
    2  FAIL  — the verdict was computed but at least one check failed (see the
               "reasons:" lines printed before the MISMATCH line).
    1  SETUP FAILURE — the run could not be executed at all (frames could not
               be built, the port could not be opened, a write failed, ...).

Run:
    uv run python examples/hil_stream.py --port COM5 --duration 120
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ahrs.common.orientation import chiaverini
from scipy.spatial.transform import Rotation

from pocketadcs_sim.dynamics import InitialState, SimulationConfig, SpacecraftConfig
from pocketadcs_sim.sensors import collect_readings, sample_history
from pocketadcs_sim.sensors.hardware_adapters import select_sensor_suite
from pocketadcs_sim.truth_environment import TruthEnvironmentConfig, run_truth_environment

# Exit codes (see module docstring).
EXIT_OK = 0
EXIT_SETUP_FAILURE = 1
EXIT_MISMATCH = 2

# Firmware flag bits (esp32/src/main.cpp). bit0 = valid frame, bit1 = sun zero
# (eclipse), bit2 = parse error, bit3 = idle tick, bit4 = degenerate TRIAD.
FLAG_VALID = 1
FLAG_PARSE_ERROR = 4
FLAG_DEGENERATE = 16
# Flags the harness may never see on an acceptance frame: a parse error or a
# degenerate TRIAD means the firmware could not determine attitude.
REJECT_FLAGS = FLAG_PARSE_ERROR | FLAG_DEGENERATE


#: Mission epoch as Unix milliseconds. Must match the ``epoch_utc`` passed to
#: ``TruthEnvironmentConfig`` in build_frames() -- the firmware's ephemeris
#: needs an absolute date, so this is a protocol constant, not a convenience.
_EPOCH_START_UNIX_MS = 1735689600000  # 2025-01-01T00:00:00Z


@dataclass(frozen=True, slots=True)
class Frame:
    seq: int
    time_s: float
    gyro: tuple[float, float, float]
    mag: tuple[float, float, float]
    sun: tuple[float, float, float]
    truth_mrp: tuple[float, float, float]
    # Position and absolute epoch: what the firmware's onboard ephemeris needs
    # to derive the reference vectors. Optional with defaults so existing
    # callers that only care about the measured vectors still construct; the
    # defaults flag the missing state rather than imitating a real orbit.
    pos_gcrs_m: tuple[float, float, float] = (float("nan"),) * 3
    epoch_ms: int = 0
    # Retained for host-side scoring only; never transmitted. The firmware
    # must derive its own reference vectors or it is not doing ADS.
    #
    # These are optional so tests and scoring utilities can build a frame from
    # the transmitted fields alone -- the defaults are deliberately NOT
    # plausible unit vectors, so a caller that forgets to supply them gets a
    # scoring error instead of a silently perfect match.
    sun_ref: tuple[float, float, float] = (float("nan"),) * 3
    mag_ref: tuple[float, float, float] = (float("nan"),) * 3

    def angle_sun_mag_deg(self) -> float:
        m = np.asarray(self.mag)
        s = np.asarray(self.sun)
        m = m / np.linalg.norm(m)
        s = s / np.linalg.norm(s)
        return float(np.arccos(np.clip(np.dot(m, s), -1.0, 1.0)) * 180.0 / math.pi)

    def truth_quaternion(self) -> np.ndarray:
        dcm = Rotation.from_mrp(np.asarray(self.truth_mrp)).inv().as_matrix()
        return chiaverini(dcm)

    def to_line(self) -> str:
        """Serialize to the firmware's S-frame.

        Version 2. V1 carried the inertial reference vectors (sun_ref,
        mag_ref), which let the host perform the attitude determination and
        reduced the firmware to a lookup. V2 carries the spacecraft's own
        position instead, and the firmware derives the references onboard.

        The epoch is sent as a Unix-milliseconds value: the firmware's
        ephemeris needs a real date, and a monotonic "seconds since start"
        would force it to re-derive the mission epoch from a magic constant
        on both sides.
        """
        g = " ".join(f"{v:.6g}" for v in self.gyro)
        m = " ".join(f"{v:.6g}" for v in self.mag)
        s = " ".join(f"{v:.4f}" for v in self.sun)
        p = " ".join(f"{v:.3f}" for v in self.pos_gcrs_m)
        return f"S2 {self.seq} {self.epoch_ms} {g} {m} {s} {p}"


def build_frames(
    duration_s: float,
    rate_hz: float,
    sensors: str = "ideal",
    seed: int | None = None,
) -> list[Frame]:
    """Simulate a truth run and turn it into HIL frames.

    ``sensors`` selects which sensor models produce the streamed vectors:
    ``ideal`` (noise-free baseline), ``hardware`` (datasheet models, ADC
    quantization only) or ``hardware_noisy`` (datasheet models with their
    The frame carries ONLY things a real spacecraft legitimately knows about
    itself: its measured sensor readings, its own position (from GPS or
    propagation), and the timestamp.

    It deliberately does NOT carry the inertial reference vectors
    (sun_direction_gcrs / magnetic_field_gcrs). Those are the answer to the
    attitude-determination problem, and if the host transmits them the host is
    doing the ADS and the firmware is just echoing a rotation matrix back.
    The firmware computes them onboard from position + clock, which is what
    a flight system does. ``truth_mrp`` is retained purely so the host can
    score the firmware's reply -- it is never transmitted.
    """
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
    truth = run_truth_environment(
        vehicle,
        initial,
        SimulationConfig(duration_s=duration_s, step_s=0.1),
        TruthEnvironmentConfig(epoch_utc="2025-01-01T00:00:00", gravity_degree=4),
    )
    readings = collect_readings(
        truth, select_sensor_suite(sensors, rate_hz=rate_hz, seed=seed)
    )
    indices = sample_history(truth, rate_hz)
    frames = []
    for seq, index in enumerate(indices):
        frames.append(
            Frame(
                seq=seq,
                time_s=float(truth.time_s[index]),
                gyro=tuple(readings["gyro"].values[seq]),
                mag=tuple(readings["magnetometer"].values[seq]),
                sun=tuple(readings["sun"].values[seq]),
                pos_gcrs_m=tuple(np.asarray(truth.position_gcrs_m[index])),
                epoch_ms=_EPOCH_START_UNIX_MS + int(round(truth.time_s[index] * 1000.0)),
                sun_ref=tuple(np.asarray(truth.sun_direction_gcrs[index])),
                mag_ref=tuple(np.asarray(truth.magnetic_field_gcrs_t[index])),
                truth_mrp=tuple(np.asarray(truth.mrp_body_from_gcrs[index])),
            )
        )
    return frames


def stream(
    port: str, frames: Iterable[Frame], timeout_s: float, capture_path: str | None = None
) -> list[dict]:
    """Stream frames, optionally appending sequence-keyed send/reply events to JSONL."""
    import serial

    frame_list = list(frames)
    serial_port = serial.Serial()
    serial_port.port = port
    serial_port.baudrate = 115200
    serial_port.timeout = 1
    serial_port.write_timeout = max(1.0, timeout_s)
    serial_port.xonxoff = False
    serial_port.rtscts = False
    serial_port.dsrdtr = False
    serial_port.dtr = False
    serial_port.rts = False
    serial_port.open()
    replies: list[dict] = []
    sent_at: dict[int, float] = {}
    answered: set[int] = set()
    with ExitStack() as stack:
        stack.callback(serial_port.close)
        capture = None
        if capture_path is not None:
            capture = stack.enter_context(Path(capture_path).open("a", encoding="utf-8"))

        def record(event: dict) -> None:
            if capture is not None:
                capture.write(json.dumps(event, allow_nan=False) + "\n")
                capture.flush()

        def receive() -> None:
            line = serial_port.readline().decode(errors="replace").strip()
            if not line.startswith("R"):
                return
            replies.append({"raw": line})
            received_at = time.monotonic()
            try:
                parsed = parse_reply(line)
            except (ValueError, IndexError) as exc:
                record({"kind": "parse_error", "raw": line, "error": str(exc)})
                return
            seq = parsed["seq"]
            if seq == -2:
                record({"kind": "idle", "raw": line, "parsed": parsed})
            elif seq in sent_at and seq not in answered:
                answered.add(seq)
                record({"kind": "reply", "seq": seq, "raw": line,
                        "parsed": parsed, "latency_s": received_at - sent_at[seq]})
            else:
                record({"kind": "unmatched", "raw": line, "parsed": parsed})

        serial_port.reset_input_buffer()
        for frame in frame_list:
            # Keep a pipelined window below the target's 64-line message queue.
            # Any matched reply frees capacity; never wait for one particular
            # sequence, since valid replies may arrive out of order.
            window_deadline = time.monotonic() + timeout_s
            while len(sent_at) - len(answered) >= 16:
                if time.monotonic() >= window_deadline:
                    raise TimeoutError("no reply freed the bounded HIL send window")
                receive()
            line = frame.to_line()
            serial_port.write((line + "\n").encode())
            serial_port.flush()
            sent_at[frame.seq] = time.monotonic()
            record({"kind": "send", "seq": frame.seq, "line": line,
                    "truth_quaternion": frame.truth_quaternion().tolist(),
                    "time_s": frame.time_s})
            # The protocol is pipelined, not request/response: frames are sent
            # as fast as the port allows and replies are joined back by seq,
            # which may arrive in any order. So only drain what has already
            # arrived -- blocking here for a particular seq would serialise
            # the stream and reject a legitimately out-of-order reply.
            while serial_port.in_waiting:
                receive()
        # Keep draining after every expected sequence is answered: a firmware
        # that answers twice, or reports a parse error late, must not slip
        # through the exactly-once criterion just because we stopped reading.
        # `receive()` blocks on the port's 1 s read timeout, so a quiet port
        # ends each pass promptly; settle_deadline bounds the extra wait.
        deadline = time.monotonic() + timeout_s
        settle_deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            if len(answered) < len(frame_list) or serial_port.in_waiting:
                settle_deadline = time.monotonic() + 0.2
            elif time.monotonic() >= settle_deadline:
                break
            receive()
        return replies


def parse_reply(line: str) -> dict:
    parts = line.split()
    if len(parts) not in (11, 13) or parts[0] != "R":
        raise ValueError(f"Expected 11 or 13 fields in R reply, got {len(parts)}")
    result = {
        "seq": int(parts[1]),
        "angle_deg": float(parts[2]),
        "rate_mag": float(parts[3]),
        "mag_norm": float(parts[4]),
        "sun_norm": float(parts[5]),
        "flags": int(parts[6]),
        "quaternion": tuple(float(x) for x in parts[7:11]),
    }
    if len(parts) == 13:
        # Zephyr times onboard ephemeris and attitude computation separately.
        # Validate the trailer rather than silently accepting arbitrary extras.
        result["ephemeris_cycles"] = int(parts[11])
        result["attitude_cycles"] = int(parts[12])
    return result


@dataclass(frozen=True, slots=True)
class Verdict:
    """The acceptance verdict for one HIL run (exit code is the verdict)."""

    ok: bool
    reasons: tuple[str, ...]
    replies: int
    expected: int
    max_angle_err: float
    max_quat_err: float
    missing_seqs: tuple[int, ...]
    duplicate_seqs: tuple[int, ...]
    bad_flag_seqs: tuple[int, ...]
    nonfinite_seqs: tuple[int, ...]
    bad_quat_seqs: tuple[int, ...]
    bad_norm_seqs: tuple[int, ...]
    parse_error_count: int


def _quat_error_deg(received: tuple[float, ...], truth: np.ndarray) -> float:
    """Attitude error between quaternions, in degrees (sign-invariant).

    The error is the ACTUAL rotation angle between the two quaternions,
    2*acos(|dot|), not the half-angle acos(|dot|). Returns NaN if either
    quaternion is not finite or not near-unit norm (the firmware reports a
    normalized unit quaternion, so a healthy reply has norm 1 within
    float rounding; a wildly off norm is a magnitude corruption).
    """
    q = np.asarray(received, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return float("nan")
    q_norm = float(np.linalg.norm(q))
    if not (0.9 <= q_norm <= 1.1):
        return float("nan")
    q = q / q_norm
    truth = np.asarray(truth, dtype=float)
    if not np.all(np.isfinite(truth)):
        return float("nan")
    t_norm = float(np.linalg.norm(truth))
    if not (t_norm > 1e-12 and t_norm < 1e6):
        return float("nan")
    truth = truth / t_norm
    dot = abs(float(np.dot(q, truth)))
    if dot > 1.0 + 1e-9:
        return float("nan")
    return float(2.0 * math.degrees(math.acos(min(1.0, dot))))


def _first_nonfinite(values: tuple[float, ...]) -> str:
    for name, value in zip(
        ("angle_deg", "rate_mag", "mag_norm", "sun_norm", "qw", "qx", "qy", "qz"),
        values,
        strict=True,
    ):
        if not math.isfinite(value):
            return name
    return ""


def _magnitude_mismatch(reported: float, vector: tuple[float, ...]) -> bool:
    """True if a reported magnitude disagrees with the vector that was sent.

    Guards the documented firmware regression where TRIAD normalized `mag`
    in place and the reply then reported mag_norm = 1.0 instead of the
    original ~1e-5 T field, and the same class of error on the rate and Sun
    magnitudes. Tolerance is relative: the firmware computes the norm of the
    same values after the ASCII round-trip, in float32.
    """
    expected = float(np.linalg.norm(np.asarray(vector, dtype=float)))
    return abs(reported - expected) > max(1e-9, 1e-3 * expected)


def evaluate_replies(
    frames: list[Frame], replies: list[dict], angle_tol_deg: float, quat_tol_deg: float
) -> Verdict:
    """Acceptance evaluation for a full HIL run.

    Checks, in order: every expected sequence answered exactly once (no
    missing, no duplicate), reply well-formedness, firmware flag sanity
    (including standalone firmware parse-error replies), finiteness of
    every reported scalar and quaternion component, the reported rate/mag/
    sun magnitudes against the ORIGINAL magnitudes of the sent vectors
    (a mag_norm reply of 1.0 is the documented in-place-normalization
    regression), quaternion sanity (finite, near-unit) and the max angle /
    attitude error tolerances (attitude error is the actual rotation
    angle, 2*acos|dot|).
    """
    reasons: list[str] = []
    expected = len(frames)
    seen: dict[int, int] = {}
    parse_error_count = 0
    firmware_parse_errors = 0
    for reply in replies:
        try:
            parsed = parse_reply(reply["raw"])
        except ValueError:
            parse_error_count += 1
            continue
        # A well-formed R -1 (flags bit2) means the firmware could not parse a
        # frame. It fails the run even when every expected sequence also has a
        # good reply: a dropped frame is a dropped measurement.
        if parsed["flags"] & FLAG_PARSE_ERROR:
            firmware_parse_errors += 1
            continue
        seq = parsed["seq"]
        if 0 <= seq < expected:
            seen[seq] = seen.get(seq, 0) + 1

    missing = tuple(sorted(set(range(expected)) - set(seen)))
    duplicates = tuple(sorted(seq for seq, count in seen.items() if count > 1))
    if missing:
        reasons.append(f"missing replies for seq {missing}")
    if duplicates:
        reasons.append(f"duplicate replies for seq {duplicates}")

    bad_flag_seqs: list[int] = []
    nonfinite_seqs: list[int] = []
    bad_quat_seqs: list[int] = []
    bad_norm_seqs: list[int] = []
    max_angle_err = 0.0
    max_quat_err = 0.0
    by_seq: dict[int, list[dict]] = {}
    for reply in replies:
        try:
            parsed = parse_reply(reply["raw"])
        except ValueError:
            continue
        if 0 <= parsed["seq"] < expected:
            by_seq.setdefault(parsed["seq"], []).append(parsed)

    for seq in range(expected):
        for parsed in by_seq.get(seq, []):
            if parsed["flags"] & REJECT_FLAGS:
                bad_flag_seqs.append(seq)
                continue
            if not (parsed["flags"] & FLAG_VALID):
                bad_flag_seqs.append(seq)
                continue
            scalar_values = (
                parsed["angle_deg"],
                parsed["rate_mag"],
                parsed["mag_norm"],
                parsed["sun_norm"],
                *parsed["quaternion"],
            )
            if _first_nonfinite(scalar_values):
                nonfinite_seqs.append(seq)
                continue
            # The firmware must report the ORIGINAL magnitudes of the vectors
            # that were sent; mag_norm = 1.0 is the documented
            # in-place-normalization regression.
            frame = frames[seq]
            if (
                _magnitude_mismatch(parsed["rate_mag"], frame.gyro)
                or _magnitude_mismatch(parsed["mag_norm"], frame.mag)
                or _magnitude_mismatch(parsed["sun_norm"], frame.sun)
            ):
                bad_norm_seqs.append(seq)
                continue
            angle_err = abs(parsed["angle_deg"] - frames[seq].angle_sun_mag_deg())
            max_angle_err = max(max_angle_err, angle_err)
            quat_err = _quat_error_deg(parsed["quaternion"], frames[seq].truth_quaternion())
            if not math.isfinite(quat_err):
                bad_quat_seqs.append(seq)
            else:
                max_quat_err = max(max_quat_err, quat_err)

    if bad_flag_seqs:
        reasons.append(f"unacceptable firmware flags on seq {_dedup(bad_flag_seqs)}")
    if nonfinite_seqs:
        reasons.append(f"non-finite values on seq {_dedup(nonfinite_seqs)}")
    if bad_quat_seqs:
        reasons.append(f"invalid quaternion on seq {_dedup(bad_quat_seqs)}")
    if bad_norm_seqs:
        reasons.append(
            f"reported rate/mag/sun magnitudes do not match the original sent "
            f"vectors on seq {_dedup(bad_norm_seqs)}"
        )
    if max_angle_err > angle_tol_deg:
        reasons.append(
            f"max angle error {max_angle_err:.4f} deg > tolerance {angle_tol_deg:g} deg"
        )
    if max_quat_err > quat_tol_deg:
        reasons.append(
            f"max attitude error {max_quat_err:.4f} deg > tolerance {quat_tol_deg:g} deg"
        )
    if parse_error_count:
        reasons.append(f"{parse_error_count} unparseable reply line(s)")
    if firmware_parse_errors:
        reasons.append(
            f"{firmware_parse_errors} firmware parse-error reply(ies): the "
            f"firmware could not parse a streamed frame"
        )

    return Verdict(
        ok=not reasons,
        reasons=tuple(reasons),
        replies=len(replies),
        expected=expected,
        max_angle_err=max_angle_err,
        max_quat_err=max_quat_err,
        missing_seqs=missing,
        duplicate_seqs=duplicates,
        bad_flag_seqs=_dedup(bad_flag_seqs),
        nonfinite_seqs=_dedup(nonfinite_seqs),
        bad_quat_seqs=_dedup(bad_quat_seqs),
        bad_norm_seqs=_dedup(bad_norm_seqs),
        parse_error_count=parse_error_count,
    )


def _dedup(seq: list[int]) -> tuple[int, ...]:
    return tuple(sorted(set(seq)))


def run_acceptance(
    frames: list[Frame],
    replies: list[dict],
    angle_tol_deg: float,
    quat_tol_deg: float,
) -> Verdict:
    """Evaluate a streamed run and print the acceptance report (no exit here)."""
    verdict = evaluate_replies(frames, replies, angle_tol_deg, quat_tol_deg)
    print(f"Expected replies: {verdict.expected}")
    print(f"Replies received: {verdict.replies}")
    print(f"Max angle error vs truth: {verdict.max_angle_err:.4f} deg")
    print(f"Max firmware-vs-truth attitude error: {verdict.max_quat_err:.4f} deg")
    for reason in verdict.reasons:
        print(f"  reason: {reason}")
    print("OK" if verdict.ok else "MISMATCH")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--dry", action="store_true", help="print frames without opening a port")
    parser.add_argument("--capture", metavar="PATH", help="append S/R events as JSONL")
    parser.add_argument("--per-frame", type=float, default=1.0,
                        help="max seconds to wait for one frame's reply")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="seconds to wait for outstanding replies after the last frame")
    parser.add_argument("--angle-tol", type=float, default=0.1,
                        help="max allowed angle error vs truth, deg")
    parser.add_argument("--quat-tol", type=float, default=2.0,
                        help="max allowed firmware-vs-truth attitude error, deg")
    parser.add_argument("--sensors", default="ideal",
                        choices=("ideal", "hardware", "hardware_noisy"),
                        help="sensor models feeding the frames: noise-free baseline, "
                             "datasheet models with ADC quantization only, or the "
                             "datasheet models with their documented noise")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for the sensor models (reproducible noise)")
    args = parser.parse_args()

    try:
        frames = build_frames(args.duration, args.rate, args.sensors, args.seed)
    except Exception as exc:  # setup failure: the run never started
        print(f"SETUP FAILURE: could not build frames: {exc}")
        raise SystemExit(EXIT_SETUP_FAILURE) from exc
    print(f"Built {len(frames)} frames for a {args.duration}s run at {args.rate} Hz")
    for frame in frames[:3]:
        print("  sample:", frame.to_line())

    if args.dry:
        return

    try:
        replies = stream(args.port, frames, timeout_s=args.timeout, capture_path=args.capture)
    except Exception as exc:
        print(f"SETUP FAILURE: {type(exc).__name__}: {exc}")
        raise SystemExit(EXIT_SETUP_FAILURE) from exc

    verdict = run_acceptance(frames, replies, args.angle_tol, args.quat_tol)
    raise SystemExit(EXIT_OK if verdict.ok else EXIT_MISMATCH)


if __name__ == "__main__":
    main()
