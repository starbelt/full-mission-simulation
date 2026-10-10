"""The versioned closed-loop application protocol, exercised on the host.

``esp32/src/adcs_protocol.h`` is the seam between the serial line and the ADCS
core: it parses the stateful closed-loop commands, runs the estimator /
reference / control / allocation stages, and formats one seq-linked reply per
line. Zephyr's ``main.cpp`` calls it; so does this test, compiled with a host
C++ compiler, so protocol defects are caught before the board is touched.

The properties pinned here are the ones that make a HIL run trustworthy:

* **A parse failure is transactional.** A malformed or refused line must leave
  the estimate, the plan and the counters exactly as they were. A partially
  applied command is how a run reports a plausible number for a state the
  device never actually held.
* **No actuator output before the first absolute fix.** The estimate starts at
  the identity attitude, which is a real rotation to nowhere; commanding
  against it before any fix is commanding against a guess.
* **Stale input coasts.** After a long sample gap the device reports stale and
  commands zero torque. It does not integrate a huge ``dt``, and it does not
  reset the rotors: wheel momentum is preserved, so a dropout coasts instead of
  silently discarding stored momentum.
* **Diagnostics are always reachable.** Boot/health uptime, RX drops, queue
  overflows and main-thread stack headroom are reported by a command, so silent
  packet loss or stack exhaustion cannot hide behind a passing run.
* **The old protocol still works.** The stateless S2/R path is untouched: this
  is an additional versioned protocol, not a silent change to the existing one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_ads_math_host import HEADER_DIR, _env_for, find_usable_compiler

from pocketadcs_sim.adcs import quaternions as quat

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "tests" / "firmware" / "adcs_protocol_driver.cpp"

OK = 0

#: Reply record types, matching the driver's leading integer.
Y_FRAME = 0
E_REFUSAL = 1
H_HEALTH = 2

SUN_REF = np.array([1.0, 0.1, -0.2])
MAG_REF = np.array([0.2, 1.0, 0.3])
MAX_TORQUE = 0.0028
CAP_TORQUE = 200.0 * 3.27e-6

# Y frame field offsets, after the leading record type.
F_STATUS = 0
F_SEQ = 1
F_HAS_FIX = 2
F_Q = slice(3, 7)
F_CTRL_Q = slice(7, 11)
F_TAU = slice(11, 14)
F_WHEEL = slice(14, 17)
F_SATURATED = 17
F_STALE = 18
F_REJECT = 19
F_CYCLES = slice(20, 24)

# H line field offsets.
H_STATUS = 0
H_UPTIME = 1
H_DROPS = 2
H_OVERFLOWS = 3
H_HEADROOM = 4
H_HAS_FIX = 5


@pytest.fixture(scope="module")
def protocol(tmp_path_factory: pytest.TempPathFactory):
    """Compile the protocol driver once; return a callable exchanging lines."""
    tmp_path = tmp_path_factory.mktemp("adcs_protocol")
    compiler = find_usable_compiler(tmp_path)
    if compiler is None:
        pytest.skip("no usable host C++ compiler (g++/clang++/MSYS2) found")
    exe = tmp_path / (
        "adcs_protocol_driver.exe" if sys.platform == "win32" else "adcs_protocol_driver"
    )
    build = subprocess.run(
        [
            str(compiler), "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-I", str(HEADER_DIR), str(DRIVER), "-o", str(exe), "-lm",
        ],
        env=_env_for(compiler),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if build.returncode != 0:
        pytest.fail(f"host build of the ADCS protocol failed with {compiler}:\n{build.stderr}")
    return exe


def exchange(protocol: Path, lines: list[str]) -> list[tuple[int, list[float]]]:
    """Send lines to a fresh device session; return ``(record type, fields)``."""
    script = lines
    proc = subprocess.run(
        [str(protocol)],
        input="\n".join(script) + "\n",
        env=_env_for(protocol),
        capture_output=True,
        text=True,
        encoding="latin1",
        timeout=120,
    )
    assert proc.returncode == 0, f"driver exited {proc.returncode}:\n{proc.stdout}{proc.stderr}"
    lines_out = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines_out) == len(script), (
        f"expected {len(script)} replies, got {len(lines_out)}:\n{proc.stdout}"
    )
    replies = []
    for line in lines_out:
        tokens = line.split()
        replies.append((int(tokens[0]), [float(token) for token in tokens[1:]]))
    return replies


@pytest.mark.parametrize(
    "invalid", ["XZ\x00 extra", "XZ\x01", "XZ\x1a", "XZ\x7f", "XZ\x80", "\x00XZ",
                "X\rZ", "X\r\rZ", "XC .3\r5 .15 .0028 200 3.27e-6 1675.5160819145563",
                "XL .\r2 0", "XS2 3 1700000000\r000 .1 0 0 0 .2 1 .3 1 .1 -.2 6900000 0 0"]
)
def test_binary_reset_framing_preserves_dynamic_and_pending_state(protocol, invalid):
    attitude = quat.from_axis_angle([1, -2, 3], .3)
    setup = [config_line(), "XL 0 0", "XR 1 0 0 0 1 0 0 0 20", "XE",
             sample_line(1, 0, attitude), sample_line(2, .05, attitude),
             "XL .2 0", "XR 1 0 0 0 1 0 0 0 20"]
    probes = ["TEST_DYNAMIC", "TEST_PLANS"]
    replies = exchange(protocol, setup + probes + [invalid] + probes +
                       [sample_line(3, .1, attitude)])
    before = replies[len(setup):len(setup) + 2]
    refusal = replies[len(setup) + 2]
    after = replies[len(setup) + 3:len(setup) + 5]
    assert refusal[0] == E_REFUSAL and refusal[1][0] == 4
    assert after == before
    assert replies[-1][0] == Y_FRAME and replies[-1][1][F_STATUS] == OK
    baseline = exchange(protocol, setup + [sample_line(3, .1, attitude)])
    assert replies[-1][1][:20] == baseline[-1][1][:20]
    assert any(abs(v) > 0 for v in before[0][1][9:12])


@pytest.mark.parametrize("reset", ["XZ", "XZ\r", "XZ \t\r", "XZ\r\r"])
def test_native_crlf_reset_and_cr_whitespace_preserve_valid_syntax(protocol, reset):
    config = config_line().replace(" ", "\r", 1)
    replies = exchange(protocol, [config, "XL\r0\r0", "XR\r1 0 0 0 1 0 0 0 20",
                                 "XE\r", reset, "TEST_STORES_CLEAR"])
    assert all(kind == E_REFUSAL and fields[0] == OK for kind, fields in replies[:-1])
    assert replies[-1] == (7, [1])


def config_line(kp: float = 0.35, kd: float = 0.15) -> str:
    return f"XC {kp} {kd} {MAX_TORQUE} 200 3.27e-6 1675.5160819145563"


def sample_line(
    seq: int,
    t_s: float,
    attitude: np.ndarray,
    gyro=(0.0, 0.0, 0.0),
    eclipse: bool = False,
) -> str:
    """One closed-loop sensor frame: what the vehicle knows about itself.

    The reference vectors are NOT in the line. The device derives them from
    position and clock through its own ephemeris, which is the whole point --
    sending them would make the host the attitude determiner.
    """
    dcm = quat.to_dcm(attitude)
    sun = np.zeros(3) if eclipse else dcm @ SUN_REF
    mag = dcm @ MAG_REF
    values = [seq, 1_700_000_000_000.0, t_s, *gyro, *mag, *sun, 6.9e6, 0.0, 0.0]
    return "XS2 " + " ".join(f"{float(v):.17g}" for v in values)


def plan_lines(initial_rad: float, segments: list[tuple]) -> list[str]:
    """Open a plan transaction, append its segments, then commit it."""
    lines = [f"XL {initial_rad:.17g} {len(segments)}"]
    for objective, start, target, duration, accel, coast, decel in segments:
        lines.append(
            f"XS {objective} {start:.17g} {target:.17g} {duration:.17g} "
            f"{accel:.17g} {coast:.17g} {decel:.17g}"
        )
    lines.append("XE")
    return lines


def frames(replies: list[tuple[int, list[float]]]) -> list[list[float]]:
    """The Y frames with seq >= 1, i.e. the closed-loop sample replies."""
    return [
        fields
        for kind, fields in replies
        if kind == Y_FRAME and fields[F_SEQ] >= 1
    ]


# ---------------------------------------------------------------------------


def test_health_is_readable_before_any_sample(protocol: Path) -> None:
    """Health must be readable before a fix exists, with no state implied."""
    kind, fields = exchange(protocol, ["XH"])[0]
    assert kind == H_HEALTH, f"health did not answer with a health line: {fields}"
    assert fields[H_STATUS] == OK
    assert fields[H_UPTIME] == 0.0, "a fresh session reports zero uptime"
    assert fields[H_DROPS] == 0.0 and fields[H_OVERFLOWS] == 0.0
    assert fields[H_HEADROOM] > 0.0, "the main thread must report positive stack headroom"
    assert fields[H_HAS_FIX] == 0.0, "no absolute fix before the first sample"


def test_no_actuator_output_before_the_first_absolute_fix(protocol: Path) -> None:
    """A wheel command before any fix is a command against the identity guess.

    The estimate starts at the identity attitude, so driving toward a target
    from there would command a real torque for a state the vehicle has not
    observed. The device must command zero torque.
    """
    attitude = np.array([1.0, 0.0, 0.0, 0.0])
    lines = [
        config_line(),
        *plan_lines(0.0, [(0, 0.0, 0.5, 100.0, 0.0, 100.0, 0.0)]),
        sample_line(1, 0.0, attitude, eclipse=True),
    ]
    frame = frames(exchange(protocol, lines))[-1]
    assert frame[F_HAS_FIX] == 0.0, "an eclipse sample cannot produce an absolute fix"
    assert all(value == 0.0 for value in frame[F_WHEEL]), (
        f"no wheel torque may be commanded before the first fix, got {frame[F_WHEEL]}"
    )
    assert all(value == 0.0 for value in frame[F_TAU]), (
        f"no body torque may be commanded before the first fix, got {frame[F_TAU]}"
    )


def test_full_closed_loop_frame_carries_estimate_control_and_wheels(protocol: Path) -> None:
    """One accepted sample must yield a complete, seq-linked reply.

    Every field a scorer needs is in the reply: the estimate, the attitude the
    controller was actually handed, the body torque and the wheel commands.
    The controller's input is echoed separately from the estimate because
    reading back the caller's intent instead of the value used is the same
    defect as a hardcoded ``truth_leakage`` flag.
    """
    plan = [(0, 0.0, 0.6, 100.0, 0.0, 100.0, 0.0)]
    lines = [config_line(), *plan_lines(0.0, plan)]
    attitudes = [quat.from_axis_angle([0.0, 0.0, 1.0], 0.05 * step) for step in range(5)]
    for seq, attitude in enumerate(attitudes, start=1):
        lines.append(sample_line(seq, 0.05 * seq, attitude, gyro=(0.0, 0.0, 0.05)))
    got = frames(exchange(protocol, lines))
    assert len(got) == 5, got
    for index, frame in enumerate(got, start=1):
        assert frame[F_STATUS] == OK, f"frame {index} refused: {frame}"
        assert frame[F_SEQ] == index, f"reply not seq-linked: {frame}"
        assert frame[F_HAS_FIX] == 1.0, f"frame {index} should hold a fix"
        # The echoed controller input is the estimate the torque came from.
        assert frame[F_CTRL_Q] == frame[F_Q], f"controller input is not the estimate: {frame}"
        for value in frame[F_TAU]:
            assert abs(value) <= MAX_TORQUE + 1e-15, (
                f"body torque exceeds the rotor limit: {frame[F_TAU]}"
            )
        # Allocation is the negative of the body torque on orthogonal wheels,
        # then capped by the commanded acceleration bound.
        for body, wheel in zip(frame[F_TAU], frame[F_WHEEL], strict=True):
            assert wheel == pytest.approx(-body, abs=1e-12) or abs(wheel) == pytest.approx(
                CAP_TORQUE, abs=1e-15
            )
    # The final frame must have driven a nonzero wheel command: a run whose
    # wheels never move proves nothing about the control path.
    assert any(abs(v) > 0.0 for v in got[-1][F_WHEEL]), f"no wheel command: {got[-1]}"


def test_stage_timings_are_reported_separately(protocol: Path) -> None:
    """Ephemeris, estimator, control and allocation must be timed apart.

    A single combined number cannot say whether a rate target is met because
    the estimator is slow or because the ephemeris is. The host injects known
    per-stage costs so the split is observable; on the device these come from
    the hardware cycle counter.
    """
    attitude = quat.from_axis_angle([0.0, 0.0, 1.0], 0.1)
    lines = [config_line(), *plan_lines(0.0, [(0, 0.0, 0.5, 100.0, 0.0, 100.0, 0.0)])]
    for seq in range(1, 4):
        lines.append(sample_line(seq, 0.05 * seq, attitude, gyro=(0.0, 0.0, 0.05)))
    frame = frames(exchange(protocol, lines))[-1]
    stages = frame[F_CYCLES]
    assert len(stages) == 4
    for index, value in enumerate(stages):
        assert value > 0.0, f"stage {index} reported no cycles: {stages}"
    # Distinct stages, distinct costs: a single lumped measurement copied into
    # four fields would defeat the purpose of reporting them separately.
    assert len(set(stages)) >= 2, f"stage timings are not independent: {stages}"


def test_reference_profile_can_be_loaded_at_all(protocol: Path) -> None:
    """A zero-segment plan plus XR must commit a reference profile.

    XL rejected a declared count below 1, and XR is only reachable through the
    plan transaction's XE, so ``XL 0 0`` + ``XR`` + ``XE`` -- the documented
    reference-profile sequence -- could never commit anything. The 3D pointing
    path was unreachable on the device, not merely untested.
    """
    # XR is qStart[4] qTarget[4] total_s = 9 numbers. A 45 deg rotation about
    # +Y, so the interpolated desired attitude differs from the start and a
    # controller that loaded nothing cannot produce a non-zero command.
    lines = [config_line(), "XL 0 0",
             "XR 1 0 0 0 0.9238795325112867 0 0.3826834323650898 0 60", "XE"]
    replies = exchange(protocol, lines)
    assert all(kind == E_REFUSAL and fields[0] == OK for kind, fields in replies), (
        f"the reference profile must load: {replies}"
    )
    # And it must actually control: a profile that loaded but was never used
    # would also report success. Sampled half-way through the profile, where
    # the interpolated desired attitude differs from the start attitude.
    lines.append(sample_line(1, 30.0, quat.from_axis_angle([0.0, 0.0, 1.0], 0.0)))
    frame = frames(exchange(protocol, lines))[-1]
    assert frame[F_STATUS] == OK
    assert frame[F_HAS_FIX] == 1.0
    assert any(abs(v) > 0.0 for v in frame[F_WHEEL]), (
        f"the loaded reference profile commanded nothing: {frame}"
    )


def test_long_gap_reports_stale_and_coasts_without_a_huge_propagation(protocol: Path) -> None:
    """A dropout must not be integrated as one enormous timestep.

    Propagating across a multi-second gap with a stale gyro produces a large
    confident error. A TRIAD fix landing afterwards would hide that, so this
    case is built so the fix CANNOT land: the post-gap sample is in eclipse, so
    the only way the estimate could move is by integrating 29.95 s of gyro.
    That is what makes this test discriminating rather than decorative.
    """
    first = quat.from_axis_angle([0.0, 0.0, 1.0], 0.1)
    second = quat.from_axis_angle([0.0, 0.0, 1.0], 0.9)
    lines = [config_line(), *plan_lines(0.0, [(0, 0.0, 0.5, 100.0, 0.0, 100.0, 0.0)])]
    lines.append(sample_line(1, 0.0, first, gyro=(0.0, 0.0, 0.1)))
    lines.append(sample_line(2, 0.05, first, gyro=(0.0, 0.0, 0.1)))
    # A 29.95 s gap. The gyro says +0.1 rad/s, so integrating it would swing the
    # estimate by ~2.99 rad -- an obvious, and entirely fictitious, rotation.
    lines.append(sample_line(3, 30.0, second, gyro=(0.0, 0.0, 0.1), eclipse=True))
    got = frames(exchange(protocol, lines))
    good, gap = got[-2], got[-1]
    assert good[F_STALE] == 0.0, "the pre-gap frame must not be stale"
    assert gap[F_STALE] == 1.0, f"a {30 - 0.05}s gap must be reported stale: {gap}"
    assert gap[F_HAS_FIX] == 1.0, "the pre-gap fix must be retained across the gap"
    assert all(value == 0.0 for value in gap[F_WHEEL]), (
        f"a stale link must coast with zero wheel torque, got {gap[F_WHEEL]}"
    )
    # The estimate must be unmoved: with the fix suppressed, any motion could
    # only have come from integrating the stale gyro across the whole gap.
    assert gap[F_Q] == good[F_Q], (
        f"the estimate moved across a stale gap: {gap[F_Q]} vs {good[F_Q]}"
    )
    assert gap[F_REJECT] == 1.0, f"the post-gap eclipse must be reported: {gap}"


def test_parse_failure_leaves_all_state_unchanged(protocol: Path) -> None:
    """A malformed line must be refused without touching anything.

    Each bad line is sent between two good samples. If any of them advanced the
    estimate, applied a partial plan or reset the session, the surrounding good
    frames stop matching the clean run -- which is exactly how a partial parse
    hides.
    """
    attitude = quat.from_axis_angle([0.0, 0.0, 1.0], 0.2)
    good_lines = [
        config_line(),
        *plan_lines(0.0, [(0, 0.0, 0.5, 100.0, 0.0, 100.0, 0.0)]),
        sample_line(1, 0.0, attitude, gyro=(0.0, 0.0, 0.1)),
        sample_line(2, 0.05, attitude, gyro=(0.0, 0.0, 0.1)),
        sample_line(3, 0.10, attitude, gyro=(0.0, 0.0, 0.1)),
    ]
    clean = frames(exchange(protocol, good_lines))
    assert len(clean) == 3

    malformed = [
        "",  # an empty line is a keepalive, not a reset
        "GARBAGE",
        "XS2 1 0 0 0 0 1 0 0 0 1 0",  # too few fields
        "XS2 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0",  # too many
        "XS2 1 0 nan 0 0 0 1 0 0 0 1 0 6.9e6 0 0",  # non-finite time
        "XL not-a-number 1",  # bad plan header
        "XE",  # end of plan with no plan open
        "XS",  # keyword only
        "X",  # bare version tag
    ]
    dirty = frames(exchange(protocol, good_lines[:2] + malformed + good_lines[2:]))
    assert len(dirty) == 3, f"a parse failure lost a good frame: {dirty}"
    for index, (got, want) in enumerate(zip(dirty, clean, strict=True)):
        # Uptime is a host-injected clock and is excluded; everything else,
        # including the estimate, the torque and the wheel commands, must match.
        assert got[0:20] == want[0:20], (
            f"frame {index} changed after a parse failure:\n got {got[0:20]}\nwant {want[0:20]}"
        )


def test_invalid_config_and_plan_are_refused_loudly(protocol: Path) -> None:
    """A refused configuration must not silently fall back to a default.

    Falling back would make the run look configured while using gains nobody
    chose -- the same class of defect as a hardcoded ``truth_leakage`` flag.
    """
    attitude = quat.from_axis_angle([0.0, 0.0, 1.0], 0.1)
    replies = exchange(
        protocol,
        [
            "XC 0 0.15 0.0028 200 3.27e-6 1675.5",  # non-positive gain
            "XC 0.35 0.15 -1 200 3.27e-6 1675.5",  # negative torque limit
            "XC 0.35",  # too few fields
            "XL 0 65",  # declares 65 segments, past the store's capacity
            config_line(),
            *plan_lines(0.0, [(0, 0.0, 0.5, 100.0, 0.0, 100.0, 0.0)]),
            sample_line(1, 0.0, attitude, gyro=(0.0, 0.0, 0.1)),
        ],
    )
    assert replies[0][0] == E_REFUSAL and replies[0][1][0] != OK, "a zero gain was accepted"
    assert replies[1][0] == E_REFUSAL and replies[1][1][0] != OK, "negative torque limit accepted"
    assert replies[2][0] == E_REFUSAL and replies[2][1][0] != OK, "truncated config accepted"
    assert replies[3][0] == E_REFUSAL and replies[3][1][0] != OK, "oversized plan accepted"
    # The session still works on the valid configuration.
    got = frames(replies)
    assert len(got) == 1 and got[0][F_STATUS] == OK, f"session did not survive: {replies}"


def test_health_counters_report_rx_drops_and_stack_headroom(protocol: Path) -> None:
    """Dropped input and exhausted stack must be visible, not inferred.

    The device is told how many lines were dropped; the health reply must carry
    that number. A run that reports a clean verdict while the link was losing
    frames is the failure this makes observable.
    """
    kind, fields = exchange(protocol, ["XD 7 3", "XH"])[-1]
    assert kind == H_HEALTH
    assert fields[H_DROPS] == 7.0, f"RX drops not reported: {fields}"
    assert fields[H_OVERFLOWS] == 3.0, f"RX overflows not reported: {fields}"
    assert fields[H_HEADROOM] > 0.0, "stack headroom must be reported"


def test_health_counters_are_set_from_the_device_side_too(protocol: Path) -> None:
    """Both counters must be settable independently and reported back.

    ``XD`` sets drops and overflows together (what a host batch reports at the
    end of a run); ``XO`` sets overflows alone, which is the device's own RX
    queue-full counter. Reporting only the host's view would let a firmware-side
    queue overflow hide.
    """
    _, both = exchange(protocol, ["XD 7 3", "XH"])[-1]
    assert both[H_DROPS] == 7.0 and both[H_OVERFLOWS] == 3.0, both
    _, only = exchange(protocol, ["XO 4", "XH"])[-1]
    assert only[H_DROPS] == 0.0 and only[H_OVERFLOWS] == 4.0, only


def test_uptime_is_reported_from_the_device_clock(protocol: Path) -> None:
    """Boot/health uptime must come back, so a stuck device is distinguishable."""
    kind, fields = exchange(protocol, ["XGS 12345", "XH"])[-1]
    assert kind == H_HEALTH
    assert fields[H_UPTIME] == 12345.0, f"uptime not reported: {fields}"


def test_plan_transaction_is_all_or_nothing(protocol: Path) -> None:
    """A plan is committed by XE or not at all.

    A half-loaded plan would track a partial maneuver and then report a small
    final error for a schedule the vehicle was never actually given. Sending
    segments and then refusing to commit must leave the previous plan in force.
    """
    attitude = quat.from_axis_angle([0.0, 0.0, 1.0], 0.0)
    lines = [
        config_line(),
        *plan_lines(0.0, [(0, 0.0, 0.5, 100.0, 0.0, 100.0, 0.0)]),
        # Open a second plan, add one segment, then abandon it: the declared
        # count is 2 and only 1 arrives, so XE must refuse.
        "XL 0 2",
        "XS 0 0.0 0.5 100.0 0.0 100.0 0.0",
        "XE",
        # Sample well into the committed plan, where it commands toward 0.5 rad.
        # At t=0 the plan's angle IS 0, so an early sample would see no error and
        # the assertion below could not tell a retained plan from a lost one.
        sample_line(1, 25.0, attitude, gyro=(0.0, 0.0, 0.0)),
    ]
    replies = exchange(protocol, lines)
    # The abandoned XE is the second-to-last line.
    assert replies[-2][0] == E_REFUSAL and replies[-2][1][0] != OK, (
        f"a short plan was committed: {replies[-2]}"
    )
    # The previously committed plan is still authoritative: the vehicle is still
    # tracking toward 0.5 rad, so a real wheel command appears.
    got = frames(replies)
    assert len(got) == 1 and got[0][F_STATUS] == OK
    assert any(abs(v) > 0.0 for v in got[0][F_WHEEL]), f"the old plan was lost: {got[0]}"


def test_stateless_s2_protocol_is_unchanged(protocol: Path) -> None:
    """The existing S2/R path must keep working alongside the new protocol.

    The closed-loop protocol is *additional*. Silently repurposing S2 would
    invalidate an already-passing regression, so the old frame is asserted here
    to still parse and still produce a seq-linked reply.
    """
    legacy = (
        "S2 1 1700000000000 0.0 0.0 0.01 20.0 1.0 -2.0 0.0 0.0 1.0 "
        "6900000.0 1000.0 -2000.0"
    )
    replies = exchange(protocol, [legacy, legacy.replace("S2 1", "S2 2")])
    assert len(replies) == 2
    for index, (kind, fields) in enumerate(replies, start=1):
        assert kind == Y_FRAME, f"legacy S2 frame {index} did not produce a Y reply"
        assert fields[F_SEQ] == index, f"legacy reply is not seq-linked: {fields}"
        assert fields[F_STATUS] == OK, f"legacy S2 frame {index} was refused: {fields}"
    # An unknown version tag is refused rather than guessed at.
    bad = exchange(protocol, ["S9 1 0 0 0 0 1 0 0 0 1 0 0 0 0"])[0]
    assert bad[0] == E_REFUSAL and bad[1][0] != OK, f"an unknown frame was accepted: {bad}"


def test_in_place_plan_validation_preserves_pending_metadata(protocol: Path) -> None:
    for segment, count, expected_status in [("XS 0 0 .5 1 .5 0 .5", 2, 4),
                                             ("XS 0 0 .5 -1 .5 0 .5", 1, 4),
                                             ("XS 0 0 .5 1 .5 0 .5", 1, 0)]:
        replies = exchange(protocol, [f"XL 0 {count}", segment, "TEST_PENDING",
                                      "XE", "TEST_PENDING"])
        assert replies[-2][1][0] == expected_status
        assert replies[-1] == replies[-3]


def test_64_segment_capacity_and_65_refusal_preserve_committed_plan(protocol: Path) -> None:
    segments = [(i, i * .01, (i + 1) * .01, 1.0, .5, 0.0, .5)
                for i in range(64)]
    committed = [config_line(), *plan_lines(0.0, segments)]
    sample = sample_line(1, 64.0, quat.from_axis_angle([0, 0, 1], 0.0))
    clean = exchange(protocol, [*committed, sample])
    assert all(kind == E_REFUSAL and fields[0] == OK
               for kind, fields in clean[:-1])
    assert frames(clean)[0][F_STATUS] == OK
    assert any(abs(v) > 0 for v in frames(clean)[0][F_WHEEL])
    dirty = exchange(protocol, [*committed, "XL 0 65", "XE", sample])
    assert dirty[-3][1][0] != OK
    assert dirty[-2][1][0] != OK
    assert frames(dirty)[0][:20] == frames(clean)[0][:20]
    # A full pending store rejects one further append without overwriting it;
    # an incomplete transaction cannot replace the previously committed plan.
    partial = exchange(protocol, [*committed, "XL 0 64", "XS 0 0 .01 1 .5 0 .5",
                                  "XE", sample])
    assert partial[-2][1][0] != OK
    assert frames(partial)[0][:20] == frames(clean)[0][:20]
    overflow = exchange(protocol, [*committed, "XL 0 64", *plan_lines(0.0, segments)[1:-1],
                                   "XS 64 .64 .65 1 .5 0 .5", "XE", sample])
    assert overflow[-3][1][0] != OK
    assert overflow[-2][1][0] == OK
    assert frames(overflow)[0][:20] == frames(clean)[0][:20]


def test_xz_clears_mission_state_but_preserves_configuration_and_health(protocol):
    attitude = quat.from_axis_angle([0, 0, 1], .2)
    populated = [config_line(.4, .2),
                 *plan_lines(.1, [(0, .1, .5, 100, 0, 100, 0)]),
                 "XR 1 0 0 0 .9238795325 0 .3826834324 0 60", "XE",
                 sample_line(1, 0, attitude, gyro=(0, 0, .1)),
                 sample_line(2, .05, attitude, gyro=(0, 0, .1)),
                 sample_line(3, 30, attitude, eclipse=True),
                 "XL .7 2", "XS 7 .7 .9 1 .5 0 .5",
                 "XR 1 0 0 0 .9238795325 0 .3826834324 0 90",
                 "XD 7 3", "XGS 12345"]
    probes = ["TEST_DYNAMIC", "TEST_PLANS", "TEST_CONFIG", "XH"]
    replies = exchange(protocol, [*populated, *probes, "XZ", *probes])
    before, after = replies[-9:-5], replies[-4:]
    assert before[0][1][6] == 1  # real absolute fix existed
    assert any(abs(v) > 0 for v in before[0][1][9:12])
    assert before[0][1][-1] == 1  # stale state existed
    assert before[1][1][5] == 1  # active reference was authoritative
    assert before[1][1][-1] == 1  # pending reference transaction existed
    fresh = exchange(protocol, [config_line(.4, .2), *probes])
    assert after[:3] == fresh[-4:-1]
    assert after[3][1][:5] == before[3][1][:5]
    assert after[3][1][-1] == 0
    assert replies[-5] == (E_REFUSAL, [0, -1, 0])
    # Idempotence and refusal to fly a cleared plan without a new commit.
    got = exchange(protocol, [*populated, "XZ", "XZ",
                              sample_line(4, 0, attitude)])
    assert got[-3] == got[-2] == (E_REFUSAL, [0, -1, 0])
    assert got[-1][0] == E_REFUSAL and got[-1][1][0] != 0


def test_xz_clears_every_stored_segment_and_reference_endpoint(protocol):
    populated = [*plan_lines(.1, [(7, .1, .5, 100, 10, 80, 10)]),
                 "XR .9238795325 .3826834324 0 0 .9238795325 0 .3826834324 0 60", "XE",
                 "XL .7 2", "XS 7 .7 .9 1 .5 0 .5",
                 "XR .9238795325 .3826834324 0 0 .9238795325 0 .3826834324 0 90"]
    replies = exchange(protocol, [*populated, "TEST_STORES_CLEAR", "XZ",
                                  "TEST_STORES_CLEAR"])
    assert replies[-3] == (7, [0])  # non-empty stores were exercised
    assert replies[-1] == (7, [1])  # every field, not just loaded/count flags


@pytest.mark.parametrize("bad", ["XZ extra", "XZ 0", "XZnan", "XZZ", "XZ nan"])
def test_malformed_xz_does_not_reset_mission(protocol, bad):
    setup = [config_line(), *plan_lines(0, [(0, 0, .5, 100, 0, 100, 0)]),
             sample_line(1, 0, quat.from_axis_angle([0, 0, 1], .2))]
    probes = ["TEST_DYNAMIC", "TEST_PLANS", "TEST_CONFIG", "XH"]
    replies = exchange(protocol, [*setup, *probes, bad, *probes])
    assert replies[-5][0] == E_REFUSAL and replies[-5][1][0] != 0
    assert replies[-9:-5] == replies[-4:]


def test_live_config_plan_and_reference_updates_preserve_dynamics(protocol):
    attitude = quat.from_axis_angle([0, 0, 1], .2)
    setup = [config_line(), *plan_lines(0, [(0, 0, .5, 100, 0, 100, 0)]),
             sample_line(1, 0, attitude), sample_line(2, .05, attitude, eclipse=True)]
    updates = [config_line(.4, .2),
               *plan_lines(.1, [(1, .1, .7, 100, 0, 100, 0)]),
               "XR 1 0 0 0 .9238795325 0 .3826834324 0 60", "XE"]
    replies = exchange(protocol, [*setup, "TEST_DYNAMIC", *updates, "TEST_DYNAMIC",
                                  sample_line(3, 0, attitude)])
    snapshots = [fields for kind, fields in replies if kind == 4]
    assert any(abs(v) > 0 for v in snapshots[0][9:12])
    assert snapshots[0] == snapshots[1]
    assert replies[-1][0] == E_REFUSAL and replies[-1][1][0] != 0


def test_first_gyro_only_sample_after_xz_cannot_inherit_absolute_fix(protocol):
    setup = [config_line(), *plan_lines(0, [(0, 0, .5, 100, 0, 100, 0)])]
    attitude = quat.from_axis_angle([0, 0, 1], .2)
    got = exchange(protocol, [*setup, sample_line(1, 0, attitude), "XZ",
                              *setup, sample_line(2, 0, attitude, eclipse=True)])
    frame = frames(got)[-1]
    assert frame[F_HAS_FIX] == 0 and frame[F_STATUS] != 0
    assert frame[F_Q] == [1, 0, 0, 0]
    assert frame[F_WHEEL] == frame[F_TAU] == [0, 0, 0]


@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("time", [-1, 0, .05])
def test_timestamp_refusal_is_transactional_even_after_stale_gap(protocol, stale, time):
    attitude = quat.from_axis_angle([0, 0, 1], .2)
    setup = [config_line(), *plan_lines(0, [(0, 0, .5, 100, 0, 100, 0)]),
             sample_line(1, 0, attitude), sample_line(2, .05, attitude)]
    if stale:
        setup.append(sample_line(3, 30, attitude, eclipse=True))
        time = 30 if time == .05 else time
    probes = ["TEST_DYNAMIC", "TEST_PLANS", "XH"]
    replies = exchange(protocol, [*setup, *probes, sample_line(4, time, attitude), *probes])
    assert replies[-4][0] == E_REFUSAL and replies[-4][1][0] != 0
    assert replies[-7:-4] == replies[-3:]


def test_overlong_line_is_refused_not_truncated(protocol: Path) -> None:
    """A line longer than the buffer must be refused.

    Truncating it would produce a syntactically valid frame with different
    numbers -- the failure mode where the first case in a capture file works
    and every later one fails for an unrelated-looking reason.
    """
    overlong = "XS2 1 0 0 " + " ".join(["0.5"] * 80)
    kind, fields = exchange(protocol, [overlong])[0]
    assert kind == E_REFUSAL and fields[0] != OK, f"an over-long line was parsed: {fields}"
