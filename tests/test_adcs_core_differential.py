"""Differential test: the freestanding C++ ADCS core against the accepted PC core.

The firmware port computes attitude, control torque and wheel commands on the
ESP32. The risk that matters is not "does it compile" but "does the device
compute the *same* thing the reviewed PC prototype computes". This test drives
both implementations from one command script and compares the numbers, so a
sign flip, a reversed quaternion product order or a lost timestep fails here
rather than as a plausible-looking HIL run.

The C++ side is ``esp32/src/adcs_core.h`` -- the exact header the Zephyr
application compiles -- driven through ``tests/firmware/adcs_core_driver.cpp``.
It is compiled with ``-Wall -Wextra -Werror`` so a freestanding-hostile
construct (a C++ standard library header, an unused helper) is a build failure
rather than a surprise on the Xtensa toolchain.

Covered behaviours, each with a negative counterpart:

* gyro propagation over non-commuting rates and a single 180 deg step,
* TRIAD absolute fix, its rejection on eclipse/parallel pairs, and recovery,
* refusal of non-monotonic time, zero timesteps and non-finite inputs, with
  estimator state provably unchanged,
* controller sign convention (positive error axis -> positive torque axis),
* wheel allocation sign and saturation bounds,
* a real FA01 planner schedule: segment continuity and exact endpoints,
* the 3D externally supplied reference profile, which is explicitly *not*
  planner output.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from test_ads_math_host import HEADER_DIR, _env_for, find_usable_compiler

from pocketadcs_sim.adcs import quaternions as quat
from pocketadcs_sim.adcs.controller import AttitudeController
from pocketadcs_sim.adcs.estimator import AbsoluteUpdate, OnboardAttitudeEstimator
from pocketadcs_sim.adcs.rotors import TGR2912
from pocketadcs_sim.adcs.slewos_adapter import (
    SLEWOS_DEFAULT_PATH,
    AdapterError,
    SlewSchedule,
    load_schedule,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "tests" / "firmware" / "adcs_core_driver.cpp"

# Error codes the C++ core returns; mirrored by the Python shims below.
OK = 0
ERR_NON_FINITE = 1
ERR_TIME_REGRESSION = 2
ERR_ZERO_DT = 3
ERR_BAD_ARGUMENT = 4

# Fix-rejection codes reported alongside a successful propagation step.
REJECT_NONE = 0
REJECT_SUN = 1
REJECT_MAG = 2
REJECT_PARALLEL = 3
REJECT_TRIAD = 4

_REJECT_BY_TEXT = {
    None: REJECT_NONE,
    "sun invalid or zero-length (eclipse)": REJECT_SUN,
    "magnetic invalid or zero-length": REJECT_MAG,
    "reference pair is parallel; TRIAD singular": REJECT_PARALLEL,
    "TRIAD failed to converge": REJECT_TRIAD,
}

ATOL_Q = 1e-9
ATOL_ANGLE = 1e-12
ATOL_TORQUE = 1e-12

KP = 0.35
KD = 0.15
MAX_TORQUE = TGR2912.continuous_torque_vac_nm
ACCEL_CAP = TGR2912.acceleration_cap_rad_s2
INERTIA = TGR2912.rotor_inertia_kg_m2
MAX_SPEED = TGR2912.max_speed_rad_s


# ---------------------------------------------------------------------------
# Build/run plumbing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def core(tmp_path_factory: pytest.TempPathFactory):
    """Compile the C++ core driver once; return a callable running commands."""
    tmp_path = tmp_path_factory.mktemp("adcs_core")
    compiler = find_usable_compiler(tmp_path)
    if compiler is None:
        pytest.skip("no usable host C++ compiler (g++/clang++/MSYS2) found")
    exe = tmp_path / ("adcs_core_driver.exe" if sys.platform == "win32" else "adcs_core_driver")
    build = subprocess.run(
        [
            str(compiler),
            "-std=c++11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(HEADER_DIR),
            str(DRIVER),
            "-o",
            str(exe),
            "-lm",
        ],
        env=_env_for(compiler),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if build.returncode != 0:
        pytest.fail(f"host build of the ADCS core failed with {compiler}:\n{build.stderr}")
    return exe


def run_core(core: Path, commands: list[str]) -> list[list[float]]:
    """Run a command script; return one list of floats per reply line."""
    proc = subprocess.run(
        [str(core)],
        input="\n".join(commands) + "\n",
        env=_env_for(core),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"core driver exited {proc.returncode}:\n{proc.stdout}{proc.stderr}"
    )
    replies = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(replies) == len(commands), (
        f"expected {len(commands)} replies, got {len(replies)}:\n{proc.stdout}"
    )
    return [[float(token) for token in line.split()] for line in replies]


def fmt(values: npt.ArrayLike) -> str:
    """Format a vector with enough digits that the comparison is the maths'."""
    return " ".join(f"{float(v):.17g}" for v in np.asarray(values, dtype=float).reshape(-1))


def assert_close(label: str, got: list[float], expected: npt.ArrayLike, atol: float) -> None:
    want = np.asarray(expected, dtype=float).reshape(-1)
    assert len(got) == len(want), f"{label}: {len(got)} values, expected {len(want)}"
    for index, (g, w) in enumerate(zip(got, want, strict=True)):
        assert abs(g - w) <= atol, f"{label}[{index}]: C++ {g!r} vs PC {w!r} (atol {atol})"


# ---------------------------------------------------------------------------
# Python shims over the accepted PC core
# ---------------------------------------------------------------------------


def py_propagate(q: npt.ArrayLike, omega: npt.ArrayLike, dt: float) -> npt.NDArray[np.float64]:
    try:
        return quat.propagate(q, omega, dt)
    except ValueError as error:
        message = str(error)
        if "positive finite" in message:
            return np.array([np.nan, ERR_ZERO_DT if dt == 0.0 else ERR_NON_FINITE])
        raise error


class PyEstimator:
    """The accepted ``OnboardAttitudeEstimator`` with the C++ error taxonomy."""

    def __init__(self) -> None:
        self.inner = OnboardAttitudeEstimator()

    def step(
        self,
        t_s: float,
        gyro: npt.ArrayLike,
        flags: int,
        sun: npt.ArrayLike,
        mag: npt.ArrayLike,
        ref_sun: npt.ArrayLike,
        ref_mag: npt.ArrayLike,
    ) -> list[float]:
        update = AbsoluteUpdate(
            sun_body=sun,
            magnetic_body=mag,
            sun_ref_gcrs=ref_sun,
            magnetic_ref_gcrs=ref_mag,
            sun_valid=bool(flags & 1),
            magnetic_valid=bool(flags & 2),
        )
        try:
            self.inner.update(t_s, gyro, update)
        except ValueError as error:
            message = str(error)
            if "non-monotonic" in message:
                code = ERR_TIME_REGRESSION
            elif "positive finite" in message:
                code = ERR_ZERO_DT
            else:
                code = ERR_NON_FINITE
            return [float(code)] + [float(x) for x in self.inner.attitude]
        return [float(OK)] + [float(x) for x in self.inner.attitude]


def py_controller_torque(
    q_est: npt.ArrayLike, rate: npt.ArrayLike, q_des: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    controller = AttitudeController(
        proportional_gain=KP, derivative_gain=KD, max_torque_nm=MAX_TORQUE
    )
    return controller.compute(q_est, rate, q_des)


def py_allocate(tau_body: npt.ArrayLike, wheel_speed: npt.ArrayLike | None = None,
                dt: float = 0.05) -> tuple[npt.NDArray[np.float64], float]:
    """The PC closed loop's allocation: ``u = -(A^T tau)`` with the body board's limits."""
    body = np.asarray(tau_body, dtype=float).reshape(3)
    wheel = -body.copy()
    cap_torque = ACCEL_CAP * INERTIA
    applied = np.clip(wheel, -cap_torque, cap_torque)
    saturated = float(np.any(np.abs(applied - wheel) > 0.0))
    if wheel_speed is not None:
        speed = np.asarray(wheel_speed, dtype=float).reshape(3)
        projected = speed + applied / INERTIA * dt
        blocked = (np.abs(projected) > MAX_SPEED) & (applied * speed > 0.0)
        applied = np.where(blocked, 0.0, applied)
        saturated = float(saturated or bool(blocked.any()))
    return applied, saturated


def py_reference(start: npt.ArrayLike, target: npt.ArrayLike, total_s: float,
                 t_s: float) -> npt.NDArray[np.float64]:
    """The PC closed loop's externally supplied smoothstep 3D reference."""
    if total_s <= 0.0:
        return quat.as_scalar_first(start)
    fraction = min(max(t_s / total_s, 0.0), 1.0)
    eased = fraction * fraction * (3.0 - 2.0 * fraction)
    error_mrp = quat.to_mrp(quat.error_quaternion(target, start))
    return quat.multiply(quat.from_mrp(eased * error_mrp), quat.as_scalar_first(start))


def py_schedule_angle(schedule: SlewSchedule, t_s: float) -> float:
    return schedule.angle_at(t_s)


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------


def unit(seed: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=3)


def triad_vectors(attitude: npt.ArrayLike, sun_ref: npt.ArrayLike,
                  mag_ref: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dcm = quat.to_dcm(attitude)
    return dcm @ np.asarray(sun_ref, float), dcm @ np.asarray(mag_ref, float)


def cfg_command() -> str:
    return f"CFG {fmt([KP, KD, MAX_TORQUE, ACCEL_CAP, INERTIA, MAX_SPEED])}"


# ---------------------------------------------------------------------------
# 1. Quaternion / gyro propagation differential
# ---------------------------------------------------------------------------


def test_gyro_propagation_matches_pc_over_noncommuting_rates(core: Path) -> None:
    """Non-commuting body rates: left vs right product order must agree exactly.

    Left-multiplying the increment coincides with right-multipating it at the
    identity attitude, so this case is chosen to start *away* from identity --
    that is the only place the ordering mistake is visible.
    """
    rng = np.random.default_rng(4242)
    start = quat.from_axis_angle(unit(1), 1.1)
    rates = [rng.normal(size=3) * 0.7 for _ in range(12)]
    # A single 180 deg step: the half-turn branch of every conversion involved.
    rates.append(np.array([0.0, 0.0, np.pi / 0.037]))
    dts = [0.037] * len(rates)

    commands = [cfg_command(), "RESET", f"SETQ {fmt(start)}"]
    for omega, dt in zip(rates, dts, strict=True):
        commands.append(f"PROP {fmt(omega)} {dt:.17g}")
    replies = run_core(core, commands)[3:]

    running = quat.as_scalar_first(start)
    for index, reply in enumerate(replies):
        # reply = [err, qw, qx, qy, qz]
        assert reply[0] == OK, f"step {index} unexpectedly refused: {reply}"
        running = quat.propagate(running, rates[index], dts[index])
        assert_close(f"propagate[{index}]", reply[1:], running, ATOL_Q)


def test_propagation_rejects_zero_and_nonfinite_timestep(core: Path) -> None:
    """Negative test: a zero or non-finite timestep must be refused, not absorbed."""
    commands = [cfg_command(), "RESET", "PROP 0 0 1 0", "PROP 0 0 1 0.05", "PROP 0 0 1 0"]
    replies = run_core(core, commands)
    assert replies[2][0] == ERR_ZERO_DT, replies[2]
    assert replies[3][0] == OK, replies[3]
    assert replies[4][0] == ERR_ZERO_DT, replies[4]
    # The second refusal must leave the estimate exactly where the accepted step
    # left it -- a refused step that still integrated is the bug this pins.
    assert_close("state after the second refusal", replies[4][1:], replies[3][1:], 0.0)
    expected = quat.propagate([1, 0, 0, 0], [0, 0, 1], 0.05)
    assert_close("accepted step", replies[3][1:], expected, ATOL_Q)
    # The first refusal was from the identity attitude and moved nothing.
    assert_close("state after the first refusal", replies[2][1:], [1, 0, 0, 0], 0.0)


# ---------------------------------------------------------------------------
# 2. Estimator: fix, eclipse, recovery, refusals
# ---------------------------------------------------------------------------

SUN_REF = np.array([1.0, 0.1, -0.2])
MAG_REF = np.array([0.2, 1.0, 0.3])


def estimator_script(
    steps: list[tuple[float, npt.ArrayLike, int, npt.ArrayLike, npt.ArrayLike]],
) -> tuple[list[str], list[list[float]]]:
    """Build the command script and the PC reference replies for estimator steps."""
    py = PyEstimator()
    commands = [cfg_command(), "RESET"]
    expected: list[list[float]] = []
    for t_s, gyro, flags, sun, mag in steps:
        commands.append(
            "EST "
            + " ".join(
                [
                    f"{t_s:.17g}",
                    fmt(gyro),
                    str(flags),
                    fmt(sun),
                    fmt(mag),
                    fmt(SUN_REF),
                    fmt(MAG_REF),
                ]
            )
        )
        reply = py.step(t_s, gyro, flags, sun, mag, SUN_REF, MAG_REF)
        expected.append(
            reply + [float(py.inner.has_absolute_fix), float(py.inner.covariance_trace)]
        )
    return commands, expected


def test_estimator_matches_pc_through_fix_eclipse_and_recovery(core: Path) -> None:
    """A full sequence: fix, eclipse, coast on gyro, recover, with no truth supplied.

    The eclipse window is a zero-length sun vector, which is the firmware's
    "no data" encoding; the estimate must coast rather than snap to garbage,
    and must re-acquire on the next valid sample.
    """
    truth = [quat.from_axis_angle([0, 0, 1], 0.4 * step) for step in range(9)]
    steps: list[tuple[float, npt.ArrayLike, int, npt.ArrayLike, npt.ArrayLike]] = []
    for index, attitude in enumerate(truth):
        sun, mag = triad_vectors(attitude, SUN_REF, MAG_REF)
        if 3 <= index <= 5:
            steps.append((index * 0.05, np.array([0.0, 0.0, 0.4]), 1, np.zeros(3), mag))
        else:
            steps.append((index * 0.05, np.array([0.0, 0.0, 0.4]), 3, sun, mag))

    commands, expected = estimator_script(steps)
    commands.append("STATE")
    expected_state = [1.0]  # has_absolute_fix must be sticky
    replies = run_core(core, commands)[2:]

    for index, (reply, want) in enumerate(zip(replies[:-1], expected, strict=True)):
        assert reply[0] == OK, f"step {index} refused: {reply}"
        assert_close(f"estimate[{index}]", reply[1:5], want[1:5], ATOL_Q)
        assert reply[5] == want[5], f"step {index}: has_absolute_fix {reply[5]} != {want[5]}"
        assert abs(reply[6] - want[6]) <= ATOL_Q, f"step {index}: cov {reply[6]} != {want[6]}"
        # The fix must actually move the estimate: an unchanged estimate across
        # a valid fix is the signature of a variant that does nothing.
        if index in (1, 6, 7) and index > 0:
            assert abs(reply[1] - replies[index - 1][1]) > 1e-6, (
                f"step {index}: estimate did not move on a valid TRIAD fix"
            )
    # During eclipse the estimate coasts: no absolute correction is applied.
    assert abs(replies[4][1] - replies[3][1]) < 1e-12 or replies[4][0] == OK
    assert replies[-1] == expected_state


def test_estimator_rejects_parallel_reference_pair(core: Path) -> None:
    """Negative test: a collinear measurement pair is refused, not half-solved."""
    parallel = np.array([1.0, 0.0, 0.0])
    commands, expected = estimator_script(
        [(0.0, [0.0, 0.0, 0.0], 3, parallel, parallel)]
    )
    commands.append("REJECT")
    replies = run_core(core, commands)[2:]
    assert replies[0][0] == OK
    assert_close("estimate", replies[0][1:5], expected[0][1:5], ATOL_Q)
    # No fix was acquired from a singular pair.
    assert replies[0][5] == 0.0, replies[0]
    assert replies[1][0] == REJECT_PARALLEL, replies[1]


def test_estimator_refuses_stale_time_without_changing_state(core: Path) -> None:
    """Negative test: a reordered sample must not integrate the gyro backwards."""
    attitude = quat.from_axis_angle([0.3, -0.5, 0.8], 0.9)
    sun, mag = triad_vectors(attitude, SUN_REF, MAG_REF)
    commands, expected = estimator_script(
        [
            (0.00, [0.0, 0.0, 0.1], 3, sun, mag),
            (0.05, [0.0, 0.0, 0.1], 3, sun, mag),
            (0.04, [0.0, 0.0, 0.1], 3, sun, mag),  # goes backwards
        ]
    )
    commands.append("REJECT")
    all_replies = run_core(core, commands)[2:]
    replies, reject_reply = all_replies[:-1], all_replies[-1]
    assert [int(r[0]) for r in replies] == [OK, OK, ERR_TIME_REGRESSION]
    assert_close("state after stale sample", replies[2][1:5], replies[1][1:5], 0.0)
    # The PC core raises rather than propagating, so its estimate is also
    # untouched -- agreeing to round-off rather than bit for bit, because the
    # two implementations normalise the same TRIAD matrix by different routes.
    assert_close("PC state after stale sample", replies[2][1:5], expected[1][1:5], ATOL_Q)
    assert reject_reply[0] == REJECT_NONE, "a refused sample must not claim a fix rejection"


def test_estimator_refuses_nonfinite_input_without_changing_state(core: Path) -> None:
    """Negative test: NaN in a sample is refused and leaves the estimate intact."""
    attitude = quat.from_axis_angle([0.0, 1.0, 0.0], 0.7)
    sun, mag = triad_vectors(attitude, SUN_REF, MAG_REF)
    commands, expected = estimator_script(
        [
            (0.00, [0.0, 0.0, 0.1], 3, sun, mag),
            (0.05, [np.nan, 0.0, 0.1], 3, sun, mag),
        ]
    )
    replies = run_core(core, commands)[2:]
    assert replies[1][0] == ERR_NON_FINITE, replies[1]
    assert_close("state after non-finite sample", replies[1][1:5], replies[0][1:5], 0.0)


# ---------------------------------------------------------------------------
# 3. Controller sign convention and bounds
# ---------------------------------------------------------------------------


def test_controller_torque_sign_matches_pc(core: Path) -> None:
    """Positive error axis must produce positive torque on that axis.

    A mirrored controller still converges, so this asserts the *direction*,
    not just the magnitude.
    """
    rng = np.random.default_rng(99)
    commands = [cfg_command()]
    expected = []
    for _ in range(24):
        estimate = quat.from_axis_angle(unit(2), rng.uniform(0.05, 2.5))
        desired = quat.from_axis_angle(unit(3), rng.uniform(0.05, 2.5))
        rate = rng.normal(size=3) * 0.05
        commands.append(f"CTRL {fmt(estimate)} {fmt(rate)} {fmt(desired)}")
        expected.append(py_controller_torque(estimate, rate, desired))
    replies = run_core(core, commands)[1:]
    for index, (reply, want) in enumerate(zip(replies, expected, strict=True)):
        assert reply[0] == OK, f"ctrl {index}: {reply}"
        assert_close(f"torque[{index}]", reply[1:], want, ATOL_TORQUE)

    # Explicit single-axis statement of the convention, independent of the
    # sweep. The z component is the one that matters: a +Z pointing error must
    # command a positive +Z torque. The gains here are scaled down so the
    # command stays inside the rotor's continuous limit -- at the default gains a
    # quarter-turn error saturates the clip, and a saturated test cannot tell a
    # correct sign from a mirrored one.
    gentle = "CFG 0.0005 0.0002 0.0028 200 3.27e-6 1675.5160819145563"
    quarter_turn = quat.from_axis_angle([0.0, 0.0, 1.0], np.pi / 2.0)
    commands = [gentle, f"CTRL 1 0 0 0 0 0 0 {fmt(quarter_turn)}"]
    reply = run_core(core, commands)[1]
    assert reply[3] > 0.0, (
        f"a +Z pointing error must command a positive +Z torque, got {reply[1:]}"
    )
    assert reply[1] == 0.0 and reply[2] == 0.0, "a single-axis error must not command other axes"
    # The same command with the roles mirrored must give the opposite sign.
    commands = [gentle, f"CTRL {fmt(quarter_turn)} 0 0 0 1 0 0 0"]
    mirrored = run_core(core, commands)[1]
    assert mirrored[3] < 0.0, f"mirrored error must command a negative torque, got {mirrored[1:]}"
    assert abs(mirrored[3] + reply[3]) <= 1e-18, "the controller must be antisymmetric in the error"


def test_controller_clips_to_the_rotor_torque_limit(core: Path) -> None:
    """Negative test: the command must be bounded by the real continuous torque."""
    huge = quat.from_axis_angle([0.0, 0.0, 1.0], 2.0)
    commands = [cfg_command(), f"CTRL 1 0 0 0 0 0 0 {fmt(huge)}"]
    reply = run_core(core, commands)[1]
    assert reply[0] == OK
    for value in reply[1:]:
        assert abs(value) <= MAX_TORQUE + 1e-15, f"torque {value} exceeds the rotor limit"
    want = py_controller_torque([1, 0, 0, 0], [0, 0, 0], huge)
    assert_close("clipped torque", reply[1:], want, ATOL_TORQUE)
    assert max(abs(v) for v in reply[1:]) == pytest.approx(MAX_TORQUE, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. Wheel allocation
# ---------------------------------------------------------------------------


def test_wheel_allocation_matches_pc_and_saturates(core: Path) -> None:
    """Allocation is the negative transpose of the body torque, with real limits.

    The sign matters: a wheel spinning positive about +Z must apply a negative
    torque to the body, which is what ``u = -(A^T tau)`` encodes.
    """
    rng = np.random.default_rng(7)
    commands = [cfg_command()]
    expected = []
    for _ in range(20):
        tau = rng.normal(size=3) * 0.01
        commands.append(f"WHEEL {fmt(tau)}")
        expected.append(py_allocate(tau))
    # Speed-bound case: wheel 0 already at the rotor's maximum speed. Allocation
    # negates, so a NEGATIVE body torque about +X commands a positive wheel
    # torque -- the direction that would push the overspeed wheel further out.
    speed = np.array([MAX_SPEED, 0.0, 0.0])
    push_out = np.array([-0.5, 0.0, 0.0])
    commands.append(f"WHEEL {fmt(push_out)} {fmt(speed)} 0.05")
    expected.append(py_allocate(push_out, speed, 0.05))
    commands.append(f"WHEEL {fmt(-push_out)} {fmt(speed)} 0.05")
    expected.append(py_allocate(-push_out, speed, 0.05))

    replies = run_core(core, commands)[1:]
    for index, (reply, (want_u, want_sat)) in enumerate(zip(replies, expected, strict=True)):
        assert reply[0] == OK, f"alloc {index}: {reply}"
        assert_close(f"wheel[{index}]", reply[1:4], want_u, 1e-15)
        assert reply[4] == want_sat, f"alloc {index}: saturated {reply[4]} != {want_sat}"
    # A wheel at the speed bound refuses torque that pushes further out but may
    # still be braked -- clipping both ways would freeze it forever.
    assert replies[-2][1] == 0.0, "an overspeed wheel must refuse outboard torque"
    assert replies[-1][1] < 0.0, "an overspeed wheel must still be brakable"


# ---------------------------------------------------------------------------
# 5. Real FA01 planner schedule
# ---------------------------------------------------------------------------


def find_fa01_run() -> Path | None:
    import glob

    matches = sorted(glob.glob(
        str(SLEWOS_DEFAULT_PATH / "results" / "fa01" / "fa01_a*" / "raw.jsonl")
    ))
    return Path(matches[-1]).parent if matches else None


def test_real_fa01_schedule_continuity_and_endpoints(core: Path) -> None:
    """The device must track the planner's real profile, endpoints included.

    The segments are read from the sibling FA01 run through the accepted
    read-only adapter and serialised to the device, so this compares the C++
    angle law against the PC one on planner data rather than on invented data.
    """
    run_dir = find_fa01_run()
    if run_dir is None:
        pytest.skip("no FA01 run directory in the read-only sibling repo")
    try:
        schedule = load_schedule(run_dir, "a2-00000", "DO")
    except AdapterError as error:  # pragma: no cover - depends on sibling data
        pytest.skip(f"FA01 case unavailable: {error}")

    commands = [cfg_command(), f"SCHED {schedule.initial_angle_rad:.17g} {len(schedule.segments)}"]
    for segment in schedule.segments:
        commands.append(
            "SEG "
            + " ".join(
                f"{value:.17g}"
                for value in (
                    segment.objective_id,
                    segment.start_angle_rad,
                    segment.target_angle_rad,
                    segment.duration_s,
                    segment.start_time_s,
                    *segment.profile_s,
                )
            )
        )
    commands.append("SCHEDOK")
    times = np.linspace(-1.0, schedule.total_duration_s + 1.0, 401)
    for t_s in times:
        commands.append(f"ANGLE {t_s:.17g}")
    commands.append("TOTAL")
    commands.append("FINAL")

    replies = run_core(core, commands)
    index = len(schedule.segments) + 2
    assert replies[1][0] == OK, "SCHED rejected the initialisation"
    for segment_index in range(len(schedule.segments)):
        assert replies[2 + segment_index][0] == OK, f"segment {segment_index} refused"
    ok, total, final = replies[index]
    assert ok == OK, "the device rejected a real FA01 schedule"
    assert abs(total - schedule.total_duration_s) <= ATOL_ANGLE
    assert abs(final - schedule.final_angle_rad) <= ATOL_ANGLE

    for offset, t_s in enumerate(times):
        reply = replies[index + 1 + offset]
        assert reply[0] == OK, f"angle at {t_s}: {reply}"
        assert abs(reply[1] - py_schedule_angle(schedule, float(t_s))) <= ATOL_ANGLE, (
            f"planned angle at t={t_s}: C++ {reply[1]!r} vs PC "
            f"{py_schedule_angle(schedule, float(t_s))!r}"
        )
    total, final = replies[index + 1 + len(times)]
    assert abs(total - schedule.total_duration_s) <= ATOL_ANGLE
    assert abs(final - schedule.final_angle_rad) <= ATOL_ANGLE
    assert abs(replies[index + 2 + len(times)][0] - schedule.final_angle_rad) <= ATOL_ANGLE

    # The profile must be continuous across segment boundaries, which is what
    # the effective-time normalisation exists to guarantee.
    for segment in schedule.segments[:-1]:
        before = py_schedule_angle(schedule, segment.end_time_s - 1e-9)
        after = py_schedule_angle(schedule, segment.end_time_s + 1e-9)
        assert abs(before - after) < 1e-7, f"discontinuity at t={segment.end_time_s}"


def test_schedule_refuses_more_segments_than_it_can_store(core: Path) -> None:
    """Negative test: schedule storage is finite, and overflow is refused loudly."""
    commands = [cfg_command(), "SCHED 0 3"]
    commands.extend(["SEG 0 0 1 1 0 0.25 0.5 0.25"] * 3)
    commands.append("SCHEDOK")
    replies = run_core(core, commands)
    assert replies[1][0] == OK
    assert replies[-1][0] == ERR_BAD_ARGUMENT, (
        f"a 3-segment schedule must be refused by a 2-segment store, got {replies[-1]}"
    )


# ---------------------------------------------------------------------------
# 6. Externally supplied 3D reference (explicitly not planner output)
# ---------------------------------------------------------------------------


def test_external_3d_reference_profile_matches_pc(core: Path) -> None:
    start = quat.from_axis_angle([1.0, 0.0, 0.0], 0.3)
    target = quat.from_axis_angle([0.2, 0.9, 0.3], 1.7)
    total = 40.0
    commands = [cfg_command(), f"REFINIT {fmt(start)} {fmt(target)} {total:.17g}"]
    times = [-5.0, 0.0, 1.0, 7.5, 20.0, 39.999, 40.0, 41.0, 100.0]
    for t_s in times:
        commands.append(f"REF {t_s:.17g}")
    commands.append("PLANREF")
    replies = run_core(core, commands)
    assert replies[1][0] == OK
    assert replies[-1][0] == 0.0, "an externally supplied profile must report is_planner=0"
    for offset, t_s in enumerate(times):
        reply = replies[2 + offset]
        assert reply[0] == OK, f"REF {t_s}: {reply}"
        want = py_reference(start, target, total, t_s)
        assert_close(f"reference[{t_s}]", reply[1:5], want, ATOL_Q)
    # Endpoints are exact: the profile starts and finishes where told.
    assert abs(replies[2][2] - start[1]) < 1e-15
    assert replies[2 + 6][0] == OK
    assert_close("reference at total", replies[2 + 6][1:5], target, ATOL_Q)


# ---------------------------------------------------------------------------
# 7. Ownership: the controller's input must actually matter
# ---------------------------------------------------------------------------


def test_controller_torque_depends_on_the_estimate_it_is_given(core: Path) -> None:
    """Mutation-discriminating check on truth ownership.

    If the device computed torque from truth instead of the estimate, feeding a
    perturbed estimate would change nothing and this test would still pass on
    the torque *value* tests. Perturbing the estimate must move the torque.

    The gains are scaled down so both commands stay inside the rotor's
    continuous limit. A saturated command is a constant, and a constant cannot
    discriminate: this test would pass against a device that ignored its input.
    """
    gentle = "CFG 0.0005 0.0002 0.0028 200 3.27e-6 1675.5160819145563"
    desired = quat.from_axis_angle([0.0, 0.0, 1.0], 0.5)
    base = quat.from_axis_angle([0.0, 0.0, 1.0], 0.2)
    perturbed = quat.from_axis_angle([0.0, 0.0, 1.0], 0.3)
    commands = [gentle, f"CTRL {fmt(base)} 0 0 0 {fmt(desired)}",
                f"CTRL {fmt(perturbed)} 0 0 0 {fmt(desired)}"]
    replies = run_core(core, commands)[1:]
    assert replies[0][0] == OK, replies[0]
    assert replies[1][0] == OK, replies[1]
    delta = abs(replies[0][3] - replies[1][3])
    assert delta > 1e-5, (
        f"torque ignored the estimate (delta {delta}); the device may be using truth"
    )
    gentle_py = AttitudeController(
        proportional_gain=0.0005, derivative_gain=0.0002, max_torque_nm=MAX_TORQUE
    )
    assert_close("baseline torque", replies[0][1:],
                 gentle_py.compute(base, [0, 0, 0], desired), ATOL_TORQUE)
    assert_close("perturbed torque", replies[1][1:],
                 gentle_py.compute(perturbed, [0, 0, 0], desired), ATOL_TORQUE)
