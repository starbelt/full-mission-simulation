"""Closed-loop ADCS simulation: plant, sensors, estimator, controller, board.

Data flow, and where truth is and is not allowed
------------------------------------------------
::

    TruthPlant  --(rates)-->  SyntheticSensors  --(measurements)-->  Estimator
        ^                                                                   |
        |                                                                   v
        +------(wheel torque)---- WheelBoard <----(torque)---- AttitudeController

The only two places truth is read are:

1. **Plant initialization** -- the physical vehicle starts in a known state.
2. **Error evaluation** -- after the loop has run, to score the estimate.

The controller is driven by the *estimated* attitude and the *measured* body
rate. The estimator receives a gyro rate and two body-frame vector measurements,
which is exactly what a real ADS receives. Nothing in
:class:`SyntheticSensors` returns truth attitude or truth rate, and
:class:`~pocketadcs_sim.adcs.estimator.OnboardAttitudeEstimator` has no
parameter through which truth could enter -- so the ownership rule is enforced by
the interface, not by convention.

That structural argument is not evidence on its own, so each run also
*measures* the property: the attitude actually handed to
:meth:`~pocketadcs_sim.adcs.controller.AttitudeController.compute` is compared
against the plant's truth attitude every step and reported in
``result.truth_cross_check`` and as the ``controller_input_vs_truth_deg`` time
series column. A hardcoded ``truth_leakage: bool`` used to sit in this record; it
was a dataclass default no code path ever set, so it asserted nothing while
reading to any consumer like a verified measurement. It is gone.

Ideal-first
-----------
With ``sensors="ideal"`` the run demonstrates the closed loop itself: the
estimator's TRIAD fix is exact, so residual error is loop and integration error
only. Noisy runs then add sensor error on top, rather than being tuned away.

Schedules
---------
Two schedule sources, kept explicitly distinct in the output record:

* ``slewos_run_dir`` set -- a real FA01 planner result, tracked as-is.
  The planner is single-axis rest-to-rest; see
  :class:`~pocketadcs_sim.adcs.slewos_adapter.PlannerScope`.
* otherwise -- a simple externally supplied reference angle profile toward
  ``target_attitude``. This exercises 3D pointing but is **not** planner output,
  and the record sets ``is_planner_generated: False``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pocketadcs_sim.adcs import quaternions as quat
from pocketadcs_sim.adcs.controller import AttitudeController
from pocketadcs_sim.adcs.estimator import AbsoluteUpdate, OnboardAttitudeEstimator
from pocketadcs_sim.adcs.plant import PlantConfig, TruthPlant
from pocketadcs_sim.adcs.rotors import WheelSet
from pocketadcs_sim.adcs.slewos_adapter import PlannerScope, load_schedule
from pocketadcs_sim.adcs.wheel_board import WheelBoard

__all__ = [
    "ClosedLoopConfig",
    "ClosedLoopResult",
    "SyntheticSensors",
    "reference_attitude",
    "run_closed_loop",
]

FloatArray = npt.NDArray[np.float64]
_RAD_TO_DEG = 180.0 / np.pi

#: Inertial reference directions. A real vehicle computes these from ephemeris
#: and time; they are knowledge, not a measurement of its own attitude.
SUN_REFERENCE_GCRS = np.array([1.0, 0.1, -0.2])
MAGNETIC_REFERENCE_GCRS = np.array([0.2, 1.0, 0.3])


@dataclass
class ClosedLoopConfig:
    """Configuration for one closed-loop run."""

    duration_s: float = 300.0
    dt_s: float = 0.05
    hub_inertia_kg_m2: tuple[float, float, float] = (0.02, 0.025, 0.03)
    start_attitude: npt.ArrayLike = (1.0, 0.0, 0.0, 0.0)
    target_attitude: npt.ArrayLike | None = None
    proportional_gain: float = 0.35
    derivative_gain: float = 0.15
    wheel_set: WheelSet | None = None
    slewos_run_dir: str | None = None
    slewos_case_id: str | None = None
    slewos_method: str | None = None
    eclipse_windows: tuple[tuple[float, float], ...] = ()
    #: One of "ideal" or "noisy". Ideal is the default so the loop's own error
    #: is visible before sensor error is layered on.
    sensors: str = "ideal"
    output_dir: str | None = None
    tag: str = "closed_loop"

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive and finite")
        for start, end in self.eclipse_windows:
            if end <= start:
                raise ValueError(f"eclipse window {start}-{end} is not ordered")
            if end > self.duration_s:
                raise ValueError(
                    f"eclipse window {start}-{end} ends after the run ({self.duration_s}s)"
                )
        if self.sensors not in ("ideal", "noisy"):
            raise ValueError("sensors must be 'ideal' or 'noisy'")
        if self.proportional_gain <= 0.0 or self.derivative_gain <= 0.0:
            raise ValueError("controller gains must be positive")
        if self.slewos_run_dir and not Path(self.slewos_run_dir).exists():
            raise ValueError(f"slewos_run_dir does not exist: {self.slewos_run_dir}")


@dataclass
class ClosedLoopResult:
    """Everything one run produced, including the configuration used."""

    config: dict[str, object]
    error: dict[str, float]
    wheels: dict[str, float]
    planner: dict[str, object]
    #: The plant's final truth attitude, and its angular distance from the
    #: *commanded target*. ``error["final_attitude_error_deg"]`` is measured
    #: against the reference, which is the right number for tracking a moving
    #: reference but says nothing on its own about whether the vehicle
    #: finished pointing where it was told to.
    plant: dict[str, object] = field(default_factory=dict)
    #: *Measured* truth-ownership evidence, in degrees: how far the attitude
    #: actually handed to the controller is from the plant's truth attitude.
    #:
    #: This replaced a hardcoded ``truth_leakage: bool`` field that no code path
    #: ever assigned. A constant ``False`` in a JSON artifact reads to a
    #: downstream consumer as a verified measurement while being nothing of the
    #: kind, and a test asserting it could not fail. The quantity is real and is
    #: recorded per step in the time series.
    #:
    #: **Reading zero.** In an ``ideal`` run this reads 0.0, and that is the
    #: expected result rather than a sign of leakage: the TRIAD fix is exact at
    #: the sample instant, so the estimate handed to the controller equals truth
    #: to machine precision, and divergence appears only across the following
    #: plant step. It becomes non-zero as soon as the fix is imperfect (noisy
    #: and eclipse runs: 0.54 deg and 1.52 deg). An unexplained 0.0 invites
    #: "fixing" the measurement; ``test_ideal_run_cross_check_is_zero_because_the
    #: _fix_is_exact_at_the_sample`` pins the correct reading instead.
    truth_cross_check: dict[str, float] = field(default_factory=dict)
    series: list[dict[str, float]] = field(default_factory=list)
    output_json_path: str | None = None
    output_csv_path: str | None = None

    def time_series(self) -> list[dict[str, float]]:
        """The recorded per-step time series."""
        return self.series

    def to_dict(self) -> dict[str, object]:
        """A JSON-serializable view of the whole run."""
        return {
            "config": self.config,
            "error": self.error,
            "wheels": self.wheels,
            "planner": self.planner,
            "plant": self.plant,
            "truth_cross_check": self.truth_cross_check,
        }


class SyntheticSensors:
    """Produces the measurements a real ADS would receive.

    Deliberately returns only sensor-shaped quantities: a body rate and two
    body-frame vectors with validity flags. It has no method that returns truth
    attitude, so the estimator downstream cannot be handed the answer even by
    accident.
    """

    def __init__(self, config: ClosedLoopConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(20260930)

    def _in_eclipse(self, t_s: float) -> bool:
        return any(start <= t_s <= end for start, end in self._config.eclipse_windows)

    def measure(
        self, t_s: float, attitude: FloatArray, body_rate: FloatArray
    ) -> tuple[FloatArray, AbsoluteUpdate | None]:
        """Return ``(gyro_body_rate, absolute_update)`` for the current state.

        The gyro is exact in ``ideal`` mode -- no bias, no noise -- so the ideal
        closed loop is a noise-free TRIAD loop and its zero estimator error is
        the expected floor, not evidence that the propagation path is
        unexercised. Only the ``noisy`` branch adds bias and noise, and that is
        the mode in which the bias is what the absolute fix has to remove. The
        gyro-only propagation path is covered directly by the estimator unit
        tests, which inject their own bias.
        """
        if self._config.sensors == "ideal":
            rate = body_rate.copy()
        else:
            bias = np.array([2.0e-4, -1.0e-4, 1.5e-4])
            rate = body_rate + bias + self._rng.normal(0.0, 1.0e-5, size=3)

        dcm = quat.to_dcm(attitude)
        sun_body = dcm @ SUN_REFERENCE_GCRS
        magnetic_body = dcm @ MAGNETIC_REFERENCE_GCRS

        sun_valid = not self._in_eclipse(t_s)
        if not sun_valid:
            # Eclipse: the array reports no signal at all. A zero-length vector
            # is "no data" -- never a real direction.
            sun_body = np.zeros(3)

        if self._config.sensors == "noisy":
            sun_body = sun_body + self._rng.normal(0.0, 2.0e-3, size=3)
            magnetic_body = magnetic_body + self._rng.normal(0.0, 1.0e-3, size=3)

        update = AbsoluteUpdate(
            sun_body=sun_body,
            magnetic_body=magnetic_body,
            sun_ref_gcrs=SUN_REFERENCE_GCRS,
            magnetic_ref_gcrs=MAGNETIC_REFERENCE_GCRS,
            sun_valid=sun_valid,
        )
        return rate, update


def reference_attitude(
    config: ClosedLoopConfig, t_s: float, total_s: float
) -> FloatArray:
    """Externally supplied reference: a smooth slerp from start to target.

    This is *not* planner output. It exists so 3D pointing can be exercised
    while the real planner's single-axis scope stays clearly separate.

    The returned attitude equals ``start_attitude`` at ``t_s == 0`` and
    ``target_attitude`` at ``t_s == total_s``, exactly. Two details make that
    true and both were once wrong:

    * the remaining error is scaled by the eased fraction ``eased``, not
      ``(1 - eased)`` -- the error must *shrink* to zero, not grow from zero;
    * the increment multiplies on the **left** of the start attitude. For the
      ``C_BN`` convention ``to_dcm(a (x) b) == to_dcm(b) @ to_dcm(a)``, so
      ``error_quaternion(target, start) = target (x) conj(start)`` is
      already the body-frame increment to the left of ``start``. Putting it on
      the right only coincides at the identity attitude, and would leave the
      reference 11.2 deg short of the target at ``t_s == total_s``.
    """
    if config.target_attitude is None:
        return quat.as_scalar_first(config.start_attitude)

    start = quat.as_scalar_first(config.start_attitude)
    target = quat.as_scalar_first(config.target_attitude)

    error_mrp = quat.to_mrp(quat.error_quaternion(target, start))
    if total_s <= 0.0:
        # A zero-length schedule has no time to move: hold the start pose.
        return start

    fraction = min(max(t_s / total_s, 0.0), 1.0)
    # Interpolate the error vector rather than the quaternion: it is the
    # well-conditioned form near the identity, and the smoothstep easing gives
    # zero rate and zero acceleration at both ends.
    eased = fraction * fraction * (3.0 - 2.0 * fraction)
    return quat.multiply(quat.from_mrp(eased * error_mrp), start)


def run_closed_loop(config: ClosedLoopConfig) -> ClosedLoopResult:
    """Run one closed-loop simulation and return its measurements."""
    wheel_set = config.wheel_set
    if wheel_set is None:
        from pocketadcs_sim.adcs.rotors import default_wheel_set

        wheel_set = default_wheel_set()

    plant = TruthPlant(
        config=PlantConfig(hub_inertia_kg_m2=config.hub_inertia_kg_m2),
        initial_attitude=config.start_attitude,
    )
    board = WheelBoard(wheel_set=wheel_set)
    estimator = OnboardAttitudeEstimator()
    sensors = SyntheticSensors(config)
    controller = AttitudeController(
        proportional_gain=config.proportional_gain,
        derivative_gain=config.derivative_gain,
    )

    schedule = None
    if config.slewos_run_dir and config.slewos_case_id:
        schedule = load_schedule(
            config.slewos_run_dir, config.slewos_case_id, config.slewos_method
        )
        # Start the plant where the plan starts, so tracking is measured, not
        # confounded by an initial offset the plan never asked for.
        plant.attitude = quat.from_axis_angle(
            np.array([0.0, 0.0, 1.0]), schedule.initial_angle_rad
        )

    total_reference_s = (
        schedule.total_duration_s if schedule is not None else config.duration_s
    )
    # A planner schedule owns the run length. Truncating it would measure
    # tracking against a still-moving reference and leave the vehicle short of
    # the planned attitude, while still reporting a tiny error.
    steps = int(round(total_reference_s / config.dt_s)) + 1

    series: list[dict[str, float]] = []
    max_error = 0.0
    max_error_during_eclipse = 0.0
    max_gap = 0.0
    saturated_steps = 0
    max_wheel_speed = 0.0
    max_controller_input_error = 0.0

    for step in range(steps):
        t_s = step * config.dt_s

        # --- Sensor measurements (the only path from plant to estimator) ---
        gyro_rate, absolute = sensors.measure(t_s, plant.attitude, plant.body_rate)
        estimator.update(time_s=t_s, gyro_body_rad_s=gyro_rate, absolute=absolute)

        # --- Reference attitude ---
        if schedule is not None:
            planned_angle = schedule.angle_at(t_s)
            desired = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), planned_angle)
        else:
            desired = reference_attitude(config, t_s, total_reference_s)

        # --- Control on ESTIMATED state only ---
        # The truth snapshot is taken here, before the plant advances: sampling
        # it after ``step_rk4`` would score this step's command against the
        # *next* step's truth -- close enough to look right, and wrong.
        truth_at_command = np.array(plant.attitude, dtype=float, copy=True)
        body_torque = controller.compute(
            estimated_attitude=estimator.attitude,
            estimated_rate_rad_s=gyro_rate,
            desired_attitude=desired,
        )

        # --- Board: body torque -> wheel torque, honouring real limits ---
        wheel_torque = -(np.asarray(wheel_set.axes, dtype=float).T @ body_torque)
        applied = board.update(wheel_torque, config.dt_s)
        gap = float(np.max(np.abs(board.commanded_torque_nm - board.applied_torque_nm)))
        max_gap = max(max_gap, gap)
        if board.saturated:
            saturated_steps += 1
        max_wheel_speed = max(
            max_wheel_speed, float(np.max(np.abs(board.wheel_speed_rad_s)))
        )

        # --- Plant advances with what the board actually delivered ---
        # RK4 rather than the adaptive solver: a 45-minute planner schedule at a
        # 50 ms control period is ~54k steps, and the adaptive path costs ~3 ms
        # each. tests/test_adcs_plant_fast.py pins the two to 1e-12.
        plant.step_rk4(config.dt_s, applied)

        # --- Scoring only: truth enters here and nowhere else ---
        truth_error = quat.rotation_angle_rad(
            quat.error_quaternion(desired, plant.attitude)
        ) * _RAD_TO_DEG
        estimate_error = quat.rotation_angle_rad(
            quat.error_quaternion(plant.attitude, estimator.attitude)
        ) * _RAD_TO_DEG
        # The measured truth-ownership quantity: how far the attitude the
        # controller *actually received* is from the plant's truth attitude.
        # This reads the controller's own record of the argument, not the
        # caller's intent, so it fails if truth is ever substituted. Zero would
        # mean the controller was handed the answer; non-zero means it was
        # working from the estimate.
        controller_input = controller.last_estimated_attitude
        if controller_input is None:
            # ``compute`` sets this on every call, so reaching here means the
            # value this number depends on does not exist. The two convenient
            # responses are both dishonest: falling back to the caller's own
            # argument scores what the caller believed it sent, and scoring
            # 0.0 reports a perfect cross-check out of a missing measurement --
            # the same defect as the hardcoded ``truth_leakage`` flag this
            # replaced. Fail the run instead.
            raise RuntimeError(
                "controller recorded no attitude argument (last_estimated_attitude "
                f"is None) at t={t_s:.3f} s; the truth cross-check cannot be measured"
            )
        controller_input_error = quat.rotation_angle_rad(
            quat.error_quaternion(truth_at_command, controller_input)
        ) * _RAD_TO_DEG
        max_controller_input_error = max(max_controller_input_error, controller_input_error)
        max_error = max(max_error, truth_error)
        in_eclipse = any(
            start <= t_s <= end for start, end in config.eclipse_windows
        )
        if in_eclipse:
            max_error_during_eclipse = max(max_error_during_eclipse, truth_error)

        series.append(
            {
                "time_s": float(t_s),
                "truth_attitude_error_deg": float(truth_error),
                "estimate_attitude_error_deg": float(estimate_error),
                "controller_input_vs_truth_deg": float(controller_input_error),
                "wheel_speed_0_rad_s": float(board.wheel_speed_rad_s[0]),
                "wheel_speed_1_rad_s": float(board.wheel_speed_rad_s[1]),
                "wheel_speed_2_rad_s": float(board.wheel_speed_rad_s[2]),
                "commanded_torque_0_nm": float(board.commanded_torque_nm[0]),
                "applied_torque_0_nm": float(board.applied_torque_nm[0]),
                "saturated": float(board.saturated),
            }
        )

    final_error = series[-1]["truth_attitude_error_deg"]
    final_estimate_error = series[-1]["estimate_attitude_error_deg"]

    # Measured evidence for the truth-ownership rule, not a declaration of it.
    truth_cross_check = {
        "max_controller_input_vs_truth_deg": float(max_controller_input_error),
        "final_controller_input_vs_truth_deg": float(
            series[-1]["controller_input_vs_truth_deg"]
        ),
    }

    error = {
        "final_attitude_error_deg": float(final_error),
        "max_attitude_error_deg": float(max_error),
        "final_estimate_error_deg": float(final_estimate_error),
        "max_estimate_error_deg": float(
            max(row["estimate_attitude_error_deg"] for row in series)
        ),
    }
    if config.eclipse_windows:
        error["max_attitude_error_deg_during_eclipse"] = float(
            max_error_during_eclipse
        )

    wheels = {
        "max_speed_bound_rad_s": float(board.max_speed_rad_s),
        "max_abs_wheel_speed_rad_s": float(max_wheel_speed),
        "max_acceleration_cap_rad_s2": float(wheel_set.acceleration_cap_rad_s2),
        "saturated_steps": float(saturated_steps),
        "max_commanded_applied_gap_nm": float(max_gap),
    }

    if schedule is not None:
        planner: dict[str, object] = {
            "is_planner_generated": True,
            "source": f"SlewOS FA01 run {config.slewos_run_dir}",
            "order": list(schedule.order),
            "case_id": schedule.case_id,
            "method": schedule.method,
            "total_duration_s": schedule.total_duration_s,
            "planner_scope": PlannerScope.current().planner_support,
        }
    else:
        planner = {
            "is_planner_generated": False,
            "source": (
                "externally supplied smoothstep reference profile; the implemented "
                "planner is single-axis, so this is not planner output"
            ),
            "order": [],
            "planner_scope": PlannerScope.current().planner_support,
        }

    # Truth is read here for scoring only: where the vehicle actually ended,
    # scored against what it was commanded to do.
    commanded_target = (
        config.target_attitude
        if schedule is None
        else quat.from_axis_angle(
            np.array([0.0, 0.0, 1.0]), schedule.final_angle_rad
        )
    )
    plant_record: dict[str, object] = {
        "final_attitude": [float(x) for x in plant.attitude],
        "final_body_rate_rad_s": [float(x) for x in plant.body_rate],
        "final_wheel_speed_rad_s": [float(x) for x in board.wheel_speed_rad_s],
    }
    if commanded_target is not None:
        plant_record["commanded_target_attitude"] = [
            float(x) for x in quat.as_scalar_first(commanded_target)
        ]
        plant_record["final_attitude_error_to_target_deg"] = float(
            quat.rotation_angle_rad(
                quat.error_quaternion(commanded_target, plant.attitude)
            )
            * _RAD_TO_DEG
        )

    result = ClosedLoopResult(
        config=asdict(config),
        error=error,
        wheels=wheels,
        planner=planner,
        plant=plant_record,
        truth_cross_check=truth_cross_check,
        series=series,
    )
    result.output_json_path, result.output_csv_path = _write_outputs(result, config)
    return result


def _write_outputs(
    result: ClosedLoopResult, config: ClosedLoopConfig
) -> tuple[str, str]:
    """Write the run's JSON record and CSV time series; return their paths."""
    if config.output_dir is None:
        return "", ""
    directory = Path(config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{config.tag}.json"
    csv_path = directory / f"{config.tag}.csv"

    record = result.to_dict()
    record["artifacts"] = {"json": json_path.name, "csv": csv_path.name}
    json_path.write_text(json.dumps(record, indent=2, default=list), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.series[0].keys()))
        writer.writeheader()
        writer.writerows(result.series)

    return str(json_path), str(csv_path)
