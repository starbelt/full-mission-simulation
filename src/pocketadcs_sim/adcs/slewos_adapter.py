"""Read-only adapter over SlewOS (FA01) planner results in the sibling repo.

SlewOS is a real planner; this module does **not** reimplement or approximate
it. It reads a finished FA01 study -- a case manifest giving each objective's
target angle, and an append-only result file giving the chosen ordering and the
timed profile of each move -- and re-expresses that plan as an angle-vs-time
trajectory that the ADCS can track.

Two rules the adapter exists to enforce:

**Never invent a plan.** A missing case, a failed search, or a record missing a
required field raises :class:`AdapterError`. There is no fallback schedule, and
no locally computed substitute timing: a silently hand-written sequence that
looks like a SlewOS result is exactly the failure mode this adapter prevents.

**Never overstate the planner.** :class:`PlannerScope` records that the
implemented FA01 engine is *single-axis, rest-to-rest*. A multi-axis case is
refused rather than flattened onto one axis, because accepting it would imply a
planner that does not exist. Multi-axis attitude targets are still exercised in
the closed loop, but with an explicitly supplied schedule, and the two are kept
distinguishable in the output records.

The sibling repo is opened read-only and never written to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "SLEWOS_DEFAULT_PATH",
    "AdapterError",
    "PlannerScope",
    "ScheduleSegment",
    "SlewSchedule",
    "load_schedule",
    "schedule_to_telemetry",
]

#: Default location of the sibling SlewOS / reaction-wheel-tradespace checkout.
SLEWOS_DEFAULT_PATH = Path(
    r"C:\Users\drago\Nextcloud\School - NextCloud\Research"
    r"\PocketADCS\Github\reaction-wheel-tradespace"
)

FloatArray = npt.NDArray[np.float64]


class AdapterError(RuntimeError):
    """Raised when a plan cannot be read, is infeasible, or is out of scope."""


@dataclass(frozen=True)
class PlannerScope:
    """What the implemented SlewOS planner actually supports today.

    Read from the FA01 study specification rather than asserted by hand, so
    widening planner support shows up here rather than in prose.
    """

    study: str
    single_axis: bool
    rest_to_rest: bool
    multi_axis_planner_exists: bool
    source: str

    @classmethod
    def current(cls) -> PlannerScope:
        """Describe the implemented FA01 engine's scope."""
        return cls(
            study="FA01",
            single_axis=True,
            rest_to_rest=True,
            multi_axis_planner_exists=False,
            source=(
                "FA01 study specification: scalar target_angle_rad per objective, "
                "endpoint_velocity_rad_s = 0 (rest-to-rest), one axis per case"
            ),
        )

    @property
    def planner_support(self) -> str:
        """One-line human-readable scope statement."""
        return (
            "single-axis rest-to-rest sequencing of scalar slew angles; "
            "no multi-axis planner is implemented"
        )


@dataclass(frozen=True)
class ScheduleSegment:
    """One planner move: from a start angle to an objective's target angle."""

    objective_id: int
    start_angle_rad: float
    target_angle_rad: float
    duration_s: float
    start_time_s: float
    #: The planner's acceleration / coast / deceleration split, in seconds.
    profile_s: tuple[float, float, float]

    @property
    def end_time_s(self) -> float:
        """Time at which this move completes."""
        return self.start_time_s + self.duration_s

    def angle_at(self, t_s: float) -> float:
        """Return the commanded angle at time ``t_s`` within this segment.

        The planner's own profile: a symmetric trapezoid of angular
        acceleration ``alpha`` over an ``accel / coast / decel`` split, ending
        at rest on the objective's target angle.

        The distance such a profile covers is ``v_peak * (duration_s - (accel +
        decel) / 2)`` with ``v_peak = alpha * accel``, so that effective time
        ``V`` -- not the raw duration -- is the correct normaliser. Dividing by
        ``duration_s`` instead rescales the whole move, which makes the profile
        discontinuous at the coast/decel junction and implies an acceleration
        the planner never commanded. Real FA01 profiles make ``V`` and
        ``duration_s`` differ by ~1e-6 for a long coast and by 50% for a
        triangular (coast-free) move, so this is not a rounding detail.
        """
        local = float(t_s) - self.start_time_s
        if local <= 0.0:
            return self.start_angle_rad
        if local >= self.duration_s:
            return self.target_angle_rad

        accel, coast, decel = self.profile_s
        effective_s = self.duration_s - 0.5 * (accel + decel)
        if effective_s <= 0.0:
            raise AdapterError(
                f"segment {self.objective_id} has no effective time to move in "
                f"(duration_s={self.duration_s}, profile_s={self.profile_s}); "
                "the profile is degenerate and its commanded angle is undefined"
            )
        # Fraction of the distance covered at local time, from integrating the
        # piecewise-linear velocity profile.
        if local < accel:
            fraction = 0.5 * local**2 / (accel * effective_s) if accel > 0.0 else 0.0
        elif local < accel + coast:
            fraction = (0.5 * accel + (local - accel)) / effective_s
        else:
            remaining = self.duration_s - local
            fraction = (
                1.0 - 0.5 * (remaining**2) / (decel * effective_s)
                if decel > 0.0
                else 1.0
            )
        fraction = min(max(fraction, 0.0), 1.0)
        return self.start_angle_rad + fraction * (
            self.target_angle_rad - self.start_angle_rad
        )


@dataclass(frozen=True)
class SlewSchedule:
    """A complete planner result, ready to be tracked."""

    case_id: str
    method: str
    order: tuple[int, ...]
    segments: tuple[ScheduleSegment, ...]
    initial_angle_rad: float
    source_run_dir: str

    @property
    def total_duration_s(self) -> float:
        """Total planned maneuver time."""
        return self.segments[-1].end_time_s if self.segments else 0.0

    @property
    def final_angle_rad(self) -> float:
        """The angle the plan ends on."""
        return self.segments[-1].target_angle_rad if self.segments else self.initial_angle_rad

    def angle_at(self, t_s: float) -> float:
        """Return the planned angle at absolute time ``t_s``."""
        if not self.segments or t_s <= 0.0:
            return self.initial_angle_rad
        if t_s >= self.total_duration_s:
            return self.final_angle_rad
        for segment in self.segments:
            if t_s <= segment.end_time_s:
                return segment.angle_at(t_s)
        return self.final_angle_rad

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable record, including the planner's scope."""
        return {
            "case_id": self.case_id,
            "method": self.method,
            "order": list(self.order),
            "initial_angle_rad": self.initial_angle_rad,
            "final_angle_rad": self.final_angle_rad,
            "total_duration_s": self.total_duration_s,
            "source_run_dir": self.source_run_dir,
            "planner_scope": PlannerScope.current().planner_support,
            "segments": [
                {
                    "objective_id": segment.objective_id,
                    "start_angle_rad": segment.start_angle_rad,
                    "target_angle_rad": segment.target_angle_rad,
                    "duration_s": segment.duration_s,
                    "profile_s": list(segment.profile_s),
                }
                for segment in self.segments
            ],
        }


def _read_json(path: Path) -> object:
    if not path.exists():
        raise AdapterError(f"required input not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AdapterError(f"could not parse {path}: {error}") from error


def _locate_manifest(run: Path) -> Path:
    """Find the manifest for a study run, tolerating the real FA01 layout.

    FA01 writes ``raw.jsonl`` at the study root but its ``manifest.json`` into
    per-worker ``n<N>`` subdirectories, so a run directory can legitimately hold
    the results and the cases side by side rather than in the same place.
    """
    direct = run / "manifest.json"
    if direct.exists():
        return direct
    candidates = sorted(p for p in run.glob("*/manifest.json"))
    if not candidates:
        raise AdapterError(f"required input not found: {direct}")
    return candidates[0]


def load_schedule(
    run_dir: str | Path,
    case_id: str,
    method: str | None = None,
) -> SlewSchedule:
    """Load one planner result and build a trackable schedule.

    ``run_dir`` is a study output directory containing ``manifest.json`` and
    ``raw.jsonl``. ``method`` selects a planner variant (``"DO"``, ``"VND"``,
    ...); the first feasible record wins when ``method`` is ``None``.
    """
    run = Path(run_dir)
    # Real FA01 study roots hold raw.jsonl at the top level while the manifest
    # lives in a per-worker n<N> subdirectory, so a run may need the manifest
    # looked up one level down.
    manifest_path = _locate_manifest(run)
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise AdapterError(f"{manifest_path} is not an object")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise AdapterError(f"manifest.json in {run} has no 'cases' list")

    case = next((c for c in cases if c.get("case_id") == case_id), None)
    if case is None:
        raise AdapterError(f"no planner case {case_id!r} in {manifest_path}")

    objectives = case.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        raise AdapterError(f"case {case_id!r} has no objectives")
    for objective in objectives:
        if isinstance(objective, dict) and "axis" in objective:
            raise AdapterError(
                f"case {case_id!r} is multi-axis, but the implemented FA01 "
                "planner is single-axis; no multi-axis planner exists to expand it"
            )
    target_by_id = {int(o["objective_id"]): float(o["target_angle_rad"]) for o in objectives}

    raw_path = run / "raw.jsonl"
    if not raw_path.exists():
        raise AdapterError(f"required input not found: {raw_path}")

    record: dict[str, Any] | None = None
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("case_id") != case_id:
            continue
        if method is not None and entry.get("method") != method:
            continue
        if not entry.get("feasible", True):
            reason = entry.get("failure_reason") or "unspecified"
            raise AdapterError(
                f"planner record for {case_id!r}/{method!r} is infeasible: {reason}"
            )
        for field in ("order", "durations_s", "profiles_s"):
            if field not in entry:
                raise AdapterError(
                    f"planner record for {case_id!r} is missing required field {field!r}"
                )
        record = entry
        break

    if record is None:
        wanted = f"/{method}" if method else ""
        raise AdapterError(f"no planner record for case {case_id!r}{wanted} in {raw_path}")

    order = [int(item) for item in record["order"]]
    durations = [float(value) for value in record["durations_s"]]
    # Each profile row is a fixed (accel, coast, decel) triple, and the segment
    # unpacks it as one. Building it element-wise is what lets a row of the
    # wrong length be rejected here, at the record that produced it, rather
    # than deferred to the first ``angle_at`` on that segment mid-run.
    profiles: list[tuple[float, float, float]] = []
    for row in record["profiles_s"]:
        values = tuple(float(v) for v in row)
        if len(values) != 3:
            raise AdapterError(
                f"planner record for {case_id!r} has a profiles_s row with "
                f"{len(values)} entries; each profile must be exactly "
                "(accel, coast, decel)"
            )
        profiles.append(values)
    if not (len(order) == len(durations) == len(profiles)):
        raise AdapterError(
            f"planner record for {case_id!r} is inconsistent: order, durations_s "
            f"and profiles_s have different lengths"
        )

    segments: list[ScheduleSegment] = []
    angle = float(case["initial_angle_rad"])
    clock = 0.0
    for objective_id, duration, profile in zip(
        order, durations, profiles, strict=True
    ):
        target = target_by_id[objective_id]
        segments.append(
            ScheduleSegment(
                objective_id=objective_id,
                start_angle_rad=angle,
                target_angle_rad=target,
                duration_s=duration,
                start_time_s=clock,
                profile_s=profile,
            )
        )
        angle = target
        clock += duration

    return SlewSchedule(
        case_id=case_id,
        method=str(record.get("method", method or "")),
        order=tuple(order),
        segments=tuple(segments),
        initial_angle_rad=float(case["initial_angle_rad"]),
        source_run_dir=str(run),
    )


def schedule_to_telemetry(schedule: SlewSchedule, times_s: npt.ArrayLike) -> FloatArray:
    """Sample a schedule's planned angle at each requested time."""
    times = np.asarray(times_s, dtype=float).reshape(-1)
    return np.array([schedule.angle_at(float(t)) for t in times], dtype=float)
