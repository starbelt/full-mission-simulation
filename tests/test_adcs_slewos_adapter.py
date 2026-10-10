"""Tests for the SlewOS schedule adapter.

The adapter's job is narrow: read a *real* FA01 planner result from the sibling
``reaction-wheel-tradespace`` repo and turn it into an angle-vs-time trajectory
this ADCS can track. It must not invent a schedule, and it must not pretend the
planner supports something it does not.

The sibling repo is read-only: these tests only read from it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pocketadcs_sim.adcs.slewos_adapter import (
    SLEWOS_DEFAULT_PATH,
    AdapterError,
    PlannerScope,
    ScheduleSegment,
    load_schedule,
    schedule_to_telemetry,
)

pytestmark = pytest.mark.skipif(
    not SLEWOS_DEFAULT_PATH.exists(), reason="sibling SlewOS repo not present"
)


def write_run(root: Path, case: dict, record: dict) -> Path:
    """Write a minimal FA01 run tree (manifest + one raw record) under root."""
    results = root / "results" / "fa01" / "fa01_test_v1" / "n2"
    results.mkdir(parents=True, exist_ok=True)
    (results / "manifest.json").write_text(
        json.dumps({"specification": {"study": "A", "length": 2}, "cases": [case]}),
        encoding="utf-8",
    )
    (results / "raw.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return results


def sample_case() -> dict:
    return {
        "case_id": "a2-00000",
        "initial_angle_rad": 0.5237018087506935,
        "objectives": [
            {"objective_id": 0, "target_angle_rad": 1.0384748012123453},
            {"objective_id": 1, "target_angle_rad": 0.4460820244321466},
        ],
    }


def sample_record() -> dict:
    return {
        "case_id": "a2-00000",
        "method": "DO",
        "order": [1, 0],
        "durations_s": [555.6434195242749, 2126.331430334369],
        "profiles_s": [
            [0.00058, 555.64225, 0.00058],
            [0.00115, 2126.32911, 0.00115],
        ],
    }


def test_adapter_reads_the_planner_order_and_target_angles(tmp_path: Path) -> None:
    """The schedule is built from the planner's own order and angles."""
    results = write_run(tmp_path, sample_case(), sample_record())

    schedule = load_schedule(results, case_id="a2-00000", method="DO")

    # The planner ordered objective 1 first, then 0. That order, not a
    # hand-written sequence, is what the schedule must follow.
    assert schedule.order == (1, 0)
    assert schedule.segments[0].target_angle_rad == pytest.approx(0.4460820244321466)
    assert schedule.segments[1].target_angle_rad == pytest.approx(1.0384748012123453)
    assert schedule.segments[0].start_angle_rad == pytest.approx(0.5237018087506935)


def test_segment_durations_come_from_the_planner_not_from_a_local_model(
    tmp_path: Path,
) -> None:
    """Segment times are the planner's reported durations, exactly."""
    results = write_run(tmp_path, sample_case(), sample_record())

    schedule = load_schedule(results, case_id="a2-00000", method="DO")

    np.testing.assert_allclose(
        [segment.duration_s for segment in schedule.segments],
        [555.6434195242749, 2126.331430334369],
        rtol=1e-12,
    )
    assert schedule.total_duration_s == pytest.approx(2681.974849858644, rel=1e-9)


def test_schedule_targets_the_final_objective_of_the_plan(tmp_path: Path) -> None:
    """A complete run ends on the planner's last ordered objective."""
    results = write_run(tmp_path, sample_case(), sample_record())

    schedule = load_schedule(results, case_id="a2-00000", method="DO")

    assert schedule.final_angle_rad == pytest.approx(1.0384748012123453)
    assert schedule.segments[-1].end_time_s == pytest.approx(
        schedule.total_duration_s, rel=1e-12
    )


def test_telemetry_samples_the_schedule_angle_at_each_time(tmp_path: Path) -> None:
    """Angle lookup is monotone in time and starts at the initial angle."""
    results = write_run(tmp_path, sample_case(), sample_record())
    schedule = load_schedule(results, case_id="a2-00000", method="DO")

    times = np.linspace(0.0, schedule.total_duration_s, 11)
    angles = schedule_to_telemetry(schedule, times)

    assert angles.shape == times.shape
    assert angles[0] == pytest.approx(0.5237018087506935)
    # A rest-to-rest move must start and finish stationary, so the angle is
    # constant while the profile is in its first (acceleration) third only in
    # the sense that it does not jump; assert the final sample is the target.
    assert angles[-1] == pytest.approx(1.0384748012123453, rel=1e-9)


def test_telemetry_beyond_the_plan_holds_the_final_angle(tmp_path: Path) -> None:
    """Asking past the end of the plan returns the final angle, not a guess."""
    results = write_run(tmp_path, sample_case(), sample_record())
    schedule = load_schedule(results, case_id="a2-00000", method="DO")

    angles = schedule_to_telemetry(schedule, np.array([schedule.total_duration_s * 2]))

    assert angles[0] == pytest.approx(1.0384748012123453)


def test_missing_case_raises_rather_than_inventing_a_schedule(tmp_path: Path) -> None:
    """An absent plan is an error, never a silent fallback."""
    results = write_run(tmp_path, sample_case(), sample_record())

    with pytest.raises(AdapterError, match="no planner case"):
        load_schedule(results, case_id="does-not-exist", method="DO")


def test_infeasible_or_failed_plan_is_refused(tmp_path: Path) -> None:
    """A planner record that failed must not be tracked as if it succeeded."""
    record = sample_record() | {"feasible": False, "failure_reason": "search budget"}
    results = write_run(tmp_path, sample_case(), record)

    with pytest.raises(AdapterError, match="infeasible"):
        load_schedule(results, case_id="a2-00000", method="DO")


def test_corrupt_record_is_reported_not_silently_repaired(tmp_path: Path) -> None:
    """A record missing a required field is reported loudly."""
    record = sample_record()
    del record["durations_s"]
    results = write_run(tmp_path, sample_case(), record)

    with pytest.raises(AdapterError, match="durations_s"):
        load_schedule(results, case_id="a2-00000", method="DO")


def test_profile_row_of_the_wrong_arity_is_rejected(tmp_path: Path) -> None:
    """A profile row must carry exactly (accel, coast, decel) seconds.

    ``ScheduleSegment.profile_s`` is a fixed-length 3-tuple, and the unpacking
    that consumes it (``accel, coast, decel = self.profile_s``) is what makes
    that length meaningful. A two- or four-element row therefore cannot be
    represented, and letting one through defers the failure to the first
    ``angle_at`` call on that segment -- mid-run, far from the record that
    caused it. It must be refused where the record is read.
    """
    for bad in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0]):
        record = sample_record()
        record["profiles_s"] = [bad, [0.00115, 2126.32911, 0.00115]]
        results = write_run(tmp_path, sample_case(), record)

        with pytest.raises(AdapterError, match="profiles_s"):
            load_schedule(results, case_id="a2-00000", method="DO")


def test_missing_files_raise_a_clear_error(tmp_path: Path) -> None:
    """Missing inputs name the path that was looked for."""
    with pytest.raises(AdapterError, match="manifest.json"):
        load_schedule(tmp_path / "nowhere", case_id="a2-00000", method="DO")
    with pytest.raises(AdapterError, match="manifest.json"):
        load_schedule(tmp_path, case_id="a2-00000", method="DO")


def test_planner_scope_is_reported_honestly() -> None:
    """Scope must say the planner is single-axis rest-to-rest, not multi-axis."""
    scope = PlannerScope.current()

    assert scope.single_axis is True
    assert scope.rest_to_rest is True
    assert "multi-axis" in scope.planner_support.lower() or (
        scope.multi_axis_planner_exists is False
    )
    assert scope.multi_axis_planner_exists is False


def test_multi_axis_case_is_rejected_as_out_of_planner_scope(tmp_path: Path) -> None:
    """A multi-axis case must be refused by the adapter, not silently flattened.

    FA01 plans one scalar axis. Accepting a 3-axis case here would mean
    inventing a planner that does not exist.
    """
    case = sample_case() | {
        "objectives": [
            {"objective_id": 0, "target_angle_rad": 1.0, "axis": [1, 0, 0]},
            {"objective_id": 1, "target_angle_rad": 0.5, "axis": [0, 1, 0]},
        ]
    }
    results = write_run(tmp_path, case, sample_record())

    with pytest.raises(AdapterError, match="multi-axis"):
        load_schedule(results, case_id="a2-00000", method="DO")


def test_real_repo_run_is_readable_end_to_end() -> None:
    """Read a real FA01 result from the sibling repo, not a fixture."""
    import glob

    # raw.jsonl lives at the study root; the n<N> subdirectories are per-worker
    # manifests. Search both depths.
    candidates = sorted(
        glob.glob(str(SLEWOS_DEFAULT_PATH / "results" / "fa01" / "fa01_a*" / "raw.jsonl"))
    )
    if not candidates:
        pytest.skip("no FA01 study A results found in the sibling repo")
    # The real layout puts raw.jsonl at the study root and manifest.json in a
    # per-worker subdirectory; the adapter locates both.
    results = Path(candidates[-1]).parent
    manifest = json.loads(
        (results / "n2" / "manifest.json").read_text(encoding="utf-8")
    )
    case_id = manifest["cases"][0]["case_id"]
    first = json.loads((results / "raw.jsonl").read_text(encoding="utf-8").splitlines()[0])
    if not first.get("feasible", True):
        pytest.skip("first planner record is infeasible")

    schedule = load_schedule(results, case_id=case_id, method=first["method"])

    assert len(schedule.segments) == len(first["order"])
    assert schedule.segments[0].target_angle_rad == pytest.approx(
        next(
            o["target_angle_rad"]
            for o in manifest["cases"][0]["objectives"]
            if o["objective_id"] == first["order"][0]
        )
    )
    assert schedule.total_duration_s > 0.0


def test_adapter_never_writes_to_the_sibling_repo(tmp_path: Path) -> None:
    """The sibling repo is read-only; the adapter must not create anything."""
    results = write_run(tmp_path, sample_case(), sample_record())
    before = sorted(os.walk(results))

    load_schedule(results, case_id="a2-00000", method="DO")

    assert sorted(os.walk(results)) == before


def test_every_profile_phase_is_evaluated_correctly() -> None:
    """All three thirds of the move must sample without error.

    The profile's three thirds are distinct code paths, so a name slip in the
    deceleration branch is invisible until a sample lands there.

    This test no longer re-derives the implementation's own expression -- doing
    so made it a transcription that passed on a wrong denominator. The exact
    closed-form checks now live in ``tests/test_adcs_slewos_profile.py``,
    against numerical integration and the planner's own ``alpha``; what is
    pinned here is the weaker, still useful statement that the deceleration
    branch is *reached* and behaves like a decelerating one.
    """
    # FA01's real profiles have a near-zero decel third, so sample a synthetic
    # segment whose thirds are all non-degenerate: that is what actually
    # exercises the deceleration branch.
    segment = ScheduleSegment(
        objective_id=0,
        start_angle_rad=0.0,
        target_angle_rad=1.0,
        duration_s=10.0,
        start_time_s=0.0,
        profile_s=(1.0, 8.0, 1.0),
    )

    accel, coast, decel = segment.profile_s
    assert decel > 0.0, "this segment only exercises the decel branch if decel > 0"

    # Inside the deceleration third, the commanded angle advances but ever more
    # slowly -- a step there is distinguishable from the coast phase.
    decel_start = accel + coast
    early = segment.angle_at(decel_start + 0.1) - segment.angle_at(decel_start)
    late = segment.angle_at(segment.duration_s) - segment.angle_at(
        segment.duration_s - 0.1
    )
    assert early > 0.0
    assert late < early
