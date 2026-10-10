"""Acceptance guards for the physical HIL adapter (no serial hardware)."""
import numpy as np
import pytest

from examples import hil_closed_loop_run as hil
from pocketadcs_sim.adcs import quaternions as quat
from pocketadcs_sim.adcs.closed_loop import ClosedLoopConfig
from pocketadcs_sim.adcs.plant import TruthPlant
from pocketadcs_sim.adcs.wheel_board import WheelBoard


def test_environment_uses_real_time_position_references_and_body_rotation():
    from astropy.coordinates import get_sun
    from astropy.time import Time
    environment = hil.NavigationHistory(1.0)
    position, sun, magnetic = environment.at(0.0)
    expected = get_sun(Time("2025-01-01T00:00:00", scale="utc")).cartesian.xyz.value
    expected /= np.linalg.norm(expected)
    assert sun == pytest.approx(expected, abs=1e-10)
    assert 1e-6 < np.linalg.norm(magnetic) < 1e-3
    assert np.linalg.norm(position) == pytest.approx(6_878_137.0)
    attitude = quat.from_axis_angle([0.3, 0.8, 0.5], 0.7)
    gyro, mag_body, sun_body = hil.measure(environment, 0.0, attitude,
                                          np.array([0.01, 0.02, 0.03]), False, 0.0)
    assert mag_body == pytest.approx(quat.to_dcm(attitude) @ magnetic)
    assert sun_body == pytest.approx(quat.to_dcm(attitude) @ sun)
    assert gyro == pytest.approx([0.01, 0.02, 0.03])


def test_offline_complete_3d_loop_scores_commanded_target(tmp_path):
    start = quat.from_axis_angle([1, 0, 0], 0.3)
    target = quat.from_axis_angle([0.3, 0.8, 0.5], 0.7)
    config = ClosedLoopConfig(duration_s=300, start_attitude=start, target_attitude=target)
    binary = hil.build_offline(tmp_path / "build")
    with hil.StdioLink(binary) as link:
        result = hil.run_loop(link, config, tmp_path / "run_v1")
    assert result["final_target_error_deg"] < 0.1
    assert result["duration_s"] == 300
    assert result["frames"] == 6001
    assert np.all(np.array(result["max_abs_rate_rad_s"]) > 1e-4)
    assert np.all(np.array(result["max_abs_wheel_speed_rad_s"]) > 1)
    assert result["stage_execution_time"] is None
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((tmp_path / "run_v1" / "manifest.json").read_text())
    required = [*sorted((root / "esp32/src").glob("*.h")),
                *sorted((root / "src/pocketadcs_sim/adcs").glob("*.py")),
                root / "examples/hil_closed_loop_run.py", root / "examples/hil_closed_loop.py",
                root / "src/pocketadcs_sim/truth_environment.py",
                root / "tests/firmware/adcs_ephemeris_stdio.cpp"]
    for source in required:
        assert manifest["source_sha256"][source.relative_to(root).as_posix()] == \
            hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["binary_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()


def test_two_complete_fresh_missions_share_persistent_device_without_reset(tmp_path):
    """A completed 3D mission followed by a fresh t0 world on the same process."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    binary = hil.build_offline(tmp_path / "persistent_build")
    with hil.StdioLink(binary) as link:
        process = link.process
        for name in ("target3d", "slewos"):
            manifest = json.loads((root / "results/hil_accepted/v3" / name /
                                   "manifest.json").read_text())
            result = hil.run_loop(link, ClosedLoopConfig(**manifest["config"]),
                                  tmp_path / name, fix_bias_rad=manifest["fix_bias_rad"])
            prior = json.loads((root / "results/hil_accepted/v3" / name /
                                "summary.json").read_text())
            assert result == prior
            wire = [json.loads(row) for row in (tmp_path / name / "wire.jsonl").read_text()
                    .splitlines()]
            assert wire[:2] == [{"direction": "tx", "line": "XZ"},
                                {"direction": "rx", "line": "E XZ 0 -1 0"}]
            assert link.process is process and process.poll() is None


@pytest.mark.parametrize("field,value", [("f3", 9), ("f1", 1.5), ("f2", 2),
                                            ("f20", -1)])
def test_invalid_device_reply_cannot_silently_pass(field, value):
    from test_hil_closed_loop_harness import y_line

    from examples.hil_closed_loop import ProtocolError, parse_y
    with pytest.raises(ProtocolError):
        parse_y(y_line(**{field: value}))


def test_host_computed_wheel_command_mutation_is_rejected(tmp_path, monkeypatch):
    from examples.hil_closed_loop import ProtocolError
    original = WheelBoard.update
    def substitute(self, command, dt):
        return original(self, np.array([1e-4, 0, 0]), dt)
    monkeypatch.setattr(WheelBoard, "update", substitute)
    binary = hil.build_offline(tmp_path / "build")
    with hil.StdioLink(binary) as link, pytest.raises(ProtocolError, match="wheel command"):
        hil.run_loop(link, ClosedLoopConfig(duration_s=0.1), tmp_path / "run_v1")


def test_verdict_scores_final_command_not_reference_tracking():
    # A perfect internal tracking number cannot bless a plant still 30deg short.
    summary = {"final_target_error_deg": 30.0, "final_reference_error_deg": 0.0}
    assert hil.verdict(summary, 0.1) == 2


@pytest.fixture(scope="module")
def device_binary(tmp_path_factory):
    return hil.build_offline(tmp_path_factory.mktemp("exact_device") / "build")


def test_cxx_truth_control_mutation_fails_guard_in_isolated_copy(device_binary, tmp_path):
    import shutil
    import subprocess
    from pathlib import Path

    from test_ads_math_host import _env_for, find_usable_compiler

    from examples.hil_closed_loop import ProtocolError
    root = Path(__file__).resolve().parents[1]
    headers = tmp_path / "mutated_headers"
    shutil.copytree(root / "esp32/src", headers)
    protocol = headers / "adcs_protocol.h"
    source = protocol.read_text()
    source = source.replace("session.controller_input = session.estimator.attitude;",
                            "session.controller_input = quatIdentity();")
    source = source.replace(
        "controllerTorque(session.config, session.estimator.attitude, gyro, desired,",
        "controllerTorque(session.config, session.controller_input, gyro, desired,")
    # Only the isolated copy is mutated; the real headers remain untouched.
    protocol.write_text(source)
    compiler = find_usable_compiler(tmp_path)
    binary = tmp_path / "truth_mutant.exe"
    subprocess.run([str(compiler), "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I", str(headers), str(root / "tests/firmware/adcs_ephemeris_stdio.cpp"),
                    "-o", str(binary), "-lm"], env=_env_for(compiler), check=True,
                   capture_output=True, text=True)
    # True initial attitude is identity; biased sensor-derived estimate is NOT.
    with hil.StdioLink(binary) as link, pytest.raises(ProtocolError, match="consume"):
        hil.run_loop(link, ClosedLoopConfig(duration_s=0.1), tmp_path / "mutant_v1",
                     fix_bias_rad=np.radians(1))


def test_truth_controller_input_mutation_is_rejected(device_binary, tmp_path):
    from examples.hil_closed_loop import ProtocolError
    class TruthEcho(hil.StdioLink):
        def read_reply(self, deadline):
            line = super().read_reply(deadline)
            if line and line.startswith("Y "):
                fields = line.split()
                # Holding identity truth with biased sensors: substitute truth
                # at the returned executed-control seam, not caller intent.
                fields[8:12] = ["1", "0", "0", "0"]
                return " ".join(fields)
            return line
    with TruthEcho(device_binary) as link, pytest.raises(ProtocolError, match="consume"):
        hil.run_loop(link, ClosedLoopConfig(duration_s=0.1), tmp_path / "mutant_v1",
                     fix_bias_rad=np.radians(1))


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "nonfinite", "incomplete",
                                      "status", "ack", "reject", "wheel_limit"])
def test_transport_corruption_cannot_pass(device_binary, tmp_path, mutation):
    from examples.hil_closed_loop import ProtocolError
    class Corrupt(hil.StdioLink):
        def __init__(self, binary):
            super().__init__(binary)
            self.last = None
        def read_reply(self, deadline):
            line = super().read_reply(deadline)
            if line and line.startswith("E XC") and mutation == "ack":
                return line.replace("XC", "XE")
            if not line or not line.startswith("Y "):
                return line
            if mutation == "missing":
                return None
            if mutation == "nonfinite":
                return line + " nan"
            if mutation == "incomplete":
                return " ".join(line.split()[:-1])
            if mutation == "status":
                return line.replace("Y 0 ", "Y 1 ", 1)
            if mutation in ("reject", "wheel_limit"):
                fields = line.split()
                fields[20 if mutation == "reject" else 15] = "2" if mutation == "reject" else "999"
                return " ".join(fields)
            if mutation == "duplicate":
                if self.last:
                    return self.last
                self.last = line
            return line
    with Corrupt(device_binary) as link, pytest.raises(ProtocolError):
        hil.run_loop(link, ClosedLoopConfig(duration_s=0.1), tmp_path / "bad_v1")


@pytest.mark.parametrize("command", ["XZ", "XC", "XL", "XR", "XE"])
@pytest.mark.parametrize("payload", [
    "0 nan nan", "0 inf 0", "0 -1 nan", "nan -1 0", "inf -1 0",
    "0 -1 inf", "0.5 -1 0", "0 -1.5 0", "0 -1 0.5", "0 0 0",
    "0 -1 2", "0 -1 -1", "0 -1 1", "1 -1 0", "0 -1", "0 -1 0 extra",
])
def test_malformed_plan_ack_cannot_pass(device_binary, tmp_path, command, payload):
    from examples.hil_closed_loop import ProtocolError

    class CorruptAck(hil.StdioLink):
        def read_reply(self, deadline):
            line = super().read_reply(deadline)
            if line and line.startswith(f"E {command} "):
                return f"E {command} {payload}"
            return line

    with CorruptAck(device_binary) as link, pytest.raises(ProtocolError, match="ACK"):
        hil.run_loop(link, ClosedLoopConfig(duration_s=0.1), tmp_path / "bad_ack_v1")


@pytest.mark.parametrize("answer", [None, "E XE 0 -1 0", "E XZ 0 -1 1"])
def test_failed_fresh_reset_stops_before_config_or_sensor_transmission(tmp_path, answer):
    from examples.hil_closed_loop import ProtocolError

    class RefuseReset:
        provenance = "deliberately refusing test transport"

        def __init__(self):
            self.sent = []

        def send(self, line):
            self.sent.append(line)

        def read_reply(self, deadline):
            return answer

    link = RefuseReset()
    with pytest.raises(ProtocolError):
        hil.run_loop(link, ClosedLoopConfig(duration_s=.1), tmp_path / "bad_reset")
    assert link.sent == ["XZ"]


def test_trailing_duplicate_is_drained_and_rejected(device_binary, tmp_path):
    from examples.hil_closed_loop import ProtocolError
    class Duplicate(hil.StdioLink):
        def read_reply(self, deadline):
            line = super().read_reply(deadline)
            if line and line.startswith("Y 0 3 "):
                self.lines.put(line)
            return line
    with Duplicate(device_binary) as link, pytest.raises(ProtocolError, match="trailing"):
        hil.run_loop(link, ClosedLoopConfig(duration_s=0.1), tmp_path / "duplicate_v1")


def test_noisy_bias_changes_physical_pointing_and_eclipse_reacquires(device_binary, tmp_path):
    config = ClosedLoopConfig(duration_s=60, target_attitude=quat.from_axis_angle([.3,.8,.5], .2),
                              eclipse_windows=((20, 40),))
    environment = hil.NavigationHistory(60)
    with hil.StdioLink(device_binary) as link:
        ideal = hil.run_loop(link, config, tmp_path / "ideal_v1", environment=environment)
    config.sensors = "noisy"
    with hil.StdioLink(device_binary) as link:
        biased = hil.run_loop(link, config, tmp_path / "biased_v1", environment=environment,
                              fix_bias_rad=np.radians(1))
    assert ideal["eclipse_samples"] == 401
    assert ideal["eclipse_max_seam_error_deg"] < 0.1
    assert ideal["reacquisition_seam_error_deg"] < 0.1
    assert biased["max_controller_input_vs_truth_deg"] > 0.8
    assert biased["final_target_error_deg"] > ideal["final_target_error_deg"] + 0.5
    assert biased["eclipse_max_seam_error_deg"] > 0.8
    assert biased["reacquisition_seam_error_deg"] > 0.8


def test_cli_exits_nonzero_on_physical_target_miss(device_binary, tmp_path):
    assert hil.main(["--offline", "--output-dir", str(tmp_path / "miss_v1"),
                     "--duration", "0.1", "--target", "0.70710678", "0", "0", "0.70710678"]) == 2


def test_refuses_existing_output_before_opening_link(tmp_path, monkeypatch):
    output = tmp_path / "existing_v1"
    output.mkdir()
    monkeypatch.setattr(hil, "build_offline",
                        lambda *_: pytest.fail("opened link before refusing overwrite"))
    assert hil.main(["--offline", "--output-dir", str(output)]) == 1


def test_adapter_reuses_accepted_coupled_plant_and_board():
    plant, board = hil.make_world(ClosedLoopConfig())
    assert type(plant) is TruthPlant
    assert type(board) is WheelBoard
    assert plant.config.hub_inertia_kg_m2 == (0.02, 0.025, 0.03)
    assert np.asarray(plant.config.wheel_set.axes) == pytest.approx(np.eye(3))
    torque = np.array([1e-4, -2e-4, 3e-4])
    expected = TruthPlant(plant.config, plant.attitude)
    applied = board.update(torque, 0.05)
    expected.step_rk4(0.05, applied)
    hil.advance_world(plant, applied, 0.05)
    assert plant.body_rate == pytest.approx(expected.body_rate)
    assert plant.wheel_speed == pytest.approx(board.wheel_speed_rad_s)
