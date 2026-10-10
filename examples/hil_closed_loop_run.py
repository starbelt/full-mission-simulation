"""Physical-world adapter for the exact C++ ADCS protocol; no host ADCS.

Offline results are native host correctness tests, NOT ESP32 performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from examples.hil_closed_loop import EPOCH_START_UNIX_MS
from pocketadcs_sim.adcs import quaternions as quat
from pocketadcs_sim.adcs.closed_loop import ClosedLoopConfig
from pocketadcs_sim.adcs.plant import PlantConfig, TruthPlant
from pocketadcs_sim.adcs.rotors import default_wheel_set
from pocketadcs_sim.adcs.wheel_board import WheelBoard


class NavigationHistory:
    """Declared circular GCRS orbit, independent Astropy Sun + existing IGRF14.

    1-second environment knots are linearly interpolated; navigation is evaluated
    analytically at every sample. No environment value is sent except position.
    No gravity/disturbance/eclipse-geometry claim is made by this bounded rig.
    """
    def __init__(self, duration_s: float):
        import astropy.units as u
        from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, get_sun
        from astropy.time import Time
        from astropy.utils import iers

        from pocketadcs_sim.truth_environment import _igrf14_itrs, _itrs_vectors_to_gcrs
        iers.conf.auto_download = False
        self.times = np.arange(0.0, np.ceil(duration_s) + 1.0)
        times = Time(EPOCH_START_UNIX_MS / 1000, format="unix", scale="utc") + self.times * u.s
        positions = np.array([self.position(t) for t in self.times])
        gcrs = GCRS(CartesianRepresentation(positions.T * u.m), obstime=times)
        itrs = gcrs.transform_to(ITRS(obstime=times)).cartesian.xyz.to_value(u.m).T
        self.mag = _itrs_vectors_to_gcrs(_igrf14_itrs(itrs, times), times)
        sun = get_sun(times).cartesian.xyz.value.T
        self.sun = sun / np.linalg.norm(sun, axis=1, keepdims=True)

    @staticmethod
    def position(t_s: float):
        radius = 6_878_137.0
        phase = np.sqrt(3.986004418e14 / radius**3) * t_s
        inclination = np.radians(51.6)
        return radius * np.array([np.cos(phase), np.sin(phase) * np.cos(inclination),
                                  np.sin(phase) * np.sin(inclination)])

    def at(self, t_s: float):
        sun = np.array([np.interp(t_s, self.times, self.sun[:, i]) for i in range(3)])
        magnetic = np.array([np.interp(t_s, self.times, self.mag[:, i]) for i in range(3)])
        return self.position(t_s), sun / np.linalg.norm(sun), magnetic


def measure(environment, t_s, attitude, body_rate, eclipse, fix_bias_rad,
            rng=None):
    _, sun, magnetic = environment.at(t_s)
    dcm = quat.to_dcm(attitude)
    # Common sensor mounting bias deliberately rotates both measured directions.
    bias = quat.to_dcm(quat.from_axis_angle([0.3, 0.8, 0.5], fix_bias_rad))
    mag_body = bias @ dcm @ magnetic
    sun_body = bias @ dcm @ sun
    gyro = body_rate.copy()
    if rng is not None:
        gyro += np.array([2e-4, -1e-4, 1.5e-4]) + rng.normal(0, 1e-5, 3)
        mag_body += rng.normal(0, 1e-8, 3)
        sun_body += rng.normal(0, 2e-4, 3)
    if eclipse:
        sun_body = np.zeros(3)
    return gyro, mag_body, sun_body


def make_world(config: ClosedLoopConfig):
    wheels = config.wheel_set or default_wheel_set()
    return (TruthPlant(PlantConfig(config.hub_inertia_kg_m2, wheels),
                       config.start_attitude), WheelBoard(wheels))


def advance_world(plant: TruthPlant, applied: np.ndarray, dt_s: float):
    plant.step_rk4(dt_s, applied)


def build_offline(directory: Path):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    compiler = shutil.which("g++") or "C:/msys64/ucrt64/bin/g++.exe"
    root = Path(__file__).resolve().parents[1]
    binary = directory / "adcs_ephemeris_stdio.exe"
    env = os.environ.copy()
    env["PATH"] = str(Path(compiler).parent) + os.pathsep + env["PATH"]
    # -static keeps the surrogate free of libstdc++-6.dll: the exe is launched
    # with only its own directory on PATH, so a dynamic link is not reliably
    # resolvable (STATUS_DLL_NOT_FOUND / 0xc0000135).
    command = [compiler, "-std=c++11", "-O2", "-static", "-Wall", "-Wextra", "-Werror",
               "-I", str(root / "esp32/src"),
               str(root / "tests/firmware/adcs_ephemeris_stdio.cpp"), "-o", str(binary), "-lm"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    (directory / "build.json").write_text(json.dumps({"command": command,
        "returncode": result.returncode, "stdout": result.stdout,
        "stderr": result.stderr}, indent=2))
    if result.returncode:
        raise RuntimeError(result.stderr)
    return binary


class StdioLink:
    provenance = "native-host exact C++ core/protocol/ephemeris surrogate; NOT hardware"

    def __init__(self, binary):
        env = os.environ.copy()
        env["PATH"] = "C:/msys64/ucrt64/bin" + os.pathsep + env["PATH"]
        self.binary = Path(binary).resolve()
        self.process = subprocess.Popen([str(self.binary)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.lines = queue.Queue()
        def receive():
            for line in self.process.stdout:
                self.lines.put(line.strip())
            self.lines.put(None)
        self.reader = threading.Thread(target=receive, daemon=True)
        self.reader.start()

    def send(self, line):
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def read_reply(self, deadline):
        try:
            return self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            return None

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.reader.join(timeout=1)
        self.process.stdout.close()
        if self.process.returncode:
            raise RuntimeError(f"surrogate exit {self.process.returncode}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def parse_identity(line: str) -> dict[str, object]:
    """Validate XI counter provenance; do not assume the configured CPU clock."""
    from examples.hil_closed_loop import ProtocolError

    fields = line.split()
    if len(fields) != 6 or fields[:2] != ["I", "1"]:
        raise ProtocolError("invalid XI identity shape/version")
    if len(fields[2]) != 64 or any(c not in "0123456789abcdef" for c in fields[2]):
        raise ProtocolError("invalid XI source digest")
    try:
        hz, line_bytes, segments = (int(v) for v in fields[3:])
    except ValueError as exc:
        raise ProtocolError("invalid XI integer field") from exc
    if not 0 < hz <= 0xFFFFFFFF or line_bytes != 320 or segments != 64:
        raise ProtocolError("invalid XI frequency/capacity")
    return {"build_id": fields[2], "counter_hz": hz,
            "line_bytes": line_bytes, "max_segments": segments}


class SerialLink:
    provenance = "serial device; cycle timing requires validated XI"
    device_timing = True

    def __init__(self, port):
        import serial
        self.port = serial.Serial()
        self.port.dtr = self.port.rts = False
        self.port.port = port
        self.port.baudrate = 115200
        self.port.timeout = 0.05
        self.port.write_timeout = 5.0
        self.port.open()

    def send(self, line):
        self.port.write((line + "\n").encode("ascii"))

    def read_reply(self, deadline):
        while time.monotonic() < deadline:
            raw = self.port.readline()
            if raw:
                return raw.decode("ascii", errors="strict").strip()
        return None

    def close(self):
        self.port.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run_loop(link, config, output_dir, *, environment=None, fix_bias_rad=0.0):
    from dataclasses import asdict

    from examples.hil_closed_loop import (
        ProtocolError,
        parse_y,
        plan_lines,
        quaternion_error_deg,
        reference_profile_lines,
    )
    from pocketadcs_sim.adcs.slewos_adapter import load_schedule
    # Exact protocol currently integrates allocation at 50 ms. Do not silently
    # misrepresent arbitrary sample periods as supported by this device core.
    if config.dt_s != 0.05:
        raise ValueError("exact device protocol requires dt_s=0.05")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    plant, board = make_world(config)
    if config.slewos_run_dir:
        schedule = load_schedule(config.slewos_run_dir, config.slewos_case_id,
                                 config.slewos_method)
        duration = schedule.total_duration_s
        plant.attitude = quat.from_axis_angle([0, 0, 1], schedule.initial_angle_rad)
        target = quat.from_axis_angle([0, 0, 1], schedule.final_angle_rad)
        instructions = plan_lines(schedule.initial_angle_rad, [
            (s.objective_id, s.start_angle_rad, s.target_angle_rad, s.duration_s,
             *s.profile_s) for s in schedule.segments])
        plan = schedule.to_dict()
    else:
        duration = config.duration_s
        target = quat.as_scalar_first(config.target_attitude if config.target_attitude is not None
                                      else config.start_attitude)
        instructions = reference_profile_lines(config.start_attitude, target, duration)
        plan = {"is_planner_generated": False, "source": "external smoothstep 3D profile"}
    root = Path(__file__).resolve().parents[1]
    # Include all shared headers (a conservative superset of the transitive
    # includes) and accepted physical-world sources, not just entry headers.
    sources = [*sorted((root / "esp32/src").glob("*.h")),
               *sorted((root / "src/pocketadcs_sim/adcs").glob("*.py")),
               root / "examples/hil_closed_loop_run.py", root / "examples/hil_closed_loop.py",
               root / "src/pocketadcs_sim/truth_environment.py",
               root / "tests/firmware/adcs_ephemeris_stdio.cpp"]
    binary = getattr(link, "binary", None)
    manifest = {"version": 2, "config": asdict(config), "plan": plan,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()
                    if binary is not None else None,
                "commanded_target": target.tolist(), "device": link.provenance,
                "stage_execution_time": None, "dt_s": config.dt_s,
                "duration_s": duration, "fix_bias_rad": fix_bias_rad,
                "timing": {"host_clock": "perf_counter_ns",
                    "host_clock_resolution_s": time.get_clock_info("perf_counter").resolution,
                    "algorithm_time_from_rtt": False, "device_identity": None,
                    "y_schema": "25 tokens legacy; optional 26th uint32 service_cycles",
                    "records": "timing.jsonl", "native_device_metrics": None,
                    "host_exchange_boundary": "before send through matched reply reception, "
                        "including host I/O/decode/logging; not algorithm time"},
                "environment": "circular GCRS r=6878137m i=51.6deg mu=3.986004418e14; "
                    "Astropy get_sun + existing ppigrf IGRF14/frame helpers; 1s interpolation; "
                    "body C_BN; gyro rad/s, magnetic T, sun unit direction; "
                    "artificial eclipse windows",
                "source_sha256": {p.relative_to(root).as_posix():
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, default=list))
    environment = environment or NavigationHistory(duration)
    rng = np.random.default_rng(20260930) if config.sensors == "noisy" else None
    times = list(np.arange(0.0, duration, config.dt_s)) + [duration]
    max_rate = np.zeros(3)
    max_wheel = np.zeros(3)
    max_seam = 0.0
    eclipse_max = 0.0
    eclipse_samples = 0
    reacquisition_error = None
    was_eclipse = False
    from pocketadcs_sim.adcs.rotors import TGR2912
    xc = f"XC {config.proportional_gain} {config.derivative_gain} {TGR2912.continuous_torque_nm} " \
         f"{board.wheel_set.acceleration_cap_rad_s2} " \
         f"{board.wheel_set.inertia} {board.max_speed_rad_s}"
    counter_hz = None
    with (directory / "wire.jsonl").open("w", encoding="utf-8") as wire, \
         (directory / "timing.jsonl").open("w", encoding="utf-8") as timing, \
         (directory / "samples.jsonl").open("w", encoding="utf-8") as samples:
        def record(direction, line):
            wire.write(json.dumps({"direction": direction, "line": line}) + "\n")
        def receive(deadline):
            while True:
                line = link.read_reply(deadline)
                if line is None:
                    return None
                record("rx", line)
                if line.startswith("R -2 "):
                    continue
                return line
        def exchange(line):
            record("tx", line)
            # QPC-backed high-resolution clock on Windows. This is application
            # transport exchange RTT, never device service/algorithm time.
            started_ns = time.perf_counter_ns()
            link.send(line)
            answer = receive(time.monotonic() + 5)
            elapsed_ns = time.perf_counter_ns() - started_ns
            if answer is None:
                raise ProtocolError(f"missing reply for {line.split()[0]}")
            entry = {"command": line.split()[0], "host_exchange_rtt_ns": elapsed_ns,
                     "counter_hz": counter_hz, "stage_cycles": None,
                     "service_cycles": None, "stage_us": None, "service_us": None}
            if answer.startswith("Y "):
                measured = parse_y(answer)
                entry["seq"] = measured.seq
                if counter_hz is not None:
                    converted = measured.cycle_times_us(counter_hz)
                    entry.update(stage_cycles=[int(v) for v in measured.cycles],
                                 service_cycles=measured.service_cycles,
                                 stage_us=converted["stages"], service_us=converted["service"])
            timing.write(json.dumps(entry) + "\n")
            return answer
        if getattr(link, "device_timing", False):
            identity = parse_identity(exchange("XI"))
            counter_hz = identity["counter_hz"]
            manifest["timing"]["device_identity"] = identity
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, default=list))
        # run_loop creates a fresh isolated plant/board world. Explicitly reset
        # device mission state first; XC and live plan changes do NOT do this.
        for line in ["XZ", xc, *instructions]:
            tokens = exchange(line).split()
            if len(tokens) != 5 or tokens[:2] != ["E", line.split()[0]]:
                raise ProtocolError(f"configuration/plan ACK mismatch: {tokens}")
            try:
                status, ack_seq, has_fix = (int(token) for token in tokens[2:])
            except ValueError as exc:
                raise ProtocolError(f"non-integer configuration/plan ACK: {tokens}") from exc
            if status != 0 or ack_seq != -1 or has_fix != 0:
                raise ProtocolError(f"invalid fresh-session configuration/plan ACK: {tokens}")
        for seq, t_s in enumerate(times, 1):
            eclipse = any(a <= t_s <= b for a, b in config.eclipse_windows)
            gyro, magnetic, sun = measure(environment, t_s, plant.attitude, plant.body_rate,
                                          eclipse, fix_bias_rad, rng)
            # Compact doubles fit the actual 320-byte input bound. Epoch/seq
            # remain decimal integers; never introduce field-shifting decimals.
            fields = [t_s, *gyro, *magnetic, *sun, *environment.position(t_s)]
            line = f"XS2 {seq} {EPOCH_START_UNIX_MS + round(t_s * 1000)} " + \
                   " ".join(f"{v:.12g}" for v in fields)
            if len(line) >= 320:
                raise ProtocolError("sensor frame exceeds actual device line capacity")
            reply = parse_y(exchange(line))
            if reply.seq != seq or reply.status != 0 or reply.stale or not reply.has_fix:
                raise ProtocolError(f"refused/unmatched sample {seq}: {reply}")
            if not eclipse and reply.reject != 0:
                raise ProtocolError(f"unexpected sensor rejection at sample {seq}")
            cap = board.wheel_set.inertia * board.wheel_set.acceleration_cap_rad_s2
            if np.any(np.abs(reply.wheel_torque) > cap + 1e-9):
                raise ProtocolError("device wheel command exceeds configured acceleration cap")
            seam = quaternion_error_deg(reply.control_input, plant.attitude)
            max_seam = max(max_seam, seam)
            if not np.allclose(reply.control_input, reply.estimate, atol=1e-9, rtol=0):
                raise ProtocolError("controller did not consume reported estimate")
            if eclipse:
                eclipse_samples += 1
                eclipse_max = max(eclipse_max, seam)
                if reply.reject != 1:
                    raise ProtocolError("absolute fix accepted during eclipse")
            elif was_eclipse:
                reacquisition_error = seam
                if reply.reject != 0:
                    raise ProtocolError("no absolute reacquisition after eclipse")
            was_eclipse = eclipse
            dt = times[seq] - t_s if seq < len(times) else 0.0
            truth = plant.attitude.copy()
            applied = board.update(reply.wheel_torque, dt) if dt else np.zeros(3)
            if dt and not np.array_equal(board.commanded_torque_nm, reply.wheel_torque):
                raise ProtocolError("board consumed a host-substituted wheel command")
            samples.write(json.dumps({"seq": seq, "time_s": t_s,
                "pre_actuation_truth": truth.tolist(), "estimate": reply.estimate.tolist(),
                "control_input": reply.control_input.tolist(), "seam_error_deg": seam,
                "eclipse": eclipse, "body_rate_rad_s": plant.body_rate.tolist(),
                "wheel_speed_rad_s": plant.wheel_speed.tolist(),
                "device_wheel_torque_nm": reply.wheel_torque.tolist(),
                "applied_wheel_torque_nm": applied.tolist()}) + "\n")
            if dt:
                advance_world(plant, applied, dt)
            max_rate = np.maximum(max_rate, np.abs(plant.body_rate))
            max_wheel = np.maximum(max_wheel, np.abs(plant.wheel_speed))
        # Read beyond the final expected seq: a duplicate is still a failure.
        extra = receive(time.monotonic() + 0.2)
        if extra is not None:
            raise ProtocolError(f"unexpected/duplicate trailing reply: {extra}")
    summary = {"frames": len(times), "duration_s": duration,
        "final_target_error_deg": quaternion_error_deg(plant.attitude, target),
        "final_truth_attitude": plant.attitude.tolist(), "commanded_target": target.tolist(),
        "max_controller_input_vs_truth_deg": max_seam,
        "max_abs_rate_rad_s": max_rate.tolist(), "max_abs_wheel_speed_rad_s": max_wheel.tolist(),
        "eclipse_samples": eclipse_samples, "eclipse_max_seam_error_deg": eclipse_max,
        "reacquisition_seam_error_deg": reacquisition_error,
        "stage_execution_time": None, "device": link.provenance}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def verdict(summary, tolerance_deg):
    error = summary["final_target_error_deg"]
    return 0 if np.isfinite(error) and error < tolerance_deg else 2


def main(argv=None):
    from examples.hil_closed_loop import ProtocolError
    parser = argparse.ArgumentParser(description=__doc__)
    link_group = parser.add_mutually_exclusive_group(required=True)
    link_group.add_argument("--offline", action="store_true")
    link_group.add_argument("--port")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slewos-run-dir")
    parser.add_argument("--case-id")
    parser.add_argument("--method", default="DO")
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--start", type=float, nargs=4, default=[1, 0, 0, 0])
    parser.add_argument("--target", type=float, nargs=4, default=[1, 0, 0, 0])
    parser.add_argument("--eclipse", type=float, nargs=2)
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument("--fix-bias-deg", type=float, default=0)
    parser.add_argument("--target-tolerance-deg", type=float, default=0.1)
    args = parser.parse_args(argv)
    try:
        output = Path(args.output_dir).resolve()
        if output.exists():
            raise FileExistsError(f"refusing overwrite: {output}")
        if args.slewos_run_dir and not args.case_id:
            raise ValueError("--case-id is required for a SlewOS run")
        config = ClosedLoopConfig(duration_s=args.duration, start_attitude=args.start,
            target_attitude=args.target, slewos_run_dir=args.slewos_run_dir,
            slewos_case_id=args.case_id, slewos_method=args.method,
            eclipse_windows=(tuple(args.eclipse),) if args.eclipse else (),
            sensors="noisy" if args.noisy else "ideal")
        link = (StdioLink(build_offline(output.with_name(output.name + "_build")))
                if args.offline else SerialLink(args.port))
        with link:
            summary = run_loop(link, config, output, fix_bias_rad=np.radians(args.fix_bias_deg))
        code = verdict(summary, args.target_tolerance_deg)
        print(json.dumps(summary, indent=2))
        print("PASS" if code == 0 else "MISMATCH: final commanded target criterion failed")
        return code
    except ProtocolError as exc:
        print(f"MISMATCH: {exc}")
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"SETUP FAILURE: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
