"""The closed-loop HIL harness parses exactly what the firmware emits.

The harness's field positions are a copy of the firmware's ``snprintf`` format.
A copy drifts silently: a field added on one side and not the other raises an
exception at run time, in the lab, mid-run -- or worse, is absorbed by a tolerant
parser and mis-scored. So the parser is pinned against real output from the
header the device compiles.

This is the test that caught the harness expecting 24 fields where the firmware
emits 24 numbers plus a one-character tag.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from examples.hil_closed_loop import (
    ProtocolError,
    format_xs2,
    parse_y,
    plan_lines,
)
from tests.test_ads_math_host import HEADER_DIR, _env_for, find_usable_compiler

DRIVER = Path("tests/firmware/adcs_protocol_driver.cpp")


@pytest.fixture(scope="module")
def exe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("protocol_pin")
    compiler = find_usable_compiler(tmp)
    binary = tmp / "pin.exe"
    subprocess.run(
        [str(compiler), "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
         "-I", str(HEADER_DIR), str(DRIVER), "-o", str(binary), "-lm"],
        env=_env_for(compiler), check=True, capture_output=True,
    )
    return binary


def exchange(exe: Path, lines: list[str]) -> list[list[float]]:
    """Run a line script and return ``[type, *fields]`` per reply, as floats."""
    result = subprocess.run([str(exe)], input="\n".join(lines),
                            env=_env_for(find_usable_compiler(Path("."))),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return [[float(t) for t in line.split()] for line in result.stdout.split("\n") if line]


def load_lines() -> list[str]:
    """Configure, load a one-segment plan, then take two closed-loop samples."""
    sun = [1.0, 0.0, 0.0]
    mag = [0.0, 1.0, 0.0]
    pos = [6.9e6, 0.0, 0.0]
    return [
        "XC 0.35 0.15 0.0028 200 3.27e-6 1675.5",
        *plan_lines(0.0, [(0, 0.0, 0.5, 100.0, 0.0, 100.0, 0.0)]),
        format_xs2(1, 0.0, [0.0, 0.0, 0.0], mag, sun, pos),
        format_xs2(2, 0.05, [0.0, 0.0, 0.01], mag, sun, pos),
    ]


def test_firmware_y_reply_is_parsed_by_the_harness(exe: Path) -> None:
    """Every ``Y`` reply the device produces must survive ``parse_y``.

    This is the contract test: if the firmware's format string and the harness's
    parser ever disagree, it fails here rather than during a board run.
    """
    replies = [r for r in exchange(exe, load_lines()) if r[0] == 0.0]
    assert len(replies) == 2, f"expected two Y frames, got {len(replies)}: {replies}"
    for raw in replies:
        # Rebuild the wire line from the numeric echo, exactly as the port would
        # deliver it, and hand it to the harness parser.
        wire = "Y " + " ".join(repr(v) for v in raw[1:])
        reply = parse_y(wire)
        assert reply.status == 0
        assert reply.seq in (1, 2)
        assert reply.has_fix, "a lit sun and magnetometer must produce a fix"
        assert np.isclose(np.linalg.norm(reply.estimate), 1.0, atol=1e-6), (
            f"the device reported a non-unit quaternion: {reply.estimate}"
        )


def test_control_input_matches_the_estimate_the_device_reported(exe: Path) -> None:
    """The device must control from the attitude it reports.

    This is the ownership property at the executed seam. If the device computed
    torque from a different attitude than the one in the reply -- truth, say --
    the two quaternions would diverge. Equal is the pass condition, and it is
    checked against the device's own output rather than against the host.
    """
    replies = [r for r in exchange(exe, load_lines()) if r[0] == 0.0]
    for raw in replies:
        wire = "Y " + " ".join(repr(v) for v in raw[1:])
        reply = parse_y(wire)
        assert reply.control_input == pytest.approx(reply.estimate, abs=1e-9), (
            "the device controlled from an attitude other than the one it reported"
        )


def test_stage_cycles_are_reported_per_stage(exe: Path) -> None:
    """All four stages must be timed separately, not collapsed into one number.

    A single total hides which stage dominates, and the ephemeris is a model
    rather than the algorithm under test, so folding it in would inflate the ADS
    figure.
    """
    raw = [r for r in exchange(exe, load_lines()) if r[0] == 0.0][-1]
    reply = parse_y("Y " + " ".join(repr(v) for v in raw[1:]))
    assert reply.cycles.shape == (4,)
    assert all(c >= 0 for c in reply.cycles), (
        f"a stage reported a negative cycle count: {reply.cycles}"
    )
    # The estimator must do real work: an all-zero cycle count would mean the
    # timer is not wired up and every timing claim downstream is void.
    assert reply.cycles[1] > 0, f"the estimator stage timed zero cycles: {reply.cycles}"


def test_refusal_lines_are_not_mistaken_for_measurements(exe: Path) -> None:
    """An ``E`` reply must never parse as a ``Y`` frame.

    Silently accepting a refusal as a measurement is how a run reports a clean
    verdict over frames the device refused to compute.
    """
    replies = exchange(exe, [*load_lines(), "XS2 3 1e3"])
    refusals = [r for r in replies if r[0] == 1.0]
    assert refusals, "a malformed frame must produce a refusal"
    for raw in replies:
        if raw[0] != 0.0:
            with pytest.raises(ProtocolError):
                parse_y("Y " + " ".join(repr(v) for v in raw[1:]))


def test_exactly_one_reply_per_sent_frame(exe: Path) -> None:
    """Every frame sent gets exactly one reply -- not zero, not two.

    Zero replies means a queue overrun or a dead application; two means a
    duplicated measurement. Both were false PASSes in this rig.
    """
    sent = load_lines()
    frames = [line for line in sent if line.startswith("XS2 ")]
    replies = exchange(exe, sent)
    seqs = [int(r[2]) for r in replies if r[0] == 0.0]
    assert len(seqs) == len(frames) == 2
    assert sorted(seqs) == [1, 2], f"expected one reply per seq, got {seqs}"
