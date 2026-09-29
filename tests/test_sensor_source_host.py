"""Pytest wrapper for the host-compiled sensor-source abstraction test.

Compiles tests/firmware/test_sensor_source.cpp against
esp32/src/sensor_source.h -- the exact header the Zephyr firmware compiles --
with a host C++ compiler, then runs the binary.

The header deliberately has no Zephyr dependency, so the seam between the
estimator and its reading path is testable on the host before any I2C/ADC
driver exists. That is the point: the abstraction is the thing most likely to
be got wrong and the hardest to debug once drivers land.

Skips when no usable host C++ compiler is available, matching
test_ads_math_host.py.
"""

import subprocess
from pathlib import Path

import pytest
from test_ads_math_host import _env_for, find_usable_compiler

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "tests" / "firmware" / "test_sensor_source.cpp"
HEADER_DIR = REPO_ROOT / "esp32" / "src"
BUILD_TIMEOUT_S = 120
RUN_TIMEOUT_S = 60


def test_sensor_source_abstraction(tmp_path: Path) -> None:
    compiler = find_usable_compiler(tmp_path)
    if compiler is None:
        pytest.skip("no usable host C++ compiler")

    binary = tmp_path / "test_sensor_source"
    build = subprocess.run(
        [
            str(compiler),
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{HEADER_DIR}",
            str(SOURCE),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT_S,
        env=_env_for(compiler),
    )
    assert build.returncode == 0, f"compilation failed:\n{build.stdout}\n{build.stderr}"

    run = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        env=_env_for(compiler),
    )
    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
    assert "all checks passed" in run.stdout
