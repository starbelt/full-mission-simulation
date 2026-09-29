"""Pytest wrapper for the host-compiled ADS firmware regression test.

Compiles tests/firmware/test_ads_math.cpp against esp32/src/ads_math.h --
the exact header the ESP32-S3 HIL firmware compiles -- with a host C++
compiler, then runs the binary. The fixture checks, on serialized S-frame
values (seq 0, 1, 2, 10, 20, 30, 39), that:

  * the compute path does not mutate its input vectors (root cause of the
    original mag_norm == 1.0 regression),
  * the emitted mag_norm / sun_norm are the ORIGINAL magnitudes,
  * the emitted sun/mag angle matches the ahrs TRIAD reference,
  * the emitted quaternion matches chiaverini(triadDcm) within 0.1 deg.

Coverage note: the fixture embeds the frame arrays after serialization
(float32 values as the firmware receives them); it does not exercise the
serial parser in esp32/src/main.cpp.

Skips when no usable host C++ compiler (g++, clang++, or an MSYS2 toolchain)
is available on this machine.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_CPP = REPO_ROOT / "tests" / "firmware" / "test_ads_math.cpp"
HEADER_DIR = REPO_ROOT / "esp32" / "src"

MSYS2_ROOTS = [
    Path("/c/msys64"),
    Path("C:/msys64"),
    Path.home() / "msys64",
]
MSYS2_BIN_SUBDIRS = ("ucrt64", "mingw64", "clang64")
COMPILER_NAMES = ("g++", "clang++", "c++")
SMOKE_SOURCE = "int main() { return 0; }\n"
BUILD_TIMEOUT_S = 120
RUN_TIMEOUT_S = 60


def _compiler_candidates() -> list[Path]:
    """Candidate host C++ compiler binaries, in preference order."""
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path).lower()
        if path.is_file() and key not in seen:
            seen.add(key)
            candidates.append(path)

    for name in COMPILER_NAMES:
        found = shutil.which(name)
        if found:
            add(Path(found))

    for root in MSYS2_ROOTS:
        for sub in MSYS2_BIN_SUBDIRS:
            if sys.platform == "win32":
                add(root / sub / "bin" / "g++.exe")
                add(root / sub / "bin" / "clang++.exe")
            else:
                add(root / sub / "bin" / "g++")
                add(root / sub / "bin" / "clang++")

    return candidates


def _env_for(compiler: Path) -> dict[str, str]:
    """Environment that lets the compiler's frontend find its DLLs/libs.

    MSYS2's cc1plus.exe sits outside <prefix>/<sub>/bin and needs that
    directory on PATH to load its dependencies; on other platforms this is
    a harmless no-op.
    """
    env = os.environ.copy()
    env["PATH"] = str(compiler.parent) + os.pathsep + env.get("PATH", "")
    return env


def _smoke_compiles(compiler: Path, tmp_path: Path) -> bool:
    """Compile a trivial TU; catches broken/partial toolchains cheaply."""
    src = tmp_path / "smoke.cpp"
    exe_name = "smoke.exe" if sys.platform == "win32" else "smoke"
    src.write_text(SMOKE_SOURCE, encoding="ascii")
    try:
        proc = subprocess.run(
            [str(compiler), "-std=c++11", str(src), "-o", str(tmp_path / exe_name)],
            env=_env_for(compiler),
            capture_output=True,
            timeout=BUILD_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def find_usable_compiler(tmp_path: Path) -> Path | None:
    for compiler in _compiler_candidates():
        if _smoke_compiles(compiler, tmp_path):
            return compiler
    return None


def test_ads_math_host_regression(tmp_path: Path) -> None:
    compiler = find_usable_compiler(tmp_path)
    if compiler is None:
        pytest.skip("no usable host C++ compiler (g++/clang++/MSYS2) found")

    exe_name = "test_ads_math.exe" if sys.platform == "win32" else "test_ads_math"
    build = subprocess.run(
        [
            str(compiler),
            "-std=c++11",
            "-O2",
            "-I",
            str(HEADER_DIR),
            str(TEST_CPP),
            "-o",
            str(tmp_path / exe_name),
        ],
        env=_env_for(compiler),
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT_S,
    )
    if build.returncode != 0:
        pytest.fail(f"host build failed with {compiler}:\n{build.stderr}")

    run = subprocess.run(
        [str(tmp_path / exe_name)],
        env=_env_for(compiler),
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
    )
    if run.returncode != 0:
        pytest.fail(
            "host ADS regression test failed "
            f"(exit {run.returncode}):\n{run.stdout}{run.stderr}"
        )
    assert "OK:" in run.stdout and "frames passed" in run.stdout, (
        f"unexpected host test output: {run.stdout!r}"
    )
