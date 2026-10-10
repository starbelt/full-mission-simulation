"""Missing ephemeris must never feed uninitialized reference vectors to TRIAD."""
import subprocess
from pathlib import Path

import pytest
from test_ads_math_host import HEADER_DIR, _env_for, find_usable_compiler

SOURCE = r"""
#include <cmath>
#include "ads_estimate.h"
int main() {
    SensorSample s = {};
    s.gyro_rad_s[0] = 0.01f;
    s.mag_ut[0] = 20.0f;
    s.sun_unit[1] = 1.0f;
    s.flags = kSensorGyro | kSensorMag | kSensorSun;
    AdsFrameResult r;
    unsigned flags = 0;
    estimateSample(s, r, flags);
    if (r.valid || !(flags & 256u) || !(flags & 16u)) return 1;
    if (r.q[0] != 1.0f || r.q[1] != 0.0f) return 2;
    s.ref_sun_unit[1] = 1.0f;
    s.ref_mag_unit[0] = 1.0f;
    s.flags |= kSensorRefSun | kSensorRefMag;
    estimateSample(s, r, flags);
    if (!r.valid || (flags & (256u | 16u))) return 3;
    return 0;
}
"""


def test_missing_references_cannot_yield_valid_attitude(tmp_path: Path):
    compiler = find_usable_compiler(tmp_path)
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    src, exe = tmp_path / "test.cpp", tmp_path / "test.exe"
    src.write_text(SOURCE, encoding="ascii")
    cmd = [str(compiler), "-std=c++17", "-O2", "-I", str(HEADER_DIR),
           str(src), "-o", str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True,
                           env=_env_for(compiler), timeout=120)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True,
                         env=_env_for(compiler), timeout=30)
    assert run.returncode == 0, run.stderr
