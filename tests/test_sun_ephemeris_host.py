"""Pin firmware Julian epoch conversion against a real calendar anchor."""
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_ads_math_host import _env_for, find_usable_compiler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sun_probe(tmp_path_factory):
    work = tmp_path_factory.mktemp("firmware_sun")
    compiler = find_usable_compiler(work)
    if compiler is None:
        pytest.skip("no host C++ compiler")
    binary = work / "sun_probe.exe"
    result = subprocess.run(
        [str(compiler), "-std=c++17", "-Wall", "-Wextra", "-Werror",
         f"-I{ROOT / 'esp32/src'}", str(ROOT / "tests/firmware/probe_sun_ephemeris.cpp"),
         "-o", str(binary)],
        capture_output=True, text=True, env=_env_for(compiler), timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    def probe(epoch_ms):
        run = subprocess.run([str(binary), str(epoch_ms)], capture_output=True,
                             text=True, env=_env_for(compiler), timeout=5, check=True)
        return np.fromstring(run.stdout, sep=" ")

    return probe


def test_j2000_anchor_is_zero_days_not_sixty_years_later(sun_probe):
    # 2000-01-01 12:00 UTC, independent Unix/calendar anchor.
    row = sun_probe(946728000000)
    assert row[0] == pytest.approx(0.0, abs=1e-10)


def test_unix_epoch_precedes_j2000(sun_probe):
    assert sun_probe(0)[0] == pytest.approx(-10957.5)


@pytest.mark.parametrize("date", ["2000-01-01T12:00:00", "2025-01-01", "2025-04-01",
                                  "2025-07-01", "2025-10-01", "2035-06-01"])
def test_sun_direction_is_gcrs_not_equator_of_date(sun_probe, date):
    from astropy.coordinates import get_sun
    from astropy.time import Time

    instant = Time(date, scale="utc")
    epoch_ms = int(round(float(instant.unix) * 1000))
    estimate = sun_probe(epoch_ms)[1:]
    truth = get_sun(instant).cartesian.xyz.value
    cosine = float(estimate @ truth / (np.linalg.norm(estimate) * np.linalg.norm(truth)))
    error_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    assert error_deg < 0.03, f"Sun reference error {error_deg:.6f} deg at {date}"
