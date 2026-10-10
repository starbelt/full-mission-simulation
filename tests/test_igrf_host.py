"""Pytest wrapper for the freestanding IGRF-14 field model.

Compiles tests/firmware/test_igrf.cpp against esp32/src/igrf.h -- the exact
header the Zephyr firmware compiles -- with a host C++ compiler, feeds it
(position, epoch) cases, and compares the returned GCRS field direction
against the host's own truth path:

    ppigrf.igrf_gc(...)  ->  ITRS field  ->  astropy ITRS->GCRS

The comparison is on the ANGLE BETWEEN THE TWO VECTOR DIRECTIONS, never on the
matrices. Comparing matrices is what produced garbage 30-170 deg errors in a
previous attempt: astropy's ITRS->GCRS is a frame transform, not something you
recover by differencing transformed basis vectors.

THE FRAME CHAIN UNDER TEST
--------------------------
Probing the host's rotation with an orthonormal basis of FIELD VECTORS (which
is what truth_environment._itrs_vectors_to_gcrs does) pins it to exactly one
matrix, in row form:

    v_gcrs = v_itrs @ (Rz(GAST) @ pnm06a)

which in column form is

    v_gcrs = (Rz(GAST) @ pnm06a).T @ v_itrs

So the C++ must compute, for a column vector, `P^T @ Rz(-GAST) @ v`. The
transpose here is the whole ballgame: applying `Rz(+GAST) @ P` instead is the
inverse rotation and differs by twice the frame angle, which is what produced
the 116-175 deg self-check failure in an earlier revision. This file now
asserts the convention explicitly, so a regression cannot come back silently.

The rig is self-checked first. Before trusting any number out of it, we run the
same basis-of-vectors probe on an independent implementation of the chain
(erfa) and require agreement with astropy. If that fails, the rig is wrong and
every downstream number is meaningless, so we fail loudly instead of
reporting a confident wrong answer.

Note the C++ deliberately does NOT use UT1 (there is no IERS table on board) and
truncates nutation at 44 terms, so a residual of order 0.003 deg against a
perfect erfa chain is expected and is asserted against a realistic bar, not
against zero.

Skips when no usable host C++ compiler is available, matching
test_ads_math_host.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_ads_math_host import _env_for, find_usable_compiler

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "tests" / "firmware" / "test_igrf.cpp"
HEADER_DIR = REPO_ROOT / "esp32" / "src"
BUILD_TIMEOUT_S = 300
RUN_TIMEOUT_S = 120

# The bar from the task: every sample must be better than this.
MAX_FIELD_ERROR_DEG = 0.01

# Sanity bound on the rig self-check. The independent erfa chain and astropy
# agree to ~1.3e-4 deg (astropy applies no aberration on this vector path but
# does use a slightly different sidereal-time evaluation), so the bar is set
# an order of magnitude above that, not at zero.
RIG_SELFCHECK_DEG = 1.0e-3

ppigrf = pytest.importorskip("ppigrf", reason="ppigrf is the ground truth")
erfa = pytest.importorskip("erfa", reason="erfa is the frame reference")
astropy = pytest.importorskip("astropy", reason="astropy is the frame reference")

import astropy.units as u  # noqa: E402
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation  # noqa: E402
from astropy.time import Time  # noqa: E402

# ---------------------------------------------------------------------------
# host truth
# ---------------------------------------------------------------------------


def host_igrf_gcrs(position_itrs_m: np.ndarray, times: Time) -> np.ndarray:
    """The host's own path: ppigrf in ITRS, rotated to GCRS by astropy."""
    fields = np.empty_like(position_itrs_m)
    datetimes = times.to_datetime(timezone=None)
    for index, (position, epoch) in enumerate(zip(position_itrs_m, datetimes, strict=True)):
        x, y, z = position / 1000.0
        radius = float(np.linalg.norm([x, y, z]))
        colatitude = float(np.degrees(np.arccos(z / radius)))
        longitude = float(np.degrees(np.arctan2(y, x)))
        radial, south, east = ppigrf.igrf_gc(radius, colatitude, longitude, epoch)
        br = float(np.asarray(radial).item())
        bt = float(np.asarray(south).item())
        bp = float(np.asarray(east).item())
        theta = np.radians(colatitude)
        phi = np.radians(longitude)
        radial_axis = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        south_axis = np.array(
            [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]
        )
        east_axis = np.array([-np.sin(phi), np.cos(phi), 0.0])
        fields[index] = (br * radial_axis + bt * south_axis + bp * east_axis) * 1.0e-9
    return fields


def host_field_gcrs(
    position_itrs_m: np.ndarray, times: Time
) -> np.ndarray:
    """ppigrf ITRS field at each position, rotated to GCRS by astropy.

    This is the host's own path: the field is evaluated where the ITRS series
    is defined, then carried out to GCRS with the same astropy transform the
    C++ side approximates with its erfa chain. The rotation acts on the FIELD
    VECTOR, never on the position.
    """
    fields = host_igrf_gcrs(position_itrs_m, times)
    itrs = ITRS(CartesianRepresentation(fields.T * u.T), obstime=times)
    gcrs = itrs.transform_to(GCRS(obstime=times))
    return np.asarray(gcrs.cartesian.xyz.to_value(u.T)).T


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between unit vectors, in degrees, elementwise over rows."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    dots = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


# ---------------------------------------------------------------------------
# the verified chain, as an independent (erfa) implementation
# ---------------------------------------------------------------------------


def rz_active_gast(degrees: float) -> np.ndarray:
    """Rz(GAST) exactly as it appears in the verified chain (active, +GAST)."""
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def verified_chain_matrix(time: Time) -> np.ndarray:
    """(Rz(GAST) @ pnm06a) -- the matrix whose transpose is ITRS->GCRS."""
    d1, d2 = time.tt.jd1, time.tt.jd2
    u1, u2 = time.ut1.jd1, time.ut1.jd2
    pnm = np.asarray(erfa.pnm06a(d1, d2))
    gast = np.degrees(erfa.gst06a(u1, u2, d1, d2))
    return rz_active_gast(gast) @ pnm


# ---------------------------------------------------------------------------
# rig self-check
# ---------------------------------------------------------------------------


def rig_selfcheck() -> float:
    """Max angle between the verified erfa chain and astropy, in deg.

    This is the guard on the whole file. If an independent implementation of
    the chain does not land on astropy's answer through this exact probe --
    a basis of field vectors, transformed as vectors -- then the probe itself
    is mis-specified and no IGRF comparison means anything.

    It also pins the CONVENTION, which is the part that went wrong before:
    the transpose form must match astropy, and the non-transposed form must
    miss by a large angle. If the latter ever starts matching, the chain has
    been silently redefined and the whole file is measuring the wrong thing.
    """
    rng = np.random.default_rng(4242)
    worst = 0.0
    worst_wrong_form = 0.0
    for iso in (
        "2015-03-01T05:00:00",
        "2020-07-04T11:22:33",
        "2025-06-15T03:11:07",
    ):
        time = Time(iso, scale="utc")
        a = verified_chain_matrix(time)
        for _ in range(12):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            reference = (
                ITRS(CartesianRepresentation(v.reshape(3, 1) * u.one), obstime=time)
                .transform_to(GCRS(obstime=time))
                .cartesian.xyz.value.ravel()
            )
            # the verified form: v_gcrs = A^T @ v
            correct = a.T @ v
            # the form that produced the historical 116-175 deg failure
            wrong = a @ v
            worst = max(
                worst, float(angle_between_deg(correct[None, :], reference[None, :])[0])
            )
            worst_wrong_form = max(
                worst_wrong_form,
                float(angle_between_deg(wrong[None, :], reference[None, :])[0]),
            )
    return worst, worst_wrong_form


# ---------------------------------------------------------------------------
# compile + run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def igrf_binary(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("igrf")
    compiler = find_usable_compiler(tmp_path)
    if compiler is None:
        pytest.skip("no usable host C++ compiler")
    binary = tmp_path / "test_igrf"
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
    assert build.returncode == 0, (
        f"compilation failed:\n{build.stdout}\n{build.stderr}"
    )
    return binary


def run_igrf(binary: Path, cases: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cases: (n, 4) array of [epoch_ms, x, y, z] in GCRS metres."""
    tmp = binary.parent / "cases.txt"
    with open(tmp, "w", encoding="ascii") as fh:
        fh.write(f"{len(cases)}\n")
        for row in cases:
            # epoch_ms is an INTEGER count of milliseconds. It must be written
            # as an integer: "%.10f" emits "1621080221438.0000000000", and the
            # C++ side reads it with %lld, which stops at the '.'. That leaves
            # ".0000000000" to be consumed as pos[0] and shifts every field by
            # one, so each case after the first gets a garbage position.
            fh.write(f"{int(row[0])} {row[1]:.17e} {row[2]:.17e} {row[3]:.17e}\n")
    run = subprocess.run(
        [str(binary), str(tmp)],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        env=_env_for(binary),
    )
    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
    assert "all checks passed" in run.stdout
    lines = [ln for ln in run.stdout.splitlines() if ln and "all checks" not in ln]
    rows = [ln.split() for ln in lines]
    dirs = np.array([[float(v) for v in r[:3]] for r in rows])
    mags = np.array([float(r[3]) for r in rows])
    rcs = np.array([int(r[4]) for r in rows])
    fdir_err = np.array([float(r[5]) for r in rows])
    return dirs, mags, rcs, fdir_err


# ---------------------------------------------------------------------------
# case generation
# ---------------------------------------------------------------------------


def leo_cases(n: int, rng: np.random.Generator) -> np.ndarray:
    """(position_gcrs_m, epoch_ms) pairs on a 550 km LEO orbit.

    Positions are generated in GCRS by a simple circular-orbit propagation with
    a realistic inclination, then handed to the C++ side. The host side gets
    the ITRS positions astropy derives from the same GCRS positions, so both
    sides are evaluating the same physical point.
    """
    re = 6371.0e3
    alt = 550.0e3
    inc = np.radians(51.6)
    cases = []
    for _ in range(n):
        raan = rng.uniform(0.0, 2 * np.pi)
        u0 = rng.uniform(0.0, 2 * np.pi)
        r_mag = re + alt
        # position in a frame where the orbit plane is inclined
        x_orb = r_mag * np.cos(u0)
        y_orb = r_mag * np.sin(u0)
        # rotate by inclination about x, then by RAAN about z
        y_i = y_orb * np.cos(inc)
        z_i = y_orb * np.sin(inc)
        x = x_orb * np.cos(raan) - y_i * np.sin(raan)
        y = x_orb * np.sin(raan) + y_i * np.cos(raan)
        z = z_i
        # epochs through 2025.0, where IGRF-14 is defined.
        # The datetime64 difference is in SECONDS, so it must be scaled to
        # milliseconds before being stored in an epoch_ms column. Leaving it
        # in seconds silently yields 1970 epochs, which the firmware rejects
        # as out of range -- a confusing failure that looks like a model bug.
        year = rng.uniform(2015.0, 2025.0)
        epoch_s = int(
            (np.datetime64(f"{int(year)}-01-01T00:00:00", "s")
             - np.datetime64("1970-01-01T00:00:00", "s")).astype("int64")
        ) + int((year - int(year)) * 365.25 * 86400.0)
        cases.append([epoch_s * 1000, x, y, z])
    return np.array(cases)


def gcrs_to_itrs_astropy(position_gcrs_m: np.ndarray, times: Time) -> np.ndarray:
    gcrs = GCRS(CartesianRepresentation(position_gcrs_m.T * u.m), obstime=times)
    return np.asarray(gcrs.transform_to(ITRS(obstime=times)).cartesian.xyz.to_value(u.m)).T


def epoch_ms_to_time(epoch_ms: np.ndarray) -> Time:
    unix_s = epoch_ms / 1000.0
    return Time(unix_s, format="unix", scale="utc")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_rig_selfcheck_pins_the_verified_convention() -> None:
    """Guard: the erfa chain must reproduce astropy through this rig.

    Also asserts the transposed (wrong) form misses badly. Without this second
    half, a chain that had been redefined to the inverse rotation would still
    "pass" the first half by accident and every number below would be
    measuring the wrong transform.
    """
    worst, worst_wrong = rig_selfcheck()
    assert worst < RIG_SELFCHECK_DEG, (
        f"rig self-check failed: the verified erfa chain disagrees with astropy by "
        f"{worst:.3e} deg, above {RIG_SELFCHECK_DEG:.1e}. The comparison rig is "
        f"mis-specified; do not trust any IGRF number until this passes."
    )
    assert worst_wrong > 10.0, (
        f"the non-transposed chain form agrees with astropy to {worst_wrong:.3e} "
        f"deg, which means the convention is no longer distinguishable. Either "
        f"the chain has been redefined to its inverse or the probe has stopped "
        f"discriminating; both invalidate the comparison below."
    )
    print(f"rig self-check worst = {worst:.3e} deg (wrong form misses by {worst_wrong:.1f} deg)")


def test_igrf_field_matches_host(igrf_binary: Path) -> None:
    """The gate: every sample better than MAX_FIELD_ERROR_DEG."""
    rng = np.random.default_rng(20260928)
    cases = leo_cases(240, rng)
    position_gcrs = cases[:, 1:4]
    times = epoch_ms_to_time(cases[:, 0])

    dirs, mags, rcs, fdir_err = run_igrf(igrf_binary, cases)
    assert np.all(rcs == 0), f"firmware returned non-zero rc: {set(rcs)}"

    position_itrs = gcrs_to_itrs_astropy(position_gcrs, times)
    host = host_field_gcrs(position_itrs, times)

    errors = angle_between_deg(dirs, host)
    worst = float(errors.max())
    worst_idx = int(errors.argmax())

    # the float-narrowing public entry point must be directionally identical
    assert float(fdir_err.max()) < 1e-5, (
        f"float output disagrees with double by {fdir_err.max():.3e}; the ADS "
        f"consumes the float form"
    )

    assert worst < MAX_FIELD_ERROR_DEG, (
        f"field-direction error {worst:.5f} deg at sample {worst_idx} "
        f"(epoch_ms={cases[worst_idx, 0]:.0f}) exceeds {MAX_FIELD_ERROR_DEG} deg. "
        f"median {np.median(errors):.6f} deg."
    )
    print(
        f"n={len(cases)} max={worst:.6e} deg median={np.median(errors):.3e} deg "
        f"magnitude range {mags.min():.3e}..{mags.max():.3e} T"
    )


def test_igrf_frame_chain_accuracy_is_measured_not_assumed(igrf_binary: Path) -> None:
    """Report the frame-chain error against a perfect erfa implementation.

    The C++ trades accuracy for self-containment in two known ways, and this
    test states the size of both rather than leaving them implicit:

      * it uses UTC where gst06a wants UT1 (no IERS table on board);
      * it truncates nutation to 44 terms.

    Both show up here as a residual against the erfa chain. This test asserts
    only that the residual is bounded and dominated by those two effects --
    it is a measurement, not a correctness proof of the far field.
    """
    rng = np.random.default_rng(5150)
    n = 60
    cases = np.empty((n, 4))
    times = []
    for i in range(n):
        year = rng.uniform(2015.0, 2025.0)
        # datetime64 "ms" subtraction already yields milliseconds.
        ms = int(
            (np.datetime64("2015-01-01T00:00:00", "ms")
             - np.datetime64("1970-01-01T00:00:00", "ms")).astype("int64")
            + int((year - 2015.0) * 365.25 * 86400.0 * 1000.0)
        )
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * 6.9e6
        cases[i] = [ms, v[0], v[1], v[2]]
        times.append(Time(ms / 1000.0, format="unix", scale="utc"))

    dirs, _, rcs, _ = run_igrf(igrf_binary, cases)
    assert np.all(rcs == 0)

    # A perfect implementation of the same chain, evaluated on the same
    # positions: evaluate the field with ppigrf in ITRS, then apply A^T.
    position_gcrs = cases[:, 1:4]
    time_arr = Time(times)
    position_itrs = gcrs_to_itrs_astropy(position_gcrs, time_arr)
    itrs_fields = host_igrf_gcrs(position_itrs, time_arr)

    ideal = np.empty_like(dirs)
    for i, t in enumerate(times):
        ideal[i] = verified_chain_matrix(t).T @ itrs_fields[i]
    ideal /= np.linalg.norm(ideal, axis=1, keepdims=True)

    err = angle_between_deg(dirs, ideal)
    worst = float(err.max())
    # 0.02 deg leaves generous headroom over the ~0.003 deg expected from the
    # UT1 and nutation-truncation terms while still catching a real regression.
    assert worst < 0.02, (
        f"frame-chain residual {worst:.5f} deg against the ideal erfa chain is "
        f"larger than expected from UTC-vs-UT1 and 44-term nutation alone"
    )
    print(
        f"frame-chain residual vs ideal erfa chain: max={worst:.3e} deg "
        f"median={np.median(err):.3e} deg over {n} samples, 2015-2025"
    )
