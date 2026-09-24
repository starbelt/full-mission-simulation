import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from pocketadcs_sim.geometry import (
    EarthTarget,
    is_target_visible,
    observe_target,
    target_line_of_sight_gcrs,
)


def test_earth_target_uses_astropy_wgs84_geodesy() -> None:
    target = EarthTarget(latitude=40 * u.deg, longitude=-105 * u.deg, height=1_600 * u.m)

    expected = EarthLocation.from_geodetic(
        lon=-105 * u.deg,
        lat=40 * u.deg,
        height=1_600 * u.m,
        ellipsoid="WGS84",
    )

    assert u.allclose(target.location.geocentric, expected.geocentric)


def test_target_line_of_sight_is_a_unit_gcrs_vector() -> None:
    time = Time("2025-01-01T00:00:00", scale="utc")
    target = EarthTarget(0 * u.deg, 0 * u.deg)
    spacecraft = EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg, 500 * u.km)
    spacecraft_position = spacecraft.get_gcrs(time).cartesian.xyz

    line_of_sight = target_line_of_sight_gcrs(spacecraft_position, target, time)

    assert u.allclose((sum(line_of_sight**2)) ** 0.5, 1 * u.one)


def test_overhead_target_is_visible() -> None:
    time = Time("2025-01-01T00:00:00", scale="utc")
    target = EarthTarget(0 * u.deg, 0 * u.deg)
    spacecraft = EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg, 500 * u.km)
    spacecraft_position = spacecraft.get_gcrs(time).cartesian.xyz

    observation = observe_target(spacecraft_position, target, time)

    assert is_target_visible(spacecraft_position, target, time)
    assert observation.visible
    assert u.isclose(observation.elevation, 90 * u.deg)
    assert u.isclose(observation.range, 500 * u.km)


def test_target_beyond_earth_limb_is_not_visible() -> None:
    time = Time("2025-01-01T00:00:00", scale="utc")
    target = EarthTarget(0 * u.deg, 180 * u.deg)
    spacecraft = EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg, 500 * u.km)

    assert not is_target_visible(spacecraft.get_gcrs(time).cartesian.xyz, target, time)
