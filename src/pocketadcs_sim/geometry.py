"""Earth-fixed targets and pointing geometry.

Geodesy and reference-frame transformations are delegated to Astropy/ERFA rather
than reproduced locally.
"""

from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
import pymap3d as pm
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
from astropy.time import Time
from astropy.units import Quantity


@dataclass(frozen=True, slots=True)
class EarthTarget:
    """A stationary target specified in WGS84 geodetic coordinates."""

    latitude: Quantity
    longitude: Quantity
    height: Quantity = field(default_factory=lambda: 0.0 * u.m)
    location: EarthLocation = field(init=False, repr=False)

    def __post_init__(self) -> None:
        location = EarthLocation.from_geodetic(
            lon=self.longitude,
            lat=self.latitude,
            height=self.height,
            ellipsoid="WGS84",
        )
        object.__setattr__(self, "location", location)


@dataclass(frozen=True, slots=True)
class TargetObservation:
    """Geometric observation of an Earth target from a spacecraft."""

    line_of_sight_gcrs: Quantity
    azimuth: Quantity
    elevation: Quantity
    range: Quantity
    visible: bool


def target_line_of_sight_gcrs(
    spacecraft_position: Quantity,
    target: EarthTarget,
    obstime: Time,
) -> Quantity:
    """Return the unit vector from the spacecraft to the target in GCRS."""

    target_position = target.location.get_gcrs(obstime).cartesian.xyz
    displacement = target_position - spacecraft_position
    return displacement / np.linalg.norm(displacement)


def observe_target(
    spacecraft_position: Quantity,
    target: EarthTarget,
    obstime: Time,
) -> TargetObservation:
    """Return WGS84 azimuth, elevation, range, visibility, and GCRS line of sight."""

    spacecraft_gcrs = GCRS(
        CartesianRepresentation(spacecraft_position),
        obstime=obstime,
    )
    spacecraft_itrs = spacecraft_gcrs.transform_to(ITRS(obstime=obstime))
    x, y, z = spacecraft_itrs.cartesian.xyz.to_value(u.m)
    azimuth, elevation, slant_range = pm.ecef2aer(
        x,
        y,
        z,
        target.latitude.to_value(u.deg),
        target.longitude.to_value(u.deg),
        target.height.to_value(u.m),
        deg=True,
    )
    return TargetObservation(
        line_of_sight_gcrs=target_line_of_sight_gcrs(spacecraft_position, target, obstime),
        azimuth=azimuth * u.deg,
        elevation=elevation * u.deg,
        range=slant_range * u.m,
        visible=bool(elevation >= 0.0),
    )


def is_target_visible(
    spacecraft_position: Quantity,
    target: EarthTarget,
    obstime: Time,
) -> bool:
    """Return whether the spacecraft is above the target's WGS84 horizon."""

    return observe_target(spacecraft_position, target, obstime).visible

