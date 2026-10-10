"""Onboard attitude estimator: gyro propagation plus TRIAD absolute fixes.

This is the component that must earn its accuracy by computing rather than by
being told. It receives only what a real ADS has:

* a gyro body rate,
* body-frame measurements of two inertial reference vectors (sun, magnetic),
* per-sensor validity.

It never receives the truth attitude, and no scoring happens here. That
separation is the whole point of the prototype, so it is enforced by the module
having no import of, or parameter for, a truth state.

Method
------
Absolute fixes use **TRIAD**, delegated to the vetted reference implementation
(``ahrs.filters.TRIAD``) rather than re-derived here, matching
:mod:`pocketadcs_sim.attitude.triad`. Between fixes the estimate propagates with
the gyro under the exact quaternion exponential
(:func:`pocketadcs_sim.adcs.quaternions.propagate`).

Validity handling follows the same rules the firmware already uses:

* a zero-length vector is *no data* (eclipse), not a real measurement, and is
  never propagated into the estimate;
* a parallel sun/magnetic pair makes TRIAD singular and is skipped;
* a non-monotonic timestamp is refused outright rather than propagated backwards.

A skipped fix leaves the estimate coasting on the gyro, which is exactly the
eclipse case: the estimate degrades gracefully and snaps back when the sun
returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from ahrs.filters import TRIAD

from pocketadcs_sim.adcs import quaternions as quat

__all__ = ["AbsoluteUpdate", "OnboardAttitudeEstimator"]

FloatArray = npt.NDArray[np.float64]

#: Below this angle the two reference vectors are too close to define attitude.
_PARALLEL_TOLERANCE = 1.0 - 1e-9


@dataclass(frozen=True)
class AbsoluteUpdate:
    """Body-frame measurements of the two inertial reference vectors.

    ``sun_body`` and ``magnetic_body`` are as measured *in body axes*;
    ``sun_ref_gcrs`` and ``magnetic_ref_gcrs`` are the corresponding inertial
    reference directions the vehicle computes from its ephemeris and time --
    knowledge a real vehicle has, and not a measurement of its own attitude.

    The reference vectors are supplied by the caller rather than defaulted to a
    fixed pair, because a TRIAD fix measures the *angle between* the two vectors:
    silently pairing a measurement with the wrong reference would produce a
    confident, wrong attitude rather than an error.

    ``sun_valid`` / ``magnetic_valid`` let the caller mark a sensor as dark or
    disconnected. A zero-length vector is additionally treated as invalid,
    because "no data" and "pointing nowhere" must not share an encoding.
    """

    sun_body: npt.ArrayLike
    magnetic_body: npt.ArrayLike
    sun_ref_gcrs: npt.ArrayLike
    magnetic_ref_gcrs: npt.ArrayLike
    sun_valid: bool = True
    magnetic_valid: bool = True


class OnboardAttitudeEstimator:
    """Gyro-propagating estimator with TRIAD absolute corrections."""

    def __init__(self) -> None:
        self.attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_time_s: float | None = None
        self.has_absolute_fix = False
        self.last_fix_rejected: str | None = None
        self.covariance_trace = 0.0

    @property
    def attitude_dcm(self) -> FloatArray:
        """The current ``C_BN`` estimate."""
        return quat.to_dcm(self.attitude)

    def update(
        self,
        time_s: float,
        gyro_body_rad_s: npt.ArrayLike,
        absolute: AbsoluteUpdate | None = None,
    ) -> None:
        """Advance the estimate to ``time_s``.

        Propagates from the previous timestamp with the gyro, then applies an
        absolute fix if one is supplied and usable. Order matters: the fix is
        applied at the new timestamp, after propagation, so the estimate is
        correct *at* the sample time rather than one step behind it.
        """
        now = float(time_s)
        if not np.isfinite(now):
            raise ValueError("time_s must be finite")
        rate = np.asarray(gyro_body_rad_s, dtype=float).reshape(3)
        if not np.all(np.isfinite(rate)):
            raise ValueError("gyro_body_rad_s must be finite")

        if self._last_time_s is not None:
            delta = now - self._last_time_s
            if delta < 0.0:
                # A reordered or stale sample: refuse it rather than integrating
                # the gyroscope backwards in time.
                raise ValueError(
                    f"non-monotonic time: {now} precedes last update {self._last_time_s}"
                )
            self.attitude = quat.propagate(self.attitude, rate, delta)

        self._last_time_s = now

        if absolute is not None:
            self._apply_absolute(absolute)

    def _apply_absolute(self, absolute: AbsoluteUpdate) -> None:
        """Apply a TRIAD fix if the measurements are usable; else coast."""
        self.last_fix_rejected = None

        sun_body = np.asarray(absolute.sun_body, dtype=float).reshape(3)
        magnetic_body = np.asarray(absolute.magnetic_body, dtype=float).reshape(3)
        sun_ref = np.asarray(absolute.sun_ref_gcrs, dtype=float).reshape(3)
        magnetic_ref = np.asarray(absolute.magnetic_ref_gcrs, dtype=float).reshape(3)
        for vector in (sun_body, magnetic_body, sun_ref, magnetic_ref):
            if not np.all(np.isfinite(vector)):
                raise ValueError("absolute measurements must be finite")

        sun_norm = float(np.linalg.norm(sun_body))
        magnetic_norm = float(np.linalg.norm(magnetic_body))
        sun_ref_norm = float(np.linalg.norm(sun_ref))
        magnetic_ref_norm = float(np.linalg.norm(magnetic_ref))

        if not absolute.sun_valid or sun_norm <= 0.0 or sun_ref_norm <= 0.0:
            self.last_fix_rejected = "sun invalid or zero-length (eclipse)"
            return
        if (
            not absolute.magnetic_valid
            or magnetic_norm <= 0.0
            or magnetic_ref_norm <= 0.0
        ):
            self.last_fix_rejected = "magnetic invalid or zero-length"
            return

        unit_sun = sun_body / sun_norm
        unit_magnetic = magnetic_body / magnetic_norm
        unit_sun_ref = sun_ref / sun_ref_norm
        unit_magnetic_ref = magnetic_ref / magnetic_ref_norm

        if (
            abs(float(np.dot(unit_sun_ref, unit_magnetic_ref)))
            >= _PARALLEL_TOLERANCE
            or abs(float(np.dot(unit_sun, unit_magnetic))) >= _PARALLEL_TOLERANCE
        ):
            self.last_fix_rejected = "reference pair is parallel; TRIAD singular"
            return

        triad = TRIAD()
        triad.v1 = unit_sun_ref
        triad.v2 = unit_magnetic_ref
        try:
            estimated = np.asarray(
                triad.estimate(w1=unit_sun, w2=unit_magnetic), dtype=float
            )
        except (ValueError, np.linalg.LinAlgError):
            self.last_fix_rejected = "TRIAD failed to converge"
            return

        self.attitude = quat.as_scalar_first(quat.from_dcm(np.asarray(estimated)))
        self.has_absolute_fix = True
        # A crude internal quality signal: how far the two measured vectors
        # were from being collinear. The estimator has no access to truth, so
        # this is the only disagreement it can actually measure.
        self.covariance_trace = float(1.0 - abs(np.dot(unit_sun, unit_magnetic)))
