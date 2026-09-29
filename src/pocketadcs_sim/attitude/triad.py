"""TRIAD attitude estimation using the reference implementation from `ahrs`.

This module does NOT re-derive TRIAD. It calls the proven reference
implementation directly:

- Harold D. Black, "Note on the satellite attitude determination by the
  triad method", NASA Technical Note 1964.
- F. Lerner, "Spacecraft attitude determination via the triad method",
  J. Spacecraft 1978 — equations 12-39a..12-45.
- `ahrs.filters.triad.TRIAD` (jrl-umi3218/ahrs, v0.4.0),
  https://ahrs.readthedocs.io/en/latest/filters/triad.html
- https://en.wikipedia.org/wiki/Triad_method

Usage: reference vectors are the Sun direction and the geomagnetic field in
the GCRS frame; measured vectors are the body-frame Sun and MAG readings from
the ideal sensor models. The resulting DCM maps reference-frame components to
body-frame components, matching the truth history's convention (see
truth_environment: `Rotation.from_mrp(sigma).inv()`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from ahrs.filters import TRIAD
from scipy.spatial.transform import Rotation

from pocketadcs_sim.sensors.base import sample_history
from pocketadcs_sim.truth_environment import TruthEnvironmentHistory

FloatArray = npt.NDArray[np.float64]


def _angle_error_deg(dcm_est: FloatArray, dcm_truth: FloatArray) -> float:
    """Angle between two attitudes: acos((tr(A_est^T A_truth) - 1) / 2)."""
    rel = np.asarray(dcm_est).T @ np.asarray(dcm_truth)
    trace = np.trace(rel)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


@dataclass(frozen=True, slots=True)
class TriadEstimator:
    """Per-sample TRIAD attitude estimate from Sun + magnetometer vectors."""

    sample_rate_hz: float = 1.0

    def estimate(
        self,
        truth: TruthEnvironmentHistory,
    ) -> dict:
        """Run the reference TRIAD at each sampled time against truth vectors.

        Returns a dict with arrays: time_s, dcm (N,3,3), truth_dcm, error_deg.
        """
        indices = np.asarray(sample_history(truth, self.sample_rate_hz))
        triad = TRIAD()
        n = len(indices)
        dcm = np.empty((n, 3, 3))
        truth_dcm = np.empty((n, 3, 3))
        errors = np.empty(n)
        for i, index in enumerate(indices):
            triad.v1 = np.asarray(truth.sun_direction_gcrs[index])
            triad.v2 = np.asarray(truth.magnetic_field_gcrs_t[index])
            dcm[i] = triad.estimate(
                w1=np.asarray(truth.sun_direction_body[index]),
                w2=np.asarray(truth.magnetic_field_body_t[index]),
            )
            truth_dcm[i] = Rotation.from_mrp(truth.mrp_body_from_gcrs[index]).inv().as_matrix()
            errors[i] = _angle_error_deg(dcm[i], truth_dcm[i])
        return {
            "time_s": np.asarray(truth.time_s[indices], dtype=float),
            "dcm": dcm,
            "truth_dcm": truth_dcm,
            "error_deg": errors,
        }

    def estimate_from_readings(
        self,
        truth: TruthEnvironmentHistory,
        sun_body: npt.ArrayLike,
        mag_body: npt.ArrayLike,
    ) -> dict:
        """Run TRIAD on explicitly supplied body-frame sensor readings.

        This is the entry point the sensor-fidelity studies use: the
        measurements come from a hardware model (ICM-40609-D, MMC5983MA,
        VEMD8080 array) rather than from the truth history, so sensor noise
        and quantization actually reach the estimator.

        Samples that cannot produce an attitude are *skipped* rather than
        allowed to corrupt the result, mirroring the firmware's handling of
        a zero sun vector and a degenerate (parallel) reference pair:

        * a zero-length sun or magnetic reading (eclipse, or the array
          returning an unresolvable direction);
        * a sun/magnetic pair that is parallel, which makes TRIAD singular.

        Such samples get a NaN error and ``valid=False``, and the remaining
        samples are still estimated. The caller is expected to check
        ``valid`` rather than assume every sample produced a solution.

        Returns:
            dict with ``time_s``, ``dcm``, ``truth_dcm``, ``error_deg`` and
            ``valid`` arrays, one entry per sampled truth time.
        """
        sun = np.asarray(sun_body, dtype=float)
        mag = np.asarray(mag_body, dtype=float)
        if sun.shape != mag.shape:
            raise ValueError("sun_body and mag_body must have the same shape")
        if sun.ndim != 2 or sun.shape[1] != 3:
            raise ValueError("readings must have shape (N, 3)")
        if not np.all(np.isfinite(sun)) or not np.all(np.isfinite(mag)):
            raise ValueError("readings must be finite")

        indices = np.asarray(sample_history(truth, self.sample_rate_hz))
        n = len(indices)
        if sun.shape[0] != n:
            raise ValueError(
                f"expected {n} readings to match the sampled truth times, "
                f"got {sun.shape[0]}"
            )

        triad = TRIAD()
        dcm = np.full((n, 3, 3), np.nan)
        truth_dcm = np.full((n, 3, 3), np.nan)
        errors = np.full(n, np.nan)
        valid = np.zeros(n, dtype=bool)

        for i, index in enumerate(indices):
            w1 = sun[i]
            w2 = mag[i]
            n1 = float(np.linalg.norm(w1))
            n2 = float(np.linalg.norm(w2))
            if n1 <= 0.0 or n2 <= 0.0:
                # Eclipse / unresolvable sample: no attitude available.
                continue
            unit1 = w1 / n1
            unit2 = w2 / n2
            # A parallel reference pair makes TRIAD singular.
            if abs(float(np.dot(unit1, unit2))) >= 1.0 - 1e-9:
                continue

            triad.v1 = np.asarray(truth.sun_direction_gcrs[index], dtype=float)
            triad.v2 = np.asarray(truth.magnetic_field_gcrs_t[index], dtype=float)
            estimated = np.asarray(
                triad.estimate(w1=np.asarray(unit1, dtype=float), w2=np.asarray(unit2, dtype=float))
            )
            reference = Rotation.from_mrp(
                np.asarray(truth.mrp_body_from_gcrs[index], dtype=float)
            ).inv().as_matrix()
            dcm[i] = estimated
            truth_dcm[i] = reference
            errors[i] = _angle_error_deg(estimated, reference)
            valid[i] = True

        return {
            "time_s": np.asarray(truth.time_s[indices], dtype=float),
            "dcm": dcm,
            "truth_dcm": truth_dcm,
            "error_deg": errors,
            "valid": valid,
        }
