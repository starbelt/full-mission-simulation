"""Attitude-accuracy and performance metrics for the PocketADCS study.

Everything here is a pure function of already-computed arrays so a study
runner can compose them freely and the results are reproducible and
citable: the same inputs always give the same numbers, and every metric
carries the full distribution rather than a single headline number.

Conventions
-----------
* Attitude error is the **full rotation angle** between the estimated and
  true attitude, in degrees, in ``[0, 180]``. It is computed as
  ``2 * acos(|q_est . q_true|)`` -- the factor of 2 and the absolute value
  are both required, because ``q`` and ``-q`` denote the same rotation and
  the half-angle would silently halve every reported error. (This exact
  mistake produced false passes in the HIL harness; see the
  ``esp32-hil-sensor-stream`` skill.)
* Quaternions are SciPy convention: ``[x, y, z, w]``.
* Dispersion uses the **sample** standard deviation (``ddof=1``) because
  these are samples of a run, not the full population.
* ``rms_deg`` is the quadratic mean including the bias term, so it is the
  right number for a total error budget; ``std_deg`` is scatter alone.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

FloatArray = npt.NDArray[np.float64]

# A quaternion must be unit length to be used in a dot-product angle.
_UNIT_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class AngleErrorStats:
    """Distribution of attitude error, in degrees."""

    count: int
    mean_deg: float
    rms_deg: float
    std_deg: float
    max_deg: float
    p50_deg: float
    p95_deg: float
    p99_deg: float
    sigma3_deg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Distribution of a per-sample duration, in milliseconds."""

    count: int
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nan_stats_float() -> float:
    return math.nan


def quaternions_from_mrp(mrp: npt.ArrayLike) -> FloatArray:
    """Convert modified Rodrigues parameters to unit quaternions.

    Args:
        mrp: shape (N, 3) or (3,) array of MRPs.

    Returns:
        (N, 4) array of unit quaternions in SciPy ``[x, y, z, w]`` order.
    """
    arr = np.asarray(mrp, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError("1-D MRP input must have 3 components")
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("mrp must have shape (N, 3) or (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("mrp entries must be finite")
    return Rotation.from_mrp(arr).as_quat()


def quaternion_rotation_error_deg(
    estimated: npt.ArrayLike, truth: npt.ArrayLike
) -> float:
    """Full rotation angle between two attitude quaternions, in degrees.

    The result is in ``[0, 180]``. Both inputs must be unit quaternions.

    Raises:
        ValueError: if either quaternion is not unit length within
            ``_UNIT_TOLERANCE``, or is non-finite. A non-unit quaternion
            (for example the all-zero quaternion an unhandled exact half
            turn can produce) is rejected rather than silently normalized.
    """
    q1 = np.asarray(estimated, dtype=float)
    q2 = np.asarray(truth, dtype=float)
    for name, q in (("estimated", q1), ("truth", q2)):
        if q.shape != (4,):
            raise ValueError(f"{name} quaternion must have shape (4,)")
        if not np.all(np.isfinite(q)):
            raise ValueError(f"{name} quaternion must be finite")
        if abs(float(np.linalg.norm(q)) - 1.0) > _UNIT_TOLERANCE:
            raise ValueError(
                f"{name} quaternion must be unit length (norm "
                f"{float(np.linalg.norm(q)):.6f})"
            )
    dot = float(np.dot(q1, q2))
    # |dot| makes the comparison sign-insensitive: q and -q are one rotation.
    return float(2.0 * math.degrees(math.acos(min(1.0, abs(dot)))))


def _validate_errors(errors: npt.ArrayLike) -> FloatArray:
    arr = np.asarray(errors, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("errors must be finite")
    if np.any(arr < 0.0):
        raise ValueError("errors must be non-negative")
    return arr


def compute_angle_error_stats(errors: npt.ArrayLike) -> AngleErrorStats:
    """Summarize a per-sample attitude error series.

    An empty input yields ``count=0`` with NaN statistics rather than
    raising, so a run with no valid estimates is representable in a
    results file.
    """
    arr = _validate_errors(errors)
    if arr.size == 0:
        nan = _nan_stats_float()
        return AngleErrorStats(
            count=0,
            mean_deg=nan,
            rms_deg=nan,
            std_deg=nan,
            max_deg=nan,
            p50_deg=nan,
            p95_deg=nan,
            p99_deg=nan,
            sigma3_deg=nan,
        )
    return AngleErrorStats(
        count=int(arr.size),
        mean_deg=float(np.mean(arr)),
        rms_deg=float(np.sqrt(np.mean(arr**2))),
        # Sample standard deviation: these are samples of a run.
        std_deg=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        max_deg=float(np.max(arr)),
        p50_deg=float(np.percentile(arr, 50.0)),
        p95_deg=float(np.percentile(arr, 95.0)),
        p99_deg=float(np.percentile(arr, 99.0)),
        sigma3_deg=float(3.0 * np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    )


def compute_latency_stats(durations_s: npt.ArrayLike) -> LatencyStats:
    """Summarize per-sample durations given in seconds, reported in ms."""
    arr = np.asarray(durations_s, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("durations must be finite")
    if np.any(arr < 0.0):
        raise ValueError("durations must be non-negative")
    if arr.size == 0:
        nan = _nan_stats_float()
        return LatencyStats(
            count=0,
            mean_ms=nan,
            std_ms=nan,
            min_ms=nan,
            p50_ms=nan,
            p95_ms=nan,
            max_ms=nan,
        )
    ms = arr * 1000.0
    return LatencyStats(
        count=int(ms.size),
        mean_ms=float(np.mean(ms)),
        std_ms=float(np.std(ms, ddof=1)) if ms.size > 1 else 0.0,
        min_ms=float(np.min(ms)),
        p50_ms=float(np.percentile(ms, 50.0)),
        p95_ms=float(np.percentile(ms, 95.0)),
        max_ms=float(np.max(ms)),
    )
