"""Reconstruct a 3-axis sun vector from the VEMD8080 six-channel array.

``VEMD8080ArrayModel`` reports six raw photocurrents, one per body face, each
proportional to ``max(0, n_i . sun)``. The attitude determination firmware
needs a single 3-axis unit sun vector, so this module provides the
reconstruction step that sits between the simulated hardware and the
estimator.

Method
------
The forward model is linear in the (unnormalized) sun direction ``u``::

    d_i = scale_i * max(0, n_i . u) + dark_i

Each channel is an inner product with a *known* face normal, so all six
channels form an overdetermined linear system in three unknowns. Solving it
in the least-squares sense and renormalizing recovers the direction exactly
when no channel is clipped and the scales are known.

That plain solution is biased whenever the sun is off the visible side of a
face, because a clipped (dark) face contributes a spurious "sunward" pull.
The estimate is therefore refined by iteratively reweighted least squares
over only the faces predicted lit, which removes the clipping bias. Two
passes drive the residual well below a tenth of a degree for the geometries
tested.

Validity
--------
The array is a convex cube, so a real sun lights at most three of the six
faces. A sample is therefore invalid when

* the whole response sits at the dark floor (eclipse), or
* more than three faces respond, which is not physically realizable and
  indicates an even/uncalibrated array carrying no reliable direction
  (for example a perfectly flat six-channel response).

Note the diagonal geometry (sun along (1,1,1)) lights three faces equally
and is *valid* -- the least-squares solve recovers it exactly.

Invalid samples must not be fed to TRIAD as if they were real vectors; this
mirrors the firmware's ``FLAG_SUN_ZERO`` and degeneracy handling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pocketadcs_sim.sensors.vemd8080_array import (
    _RESPONSE_EXPONENT,
    CHANNEL_NORMALS,
)

FloatArray = npt.NDArray[np.float64]

# Normalized least-squares design matrix (6 x 3): the six known face normals.
_NORMALS = np.asarray(CHANNEL_NORMALS, dtype=float)

# Pseudoinverse of the normal matrix, used for the initial projection and
# for propagating measurement noise into an angular uncertainty.
_NORMAL_PINV = np.linalg.pinv(_NORMALS.T)  # (3, 6)
_NORMAL_COV = np.linalg.pinv(_NORMALS.T @ _NORMALS)  # (3, 3)

# A response below this fraction of the array's own maximum is treated as
# dark current rather than as a lit face. Dimensionless.
#: A face counts as illuminated when it reads at least this fraction of the
#: brightest channel in the same sample. Relative rather than absolute, so
#: the dark-current floor and analog noise on an unlit face cannot inflate
#: the count: at 0.1% even 0.05 uA of noise (250x the 0.2 nA dark floor) was
#: enough to push 2 genuinely-lit faces to 5 and get 58% of sunlit samples
#: discarded as physically implausible. At 20% the count is stable to >1 uA
#: of noise while still catching the degenerate patterns the test exists for
#: (six equally-bright faces, and an all-zero sample).
_FLOOR_FRACTION = 0.2

# A face predicted to respond at or above this fraction of the array maximum
# counts as lit during the clipped refinement. Dimensionless.
_LIT_FRACTION = 1.0e-2

#: Reciprocal of the array's ``cos ** q`` angular-response exponent. The
#: forward model and the reconstruction must use matching exponents, so this
#: is derived from the model's own constant rather than hard-coded.
_INVERSE_RESPONSE_EXPONENT = 1.0 / _RESPONSE_EXPONENT

# A convex cube cannot light more than three faces; more means the array is
# flat/uncalibrated and the direction is not recoverable.
_MAX_LIT_FACES = 3

# Iteratively-reweighted refinement passes. The plain least-squares solution
# is already close; two passes remove the clipping bias.
_REFINEMENT_PASSES = 2


@dataclass(frozen=True, slots=True)
class SunVectorEstimate:
    """One reconstructed sun direction, its validity, and its uncertainty.

    Attributes:
        vector: unit sun direction in the body frame, or the zero vector
            when ``valid`` is False.
        valid: False during eclipse, or for a physically impossible
            (non-dominant) illumination pattern.
        total_current: sum of the six raw channel readings, in the input
            units (diagnostic for exposure and clipping studies).
        dominant_channel: index of the brightest channel, or None when the
            sample is invalid.
        uncertainty_deg: propagated 1-sigma angular uncertainty, or
            ``inf`` when the sample is invalid.
    """

    vector: FloatArray
    valid: bool
    total_current: float
    dominant_channel: int | None
    uncertainty_deg: float


def _as_row_batch(channels: npt.ArrayLike, name: str) -> np.ndarray:
    """Coerce input to a validated (N, 6) float array."""
    arr = np.asarray(channels, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 6:
            raise ValueError(f"{name} must have 6 entries")
        arr = arr.reshape(1, 6)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"{name} must have shape (N, 6) or (6,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Normalize each row by its own max so channel scales compare uniformly."""
    row_max = np.max(arr, axis=1, keepdims=True)
    safe = np.where(row_max > 0.0, row_max, 1.0)
    return arr / safe


def _linearize(weights: FloatArray) -> FloatArray:
    """Undo the array's angular response so channels are proportional to cos.

    ``VEMD8080ArrayModel`` reports ``cos(theta) ** q`` with ``q`` chosen so
    the response is 0.5 at the datasheet's 65 deg half-sensitivity angle.
    The least-squares solve below assumes each channel is proportional to
    ``max(0, n . sun)``, so the power must be undone first:
    ``cos = response ** (1/q)``.

    Without this the estimate is biased toward the array's brightest face
    (roughly 15% at the half-sensitivity angle) and the bias varies with
    incidence, so it does not cancel in the normalization.
    """
    return weights ** _INVERSE_RESPONSE_EXPONENT


def _solve_direction(weights: FloatArray) -> FloatArray:
    """Least-squares unit direction from per-row weighted face responses."""
    normalized = _normalize_rows(_linearize(weights))
    # _NORMAL_PINV is (6, 3): (N, 6) @ (6, 3) -> (N, 3) unit directions.
    solution = normalized @ _NORMAL_PINV
    norms = np.linalg.norm(solution, axis=1, keepdims=True)
    return np.divide(
        solution, norms, out=np.zeros_like(solution), where=norms > 0.0
    )


def _lit_face_count(row: np.ndarray) -> int:
    """How many faces respond above the dark floor for one sample."""
    peak = float(np.max(row))
    if peak <= 0.0:
        return 0
    return int(np.count_nonzero(row >= _FLOOR_FRACTION * peak))


def _refine(weights: FloatArray) -> FloatArray:
    """Iteratively reweighted refinement that respects one-sided clipping.

    Only faces predicted lit contribute, so a dark backlit face does not drag
    the estimate toward the sensor's unlit side.
    """
    directions = _solve_direction(weights)
    for _ in range(_REFINEMENT_PASSES):
        predicted = np.clip(directions @ _NORMALS.T, 0.0, None)
        peaks = np.max(predicted, axis=1, keepdims=True)
        lit = predicted >= _LIT_FRACTION * peaks
        masked = np.where(lit, weights, 0.0)
        alive = np.max(masked, axis=1) > 0.0
        if not np.any(alive):
            break
        refined = _solve_direction(masked)
        directions = np.where(alive[:, None], refined, directions)
    return directions


def _uncertainty_deg(weights: FloatArray, directions: FloatArray) -> FloatArray:
    """Propagate the least-squares residual into a 1-sigma angular error.

    The residual of the linear fit is the best available in-situ noise
    estimate, so the covariance of the direction is
    ``sigma^2 * pinv(N^T N)`` and the angular error follows from its trace.
    A clean signal leaves a near-zero residual and therefore a near-zero
    uncertainty; a noisy array reports a larger one.
    """
    normalized = _normalize_rows(weights)
    predicted = np.clip(directions @ _NORMALS.T, 0.0, None)
    residual = normalized - predicted
    dof = max(6 - 3, 1)
    sigma_sq = np.sum(residual * residual, axis=1) / dof
    trace = float(np.trace(_NORMAL_COV))
    angular_rad = np.sqrt(np.maximum(sigma_sq * trace, 0.0))
    return np.degrees(angular_rad)


def reconstruct_sun_vector_batch(
    channels: npt.ArrayLike, lit_threshold: float | None = None
) -> list[SunVectorEstimate]:
    """Reconstruct sun vectors for a batch of six-channel samples.

    Args:
        channels: (N, 6) or (6,) array of raw channel readings.
        lit_threshold: absolute illumination floor, in the same units as
            ``channels``. A sample whose brightest channel is below this is
            treated as unlit and marked invalid.

            This is required for a correct eclipse test. The array model
            clips readings at zero, which rectifies negative noise; in deep
            eclipse that leaves some channels slightly above zero and the
            rest at exactly zero -- a *shape* that a purely relative test
            reads as a lit face, producing a garbage direction. Supplying
            the expected lit level makes validity an absolute statement
            ("the array is receiving light") rather than a relative one
            ("the channels are unequal"). Pass a small fraction of the
            model's configured ``scale``.

    Returns:
        One ``SunVectorEstimate`` per input sample. Samples carrying no
        directional information (eclipse, or more than three lit faces)
        come back with ``valid=False``, a zero vector, and infinite
        uncertainty.

    Raises:
        ValueError: if the input is not (N, 6)/(6,), holds non-finite
            values, or ``lit_threshold`` is negative or non-finite.
    """
    if lit_threshold is not None and (
        not np.isfinite(lit_threshold) or lit_threshold < 0.0
    ):
        raise ValueError("lit_threshold must be a finite non-negative value")

    arr = _as_row_batch(channels, "channels")
    directions = _refine(arr)
    uncertainties = _uncertainty_deg(arr, directions)

    estimates: list[SunVectorEstimate] = []
    for row, direction, sigma_deg in zip(arr, directions, uncertainties, strict=True):
        total = float(np.sum(row))
        peak = float(np.max(row))
        lit = _lit_face_count(row)
        below_floor = lit_threshold is not None and peak < lit_threshold
        # Eclipse (everything at the dark floor, or explicitly below the
        # supplied illumination threshold) or a physically impossible number
        # of lit faces.
        if peak <= 0.0 or lit == 0 or lit > _MAX_LIT_FACES or below_floor:
            estimates.append(
                SunVectorEstimate(
                    vector=np.zeros(3, dtype=float),
                    valid=False,
                    total_current=total,
                    dominant_channel=None,
                    uncertainty_deg=math.inf,
                )
            )
            continue
        estimates.append(
            SunVectorEstimate(
                vector=np.asarray(direction, dtype=float),
                valid=True,
                total_current=total,
                dominant_channel=int(np.argmax(row)),
                uncertainty_deg=float(sigma_deg),
            )
        )
    return estimates


def reconstruct_sun_vector(
    channels: npt.ArrayLike, lit_threshold: float | None = None
) -> SunVectorEstimate:
    """Reconstruct a single six-channel sample into a sun vector estimate."""
    return reconstruct_sun_vector_batch(channels, lit_threshold=lit_threshold)[0]


def estimate_sun_vector_uncertainty_deg(estimate: SunVectorEstimate) -> float:
    """Return the propagated 1-sigma angular uncertainty for one estimate.

    An invalid estimate reports ``inf``.
    """
    if not estimate.valid:
        return math.inf
    return float(estimate.uncertainty_deg)
