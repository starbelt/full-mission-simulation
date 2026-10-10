"""Reconstruct a 3-axis sun vector from the VEMD8080 six-channel array.

``VEMD8080ArrayModel`` reports six raw photocurrents, one per body face, each
proportional to ``max(0, n_i . sun)``. The attitude determination firmware
needs a single 3-axis unit sun vector, so this module provides the
reconstruction step that sits between the simulated hardware and the
estimator.

Method
------
The forward model is, per face, ``d_i = scale_i * f(max(0, n_i . u)) +
dark_i`` where ``f`` is the array's angular response -- a power law plus a
reflection tail, ``c ** q + t * (1 - c)`` (see
``vemd8080_array._angular_response``). Because ``f`` is strictly
increasing, its inverse turns each channel back into a linear quantity:
``c_i = f ** -1 (d_i - dark_i) / scale_i`` is the cosine ``max(0, n_i .
u)``, so the six channels form an overdetermined linear system in the
three unknowns of ``u``. Solving it in the least-squares sense and
renormalizing recovers the direction exactly when no channel is clipped
and the scales are known.

That plain solution is biased whenever the sun is off the visible side of
a face, because a clipped (dark) face contributes a spurious "sunward"
pull. The estimate is therefore refined by iteratively reweighted least
squares over only the faces predicted lit, which removes the clipping
bias. Two passes drive the residual well below a tenth of a degree for the
geometries tested.

Because the tail breaks the scale-homogeneity a per-row max-normalization
relies on, the cosines are recovered with a fixed point on each row's peak
cosine (see ``_solve_direction``), not a plain elementwise inverse: the
latter left the estimate ~0.1 deg off, biased toward grazing incidence.
The fixed point is exact (to machine precision) for uniform channel scales
and robust to the modest per-channel spread and dark floor a real build
carries.

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
    CHANNEL_NORMALS,
    _angular_response,
    invert_angular_response,
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

# Passes of the peak-cosine fixed point in the response inversion. The
# mapping contracts for the cube geometries that occur, so a dozen passes
# are ample and cheap (each is one vectorized bisection and one 3x3 solve).
_FIXED_POINT_PASSES = 12

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


def _solve_direction(weights: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Least-squares unit direction from per-row weighted face responses.

    Inverts the array's power-law-plus-tail response with a fixed point on
    the row's peak cosine. Each row satisfies
    ``w_i = s_i * f(max(0, n_i . u))`` for a strictly-increasing response
    ``f`` (``c ** q + t * (1 - c)``). Normalizing by the row maximum gives
    ``r_i = w_i / max(w) = f(c_i) / f(m)`` where ``m`` is the largest face
    cosine, so for a trial ``m`` the per-face cosines are
    ``c_i = f ** -1 (r_i * f(m))`` and the direction is the least-squares
    solve ``u ~ pinv(N^T N) ** -1 ... c`` -- the same homogeneous solve the
    array is built for. The next trial ``m`` is the peak cosine the new
    direction predicts; the fixed point is the true peak cosine, and it
    converges in a handful of passes for any convex-cube geometry.

    A plain elementwise ``f ** -1`` (the inverse of the old tail-less pure
    power law) is *not* valid here: the tail breaks the scale-homogeneity
    the row normalization relies on, and left un-inverted it biases the
    estimate ~0.1 deg toward grazing incidence. The fixed point is exact
    (to machine precision) for uniform channel scales, and
    robust to the modest per-channel scale spread and dark floor a real
    build carries.

    Returns:
        (directions, cosines): unit directions and the corresponding
        per-face linearized cosines for the clipping-aware refinement.
    """
    row_max = np.max(weights, axis=1, keepdims=True)
    safe = np.where(row_max > 0.0, row_max, 1.0)
    r = weights / safe
    alive = row_max[:, 0] > 0.0
    m = np.ones(r.shape[0])
    directions = np.zeros((r.shape[0], 3), dtype=float)
    for _ in range(_FIXED_POINT_PASSES):
        cosines = invert_angular_response(r * _angular_response(m)[:, None])
        solution = cosines @ _NORMAL_PINV
        norms = np.linalg.norm(solution, axis=1, keepdims=True)
        directions = np.divide(
            solution, norms, out=np.zeros_like(solution), where=norms > 0.0
        )
        m = np.where(
            alive,
            np.clip(directions @ _NORMALS.T, 0.0, None).max(axis=1),
            1.0,
        )
    cosines = invert_angular_response(r * _angular_response(m)[:, None])
    return directions, cosines


def _lit_face_count(row: np.ndarray) -> int:
    """How many faces respond above the dark floor for one sample."""
    peak = float(np.max(row))
    if peak <= 0.0:
        return 0
    return int(np.count_nonzero(row >= _FLOOR_FRACTION * peak))


def _refine(weights: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Iteratively reweighted refinement that respects one-sided clipping.

    Only faces predicted lit contribute, so a dark backlit face does not drag
    the estimate toward the sensor's unlit side. Operates on the
    linearized (response-inverted) cosines from the fixed-point solve.

    Returns:
        (directions, cosines): the refined unit directions and the full
        (unmasked) linearized cosines, which the uncertainty propagation
        uses as the least-squares residual target.
    """
    directions, cosines = _solve_direction(weights)
    for _ in range(_REFINEMENT_PASSES):
        predicted = np.clip(directions @ _NORMALS.T, 0.0, None)
        peaks = np.max(predicted, axis=1, keepdims=True)
        lit = predicted >= _LIT_FRACTION * peaks
        masked = np.where(lit, cosines, 0.0)
        alive = np.max(masked, axis=1) > 0.0
        if not np.any(alive):
            break
        solution = masked @ _NORMAL_PINV
        norms = np.linalg.norm(solution, axis=1, keepdims=True)
        refined = np.divide(
            solution, norms, out=np.zeros_like(solution), where=norms > 0.0
        )
        directions = np.where(alive[:, None], refined, directions)
    return directions, cosines


def _uncertainty_deg(cosines: FloatArray, directions: FloatArray) -> FloatArray:
    """Propagate the least-squares residual into a 1-sigma angular error.

    The residual of the linear fit is the best available in-situ noise
    estimate, so the covariance of the direction is
    ``sigma^2 * pinv(N^T N)`` and the angular error follows from its trace.
    A clean signal leaves a near-zero residual and therefore a near-zero
    uncertainty; a noisy array reports a larger one.

    The residual is taken on the *linearized* cosines (the response
    inverted), not the raw channel weights: the array's reflection tail
    leaves edge-on faces reading a finite value on the raw weights, which
    would inflate the residual of a perfectly clean signal. The linearized
    cosines map that tail back to zero, so a clean signal is residual-free.
    """
    predicted = np.clip(directions @ _NORMALS.T, 0.0, None)
    residual = cosines - predicted
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
    directions, cosines = _refine(arr)
    uncertainties = _uncertainty_deg(cosines, directions)

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
