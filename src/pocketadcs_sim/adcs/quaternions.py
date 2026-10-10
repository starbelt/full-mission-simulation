"""Quaternion kinematics for the closed-loop ADCS prototype.

Conventions, stated once so every other module can rely on them:

* Quaternions are **scalar-first** Hamilton quaternions, numpy shape ``(4,)``:
  ``[w, x, y, z]``.
* A quaternion ``q`` denotes the DCM ``C(q)`` mapping inertial-frame components
  to **body-frame** components, i.e. ``C_BN``. This matches
  :mod:`pocketadcs_sim.attitude.triad`, whose DCM is
  ``Rotation.from_mrp(mrp).inv()``.
* ``omega`` is the body rate ``omega_BN`` in rad/s, in body axes.
* Kinematics ``qdot = 0.5 * q (x) [0, omega]`` for ``C_BN`` quaternions
  (Markley, "Attitude Error Representations for Kalman Filtering", 2003).
* ``q`` and ``-q`` denote the same rotation, so every function here is
  sign-invariant in its quaternion inputs.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "as_scalar_first",
    "conjugate",
    "error_quaternion",
    "from_axis_angle",
    "from_dcm",
    "from_mrp",
    "multiply",
    "propagate",
    "rotation_angle_rad",
    "to_dcm",
    "to_mrp",
]


def _raw_multiply(left: FloatArray, right: FloatArray) -> FloatArray:
    """Unnormalized Hamilton product of two scalar-first 4-vectors."""
    aw, av = left[0], left[1:]
    bw, bv = right[0], right[1:]
    return np.concatenate(([aw * bw - av @ bv], aw * bv + bw * av + np.cross(av, bv)))


def as_scalar_first(value: npt.ArrayLike) -> FloatArray:
    """Return a normalized copy of a scalar-first quaternion, preserving its sign."""
    q = np.asarray(value, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("quaternion must be finite with a non-zero norm")
    return q / norm


def conjugate(q: npt.ArrayLike) -> FloatArray:
    """Return the inverse rotation of a unit quaternion."""
    product = as_scalar_first(q) * np.array([1.0, -1.0, -1.0, -1.0])
    return np.asarray(product, dtype=float)


def multiply(left: npt.ArrayLike, right: npt.ArrayLike) -> FloatArray:
    """Return the normalized Hamilton product ``left (x) right``."""
    return as_scalar_first(_raw_multiply(as_scalar_first(left), as_scalar_first(right)))


def from_axis_angle(axis: npt.ArrayLike, angle_rad: float) -> FloatArray:
    """Return the scalar-first quaternion rotating ``angle_rad`` about ``axis``."""
    unit = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(unit))
    if norm <= 0.0:
        raise ValueError("rotation axis must be non-zero")
    half = 0.5 * float(angle_rad)
    return np.concatenate(([np.cos(half)], np.sin(half) * (unit / norm)))


def from_mrp(sigma: npt.ArrayLike) -> FloatArray:
    """Return the ``C_BN`` quaternion for a modified Rodrigues parameter.

    Matches ``scipy.spatial.transform.Rotation.from_mrp``.
    """
    s = np.asarray(sigma, dtype=float).reshape(3)
    magnitude_sq = float(s @ s)
    return as_scalar_first(np.concatenate(([1.0 - magnitude_sq], 2.0 * s)) / (1.0 + magnitude_sq))


def to_mrp(q: npt.ArrayLike) -> FloatArray:
    """Return the MRP for a ``C_BN`` quaternion, matching scipy's convention.

    The scalar part is forced non-negative so the result is a function of the
    rotation alone. A 180 deg rotation lands on ``|sigma| == 1`` (the map is
    singular there); that value round-trips correctly through :func:`from_mrp`
    and is what ``scipy.spatial.transform.Rotation.as_mrp`` returns, so it is
    left alone rather than flipped into the shadow set.
    """
    unit = as_scalar_first(q)
    if unit[0] < 0.0:
        unit = -unit
    return np.asarray(unit[1:] / (1.0 + unit[0]), dtype=float)


def to_dcm(q: npt.ArrayLike) -> FloatArray:
    """Return the ``(3, 3)`` DCM ``C_BN`` mapping inertial to body components.

    ``C_BN`` maps ``v_N`` to ``v_B``, so it is the **transpose** of the active
    rotation matrix for the same quaternion: for ``R`` the active rotation,
    ``C_BN == R.T``. This is the repository convention -- ``triad.py`` builds
    ``truth_dcm`` as ``Rotation.from_mrp(mrp).inv()`` -- and getting the
    direction backwards silently mirrors every estimate.

    A consequence worth stating: quaternion products compose in the opposite
    order to matrix products in this convention, because both are transposed:

        ``to_dcm(a (x) b) == to_dcm(b) @ to_dcm(a)``
    """
    w, x, y, z = as_scalar_first(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
            [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)],
            [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def from_dcm(value: npt.ArrayLike) -> FloatArray:
    """Return the ``C_BN`` quaternion for a proper rotation matrix.

    ``to_dcm`` returns the transpose of the active rotation matrix, so this
    recovers the quaternion of ``c.T`` -- feeding the inverse rotation in by
    mistake would negate the vector part and silently mirror every estimate.
    """
    c = np.asarray(value, dtype=float).reshape(3, 3)
    if not np.allclose(c.T @ c, np.eye(3), atol=1e-8) or not np.isclose(
        float(np.linalg.det(c)), 1.0, atol=1e-8
    ):
        raise ValueError("dcm must be a proper rotation matrix")
    c = c.T
    trace = float(np.trace(c))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(1.0 + trace)
        return as_scalar_first(
            np.array(
                [
                    0.25 * scale,
                    (c[2, 1] - c[1, 2]) / scale,
                    (c[0, 2] - c[2, 0]) / scale,
                    (c[1, 0] - c[0, 1]) / scale,
                ]
            )
        )
    # trace <= 0. The trace-based formulas lose all precision here, so drive the
    # largest diagonal term: it is the one component recoverable from a single
    # square root. Note that the scalar part uses a *difference* of the two
    # off-diagonal terms while the other two use *sums*; using differences
    # throughout silently returns the axis unit vector instead of the rotation.
    index = int(np.argmax(np.diag(c)))
    others = [i for i in range(3) if i != index]
    j, k = others
    if index == 0:
        scale = 2.0 * np.sqrt(1.0 + c[0, 0] - c[1, 1] - c[2, 2])
    elif index == 1:
        scale = 2.0 * np.sqrt(1.0 - c[0, 0] + c[1, 1] - c[2, 2])
    else:
        scale = 2.0 * np.sqrt(1.0 - c[0, 0] - c[1, 1] + c[2, 2])
    result = np.zeros(4)
    result[0] = (c[j, k] - c[k, j]) / scale
    result[1 + index] = 0.25 * scale
    result[1 + j] = (c[index, j] + c[j, index]) / scale
    result[1 + k] = (c[index, k] + c[k, index]) / scale
    return as_scalar_first(result)


def propagate(q: npt.ArrayLike, omega_body_rad_s: npt.ArrayLike, dt_s: float) -> FloatArray:
    """Propagate ``C_BN`` from ``q`` under a constant body rate for ``dt_s``.

    Uses the exact exponential of the body-rate generator, so a constant rate
    reproduces a known-axis rotation to machine precision and splitting an
    interval into pieces changes nothing. That matters for the ADS: the
    gyro-only path in eclipse accumulates this propagation every step, and a
    first-order truncation would leak a bias proportional to ``dt`` that no
    absolute correction can remove.

    **Sign/side note.** Because ``C_BN`` maps inertial components to body
    components, a body rotation by ``+theta`` about ``+Z`` is ``C = Rz(-theta)``,
    so ``C_dot = -[omega]x C`` and the increment multiplies on the **right** of
    the quaternion: ``q_new = q (x) exp(-omega dt / 2)``. The opposite ordering
    gives ``C_dot = C [omega]x``, which coincides at the identity attitude only
    -- so that mistake is invisible from a zero attitude and then silently
    mirrors every body-rate update once the vehicle is slewing.
    """
    unit = as_scalar_first(q)
    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be a positive finite number")
    omega = np.asarray(omega_body_rad_s, dtype=float).reshape(3)
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega_body_rad_s must be finite")
    angle = float(np.linalg.norm(omega) * dt)
    if angle == 0.0:
        return unit
    delta = from_axis_angle(omega, angle)
    return as_scalar_first(_raw_multiply(unit, delta))


def error_quaternion(q_desired: npt.ArrayLike, q_actual: npt.ArrayLike) -> FloatArray:
    """Return the shortest rotation from the actual body frame to the desired one.

    The result is ``q_desired (x) conjugate(q_actual)`` with a non-negative
    scalar part, so its vector part is the rotation that must still be applied
    to reach the desired attitude, expressed in desired body axes.
    """
    error = _raw_multiply(as_scalar_first(q_desired), conjugate(q_actual))
    if error[0] < 0.0:
        error = -error
    return as_scalar_first(error)


def rotation_angle_rad(q: npt.ArrayLike) -> float:
    """Return the rotation angle of ``q`` in rad, in ``[0, pi]``."""
    unit = as_scalar_first(q)
    if unit[0] < 0.0:
        unit = -unit
    return float(2.0 * np.arccos(np.clip(unit[0], -1.0, 1.0)))
