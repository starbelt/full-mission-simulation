"""Quaternion attitude controller (PD on the attitude error).

The controller is deliberately the simplest thing that can hold attitude, so
that a failure in the loop means a bug in the plant, the estimator or the board
rather than in the control law. It sees only the *estimated* attitude and rate —
never truth — which is what makes the closed-loop test an end-to-end check of
the truth-ownership rule.

Law
---
The error is a body-frame rotation vector from the current to the desired
attitude. A PD law on that error plus the estimated rate gives the body torque::

    tau = Kp * e - Kd * omega_est

Sign convention matters and is easy to get backwards: a *positive* ``e_z`` means
the body must rotate about ``+Z`` to reach the target, so the commanded torque
about ``+Z`` must be positive. A mirrored controller still settles, just the
wrong way, so ``test_attitude_error_drives_torque_in_the_reducing_direction``
names the direction explicitly rather than only asserting convergence.

The torque is clipped to the board's continuous limit. With three orthogonal
wheels the allocable body torque is bounded by ``sum |u_max| / sqrt(3)`` for a
balanced command, and a pure single-axis command by ``u_max``; the controller
uses the single-axis value, which is always feasible.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pocketadcs_sim.adcs import quaternions as quat
from pocketadcs_sim.adcs.rotors import TGR2912

__all__ = ["AttitudeController"]

FloatArray = npt.NDArray[np.float64]

DEFAULT_PROPORTIONAL_GAIN = 0.05  # rad -> N.m
DEFAULT_DERIVATIVE_GAIN = 0.02  # N.m.s -> N.m


class AttitudeController:
    """PD attitude controller producing a body-frame torque command."""

    def __init__(
        self,
        proportional_gain: float = DEFAULT_PROPORTIONAL_GAIN,
        derivative_gain: float = DEFAULT_DERIVATIVE_GAIN,
        max_torque_nm: float | None = None,
    ) -> None:
        if proportional_gain <= 0.0 or derivative_gain <= 0.0:
            raise ValueError("controller gains must be positive")
        limit = TGR2912.continuous_torque_vac_nm if max_torque_nm is None else float(max_torque_nm)
        if limit <= 0.0:
            raise ValueError("max_torque_nm must be positive")
        self.proportional_gain = float(proportional_gain)
        self.derivative_gain = float(derivative_gain)
        self.max_torque_nm = limit
        #: The attitude argument this controller was actually handed on its most
        #: recent :meth:`compute` call, copied on the way in.
        #:
        #: The closed loop's truth cross-check reads this back so that it scores
        #: the value a torque was really computed from. Reading the *caller's*
        #: snapshot instead would score whatever the caller believed it sent --
        #: which is the same failure as the hardcoded ``truth_leakage`` flag this
        #: replaced: a number that cannot detect the thing it claims to measure.
        #: It is copied rather than aliased so a caller reusing one buffer across
        #: steps cannot rewrite the record after the fact.
        self.last_estimated_attitude: FloatArray | None = None

    def compute(
        self,
        estimated_attitude: npt.ArrayLike,
        estimated_rate_rad_s: npt.ArrayLike,
        desired_attitude: npt.ArrayLike,
        wheel_axes: npt.ArrayLike | None = None,
    ) -> FloatArray:
        """Return the body-frame torque command for the estimated state.

        ``wheel_axes`` is accepted for interface completeness and is checked for
        feasibility; the three-wheel board applies whatever it is given along
        its own axes, so allocation happens on the board side.
        """
        estimate = quat.as_scalar_first(estimated_attitude)
        self.last_estimated_attitude = np.array(estimate, dtype=float, copy=True)
        desired = quat.as_scalar_first(desired_attitude)
        rate = np.asarray(estimated_rate_rad_s, dtype=float).reshape(3)

        if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(desired)):
            raise ValueError("attitudes must be finite")
        if not np.all(np.isfinite(rate)):
            raise ValueError("estimated_rate_rad_s must be finite")
        if wheel_axes is not None:
            axes = np.asarray(wheel_axes, dtype=float).reshape(3, 3)
            if not np.allclose(np.linalg.det(axes), 1.0, atol=1e-6) and not np.allclose(
                np.abs(np.linalg.det(axes)), 1.0, atol=1e-6
            ):
                raise ValueError("wheel_axes must be a right-handed triad")

        # The MRP is this module's rotation-vector representation: bounded, and
        # nonsingular where the exponential map of the error quaternion is not.
        error = quat.to_mrp(quat.error_quaternion(desired, estimate))
        torque = self.proportional_gain * error - self.derivative_gain * rate
        return np.clip(torque, -self.max_torque_nm, self.max_torque_nm)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable record of the control law."""
        return {
            "law": "tau = Kp * error_vector - Kd * estimated_body_rate",
            "proportional_gain_rad_per_nm": self.proportional_gain,
            "derivative_gain_nm_s_per_rad": self.derivative_gain,
            "max_torque_nm": self.max_torque_nm,
            "feedback_sign_convention": "positive error axis -> positive torque axis",
        }
