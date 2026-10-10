"""Truth rigid body plus three momentum wheels.

The plant is the *physical world* of the prototype: it integrates the hub's
rigid-body dynamics with the reaction torques of three TGR 29-12 rotors. It
knows nothing about the estimator or controller, and its attitude/rate are
strictly truth -- scoring reads them, the ADS does not.

Physics
-------
Hub inertia ``I`` is diagonal in body axes. Wheel ``i`` spins about body axis
``g_i`` with rotor inertia ``J``. Rotor speed is stored **relative to the body**
(``Omega_i``), which is what a wheel actually reports; the inertial rotor rate
is ``omega . g_i + Omega_i``. Summing momenta in the body frame gives

    I_eff = I + sum_i J g_i g_i^T
    I_eff omega_dot + omega x (I_eff omega) = tau_external - sum_i J Omega_dot_i g_i
                                             + omega x (sum_i J Omega_i g_i)

The last cross-product term is the gyroscopic term and is what makes a 3D
momentum wheel different from a 1D one. Both it and the rotor inertia in
``I_eff`` are included; omitting them is the standard way a hand-rolled wheel
model quietly gets 3D behavior wrong.

Sign convention: a positive wheel torque ``u_i`` **accelerates** rotor ``i``
(``Omega_dot_i = u_i / J``) and therefore applies ``-u_i`` about ``g_i`` to the
hub. Momentum is conserved: rotor and hub torques are equal and opposite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from pocketadcs_sim.adcs import quaternions as quat
from pocketadcs_sim.adcs.rotors import WheelSet, default_wheel_set

__all__ = ["PlantConfig", "PlantState", "TruthPlant"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PlantConfig:
    """Physical parameters of the truth plant.

    ``hub_inertia_kg_m2`` is the bus plus structure inertia, excluding the
    rotors; rotor inertia enters through the wheel set, not here.
    """

    hub_inertia_kg_m2: tuple[float, float, float]
    wheel_set: WheelSet = field(default_factory=default_wheel_set)

    @property
    def hub_inertia(self) -> FloatArray:
        """Diagonal hub inertia as a matrix."""
        return np.diag(np.asarray(self.hub_inertia_kg_m2, dtype=float))

    @property
    def effective_inertia(self) -> FloatArray:
        """``I + sum_i J g_i g_i^T``: the inertia seen by the free hub."""
        axes = np.asarray(self.wheel_set.axes, dtype=float)
        return np.asarray(
            self.hub_inertia + self.wheel_set.inertia * (axes.T @ axes), dtype=float
        )


@dataclass
class PlantState:
    """A snapshot of truth state at one instant."""

    time_s: float
    attitude: FloatArray  # scalar-first quaternion, C_BN
    body_rate_rad_s: FloatArray  # omega_BN, rad/s
    wheel_speed_rad_s: FloatArray  # rotor speed relative to the body, rad/s


class TruthPlant:
    """Rigid-body hub with three momentum wheels.

    Two integrators share one ``_derivative``. :meth:`step` uses the adaptive
    ``solve_ivp(method="RK45")`` and is the reference. :meth:`step_rk4` is a
    fixed-step classical RK4 of the same derivative, used by the closed loop
    because a 45-minute planner schedule at a 50 ms control period is ~54k
    steps and the adaptive path costs ~3 ms each; the step is exposed because
    the wheel dynamics have a time constant set by ``J / u_max``, and too
    coarse a step integrates the rotor speed noticeably wrong. That error is
    the integrator's, not the model's, so the closed-loop tolerances are
    stated with it in mind. ``tests/test_adcs_plant_fast.py`` pins the two to
    1e-12 and checks fourth-order convergence.
    """

    def __init__(
        self,
        config: PlantConfig,
        initial_attitude: npt.ArrayLike,
        initial_body_rate_rad_s: npt.ArrayLike = (0.0, 0.0, 0.0),
        initial_wheel_speed_rad_s: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> None:
        self.config = config
        self._axes = np.asarray(config.wheel_set.axes, dtype=float)
        self.attitude = quat.as_scalar_first(initial_attitude)
        self.body_rate = np.asarray(initial_body_rate_rad_s, dtype=float).reshape(3).copy()
        self.wheel_speed = np.asarray(initial_wheel_speed_rad_s, dtype=float).reshape(3).copy()
        self.time_s = 0.0

    @property
    def effective_inertia(self) -> FloatArray:
        """Inertia including rotor contributions."""
        return self.config.effective_inertia

    @property
    def total_angular_momentum_body(self) -> FloatArray:
        """Angular momentum of hub plus rotors, expressed in body axes.

        ``I_eff omega + sum_i J Omega_i g_i`` with ``Omega_i`` relative to the
        body. Folding the rotor inertia into ``I_eff`` and using the *relative*
        rotor speed are two halves of one identity -- mixing them double-counts.
        """
        axes = self._axes
        return np.asarray(
            self.effective_inertia @ self.body_rate
            + (self.config.wheel_set.inertia * (axes.T @ self.wheel_speed)),
            dtype=float,
        )

    def rotor_torque_on_hub(self, wheel_torque_nm: FloatArray) -> FloatArray:
        """Return the hub reaction torque from wheel torques (equal and opposite).

        ``u_i`` spins rotor ``i`` up, so the hub feels ``-u_i`` about ``g_i``.
        """
        return np.asarray(
            -(self._axes.T @ np.asarray(wheel_torque_nm, dtype=float).reshape(3)),
            dtype=float,
        )

    def _derivative(self, state: FloatArray, wheel_torque_nm: FloatArray) -> FloatArray:
        """Right-hand side of the coupled hub/rotor equations.

        With rotor speed ``Omega_i`` measured relative to the body, the total
        angular momentum in body axes is

            H = I_eff omega + sum_i J Omega_i g_i,      I_eff = I + sum_i J g_i g_i^T

        and ``dH/dt|_inertial = tau`` becomes ``dH/dt|_body + omega x H = tau``.
        Differentiating (the axes are body-fixed, so ``dI_eff/dt`` vanishes and
        ``Omega_dot_i = u_i / J``) gives

            I_eff omega_dot = tau - omega x I_eff omega
                                 - sum_i J Omega_dot_i g_i
                                 - omega x (sum_i J Omega_i g_i)

        The last two terms are the wheel reaction torque and the gyroscopic term.
        Note the gyroscopic momentum uses ``Omega_i`` **relative to the body**;
        adding the hub rate ``omega`` there instead double-counts the rotor
        inertia that is already inside ``I_eff``, and flips the precession sense
        for a spinning body.
        """
        attitude = quat.as_scalar_first(state[:4])
        rate = state[4:7]
        wheel_speed = state[7:10]
        rotor_inertia = self.config.wheel_set.inertia

        wheel_acceleration = wheel_torque_nm / rotor_inertia
        reaction = -(self._axes.T @ wheel_torque_nm)
        rotor_momentum = rotor_inertia * (self._axes.T @ wheel_speed)

        rate_dot = np.linalg.solve(
            self.effective_inertia,
            reaction
            - np.cross(rate, self.effective_inertia @ rate)
            - np.cross(rate, rotor_momentum),
        )
        # Body-rate kinematics: C_dot = C [omega]x, i.e. the increment multiplies
        # on the right of the C_BN quaternion (see quaternions.propagate).
        attitude_dot = 0.5 * quat._raw_multiply(
            quat.as_scalar_first(attitude), np.concatenate(([0.0], rate))
        )
        return np.concatenate((attitude_dot, rate_dot, wheel_acceleration))

    def step(self, dt_s: float, wheel_torque_nm: npt.ArrayLike) -> None:
        """Advance the plant by ``dt_s`` under a constant wheel torque command."""
        dt = float(dt_s)
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")
        torque = np.asarray(wheel_torque_nm, dtype=float).reshape(3)
        if not np.all(np.isfinite(torque)):
            raise ValueError("wheel torque must be finite")

        state = np.concatenate((self.attitude, self.body_rate, self.wheel_speed))

        def rhs(_t: float, y: FloatArray) -> FloatArray:
            return self._derivative(y, torque)

        solution = solve_ivp(
            rhs,
            (0.0, dt),
            state,
            method="RK45",
            rtol=1e-12,
            atol=1e-14,
            t_eval=[dt],
        )
        if not solution.success:
            raise RuntimeError(f"plant integration failed: {solution.message}")
        final = solution.y[:, -1]

        self.attitude = quat.as_scalar_first(final[:4])
        self.body_rate = final[4:7]
        self.wheel_speed = final[7:10]
        self.time_s += dt

    def step_rk4(self, dt_s: float, wheel_torque_nm: npt.ArrayLike) -> None:
        """Advance by ``dt_s`` with a single fixed-step RK4 update.

        :meth:`step` integrates adaptively to ``rtol=1e-12`` at roughly 3 ms per
        call, which is far too slow to close a 45-minute planner schedule at a
        50 ms control period. This is the same :meth:`_derivative` integrated by
        classical RK4, renormalizing the quaternion afterwards so the state
        stays on the rotation manifold.

        This is a *numerical* substitution, not a physical one: the dynamics are
        unchanged, and ``tests/test_adcs_plant_fast.py`` pins the result against
        the adaptive solver to integrator tolerance.
        """
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        torque = np.asarray(wheel_torque_nm, dtype=float).reshape(3)
        if not np.all(np.isfinite(torque)):
            raise ValueError("wheel torque must be finite")

        state = np.concatenate((self.attitude, self.body_rate, self.wheel_speed))

        k1 = self._derivative(state, torque)
        k2 = self._derivative(state + 0.5 * dt * k1, torque)
        k3 = self._derivative(state + 0.5 * dt * k2, torque)
        k4 = self._derivative(state + dt * k3, torque)
        final = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.attitude = quat.as_scalar_first(final[:4])
        self.body_rate = final[4:7]
        self.wheel_speed = final[7:10]
        self.time_s += dt

    def snapshot(self) -> PlantState:
        """Return the current truth state."""
        return PlantState(
            time_s=self.time_s,
            attitude=self.attitude.copy(),
            body_rate_rad_s=self.body_rate.copy(),
            wheel_speed_rad_s=self.wheel_speed.copy(),
        )
