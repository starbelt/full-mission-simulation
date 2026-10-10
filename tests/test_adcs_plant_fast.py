"""Tests for the fast fixed-step plant integrator.

``TruthPlant.step`` integrates with ``solve_ivp`` for maximum fidelity, which
costs ~3 ms per call -- far too slow to close a 45-minute planner schedule at a
50 ms control period. The closed loop therefore uses a fixed-step RK4 update,
which must agree with the adaptive solver to integrator tolerance.

This file pins that agreement, so the speed-up is a documented numerical choice
rather than an unexamined swap.
"""

from __future__ import annotations

import numpy as np
import pytest

from pocketadcs_sim.adcs import quaternions as quat
from pocketadcs_sim.adcs.plant import PlantConfig, TruthPlant

CONFIG = PlantConfig(hub_inertia_kg_m2=(0.02, 0.025, 0.03))


def make_plant(**kwargs: object) -> TruthPlant:
    return TruthPlant(
        config=CONFIG,
        initial_attitude=[1.0, 0.0, 0.0, 0.0],
        **kwargs,  # type: ignore[arg-type]
    )


def test_rk4_step_agrees_with_the_adaptive_solver() -> None:
    """The fast path must not change the trajectory beyond integrator error."""
    slow = make_plant()
    fast = make_plant()
    torque = [2.0e-4, -1.0e-4, 5.0e-5]

    for _ in range(20):
        slow.step(0.05, torque)
        fast.step_rk4(0.05, torque)

    assert fast.body_rate == pytest.approx(slow.body_rate, abs=1e-12)
    assert fast.attitude == pytest.approx(slow.attitude, abs=1e-12)
    assert fast.wheel_speed == pytest.approx(slow.wheel_speed, abs=1e-12)


def test_rk4_step_converges_as_the_step_shrinks() -> None:
    """Halving dt moves the RK4 result toward the reference, fourth order."""
    # The reference must span the same 20 s as the RK4 runs below.
    reference = make_plant()
    for _ in range(400):
        reference.step(0.05, [1e-4, 0.0, 0.0])

    # The body rate is near-linear under this torque, so its own truncation
    # error stays at the rounding floor. Attitude error is the observable that
    # actually accumulates, so measure that against the adaptive reference.
    def deviation(dt_s: float) -> float:
        plant = make_plant()
        for _ in range(round(20.0 / dt_s)):
            plant.step_rk4(dt_s, [1e-4, 0.0, 0.0])
        return float(
            np.linalg.norm(quat.to_dcm(plant.attitude) - quat.to_dcm(reference.attitude))
        )

    # Deliberately coarse: at the 50 ms control step the RK4 error is already
    # ~6e-13 over 20 s, far below anything the closed loop observes, so the
    # order is demonstrated where truncation error actually dominates.
    coarse = deviation(1.0)
    fine = deviation(0.5)

    assert coarse > 1e-9
    assert fine < coarse
    assert fine / coarse < 0.1  # ~2^-4 = 0.0625


def test_rk4_preserves_total_momentum_magnitude() -> None:
    """Momentum conservation is a physical invariant, not an integrator detail."""
    plant = make_plant(initial_body_rate_rad_s=(0.01, -0.02, 0.03))
    initial = float(np.linalg.norm(plant.total_angular_momentum_body))

    for _ in range(500):
        plant.step_rk4(0.05, [1e-4, 0.0, 0.0])

    assert float(np.linalg.norm(plant.total_angular_momentum_body)) == pytest.approx(
        initial, rel=1e-10
    )


def test_rk4_keeps_the_quaternion_normalized() -> None:
    """RK4 on the quaternion manifold must not let the norm drift."""
    plant = make_plant()
    for _ in range(2000):
        plant.step_rk4(0.05, [3e-4, 1e-4, -2e-4])

    assert float(np.linalg.norm(plant.attitude)) == pytest.approx(1.0, abs=1e-12)


def test_rk4_rejects_invalid_steps() -> None:
    """A bad dt must raise rather than silently integrating garbage."""
    plant = make_plant()

    with pytest.raises(ValueError):
        plant.step_rk4(0.0, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        plant.step_rk4(-0.01, [0.0, 0.0, 0.0])
