"""Tests for the truth plant: momentum, gyroscopic coupling, analytic agreement.

These are the tests that decide whether the plant is a physical model or a
plausible-looking one, so they check conservation laws and an independent
implementation rather than round-tripping the plant against itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from pocketadcs_sim.adcs import quaternions as quat
from pocketadcs_sim.adcs.plant import PlantConfig, TruthPlant

INERTIA = (0.02, 0.025, 0.03)
J = 3.27e-6


def make_plant(**kwargs: object) -> TruthPlant:
    """Return a plant with the TGR 29-12 triad and the given overrides."""
    config = PlantConfig(hub_inertia_kg_m2=INERTIA)
    return TruthPlant(config=config, initial_attitude=[1.0, 0.0, 0.0, 0.0], **kwargs)  # type: ignore[arg-type]


def test_effective_inertia_includes_rotor_inertia_along_each_axis() -> None:
    """Each rotor adds J to its own axis and nothing to the others."""
    plant = make_plant()

    expected = np.diag(np.asarray(INERTIA)) + J * np.eye(3)
    np.testing.assert_allclose(plant.effective_inertia, expected, atol=1e-18)


def test_single_wheel_matches_the_analytic_momentum_wheel_solution() -> None:
    """One wheel on a free hub: the hub accelerates at exactly -u / I_eff.

    The textbook 1D momentum-wheel result. A sign error or a missing rotor
    inertia shows up here because there is no 3D coupling to hide behind. Note
    the hub does **not** settle: spinning a wheel up at constant torque drives
    the hub at a constant counter-rate for as long as the torque is applied.
    Only the rotor speed is unbounded.
    """
    plant = make_plant()
    dt = 0.01
    steps = 2000
    torque = np.array([1e-4, 0.0, 0.0])
    inertia = np.asarray(INERTIA) + J

    for _ in range(steps):
        plant.step(dt, torque)
    time = steps * dt

    np.testing.assert_allclose(plant.wheel_speed, [torque[0] * time / J, 0.0, 0.0], rtol=1e-9)
    np.testing.assert_allclose(
        plant.body_rate[0], -torque[0] * time / inertia[0], rtol=1e-6
    )
    np.testing.assert_allclose(plant.body_rate[1:], [0.0, 0.0], atol=1e-14)


def test_rotor_torque_on_hub_is_equal_and_opposite() -> None:
    """Hub feels exactly the negative of what the rotors are given."""
    plant = make_plant()
    torque = np.array([1e-5, -2e-5, 3e-6])

    reaction = plant.rotor_torque_on_hub(torque)

    np.testing.assert_allclose(reaction, -torque, atol=1e-18)


def test_total_angular_momentum_is_conserved_with_no_external_torque() -> None:
    """Torque-free: hub + rotor momentum stays constant in the inertial frame.

    Checked in the inertial frame, which is the frame where "conserved" means
    something. Body-frame components legitimately change as the body rotates.
    """
    plant = make_plant(initial_body_rate_rad_s=[0.01, -0.02, 0.03])
    def inertial_momentum() -> np.ndarray:
        return quat.to_dcm(plant.attitude).T @ plant.total_angular_momentum_body

    initial = inertial_momentum()
    for _ in range(500):
        plant.step(0.01, np.zeros(3))

    np.testing.assert_allclose(inertial_momentum(), initial, rtol=1e-8, atol=1e-15)


def test_rotor_momentum_precesses_the_body_axis() -> None:
    """Locked rotor momentum drives the body rate off its spin axis.

    With the wheels held at speed, their momentum ``J * Omega`` acts as an
    external-looking gyroscopic term, so a body spinning about +X develops a
    +Y rate. This is genuine torque-free 3D motion (Euler precession about the
    total momentum vector), so it is checked as a *qualitative but sharp* fact --
    the off-axis rate must appear and be the same sign -- rather than against a
    linearization that does not describe this regime.
    """
    plant = make_plant(
        initial_body_rate_rad_s=[0.05, 0.0, 0.0],
        initial_wheel_speed_rad_s=[0.0, 0.0, 400.0],
    )
    baseline = make_plant(initial_body_rate_rad_s=[0.05, 0.0, 0.0])

    for _ in range(2000):
        plant.step(0.005, np.zeros(3))
        baseline.step(0.005, np.zeros(3))

    assert abs(baseline.body_rate[1]) < 1e-12, "stopped wheels must not precess"
    assert plant.body_rate[1] > 1e-3, "rotor momentum must drive an off-axis rate"

    # Conservation pins the motion: |H| is invariant, so the spinning case's
    # total momentum (hub + rotor) must still have its initial magnitude.
    # The stopped-wheel baseline is a different state and is *not* comparable
    # here -- it carries no rotor momentum at all.
    inertia = plant.effective_inertia
    initial_momentum = inertia @ np.array([0.05, 0.0, 0.0]) + J * np.array(
        [0.0, 0.0, 400.0]
    )
    final_momentum = inertia @ plant.body_rate + J * plant.wheel_speed
    assert np.linalg.norm(final_momentum) == pytest.approx(
        np.linalg.norm(initial_momentum), rel=1e-9
    )


def test_three_wheel_torque_produces_the_expected_coupled_body_rate() -> None:
    """A 3-axis command matches an independent integration of the same physics.

    The reference here re-derives the coupled ODE from the momentum equations
    and integrates it with a different solver and step size, so it is a real
    cross-check of the plant rather than a restatement of its internals. The
    earlier version of this test used forward Euler on a linearization that
    dropped both the rotor inertia and the Euler cross term, which is only
    valid at first order in the rate -- it disagreed with the plant by 2.8%.
    """
    torque = np.array([4e-5, -3e-5, 2e-5])
    wheel_speed = np.array([10.0, -20.0, 30.0])
    initial_rate = np.array([0.02, -0.01, 0.015])
    plant = make_plant(
        initial_body_rate_rad_s=initial_rate, initial_wheel_speed_rad_s=wheel_speed
    )

    duration = 2.0
    for _ in range(2000):
        plant.step(0.001, torque)

    inertia = np.diag(np.asarray(INERTIA)) + J * np.eye(3)

    def reference(_t: float, state: np.ndarray) -> np.ndarray:
        rate = state[:3]
        wheel = state[3:6]
        rate_dot = np.linalg.solve(
            inertia,
            -torque
            - np.cross(rate, inertia @ rate)
            - np.cross(rate, J * wheel),
        )
        return np.concatenate((rate_dot, torque / J))

    from scipy.integrate import solve_ivp

    solution = solve_ivp(
        reference,
        (0.0, duration),
        np.concatenate((initial_rate, wheel_speed)),
        method="Radau",
        rtol=1e-12,
        atol=1e-15,
        t_eval=[duration],
    )
    reference_rate = solution.y[:3, -1]
    reference_wheel = solution.y[3:6, -1]

    np.testing.assert_allclose(plant.wheel_speed, reference_wheel, rtol=1e-8)
    np.testing.assert_allclose(plant.body_rate, reference_rate, rtol=1e-6, atol=1e-12)


def test_torque_free_effective_momentum_is_preserved_without_wheels() -> None:
    """With zero rotor speed the invariant is |I_eff omega|, not |omega|.

    |omega| is only constant for a spherical body; with anisotropic inertia it
    wobbles by ~0.1% while |I_eff omega| holds to machine precision. Asserting
    the wrong invariant would either fail spuriously or, worse, pass for a
    wrong reason at a loose tolerance.
    """
    plant = make_plant(initial_body_rate_rad_s=[0.01, 0.02, -0.03])
    effective = plant.effective_inertia
    initial = float(np.linalg.norm(effective @ plant.body_rate))

    for _ in range(1000):
        plant.step(0.01, np.zeros(3))

    assert float(np.linalg.norm(effective @ plant.body_rate)) == pytest.approx(
        initial, rel=1e-9
    )


def test_initial_wheel_speed_is_preserved_when_no_torque_is_applied() -> None:
    """A command timeout must not reset rotor speed; the wheels coast."""
    plant = make_plant(initial_wheel_speed_rad_s=[100.0, -200.0, 300.0])
    start = plant.wheel_speed.copy()

    for _ in range(100):
        plant.step(0.01, np.zeros(3))

    np.testing.assert_allclose(plant.wheel_speed, start, rtol=1e-12, atol=1e-9)


def test_step_rejects_invalid_dt_and_nonfinite_torque() -> None:
    """Bad inputs are refused loudly rather than producing silent NaN state."""
    plant = make_plant()

    with pytest.raises(ValueError):
        plant.step(0.0, np.zeros(3))
    with pytest.raises(ValueError):
        plant.step(-0.01, np.zeros(3))
    with pytest.raises(ValueError):
        plant.step(0.01, np.array([np.nan, 0.0, 0.0]))


def test_reversing_wheel_torque_brings_the_hub_rate_back_to_zero() -> None:
    """A wheel can be spun down and the hub returns to rest (momentum exchange)."""
    plant = make_plant()
    for _ in range(1000):
        plant.step(0.01, np.array([1e-4, 0.0, 0.0]))
    spun_up = plant.wheel_speed[0]
    assert spun_up > 0.0
    assert plant.body_rate[0] < 0.0

    for _ in range(1000):
        plant.step(0.01, np.array([-1e-4, 0.0, 0.0]))

    assert plant.wheel_speed[0] == pytest.approx(0.0, abs=1e-6)
    assert plant.body_rate[0] == pytest.approx(0.0, abs=1e-9)
