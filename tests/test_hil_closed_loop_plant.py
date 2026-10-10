"""Physical HIL world uses the accepted coupled hub/rotor model.

The removed provisional tests asserted artificial viscous damping and invented
inertia. They tested a rejected plant, not the accepted physical configuration.
The original sign assertion also failed before replacement (see task baseline).
"""
import numpy as np
import pytest

from examples.hil_closed_loop_run import advance_world, make_world
from pocketadcs_sim.adcs import quaternions as quat
from pocketadcs_sim.adcs.closed_loop import ClosedLoopConfig
from pocketadcs_sim.adcs.rotors import TGR2912


def test_free_coupled_system_conserves_inertial_momentum_without_damping():
    plant, board = make_world(ClosedLoopConfig())
    plant.body_rate = np.array([0.01, -0.02, 0.03])
    plant.wheel_speed = np.array([30., -40., 50.])
    before = quat.to_dcm(plant.attitude).T @ plant.total_angular_momentum_body
    for _ in range(100):
        advance_world(plant, np.zeros(3), 0.05)
    after = quat.to_dcm(plant.attitude).T @ plant.total_angular_momentum_body
    assert after == pytest.approx(before, abs=1e-12)
    assert np.linalg.norm(plant.body_rate) > 0.03
    assert board.wheel_set.inertia == TGR2912.rotor_inertia_kg_m2


def test_returned_positive_wheel_torque_reacts_negatively_on_hub():
    plant, board = make_world(ClosedLoopConfig())
    applied = board.update([1e-4, 0, 0], 0.05)
    advance_world(plant, applied, 0.05)
    assert plant.body_rate[0] < 0
    assert plant.wheel_speed[0] > 0
    assert plant.wheel_speed == pytest.approx(board.wheel_speed_rad_s)


def test_wheel_board_applies_real_cap_before_coupled_integration():
    plant, board = make_world(ClosedLoopConfig())
    applied = board.update([999, -999, 999], 0.05)
    cap = board.wheel_set.inertia * board.wheel_set.acceleration_cap_rad_s2
    assert applied == pytest.approx([cap, -cap, cap])
    advance_world(plant, applied, 0.05)
    assert plant.wheel_speed == pytest.approx([10, -10, 10])
