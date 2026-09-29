"""Ideal three-axis gyroscope model."""

from __future__ import annotations

from pocketadcs_sim.sensors.base import IdealSensorModel
from pocketadcs_sim.truth_environment import TruthEnvironmentHistory


class GyroscopeModel(IdealSensorModel):
    """Reports the true body-frame angular rate with no bias or noise."""

    def truth_quantity(self, truth: TruthEnvironmentHistory) -> float:
        return truth.angular_velocity_body_rad_s
