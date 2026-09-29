"""Ideal three-axis sun-sensor model."""

from __future__ import annotations

from pocketadcs_sim.sensors.base import IdealSensorModel
from pocketadcs_sim.truth_environment import TruthEnvironmentHistory


class SunSensorModel(IdealSensorModel):
    """Reports the true body-frame unit Sun direction, including eclipsed runs."""

    def truth_quantity(self, truth: TruthEnvironmentHistory) -> float:
        return truth.sun_direction_body
