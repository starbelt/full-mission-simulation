"""Ideal three-axis magnetometer model."""

from __future__ import annotations

from pocketadcs_sim.sensors.base import IdealSensorModel
from pocketadcs_sim.truth_environment import TruthEnvironmentHistory


class MagnetometerModel(IdealSensorModel):
    """Reports the true body-frame magnetic field vector with no error terms."""

    def truth_quantity(self, truth: TruthEnvironmentHistory) -> float:
        return truth.magnetic_field_body_t
