"""Typed sensor readings sampled from the truth histories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pocketadcs_sim.sensors.base import IdealSensorModel
from pocketadcs_sim.truth_environment import TruthEnvironmentHistory

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SensorReading:
    """One sampled measurement from one sensor, in SI units."""

    sensor: str
    sample_rate_hz: float
    values: FloatArray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("values must be a (N, 3) array")


def collect_readings(
    truth: TruthEnvironmentHistory,
    sensors: dict[str, IdealSensorModel],
) -> dict[str, SensorReading]:
    """Sample every ideal sensor model over the truth history."""
    readings = {}
    for name, sensor in sensors.items():
        readings[name] = SensorReading(
            sensor=name,
            sample_rate_hz=sensor.sample_rate_hz,
            values=sensor.measure(truth),
        )
    return readings
