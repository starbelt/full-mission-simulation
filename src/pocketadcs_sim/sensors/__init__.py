"""Ideal sensor models for the PocketQube attitude determination simulation.

Every model is deliberately bare-bones: it reports the true body-frame
quantity at the sensor sampling rate and carries no noise, bias, scale
factor, misalignment, or dropout. Error terms are added later against
measured hardware data.

The datasheet-faithful hardware models (``icm40609d``,
``hardware_models``, ``vemd8080_array``) are deliberately NOT re-exported
here: they do not share the ``IdealSensorModel`` interface. Use
``hardware_adapters`` to wrap them so they can be dropped into
``collect_readings`` alongside the ideal models, which is what makes a
like-for-like noise ablation possible.
"""

from pocketadcs_sim.sensors.base import IdealSensorModel, sample_history
from pocketadcs_sim.sensors.gyroscope import GyroscopeModel
from pocketadcs_sim.sensors.hardware_adapters import (
    IMUAdapter,
    MagnetometerAdapter,
    SunArrayAdapter,
    build_hardware_sensor_suite,
)
from pocketadcs_sim.sensors.magnetometer import MagnetometerModel
from pocketadcs_sim.sensors.readings import SensorReading, collect_readings
from pocketadcs_sim.sensors.sun_reconstruction import (
    SunVectorEstimate,
    estimate_sun_vector_uncertainty_deg,
    reconstruct_sun_vector,
    reconstruct_sun_vector_batch,
)
from pocketadcs_sim.sensors.sun_sensor import SunSensorModel

__all__ = [
    "IdealSensorModel",
    "GyroscopeModel",
    "MagnetometerModel",
    "SensorReading",
    "SunSensorModel",
    "collect_readings",
    "sample_history",
    # Hardware adapter layer
    "IMUAdapter",
    "MagnetometerAdapter",
    "SunArrayAdapter",
    "build_hardware_sensor_suite",
    # Sun-vector reconstruction from the VEMD8080 array
    "SunVectorEstimate",
    "reconstruct_sun_vector",
    "reconstruct_sun_vector_batch",
    "estimate_sun_vector_uncertainty_deg",
]
