"""Tests for the hardware-sensor adapters that feed the HIL harness.

The three hardware-faithful models (ICM-40609-D, MMC5983MA, VEMD8080 array)
were written as standalone classes that do not share a common interface:
they do not subclass ``IdealSensorModel``, the magnetometer names its rate
``odr_hz`` rather than ``sample_rate_hz``, the IMU exposes ``measure_gyro``
instead of ``measure``, and the sun array reports six raw currents rather
than a 3-axis vector. ``collect_readings`` therefore cannot consume them.

These tests pin the adapter layer that makes the hardware models usable
anywhere an ``IdealSensorModel`` is expected.
"""

import numpy as np
import pytest

from pocketadcs_sim.sensors.base import IdealSensorModel, sample_history
from pocketadcs_sim.sensors.hardware_adapters import (
    IMUAdapter,
    MagnetometerAdapter,
    SunArrayAdapter,
    build_hardware_sensor_suite,
)
from pocketadcs_sim.sensors.hardware_models import MMC5983MAModel
from pocketadcs_sim.sensors.icm40609d import ICM40609DModel
from pocketadcs_sim.sensors.vemd8080_array import VEMD8080ArrayModel


class _FakeTruth:
    """A minimal stand-in exposing only what the adapters sample."""

    def __init__(self, n=64, dt=0.1):
        self.time_s = np.arange(n) * dt
        self.angular_velocity_body_rad_s = np.tile(
            np.array([[0.01, -0.02, 0.03]]), (n, 1)
        )
        self.magnetic_field_body_t = np.tile(
            np.array([[2.0e-5, -1.0e-5, 4.0e-5]]), (n, 1)
        )
        self.sun_direction_body = np.tile(np.array([[0.3, 0.4, 0.8660254]]), (n, 1))
        self.sun_direction_body /= np.linalg.norm(self.sun_direction_body, axis=1)[
            :, None
        ]
        self.illumination_fraction = np.ones(n)


def test_imu_adapter_is_an_ideal_sensor_model():
    adapter = IMUAdapter(ICM40609DModel(sample_rate_hz=10.0))
    assert isinstance(adapter, IdealSensorModel)


def test_imu_adapter_measure_matches_process_gyro():
    truth = _FakeTruth()
    model = ICM40609DModel(sample_rate_hz=10.0)
    adapter = IMUAdapter(model)
    out = adapter.measure(truth)
    indices = sample_history(truth, 10.0)
    assert out.shape == (len(indices), 3)
    assert np.allclose(out, model.process_gyro(truth.angular_velocity_body_rad_s[indices]))


def test_imu_adapter_exposes_sample_rate_hz():
    adapter = IMUAdapter(ICM40609DModel(sample_rate_hz=25.0))
    assert adapter.sample_rate_hz == 25.0


def test_magnetometer_adapter_is_an_ideal_sensor_model():
    adapter = MagnetometerAdapter(MMC5983MAModel(odr_hz=20.0))
    assert isinstance(adapter, IdealSensorModel)
    assert adapter.sample_rate_hz == 20.0


def test_magnetometer_adapter_measure_shape():
    truth = _FakeTruth()
    out = MagnetometerAdapter(MMC5983MAModel(odr_hz=20.0)).measure(truth)
    assert out.ndim == 2 and out.shape[1] == 3


def test_sun_array_adapter_is_an_ideal_sensor_model():
    adapter = SunArrayAdapter(VEMD8080ArrayModel(sample_rate_hz=10.0))
    assert isinstance(adapter, IdealSensorModel)


def test_sun_array_adapter_returns_unit_vectors():
    truth = _FakeTruth()
    out = SunArrayAdapter(VEMD8080ArrayModel(sample_rate_hz=10.0)).measure(truth)
    assert out.ndim == 2 and out.shape[1] == 3
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0)


def test_sun_array_adapter_zero_vector_during_eclipse():
    """Eclipse must yield a zero (invalid) sun vector, never a fake one."""
    truth = _FakeTruth()
    truth.illumination_fraction = np.zeros(len(truth.time_s))
    out = SunArrayAdapter(
        VEMD8080ArrayModel(sample_rate_hz=10.0, dark_current_a=0.0)
    ).measure(truth)
    assert np.allclose(out, 0.0)


def test_build_hardware_sensor_suite_keys():
    suite = build_hardware_sensor_suite(rate_hz=10.0)
    assert set(suite) == {"gyro", "magnetometer", "sun"}
    for value in suite.values():
        assert isinstance(value, IdealSensorModel)


def test_build_hardware_sensor_suite_feeds_collect_readings():
    from pocketadcs_sim.sensors.readings import collect_readings

    truth = _FakeTruth()
    readings = collect_readings(truth, build_hardware_sensor_suite(rate_hz=10.0))
    assert set(readings) == {"gyro", "magnetometer", "sun"}
    for reading in readings.values():
        assert reading.values.ndim == 2 and reading.values.shape[1] == 3


def test_build_hardware_sensor_suite_noise_knobs_are_wired():
    suite = build_hardware_sensor_suite(
        rate_hz=10.0, gyro_noise_rms_dps=0.5, mag_noise_rms_t=1e-7, sun_noise_rms_a=1e-8
    )
    assert suite["gyro"].model.gyro_noise_rms_dps == 0.5
    assert suite["magnetometer"].model.noise_rms_t == 1e-7
    assert suite["sun"].model.noise_rms_a == 1e-8


def test_build_hardware_sensor_suite_rejects_bad_rate():
    with pytest.raises(ValueError):
        build_hardware_sensor_suite(rate_hz=0.0)
    with pytest.raises(ValueError):
        build_hardware_sensor_suite(rate_hz=-5.0)


def test_adapters_are_deterministic_for_a_seed():
    truth = _FakeTruth()
    suite_a = build_hardware_sensor_suite(rate_hz=10.0, seed=1234)
    suite_b = build_hardware_sensor_suite(rate_hz=10.0, seed=1234)
    for key in ("gyro", "magnetometer", "sun"):
        assert np.array_equal(suite_a[key].measure(truth), suite_b[key].measure(truth))


class TestHarnessIntegration:
    """The HIL harness must be able to stream hardware-modelled frames.

    Before the adapter layer existed, the harness could only use the ideal
    noise-free models, so the datasheet work never reached the board. These
    tests pin the sensor-suite selection that the harness now exposes via
    ``--sensors``.
    """

    def test_select_ideal_returns_the_ideal_models(self):
        from pocketadcs_sim.sensors.hardware_adapters import select_sensor_suite

        suite = select_sensor_suite("ideal", rate_hz=10.0)
        assert isinstance(suite["gyro"], IdealSensorModel)
        # The ideal models are the originals, not adapters.
        from pocketadcs_sim.sensors import GyroscopeModel, MagnetometerModel, SunSensorModel

        assert isinstance(suite["gyro"], GyroscopeModel)
        assert isinstance(suite["magnetometer"], MagnetometerModel)
        assert isinstance(suite["sun"], SunSensorModel)

    def test_select_hardware_returns_adapters(self):
        from pocketadcs_sim.sensors.hardware_adapters import (
            IMUAdapter,
            MagnetometerAdapter,
            SunArrayAdapter,
            select_sensor_suite,
        )

        suite = select_sensor_suite("hardware", rate_hz=10.0)
        assert isinstance(suite["gyro"], IMUAdapter)
        assert isinstance(suite["magnetometer"], MagnetometerAdapter)
        assert isinstance(suite["sun"], SunArrayAdapter)

    def test_hardware_noisy_is_the_noisy_hardware_suite(self):
        from pocketadcs_sim.sensors.hardware_adapters import select_sensor_suite

        suite = select_sensor_suite("hardware_noisy", rate_hz=10.0, seed=3)
        assert suite["gyro"].model.gyro_noise_rms_dps > 0.0
        assert suite["magnetometer"].model.noise_rms_t > 0.0
        assert suite["sun"].model.noise_rms_a > 0.0

    def test_unknown_sensor_suite_raises(self):
        from pocketadcs_sim.sensors.hardware_adapters import select_sensor_suite

        with pytest.raises(ValueError):
            select_sensor_suite("nonsense", rate_hz=10.0)

    def test_select_suite_rejects_bad_rate(self):
        from pocketadcs_sim.sensors.hardware_adapters import select_sensor_suite

        with pytest.raises(ValueError):
            select_sensor_suite("hardware", rate_hz=0.0)

    def test_all_suites_have_the_same_keys(self):
        from pocketadcs_sim.sensors.hardware_adapters import select_sensor_suite

        keys = [
            set(select_sensor_suite(name, rate_hz=10.0))
            for name in ("ideal", "hardware", "hardware_noisy")
        ]
        assert keys[0] == keys[1] == keys[2]
