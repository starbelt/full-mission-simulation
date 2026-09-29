"""Adapters presenting the hardware-faithful sensors through one interface.

The datasheet-driven models in this package were each written standalone and
do not share a common interface:

* ``MMC5983MAModel`` names its sampling rate ``odr_hz``, not
  ``sample_rate_hz``.
* ``ICM40609DModel`` exposes ``measure_gyro`` (and
  ``process_specific_force``) rather than ``measure``, because the truth
  environment carries no specific-force quantity.
* ``VEMD8080ArrayModel`` reports six raw photocurrents, not a 3-axis sun
  vector, so it needs a reconstruction step before it can feed TRIAD.

None of them subclass ``IdealSensorModel``, so ``collect_readings`` -- which
requires ``sensor.measure(truth)`` and ``sensor.sample_rate_hz`` -- cannot
consume them, and the HIL harness was therefore forced to use the ideal
noise-free models.

This module provides thin adapters that wrap each hardware model in the
``IdealSensorModel`` interface without modifying the models themselves, so
both ideal and hardware-faithful sensor suites drop into the same harness
and become directly comparable in an ablation study.

Eclipse handling
----------------
``SunArrayAdapter`` returns the **zero vector** for samples the array cannot
resolve (eclipse, or a non-dominant illumination pattern). This is
deliberate: the firmware treats a zero sun vector as an invalid reference
and flags it, so an unresolvable sample degrades gracefully instead of
injecting a fabricated direction into the attitude solution.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pocketadcs_sim.sensors.base import IdealSensorModel, sample_history
from pocketadcs_sim.sensors.hardware_models import MMC5983MAModel
from pocketadcs_sim.sensors.icm40609d import ICM40609DModel
from pocketadcs_sim.sensors.sun_reconstruction import reconstruct_sun_vector_batch
from pocketadcs_sim.sensors.vemd8080_array import (
    VEMD8080_DEFAULT_FULL_SCALE_A,
    VEMD8080ArrayModel,
)
from pocketadcs_sim.truth_environment import TruthEnvironmentHistory

FloatArray = npt.NDArray[np.float64]


class _HardwareAdapter(IdealSensorModel):
    """Base class wiring a hardware model into ``IdealSensorModel``."""

    def __init__(self, model: object, sample_rate_hz: float) -> None:
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be a finite positive value")
        self.model = model
        super().__init__(sample_rate_hz=sample_rate_hz)

    def truth_quantity(self, truth: TruthEnvironmentHistory) -> FloatArray:
        raise NotImplementedError


class IMUAdapter(_HardwareAdapter):
    """Present an ``ICM40609DModel`` gyro as an ``IdealSensorModel``.

    The truth environment exposes angular velocity but no specific force,
    so this adapter samples the **gyroscope** only. Accelerometer sampling
    requires an explicit specific-force input and is available through the
    wrapped model via ``process_specific_force``.
    """

    def __init__(self, model: ICM40609DModel) -> None:
        super().__init__(model, sample_rate_hz=model.sample_rate_hz)

    def truth_quantity(self, truth: TruthEnvironmentHistory) -> FloatArray:
        return np.asarray(truth.angular_velocity_body_rad_s, dtype=float)

    def measure(self, truth: TruthEnvironmentHistory) -> FloatArray:
        indices = sample_history(truth, self.sample_rate_hz)
        sampled = np.asarray(truth.angular_velocity_body_rad_s, dtype=float)[indices]
        return self.model.process_gyro(sampled)


class MagnetometerAdapter(_HardwareAdapter):
    """Present an ``MMC5983MAModel`` as an ``IdealSensorModel``.

    Normalizes the model's ``odr_hz`` to the shared ``sample_rate_hz`` name.
    """

    def __init__(self, model: MMC5983MAModel) -> None:
        super().__init__(model, sample_rate_hz=model.odr_hz)

    def truth_quantity(self, truth: TruthEnvironmentHistory) -> FloatArray:
        return np.asarray(truth.magnetic_field_body_t, dtype=float)

    def measure(self, truth: TruthEnvironmentHistory) -> FloatArray:
        indices = sample_history(truth, self.sample_rate_hz)
        return self.model.process_field(
            np.asarray(truth.magnetic_field_body_t, dtype=float)[indices]
        )


class SunArrayAdapter(_HardwareAdapter):
    """Present a ``VEMD8080ArrayModel`` as a 3-axis ``IdealSensorModel``.

    Runs the six raw channels through the least-squares sun reconstruction
    so the harness receives the unit sun vector the firmware expects.
    Samples that cannot be resolved (eclipse / degenerate illumination) come
    back as the zero vector, which the firmware flags as an invalid
    reference rather than trusting.

    The adapter supplies an absolute ``lit_threshold`` to the reconstruction.
    This is not optional: the array model clips readings at zero, which
    rectifies negative noise, so a deep-eclipse sample otherwise looks like
    a lit face and yields a garbage direction (~90 deg attitude errors). The
    threshold is derived from the model's own per-channel scale, so it tracks
    whatever calibration the model was configured with.
    """

    #: Fraction of the configured per-channel scale that must be exceeded for
    #: a sample to count as lit. Well above the dark current and noise floor,
    #: well below a real lit reading.
    LIT_FRACTION = 0.05

    def __init__(self, model: VEMD8080ArrayModel) -> None:
        super().__init__(model, sample_rate_hz=model.sample_rate_hz)
        self._lit_threshold = self.LIT_FRACTION * float(np.max(model.scale))

    def truth_quantity(self, truth: TruthEnvironmentHistory) -> FloatArray:
        return np.asarray(truth.sun_direction_body, dtype=float)

    def measure(self, truth: TruthEnvironmentHistory) -> FloatArray:
        indices = sample_history(truth, self.sample_rate_hz)
        channels = self.model.process_sun(
            np.asarray(truth.sun_direction_body, dtype=float)[indices],
            np.asarray(truth.illumination_fraction, dtype=float)[indices],
        )
        estimates = reconstruct_sun_vector_batch(
            channels, lit_threshold=self._lit_threshold
        )
        if not estimates:
            return np.zeros((0, 3), dtype=float)
        return np.vstack([np.asarray(est.vector, dtype=float) for est in estimates])


def select_sensor_suite(
    name: str,
    rate_hz: float,
    seed: int | None = None,
    gyro_noise_rms_dps: float = 0.0,
    mag_noise_rms_t: float = 0.0,
    sun_noise_rms_a: float = 0.0,
) -> dict[str, IdealSensorModel]:
    """Choose a sensor suite by name, for the HIL harness ``--sensors`` flag.

    All three options return the same ``{"gyro", "magnetometer", "sun"}``
    keys so the harness, the study runner and any caller can swap one for
    another without special-casing.

    * ``ideal`` -- the original noise-free models. Reproduces the
      pre-datasheet behaviour exactly, which is what makes it the correct
      baseline for a like-for-like comparison.
    * ``hardware`` -- the datasheet-faithful models (ICM-40609-D,
      MMC5983MA, VEMD8080 array) with analog noise off but ADC
      quantization active.
    * ``hardware_noisy`` -- the same, with the datasheet noise levels on.

    Raises:
        ValueError: on an unknown name or a non-positive ``rate_hz``.
    """
    from pocketadcs_sim.sensors.gyroscope import GyroscopeModel
    from pocketadcs_sim.sensors.magnetometer import MagnetometerModel
    from pocketadcs_sim.sensors.sun_sensor import SunSensorModel

    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("rate_hz must be a finite positive value")

    if name == "ideal":
        return {
            "gyro": GyroscopeModel(sample_rate_hz=rate_hz),
            "magnetometer": MagnetometerModel(sample_rate_hz=rate_hz),
            "sun": SunSensorModel(sample_rate_hz=rate_hz),
        }
    if name in ("hardware", "hardware_noisy"):
        noisy = name == "hardware_noisy"
        # `hardware_noisy` with no explicit levels means the datasheet
        # defaults, so the flag is useful without further arguments.
        if noisy:
            from pocketadcs_sim.study import GYRO_NOISE_DPS, MAG_NOISE_T, SUN_NOISE_A

            gyro_level = gyro_noise_rms_dps or GYRO_NOISE_DPS
            mag_level = mag_noise_rms_t or MAG_NOISE_T
            sun_level = sun_noise_rms_a or SUN_NOISE_A
        else:
            gyro_level = mag_level = sun_level = 0.0
        return build_hardware_sensor_suite(
            rate_hz=rate_hz,
            gyro_noise_rms_dps=gyro_level,
            mag_noise_rms_t=mag_level,
            sun_noise_rms_a=sun_level,
            seed=seed,
        )
    raise ValueError(
        f"unknown sensor suite {name!r}; expected 'ideal', 'hardware' or 'hardware_noisy'"
    )


def build_hardware_sensor_suite(
    rate_hz: float,
    gyro_noise_rms_dps: float = 0.0,
    gyro_bias_dps: float = 0.0,
    mag_noise_rms_t: float = 0.0,
    mag_bias_counts: float = 0.0,
    sun_noise_rms_a: float = 0.0,
    sun_full_scale_a: float = VEMD8080_DEFAULT_FULL_SCALE_A,
    seed: int | None = None,
) -> dict[str, IdealSensorModel]:
    """Build the full datasheet-faithful sensor suite at one common rate.

    Every sensor is sampled at ``rate_hz`` so the three readings in a frame
    are time-coherent, which the TRIAD estimator requires.

    ``sun_full_scale_a`` is the analog front end's range. It must be stated
    in the same units as ``sun_noise_rms_a`` or the ADC grid will swallow the
    noise whole: at 1 A full scale a 1 uA rms noise is 8 orders of magnitude
    below one 12-bit code and the sun channel reports no noise at all.

    All noise defaults are 0 (deterministic zero-noise mode), so the suite
    reproduces the ideal models until noise is deliberately injected. This
    is what makes the noise ablation meaningful: turning one knob on at a
    time isolates that sensor's contribution to the error budget.

    Args:
        rate_hz: common sampling rate for all three sensors.
        gyro_noise_rms_dps: gyro white-noise RMS in deg/s.
        gyro_bias_dps: constant per-axis gyro bias in deg/s.
        mag_noise_rms_t: magnetometer noise RMS in Tesla.
        mag_bias_counts: magnetometer bias in ADC counts.
        sun_noise_rms_a: per-channel photodiode noise RMS in amperes.
        seed: shared RNG seed for reproducibility.

    Returns:
        Mapping of ``{"gyro", "magnetometer", "sun"}`` to adapters that
        satisfy the ``IdealSensorModel`` interface.
    """
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("rate_hz must be a finite positive value")

    gyro = ICM40609DModel(
        sample_rate_hz=rate_hz,
        gyro_noise_rms_dps=gyro_noise_rms_dps,
        gyro_bias_dps=gyro_bias_dps,
        seed=seed,
    )
    magnetometer = MMC5983MAModel(
        odr_hz=rate_hz, noise_rms_t=mag_noise_rms_t, bias_counts=mag_bias_counts, seed=seed
    )
    sun = VEMD8080ArrayModel(
        sample_rate_hz=rate_hz,
        noise_rms_a=sun_noise_rms_a,
        full_scale_a=sun_full_scale_a,
        seed=seed,
    )
    return {
        "gyro": IMUAdapter(gyro),
        "magnetometer": MagnetometerAdapter(magnetometer),
        "sun": SunArrayAdapter(sun),
    }
