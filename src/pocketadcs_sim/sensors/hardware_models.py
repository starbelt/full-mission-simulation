"""Hardware-faithful sensor models for the ADS-2 flight hardware.

These models are intentionally separate from the ideal models in
``base.py`` / ``magnetometer.py`` / ``sun_sensor.py``: they start from the
same body-frame truth quantity and then apply the documented behavior of
the selected flight hardware -- digital quantization, full-scale
saturation, configurable bias, and reproducible noise. The ideal models
and their public API are untouched.

Documented spec source (full citations in ``docs/hardware_sensor_models.md``):

* MEMSIC MMC5983MA datasheet, Rev A (2019-04-03), primary:
  https://www.memsic.com/Public/Uploads/uploadfile/files/20220119/MMC5983MADatasheetRevA.pdf

Deferred to a later bounded workstream (NOT modeled here, no values
assumed): TDK ICM-40609-D gyro/accelerometer and the Vishay VEMD8080
six-photodiode body-frame sun array.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pocketadcs_sim.sensors.base import sample_history
from pocketadcs_sim.truth_environment import TruthEnvironmentHistory

FloatArray = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# MEMSIC MMC5983MA -- documented datasheet values (Rev A). See the docs
# file for the exact source locations and verification status.
# ---------------------------------------------------------------------------
GAUSS_TO_TESLA = 1.0e-4
MMC5983_MAX_ODR_HZ = 1000.0  # BW=11 with CM_Freq=111

# 18-bit operating mode (documented): 16384 counts/G sensitivity and a
# 131072-count null-field offset; output codes are unsigned.
MMC5983_18BIT = {
    "null_offset_counts": 131072,
    "counts_per_gauss": 16384,
    "lsb_gauss": 1.0 / 16384.0,
    "code_min": 0,
    "code_max": (1 << 18) - 1,  # 262143
}

# 16-bit operating mode (documented): 4096 counts/G, 32768-count offset.
MMC5983_16BIT = {
    "null_offset_counts": 32768,
    "counts_per_gauss": 4096,
    "lsb_gauss": 1.0 / 4096.0,
    "code_min": 0,
    "code_max": (1 << 16) - 1,  # 65535
}

# BW bit setting -> documented ODR and total RMS noise (1.2 mG also
# applies to the 1000 Hz ODR of BW=11/CM_Freq=111).
MMC5983_BANDS: dict[int, dict[str, float]] = {
    0: {"odr_hz": 50.0, "noise_rms_t": 0.4e-6},
    1: {"odr_hz": 100.0, "noise_rms_t": 0.6e-6},
    2: {"odr_hz": 225.0, "noise_rms_t": 0.8e-6},
    3: {"odr_hz": 580.0, "noise_rms_t": 1.2e-6},
}


def _mode_table(bits: int) -> dict[str, float | int]:
    if bits == 18:
        return MMC5983_18BIT
    if bits == 16:
        return MMC5983_16BIT
    raise ValueError("bits must be 16 or 18 (documented MMC5983 operating modes)")


def field_to_counts(field_t: npt.ArrayLike, bits: int = 18) -> np.ndarray:
    """Quantize a magnetic field in Tesla to documented MMC5983 unsigned codes.

    Transfer function (datasheet): code = round(null_offset +
    field[Gauss] * counts_per_gauss), clipped to the unsigned code range
    (full-scale saturation at the +/-8 G FSR).
    """
    x = np.asarray(field_t, dtype=float)
    table = _mode_table(bits)
    gauss = x / GAUSS_TO_TESLA
    counts = np.rint(gauss * table["counts_per_gauss"] + table["null_offset_counts"])
    return np.clip(counts, table["code_min"], table["code_max"]).astype(np.int64)


def counts_to_field_t(counts: npt.ArrayLike, bits: int = 18) -> np.ndarray:
    """Dequantize MMC5983 unsigned codes back to a field in Tesla."""
    x = np.asarray(counts, dtype=float)
    table = _mode_table(bits)
    return (x - table["null_offset_counts"]) / table["counts_per_gauss"] * GAUSS_TO_TESLA


class MMC5983MAModel:
    """Simulated MEMSIC MMC5983MA 3-axis magnetometer.

    Output: body-frame magnetic field in Tesla, shape (N, 3), sampled at
    ``odr_hz``. Signal chain: truth field -> additive Gaussian noise (zero
    when ``noise_rms_t`` is 0, i.e. the deterministic zero-noise mode) ->
    per-axis digital quantization to the documented unsigned code range
    (full-scale saturation at the +/-8 G FSR) -> additive bias in counts ->
    dequantization to Tesla.

    Determinism: for a fixed ``seed`` and a given input array, repeated
    calls return identical outputs (the noise stream is re-seeded from
    ``seed`` on every call and depends only on the input shape).

    Config validation mirrors the documented device limits; values marked
    "assumption" in ``docs/hardware_sensor_models.md`` are not datasheet
    figures.
    """

    def __init__(
        self,
        odr_hz: float = 50.0,
        bits: int = 18,
        noise_rms_t: float = 0.0,
        bias_counts: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if odr_hz <= 0.0:
            raise ValueError("odr_hz must be positive")
        if odr_hz > MMC5983_MAX_ODR_HZ:
            raise ValueError(
                f"odr_hz must be <= {MMC5983_MAX_ODR_HZ:.0f} Hz (documented maximum)"
            )
        if bits not in (16, 18):
            raise ValueError("bits must be 16 or 18 (documented operating modes)")
        if not np.isfinite(noise_rms_t) or noise_rms_t < 0.0:
            raise ValueError("noise_rms_t must be a finite value >= 0")
        if not np.isfinite(bias_counts):
            raise ValueError("bias_counts must be finite")
        self.odr_hz = float(odr_hz)
        self.bits = bits
        self.noise_rms_t = float(noise_rms_t)
        self.bias_counts = float(bias_counts)
        self.seed = seed
        self._table = _mode_table(bits)

    def process_field(self, field_t: npt.ArrayLike) -> FloatArray:
        """Apply noise, quantization, saturation and bias to a field array.

        Accepts (N, 3) or (3,) Tesla input; always returns (N, 3) Tesla.
        """
        x = np.asarray(field_t, dtype=float)
        if x.ndim == 1:
            if x.shape[0] != 3:
                raise ValueError("1-D field input must have 3 axes")
            x = x.reshape(1, 3)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("field_t must have shape (N, 3) or (3,)")
        if self.noise_rms_t > 0.0:
            rng = np.random.default_rng(self.seed)
            x = x + rng.normal(0.0, self.noise_rms_t, size=x.shape)
        counts = field_to_counts(x, bits=self.bits).astype(float)
        counts = np.clip(counts + self.bias_counts,
                         self._table["code_min"], self._table["code_max"])
        return counts_to_field_t(counts, bits=self.bits)

    def measure(self, truth: TruthEnvironmentHistory) -> FloatArray:
        """Sample the truth body-frame magnetic field at the ODR and process it."""
        indices = sample_history(truth, self.odr_hz)
        return self.process_field(truth.magnetic_field_body_t[indices])
