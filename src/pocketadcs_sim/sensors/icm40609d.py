"""Hardware-faithful TDK ICM-40609-D three-axis gyroscope + accelerometer model.

This is a self-contained new module; the ideal models in ``base.py`` /
``gyroscope.py``, the hardware magnetometer in ``hardware_models.py`` and
the VEMD8080 array are untouched. The firmware worker's ``esp32/`` tree and
``examples/hil_stream.py`` are also out of scope.

The model reproduces the documented signal behavior of the selected flight
hardware -- a TDK ICM-40609-D MEMS IMU providing a 3-axis angular-velocity
gyro and a 3-axis accelerometer -- plus the raw 16-bit ADC response each
axis would deliver. Signal chain per axis:

    gyro:      dps = omega_dps + gyro_bias_dps + gyro_noise_dps
               counts = clip(round(dps * 32768 / FSR_dps), -32768, 32767)
               reading = counts * (FSR_dps / 32768) * (pi / 180)   [rad/s]
    accel:     f_g = (f_m_s2 + accel_bias_m_s2 + accel_noise_m_s2) / g0
               counts = clip(round(f_g * 32768 / FSR_g), -32768, 32767)
               reading = counts * (FSR_g / 32768) * g0            [m/s^2]

The accelerometer output is **specific force** in m/s^2, NOT inertial
acceleration: no gravity term is subtracted or added anywhere in this
model, and nothing here derives specific force from the truth environment
(the truth history carries angular velocity but no specific-force
quantity -- see the note on ``measure_gyro`` below).

Noise and bias model
--------------------

Two separate error sources are modeled, and they are deliberately kept
distinct because they behave completely differently in a filter:

* **White noise, specified as a noise DENSITY.** The datasheet specifies
  noise per unit bandwidth (deg/s per sqrt(Hz)), not per sample, so the
  per-sample sigma is derived as ``sigma = density * sqrt(ENBW)``. This
  makes the output noise rate-dependent in the physically correct way:
  halving ``sample_rate_hz`` halves the bandwidth and therefore reduces
  the per-sample sigma by 1/sqrt(2) at a fixed density. A rate-independent
  sigma (an earlier revision) overstated the noise at low rates and
  understated it at high ones.

* **Bias instability, a slowly-varying (flicker) term plus a
  turn-on-to-turn-on spread.** A real bias is not a constant: it drifts
  within a power cycle and differs from one power cycle to the next. The
  drift is modeled as a first-order Gauss-Markov (AR(1)) flicker process,
  which is the standard engineering stand-in for the 1/f^0.5 corner power
  law and is far cheaper than synthesizing an actual 1/f series. The
  turn-on-to-turn-on term is one draw per model instance -- a constant
  within a run, different across runs -- so an ensemble of runs covers
  the part of the bias budget that no amount of filtering can remove.

Constant values and constant units
----------------------------------

Documented spec values (verified against the primary datasheet; full
citations and the unverified-item list in ``docs/icm40609d.md``):

* TDK ICM-40609-D datasheet, DS-000330 v1.2:
  https://product.tdk.com/system/files/dam/doc/product/sensor/mortion-inertial/imu/data_sheet/ds-000330_icm-40609-d_v1.2.pdf
* Gyro FSR options verified in the datasheet sensitivity table: +/-2000
  dps (listed sensitivity 16.4 LSB/dps) and +/-15.625 dps (listed
  sensitivity 2097.2 LSB/dps).
* Default accelerometer full scale here is +/-4 g.

Everything else -- the 16-bit signed quantization grid
``32768 / FSR`` counts per full-scale unit (a derived engineering choice,
distinct from the listed sensitivities above), round-to-nearest (ties to
even) quantization, the clipping rule, the bias/noise model shape and the
default values -- is an *engineering assumption*, labeled as such below
and in the docs file. The full FSR option lists, the ODR range and the
noise/ARW/bias-stability figures were NOT exhaustively re-verified in this
session and are treated as unverified.

Why ``32768 / FSR`` and not the listed 16.4 LSB/dps: the listed sensitivity
is a *rounded* presentation of the true digital scale, and the two differ by
a systematic 0.098% (~3.5 deg/hour of gyro-integrated error at 1 deg/s)
that no filter can average out. ArduPilot's production ICM-40609-D driver
(``libraries/AP_InertialSensor/AP_InertialSensor_Invensensev3.h``) carries the
"16.4 LSB/DPS" figure in a comment while computing
``radians(1) / (SCALE_RANGE_16BIT / 2000.0)`` = 16.384 in code, i.e. the
16-bit converter is taken as exact and the spec number as rounded. This
model follows that. The accel grid is unaffected: 32768/4 = 8192 LSB/g
matches the datasheet exactly.

Determinism: for an integer ``seed`` and a given input, repeated calls return
identical outputs (the noise stream is re-seeded on every call and depends
only on the input shape). With ``seed=None``, nonzero noise draws fresh
entropy on each call. All ``*_rms_* = 0`` (the default) gives the
deterministic zero-noise mode.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pocketadcs_sim.sensors.base import sample_history
from pocketadcs_sim.truth_environment import TruthEnvironmentHistory

FloatArray = npt.NDArray[np.float64]

# Standard gravity, exact by definition (used for the m/s^2 <-> g bridge).
G_STANDARD_M_S2 = 9.80665
_DEG_TO_RAD = np.pi / 180.0
_RAD_TO_DEG = 180.0 / np.pi

# Signed 16-bit ADC output limits.
INT16_MIN = -32768
INT16_MAX = 32767

# Verified datasheet values (DS-000330 v1.2; see docs/icm40609d.md).
ICM40609D_GYRO_FSR_DEFAULT_DPS = 2000.0  # gyro FSR option, paired with 16.4 LSB/dps
ICM40609D_GYRO_FSR_ALT_DPS = 15.625  # gyro FSR option, paired with 2097.2 LSB/dps
ICM40609D_ACCEL_FSR_DEFAULT_G = 4.0  # default accelerometer FSR


# --- Noise density (DATASHEET) ------------------------------------------------
# The datasheet specifies noise as a density per sqrt(Hz) of bandwidth, not
# as a per-sample RMS, so the model stores the density and derives sigma.
ICM40609D_GYRO_NOISE_DENSITY_DPS_SQRT_HZ = 4.5e-3
"""DATASHEET: gyro noise density, 4.5 mdps/sqrt(Hz)."""

ICM40609D_ACCEL_NOISE_DENSITY_M_S2_SQRT_HZ = 100e-6 * G_STANDARD_M_S2
"""DATASHEET: accel noise density, 100 ug/sqrt(Hz), converted to m/s^2/sqrt(Hz)."""

# --- Bias instability (DATASHEET magnitudes, ENGINEERING time constant) -----
ICM40609D_GYRO_BIAS_INSTABILITY_DPS = 2.0
"""DATASHEET: gyro bias instability, 2 deg/s peak-to-peak band (1 sigma-ish)."""

ICM40609D_ACCEL_BIAS_INSTABILITY_M_S2 = 0.25e-3 * G_STANDARD_M_S2
"""DATASHEET: accel bias instability, 0.25 mg, converted to m/s^2."""

ICM40609D_BIAS_CORRELATION_TIME_S = 60.0
"""ENGINEERING ASSUMPTION: bias correlation time constant, 60 s.

The datasheet gives the *magnitude* of bias instability but not the power
spectral density that would fix a correlation time. 60 s is chosen so the
AR(1) flicker is slow relative to any realistic filter time constant while
still wandering appreciably over a typical run. A true 1/f^0.5 spectrum is
not synthesized; see the module docstring.
"""


def _validate_nonneg(value: float, name: str) -> float:
    val = float(value)
    if not np.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be a finite value >= 0")
    return val


def _optional_nonneg(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return _validate_nonneg(value, name)


def _validate_nonneg_bias(value: npt.ArrayLike, name: str) -> np.ndarray:
    """A non-negative scalar or (3,) bias magnitude, returned as shape (3,)."""
    arr = _validate_bias(value, name)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} values must be >= 0")
    return arr


def _validate_bias(bias: npt.ArrayLike, name: str) -> np.ndarray:
    bias_arr = np.asarray(bias, dtype=float)
    if bias_arr.ndim == 0:
        bias_arr = np.full(3, float(bias_arr), dtype=float)
    if bias_arr.shape != (3,):
        raise ValueError(f"{name} must be a scalar or have shape (3,)")
    if not np.all(np.isfinite(bias_arr)):
        raise ValueError(f"{name} values must be finite")
    return bias_arr


def _validate_vector3_history(values: npt.ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError(f"1-D {name} input must have 3 axes")
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3) or (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} entries must be finite")
    return arr


class ICM40609DModel:
    """Simulated TDK ICM-40609-D 3-axis gyroscope and accelerometer.

    Outputs (always shape (N, 3), body frame):

    * ``process_gyro(omega_body_rad_s)`` -- angular velocity readings in
      rad/s for an explicitly supplied body-frame angular velocity.
    * ``process_specific_force(f_body_m_s2)`` -- specific force readings in
      m/s^2 for an explicitly supplied body-frame specific force.
    * ``measure_gyro(truth)`` -- gyro readings sampled from the truth
      history's ``angular_velocity_body_rad_s`` at ``sample_rate_hz``.

    There is deliberately NO ``measure(truth)`` method: the truth
    environment provides no specific-force quantity, so sampling an
    accelerometer from it would have to invent a physical state. Supply
    specific force explicitly (``process_specific_force``) or use
    ``measure_gyro`` for gyro-only truth sampling.

    Configuration:

    * ``sample_rate_hz`` -- simulation output rate. The device's supported
      ODR range is not modeled as a constraint; the rate is a simulation
      choice (default 100 Hz, an engineering assumption).
    * ``gyro_full_scale_dps`` -- gyro FSR in deg/s (default +/-2000 dps, a
      verified datasheet option).
    * ``accel_full_scale_g`` -- accelerometer FSR in g (default +/-4 g).
    * ``gyro_bias_dps`` -- per-axis constant gyro bias in deg/s (default
      0; an engineering assumption).
    * ``gyro_noise_rms_dps`` -- white Gaussian gyro noise RMS in deg/s
      (default 0 => exact zero-noise mode).
    * ``accel_bias_m_s2`` -- per-axis constant accelerometer bias in m/s^2
      (default 0; an engineering assumption).
    * ``accel_noise_rms_m_s2`` -- white Gaussian accel noise RMS in m/s^2
      (default 0 => exact zero-noise mode). Legacy per-sample override.
    * ``gyro_noise_density_dps_sqrt_hz`` / ``accel_noise_density_m_s2_sqrt_hz``
      -- noise DENSITY, the datasheet's own units (deg/s per sqrt(Hz) and
      m/s^2 per sqrt(Hz)). The per-sample sigma is derived as
      ``density * sqrt(ENBW)``; pass one of these *instead of* the
      ``*_noise_rms_*`` override to get rate-dependent noise. Default None
      (use the ``*_noise_rms_*`` override, which defaults to 0). The
      datasheet values are ``ICM40609D_GYRO_NOISE_DENSITY_DPS_SQRT_HZ``
      (4.5 mdps/sqrt(Hz)) and
      ``ICM40609D_ACCEL_NOISE_DENSITY_M_S2_SQRT_HZ`` (100 ug/sqrt(Hz)).
    * ``noise_bandwidth_hz`` -- effective noise bandwidth in Hz, default
      ``sample_rate_hz / 2``. ENGINEERING ASSUMPTION: the datasheet quotes
      densities over a 0--10 Hz (gyro) / 0--100 Hz (accel) measurement
      bandwidth that is a property of the characterization setup, not of the
      part. Deriving ENBW from the output rate keeps the noise power
      proportional to the rate the caller actually consumes, and
      ``sqrt(fs/2)`` is the single-sided convention that makes sigma halve
      as 1/sqrt(2) when the rate halves. Pass an explicit value to pin it.
    * ``gyro_bias_instability_dps`` / ``accel_bias_instability_m_s2`` --
      1-sigma magnitude of the slowly-varying (AR(1) flicker) bias term,
      scalar or per-axis. Default 0 (no flicker, constant bias, so the
      model stays bit-exact deterministic). Datasheet magnitudes are
      ``ICM40609D_GYRO_BIAS_INSTABILITY_DPS`` and
      ``ICM40609D_ACCEL_BIAS_INSTABILITY_M_S2``.
    * ``gyro_turn_on_bias_std_dps`` / ``accel_turn_on_bias_std_m_s2`` --
      1-sigma turn-on-to-turn-on bias spread, drawn once per model instance
      (constant within a run, different across runs). Default 0.
    * ``bias_correlation_time_s`` -- AR(1) correlation time of the flicker
      term, default ``ICM40609D_BIAS_CORRELATION_TIME_S`` (60 s), an
      ENGINEERING ASSUMPTION: the datasheet gives the bias-instability
      magnitude but not the spectrum that would fix this value.
    * ``seed`` -- noise RNG seed (None => fresh entropy for nonzero noise).

    All defaults are zero, so ``ICM40609DModel()`` is the exact
    deterministic zero-noise model and every datasheet value is opt-in via
    the module constants above. Note the deliberate asymmetry: noise
    density and bias instability are *specified* by the datasheet but are
    not enabled by default, because enabling them would break the exact
    zero mode that the rest of the study harness relies on for its
    noise-free baseline.
    """

    def __init__(
        self,
        sample_rate_hz: float = 100.0,
        gyro_full_scale_dps: float = ICM40609D_GYRO_FSR_DEFAULT_DPS,
        accel_full_scale_g: float = ICM40609D_ACCEL_FSR_DEFAULT_G,
        gyro_bias_dps: npt.ArrayLike = 0.0,
        gyro_noise_rms_dps: float = 0.0,
        accel_bias_m_s2: npt.ArrayLike = 0.0,
        accel_noise_rms_m_s2: float = 0.0,
        gyro_noise_density_dps_sqrt_hz: float | None = None,
        accel_noise_density_m_s2_sqrt_hz: float | None = None,
        noise_bandwidth_hz: float | None = None,
        gyro_bias_instability_dps: npt.ArrayLike = 0.0,
        accel_bias_instability_m_s2: npt.ArrayLike = 0.0,
        gyro_turn_on_bias_std_dps: npt.ArrayLike = 0.0,
        accel_turn_on_bias_std_m_s2: npt.ArrayLike = 0.0,
        bias_correlation_time_s: float = ICM40609D_BIAS_CORRELATION_TIME_S,
        seed: int | None = None,
    ) -> None:
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be a finite positive value")
        if not np.isfinite(gyro_full_scale_dps) or gyro_full_scale_dps <= 0.0:
            raise ValueError("gyro_full_scale_dps must be a finite value > 0")
        if not np.isfinite(accel_full_scale_g) or accel_full_scale_g <= 0.0:
            raise ValueError("accel_full_scale_g must be a finite value > 0")
        self.sample_rate_hz = float(sample_rate_hz)
        self.gyro_full_scale_dps = float(gyro_full_scale_dps)
        self.accel_full_scale_g = float(accel_full_scale_g)
        self.gyro_bias_dps = _validate_bias(gyro_bias_dps, "gyro_bias_dps")
        self.accel_bias_m_s2 = _validate_bias(accel_bias_m_s2, "accel_bias_m_s2")
        self.seed = seed
        # One generator PER SENSOR, advanced across calls, both spawned from
        # one SeedSequence so a seed still fully determines the run.
        #
        # Per-sensor streams (a single shared generator does not work): the
        # gyro and accel paths are driven by separate public methods, and a
        # caller may interleave them in any order. With one shared stream, the
        # values the gyro receives would depend on how many accel samples
        # were drawn in between -- i.e. the call pattern would change the
        # physical noise realization. Independent streams make each sensor's
        # output a function of its own inputs only.
        #
        # Re-seeding inside every processing call (as an earlier revision did)
        # makes chunked streaming replay the identical noise realization in
        # every chunk, which puts spurious spectral lines into any PSD or
        # Allan-deviation estimate taken from the output. A single batch
        # still reproduces exactly for a given seed.
        self._gyro_rng, self._accel_rng = (
            np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(2)
        )

        # Effective noise bandwidth: explicit, else fs/2. See the class
        # docstring for why fs/2 rather than the datasheet's measurement
        # bandwidth.
        if noise_bandwidth_hz is None:
            self._noise_bandwidth_hz = 0.5 * self.sample_rate_hz
        else:
            if not np.isfinite(noise_bandwidth_hz) or noise_bandwidth_hz <= 0.0:
                raise ValueError("noise_bandwidth_hz must be a finite value > 0")
            self._noise_bandwidth_hz = float(noise_bandwidth_hz)

        # Noise is specified as a DENSITY (datasheet units) and the per-sample
        # sigma is derived as density * sqrt(ENBW). An explicit
        # ``*_noise_rms_*`` overrides the density outright, which is what
        # keeps existing call sites (study.py, hardware_adapters) working with
        # unchanged semantics.
        self.gyro_noise_density_dps_sqrt_hz = _optional_nonneg(
            gyro_noise_density_dps_sqrt_hz, "gyro_noise_density_dps_sqrt_hz"
        )
        self.accel_noise_density_m_s2_sqrt_hz = _optional_nonneg(
            accel_noise_density_m_s2_sqrt_hz, "accel_noise_density_m_s2_sqrt_hz"
        )
        rms_scale = np.sqrt(self._noise_bandwidth_hz)
        if self.gyro_noise_density_dps_sqrt_hz is not None and gyro_noise_rms_dps:
            # An explicit per-sample RMS wins outright: the legacy knob is
            # what existing call sites (study.py, hardware_adapters) set, and
            # silently ignoring it in favour of a density would change their
            # noise level without any visible cause.
            self.gyro_noise_density_dps_sqrt_hz = None
        if self.accel_noise_density_m_s2_sqrt_hz is not None and accel_noise_rms_m_s2:
            self.accel_noise_density_m_s2_sqrt_hz = None
        if self.gyro_noise_density_dps_sqrt_hz is not None:
            self.gyro_noise_rms_dps = self.gyro_noise_density_dps_sqrt_hz * rms_scale
        else:
            self.gyro_noise_rms_dps = _validate_nonneg(gyro_noise_rms_dps, "gyro_noise_rms_dps")
        if self.accel_noise_density_m_s2_sqrt_hz is not None:
            self.accel_noise_rms_m_s2 = (
                self.accel_noise_density_m_s2_sqrt_hz * rms_scale
            )
        else:
            self.accel_noise_rms_m_s2 = _validate_nonneg(
                accel_noise_rms_m_s2, "accel_noise_rms_m_s2"
            )

        # Bias instability: a slowly-varying AR(1) flicker term, plus a
        # turn-on-to-turn-on constant drawn once per power cycle.
        if not np.isfinite(bias_correlation_time_s) or bias_correlation_time_s <= 0.0:
            raise ValueError("bias_correlation_time_s must be a finite value > 0")
        self.bias_correlation_time_s = float(bias_correlation_time_s)
        self.gyro_bias_instability_dps = _validate_nonneg_bias(
            gyro_bias_instability_dps, "gyro_bias_instability_dps"
        )
        self.accel_bias_instability_m_s2 = _validate_nonneg_bias(
            accel_bias_instability_m_s2, "accel_bias_instability_m_s2"
        )
        self.gyro_turn_on_bias_std_dps = _validate_nonneg_bias(
            gyro_turn_on_bias_std_dps, "gyro_turn_on_bias_std_dps"
        )
        self.accel_turn_on_bias_std_m_s2 = _validate_nonneg_bias(
            accel_turn_on_bias_std_m_s2, "accel_turn_on_bias_std_m_s2"
        )
        # AR(1) pole from the correlation time. sigma_flicker is the
        # stationary 1-sigma level, so the innovation scale carries the
        # sqrt(1 - a^2) factor that keeps the stationary variance correct.
        dt = 1.0 / self.sample_rate_hz
        self._bias_pole = float(np.exp(-dt / self.bias_correlation_time_s))
        one_minus_a2 = 1.0 - self._bias_pole**2
        self._gyro_flicker_step = np.sqrt(np.maximum(one_minus_a2, 0.0))
        self._accel_flicker_step = np.sqrt(np.maximum(one_minus_a2, 0.0))
        self._gyro_bias_state = np.zeros(3, dtype=float)
        self._accel_bias_state = np.zeros(3, dtype=float)
        # Turn-on-to-turn-on: constant within a run, one draw per instance.
        self._gyro_turn_on_bias_dps = self._gyro_rng.normal(0.0, 1.0, size=3) * (
            self.gyro_turn_on_bias_std_dps
        )
        self._accel_turn_on_bias_m_s2 = self._accel_rng.normal(0.0, 1.0, size=3) * (
            self.accel_turn_on_bias_std_m_s2
        )

    @property
    def noise_bandwidth_hz(self) -> float:
        """Effective noise bandwidth used to convert density to sigma."""
        return self._noise_bandwidth_hz

    def _draw_noise(
        self,
        rng: np.random.Generator,
        sigma: float,
        instability: np.ndarray,
        pole: float,
        step: float,
        state: np.ndarray,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw one call's worth of white noise and flicker bias.

        Returns ``(white, flicker)``, both shape ``(n, 3)``.

        Both sources are drawn from a SINGLE ``(n, 3, 2)`` draw rather than
        two separate draws. That is what makes a chunked call reproduce a
        batched call exactly: numpy's generator fills a request in C order,
        so ``(n, 3, 2)`` yields the identical stream as two consecutive
        ``(n1, 3, 2)`` / ``(n2, 3, 2)`` requests that sum to ``n``. Drawing
        noise and flicker separately would interleave them differently at a
        chunk boundary and break that identity.

        The flicker recursion is then carried forward on ``state``, so the
        walk continues across calls exactly as one long batch would.
        """
        if n == 0:
            return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
        draws = rng.normal(0.0, 1.0, size=(n, 3, 2))
        white = draws[:, :, 0] * sigma
        if not np.any(instability):
            return white, np.zeros((n, 3), dtype=float)
        innovation = draws[:, :, 1] * step * instability
        # y[k] = pole*y[k-1] + innovation[k], seeded from the carried state.
        flicker = np.empty((n, 3), dtype=float)
        prev = state
        for k in range(n):
            prev = pole * prev + innovation[k]
            flicker[k] = prev
        state[...] = prev
        return white, flicker

    @property
    def gyro_counts_per_dps(self) -> float:
        """Quantization grid: 32768 counts per full-scale deg/s."""
        return 32768.0 / self.gyro_full_scale_dps

    @property
    def accel_counts_per_g(self) -> float:
        """Quantization grid: 32768 counts per full-scale g."""
        return 32768.0 / self.accel_full_scale_g

    def process_gyro(self, omega_body_rad_s: npt.ArrayLike) -> FloatArray:
        """Apply bias, noise, 16-bit quantization and clipping to a gyro input.

        Args:
            omega_body_rad_s: body-frame angular velocity, rad/s, shape
                (N, 3) or (3,). Rows with non-finite entries raise
                ``ValueError``.

        Returns:
            (N, 3) array of quantized gyro readings in rad/s, with each
            axis bounded to [-FSR, FSR * 32767/32768] rad/s.
        """
        omega = _validate_vector3_history(omega_body_rad_s, "omega_body_rad_s")
        white, flicker = self._draw_noise(
            self._gyro_rng,
            self.gyro_noise_rms_dps,
            self.gyro_bias_instability_dps,
            self._bias_pole,
            self._gyro_flicker_step,
            self._gyro_bias_state,
            omega.shape[0],
        )
        signal_dps = (
            omega * _RAD_TO_DEG
            + self.gyro_bias_dps
            + self._gyro_turn_on_bias_dps
            + flicker
            + white
        )
        counts = np.rint(signal_dps * self.gyro_counts_per_dps)
        counts = np.clip(counts, INT16_MIN, INT16_MAX)
        return counts * (self.gyro_full_scale_dps / 32768.0) * _DEG_TO_RAD

    def process_specific_force(self, f_body_m_s2: npt.ArrayLike) -> FloatArray:
        """Apply bias, noise, 16-bit quantization and clipping to specific force.

        Args:
            f_body_m_s2: body-frame specific force, m/s^2 (NOT inertial
                acceleration; no gravity term is involved), shape (N, 3)
                or (3,). Rows with non-finite entries raise ``ValueError``.

        Returns:
            (N, 3) array of quantized specific force readings in m/s^2,
            with each axis bounded to [-FSR*g0, FSR*g0 * 32767/32768].
        """
        f = _validate_vector3_history(f_body_m_s2, "f_body_m_s2")
        white, flicker = self._draw_noise(
            self._accel_rng,
            self.accel_noise_rms_m_s2,
            self.accel_bias_instability_m_s2,
            self._bias_pole,
            self._accel_flicker_step,
            self._accel_bias_state,
            f.shape[0],
        )
        signal_g = (
            f + self.accel_bias_m_s2 + self._accel_turn_on_bias_m_s2 + flicker + white
        ) / G_STANDARD_M_S2
        counts = np.rint(signal_g * self.accel_counts_per_g)
        counts = np.clip(counts, INT16_MIN, INT16_MAX)
        return counts * (self.accel_full_scale_g / 32768.0) * G_STANDARD_M_S2

    def measure_gyro(self, truth: TruthEnvironmentHistory) -> FloatArray:
        """Sample the truth body-frame angular velocity at the sensor rate.

        Uses ``truth.angular_velocity_body_rad_s`` (a verified truth
        quantity) via nearest-truth-sample sampling. There is no
        accelerometer equivalent: the truth history carries no specific
        force quantity.
        """
        indices = sample_history(truth, self.sample_rate_hz)
        return self.process_gyro(truth.angular_velocity_body_rad_s[indices])
