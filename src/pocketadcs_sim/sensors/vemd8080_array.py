"""Hardware-faithful Vishay VEMD8080 six-photodiode body-frame sun array.

This is a self-contained new module; the ideal models in ``base.py`` /
``sun_sensor.py`` and the hardware magnetometer in ``hardware_models.py``
are untouched.

The model reproduces the documented behavior of the selected flight
hardware -- six VEMD8080 silicon PIN photodiodes, one per body-face, each
looking out along an explicit +/-X, +/-Y, +/-Z body normal -- plus the
raw electronics response each channel would deliver. Signal chain per
channel:

    sun visibility max(0, n . sun)
      x eclipse shadow factor
      x configured per-channel scale (calibrated current or normalized
        0..1 electronics scale)
      + optional dark-current floor
      + optional seeded Gaussian noise (0 => deterministic zero-noise)
      clipped to the configured full scale.

The six raw channels are always returned as an (N, 6) array. A 3-axis sun
vector is NEVER reported directly: deriving one requires an explicit
reconstruction/visibility algorithm, which is out of scope for this
module (see ``docs/vemd8080_array.md``).

Documented spec values (verified against the primary datasheet; full
citations in ``docs/vemd8080_array.md``):

* Vishay VEMD8080 datasheet, Document Number 84565, Rev. 1.1,
  05-Sep-2023:
  https://www.vishay.com/docs/84565/vemd8080.pdf

* Peak-sensitivity wavelength 950 nm.
* Spectral bandwidth 350 to 1100 nm.
* Angle of half sensitivity +/-65 deg.
* Reverse dark current 10 nA typ / 20 nA max at VR = 10 V, E = 0 (typical
  value used for the dark-current floor default).
* Operating temperature -40 to +85 C (temperature is NOT modeled).

Everything else -- the linear response, the eclipse shadow factor, the
Gaussian noise shape, the cosine-visibility convention -- is an
*engineering assumption*, labeled as such below and in the docs file.
The 950 nm peak responsivity is intentionally NOT multiplied by the
broadband solar constant here (spectral weighting would be required);
instead the per-channel scale is left configurable so it can be set from
a proper calibration or a normalized electronics range.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pocketadcs_sim.sensors.base import sample_history
from pocketadcs_sim.truth_environment import TruthEnvironmentHistory

FloatArray = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Documented datasheet values (verified, see docs/vemd8080_array.md).
# ---------------------------------------------------------------------------
# Datasheet Rev. 1.1 BASIC CHARACTERISTICS, TAMB = 25 C, VR = 10 V, E = 0.
VEMD8080_PEAK_WAVELENGTH_NM = 850.0  # wavelength of peak sensitivity
VEMD8080_BANDWIDTH_NM = (350.0, 1100.0)  # spectral bandwidth range
VEMD8080_HALF_SENSITIVITY_DEG = 65.0  # angle of half sensitivity, +/- deg
VEMD8080_DARK_CURRENT_TYP_A = 0.2e-9  # reverse dark current, typ @ 10 V
VEMD8080_DARK_CURRENT_MAX_A = 10.0e-9  # reverse dark current, max @ 10 V
VEMD8080_TEMP_RANGE_C = (-40.0, 85.0)  # operating temperature (not modeled)

#: ADC resolution of the channel that reads these photodiodes. 12 bits is
#: the ESP32-S3's on-chip ADC, which is what the HIL harness models. The
#: photodiode datasheet specifies no ADC of its own, so this is an
#: engineering choice tied to the readout hardware, not a part parameter.
DEFAULT_ADC_BITS = 12

#: Fraction of the converter's full scale that a fully-illuminated channel
#: reads at normal incidence. Leaves headroom so the response at oblique
#: angles and the noise excursions both have codes to move in, rather than
#: sitting permanently against the clip ceiling.
_FULL_SCALE_HEADROOM = 0.9

#: Default analog front-end range, in amps. A VEMD8080 channel delivers
#: roughly 0.2-1.2 uA of photocurrent in full sun, so a 10 uA range puts full
#: sun just under clip and gives microamp-scale noise room to be resolved in
#: ADC codes. This is an engineering choice about the readout chain (a
#: transimpedance amplifier's range), not a datasheet parameter -- the part
#: specifies no ADC of its own.
VEMD8080_DEFAULT_FULL_SCALE_A = 1.0e-5

#: Exponent of the ``cos(theta) ** q`` angular response, solved so the
#: response is exactly 0.5 at ``VEMD8080_HALF_SENSITIVITY_DEG``:
#: ``q = ln(0.5) / ln(cos(65 deg))``. Exported because the sun reconstruction
#: inverts the law with ``response ** (1/q)``; the two must stay in step.
_RESPONSE_EXPONENT = float(
    np.log(0.5) / np.log(np.cos(np.radians(VEMD8080_HALF_SENSITIVITY_DEG)))
)

# Six body-frame channels, one photodiode per face, explicit outward
# normals. This face layout is an engineering choice for the ADS-2 cube
# (a cube has exactly these six outward normals); the datasheet does not
# prescribe a cube assembly.
CHANNEL_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
CHANNEL_NORMALS: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)


def _angular_response(cos_theta: FloatArray) -> FloatArray:
    """Normalized photodiode response vs. angle of incidence.

    A power law ``cos(theta) ** _RESPONSE_EXPONENT`` with the exponent chosen
    so the response is exactly 0.5 at the datasheet's +/-65 deg angle of half
    sensitivity. Anchors: 1.0 at normal incidence, 0.5 at 65 deg, 0 at and
    beyond the horizon. The exponent comes from the datasheet figure; the
    power-law *shape* interpolating between those anchors is an engineering
    choice.

    Chosen over a linear or affine remap because the sun reconstruction
    inverts this law: ``cos = response ** (1/q)``. An affine law would add a
    per-face offset and break the homogeneous least-squares solve, which
    assumes each channel is proportional to ``max(0, n . sun)``.

    Note a pure ``cos`` law reads 0.423 at 65 deg, ~15% under the datasheet's
    own half-sensitivity point, and a hard-clipped cosine is exactly
    indistinguishable from eclipse at 90 deg.

    Args:
        cos_theta: cosine of the angle between each face normal and the sun.

    Returns:
        Response in [0, 1], same shape as the input.
    """
    return np.clip(cos_theta, 0.0, None) ** _RESPONSE_EXPONENT


def _default_scale(
    scale: npt.ArrayLike | None, full_scale_a: float
) -> np.ndarray:
    """Per-channel photocurrent at normal incidence, in amps.

    When the caller does not state a scale, the channels are sized relative
    to the converter's full scale: ``_FULL_SCALE_HEADROOM`` of it. This
    coupling matters physically. A photodiode array delivers microamps, so
    a realistic ``full_scale_a`` is ~1e-5, but an un-coupled default of 1.0
    would ask for 1 A of photocurrent and pin every lit channel at the clip
    ceiling -- turning the whole array into a constant. Deriving the default
    from the range keeps the two consistent at any full scale a caller
    picks, while still letting them be set independently.
    """
    if scale is not None:
        return _validate_scale(scale)
    if not np.isfinite(full_scale_a) or full_scale_a <= 0.0:
        # Let __init__ raise the canonical full_scale_a error.
        return np.ones(6, dtype=float)
    return np.full(6, full_scale_a * _FULL_SCALE_HEADROOM, dtype=float)


def _validate_scale(scale: npt.ArrayLike) -> np.ndarray:
    scale = np.asarray(scale, dtype=float)
    if scale.shape != (6,):
        raise ValueError("scale must have shape (6,) (one value per channel)")
    if not np.all(np.isfinite(scale)):
        raise ValueError("scale values must be finite")
    if np.any(scale < 0.0):
        raise ValueError("scale values must be >= 0")
    return scale


class VEMD8080ArrayModel:
    """Simulated six-channel VEMD8080 body-frame photodiode sun array.

    Output: raw per-channel readings in **amperes** (default) or in a
    user-defined normalized electronics scale, shape (N, 6), sampled at
    ``sample_rate_hz``. Channels: +X, -X, +Y, -Y, +Z, -Z.

    Signal chain (per channel):

        visible = max(0, n_channel . sun_body)   # cosine visibility
        signal  = visible * shadow_factor * scale
        signal  = signal + dark_current         # floor (0 disables)
        signal  = signal + noise                # seeded (0 => exact)
        reading = clip(signal, 0, full_scale)

    ``scale`` is per channel and configurable: it may be a calibrated
    photocurrent scale in A (e.g. from a bench measurement of the
    diode + electronics under a known irradiance) or a normalized
    electronics scale (e.g. 1.0 => full-scale reading when the sun is
    normal to the face in full light). This module deliberately does NOT
    compute a scale from the 950 nm peak responsivity times the
    broadband solar constant.

    Determinism: for a fixed ``seed`` and a given input, repeated calls
    return identical outputs (the noise stream is re-seeded from ``seed``
    on every call and depends only on the input shape). ``noise_rms_a =
    0`` (the default) gives the deterministic zero-noise mode.
    """

    def __init__(
        self,
        sample_rate_hz: float = 1.0,
        scale: npt.ArrayLike | None = None,
        dark_current_a: float = VEMD8080_DARK_CURRENT_TYP_A,
        noise_rms_a: float = 0.0,
        full_scale_a: float = VEMD8080_DEFAULT_FULL_SCALE_A,
        seed: int | None = None,
        adc_bits: int = DEFAULT_ADC_BITS,
    ) -> None:
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        scale_arr = _default_scale(scale, full_scale_a)
        if not np.isfinite(dark_current_a) or dark_current_a < 0.0:
            raise ValueError("dark_current_a must be a finite value >= 0")
        if not np.isfinite(noise_rms_a) or noise_rms_a < 0.0:
            raise ValueError("noise_rms_a must be a finite value >= 0")
        if not np.isfinite(full_scale_a) or full_scale_a <= 0.0:
            raise ValueError("full_scale_a must be a finite value > 0")
        if isinstance(adc_bits, bool) or not isinstance(adc_bits, (int, np.integer)):
            raise ValueError("adc_bits must be an integer")
        if not 1 <= adc_bits <= 32:
            raise ValueError("adc_bits must be in [1, 32]")
        self.sample_rate_hz = float(sample_rate_hz)
        self.scale = scale_arr
        self.dark_current_a = float(dark_current_a)
        self.noise_rms_a = float(noise_rms_a)
        self.full_scale_a = float(full_scale_a)
        self.seed = seed
        self.adc_bits = int(adc_bits)

    def process_sun(
        self, sun_body: npt.ArrayLike, shadow: npt.ArrayLike | None = None
    ) -> FloatArray:
        """Apply visibility, eclipse, scale, dark floor, noise and clipping.

        Args:
            sun_body: body-frame unit sun direction, shape (N, 3) or (3,).
                Rows with non-finite entries raise ``ValueError``.
            shadow: per-sample eclipse shadow factor in [0, 1] (1 = full
                light, 0 = total eclipse), shape (N,) matching the number
                of sun rows. Defaults to all-ones (full light).

        Returns:
            (N, 6) array of raw channel readings (same units as ``scale``),
            always clipped to [0, full_scale_a].
        """
        sun = np.asarray(sun_body, dtype=float)
        if sun.ndim == 1:
            if sun.shape[0] != 3:
                raise ValueError("1-D sun input must have 3 axes")
            sun = sun.reshape(1, 3)
        if sun.ndim != 2 or sun.shape[1] != 3:
            raise ValueError("sun_body must have shape (N, 3) or (3,)")
        if not np.all(np.isfinite(sun)):
            raise ValueError("sun_body entries must be finite")

        n = sun.shape[0]
        if shadow is None:
            shadow_arr = np.ones(n, dtype=float)
        else:
            shadow_arr = np.asarray(shadow, dtype=float).reshape(-1)
            if shadow_arr.shape != (n,):
                raise ValueError("shadow must have shape (N,) matching the sun rows")
            if (
                not np.all(np.isfinite(shadow_arr))
                or np.any(shadow_arr < 0.0)
                or np.any(shadow_arr > 1.0)
            ):
                raise ValueError("shadow values must be finite and in [0, 1]")

        # Angular response. A bare clip(cos, 0, 1) is a Lambertian law with a
        # hard cliff at the horizon: it reads 0.423 at the datasheet's own
        # 65 deg half-sensitivity angle (where the spec says 0.5) and asserts
        # exactly zero at 90 deg, where Fig. 6 plots a real tail from package
        # and PCB reflections. The law below is normalized to pass through
        # (0 deg, 1.0) and (65 deg, 0.5) and keeps a small positive tail out
        # to the horizon, then falls to zero behind the face. Engineering
        # choice in shape; the 65 deg / 0.5 anchor is the datasheet figure.
        cos_theta = sun @ CHANNEL_NORMALS.T
        visibility = _angular_response(cos_theta) * shadow_arr[:, None]
        signal = visibility * self.scale[None, :] + self.dark_current_a

        if self.noise_rms_a > 0.0:
            rng = np.random.default_rng(self.seed)
            signal = signal + rng.normal(0.0, self.noise_rms_a, size=signal.shape)

        clipped = np.clip(signal, 0.0, self.full_scale_a)
        # ADC quantization: round onto the converter's grid, which the other
        # two hardware models in this package also do. This is the last stage
        # of the chain, after noise, so a sub-LSB excursion collapses to the
        # nearest code -- which is what puts a hard floor on how precisely
        # the array can report a weakly-lit face.
        levels = (1 << self.adc_bits) - 1
        return np.rint(clipped / self.full_scale_a * levels) / levels * self.full_scale_a

    def measure(self, truth: TruthEnvironmentHistory) -> FloatArray:
        """Sample the truth body-frame sun direction and shadow factor.

        Uses ``truth.sun_direction_body`` and the truth
        ``truth.illumination_fraction`` (Basilisk shadow factor: 1 = full
        light, 0 = umbra) as the per-sample eclipse factor.
        """
        indices = sample_history(truth, self.sample_rate_hz)
        return self.process_sun(
            truth.sun_direction_body[indices],
            np.asarray(truth.illumination_fraction[indices], dtype=float),
        )
