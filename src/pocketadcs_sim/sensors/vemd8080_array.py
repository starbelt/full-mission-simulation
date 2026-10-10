"""Hardware-faithful Vishay VEMD8080 six-photodiode body-frame sun array.

This is a self-contained new module; the ideal models in ``base.py`` /
``sun_sensor.py`` and the hardware magnetometer in ``hardware_models.py``
are untouched.

The model reproduces the documented behavior of the selected flight
hardware -- six VEMD8080 silicon PIN photodiodes, one per body-face, each
looking out along an explicit +/-X, +/-Y, +/-Z body normal -- plus the
raw electronics response each channel would deliver. Signal chain per
channel:

    c          = n_channel . sun_body              # cosine of incidence
    response   = c ** q + t * (1 - c)  for c >= 0  (exactly 0 for c < 0)
    signal     = response * shadow_face * scale    # per-face eclipse +
    signal    += dark_current                      # per-channel floor
    signal    += Gaussian noise (optional)         # seeded; 0 => off
    signal    += shot noise (optional)             # std ~ sqrt(signal I)
    reading    = clip(signal, 0, full_scale) onto the ADC grid

The angular response is a power law plus a linear reflection tail, with
the exponent solved so the datasheet's half-sensitivity point (0.5 at
65 deg) is exact by construction:

    q = ln(0.5 - t * (1 - cos 65 deg)) / ln(cos 65 deg)

so the response is 1.0 at normal incidence, exactly 0.5 at the
datasheet's +/-65 deg angle of half sensitivity, a finite tail ``t`` at
90 deg (the real residual the datasheet plots in Fig. 6, from package
and PCB reflections, instead of the hard cliff of a clipped cosine), and
0 beyond the horizon. The law is strictly monotone on [0, 1], so the sun
reconstruction inverts it numerically; both sides derive from the same
constants and cannot drift apart.

The six raw channels are always returned as an (N, 6) array. A 3-axis sun
vector is NEVER reported directly: deriving one requires an explicit
reconstruction/visibility algorithm, which is out of scope for this
module (see ``docs/vemd8080_array.md``).

Documented spec values (verified against the primary datasheet; full
citations in ``docs/vemd8080_array.md``):

* Vishay VEMD8080 datasheet, Document Number 84565, Rev. 1.1,
  05-Sep-2023:
  https://www.vishay.com/docs/84565/vemd8080.pdf

* Peak-sensitivity wavelength 850 nm.
* Spectral bandwidth 350 to 1100 nm.
* Angle of half sensitivity +/-65 deg.
* Reverse dark current 0.2 nA typ / 10 nA max at VR = 10 V, E = 0 (typical
  value used for the dark-current floor default).
* Operating temperature -40 to +85 C (temperature is NOT modeled).

Everything else -- the power-law-plus-tail angular shape, the eclipse
shadow factor, the Gaussian and shot-noise shapes, the cosine-visibility
convention -- is an *engineering assumption*, labeled as such below and
in the docs file. The 850 nm peak responsivity is intentionally NOT
multiplied by the broadband solar constant here (spectral weighting would
be required); instead the per-channel scale is left configurable so it can
be set from a proper calibration or a normalized electronics range.
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

#: Fraction of the peak response a face still reports at exactly 90 deg.
#: Engineering assumption: the datasheet's Fig. 6 plots a finite residual
#: at the horizon (package and PCB reflections) instead of the hard cliff
#: of a clipped cosine; 5% is a representative value. It is a module
#: constant rather than a constructor knob because the sun reconstruction
#: inverts the law, so the two must stay in step.
VEMD8080_REFLECTION_TAIL_FRACTION = 0.05

#: Tolerance of the unit-vector validation in ``process_sun``. The truth
#: environment supplies unit sun directions to machine precision; this
#: exists to catch unnormalized input, not float32 rounding.
_UNIT_VECTOR_RTOL = 1.0e-6

#: Elementary charge (exact in the 2019 SI), for the shot-noise term.
_ELEMENTARY_CHARGE_C = 1.602176634e-19

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

# ---------------------------------------------------------------------------
# Angular response law and its numerical inverse.
# ---------------------------------------------------------------------------
_COS_HALF_SENSITIVITY = float(np.cos(np.radians(VEMD8080_HALF_SENSITIVITY_DEG)))
_TAIL = VEMD8080_REFLECTION_TAIL_FRACTION

#: Exponent of the ``cos(theta) ** q`` direct term, solved so the TOTAL
#: response ``c ** q + t * (1 - c)`` is exactly 0.5 at the datasheet's
#: half-sensitivity angle:
#: ``q = ln(0.5 - t * (1 - cos 65 deg)) / ln(cos 65 deg)``.
#: Exported because the sun reconstruction inverts the law; the two must
#: stay in step (both derive from the constants below).
_RESPONSE_EXPONENT = float(
    np.log(0.5 - _TAIL * (1.0 - _COS_HALF_SENSITIVITY)) / np.log(_COS_HALF_SENSITIVITY)
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

    ``c ** q + t * (1 - c)`` for ``c >= 0`` and exactly 0 behind the face
    (``c < 0``), where ``c`` is the cosine of the angle between the face
    normal and the sun, ``q`` = ``_RESPONSE_EXPONENT`` and ``t`` the
    reflection-tail fraction. Anchors: 1.0 at normal incidence, exactly
    0.5 at the datasheet's +/-65 deg half-sensitivity angle (exact by
    construction of ``q``), a finite tail ``t`` at 90 deg (package/PCB
    reflections, the datasheet's Fig. 6), and 0 beyond the horizon.

    The power-law *shape* and the linear tail *shape* are engineering
    choices interpolating the datasheet anchors; the anchors themselves
    are the datasheet figures. A pure ``cos`` law reads 0.423 at 65 deg,
    ~15% under the datasheet's own half-sensitivity point, and a
    hard-clipped cosine is exactly indistinguishable from eclipse at
    90 deg -- which is what makes grazing-incidence and terminator-sweep
    cases biased.

    The law is strictly increasing on [0, 1], so it is invertible; see
    :func:`invert_angular_response`, which the sun reconstruction uses.

    Args:
        cos_theta: cosine of the angle between each face normal and the sun.

    Returns:
        Response in [0, 1], same shape as the input.
    """
    c = np.clip(cos_theta, 0.0, None)
    response = c ** _RESPONSE_EXPONENT + _TAIL * (1.0 - c)
    return np.where(cos_theta >= 0.0, response, 0.0)


def invert_angular_response(response: FloatArray) -> FloatArray:
    """Numerically invert :func:`_angular_response`.

    Maps a normalized response in [0, 1] back to the cosine of the
    incidence angle in [0, 1]. Values at or below the reflection-tail
    floor map to 0 (at or beyond the horizon), matching the law's 0.05
    response exactly at 90 deg.

    The law is strictly increasing on [0, 1] (its derivative
    ``q * c ** (q - 1) + t`` never vanishes there), so a vectorized
    bisection converges to machine precision; no lookup grid, no
    interpolation error.

    Private API, but imported by the sun reconstruction, which inverts
    the array's law before its least-squares solve; both sides derive
    from the same constants, so they cannot drift apart.
    """
    r = np.clip(np.asarray(response, dtype=float), _TAIL, 1.0)
    lo = np.zeros_like(r)
    hi = np.ones_like(r)
    for _ in range(60):  # 2**-60 < 1e-18: machine precision on [0, 1]
        mid = 0.5 * (lo + hi)
        f = mid ** _RESPONSE_EXPONENT + _TAIL * (1.0 - mid)
        lo = np.where(f <= r, mid, lo)
        hi = np.where(f > r, mid, hi)
    return 0.5 * (lo + hi)


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

    The default is the *nominal* (matched-unit) case: a real build shows
    per-unit gain spread, which the caller models by passing a ``scale``
    array with six distinct values.
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


def _validate_dark_current(dark_current_a: npt.ArrayLike) -> np.ndarray:
    """Per-channel dark-current floor, in amps.

    A scalar applies the same floor to all six channels (the nominal part);
    a (6,) array models the per-unit dark-current spread of a real build
    (the datasheet's 0.2 nA typ / 10 nA max range is 50x, and individual
    diodes in one assembly land anywhere in it).
    """
    dc = np.asarray(dark_current_a, dtype=float)
    dc = np.full(6, float(dc)) if dc.ndim == 0 else dc.reshape(-1)
    if dc.shape != (6,):
        raise ValueError("dark_current_a must be a scalar or have shape (6,)")
    if not np.all(np.isfinite(dc)):
        raise ValueError("dark_current_a values must be finite")
    if np.any(dc < 0.0):
        raise ValueError("dark_current_a values must be >= 0")
    return dc


class VEMD8080ArrayModel:
    """Simulated six-channel VEMD8080 body-frame photodiode sun array.

    Output: raw per-channel readings in **amperes** (default) or in a
    user-defined normalized electronics scale, shape (N, 6), sampled at
    ``sample_rate_hz``. Channels: +X, -X, +Y, -Y, +Z, -Z.

    Signal chain (per channel, per sample):

        c         = n_channel . sun_body            # cosine of incidence
        response  = c ** q + t * (1 - c)  (c >= 0)  # 0 behind the face
        signal    = response * shadow * scale
        signal   += dark_current
        signal   += Gaussian noise                  # seeded (0 => off)
        signal   += shot noise                      # optional, std ~ sqrt(I)
        reading   = clip(signal, 0, full_scale) onto the ADC grid

    ``scale`` is per channel and configurable: it may be a calibrated
    photocurrent scale in A (e.g. from a bench measurement of the
    diode + electronics under a known irradiance) or a normalized
    electronics scale. Passing six distinct values models the per-unit
    gain spread of a real build. This module deliberately does NOT
    compute a scale from the 850 nm peak responsivity times the
    broadband solar constant.

    ``dark_current_a`` is per channel: a scalar (the nominal part) or a
    (6,) array for the per-unit dark-current spread of a real build.

    ``shadow`` is per face: one factor per sample (a global eclipse factor
    on all six faces) or one factor per face, so a real array's per-face
    occlusion -- the body shadows the anti-sun face during a backside
    pass -- can be expressed. ``measure`` derives the per-face factor from
    the truth: the eclipse factor times a body-shadow indicator (a face
    whose normal points away from the sun is occluded by the body).

    ``shot_noise=True`` adds the diode's physical shot noise on top of
    ``noise_rms_a``: per sample, per channel, a zero-mean Gaussian with
    standard deviation ``sqrt(2 * e * B * I)`` where ``I`` is the
    deterministic signal current (photocurrent + dark floor, clipped at
    zero) and ``B`` is the single-pole bandwidth, taken as the Nyquist of
    the sample rate. That is the ``sqrt(I)`` scaling a real diode has.
    ``noise_rms_a`` remains the electronics' readout floor; at the
    array's actual (low) bandwidth the readout floor dominates, which is
    why shot noise is a separate opt-in term.

    Determinism: for a fixed ``seed`` and a given input, repeated calls
    return identical outputs (the noise stream is re-seeded from ``seed``
    on every call and depends only on the input shape). ``noise_rms_a =
    0`` and ``shot_noise = False`` (the defaults) give the deterministic
    zero-noise mode.
    """

    def __init__(
        self,
        sample_rate_hz: float = 1.0,
        scale: npt.ArrayLike | None = None,
        dark_current_a: float | npt.ArrayLike = VEMD8080_DARK_CURRENT_TYP_A,
        noise_rms_a: float = 0.0,
        full_scale_a: float = VEMD8080_DEFAULT_FULL_SCALE_A,
        seed: int | None = None,
        adc_bits: int = DEFAULT_ADC_BITS,
        shot_noise: bool = False,
    ) -> None:
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        scale_arr = _default_scale(scale, full_scale_a)
        dark = _validate_dark_current(dark_current_a)
        if not np.isfinite(noise_rms_a) or noise_rms_a < 0.0:
            raise ValueError("noise_rms_a must be a finite value >= 0")
        if not np.isfinite(full_scale_a) or full_scale_a <= 0.0:
            raise ValueError("full_scale_a must be a finite value > 0")
        if isinstance(adc_bits, bool) or not isinstance(adc_bits, (int, np.integer)):
            raise ValueError("adc_bits must be an integer")
        if not 1 <= adc_bits <= 32:
            raise ValueError("adc_bits must be in [1, 32]")
        if not isinstance(shot_noise, (bool, np.bool_)):
            raise ValueError("shot_noise must be a boolean")
        self.sample_rate_hz = float(sample_rate_hz)
        self.scale = scale_arr
        self.dark_current_a = dark
        self.noise_rms_a = float(noise_rms_a)
        self.full_scale_a = float(full_scale_a)
        self.seed = seed
        self.adc_bits = int(adc_bits)
        self.shot_noise = bool(shot_noise)

    def process_sun(
        self, sun_body: npt.ArrayLike, shadow: npt.ArrayLike | None = None
    ) -> FloatArray:
        """Apply visibility, eclipse, scale, dark floor, noise and clipping.

        Args:
            sun_body: body-frame unit sun direction, shape (N, 3) or (3,).
                Rows must be unit vectors (norm 1 within 1e-6); non-finite
                entries or non-unit rows raise ``ValueError``.
            shadow: per-face eclipse shadow factor in [0, 1] (1 = full
                light, 0 = total eclipse). Either one value per sample,
                shape (N,) (a global factor on all six faces), or one
                value per face, shape (N, 6) (a real array is occluded
                per face -- the body shadows the anti-sun face during a
                backside pass). Defaults to all-ones (full light).

        Returns:
            (N, 6) array of raw channel readings (same units as ``scale``),
            always clipped to [0, full_scale_a] and quantized to the ADC
            grid.
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
        norms = np.linalg.norm(sun, axis=1)
        if np.any(np.abs(norms - 1.0) > _UNIT_VECTOR_RTOL):
            raise ValueError(
                "sun_body rows must be unit vectors "
                f"(norm within {_UNIT_VECTOR_RTOL:g} of 1)"
            )

        n = sun.shape[0]
        if shadow is None:
            shadow_arr = np.ones((n, 6), dtype=float)
        else:
            sh = np.asarray(shadow, dtype=float)
            if sh.ndim == 1:
                if sh.shape != (n,):
                    raise ValueError(
                        "1-D shadow must have shape (N,) matching the sun rows"
                    )
                shadow_arr = sh[:, None]  # one factor, broadcast to all faces
            elif sh.ndim == 2:
                if sh.shape != (n, 6):
                    raise ValueError(
                        "2-D shadow must have shape (N, 6), one factor per face"
                    )
                shadow_arr = sh
            else:
                raise ValueError("shadow must have shape (N,) or (N, 6)")
            if (
                not np.all(np.isfinite(shadow_arr))
                or np.any(shadow_arr < 0.0)
                or np.any(shadow_arr > 1.0)
            ):
                raise ValueError("shadow values must be finite and in [0, 1]")

        # Angular response: the power law plus the reflection tail, with
        # the exponent anchored to the datasheet's 0.5-at-65-deg
        # half-sensitivity point. A bare clip(cos, 0, 1) reads 0.423 at
        # 65 deg (~15% under spec) and exactly zero at 90 deg, where the
        # datasheet plots a real tail.
        cos_theta = sun @ CHANNEL_NORMALS.T
        visibility = _angular_response(cos_theta) * shadow_arr
        photocurrent = visibility * self.scale[None, :] + self.dark_current_a[None, :]

        signal = photocurrent
        if self.noise_rms_a > 0.0 or self.shot_noise:
            rng = np.random.default_rng(self.seed)
        else:
            rng = None
        if self.noise_rms_a > 0.0:
            signal = signal + rng.normal(0.0, self.noise_rms_a, size=signal.shape)
        if self.shot_noise:
            # Shot noise of the diode's own current: a Poisson process
            # with rate proportional to I, so a zero-mean Gaussian
            # approximation with std sqrt(2 e B I) (B = the single-pole
            # bandwidth, taken as the Nyquist of the sample rate). I is
            # the deterministic signal current (photocurrent + dark
            # floor), clipped at zero -- a negative excursion carries no
            # current.
            bandwidth_hz = self.sample_rate_hz / 2.0
            shot_std = np.sqrt(
                2.0
                * _ELEMENTARY_CHARGE_C
                * bandwidth_hz
                * np.maximum(photocurrent, 0.0)
            )
            signal = signal + rng.standard_normal(signal.shape) * shot_std

        clipped = np.clip(signal, 0.0, self.full_scale_a)
        # ADC quantization: round onto the converter's grid, which the other
        # two hardware models in this package also do. This is the last stage
        # of the chain, after noise, so a sub-LSB excursion collapses to the
        # nearest code -- which is what puts a hard floor on how precisely
        # the array can report a weakly-lit face.
        levels = (1 << self.adc_bits) - 1
        return np.rint(clipped / self.full_scale_a * levels) / levels * self.full_scale_a

    def measure(self, truth: TruthEnvironmentHistory) -> FloatArray:
        """Sample the truth body-frame sun direction and per-face shadow.

        Uses ``truth.sun_direction_body`` and the truth
        ``truth.illumination_fraction`` (Basilisk shadow factor: 1 = full
        light, 0 = umbra) as the eclipse factor, and applies the cube's
        own body shadow per face: a face whose normal points away from
        the sun (``normal . sun < 0``) is occluded by the body, so its
        shadow factor is zero even in full light. This is the per-face
        occlusion a real array has -- the body shadows the anti-sun face
        during a backside pass -- which a single global eclipse scalar
        cannot express.
        """
        indices = sample_history(truth, self.sample_rate_hz)
        sun = np.asarray(truth.sun_direction_body, dtype=float)[indices]
        illumination = np.asarray(truth.illumination_fraction[indices], dtype=float)
        body_lit = sun @ CHANNEL_NORMALS.T > 0.0
        per_face_shadow = illumination[:, None] * body_lit
        return self.process_sun(sun, per_face_shadow)
