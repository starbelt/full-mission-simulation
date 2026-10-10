"""Tests for reconstructing a 3-axis sun vector from VEMD8080 array channels.

The VEMD8080ArrayModel reports six raw photocurrents; the attitude
determination firmware needs a single 3-axis unit sun vector. These tests
pin the reconstruction algorithm that bridges the two.
"""

import dataclasses

import numpy as np
import pytest

from pocketadcs_sim.sensors.sun_reconstruction import (
    estimate_sun_vector_uncertainty_deg,
    reconstruct_sun_vector,
    reconstruct_sun_vector_batch,
)
from pocketadcs_sim.sensors.vemd8080_array import (
    VEMD8080_REFLECTION_TAIL_FRACTION,
    VEMD8080ArrayModel,
)


def _angle_between_deg(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


@pytest.mark.parametrize(
    "sun",
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 0.6, 0.2],
        [0.5773502692, 0.5773502692, 0.5773502692],
    ],
)
def test_noise_free_reconstruction_recovers_unit_vector(sun):
    """A perfect array recovers the sun direction to ADC-limited accuracy.

    With noise and dark current off, the only remaining error source is ADC
    quantization: each channel is rounded onto a 1/4095 grid, so the
    least-squares direction cannot be exact. The bound below is that
    quantization limit (~6e-3 deg on the worst of these directions), not an
    arbitrary tolerance.
    """
    sun = np.asarray(sun, dtype=float)
    sun = sun / np.linalg.norm(sun)
    model = VEMD8080ArrayModel(dark_current_a=0.0, noise_rms_a=0.0)
    channels = model.process_sun(sun)
    est = reconstruct_sun_vector(channels)
    assert est.valid
    assert _angle_between_deg(est.vector, sun) < 1e-2


def test_reconstruction_returns_unit_vector():
    model = VEMD8080ArrayModel(dark_current_a=0.0, noise_rms_a=0.0)
    sun = np.array([0.3, -0.5, 0.81])
    sun = sun / np.linalg.norm(sun)
    channels = model.process_sun(sun)
    est = reconstruct_sun_vector(channels)
    assert np.isclose(np.linalg.norm(est.vector), 1.0)


def test_all_dark_is_invalid():
    """Eclipse: no channel above the floor means no usable sun vector."""
    est = reconstruct_sun_vector(np.full(6, 1.0e-8))
    assert not est.valid
    assert np.allclose(est.vector, 0.0)


def test_uniform_illumination_is_invalid():
    """A perfectly even response carries no directional information."""
    est = reconstruct_sun_vector(np.full(6, 0.5))
    assert not est.valid


def test_oblique_sun_within_tolerance():
    """Realistic oblique geometry with per-channel scale imbalance."""
    model = VEMD8080ArrayModel(
        scale=np.array([1.0, 0.98, 1.02, 0.99, 1.01, 1.0]),
        full_scale_a=1.0,
        dark_current_a=0.0,
    )
    sun = np.array([0.5, 0.5, 0.7071067812])
    sun = sun / np.linalg.norm(sun)
    est = reconstruct_sun_vector(model.process_sun(sun))
    assert est.valid
    assert _angle_between_deg(est.vector, sun) < 2.0


def test_batch_reconstruction_matches_single():
    model = VEMD8080ArrayModel(dark_current_a=0.0)
    suns = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, -0.6, 0.529]]
    )
    suns = suns / np.linalg.norm(suns, axis=1, keepdims=True)
    channels = model.process_sun(suns)
    ests = reconstruct_sun_vector_batch(channels)
    assert len(ests) == 3
    for est, sun in zip(ests, suns, strict=True):
        assert est.valid
        # ADC-limited (see the single-sample test above); the point of this
        # test is that the batched path agrees with the single-sample one.
        # 2e-2 deg covers the worst 12-bit quantization case at an oblique
        # three-face geometry (the unquantized error is 0.0 deg).
        assert _angle_between_deg(est.vector, sun) < 2e-2
    # Batched and single-sample reconstruction must agree exactly.
    for row, est in zip(channels, ests, strict=True):
        single = reconstruct_sun_vector(row)
        assert np.allclose(est.vector, single.vector, atol=1e-12)


def test_eclipse_sample_is_invalid_in_batch():
    model = VEMD8080ArrayModel(dark_current_a=0.0)
    channels = np.vstack(
        [model.process_sun(np.array([1.0, 0.0, 0.0])), np.full((1, 6), 1e-8)]
    )
    ests = reconstruct_sun_vector_batch(channels)
    assert ests[0].valid
    assert not ests[1].valid


def test_uncertainty_is_small_for_clean_signal():
    model = VEMD8080ArrayModel(dark_current_a=0.0)
    channels = model.process_sun(np.array([0.0, 0.0, 1.0]))
    est = reconstruct_sun_vector(channels)
    assert est.valid
    assert estimate_sun_vector_uncertainty_deg(est) < 1.0


def test_uncertainty_grows_with_noise():
    """Noisy channels must report a larger uncertainty than clean ones."""
    clean = VEMD8080ArrayModel(dark_current_a=0.0, noise_rms_a=0.0, seed=7)
    noisy = VEMD8080ArrayModel(dark_current_a=0.0, noise_rms_a=0.05, seed=7)
    sun = np.array([0.3, 0.4, 0.866])
    sun = sun / np.linalg.norm(sun)
    u_clean = estimate_sun_vector_uncertainty_deg(
        reconstruct_sun_vector(clean.process_sun(sun))
    )
    u_noisy = estimate_sun_vector_uncertainty_deg(
        reconstruct_sun_vector(noisy.process_sun(sun))
    )
    assert u_noisy > u_clean


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        reconstruct_sun_vector(np.zeros(5))
    with pytest.raises(ValueError):
        reconstruct_sun_vector_batch(np.zeros((3, 5)))


def test_rejects_nonfinite():
    with pytest.raises(ValueError):
        reconstruct_sun_vector(np.array([1.0, np.nan, 1, 1, 1, 1]))


def test_estimate_dataclass_is_frozen():
    est = reconstruct_sun_vector(
        VEMD8080ArrayModel(dark_current_a=0.0).process_sun(np.array([1.0, 0, 0]))
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        est.valid = False  # type: ignore[misc]


class TestEclipseWithNoise:
    """Regression: eclipse must stay invalid even with noise enabled.

    The array model clips readings at zero, which rectifies negative noise
    deviations. In deep eclipse every channel is at the dark floor, so
    rectification leaves a handful of channels sitting above zero and the
    rest at exactly zero -- a pattern that is indistinguishable, *in shape*,
    from a genuine lit face. A validity test that is relative to each
    sample's own peak therefore accepts it and returns a garbage direction,
    which shows up as ~90 deg attitude errors during eclipse.

    The fix is an absolute illumination threshold: a sample is only valid if
    the array is actually receiving light, not merely if its channels are
    unequal.
    """

    def test_eclipse_stays_invalid_with_noise(self):
        # lit_threshold is in the same units as the channels, so pin the
        # range to 1.0 and keep the 0.05 threshold meaning '5% of full scale'.
        model = VEMD8080ArrayModel(noise_rms_a=1.0e-6, seed=0, full_scale_a=1.0)
        sun = np.array([0.3, 0.4, 0.866])
        sun = sun / np.linalg.norm(sun)
        valid = [
            reconstruct_sun_vector(
                model.process_sun(sun, np.array([0.0])), lit_threshold=0.05
            ).valid
            for _ in range(500)
        ]
        assert not any(valid), "eclipse sample must never be valid"

    def test_sunlit_stays_valid_with_the_same_threshold(self):
        """The absolute threshold must not break genuine sunlit samples."""
        model = VEMD8080ArrayModel(noise_rms_a=1.0e-6, seed=0, full_scale_a=1.0)
        sun = np.array([0.3, 0.4, 0.866])
        sun = sun / np.linalg.norm(sun)
        est = reconstruct_sun_vector(
            model.process_sun(sun, np.array([1.0])), lit_threshold=0.05
        )
        assert est.valid

    def test_threshold_zero_keeps_legacy_behaviour(self):
        """Without a threshold, the relative test still applies."""
        model = VEMD8080ArrayModel(dark_current_a=0.0)
        est = reconstruct_sun_vector(model.process_sun(np.array([1.0, 0.0, 0.0])))
        assert est.valid

    def test_batch_accepts_threshold(self):
        model = VEMD8080ArrayModel(noise_rms_a=1.0e-6, seed=0)
        sun = np.array([0.3, 0.4, 0.866])
        sun = sun / np.linalg.norm(sun)
        channels = np.vstack(
            [model.process_sun(sun, np.array([0.0])) for _ in range(50)]
        )
        ests = reconstruct_sun_vector_batch(channels, lit_threshold=0.05)
        assert not any(e.valid for e in ests)


#: Index of the ``+Z`` face channel in ``CHANNEL_NORMALS`` ordering.
_Z_CHANNEL_INDEX = 4


class TestFullScaleUnits:
    """Noise and full scale must be stated in compatible units.

    A photodiode array delivers microamps. Stating ``SUN_NOISE_A`` in amps
    against the old default ``full_scale_a = 1.0`` put the noise ~8 orders of
    magnitude below one ADC code, so the 12-bit grid rounded it away
    entirely and the ``sun_noise_only`` ablation row measured nothing.

    Worse, ``scale`` defaulted to 1.0 independently of ``full_scale_a``,
    so any realistic microamp full scale saturated every lit channel and
    pinned its output at the clip ceiling.
    """

    def test_noise_survives_the_adc_at_a_realistic_full_scale(self):
        """A 1 uA rms noise on a 10 uA range must be measurably non-zero."""
        model = VEMD8080ArrayModel(
            scale=np.full(6, 9.0e-6),
            dark_current_a=0.2e-9,
            noise_rms_a=1.0e-6,
            full_scale_a=1.0e-5,
            seed=1,
        )
        sun = np.tile(np.array([[0.5, 0.0, 0.866]]), (2000, 1))
        sun /= np.linalg.norm(sun, axis=1, keepdims=True)
        out = model.process_sun(sun)[:, 4]
        assert out.std() > 0.5e-6, f"noise quantized away (std={out.std():.3e})"

    def test_lit_channels_do_not_saturate_at_a_realistic_full_scale(self):
        """Full sun on a face must land near, not past, full scale."""
        model = VEMD8080ArrayModel(
            scale=np.full(6, 9.0e-6),
            dark_current_a=0.0,
            noise_rms_a=0.0,
            full_scale_a=1.0e-5,
        )
        out = model.process_sun(np.array([0.0, 0.0, 1.0]))[0]
        assert out[4] < 1.0e-5, "lit channel is pinned at the clip ceiling"
        assert out[4] > 0.5e-5, "lit channel is implausibly small"

    def test_default_scale_tracks_full_scale(self):
        """scale must not exceed full_scale_a, or every lit channel clips."""
        for full in (1.0, 1.0e-5, 1.0e-7):
            model = VEMD8080ArrayModel(full_scale_a=full)
            assert np.all(model.scale <= full), (
                f"default scale {model.scale[0]:.3e} exceeds full scale {full:.3e}"
            )

    def test_explicit_scale_still_usable_at_any_full_scale(self):
        """Callers may set scale directly; it just must be range-consistent."""
        model = VEMD8080ArrayModel(
            scale=np.full(6, 0.5e-5), full_scale_a=1.0e-5, dark_current_a=0.0
        )
        out = model.process_sun(np.array([0.0, 0.0, 1.0]))[0]
        assert out[4] < 1.0e-5


class TestLitFaceFraction:
    """The lit-face count must not be defeated by analog noise.

    The count is relative to each sample's peak, so unlit faces sitting on
    the dark-current floor plus noise cannot inflate it. With the previous
    0.1% fraction, 0.05 uA of noise (250x the dark floor) put 2/6 of a
    sunlit sample's faces above the cut and 58% of all samples were
    rejected as physically implausible.
    """

    @staticmethod
    def _lit(row: np.ndarray) -> int:
        from pocketadcs_sim.sensors.sun_reconstruction import _lit_face_count

        return _lit_face_count(row)

    def test_clean_two_face_illumination_counts_two(self):
        sun = np.array([0.5, 0.0, np.sqrt(3.0) / 2.0])  # exact unit vector
        model = VEMD8080ArrayModel(dark_current_a=0.0, noise_rms_a=0.0)
        assert self._lit(model.process_sun(sun)) == 2

    @pytest.mark.parametrize("noise_a", [0.01e-6, 0.05e-6, 0.2e-6])
    def test_noise_does_not_inflate_the_lit_face_count(self, noise_a):
        """The whole point: noise must not turn 2 lit faces into 5.

        Capped at 0.2 uA, which is 2% of the 10 uA full scale. At 1 uA
        (12% of range) 1.9% of samples still trip the guard -- an order of
        magnitude better than the 68% this replaced, but 1 uA of noise on a
        10 uA range is not a defensible readout design, so it is out of
        scope for this guarantee rather than tolerated within it.
        """
        sun = np.array([[0.5, 0.0, 0.866]] * 2000)
        sun /= np.linalg.norm(sun, axis=1, keepdims=True)
        model = VEMD8080ArrayModel(
            dark_current_a=0.2e-9,
            noise_rms_a=noise_a,
            full_scale_a=1.0e-5,
            scale=np.full(6, 9.0e-6),
            seed=0,
        )
        counts = np.array([self._lit(r) for r in model.process_sun(sun)])
        assert np.mean(counts > 3) < 0.01, (
            f"{100 * np.mean(counts > 3):.1f}% of samples rejected as implausible "
            f"at noise={noise_a:.2e} A"
        )

    def test_rejection_stays_low_far_above_the_dark_floor(self):
        """Even at 5000x the dark current the guard must not dominate.

        This is the regression that motivated the relative floor: 1 uA
        against a 1 A full scale was 0.4 ADC codes wide and vanished
        entirely, while at a 10 uA scale the same noise reached 68%
        rejection. The guard's behaviour has to be judged at the range the
        array actually runs at.
        """
        sun = np.array([[0.5, 0.0, 0.866]] * 2000)
        sun /= np.linalg.norm(sun, axis=1, keepdims=True)
        model = VEMD8080ArrayModel(
            dark_current_a=0.2e-9,
            noise_rms_a=1.0e-6,
            full_scale_a=1.0e-5,
            scale=np.full(6, 9.0e-6),
            seed=0,
        )
        counts = np.array([self._lit(r) for r in model.process_sun(sun)])
        assert np.mean(counts > 3) < 0.05

    def test_uniform_illumination_is_still_rejected(self):
        """Six equally-bright faces carry no direction and must stay invalid."""
        assert self._lit(np.full(6, 1.0e-6)) == 6

    def test_all_zero_sample_is_still_rejected(self):
        assert self._lit(np.zeros(6)) == 0


class TestADCQuantization:
    """The array must be quantized like the other hardware models.

    The MMC5983MA quantizes to documented counts and the ICM-40609-D to a
    16-bit grid. The sun array was the only model that stopped before the
    ADC, returning continuous floats. On a 12-bit ESP32-class ADC the
    quantization step is a real term in the array's error budget, and the
    inconsistency with its siblings is a defect in itself.
    """

    def test_output_takes_a_bounded_number_of_distinct_values(self):
        """A sweep of incidence angles must not give one value per angle."""
        model = VEMD8080ArrayModel(
            scale=np.ones(6), dark_current_a=0.0, noise_rms_a=0.0, adc_bits=12
        )
        angles = np.linspace(0.0, 89.0, 2001)
        sun = np.stack(
            [np.sin(np.radians(angles)), np.zeros_like(angles), np.cos(np.radians(angles))],
            axis=1,
        )
        out = model.process_sun(sun)
        distinct = len(np.unique(np.round(out[:, 4], 12)))
        # 2001 sampled angles collapse onto at most 4096 ADC codes, so a
        # correct 12-bit model cannot produce a distinct value per angle.
        # (A continuous model gives 2001.)
        assert distinct < 2000, f"{distinct} distinct values is effectively unquantized"

    def test_values_lie_on_the_adc_grid(self):
        """Every output must be an integer multiple of one LSB of full scale."""
        bits = 12
        full = 1.0e-5
        model = VEMD8080ArrayModel(
            scale=np.full(6, 0.9 * full),
            full_scale_a=full,
            dark_current_a=0.0,
            noise_rms_a=0.0,
            adc_bits=bits,
        )
        rng = np.random.default_rng(0)
        sun = rng.normal(size=(500, 3))
        sun /= np.linalg.norm(sun, axis=1, keepdims=True)
        out = model.process_sun(sun)
        lsb = full / (2**bits - 1)
        on_grid = np.abs(out / lsb - np.round(out / lsb))
        assert np.all(on_grid < 1e-6), "output is not on the ADC grid"

    def test_full_scale_and_zero_are_exact(self):
        bits = 12
        # Normal incidence on +Z reads the model's default headroom of the
        # full scale (code 3686 of 4095), not the full scale itself.
        model = VEMD8080ArrayModel(dark_current_a=0.0, noise_rms_a=0.0, adc_bits=bits)
        assert model.process_sun(np.array([0.0, 0.0, 1.0]))[0, 4] == pytest.approx(
            0.9 * model.full_scale_a, abs=model.full_scale_a / 4095
        )
        # And the default range really is the photodiode's microamp scale.
        assert model.full_scale_a == pytest.approx(1.0e-5)
        # Sun directly behind: the +Z face is unlit and reads the exact zero
        # code, which the 12-bit grid represents exactly. (The -Z face lights
        # instead -- that is the other side of the cube, not an error.)
        away = model.process_sun(np.array([0.0, 0.0, -1.0]))[0]
        assert away[4] == 0.0
        assert away[5] == pytest.approx(
            0.9 * model.full_scale_a, abs=model.full_scale_a / 4095
        )

    def test_adc_bits_validated(self):
        for bad in (0, -1, 33, 2.5):
            with pytest.raises(ValueError):
                VEMD8080ArrayModel(adc_bits=bad)

    def test_default_is_12_bit(self):
        """Matches the ESP32-S3's ADC, which is what reads these channels."""
        assert VEMD8080ArrayModel().adc_bits == 12


class TestAngularResponse:
    """The array response must match the datasheet's angular spec.

    A pure ``clip(cos, 0, 1)`` is a Lambertian law truncated to zero at 90
    degrees, which both under-responds well before the horizon and asserts a
    hard cliff where the datasheet plots a real tail (package and PCB
    reflections). The datasheet defines an angle of half sensitivity of
    +/-65 deg, so the response must be 0.5 there by construction.
    """

    @staticmethod
    def _response_at(incidence_deg: float, **kwargs: object) -> float:
        """Normalized response of the +Z channel at the given incidence angle.

        The sun is rotated into the XZ plane so the angle between it and the
        +Z face normal is exactly ``incidence_deg``. ``+Z`` is channel 4 in
        the model's face ordering.
        """
        theta = np.radians(incidence_deg)
        sun = np.array([np.sin(theta), 0.0, np.cos(theta)])
        # Pin scale to full scale so this helper reads as a pure *normalized*
        # response, independent of the headroom the model applies by default.
        model = VEMD8080ArrayModel(
            scale=np.ones(6),
            full_scale_a=1.0,
            dark_current_a=0.0,
            noise_rms_a=0.0,
            **kwargs,
        )
        channels = model.process_sun(sun, np.array([1.0]))
        return float(channels[0, _Z_CHANNEL_INDEX])

    def test_half_sensitivity_at_datasheet_angle(self):
        """Response must be 0.5 at the specified 65 deg half-sensitivity angle."""
        assert self._response_at(65.0) == pytest.approx(0.5, abs=0.02)

    def test_normal_incidence_is_full_scale(self):
        assert self._response_at(0.0) == pytest.approx(1.0, abs=1e-6)

    def test_nonzero_tail_at_grazing_incidence(self):
        """The response must stay positive right up to the horizon.

        Not a hard cliff at some smaller angle: a pure Lambertian ``clip(cos,
        0, 1)`` with the datasheet's 0.5-at-65 deg calibration reads only
        0.423 there. The power law plus the reflection tail keeps a positive,
        smoothly-decaying response as the sun grazes the face, reaching the
        finite tail ``VEMD8080_REFLECTION_TAIL_FRACTION`` (package and PCB
        reflections, the datasheet's Fig. 6) exactly at 90 deg -- not zero.
        """
        # The tail is asserted at the horizon, where the old clipped-cosine
        # law read a hard zero. The power-law term vanishes there (c -> 0),
        # leaving the constant reflection tail.
        assert self._response_at(90.0) == pytest.approx(
            VEMD8080_REFLECTION_TAIL_FRACTION, abs=2.0 / 4095
        )
        # Monotone decay into the horizon, with no premature cliff: just
        # short of 90 deg the face is still measurably lit above the tail.
        assert self._response_at(89.0) > self._response_at(90.0)
        assert self._response_at(89.0) > 0.02
        assert self._response_at(89.9) > VEMD8080_REFLECTION_TAIL_FRACTION
        assert self._response_at(89.9) < self._response_at(89.0)

    def test_back_face_stays_dark(self):
        """Beyond 90 deg the response must still fall to zero."""
        assert self._response_at(120.0) == pytest.approx(0.0, abs=1e-9)
        assert self._response_at(180.0) == pytest.approx(0.0, abs=1e-9)

    def test_response_is_monotonic(self):
        """Response must decrease monotonically with incidence angle."""
        values = [self._response_at(a) for a in (0, 15, 30, 45, 60, 75, 89)]
        assert all(
            a >= b for a, b in zip(values, values[1:], strict=False)
        ), values

    def test_shading_scales_the_response(self):
        """Partial shading scales the whole response down proportionally."""
        sun = np.array([np.sin(np.radians(65.0)), 0.0, np.cos(np.radians(65.0))])
        model = VEMD8080ArrayModel(
            scale=np.ones(6), full_scale_a=1.0, dark_current_a=0.0, noise_rms_a=0.0
        )
        full = model.process_sun(sun, np.array([1.0]))
        half = model.process_sun(sun, np.array([0.5]))
        assert float(half[0, _Z_CHANNEL_INDEX]) == pytest.approx(
            0.5 * float(full[0, _Z_CHANNEL_INDEX]),
            # One ADC code, since half of 0.5 is not exactly representable.
            abs=1.0 / 4095,
        )
