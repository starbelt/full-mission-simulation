"""Tests for the hardware-faithful VEMD8080 six-photodiode sun array."""

import numpy as np
import pytest

from pocketadcs_sim.sensors.vemd8080_array import (
    _RESPONSE_EXPONENT,
    CHANNEL_NAMES,
    CHANNEL_NORMALS,
    VEMD8080ArrayModel,
)

#: Normalized range used by the geometry/shape tests below. The model's real
#: default is 1e-5 A (a photodiode's actual photocurrent), but these tests
#: assert on *fractions* of full scale, so they pin the range to 1.0 and keep
#: ``scale`` in the same units. Tests that care about absolute amps or about
#: noise surviving the ADC state the microamp range explicitly instead.
_UNIT_FULL_SCALE_A = 1.0


def _unit_model(**kwargs) -> VEMD8080ArrayModel:
    """A model whose full scale is 1.0, so scale values read as fractions."""
    kwargs.setdefault("full_scale_a", _UNIT_FULL_SCALE_A)
    if "scale" not in kwargs:
        kwargs["scale"] = np.ones(6)
    return VEMD8080ArrayModel(**kwargs)


def test_channel_names_and_normals_match_faces():
    assert CHANNEL_NAMES == ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
    assert CHANNEL_NORMALS.shape == (6, 3)
    # Each normal is a unit vector along its face axis, sign per name.
    expected = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    assert np.array_equal(CHANNEL_NORMALS, expected)
    assert np.allclose(np.linalg.norm(CHANNEL_NORMALS, axis=1), 1.0)


def test_face_geometry_positive_x_sun():
    # Sun exactly along +X body axis: only the +X channel is lit (in full
    # light, no dark floor), at its configured scale. scale is pinned equal
    # to full scale so "at its configured scale" is exactly full scale.
    model = VEMD8080ArrayModel(dark_current_a=0.0, scale=np.ones(6), full_scale_a=1.0)
    out = model.process_sun(np.array([1.0, 0.0, 0.0]))
    assert out.shape == (1, 6)
    assert out[0, 0] == pytest.approx(1.0)  # +X
    assert out[0, 1] == pytest.approx(0.0)  # -X
    assert out[0, 2] == pytest.approx(0.0)  # +Y
    assert out[0, 3] == pytest.approx(0.0)  # -Y
    assert out[0, 4] == pytest.approx(0.0)  # +Z
    assert out[0, 5] == pytest.approx(0.0)  # -Z


def test_face_geometry_oblique_sun_angular_response():
    # Sun at 45 deg between +X and +Y: both lit with the same response, all
    # others exactly zero (no negative-visibility leakage). The response is
    # cos ** q, q chosen so the value is 0.5 at the datasheet's 65 deg
    # half-sensitivity angle, not a bare cosine.
    c = np.cos(np.pi / 4.0)
    expected = c**_RESPONSE_EXPONENT
    model = _unit_model(dark_current_a=0.0)
    out = model.process_sun(np.array([c, c, 0.0]))
    # Compared to within one ADC LSB: the model quantizes, so an exact
    # match to the continuous ideal is not the right assertion.
    lsb = 1.0 / 4095
    assert out[0, 0] == pytest.approx(expected, abs=lsb)
    assert out[0, 2] == pytest.approx(expected, abs=lsb)
    assert out[0, 1] == pytest.approx(0.0)
    assert out[0, 3] == pytest.approx(0.0)
    assert out[0, 4] == pytest.approx(0.0)
    assert out[0, 5] == pytest.approx(0.0)


def test_eclipse_total_shadows_all_channels():
    # Total eclipse: all channels read only the dark-current floor. The floor
    # is ~8 orders of magnitude below one 12-bit LSB at the default 1 A full
    # scale, so the ADC collapses it to exactly zero -- which is the correct
    # and expected outcome, not a bug. A real deployment would run the array
    # in transimpedance mode with an analog gain stage so the dark current
    # lands mid-scale; that is an analog front-end choice, not a sensor one.
    model = _unit_model(dark_current_a=10.0e-9)
    out = model.process_sun(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                           shadow=np.array([0.0, 0.0]))
    assert np.all(out == 0.0)
    # With a full scale matched to the dark current, it is resolved instead.
    fine = VEMD8080ArrayModel(scale=np.ones(6), dark_current_a=10.0e-9,
                              full_scale_a=1.0e-7)
    fine_out = fine.process_sun(np.array([1.0, 0.0, 0.0]), shadow=np.array([0.0]))
    assert fine_out[0, 0] > 0.0
    # Zero dark current: total eclipse is exactly zero on every channel.
    dark = VEMD8080ArrayModel(dark_current_a=0.0)
    assert np.all(dark.process_sun(np.array([1.0, 0.0, 0.0]), shadow=np.array([0.0])) == 0.0)


def test_partial_eclipse_scales_visible_channels():
    # Half shadow: visible channels read exactly half of full light.
    model = _unit_model(dark_current_a=0.0)
    out = model.process_sun(np.array([1.0, 0.0, 0.0]), shadow=np.array([0.5]))
    # Half of full scale quantizes to code 2048 of 4095, not to 0.5 exactly.
    assert out[0, 0] == pytest.approx(2048 / 4095, abs=1e-12)
    assert out[0, 1] == pytest.approx(0.0)


def test_per_channel_scale():
    # Each channel's scale applies independently.
    model = _unit_model(scale=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2]),
                      dark_current_a=0.0, full_scale_a=2.0)
    out = model.process_sun(np.array([1.0, 0.0, 0.0]), shadow=np.array([1.0]))
    # scale is in amps against a 2.0 A range here, so 0.2 reads 0.2 A and
    # quantizes to code 0.2/2.0*4095 = 410.
    assert out[0, 0] == pytest.approx(0.2, abs=1.0 / 4095)
    # -X face is dark, so its scale does not appear.
    assert out[0, 1] == pytest.approx(0.0)


def test_full_scale_clipping():
    # Scale above the full scale saturates the lit channel at full scale.
    model = VEMD8080ArrayModel(scale=np.array([5.0, 0.5, 0.5, 0.5, 0.5, 0.5]),
                               dark_current_a=0.0, full_scale_a=1.0)
    out = model.process_sun(np.array([1.0, 0.0, 0.0]))
    assert out[0, 0] == pytest.approx(1.0)
    # Sun along +X: only the +X face is lit; all others read 0.
    assert np.all(out[0, 1:] == 0.0)
    # Dark floor above full scale also clips.
    floor = VEMD8080ArrayModel(dark_current_a=2.0, full_scale_a=1.0)
    assert np.all(floor.process_sun(np.array([1.0, 0.0, 0.0])) == 1.0)


def test_noise_seed_determinism():
    sun = np.array([[0.5, 0.5, 0.5], [-0.3, 0.6, -0.6]])
    a = VEMD8080ArrayModel(dark_current_a=0.0, noise_rms_a=0.01, seed=42).process_sun(sun)
    b = VEMD8080ArrayModel(dark_current_a=0.0, noise_rms_a=0.01, seed=42).process_sun(sun)
    c = _unit_model(dark_current_a=0.0, noise_rms_a=0.01, seed=7).process_sun(sun)
    again = VEMD8080ArrayModel(dark_current_a=0.0, noise_rms_a=0.01, seed=42).process_sun(sun)
    assert np.array_equal(a, b)
    assert np.array_equal(a, again)
    assert not np.array_equal(a, c)


def test_noise_scale_matches_config():
    n = 200_000
    # Sun normal to +X so only channel 0 carries the signal; channels 1-5
    # sit at the zero floor and read pure noise (clipped at 0).
    sun = np.ones((n, 3)) * np.array([1.0, 0.0, 0.0])
    model = _unit_model(dark_current_a=0.0, noise_rms_a=0.01, seed=123)
    out = model.process_sun(sun)
    # Channel 0: signal (1.0) + N(0, 0.01) noise, below the 1.0 full scale.
    assert 0.5 * 0.01 < np.std(out[:, 0]) < 2.0 * 0.01
    # Dark channels: clipped at the 0 floor, so only positive noise leaks.
    assert out[:, 1].min() == pytest.approx(0.0)
    assert np.mean(out[:, 1]) > 0.0  # half-wave-rectified mean


def test_zero_noise_mode_is_exact_and_repeatable():
    sun = np.array([[0.7, -0.1, 0.5], [-0.2, 0.9, -0.4]])
    model = VEMD8080ArrayModel(
        dark_current_a=0.0,
        noise_rms_a=0.0,
        seed=1,
        scale=np.ones(6),
        full_scale_a=1.0,
    )
    a = model.process_sun(sun)
    assert np.array_equal(a, model.process_sun(sun))

    def resp(value: float) -> float:
        return max(value, 0.0) ** _RESPONSE_EXPONENT

    def quantized(value: float) -> float:
        # The model rounds onto the 12-bit grid, so the expected value must
        # go through the same rounding to be compared exactly.
        return np.rint(resp(value) * 4095) / 4095

    # Exact angular-response values up to one ADC code, no numerical drift.
    lsb = 1.0 / 4095
    assert a[0, 0] == pytest.approx(quantized(0.7), abs=lsb)
    assert a[0, 1] == pytest.approx(0.0)
    assert a[0, 4] == pytest.approx(quantized(0.5), abs=lsb)
    assert a[1, 2] == pytest.approx(quantized(0.9), abs=lsb)
    assert a[1, 3] == pytest.approx(0.0)
    # Determinism must hold exactly, code for code.
    assert np.array_equal(a, model.process_sun(sun))


def test_invalid_config():
    for kwargs in (
        {"sample_rate_hz": 0.0},
        {"sample_rate_hz": -1.0},
        {"scale": np.ones(3)},  # wrong length
        {"scale": np.array([1.0, 0.0, -1.0, 1.0, 1.0, 1.0])},  # negative
        {"scale": np.full(6, float("nan"))},
        {"dark_current_a": -1.0},
        {"noise_rms_a": -1.0},
        {"noise_rms_a": float("inf")},
        {"full_scale_a": 0.0},
        {"full_scale_a": float("nan")},
    ):
        with pytest.raises(ValueError):
            VEMD8080ArrayModel(**kwargs)


def test_invalid_inputs():
    model = VEMD8080ArrayModel()
    with pytest.raises(ValueError):
        model.process_sun(np.ones((2, 2)))
    with pytest.raises(ValueError):
        model.process_sun(np.ones(4))
    with pytest.raises(ValueError):
        model.process_sun(np.array([[1.0, 0.0, float("nan")]]))
    with pytest.raises(ValueError):
        model.process_sun(np.array([1.0, 0.0, 0.0]), shadow=np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        model.process_sun(np.array([1.0, 0.0, 0.0]), shadow=np.array([1.5]))
    with pytest.raises(ValueError):
        model.process_sun(np.array([1.0, 0.0, 0.0]), shadow=np.array([-0.1]))


def test_measure_with_truth_history():
    """Integration path: sample the truth environment at the sensor rate."""
    from pocketadcs_sim.dynamics import InitialState, SimulationConfig, SpacecraftConfig
    from pocketadcs_sim.sensors.base import sample_history
    from pocketadcs_sim.truth_environment import TruthEnvironmentConfig, run_truth_environment

    vehicle = SpacecraftConfig(
        mass_kg=0.25,
        inertia_kg_m2=np.diag([8.0e-4, 9.0e-4, 7.0e-4]),
    )
    initial = InitialState(
        position_m=np.array([6_878_137.0, 0.0, 0.0]),
        velocity_m_s=np.array([0.0, 7_612.6, 0.0]),
        mrp_body_from_inertial=np.array([0.1, -0.2, 0.05]),
        angular_velocity_body_rad_s=np.array([0.01, -0.02, 0.03]),
    )
    truth = run_truth_environment(
        vehicle,
        initial,
        SimulationConfig(duration_s=2.0, step_s=0.5),
        TruthEnvironmentConfig(epoch_utc="2025-01-01T00:00:00", gravity_degree=4),
    )
    model = VEMD8080ArrayModel(dark_current_a=0.0)  # zero-noise by default
    out = model.measure(truth)
    indices = sample_history(truth, model.sample_rate_hz)
    assert out.shape == (len(indices), 6)
    # Every reading is non-negative and within the full scale.
    assert (out >= 0.0).all()
    assert (out <= model.full_scale_a).all()
    # In full light, at least one face must be lit (sun is not exactly
    # perpendicular to all six faces).
    assert (out.sum(axis=1) > 0.0).all()
    # During shadow (shadow factor 0), every channel reads the floor (0).
    shadow = np.asarray(truth.illumination_fraction[indices], dtype=float)
    if (shadow == 0.0).any():
        assert np.all(out[shadow == 0.0] == 0.0)
