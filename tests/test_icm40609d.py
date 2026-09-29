"""Tests for the hardware-faithful TDK ICM-40609-D gyro + accelerometer model."""

import numpy as np
import pytest

from pocketadcs_sim.sensors import icm40609d as icm
from pocketadcs_sim.sensors.icm40609d import (
    G_STANDARD_M_S2,
    ICM40609D_GYRO_FSR_DEFAULT_DPS,
    ICM40609DModel,
)

# The quantization LSB at the default 2000 dps FSR. Every statistical test
# below is sized so its signal sits orders of magnitude above this, otherwise
# the ADC grid would swallow the effect being measured and the test would
# pass for the wrong reason.
GYRO_LSB_DPS = ICM40609D_GYRO_FSR_DEFAULT_DPS / 32768.0  # 0.061 dps
ACCEL_LSB_M_S2 = 4.0 * G_STANDARD_M_S2 / 32768.0  # ~1.2e-3 m/s^2

# ---------------------------------------------------------------------------
# Conversion / quantization
# ---------------------------------------------------------------------------


class TestNoiseStreamContinuity:
    """Regression: the noise stream must advance across calls.

    Re-seeding inside every processing call made chunked streaming replay the
    identical noise realization in every chunk. A standard-deviation test
    still passes under that bug, so only a chunk-vs-batch comparison catches
    it -- and the consequence (spurious spectral lines in any PSD or
    Allan-deviation estimate) silently corrupts a noise study.
    """

    def test_successive_chunks_differ(self):
        """Two chunks drawn one after another must not be the same data.

        The noise level here is well above the 2000 dps LSB (0.061 dps), so
        the samples survive quantization -- a sub-LSB noise level quantizes to
        all-zero and would make this test vacuous.
        """
        m = ICM40609DModel(gyro_noise_rms_dps=2.0, sample_rate_hz=100.0, seed=7)
        x = np.zeros((500, 3))
        first = m.process_gyro(x[:250])
        second = m.process_gyro(x[250:])
        assert not np.array_equal(first[10:20], second[10:20]), (
            "successive chunks replay the same noise realization"
        )

    def test_batched_still_reproduces_for_a_seed(self):
        """A single batch must still be exactly reproducible."""
        x = np.zeros((100, 3))
        a = ICM40609DModel(gyro_noise_rms_dps=2.0, seed=7).process_gyro(x)
        b = ICM40609DModel(gyro_noise_rms_dps=2.0, seed=7).process_gyro(x)
        assert np.array_equal(a, b)

    def test_chunked_matches_single_batch(self):
        """Driving the model in chunks must equal one batched call."""
        x = np.zeros((400, 3))
        batched = ICM40609DModel(gyro_noise_rms_dps=2.0, seed=7).process_gyro(x)
        m = ICM40609DModel(gyro_noise_rms_dps=2.0, seed=7)
        chunked = np.vstack([m.process_gyro(x[:150]), m.process_gyro(x[150:])])
        assert np.array_equal(batched, chunked)


# ---------------------------------------------------------------------------
def test_zero_input_is_exact_zero():
    model = ICM40609DModel()
    out = model.process_gyro(np.zeros(3))
    assert out.shape == (1, 3)
    assert np.array_equal(out, np.zeros((1, 3)))
    out_f = model.process_specific_force(np.zeros(3))
    assert np.array_equal(out_f, np.zeros((1, 3)))


def test_gyro_conversion_round_trip():
    # Half-scale on all axes: quantization error <= 1/2 LSB.
    model = ICM40609DModel()
    dps = model.gyro_full_scale_dps / 2.0
    omega = np.full((2, 3), dps * np.pi / 180.0)
    out = model.process_gyro(omega)
    lsb_rad_s = (model.gyro_full_scale_dps / 32768.0) * np.pi / 180.0
    assert np.abs(out - omega).max() <= 0.5 * lsb_rad_s


def test_gyro_quantization_step_is_one_count():
    # Inputs separated by one LSB (in dps) map to adjacent counts.
    model = ICM40609DModel()
    dps_per_count = model.gyro_full_scale_dps / 32768.0
    a = np.array([[0.0, 0.0, 0.0]], dtype=float) * (np.pi / 180.0)
    b = np.array([[dps_per_count, 0.0, 0.0]], dtype=float) * (np.pi / 180.0)
    out_a = model.process_gyro(a)
    out_b = model.process_gyro(b)
    assert out_b[0, 0] - out_a[0, 0] == pytest.approx(dps_per_count * np.pi / 180.0)


def test_accel_conversion_round_trip():
    model = ICM40609DModel()
    f_g = model.accel_full_scale_g / 2.0
    f = np.full((1, 3), f_g * G_STANDARD_M_S2)
    out = model.process_specific_force(f)
    lsb_m_s2 = (model.accel_full_scale_g / 32768.0) * G_STANDARD_M_S2
    assert np.abs(out - f).max() <= 0.5 * lsb_m_s2


# ---------------------------------------------------------------------------
# FSR configuration
# ---------------------------------------------------------------------------
def test_full_scale_options_quantize_to_their_own_grid():
    m_default = ICM40609DModel()  # +/-2000 dps default
    m_alt = ICM40609DModel(gyro_full_scale_dps=15.625)
    assert m_default.gyro_counts_per_dps == pytest.approx(16.384)
    # Engineering quantization grid 32768/FSR (the datasheet's stated
    # 2097.2 LSB/dps sensitivity is NOT what the model uses).
    assert m_alt.gyro_counts_per_dps == pytest.approx(2097.152)
    # 1000 dps: half-scale on the default FSR, and far above the alt FSR,
    # so the alt model must clip it to positive full scale.
    omega_dps = np.array([[1000.0, 0.0, 0.0]], dtype=float) * (np.pi / 180.0)
    default_out = m_default.process_gyro(omega_dps)[0, 0]
    assert 0.0 < default_out <= m_default.gyro_full_scale_dps * np.pi / 180.0
    alt_out = m_alt.process_gyro(omega_dps)[0, 0]
    assert alt_out == pytest.approx(
        m_alt.gyro_full_scale_dps * 32767.0 / 32768.0 * np.pi / 180.0
    )
    # 10 dps: comfortably in range on the alt FSR -- must land within
    # half an alt-LSB of the input.
    omega_small = np.array([[10.0, 0.0, 0.0]], dtype=float) * (np.pi / 180.0)
    alt_small = m_alt.process_gyro(omega_small)[0, 0]
    alt_lsb = (m_alt.gyro_full_scale_dps / 32768.0) * np.pi / 180.0
    assert abs(alt_small - 10.0 * np.pi / 180.0) <= 0.5 * alt_lsb
    # Both are within their own full-scale limits.
    assert default_out <= m_default.gyro_full_scale_dps * np.pi / 180.0


def test_accel_full_scale_configuration():
    m1g = ICM40609DModel(accel_full_scale_g=1.0)
    m4g = ICM40609DModel(accel_full_scale_g=4.0)
    f = np.array([[0.0, 0.0, 0.0]])
    # 3 m/s^2 (~0.306 g): in range for both 1 g and 4 g settings.
    f3 = np.array([[0.0, 3.0, 0.0]])
    out4 = m4g.process_specific_force(f3)
    assert abs(out4[0, 1] - 3.0) <= 0.5 * (4.0 / 32768.0) * G_STANDARD_M_S2
    out1 = m1g.process_specific_force(f3)
    assert abs(out1[0, 1] - 3.0) <= 0.5 * (1.0 / 32768.0) * G_STANDARD_M_S2
    assert m1g.process_specific_force(f)[0, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------
def test_gyro_clipping_at_full_scale():
    model = ICM40609DModel()  # +/-2000 dps
    big = np.array([[3000.0, 0.0, 0.0]]) * (np.pi / 180.0)
    out = model.process_gyro(big)
    # Positive saturation is 32767 counts (one count short of FSR).
    assert out[0, 0] == pytest.approx(
        model.gyro_full_scale_dps * 32767.0 / 32768.0 * np.pi / 180.0
    )
    neg_big = np.array([[-3000.0, 0.0, 0.0]]) * (np.pi / 180.0)
    out_neg = model.process_gyro(neg_big)
    # Negative saturation is exactly -32768 counts = -FSR.
    assert out_neg[0, 0] == pytest.approx(-model.gyro_full_scale_dps * np.pi / 180.0)


def test_gyro_clip_is_asymmetric_by_one_count():
    # Positive saturation is 32767 counts, negative is -32768 counts.
    model = ICM40609DModel()
    dps = model.gyro_full_scale_dps
    pos = model.process_gyro(np.array([[dps, 0.0, 0.0]]) * (np.pi / 180.0))[0, 0]
    neg = model.process_gyro(np.array([[-dps, 0.0, 0.0]]) * (np.pi / 180.0))[0, 0]
    lsb = (dps / 32768.0) * np.pi / 180.0
    assert pos == pytest.approx(32767.0 * lsb)
    assert neg == pytest.approx(-32768.0 * lsb)
    assert abs(neg) - pos == pytest.approx(lsb)


def test_accel_clipping_at_full_scale():
    model = ICM40609DModel()  # +/-4 g
    big = np.array([[0.0, 0.0, 50.0]])  # m/s^2 ~ 5.1 g
    out = model.process_specific_force(big)
    assert out[0, 2] == pytest.approx(
        model.accel_full_scale_g * 32767.0 / 32768.0 * G_STANDARD_M_S2
    )
    neg = np.array([[0.0, 0.0, -50.0]])
    out_neg = model.process_specific_force(neg)
    assert out_neg[0, 2] == pytest.approx(
        -model.accel_full_scale_g * G_STANDARD_M_S2
    )


# ---------------------------------------------------------------------------
# Bias / noise
# ---------------------------------------------------------------------------
def test_gyro_bias_is_applied_before_quantization():
    model = ICM40609DModel(gyro_bias_dps=np.array([1.0, 2.0, 3.0]))
    out = model.process_gyro(np.zeros((1, 3)))
    expected_dps = np.array([1.0, 2.0, 3.0]) * np.pi / 180.0
    lsb = (model.gyro_full_scale_dps / 32768.0) * np.pi / 180.0
    assert np.abs(out[0] - expected_dps).max() <= 0.5 * lsb


def test_accel_bias_is_applied_before_quantization():
    model = ICM40609DModel(accel_bias_m_s2=np.array([0.1, 0.2, 0.3]))
    out = model.process_specific_force(np.zeros((1, 3)))
    lsb = (model.accel_full_scale_g / 32768.0) * G_STANDARD_M_S2
    assert np.abs(out[0] - np.array([0.1, 0.2, 0.3])).max() <= 0.5 * lsb


def test_noise_seed_determinism():
    omega = np.random.default_rng(0).normal(size=(50, 3)) * 0.5
    f = np.random.default_rng(1).normal(size=(50, 3)) * 2.0
    a = ICM40609DModel(gyro_noise_rms_dps=1.0, accel_noise_rms_m_s2=0.01, seed=42)
    b = ICM40609DModel(gyro_noise_rms_dps=1.0, accel_noise_rms_m_s2=0.01, seed=42)
    c = ICM40609DModel(gyro_noise_rms_dps=1.0, accel_noise_rms_m_s2=0.01, seed=7)
    assert np.array_equal(a.process_gyro(omega), b.process_gyro(omega))
    assert np.array_equal(a.process_specific_force(f), b.process_specific_force(f))
    assert not np.array_equal(a.process_gyro(omega), c.process_gyro(omega))
    assert not np.array_equal(a.process_specific_force(f), c.process_specific_force(f))


def test_noise_rms_is_close_to_config():
    n = 300_000
    omega = np.zeros((n, 3))
    f = np.zeros((n, 3))
    gyro = ICM40609DModel(gyro_noise_rms_dps=1.0, seed=123)
    out_gyro = gyro.process_gyro(omega)
    noise_dps = out_gyro * (180.0 / np.pi)
    # Quantization (16.4 counts/dps at 2000 dps) is small compared to 1 dps.
    assert 0.8 * 1.0 < np.std(noise_dps) < 1.2 * 1.0
    accel = ICM40609DModel(accel_noise_rms_m_s2=0.05, seed=124)
    out_accel = accel.process_specific_force(f)
    assert 0.8 * 0.05 < np.std(out_accel) < 1.2 * 0.05


def test_zero_noise_mode_is_exact_and_repeatable():
    omega = np.array([[0.7, -0.1, 0.5], [-0.2, 0.9, -0.4]]) * (np.pi / 180.0)
    f = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]])
    model = ICM40609DModel(seed=1)  # zero-noise by default
    assert np.array_equal(model.process_gyro(omega), model.process_gyro(omega))
    assert np.array_equal(model.process_specific_force(f), model.process_specific_force(f))
    # With zero bias/noise the output must be a multiple of one count.
    lsb_gyro = (model.gyro_full_scale_dps / 32768.0) * np.pi / 180.0
    gyro_counts = model.process_gyro(omega) / lsb_gyro
    assert np.allclose(gyro_counts, np.round(gyro_counts))
    lsb_accel = (model.accel_full_scale_g / 32768.0) * G_STANDARD_M_S2
    accel_counts = model.process_specific_force(f) / lsb_accel
    assert np.allclose(accel_counts, np.round(accel_counts))


# ---------------------------------------------------------------------------
# Sample rate / ODR
# ---------------------------------------------------------------------------
def test_sample_rate_validation_and_storage():
    model = ICM40609DModel(sample_rate_hz=4096.0)
    assert model.sample_rate_hz == pytest.approx(4096.0)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            ICM40609DModel(sample_rate_hz=bad)


# ---------------------------------------------------------------------------
# Malformed inputs
# ---------------------------------------------------------------------------
def test_invalid_config():
    for kwargs in (
        {"sample_rate_hz": 0.0},
        {"gyro_full_scale_dps": 0.0},
        {"gyro_full_scale_dps": float("nan")},
        {"accel_full_scale_g": -1.0},
        {"gyro_noise_rms_dps": -0.1},
        {"accel_noise_rms_m_s2": float("inf")},
        {"gyro_bias_dps": np.ones(2)},  # wrong shape
        {"gyro_bias_dps": np.array([0.0, float("nan"), 0.0])},
        {"accel_bias_m_s2": [1.0, 2.0, 3.0, 4.0]},
    ):
        with pytest.raises(ValueError):
            ICM40609DModel(**kwargs)


def test_invalid_inputs():
    model = ICM40609DModel()
    with pytest.raises(ValueError):
        model.process_gyro(np.ones((2, 2)))
    with pytest.raises(ValueError):
        model.process_gyro(np.ones(4))
    with pytest.raises(ValueError):
        model.process_gyro(np.array([[1.0, 0.0, float("nan")]]))
    with pytest.raises(ValueError):
        model.process_specific_force(np.ones(3, dtype=object).astype(float) * np.nan)
    with pytest.raises(ValueError):
        model.process_specific_force(np.zeros(2))


# ---------------------------------------------------------------------------
# Truth integration (gyro path only)
# ---------------------------------------------------------------------------
def test_measure_gyro_with_truth_history():
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
    model = ICM40609DModel(sample_rate_hz=2.0)  # zero-noise by default
    out = model.measure_gyro(truth)
    indices = sample_history(truth, model.sample_rate_hz)
    assert out.shape == (len(indices), 3)
    # Readings equal the quantized truth angular velocity at the sampled rows.
    expected = model.process_gyro(truth.angular_velocity_body_rad_s[indices])
    assert np.array_equal(out, expected)
    # Bounded by the gyro full scale, which is the default FSR.
    assert model.gyro_full_scale_dps == ICM40609D_GYRO_FSR_DEFAULT_DPS
    bound = model.gyro_full_scale_dps * np.pi / 180.0
    assert (np.abs(out) <= bound).all()


# ---------------------------------------------------------------------------
# Density-based (ARW) noise
# ---------------------------------------------------------------------------


def test_datasheet_density_constants():
    """The published noise densities are the model's documented defaults."""
    assert pytest.approx(4.5e-3) == icm.ICM40609D_GYRO_NOISE_DENSITY_DPS_SQRT_HZ
    assert pytest.approx(100e-6 * G_STANDARD_M_S2) == (
        icm.ICM40609D_ACCEL_NOISE_DENSITY_M_S2_SQRT_HZ
    )


def test_noise_sigma_is_density_times_sqrt_bandwidth():
    """Per-sample sigma follows the density * sqrt(ENBW) relation."""
    m = ICM40609DModel(
        sample_rate_hz=200.0,
        gyro_noise_density_dps_sqrt_hz=4.5e-3,
        seed=5,
    )
    assert m.gyro_noise_rms_dps == pytest.approx(4.5e-3 * np.sqrt(100.0))


def test_default_bandwidth_is_half_the_sample_rate():
    """ENBW defaults to fs/2 (an engineering assumption, documented)."""
    m = ICM40609DModel(sample_rate_hz=200.0, gyro_noise_density_dps_sqrt_hz=4.5e-3)
    assert m.noise_bandwidth_hz == pytest.approx(100.0)


def test_halving_sample_rate_scales_noise_by_one_over_sqrt_two():
    """THE core density test: fixed density, half the rate -> sigma/sqrt(2).

    A rate-independent sigma (the pre-existing behaviour) would give a ratio
    of 1.0 here, so this test fails until noise is reparametrized by density.

    The density is chosen so the per-sample sigma is ~1000x the 0.061 dps
    quantization LSB, which keeps the effect far above the ADC floor.
    """
    density = 0.5  # dps/sqrt(Hz) -- an oversized, easily-measured value
    n = 200_000
    fast = ICM40609DModel(
        sample_rate_hz=200.0, gyro_noise_density_dps_sqrt_hz=density, seed=99
    )
    slow = ICM40609DModel(
        sample_rate_hz=100.0, gyro_noise_density_dps_sqrt_hz=density, seed=99
    )
    sigma_fast = np.std(fast.process_gyro(np.zeros((n, 3))) * (180.0 / np.pi))
    sigma_slow = np.std(slow.process_gyro(np.zeros((n, 3))) * (180.0 / np.pi))
    # Both sigmas sit orders of magnitude above the LSB, so neither is
    # quantization-limited and the comparison is meaningful.
    assert sigma_fast > 20.0 * GYRO_LSB_DPS
    assert sigma_slow > 20.0 * GYRO_LSB_DPS
    assert sigma_slow / sigma_fast == pytest.approx(1.0 / np.sqrt(2.0), rel=0.02)


def test_accel_noise_also_scales_as_sqrt_bandwidth():
    density = 0.2  # m/s^2/sqrt(Hz)
    n = 200_000
    fast = ICM40609DModel(
        sample_rate_hz=400.0, accel_noise_density_m_s2_sqrt_hz=density, seed=13
    )
    slow = ICM40609DModel(
        sample_rate_hz=200.0, accel_noise_density_m_s2_sqrt_hz=density, seed=13
    )
    s_fast = np.std(fast.process_specific_force(np.zeros((n, 3))))
    s_slow = np.std(slow.process_specific_force(np.zeros((n, 3))))
    assert s_fast > 20.0 * ACCEL_LSB_M_S2
    assert s_slow / s_fast == pytest.approx(1.0 / np.sqrt(2.0), rel=0.02)


def test_explicit_bandwidth_overrides_the_rate_derived_default():
    """A pinned ENBW makes the rate irrelevant, isolating sigma = n*sqrt(BW)."""
    density = 0.5
    a = ICM40609DModel(
        sample_rate_hz=200.0,
        gyro_noise_density_dps_sqrt_hz=density,
        noise_bandwidth_hz=10.0,
        seed=3,
    )
    b = ICM40609DModel(
        sample_rate_hz=100.0,
        gyro_noise_density_dps_sqrt_hz=density,
        noise_bandwidth_hz=10.0,
        seed=3,
    )
    assert a.noise_bandwidth_hz == b.noise_bandwidth_hz == 10.0
    assert a.gyro_noise_rms_dps == pytest.approx(b.gyro_noise_rms_dps)
    assert a.gyro_noise_rms_dps == pytest.approx(density * np.sqrt(10.0))


def test_explicit_rms_overrides_density():
    """The legacy per-sample knob still wins, so old call sites are unaffected."""
    m = ICM40609DModel(
        sample_rate_hz=100.0,
        gyro_noise_density_dps_sqrt_hz=4.5e-3,
        gyro_noise_rms_dps=1.25,
    )
    assert m.gyro_noise_rms_dps == pytest.approx(1.25)


def test_density_and_bias_instability_validation():
    for kwargs in (
        {"gyro_noise_density_dps_sqrt_hz": -1.0},
        {"accel_noise_density_m_s2_sqrt_hz": float("nan")},
        {"noise_bandwidth_hz": 0.0},
        {"noise_bandwidth_hz": -5.0},
        {"gyro_bias_instability_dps": -0.1},
        {"accel_bias_instability_m_s2": float("inf")},
        {"gyro_bias_instability_dps": np.ones(2)},
    ):
        with pytest.raises(ValueError):
            ICM40609DModel(**kwargs)


# ---------------------------------------------------------------------------
# Bias instability (flicker random walk + turn-on-to-turn-on spread)
# ---------------------------------------------------------------------------


def test_bias_instability_moves_over_a_long_run():
    """The bias must not be frozen: it random-walks (flicker) within a run.

    Averages over long blocks to separate the slow bias wander from the much
    smaller white noise, then checks the block means actually move and that
    the total spread exceeds the configured flicker sigma.
    """
    m = ICM40609DModel(
        sample_rate_hz=100.0,
        gyro_bias_instability_dps=0.5,
        seed=21,
    )
    n = 100_000
    out = m.process_gyro(np.zeros((n, 3))) * (180.0 / np.pi)
    # The flicker component alone must be well above the LSB or this test
    # would be measuring quantization, not bias instability.
    assert np.all(m.gyro_bias_instability_dps > 4.0 * GYRO_LSB_DPS)
    block = out[:, 0].reshape(-1, 1000).mean(axis=1)
    assert block.std() > 0.0, "bias is constant across a 1000 s run"
    # The flicker reaches a substantial fraction of its configured 1-sigma
    # level over 1000 s (1000 s is ~16 correlation times, so the AR(1) has
    # long since settled to its stationary spread).
    assert out[:, 0].std() > 0.5 * float(m.gyro_bias_instability_dps[0])


def test_bias_instability_is_slow_relative_to_white_noise():
    """Flicker is low-frequency: block means track, sample-to-sample does not.

    Guards against implementing 'bias instability' as plain per-sample white
    noise, which would masquerade as a moving bias in the previous test.
    """
    m = ICM40609DModel(sample_rate_hz=100.0, gyro_bias_instability_dps=2.0, seed=22)
    n = 40_000
    out = m.process_gyro(np.zeros((n, 3)))[:, 0] * (180.0 / np.pi)
    step_var = np.var(np.diff(out))
    slow = out[: (n // 1000) * 1000].reshape(-1, 1000)
    # If bias were white, the variance of the 1000-sample block means would
    # be 1/1000 of the sample variance. For a slow bias it is far larger.
    assert np.var(slow.mean(axis=1)) > 100.0 * step_var / 1000.0


def test_turn_on_spread_differs_between_instances():
    """Turn-on-to-turn-on spread: two models, same config, different draws."""
    a = ICM40609DModel(sample_rate_hz=100.0, gyro_turn_on_bias_std_dps=1.0, seed=1)
    b = ICM40609DModel(sample_rate_hz=100.0, gyro_turn_on_bias_std_dps=1.0, seed=2)
    z = np.zeros((200, 3))
    assert not np.allclose(
        a.process_gyro(z)[:, 0] * (180.0 / np.pi), b.process_gyro(z)[:, 0] * (180.0 / np.pi)
    )


def test_turn_on_spread_is_deterministic_for_a_seed():
    a = ICM40609DModel(sample_rate_hz=100.0, gyro_turn_on_bias_std_dps=1.0, seed=5)
    b = ICM40609DModel(sample_rate_hz=100.0, gyro_turn_on_bias_std_dps=1.0, seed=5)
    z = np.zeros((50, 3))
    assert np.array_equal(a.process_gyro(z), b.process_gyro(z))


def test_bias_instability_reproduces_and_chunks_identically():
    """Batched == chunked, with the bias random walk and noise both active.

    The random walk is stateful, so a re-seeded-per-call implementation would
    restart the walk in every chunk; only this comparison catches it.
    """
    kwargs = dict(
        sample_rate_hz=100.0,
        gyro_bias_instability_dps=1.0,
        gyro_turn_on_bias_std_dps=0.5,
        accel_bias_instability_m_s2=0.05,
        accel_turn_on_bias_std_m_s2=0.02,
        gyro_noise_density_dps_sqrt_hz=0.2,
        seed=31,
    )
    x = np.zeros((600, 3))
    f = np.zeros((600, 3))
    batch = ICM40609DModel(**kwargs)
    got_g, got_f = batch.process_gyro(x), batch.process_specific_force(f)
    again = ICM40609DModel(**kwargs)
    assert np.array_equal(got_g, again.process_gyro(x))
    assert np.array_equal(got_f, again.process_specific_force(f))
    stream = ICM40609DModel(**kwargs)
    chunked_g = np.vstack([stream.process_gyro(x[:137]), stream.process_gyro(x[137:])])
    chunked_f = np.vstack(
        [stream.process_specific_force(f[:137]), stream.process_specific_force(f[137:])]
    )
    assert np.array_equal(got_g, chunked_g)
    assert np.array_equal(got_f, chunked_f)


def test_default_bias_instability_is_off_and_exact_zero_survives():
    """Zero defaults must still give bit-exact zero output."""
    m = ICM40609DModel(sample_rate_hz=200.0)
    assert np.all(m.gyro_bias_instability_dps == 0.0)
    assert np.all(m.accel_bias_instability_m_s2 == 0.0)
    assert np.array_equal(m.process_gyro(np.zeros((5, 3))), np.zeros((5, 3)))
    assert np.array_equal(
        m.process_specific_force(np.zeros((5, 3))), np.zeros((5, 3))
    )


def test_datasheet_bias_instability_constants_are_labeled():
    """New module constants must be traceable to datasheet vs assumption."""
    assert pytest.approx(2.0) == icm.ICM40609D_GYRO_BIAS_INSTABILITY_DPS
    assert pytest.approx(0.25e-3 * G_STANDARD_M_S2) == (
        icm.ICM40609D_ACCEL_BIAS_INSTABILITY_M_S2
    )
    assert icm.ICM40609D_BIAS_CORRELATION_TIME_S > 0.0
