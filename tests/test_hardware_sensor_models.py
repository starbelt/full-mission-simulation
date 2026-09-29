"""Tests for the hardware-faithful MMC5983MA magnetometer model."""

import numpy as np
import pytest

from pocketadcs_sim.sensors.hardware_models import (
    GAUSS_TO_TESLA,
    MMC5983_BANDS,
    MMC5983MAModel,
    counts_to_field_t,
    field_to_counts,
)

G = GAUSS_TO_TESLA  # 1 Gauss in Tesla
HALF_LSB_18BIT = 0.5 / 16384.0 * G  # half-LSB in Tesla


def test_null_field_gives_null_offset_and_zero_tesla():
    counts = field_to_counts(np.zeros((1, 3)))
    assert counts.shape == (1, 3)
    assert (counts == 131072).all()
    out = MMC5983MAModel().process_field(np.zeros((3,)))
    assert out.shape == (1, 3)
    assert np.allclose(out, 0.0)


def test_full_scale_clipping_at_fsr():
    model = MMC5983MAModel()
    # +8 G FSR quantizes to the max 18-bit code 262143, i.e. just under +8 G.
    plus_limit = (262143 - 131072) / 16384.0 * G
    assert np.allclose(model.process_field(np.full((1, 3), 8.0 * G))[0], plus_limit)
    # -8 G FSR maps exactly to code 0.
    assert np.allclose(model.process_field(np.full((1, 3), -8.0 * G))[0], -8.0 * G)
    # Beyond the FSR clips at the same saturated value.
    assert np.allclose(model.process_field(np.full((1, 3), 12.0 * G))[0], plus_limit)


def test_unit_conversion_roundtrip_within_half_lsb():
    model = MMC5983MAModel()
    field = np.array([[0.42 * G, -0.17 * G, 0.091 * G]])  # ~Earth-like field
    out = model.process_field(field)
    assert (np.abs(out - field) <= HALF_LSB_18BIT * (1.0 + 1e-9)).all()


def test_dequantize_matches_quantize():
    codes = field_to_counts(np.array([[2.0 * G, -1.0 * G, 0.5 * G]]))
    assert codes.dtype.kind in ("i", "u")
    back = counts_to_field_t(codes)
    assert (np.abs(back - np.array([[2.0 * G, -1.0 * G, 0.5 * G]]))
            <= HALF_LSB_18BIT * (1.0 + 1e-9)).all()


def test_seed_determinism_and_repeatability():
    field = np.linspace(-4.0 * G, 4.0 * G, 200).reshape(-1, 1) * np.ones((1, 3))
    a = MMC5983MAModel(noise_rms_t=0.4e-6, seed=42).process_field(field)
    b = MMC5983MAModel(noise_rms_t=0.4e-6, seed=42).process_field(field)
    c = MMC5983MAModel(noise_rms_t=0.4e-6, seed=7).process_field(field)
    assert np.array_equal(a, b)
    # Repeated calls on the same instance replay the same seeded stream.
    again = MMC5983MAModel(noise_rms_t=0.4e-6, seed=42).process_field(field)
    assert np.array_equal(a, again)
    # A different seed changes the noise realization.
    assert not np.array_equal(a, c)


def test_noise_scale_matches_config():
    n = 200_000
    model = MMC5983MAModel(noise_rms_t=0.4e-6, seed=123)
    out = model.process_field(np.zeros((n, 3))).reshape(-1)
    # Noise (0.4 uT) dominates the 18-bit quantization step (0.0625 uT),
    # so the residual std must track the configured RMS within a factor 2.
    assert 0.5 * 0.4e-6 < np.std(out) < 2.0 * 0.4e-6


def test_zero_noise_mode_is_exact():
    model = MMC5983MAModel(noise_rms_t=0.0, seed=1)
    field = np.array([[1.23456789e-4, -2.1e-5, 3.05e-4]])
    out = model.process_field(field)
    # Deterministic: identical on re-call and within half an LSB of truth.
    assert np.array_equal(out, model.process_field(field))
    assert (np.abs(out - field) <= HALF_LSB_18BIT * (1.0 + 1e-9)).all()


def test_bias_shifts_output_by_bias_counts():
    no_bias = MMC5983MAModel().process_field(np.zeros((1, 3)))
    biased = MMC5983MAModel(bias_counts=1000.0).process_field(np.zeros((1, 3)))
    shift = (1000.0 / 16384.0) * G
    assert np.allclose(biased[0] - no_bias[0], shift)


def test_band_table_matches_datasheet():
    assert MMC5983_BANDS[0] == {"odr_hz": 50.0, "noise_rms_t": 0.4e-6}
    assert MMC5983_BANDS[1] == {"odr_hz": 100.0, "noise_rms_t": 0.6e-6}
    assert MMC5983_BANDS[2] == {"odr_hz": 225.0, "noise_rms_t": 0.8e-6}
    assert MMC5983_BANDS[3] == {"odr_hz": 580.0, "noise_rms_t": 1.2e-6}


def test_config_errors():
    for kwargs in (
        {"odr_hz": 0.0},
        {"odr_hz": -5.0},
        {"odr_hz": 2000.0},
        {"bits": 17},
        {"noise_rms_t": -1.0},
        {"noise_rms_t": float("nan")},
        {"bias_counts": float("inf")},
    ):
        with pytest.raises(ValueError):
            MMC5983MAModel(**kwargs)
    with pytest.raises(ValueError):
        MMC5983MAModel().process_field(np.ones((2, 2)))
    with pytest.raises(ValueError):
        MMC5983MAModel().process_field(np.ones(4))


def test_measure_with_truth_history():
    """Integration path: sample the truth environment at the ODR."""
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
    model = MMC5983MAModel(odr_hz=10.0)  # zero-noise by default
    out = model.measure(truth)
    indices = sample_history(truth, 10.0)
    assert out.shape == (len(indices), 3)
    ideal = truth.magnetic_field_body_t[indices]
    assert (np.abs(out - ideal) <= HALF_LSB_18BIT * (1.0 + 1e-9)).all()
    # LEO field (~0.3-0.6 G) stays inside the +/-8 G FSR: no saturation.
    assert np.abs(out).max() < 8.0 * G
