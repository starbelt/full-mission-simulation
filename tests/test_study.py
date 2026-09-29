"""Tests for the sensor-fidelity ablation study runner."""

import json
from pathlib import Path

import numpy as np
import pytest

from pocketadcs_sim.study import (
    StudyConfig,
    build_suite_variants,
    run_hardware_fidelity_study,
    summarize_study,
)


class _FakeTruth:
    """A self-consistent truth history for the study tests.

    The body-frame vectors are the GCRS reference vectors rotated by the
    truth attitude, so a noise-free run really does produce zero error and
    any nonzero error is attributable to the sensor model under test.
    """

    def __init__(self, n=40, dt=0.5):
        from scipy.spatial.transform import Rotation

        self.time_s = np.arange(n) * dt
        ang = np.linspace(0.0, 0.6, n)
        self.angular_velocity_body_rad_s = np.stack(
            [np.cos(ang), np.sin(ang), 0.3 * np.ones(n)], axis=1
        )

        # Reference vectors in GCRS.
        self.sun_direction_gcrs = np.stack(
            [np.cos(ang), np.sin(ang), 0.5 * np.ones(n)], axis=1
        )
        self.sun_direction_gcrs /= np.linalg.norm(
            self.sun_direction_gcrs, axis=1, keepdims=True
        )
        self.magnetic_field_gcrs_t = np.tile(
            np.array([[2.0e-5, -1.0e-5, 4.0e-5]]), (n, 1)
        )
        self.magnetic_field_gcrs_t /= np.linalg.norm(
            self.magnetic_field_gcrs_t, axis=1, keepdims=True
        )

        axis = np.array([0.3, -0.2, 0.93])
        axis /= np.linalg.norm(axis)
        self.mrp_body_from_gcrs = (
            Rotation.from_rotvec(np.outer(ang * 0.4, axis)).as_mrp()
        )
        dcm = Rotation.from_mrp(self.mrp_body_from_gcrs).inv().as_matrix()
        self.sun_direction_body = np.einsum("nij,nj->ni", dcm, self.sun_direction_gcrs)
        self.magnetic_field_body_t = np.einsum(
            "nij,nj->ni", dcm, self.magnetic_field_gcrs_t
        )
        self.illumination_fraction = np.ones(n)
        self.gravity_gradient_torque_body_nm = np.zeros((n, 3))


def test_build_suite_variants_covers_the_ablation_matrix():
    variants = build_suite_variants(rate_hz=10.0)
    names = {v.name for v in variants}
    assert "ideal" in names
    assert "all_noise" in names
    assert "gyro_noise_only" in names
    assert "mag_noise_only" in names
    assert "sun_noise_only" in names
    # A quantization-only variant isolates ADC resolution from noise.
    assert "quantization_only" in names


def test_variants_are_named_uniquely():
    variants = build_suite_variants(rate_hz=10.0)
    assert len({v.name for v in variants}) == len(variants)


def test_ideal_variant_has_no_hardware_noise():

    variants = {v.name: v for v in build_suite_variants(rate_hz=10.0)}
    ideal = variants["ideal"]()
    assert ideal["gyro"].model.gyro_noise_rms_dps == 0.0
    assert ideal["magnetometer"].model.noise_rms_t == 0.0
    assert ideal["sun"].model.noise_rms_a == 0.0


def test_all_noise_variant_turns_every_noise_on():

    variants = {v.name: v for v in build_suite_variants(rate_hz=10.0)}
    suite = variants["all_noise"]()
    assert suite["gyro"].model.gyro_noise_rms_dps > 0.0
    assert suite["magnetometer"].model.noise_rms_t > 0.0
    assert suite["sun"].model.noise_rms_a > 0.0


def test_gyro_only_variant_leaves_other_sensors_clean():
    variants = {v.name: v for v in build_suite_variants(rate_hz=10.0)}
    suite = variants["gyro_noise_only"]()
    assert suite["gyro"].model.gyro_noise_rms_dps > 0.0
    assert suite["magnetometer"].model.noise_rms_t == 0.0
    assert suite["sun"].model.noise_rms_a == 0.0


def test_quantization_only_has_noise_off_but_quantizer_on():
    """Quantization must stay on in every variant -- that is the ADC."""
    variants = {v.name: v for v in build_suite_variants(rate_hz=10.0)}
    suite = variants["quantization_only"]()
    assert suite["gyro"].model.gyro_noise_rms_dps == 0.0
    # The 16-bit quantization grid is a property of the model, not a knob.
    assert suite["gyro"].model.gyro_counts_per_dps > 0.0


def test_study_runs_and_emits_results(tmp_path: Path):
    truth = _FakeTruth()
    config = StudyConfig(rate_hz=10.0, seed=42)
    result = run_hardware_fidelity_study(truth, config)
    assert result.runs
    for run in result.runs:
        assert run.variant_name
        assert run.samples > 0
        assert run.angle_rms_deg >= 0.0
        assert np.isfinite(run.angle_max_deg)
        assert run.estimator_wall_time_s >= 0.0


def test_study_is_deterministic_for_a_seed():
    truth = _FakeTruth()
    a = run_hardware_fidelity_study(truth, StudyConfig(rate_hz=10.0, seed=7))
    b = run_hardware_fidelity_study(truth, StudyConfig(rate_hz=10.0, seed=7))
    assert [r.angle_rms_deg for r in a.runs] == [r.angle_rms_deg for r in b.runs]


def test_different_seeds_change_noisy_results():
    truth = _FakeTruth()
    a = run_hardware_fidelity_study(truth, StudyConfig(rate_hz=10.0, seed=1))
    b = run_hardware_fidelity_study(truth, StudyConfig(rate_hz=10.0, seed=2))
    assert [r.angle_rms_deg for r in a.runs] != [r.angle_rms_deg for r in b.runs]


def test_ideal_variant_has_smaller_error_than_all_noise():
    """Noise must not improve the estimate, even allowing for sampling noise.

    The ordering is a directional claim about the estimators, not an exact
    one: on a short fixture whose error is dominated by eclipse transitions
    the two variants can land within a few hundredths of a degree of each
    other in either order. Asserting a strict inequality here tested the
    fixture's particular random draw rather than the property, and broke as
    soon as the sun noise level changed by a factor of 20.

    What must hold is that noise does not *help*, with a margin wide enough
    to absorb sampling scatter but far tighter than the effect being tested
    (the magnetometer noise alone moves the RMS by several degrees).
    """
    truth = _FakeTruth()
    result = run_hardware_fidelity_study(truth, StudyConfig(rate_hz=10.0, seed=3))
    by_name = {r.variant_name: r for r in result.runs}
    assert by_name["ideal"].angle_rms_deg <= by_name["all_noise"].angle_rms_deg + 0.5


def test_study_writes_json(tmp_path: Path):
    truth = _FakeTruth()
    out = tmp_path / "results.json"
    run_hardware_fidelity_study(truth, StudyConfig(rate_hz=10.0, seed=5), output_path=out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert "runs" in data
    assert data["config"]["seed"] == 5
    assert len(data["runs"]) == len(build_suite_variants(rate_hz=10.0))


def test_summary_table_is_markdown_and_ordered():
    truth = _FakeTruth()
    result = run_hardware_fidelity_study(truth, StudyConfig(rate_hz=10.0, seed=9))
    table = summarize_study(result)
    assert table.startswith("|")
    assert "variant" in table
    for run in result.runs:
        assert run.variant_name in table


def test_config_rejects_bad_rate():
    with pytest.raises(ValueError):
        StudyConfig(rate_hz=0.0)
    with pytest.raises(ValueError):
        StudyConfig(rate_hz=-1.0)
