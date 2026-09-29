"""Tests for the attitude accuracy / performance metric computations."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pocketadcs_sim.metrics import (
    AngleErrorStats,
    LatencyStats,
    compute_angle_error_stats,
    compute_latency_stats,
    quaternion_rotation_error_deg,
    quaternions_from_mrp,
)


class TestQuaternionHelpers:
    def test_quaternions_from_mrp_shape(self):
        mrp = np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.05]])
        quats = quaternions_from_mrp(mrp)
        assert quats.shape == (2, 4)
        assert np.allclose(np.linalg.norm(quats, axis=1), 1.0)

    def test_quaternions_from_mrp_single_vector(self):
        quats = quaternions_from_mrp(np.array([0.0, 0.0, 0.0]))
        assert quats.shape == (1, 4)

    def test_quaternions_from_mrp_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            quaternions_from_mrp(np.zeros((3, 2)))

    def test_quaternion_rotation_error_is_zero_for_identical(self):
        mrp = np.array([0.1, -0.2, 0.05])
        quat = quaternions_from_mrp(mrp.reshape(1, 3))
        err = quaternion_rotation_error_deg(quat[0], quat[0])
        assert err == pytest.approx(0.0, abs=1e-9)

    def test_quaternion_rotation_error_matches_known_angle(self):
        """A rotation of exactly 30 deg about z must report 30 deg."""
        truth = Rotation.from_euler("z", 30.0, degrees=True).as_quat()
        identity = np.array([0.0, 0.0, 0.0, 1.0])  # SciPy [x, y, z, w]
        err = quaternion_rotation_error_deg(truth, identity)
        assert err == pytest.approx(30.0, abs=1e-9)

    def test_quaternion_rotation_error_is_sign_insensitive(self):
        q = Rotation.from_euler("x", 42.0, degrees=True).as_quat()
        assert quaternion_rotation_error_deg(q, -q) == pytest.approx(0.0, abs=1e-9)

    def test_quaternion_rotation_error_rejects_nonunit(self):
        with pytest.raises(ValueError):
            quaternion_rotation_error_deg(
                np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])
            )

    def test_quaternion_rotation_error_range(self):
        """The error is the full rotation angle, never above 180 deg."""
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        for angle in (10.0, 90.0, 170.0, 180.0):
            a = Rotation.from_euler("y", angle, degrees=True).as_quat()
            err = quaternion_rotation_error_deg(a, identity)
            assert err == pytest.approx(angle, abs=1e-9)
            assert 0.0 <= err <= 180.0


class TestAngleErrorStats:
    def test_known_constant_error(self):
        stats = compute_angle_error_stats(np.full(10, 2.0))
        assert isinstance(stats, AngleErrorStats)
        assert stats.mean_deg == pytest.approx(2.0)
        assert stats.rms_deg == pytest.approx(2.0)
        assert stats.max_deg == pytest.approx(2.0)
        # A perfectly constant error has no scatter, so the 3-sigma band
        # collapses to zero; the bias is carried by mean/rms, not sigma.
        assert stats.sigma3_deg == pytest.approx(0.0)
        assert stats.std_deg == pytest.approx(0.0)
        assert stats.count == 10

    def test_sigma3_is_three_times_sample_std(self):
        errs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        stats = compute_angle_error_stats(errs)
        assert stats.sigma3_deg == pytest.approx(3.0 * np.std(errs, ddof=1))

    def test_empty_is_nan_not_crash(self):
        stats = compute_angle_error_stats(np.zeros(0))
        assert stats.count == 0
        assert np.isnan(stats.mean_deg)
        assert np.isnan(stats.rms_deg)

    def test_rms_uses_population_division(self):
        errs = np.array([0.0, 4.0])
        stats = compute_angle_error_stats(errs)
        # RMS of [0,4] is sqrt(8) regardless of ddof.
        assert stats.rms_deg == pytest.approx(np.sqrt(8.0))

    def test_std_dev_is_sample_based(self):
        errs = np.array([1.0, 2.0, 3.0, 4.0])
        stats = compute_angle_error_stats(errs)
        assert stats.std_deg == pytest.approx(np.std(errs, ddof=1))

    def test_rejects_negative_errors(self):
        with pytest.raises(ValueError):
            compute_angle_error_stats(np.array([-1.0, 1.0]))

    def test_rejects_nonfinite(self):
        with pytest.raises(ValueError):
            compute_angle_error_stats(np.array([1.0, np.nan]))

    def test_to_dict_is_json_safe(self):
        import json

        stats = compute_angle_error_stats(np.array([1.0, 2.0]))
        json.dumps(stats.to_dict(), allow_nan=False)

    def test_quantiles_present(self):
        stats = compute_angle_error_stats(np.arange(100, dtype=float))
        assert stats.p50_deg <= stats.p95_deg <= stats.p99_deg
        assert stats.p99_deg <= stats.max_deg


class TestLatencyStats:
    def test_basic_stats(self):
        stats = compute_latency_stats(np.array([0.01, 0.02, 0.03]))
        assert isinstance(stats, LatencyStats)
        assert stats.count == 3
        assert stats.mean_ms == pytest.approx(20.0)
        assert stats.max_ms == pytest.approx(30.0)
        assert stats.p95_ms >= stats.p50_ms

    def test_empty_latency(self):
        stats = compute_latency_stats(np.zeros(0))
        assert stats.count == 0
        assert np.isnan(stats.mean_ms)

    def test_rejects_negative_latency(self):
        with pytest.raises(ValueError):
            compute_latency_stats(np.array([-0.1]))

    def test_to_dict_is_json_safe(self):
        import json

        json.dumps(compute_latency_stats(np.array([0.01])).to_dict(), allow_nan=False)


class TestEndToEnd:
    def test_perfect_estimator_has_zero_error(self):
        truth_mrp = np.array([[0.1, -0.2, 0.05], [0.11, -0.19, 0.06]])
        est_quats = quaternions_from_mrp(truth_mrp)
        truth_quats = quaternions_from_mrp(truth_mrp)
        errors = np.array(
            [
                quaternion_rotation_error_deg(e, t)
                for e, t in zip(est_quats, truth_quats, strict=True)
            ]
        )
        # MRP->quaternion->angle round-trip is exact to floating-point
        # precision, not to the last bit.
        assert np.allclose(errors, 0.0, atol=1e-5)

    def test_noise_increases_error(self):
        """A perturbation of the estimated attitude increases the error."""
        rng = np.random.default_rng(0)
        truth_quat = quaternions_from_mrp(np.array([0.1, -0.2, 0.05]))[0]
        # An estimate that exactly equals the truth: zero error by construction.
        exact = quaternion_rotation_error_deg(truth_quat, truth_quat)
        # The same attitude nudged by a small random rotation: larger error.
        perturbed = (
            Rotation.from_quat(truth_quat)
            * Rotation.from_rotvec(rng.normal(0, 0.02, 3))
        ).as_quat()
        nudged = quaternion_rotation_error_deg(perturbed, truth_quat)
        assert exact < 1e-5
        assert nudged > exact
        assert 0.0 < nudged < 10.0

    def test_shape_mismatch_raises(self):
        # (3, 2) is not a valid (N, 3) MRP array.
        with pytest.raises(ValueError):
            quaternions_from_mrp(np.zeros((3, 2)))
        with pytest.raises(ValueError):
            quaternions_from_mrp(np.zeros((2, 3, 3)))
