"""Tests for running TRIAD on externally supplied sensor readings.

``TriadEstimator.estimate`` pulls its measurements straight out of the truth
history, which is fine for a noise-free sanity check but cannot exercise a
sensor model. These tests pin a method that accepts explicit measured
vectors, which is what lets the ablation study feed the estimator through the
same hardware models that produce the telemetry.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pocketadcs_sim.attitude.triad import TriadEstimator
from pocketadcs_sim.sensors.base import sample_history


class _FakeTruth:
    """A self-consistent truth history: body vectors are the GCRS
    reference vectors rotated into the body frame by the truth attitude.

    Consistency matters here: if the body vectors simply equalled the GCRS
    ones while the truth MRP claimed a rotated attitude, TRIAD would
    correctly return a solution that disagrees with the (impossible) truth,
    and a noise-free run would show a nonzero error for reasons that have
    nothing to do with the code under test.
    """

    def __init__(self, n=32, dt=0.5):

        self.time_s = np.arange(n) * dt
        ang = np.linspace(0.0, 0.5, n)
        self.angular_velocity_body_rad_s = np.stack(
            [np.cos(ang), np.sin(ang), 0.2 * np.ones(n)], axis=1
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

        # A varying truth attitude, so the study is not a trivial identity.
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


def _sampled(truth, rate_hz=5.0):
    """Readings resampled to match what the estimator will consume."""
    idx = sample_history(truth, rate_hz)
    return (
        truth.sun_direction_body[idx],
        truth.magnetic_field_body_t[idx],
        truth.sun_direction_gcrs[idx],
        truth.magnetic_field_gcrs_t[idx],
    )


def test_estimate_from_readings_shapes():
    truth = _FakeTruth()
    est = TriadEstimator(sample_rate_hz=5.0)
    sun, mag, _, _ = _sampled(truth)
    out = est.estimate_from_readings(truth, sun_body=sun, mag_body=mag)
    n_expected = len(sample_history(truth, 5.0))
    assert out["dcm"].shape == (n_expected, 3, 3)
    assert out["error_deg"].shape == (n_expected,)
    assert out["time_s"].shape == (n_expected,)


def test_noise_free_readings_give_zero_error():
    truth = _FakeTruth()
    est = TriadEstimator(sample_rate_hz=5.0)
    sun, mag, _, _ = _sampled(truth)
    out = est.estimate_from_readings(truth, sun_body=sun, mag_body=mag)
    # The DCMs agree to ~1e-16, but `_angle_error_deg` evaluates
    # arccos((tr-1)/2) which loses precision as the trace approaches 3.
    # That sets a numerical floor of order 1e-6 deg -- orders of magnitude
    # below any real sensor error, so it is the floor to assert against.
    assert out["error_deg"].max() < 1e-4


def test_perturbed_sun_increases_error():
    truth = _FakeTruth()
    est = TriadEstimator(sample_rate_hz=5.0)
    sun, mag, _, _ = _sampled(truth)
    clean = est.estimate_from_readings(truth, sun_body=sun, mag_body=mag)
    rng = np.random.default_rng(0)
    noisy = sun + rng.normal(0, 0.01, sun.shape)
    noisy = noisy / np.linalg.norm(noisy, axis=1, keepdims=True)
    dirty = est.estimate_from_readings(truth, sun_body=noisy, mag_body=mag)
    assert dirty["error_deg"].max() > clean["error_deg"].max()


def test_zero_sun_vector_is_skipped_not_crashed():
    """Eclipse must not produce a NaN attitude; the sample is dropped."""
    truth = _FakeTruth()
    est = TriadEstimator(sample_rate_hz=5.0)
    sun, mag, _, _ = _sampled(truth)
    sun = sun.copy()
    sun[0] = 0.0  # eclipse sample
    out = est.estimate_from_readings(truth, sun_body=sun, mag_body=mag)
    assert np.all(np.isfinite(out["error_deg"][out["valid"]]))
    assert out["valid"].sum() == len(out["error_deg"]) - 1
    assert np.isnan(out["error_deg"][0])


def test_parallel_vectors_are_skipped():
    """Sun and mag parallel is a degenerate TRIAD case, not a crash."""
    truth = _FakeTruth()
    est = TriadEstimator(sample_rate_hz=5.0)
    sun, mag, _, _ = _sampled(truth)
    sun, mag = sun.copy(), mag.copy()
    mag[1] = sun[1] * 1e-5  # parallel -> degenerate
    out = est.estimate_from_readings(truth, sun_body=sun, mag_body=mag)
    assert np.all(np.isfinite(out["error_deg"][out["valid"]]))
    assert out["valid"].sum() == len(out["error_deg"]) - 1


def test_length_mismatch_raises():
    truth = _FakeTruth()
    est = TriadEstimator(sample_rate_hz=5.0)
    sun, mag, _, _ = _sampled(truth)
    with pytest.raises(ValueError):
        est.estimate_from_readings(truth, sun_body=sun[:-1], mag_body=mag[:-1])


def test_nonfinite_input_raises():
    truth = _FakeTruth()
    est = TriadEstimator(sample_rate_hz=5.0)
    sun, mag, _, _ = _sampled(truth)
    sun = sun.copy()
    sun[0, 0] = np.nan
    with pytest.raises(ValueError):
        est.estimate_from_readings(truth, sun_body=sun, mag_body=mag)
