"""Shared interface for the bare-bones ideal sensor models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pocketadcs_sim.truth_environment import TruthEnvironmentHistory

FloatArray = npt.NDArray[np.float64]


def sample_history(truth: TruthEnvironmentHistory, sample_rate_hz: float) -> FloatArray:
    """Return the sampled times for a sensor rate, using nearest truth samples."""
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    times = np.arange(0.0, truth.time_s[-1] + 1e-9, 1.0 / sample_rate_hz)
    # Nearest truth sample for each requested time.
    left = np.searchsorted(truth.time_s, times, side="left")
    candidate = np.clip(left, 0, len(truth.time_s) - 1)
    # Compare the sample at `left` with the one before it (if it exists).
    before = np.clip(left - 1, 0, len(truth.time_s) - 1)
    pick_left = np.abs(truth.time_s[before] - times) < np.abs(truth.time_s[candidate] - times)
    indices = np.where(pick_left, before, candidate)
    return np.clip(indices, 0, len(truth.time_s) - 1)


class IdealSensorModel(ABC):
    """A noise-free sensor reporting the true body-frame quantity it sees."""

    def __init__(self, sample_rate_hz: float = 1.0) -> None:
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        self.sample_rate_hz = float(sample_rate_hz)

    @abstractmethod
    def truth_quantity(self, truth: TruthEnvironmentHistory) -> FloatArray:
        """Return the full-history body-frame truth quantity for this sensor."""

    def measure(self, truth: TruthEnvironmentHistory) -> FloatArray:
        """Return sampled true measurements at the sensor sampling rate."""
        indices = sample_history(truth, self.sample_rate_hz)
        return np.asarray(self.truth_quantity(truth)[indices], dtype=float)
