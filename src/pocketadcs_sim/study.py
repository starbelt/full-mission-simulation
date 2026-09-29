"""Sensor-fidelity ablation study: how much error does each sensor add?

Purpose
-------
Answer one question defensibly: **where does the attitude-determination error
come from?** The study runs the same TRIAD estimator over the same truth
history under a small set of sensor configurations that differ only in which
error sources are enabled, so each source can be attributed a number.

Design
------
The matrix is deliberately small -- five variants, not a combinatorial
sweep -- because the useful result is an *attribution*, and attribution
needs each source isolated, not every combination enumerated:

=========================  ======  ======  =====
variant                    gyro   mag     sun
=========================  ======  ======  =====
``ideal``                  -       -       -
``quantization_only``      quant  quant    quant
``gyro_noise_only``        noise  -        -
``mag_noise_only``         -       noise    -
``sun_noise_only``         -       -        noise
``all_noise``              noise  noise    noise
=========================  ======  ======  ======

``ideal`` is the noise-free, zero-bias reference. ``quantization_only``
isolates the ADC grid (16-bit IMU, 18-bit MMC5983MA) from analog noise, which
matters because quantization error is a *deterministic floor* while noise is
a *random* term. The single-sensor variants then attribute error to one
sensor each, and ``all_noise`` gives the total.

Every variant is a real hardware-faithful model, not a stub: the harness and
the estimator see the same code path they would in flight.

Reproducibility
---------------
All randomness derives from ``StudyConfig.seed`` through
``build_hardware_sensor_suite``, so a rerun with the same seed reproduces the
same numbers exactly. Results are written as JSON (``allow_nan=False``, so
the file is valid for any consumer) with the full config recorded alongside
the numbers, which is what makes a table citable in a thesis appendix.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy

from pocketadcs_sim.attitude.triad import TriadEstimator
from pocketadcs_sim.metrics import compute_angle_error_stats
from pocketadcs_sim.sensors import build_hardware_sensor_suite
from pocketadcs_sim.truth_environment import TruthEnvironmentHistory

# Datasheet-representative noise levels used by the ablation. The gyro and
# magnetometer figures are the datasheet RMS noise at their default
# full-scale settings; the photodiode figure is an engineering value
# (the array's datasheet specifies no RMS noise), so the sun variant is
# explicitly labelled as an assumption in the results.
GYRO_NOISE_DPS = 0.01
MAG_NOISE_T = 0.6e-6

#: Sun array analog noise, rms in amps. This is an ENGINEERING ASSUMPTION,
#: not a datasheet figure: the VEMD8080 specifies irradiance response and dark
#: current but no output-referred RMS noise, and no transimpedance gain is
#: stated either. The value is therefore a readout-chain assumption and must
#: be quoted as one.
#:
#: It also has to be small compared with the DARK CURRENT for the array's own
#: validity logic to work. The reconstruction rejects a sample whose lit-face
#: count exceeds 3; an unlit face sits at the 0.2 nA dark floor, so noise much
#: above that makes unlit faces read as lit and every sample is discarded.
#: At 1 uA (noise/dark = 5000) that happens even in full sun, and the whole
#: orbit was scored as invalid -- a 93 deg mean error that is a rejected-sample
#: artifact, not a sensor result.
#:
#: 0.05 uA keeps the noise ~250x the dark floor (so the dark current still
#: matters and the unlit faces stay unlit) while spanning ~20 ADC codes at a
#: 10 uA full scale, so it is a real, measurable noise term.
SUN_NOISE_A = 0.05e-6

#: Full-scale range of the sun array's analog front end. A VEMD8080 channel
#: delivers roughly 0.2-1.2 uA of photocurrent in full sun, so a 10 uA range
#: (with the model's default 0.9 headroom) puts full sun just under clip and
#: leaves the noise about 20 codes wide. Stating the range as 1 A instead --
#: the old default -- put the noise ~8 orders of magnitude below one 12-bit
#: code, so the grid rounded it away and the ``sun_noise_only`` ablation row
#: measured nothing at all. Units, not a sensor property.
SUN_FULL_SCALE_A = 1.0e-5

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StudyConfig:
    """Knobs for one study run."""

    rate_hz: float = 10.0
    seed: int | None = 0
    gyro_noise_rms_dps: float = GYRO_NOISE_DPS
    mag_noise_rms_t: float = MAG_NOISE_T
    sun_noise_rms_a: float = SUN_NOISE_A

    def __post_init__(self) -> None:
        if not np.isfinite(self.rate_hz) or self.rate_hz <= 0.0:
            raise ValueError("rate_hz must be a finite positive value")


@dataclass(frozen=True, slots=True)
class SuiteVariant:
    """One row of the ablation matrix.

    ``build`` takes an optional :class:`StudyConfig`; called with no
    argument it uses the default configuration, which is what makes a
    variant easy to inspect in isolation.
    """

    name: str
    description: str
    build: Callable[..., dict[str, Any]]

    def __call__(self, config: StudyConfig | None = None) -> dict[str, Any]:
        return self.build(config or StudyConfig())


@dataclass(frozen=True, slots=True)
class RunResult:
    """Metrics for one variant of the ablation."""

    variant_name: str
    description: str
    samples: int
    valid_samples: int
    invalid_samples: int
    angle_mean_deg: float
    angle_rms_deg: float
    angle_max_deg: float
    angle_p95_deg: float
    angle_sigma3_deg: float
    estimator_wall_time_s: float
    per_sample_us: float


@dataclass(frozen=True, slots=True)
class StudyResult:
    """All variants plus the environment they were produced in."""

    config: StudyConfig
    runs: list[RunResult]
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "environment": self.environment,
            "runs": [asdict(run) for run in self.runs],
        }


def _environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def build_suite_variants(rate_hz: float = 10.0) -> list[SuiteVariant]:
    """The ablation matrix.

    Every variant is a closure over ``rate_hz`` so the caller can build the
    same matrix at any sampling rate.
    """
    def _clean(config: StudyConfig) -> dict[str, Any]:
        return build_hardware_sensor_suite(
            rate_hz=rate_hz, gyro_noise_rms_dps=0.0, mag_noise_rms_t=0.0, sun_noise_rms_a=0.0
        )

    def _quantization(config: StudyConfig) -> dict[str, Any]:
        # Noise off; the ADC grid is intrinsic to each model and stays on.
        return _clean(config)

    def _gyro(config: StudyConfig) -> dict[str, Any]:
        return build_hardware_sensor_suite(
            rate_hz=rate_hz,
            gyro_noise_rms_dps=config.gyro_noise_rms_dps,
            mag_noise_rms_t=0.0,
            sun_noise_rms_a=0.0,
            seed=config.seed,
        )

    def _mag(config: StudyConfig) -> dict[str, Any]:
        return build_hardware_sensor_suite(
            rate_hz=rate_hz,
            gyro_noise_rms_dps=0.0,
            mag_noise_rms_t=config.mag_noise_rms_t,
            sun_noise_rms_a=0.0,
            seed=config.seed,
        )

    def _sun(config: StudyConfig) -> dict[str, Any]:
        return build_hardware_sensor_suite(
            rate_hz=rate_hz,
            gyro_noise_rms_dps=0.0,
            mag_noise_rms_t=0.0,
            sun_noise_rms_a=config.sun_noise_rms_a,
            seed=config.seed,
        )

    def _all(config: StudyConfig) -> dict[str, Any]:
        return build_hardware_sensor_suite(
            rate_hz=rate_hz,
            gyro_noise_rms_dps=config.gyro_noise_rms_dps,
            mag_noise_rms_t=config.mag_noise_rms_t,
            sun_noise_rms_a=config.sun_noise_rms_a,
            seed=config.seed,
        )

    return [
        SuiteVariant("ideal", "noise-free reference (ADC grid still active)", _clean),
        SuiteVariant("quantization_only", "ADC quantization only, no analog noise", _quantization),
        SuiteVariant("gyro_noise_only", "ICM-40609-D gyro noise only", _gyro),
        SuiteVariant("mag_noise_only", "MMC5983MA magnetometer noise only", _mag),
        SuiteVariant("sun_noise_only", "VEMD8080 array noise only (assumed level)", _sun),
        SuiteVariant("all_noise", "all three sensor noise sources together", _all),
    ]


def _run_variant(
    truth: TruthEnvironmentHistory,
    config: StudyConfig,
    variant: SuiteVariant,
) -> RunResult:
    suite = variant.build(config)
    sun_reading = suite["sun"].measure(truth)
    mag_reading = suite["magnetometer"].measure(truth)

    estimator = TriadEstimator(sample_rate_hz=config.rate_hz)
    start = time.perf_counter()
    output = estimator.estimate_from_readings(
        truth, sun_body=sun_reading, mag_body=mag_reading
    )
    elapsed = time.perf_counter() - start

    errors = output["error_deg"]
    valid = output["valid"]
    good = np.asarray(errors[valid], dtype=float)
    stats = compute_angle_error_stats(good)
    samples = int(len(errors))
    per_sample_us = (elapsed / samples * 1e6) if samples else float("nan")

    return RunResult(
        variant_name=variant.name,
        description=variant.description,
        samples=samples,
        valid_samples=int(valid.sum()),
        invalid_samples=int(samples - valid.sum()),
        angle_mean_deg=stats.mean_deg,
        angle_rms_deg=stats.rms_deg,
        angle_max_deg=stats.max_deg,
        angle_p95_deg=stats.p95_deg,
        angle_sigma3_deg=stats.sigma3_deg,
        estimator_wall_time_s=float(elapsed),
        per_sample_us=float(per_sample_us),
    )


def run_hardware_fidelity_study(
    truth: TruthEnvironmentHistory,
    config: StudyConfig | None = None,
    output_path: Path | None = None,
) -> StudyResult:
    """Run the full ablation matrix over one truth history.

    Args:
        truth: the truth environment history to run against.
        config: study knobs; defaults to :class:`StudyConfig`.
        output_path: if given, write the results as JSON. NaN is not valid
            JSON, so a variant with no valid samples is written with its
            NaN fields set to null rather than a non-standard token.

    Returns:
        A :class:`StudyResult` holding one :class:`RunResult` per variant.
    """
    config = config or StudyConfig()
    variants = build_suite_variants(rate_hz=config.rate_hz)
    runs = [_run_variant(truth, config, variant) for variant in variants]
    result = StudyResult(config=config, runs=runs, environment=_environment())

    if output_path is not None:
        payload = _json_safe(result.to_dict())
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return result


def _json_safe(value: Any) -> Any:
    """Replace NaN/Inf with None so the file is strict-JSON valid."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def summarize_study(result: StudyResult) -> str:
    """Render the study as a markdown table for a report or thesis."""
    header = (
        "| variant | valid | RMS err (deg) | mean (deg) | p95 (deg) | max (deg) | us/sample |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = []
    for run in result.runs:
        rows.append(
            f"| {run.variant_name} | {run.valid_samples}/{run.samples} "
            f"| {run.angle_rms_deg:.4f} | {run.angle_mean_deg:.4f} "
            f"| {run.angle_p95_deg:.4f} | {run.angle_max_deg:.4f} "
            f"| {run.per_sample_us:.1f} |"
        )
    return "\n".join([header, *rows])
