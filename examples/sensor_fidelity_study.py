"""Run the sensor-fidelity ablation study against a real truth environment.

Emits ``results/sensor_fidelity/<tag>/results.json`` and prints the summary
table. Every run records its config, seed and environment in the JSON, so a
table in the thesis can be traced back to the exact numbers that produced it.

Examples::

    # One orbit, deterministic.
    python examples/sensor_fidelity_study.py --duration 5400 --rate 10

    # Several seeds at once, for a confidence spread on the noisy variants.
    python examples/sensor_fidelity_study.py --seeds 0 1 2 3 4

    # Sweep the sampling rate.
    python examples/sensor_fidelity_study.py --rates 1 5 10 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pocketadcs_sim.dynamics import InitialState, SimulationConfig, SpacecraftConfig
from pocketadcs_sim.study import StudyConfig, run_hardware_fidelity_study, summarize_study
from pocketadcs_sim.truth_environment import (
    TruthEnvironmentConfig,
    run_truth_environment,
)

RESULTS_ROOT = Path("results") / "sensor_fidelity"


def build_truth(duration_s: float, step_s: float = 0.5):
    """A representative LEO truth run for the study."""
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
    return run_truth_environment(
        vehicle,
        initial,
        SimulationConfig(duration_s=duration_s, step_s=step_s),
        TruthEnvironmentConfig(epoch_utc="2025-01-01T00:00:00", gravity_degree=4),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=1800.0, help="truth run length, s")
    parser.add_argument("--step", type=float, default=0.5, help="truth integrator step, s")
    parser.add_argument("--rate", type=float, default=10.0, help="sensor sample rate, Hz")
    parser.add_argument("--rates", type=float, nargs="*", default=None, help="sweep several rates")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0], help="seeds to run")
    parser.add_argument("--tag", default=None, help="output subdirectory name")
    args = parser.parse_args()

    rates = args.rates if args.rates else [args.rate]
    tag = args.tag or f"d{int(args.duration)}_r{'-'.join(str(r) for r in rates)}"
    out_dir = RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"building truth environment ({args.duration:.0f} s, step {args.step} s) ...")
    truth = build_truth(args.duration, args.step)
    print(f"  {len(truth.time_s)} truth samples")

    manifest = []
    for rate in rates:
        for seed in args.seeds:
            config = StudyConfig(rate_hz=float(rate), seed=seed)
            label = f"rate{rate:g}_seed{seed}"
            result = run_hardware_fidelity_study(
                truth, config, output_path=out_dir / f"{label}.json"
            )
            print(f"\n=== rate {rate:g} Hz, seed {seed} ===")
            print(summarize_study(result))
            manifest.append(
                {
                    "label": label,
                    "rate_hz": float(rate),
                    "seed": seed,
                    "file": f"{label}.json",
                }
            )

    (out_dir / "manifest.json").write_text(
        json.dumps({"tag": tag, "runs": manifest}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
