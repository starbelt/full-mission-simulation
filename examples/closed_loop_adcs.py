"""Run the PC-first closed-loop ADCS prototype and write real measurements.

Two runs, both written to disk as JSON + CSV:

1. **SlewOS** -- tracks a *real* FA01 planner result read from the sibling
   ``reaction-wheel-tradespace`` repo (single-axis rest-to-rest; see
   :class:`~pocketadcs_sim.adcs.slewos_adapter.PlannerScope` for what the
   planner does and does not support).
2. **3D target** -- tracks a nontrivial 3D attitude with an *externally
   supplied* reference profile. This exercises 3D pointing and is deliberately
   not presented as planner output.

Run::

    PYTHONPATH=src .venv/Scripts/python.exe examples/closed_loop_adcs.py
    PYTHONPATH=src .venv/Scripts/python.exe examples/closed_loop_adcs.py --noisy
    PYTHONPATH=src .venv/Scripts/python.exe examples/closed_loop_adcs.py \\
        --slewos-run <path> --case a2-00000 --method DO

``--noisy`` adds gyro bias/noise and sensor noise so the ADS correction is
visible under error; the default ideal run shows the loop's own floor first.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from pocketadcs_sim.adcs import quaternions as quat
from pocketadcs_sim.adcs.closed_loop import ClosedLoopConfig, run_closed_loop
from pocketadcs_sim.adcs.rotors import DATASHEET_SOURCE, TGR2912
from pocketadcs_sim.adcs.slewos_adapter import SLEWOS_DEFAULT_PATH, PlannerScope

OUTPUT_ROOT = Path("results") / "adcs_closed_loop" / "v1"


def find_slewos_run() -> Path | None:
    """Return a real FA01 study A run directory, or None if absent."""
    candidates = sorted(
        glob.glob(str(SLEWOS_DEFAULT_PATH / "results" / "fa01" / "fa01_a*" / "raw.jsonl"))
    )
    return Path(candidates[-1]).parent if candidates else None


def describe_rotors() -> dict[str, object]:
    """Return the rotor record, including which torque column is in use."""
    return TGR2912.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy", action="store_true", help="add sensor and gyro error")
    parser.add_argument("--slewos-run", default=None, help="FA01 run directory")
    parser.add_argument("--case", default="a2-00000", help="planner case id")
    parser.add_argument("--method", default="DO", help="planner method")
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_ROOT), help="where to write JSON/CSV"
    )
    parser.add_argument(
        "--eclipse", action="store_true", help="include a mid-run eclipse window"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "noisy" if args.noisy else "ideal"
    scope = PlannerScope.current()

    print("PC-first closed-loop ADCS prototype")
    print(f"  sensors            : {mode}")
    print(f"  rotor source       : {DATASHEET_SOURCE}")
    print(f"  torque in use      : {TGR2912.continuous_torque_nm} N.m (vacuum)")
    print(f"  accel cap          : {TGR2912.acceleration_cap_rad_s2} rad/s^2 "
          "(engineering parameter, NOT a datasheet/stall claim)")
    print(f"  planner scope      : {scope.planner_support}")
    print()

    rotor_record = describe_rotors()
    (output_dir / "rotor_configuration.json").write_text(
        json.dumps(rotor_record, indent=2), encoding="utf-8"
    )

    runs: dict[str, object] = {}

    # --- 1. Real SlewOS plan -------------------------------------------------
    run_dir = Path(args.slewos_run) if args.slewos_run else find_slewos_run()
    if run_dir is not None:
        slew_config = ClosedLoopConfig(
            slewos_run_dir=str(run_dir),
            slewos_case_id=args.case,
            slewos_method=args.method,
            sensors=mode,
            output_dir=str(output_dir),
            tag=f"slewos_{args.case}_{args.method}_{mode}",
        )
        slew_result = run_closed_loop(slew_config)
        runs["slewos"] = slew_result.to_dict()
        print("SlewOS one-axis rest-to-rest plan (planner generated)")
        print(f"  run dir            : {run_dir}")
        print(f"  order              : {slew_result.planner['order']}")
        print(f"  planned duration   : {slew_result.planner['total_duration_s']:.1f} s")
        print(f"  final error        : {slew_result.error['final_attitude_error_deg']:.4f} deg")
        print(f"  max error          : {slew_result.error['max_attitude_error_deg']:.4f} deg")
        print(f"  error vs plan end  : "
              f"{slew_result.plant['final_attitude_error_to_target_deg']:.4f} deg")
        print(f"  max |wheel speed|  : {slew_result.wheels['max_abs_wheel_speed_rad_s']:.3f} "
              f"rad/s of {slew_result.wheels['max_speed_bound_rad_s']:.1f} rad/s bound")
        print(f"  saturated steps    : {slew_result.wheels['saturated_steps']:.0f}")
        print()
    else:
        print("SlewOS run skipped: no FA01 study A results found.")
        print(f"  looked under       : {SLEWOS_DEFAULT_PATH}")
        print()

    # --- 2. 3D target, externally supplied schedule --------------------------
    target = quat.from_axis_angle(np.array([0.3, 0.8, 0.5]), 0.7)
    three_d_config = ClosedLoopConfig(
        target_attitude=target,
        start_attitude=quat.from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.3),
        duration_s=300.0,
        sensors=mode,
        eclipse_windows=((100.0, 200.0),) if args.eclipse else (),
        output_dir=str(output_dir),
        tag=f"target_3d_{mode}" + ("_eclipse" if args.eclipse else ""),
    )
    three_d_result = run_closed_loop(three_d_config)
    runs["target_3d"] = three_d_result.to_dict()
    print("3D target attitude (EXTERNALLY SUPPLIED schedule, not planner output)")
    print(f"  final error        : {three_d_result.error['final_attitude_error_deg']:.4f} deg")
    print(f"  max error          : {three_d_result.error['max_attitude_error_deg']:.4f} deg")
    print(f"  error vs target    : "
          f"{three_d_result.plant['final_attitude_error_to_target_deg']:.4f} deg")
    if "max_attitude_error_deg_during_eclipse" in three_d_result.error:
        print(f"  error in eclipse   : "
              f"{three_d_result.error['max_attitude_error_deg_during_eclipse']:.4f} deg")
    print(f"  max |wheel speed|  : "
          f"{three_d_result.wheels['max_abs_wheel_speed_rad_s']:.3f} rad/s")
    print(f"  saturated steps    : {three_d_result.wheels['saturated_steps']:.0f}")
    print()

    summary = {
        "sensors": mode,
        "planner_scope": scope.planner_support,
        "rotor_configuration": rotor_record,
        "runs": runs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=list), encoding="utf-8"
    )
    print(f"Wrote JSON + CSV to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
