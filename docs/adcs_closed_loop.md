# PC-first closed-loop ADCS prototype

A desktop (PC) closed-loop attitude determination and control prototype for
PocketADCS: gyro-propagating attitude estimation, a quaternion feedback
controller, and a three-wheel reaction-wheel board stand-in built from real
TGR 29-12 rotor parameters. Firmware and hardware are untouched by this work.

## Reproduction

```bash
cd "C:/Users/drago/Nextcloud/School - NextCloud/Research/pocketadcs-sim"

# Fresh project setup (includes declared AHRS and serial dependencies)
uv sync --locked

# Tests for every module in the prototype
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/ -q -k adcs

# The executable example (writes real JSON + CSV)
PYTHONPATH=src .venv/Scripts/python.exe examples/closed_loop_adcs.py
PYTHONPATH=src .venv/Scripts/python.exe examples/closed_loop_adcs.py --noisy --eclipse

# Independent cross-check of the plant against Basilisk
PYTHONPATH=src .venv/Scripts/python.exe examples/validate_plant_vs_basilisk.py
```

Outputs land in `results/adcs_closed_loop/v1/` (a new versioned directory; no
existing schema or output directory is modified).

## Architecture and the truth boundary

```
TruthPlant  --(body rate)-->  SyntheticSensors  --(gyro + 2 vectors)-->  Estimator
    ^                                                                       |
    |                                                                       v
    +----(wheel torque)---- WheelBoard <----(torque cmd)---- AttitudeController
```

| Module | File | Role |
| --- | --- | --- |
| Quaternions | `adcs/quaternions.py` | Scalar-first Hamilton quaternions, `C_BN` convention, exact exponential propagation |
| Rotors | `adcs/rotors.py` | TGR 29-12 record, three-wheel set, acceleration cap |
| Plant | `adcs/plant.py` | Truth rigid body + three momentum wheels (truth) |
| Sensors | `adcs/closed_loop.py` | `SyntheticSensors` — gyro rate + body-frame sun/mag readings |
| Estimator | `adcs/estimator.py` | Gyro propagation + TRIAD absolute fix (onboard-like) |
| Controller | `adcs/controller.py` | PD on attitude error + estimated body rate |
| Board | `adcs/wheel_board.py` | Wheel-command executor with limits and honest reporting |
| SlewOS adapter | `adcs/slewos_adapter.py` | Read-only reader of real FA01 planner results |
| Loop | `adcs/closed_loop.py` | Wires the above together and scores the result |

**Truth is read in exactly two places:** initializing the plant's physical
state, and computing errors *after* the loop has run. The controller is driven
by the estimated attitude and the *measured* gyro rate. This is enforced by
interface, not convention:

* `SyntheticSensors` exposes no method returning truth attitude or truth rate.
* `OnboardAttitudeEstimator` has no parameter through which truth could enter.
* `AttitudeController.compute` takes an estimate, not a truth state.

That structural argument is not evidence, so it is also **measured**. Every
step, the attitude the controller actually received is compared against the
plant's truth attitude and reported in `truth_cross_check` and in the
`controller_input_vs_truth_deg` time-series column. The comparison reads
`AttitudeController.last_estimated_attitude` — the controller's own record of
its argument, copied on the way in — not the caller's snapshot, so it measures
the value a torque was really computed from.

Three tests hold this down, and all three fail if truth enters the loop:

* `test_controller_is_handed_the_estimate_and_not_the_plant_attitude` — wraps
  the estimator, controller and plant, samples all three at the same point in
  the loop, and asserts the controller's argument is the estimator's attitude
  bit for bit while the two are measurably far apart.
* `test_loop_follows_a_biased_estimate_rather_than_the_plant` — rotates every
  sensor fix by a known 10°. The vehicle then settles 10° from the commanded
  target, because the loop is closed through the estimate. A loop touching truth
  would still converge on the target.
* `test_result_reports_a_measured_truth_cross_check` — the recorded
  measurement exists, is non-zero where the two genuinely differ, and the
  unearned `truth_leakage` / `truth_used_for` fields are gone.

**Reading the cross-check.** In `ideal` runs it reads **0.0**, which is the
expected result and not a sign of leakage: the TRIAD fix is exact at the sample
instant, so the estimate handed to the controller equals truth to machine
precision, and divergence appears only across the following plant step — the
post-step estimator error in the same run is larger (0.0096° for the 3D case).
It becomes non-zero as soon as the fix is imperfect: 0.54° (noisy SlewOS) and
1.52° (noisy 3D with eclipse). An unexplained 0.0 invites "fixing" the
measurement, so `test_ideal_run_cross_check_is_zero_because_the_fix_is_exact_at
_the_sample` pins the correct reading.

A hardcoded `truth_leakage: false` used to appear in all four JSON artifacts. It
was a dataclass default no code path ever assigned, so the test asserting it
compared `False` with `False` — a check that passes exactly when the thing it
exists to prevent has happened. A constant in a result file reads to any
downstream consumer as a verified measurement.

A suspiciously good result would be a bug report, not a success, so it is worth
being precise about what the ideal run does and does not exercise. The ideal
gyro carries **no** bias and no noise: `SyntheticSensors.measure` adds both only
in the `noisy` branch, and the ideal run's `max_estimate_error_deg` is 0.0,
including through a commanded eclipse. That zero is the expected floor of a
noise-free TRIAD loop, exactly as the cross-check paragraph above says -- not
evidence that the propagation path is untested. The gyro-only path is genuinely
exercised by the estimator unit tests, which inject their own bias;
`test_gyro_only_drift_grows_without_becoming_the_truth` pins that an un-aided
estimate must drift rather than match truth. The `noisy` run is the one in which
the absolute fix has real work to do.

## Conventions (explicit, because the sign errors here are silent)

* Quaternions are **scalar-first** Hamilton, `[w, x, y, z]`.
* `to_dcm(q)` returns **`C_BN`**, the transpose of the active rotation matrix.
* For `C_BN` and a **body** rate: `C_dot = -[omega]x C`, so a body rotation by
  `+theta` about `+Z` is `C = Rz(-theta)`, and the rotation increment
  **multiplies on the right** of the quaternion. Left-multiplying is correct
  only at the identity attitude — the mistake that motivated
  `test_propagation_composes_correctly_from_a_nonzero_attitude`.
* Rotor speed `Omega_i` is **relative to the body**, so
  `I_eff = I + sum J g g^T` and the hub momentum is `I_eff omega + sum J Omega g`.
* Spinning a rotor up applies **`-u_i`** to the hub, so a desired body torque
  `t` is commanded as wheel torque `u = -t`.
* Total body momentum is `I_eff omega + sum_i J Omega_i g_i`; the gyroscopic
  term is `-omega x (I_eff omega) - omega x (sum_i J Omega_i g_i)`.

## Estimator

Gyro propagation under the exact quaternion exponential, bridged by TRIAD
absolute fixes. TRIAD is delegated to the vetted `ahrs.filters.TRIAD`
reference (matching `pocketadcs_sim/attitude/triad.py`) rather than
re-derived.

Validity handling:

* A **zero-length vector is "no data"** (eclipse), never a real direction — the
  two states must not share an encoding.
* A parallel sun/mag pair makes TRIAD singular and is skipped, not forced.
* A non-monotonic timestamp **raises**, rather than integrating the gyro
  backwards and silently diverging.

Reference vectors are supplied per update rather than defaulted, because a
TRIAD fix measures the *angle between* the two vectors; pairing a measurement
with the wrong reference would give a confident wrong attitude.

## Controller

`tau = Kp * e - Kd * omega_est`, where `e` is the MRP of the error rotation.
A positive error axis produces a positive torque on that axis; a mirrored
controller still settles, just the wrong way, so the direction is asserted
explicitly and by a closed-loop convergence test.

## Rotor configuration (TGR 29-12 Rev B, p.1 Motor Parameter Table)

| Parameter | Value | Note |
| --- | --- | --- |
| Rotor inertia | 3.27e-6 kg·m² | rotor only; no extra flywheel |
| Rotor mass | 0.028 kg | |
| Max speed | 16000 rpm = 1675.5 rad/s | |
| Continuous torque (vacuum) | **0.0028 N·m** | the column actually used |
| Continuous torque (atmosphere) | 0.0135 N·m | kept in the record, *not* used |
| Acceleration cap | 200 rad/s² | **engineering parameter** |

The acceleration cap is **not** a datasheet quantity and **not** a proven
no-stall limit. It is ~23% of the torque-implied maximum
(`0.0028 / 3.27e-6 = 857 rad/s²`), and the record carries
`acceleration_cap_is_datasheet_claim: False`. The datasheet gives no stall
torque, so nothing here claims the cap prevents stall. The torque-implied
maximum is reported alongside it for comparison.

Both the atmosphere and vacuum columns are retained in the output record
because choosing the wrong one is a silent ~5x error in control authority.

## Board stand-in

Executes local wheel-torque commands against continuous rotor state. Limits
are applied in order:

1. **Acceleration cap** bounds `u/J` for the step.
2. **Speed bound** blocks torque that would push a wheel further out, but
   **still allows braking** — clipping symmetrically would freeze an overspeeding
   wheel that could not recover.

A `None` command models a **command timeout**: torque goes to zero but rotor
speed is **preserved** (the wheels coast). Zero-filling wheel state on timeout
would turn a comms dropout into a full attitude excursion by silently deleting
stored momentum.

The board reports **commanded vs applied torque separately**, so a limit is
distinguishable from an actuator fault.

## Planner scope — what exists and what does not

The implemented SlewOS/FA01 planner is **single-axis, rest-to-rest** sequencing
of scalar slew angles. There is **no multi-axis planner**. This is reported in
every output record via `PlannerScope`, and the adapter *refuses* a multi-axis
case rather than flattening it onto one axis, since accepting it would imply a
planner that does not exist.

The adapter reads real planner results from the sibling repo (read-only):

* `manifest.json` — per-objective `target_angle_rad` (in FA01 study roots the
  manifest lives in a per-worker `n<N>` subdirectory while `raw.jsonl` sits at
  the study root; the adapter locates both).
* `raw.jsonl` — the chosen `order` and the per-move `durations_s` / `profiles_s`.

It never invents a plan: a missing case, an infeasible record, or a missing
required field raises `AdapterError`. There is no fallback schedule and no
locally computed substitute timing.

The angle-vs-time reference is the planner's own symmetric trapezoid, built from
`profiles_s`. Such a profile covers `v_peak * (duration - (accel + decel) / 2)`,
so it is that **effective time** — not the raw duration — that normalises the
profile. Normalising by the duration instead rescales the whole move: the
commanded angle steps at the coast/decel junction, and the implied angular
acceleration is not the `alpha` the planner optimized. For real `a2-00000` the
two denominators differ by 1e-6 rad, but 5.1% of the 930,507 real FA01 segments
exceed a 0.1% error and the worst is 50%, so the distinction is a property of
the corpus, not of this one case.

The prototype therefore runs two clearly distinguished cases:

| Case | Schedule source | Meaning |
| --- | --- | --- |
| `slewos_*` | **real FA01 planner result** | one-axis rest-to-rest tracking |
| `target_3d_*` | **externally supplied** smoothstep reference | 3D pointing; *not* planner output |

## Measured results (ideal sensors)

Real SlewOS plan `a2-00000` / `DO` from `fa01_a_v2` — order `[1, 0]`, 30.0° →
25.6° → 59.5°, planned duration 2681.97 s:

| Quantity | Value |
| --- | --- |
| Final attitude error (vs reference) | 0.0266° |
| Max attitude error (vs reference) | 0.0266° |
| **Final attitude error vs the planner's final angle** | **0.0270°** |
| Final estimator-vs-truth error | 0.0008° |
| Max \|wheel speed\| | 2.556 rad/s of 1675.5 rad/s bound (0.15%) |
| Saturated steps | 0 |

The run lasts the full planned maneuver: when a schedule is loaded, the plan's
duration sets the run length, so "tracks the plan" is a statement about the
whole maneuver and not about its first 300 s.

3D target (externally supplied schedule), start 38.4° from target:

| Quantity | Value |
| --- | --- |
| Final attitude error (vs reference) | 0.0064° |
| **Final attitude error vs the commanded target** | **0.0064°** |
| Max attitude error (vs reference) | 0.3208° |
| Max \|wheel speed\| | 23.4 rad/s |
| Saturated steps | 0 |

`max_attitude_error_deg` of 0.32° is the loop's own overshoot during the
slew; the 38.4° start-to-target separation is the *commanded* maneuver, not
tracking error, and the number that says whether the vehicle pointed at the
target is `plant.final_attitude_error_to_target_deg` — scored from the plant's
final truth attitude against `target_attitude`, independently of the
reference the loop was given.

The reference is pinned by its own tests, not only by the loop tracking it:
`reference_attitude(config, 0)` equals the start attitude and
`reference_attitude(config, total)` equals the target to 1e-12, with the
error monotonically decreasing in between. A reference that ended somewhere
else would let the plant track it perfectly and report a tiny error while
never reaching the commanded attitude.

Tolerances are appropriate to ideal sensors: with perfect TRIAD fixes the
residual measures the **loop**, not sensor error. The `--noisy` run adds gyro
bias/noise and sensor noise on top; ADS corrections are not tuned away.

Noisy run (`--noisy --eclipse`, ideal shown first, as above): the SlewOS plan
finishes 0.0601° from the planner's final angle (max 0.0993°, wheels 18.4 rad/s,
51 saturated steps), and the 3D case finishes 0.0339° from the target, degrading
to 1.4553° through the commanded eclipse and recovering after it.

## Validation

The plant is cross-checked against Basilisk's own
`ReactionWheelStateEffector` (`examples/validate_plant_vs_basilisk.py`): same
inertias, same torque, same initial rotor speeds, and a non-identity initial
attitude, over 20 s.

* `max |domega| = 1.5e-05 rad/s`
* `max |dOmega| = 4.4e-02 rad/s`

The wheel-speed residual is **Basilisk's** error: this plant matches the
closed-form integral `Omega = Omega_0 + (u/J)t` to every printed digit
(254.64831804), while Basilisk's drift from that integral grows linearly with
step count at a constant relative ~1.8e-4 — its effector uses a first-order
rotor update, and halving `dt` does not shrink it. The script therefore reports
a band and explains why, rather than asserting agreement with a discretization
error.

Numerics: the closed loop uses a fixed-step RK4 plant update
(`TruthPlant.step_rk4`) because a 45-minute schedule at 50 ms is ~54k steps and
the adaptive `solve_ivp` path costs ~3 ms each.
`tests/test_adcs_plant_fast.py` pins RK4 to the adaptive solver (1e-12) and
confirms 4th-order convergence; at the control step its attitude error is
already ~6e-13 over 20 s.

## Test coverage

`tests/test_adcs_*.py`, 129 collected tests:

* **quaternions** — known-axis propagation, non-commuting rate changes, sign
  invariance, 180° conditioning, MRP round trips.
* **rotors** — datasheet values pinned; tests fail if the torque column or the
  `is_datasheet_claim` flag is flipped (verified by mutation).
* **plant** — momentum conservation, gyroscopic coupling, analytic wheel
  solution, RK4 vs adaptive.
* **wheel_board** — acceleration cap, speed bound, braking while overspeed,
  timeout preserving wheel state, per-wheel saturation reporting.
* **estimator** — TRIAD fix, gyro-only drift, eclipse coast and recovery, zero
  vector and parallel-pair rejection, non-monotonic time, no truth attribute.
* **controller** — error direction, rate damping, clipping, a closed-loop
  convergence test that pins the loop sign, and that `last_estimated_attitude`
  records the argument actually received (overwritten per call, and copied so a
  caller cannot rewrite it through a reused buffer).
* **slewos_adapter** — real plan read end-to-end, planner scope honesty,
  multi-axis refusal, no fallback, and that the sibling repo is never written.
* **slewos_profile** — the tracked reference against *independent* references,
  never against the implementation's own expression: numerical integration of
  the velocity ramp, continuity at the coast/decel junction, exact endpoints, the
  triangular (coast-free) profile FA01 really emits, the angular acceleration
  implied by `angle_at` equal to the planner's `alpha`, a degenerate profile
  refused rather than divided by, and every real segment of an actual FA01 plan
  checked against the sibling repo's own `physics.trapezoid_timing`. Reverting
  the normaliser to `duration_s` fails 13 of these 17; swapping `accel` and
  `decel` in either ramp branch survives every symmetric profile, so an
  asymmetric one is pinned too.
* **closed_loop** — SlewOS tracking within tolerance over the *whole* planned
  maneuver, 3D target, the reference profile's own endpoints, eclipse, wheel
  bounds, determinism, real JSON/CSV output, and the three truth-ownership
  checks described above (seam identity, biased-sensor end-to-end, measured
  cross-check).
