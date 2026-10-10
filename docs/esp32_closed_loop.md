# ESP32-S3 closed-loop ADCS (firmware v1)

The shared closed-loop core is integrated into the Zephyr ESP32-S3 target and
host-tested; this build-only stage has not flashed or validated that port on
hardware. The firmware implements the closed-loop attitude determination and
control loop.
The host simulates the orbit, the sensor hardware and the spacecraft's
rotational dynamics; the device computes the reference vectors, the attitude,
the control torque and the wheel commands, and the host scores the result
against truth it never sent.

## Why a new protocol, and not the existing S2

The legacy `S2` / `R` exchange is stateless and already has a firmware
regression suite and an operator soak behind it on COM5. Closed-loop control is
stateful -- it needs a plan, an initial estimate, rotor momentum, and a
time base -- and reusing `S2`'s tag for that would leave those results
describing behaviour their names do not imply. So closed loop is a **separate,
versioned protocol family** and the legacy path is byte-for-byte unchanged.

Both are answered on the same port. Neither is repurposed.

| | legacy (unchanged) | closed loop (v1) |
|---|---|---|
| frame | `S2` | `XS2` |
| reply | `R` | `Y` |
| state | none | plan, estimate, wheel momentum |
| timing | none (stateless) | per-frame, rejects regression |
| wheels | not commanded | torque and saturation reported |

## The ownership boundary

This is the part that decides whether any accuracy number means anything.

**The device computes** — everything in this column has no field to travel in,
because the protocol has no room for it:

- the inertial Sun and magnetic reference vectors, from its own GCRS position
  and clock (`EphemerisReferenceProvider`: IGRF-14 + Meeus),
- the TRIAD attitude solution,
- propagation between fixes, and the decision to reject a fix,
- the MRP attitude error, the PD control law, the torque clip,
- the wheel allocation and its speed and acceleration limits,
- the safety logic (stale gap, no fix, over-long line, plan transaction).

**The host provides** — a stand-in for reality, and never an answer:

- position, gyro, magnetometer, sun sensor (with hardware error applied),
- the plan: gains, limits, and a schedule to fly. A plan is an *instruction*,
  not a state. No truth attitude, no truth rate, no precomputed reference
  vector is transmitted, and the harness has a test asserting the frame
  serializer never mentions one.

**The host alone knows** the plant — spacecraft rotational dynamics. It is not
simulated on the device, because it is not an ADCS function.

Small error alone establishes neither correctness nor truth leakage. Validate
ownership at the executed controller seam, including deliberately biased/noisy
sensors and eclipse, and compare the physical plant to the final commanded
target. The corrected Sun epoch/frame model is independently tested; the old
0.3-degree residual was a coordinate-frame defect, not an accuracy floor.

## Wire format

All lines are ASCII, newline-terminated. Numbers are plain decimal; the device
parses closed-loop fields with bounded `strtod` reads. The 320-byte line buffer
accepts at most 319 ASCII payload bytes plus NUL (newline is not stored).
The framing boundary rejects embedded NUL, non-ASCII bytes, DEL and other
control bytes (except tab, CR preserved as space and terminating LF). CR must
never join split operator/numeric tokens: `X\rZ` is not `XZ`, and `.3\r5` is
not `.35`. Each CR consumes a payload byte; LF is the only line delimiter. A malformed line is
drained through LF and refused once without dispatching its prefix or suffix;
the next valid line is processed normally. Binary refusal is not a line-length
overflow and does not reset cumulative RX health.

### Configuration

```
XC kp kd max_torque_nm accel_cap_rad_s2 rotor_inertia_kg_m2 max_speed_rad_s
```

Non-positive gains, limits or inertia are refused. Refusing is loud: a bad
configuration must not silently become a controller that cannot act.

### Loading a plan (two-step transaction)

```
XL initial_angle_rad declared_segments
XS objective_id start_angle_rad target_angle_rad duration_s accel_s coast_s decel_s
  (repeat)
XE
```

Up to 64 segments are supported; 65 is refused without replacing the live plan.
Committed and pending schedules reside in a static session, not on the main
thread stack. Nothing is committed until `XE`, and `XE` refuses unless exactly
`declared_segments` segments arrived — so a mis-ordered or truncated plan cannot
half-apply. `XL 0 0` opens a reference-profile transaction instead (below).

Segments are contiguous: the device assigns each segment's start time from the
durations before it, which is what makes a reordered plan detectable at `XE`.

### Loading a 3D reference profile

```
XL 0 0
XR qStart[4] qTarget[4] total_s
XE
```

The device interpolates start to target over `total_s` with a smoothstep ease
and controls to that. This is **an externally supplied profile, not planner
output**: the implemented FA01 engine is single-axis rest-to-rest, so 3D
pointing is exercised with a profile from outside and recorded as such.

### Fresh isolated experiment versus live mission updates

`XZ` is the explicit boundary for a fresh isolated HIL experiment. Send it only
when the host intentionally initializes a new physical plant and wheel-board
stand-in. A validated `XZ` clears estimator attitude, timestamp history,
absolute-fix/rejection/covariance state, control-input validity, sample/stale
state, wheel-speed bookkeeping, active schedule/reference (including
`use_reference`), both pending transactions and their counters. It preserves
validated configuration, stale-gap configuration, external sensor calibration,
platform cycle source, and cumulative real uptime/RX/stack health. It resets in
place, without rebooting or copying a large `Session` onto the main stack.

The ACK must be exactly `E XZ 0 -1 0`. Extra arguments or suffixes are refused
without changing mission state. After reset, a plan/profile must be committed
and a new absolute fix acquired; gyro-only data cannot inherit a previous fix
or actuate against a prior mission's estimate. Repeating `XZ` is idempotent.
The physical adapter sends and validates `XZ` before `XC` on each `run_loop`,
because each call constructs a fresh host world. A failed reset ACK stops before
configuration or sensor transmission. No timestamp rebasing is used.

`XC`, `XL`/`XS`/`XE` and `XR`/`XE` are ordinary live updates, not fresh-session
initializers. They preserve estimator time/state, wheel momentum and control
history. Within a mission, duplicate or regressed sample time remains refused,
including duplicates/regressions following a stale sample, before state changes.
Do not use `XZ` for an in-flight plan/gain change or as a dropout recovery.

Migration is image-manifest gated: the historical v1 image had plan-only `XZ`
and is not compatible with the corrected persistent-session lifecycle. Preserve
its evidence; v2 review found a malformed binary-reset framing bypass, and
v3 was withdrawn after an internal CR token-joining reset alias was reproduced.
Review and approve the corrected v5 measurement source and flash-artifact
manifest (v4 lifecycle/framing remains the behavioral baseline),
then have the designated hardware owner replace the acceptance manifest. A
successful build does not authorize flashing or prove hardware acceptance.

### The control frame

```
XS2 seq epoch_ms t_s gyro[3] mag[3] sun[3] pos_gcrs_m[3]
```

15 numbers. `epoch_ms` is absolute Unix milliseconds (the onboard ephemeris
needs a real date; "seconds since start" would force a mission-epoch constant to
stay in sync on two sides). `pos_gcrs_m` is read as double — float32 resolves
about 0.5 m at LEO radius, which shows up directly as reference error. An
all-zero `sun` means "no sun measurement" (eclipse), which is not the same as a
real direction, and the device treats it as absent.

### The reply

```
Y status seq has_fix q[4] ctrl_q[4] tau[3] u[3] saturated stale reject cyc[4] [service_cycles]
```

- `q` is the attitude the device **reported**; `ctrl_q` is the attitude the
  controller **used**. They must be identical. They are reported separately so a
  truth leak into the control path is visible rather than assumed away — the
  host test asserts equality against the device's own output.
- `cyc` preserves the four original hardware stage deltas, ordered ephemeris,
  estimator, controller and allocation. Do not sum these and call the result
  complete frame service time: parsing, profile evaluation and formatting are
  outside those intervals.
- v5 Zephyr appends one optional trailing `service_cycles` field to a fully
  formatted `Y` (including no-fix/stale Y frames). Original/native/historical
  `Y` has 25 whitespace-separated tokens; the instrumented target has 26.
  Missing service is **unavailable (`None`/JSON null)**, never inferred or zero
  filled. An explicit measured zero remains zero. Stage and service fields are
  finite, integral uint32 values, 0 through 4294967295. The host validates exact
  decimal values before float rounding; fractional, negative, overflow,
  non-finite or extra fields are refused.
- The service start is the first `k_cycle_get_32()` after a successful dequeue,
  before uptime/stack/RX health sampling, command routing and `handleLine`.
  The stop is immediately after `handleLine` returns, including bounded numeric
  parsing, schedule/reference-profile evaluation, onboard ephemeris, gyro and
  absolute estimation, controller, allocation/bookkeeping and original Y reply
  formatting. This is **elapsed on-device queued-command service**, not a sum
  of stages or a pure algorithm-only CPU-time metric. Counter reads and any
  ISR/scheduler preemption during the interval are included.
- Excluded: UART input/framing/ISR assembly and enqueue; queue wait and dequeue
  copy; appended metric formatting/copy and its capacity validation; reply
  newline and polling UART/USB transport; subsequent idle/heartbeat processing.
  The append uses a separate 12-byte suffix and copies only after validating the
  would-be length and remaining capacity; an insufficient buffer emits the
  existing `E SHORT 4 -1 0` instead of transmitting a partial Y.
- Both deltas use Xtensa CCOUNT via Zephyr `k_cycle_get_32()`, with unsigned
  subtraction permitting a wrap only when the interval is shorter than 2^32
  cycles. `XI` reports the actual counter frequency (240000000 Hz in this build;
  the interval limit is approximately 17.9 seconds at that frequency). The host
  converts each stage and service count independently as `cycles * 1e6 / hz`,
  using validated XI, never an assumed CPU frequency. No v5 target runtime
  measurements or performance/rate ceiling are claimed until authorized hardware
  acceptance. Legacy `S2/R`, ACK, XI and XH shapes are unchanged.
- The physical adapter writes a separate `timing.jsonl` exchange journal, with
  integer `host_exchange_rtt_ns` from Windows high-resolution `perf_counter_ns`
  (clock resolution recorded in the run manifest). The span is immediately
  before host send through reply reception, including host I/O, decode, reply
  logging and any skipped idle heartbeats. Deadline management still uses
  `monotonic()`. RTT is **transport/application latency**, never algorithm time.
  A serial link queries/strictly validates XI before collecting converted cycle
  metrics. Native callback/stub cycles remain test-only: the journal leaves
  their stage/service counts and times null even if a test target emits them.
  Native and historical service are not hardware performance evidence.

### Diagnostics

```
XI                  identity: I 1 source_sha256 counter_hz line_bytes max_segments
XH                  health: H status uptime_ms queue_put_failures line_overflows headroom_bytes has_fix
XZ                  explicit fresh isolated HIL mission reset (not live reconfiguration)
```

`XH` reads actual main-thread unused stack bytes through
`k_thread_stack_space_get`, with INIT_STACKS and THREAD_STACK_INFO enabled.
This is a historical high-watermark, not a current stack-pointer measurement;
-1 explicitly means unavailable/error. Health is sampled before dispatching a
line, so it includes previous work, not that line's peak. After draining the
last frame, explicitly issue `XH` to capture the final processing high-watermark
and RX counters. Hardware runtime values remain unverified until flashing.
Queue failures and line-overflow drops are separate atomic ISR counters; a
successful queue put never increments the queue-failure count. An overflowing
line is refused as a whole, and assembly recovers at its newline. Counters are
uint32 modulo values. `XD`, `XO` and `XGS <uptime>` setters still exist for host
protocol tests but are refused on the hardware target: they cannot forge health.

Timing boundaries are ephemeris reference callback; gyro propagation/absolute
fix; estimate echo plus PD controller; allocation plus open-loop speed
bookkeeping. Desired plan/profile evaluation, parsing, reply formatting and USB
transport are outside these intervals; do not sum them and call it total frame
execution time. Native-surrogate timing is unavailable, not ESP32 performance.

## Safety behaviour, and why each exists

| Situation | Device behaviour | The failure it prevents |
|---|---|---|
| No absolute fix yet | torque 0, status reports it | acting on an initial guess |
| Sample gap over threshold | **do not propagate**; report stale, torque 0 | integrating 30 s of stale gyro into a large confident error |
| Eclipse (zero sun) | reject the fix, keep the last fix | a TRIAD fix on an undefined direction |
| Reference pair parallel | reject the fix | a degenerate rotation matrix |
| Line longer than the buffer | refuse it whole | truncation parsing as a *different valid frame* |
| Field count or parse error | refuse, advance nothing | partial state from a bad line |
| Plan segment count mismatch at `XE` | refuse, keep the old plan | a half-applied schedule |
| Wheel speed or acceleration at the bound | clamp the increment, not the command | exceeding a real rotor limit |
| Torque above the limit | clip to the limit, report saturation | a command the hardware cannot execute |

The stale-gap row is the subtle one. Propagating across a dropout is the
obvious implementation and it is wrong: it multiplies a stale, possibly biased
gyro by tens of seconds. If a TRIAD fix happens to land afterwards it hides the
damage, so the defect survives into a run that looks clean — which is why the
test for it builds a case where the fix *cannot* land.

Rotor momentum is preserved across a suppressed cycle, never zeroed. Dropping
it would turn a comms gap into a full attitude excursion as the stored momentum
silently disappears.

## What is not implemented

- **No motor hardware.** Wheel speeds are open-loop bookkeeping of what the
  device commanded, integrated from the commanded torque. There is no sensed
  wheel speed on this board; the host's rotor stand-in holds the real state.
- **No IMU driver.** Gyro, magnetometer and sun readings arrive over serial.
  `HilStreamSource` in `zephyr/src/main.cpp` is the single line where real I2C or
  ADC drivers would be substituted.
- **No fault detection or thruster mode.** Reaction wheels only.

## Running it

```bash
# Build (stages sources outside the space-containing repo path first)
bash esp32/build_zephyr.sh

# Flash
bash esp32/flash_zephyr.sh COM5

# Host harness: streams frames, scores attitude error against withheld truth
PYTHONPATH=src .venv/Scripts/python.exe -m examples.hil_closed_loop_run --port COM5 \
  --output-dir results/hil_hardware/new_version --duration 300
```

Exit code is the verdict: 0 pass, 2 a check failed, 1 the run could not be
executed. See `examples/hil_stream.py` for the legacy harness's equivalent.

## Host tests

| File | What it pins |
|---|---|
| `tests/test_adcs_core_differential.py` | C++ core vs the PC prototype, value by value, including rejection and allocation |
| `tests/test_adcs_protocol_host.py` | the protocol: transactions, refusals, stale gaps, exactly-once, health |
| `tests/test_hil_closed_loop_harness.py` | the harness's scoring, incl. the false PASSes that have actually occurred |
| `tests/test_hil_closed_loop_protocol_pin.py` | harness parser against real firmware output, and the ownership property |

The first two compile `esp32/src/adcs_core.h` and `esp32/src/adcs_protocol.h` on
the host with `-Wall -Wextra -Werror`, so the code under test is the code that
ships.
