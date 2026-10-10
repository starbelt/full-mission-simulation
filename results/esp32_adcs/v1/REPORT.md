# ESP32 ADCS hardware acceptance — partial, blocked

Task: t_ae7823e4. Actual ESP32-S3 on COM5 (USB 303A:1001; MAC 7c:4f:ad:0a:88:3c). COM7 was not accessed. No firmware rebuild/modification, host physical-plant modification, or git commit was made.

## Provenance / flash

Approved manifest: results/zephyr_adcs/v1/build_manifest.json.
Build ID: e6b87a559797bc135e9ec7ad8a2d36acf33f005a7b0a2e302f61bf36925309c4.
27 manifest source hashes plus both flash image hashes matched before flashing. Flash used esptool 5.4.0, watchdog-reset, exact retained images, no flash-parameter overrides. Both esptool write hash verifications passed. Complete readback hashes matched:

- MCUboot @0x0, 37344 bytes: e1b0a4ee16de5083bc2094677c41a72e53c96c919916babe8770e68777312306.
- Application @0x20000, 188204 bytes: 2efaace3049efc60f56c7fecc9b42dff88213c75c0cb1b061c18253f028bbc56.

Live XI exactly matched manifest identity, including 240000000 Hz counter, 320-byte line capacity and 64 segments. Read-only idle heartbeat observed. Bringup XH: uptime23916ms, drops0, overflows0, headroom14732 bytes.

## Completed physical case

Unchanged approved target3d configuration, accepted TruthPlant / WheelBoard, raw gyro/magnetic/Sun body measurements plus absolute epoch and position. No truth attitude or inertial references transmitted. Device wheel commands were applied to the accepted board and plant. All host/firmware fixture source hashes remained unchanged.

- 6001 unique TX XS2 / Y replies / physical samples, seq1..6001; complete strict joins audited.
- Simulated mission duration300s; observed wall duration171.812s including environment preparation. This was unpaced lockstep closed-loop, NOT a 20Hz paced demonstration.
- Final physical pointing error0.0075581127842401456deg against commanded final target.
- Controller input equals emitted estimator quaternion on all samples. Maximum estimator/controller input versus simultaneous pre-actuation truth0.002693425149325349deg.
- All status0, has_fix1, stale0, reject0. Wheel acceleration torque cap and speed limit checks pass; max wheel speed by axis2.3522171,23.4183945,12.1693884rad/s, below1675.5160819rad/s bound.
- XH before, every1000 frames and after case: no RX drops/overflows, strictly increasing uptime, minimum observed main-thread headroom5068 bytes. No boot/reset messages observed.

Measured on target CCOUNT, converted with live XI counter frequency:

| Stage, ms | p50 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: |
| Ephemeris | 11.769088 | 11.792275 | 11.799338 | 12.195863 |
| Estimator | 0.549350 | 0.560950 | 0.562688 | 0.565100 |
| Controller | 0.103808 | 0.106917 | 0.107238 | 0.127942 |
| Allocation | 0.058671 | 0.060863 | 0.061142 | 0.061604 |
| Sum of instrumented stages | 12.479667 | 12.504608 | 12.513083 | 12.811979 |

No instrumented-stage sum exceeded50ms. Full core/control-loop execution time is UNAVAILABLE: firmware counters exclude reference-profile evaluation, parsing, formatting and transport. Therefore this is not complete deadline acceptance. Target timings are not native-host timings.

Serial RTT p50/p95/p99/max31/32/32/32ms from retained timestamped transport. These are COARSE: this Python interpreter's monotonic clock uses GetTickCount64 with15.625ms resolution. They must not be interpreted as precision latency measurements.

## Blocking defect and bounded reproduction

After completing target3d, the host closed and reopened COM5 with DTR/RTS false before opening. XI still matched. Fresh approved full SlewOS plan was accepted (XC, XL, both XS, XE all status0), but its first raw sensor frame at mission time0 returned `E EST 2 1 1`: retained estimator time rejected the new mission's time origin. No Y frame and no physical SlewOS sample were produced. Planned duration2681.974849858644s /53641 frames was NOT completed or shortened.

A separate bounded diagnostic confirmed XZ did not repair this: XH before345461ms, XZ ACK`E XZ 0 -1 1`, accepted reconfiguration/plan, then same first XS2 returned`E EST 2 1 1`. XH after345488ms, drops0, overflows0, stack5068 bytes. Target remained live without reset. Diagnostic is transport/session evidence, not pointing evidence.

Source localization: examples/hil_closed_loop_run.py:283 loads configuration/mission but no explicit estimator session lifecycle. adcs_protocol.h:540 XC updates config only; :750 XZ clears plan stores but leaves estimator, timestamp, has_last_sample, wheel bookkeeping and use_reference. runControlCycle:387 calls estimator.propagate, which refuses time regression. The harness creates a fresh physical world for each mission while the persistent device retains old mission state. A separate corrective task must define safe same-mission reconfiguration versus fresh isolated-HIL-session reset semantics; do not silently reset physical wheel momentum on live plan changes.

## Not completed / disposition

Stopped as task directed when hardware exposed a defect. No workaround reboot, source edit, firmware rebuild or alternative plant was used. Full SlewOS, eclipse, biased/noisy, bounded20Hz streaming, malformed/stale-frame recovery, overload/recovery and legacyS2 tests remain pending. Biased-case ownership discrimination has not been demonstrated on target. Normal stack/performance evidence applies only to the completed3D case. Hardware acceptance is BLOCKED, not PASS; no flight-certification claim.

## Reproduction / artifacts

From repository root:

    PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/Scripts/python.exe results/esp32_adcs/v1/benchmark.py

Benchmark requires a fresh boot/session and unused output directories; it refuses to overwrite case directories. It exercised the completed case then stopped on failure. Do not rerun over existing evidence. On the currently retained post-case target state:

    PYTHONPATH=src .venv/Scripts/python.exe results/esp32_adcs/v1/reproduce_reconfigure.py

Diagnostic also refuses overwrite of its raw output. Offline audit (no serial access):

    PYTHONPATH=src .venv/Scripts/python.exe results/esp32_adcs/v1/audit.py

Offline audit completed successfully; 114 focused existing adapter/harness/protocol-pin/Zephyr-integration tests passed in34.55s. No project sources were edited.

Evidence: flash.log; readback_boot/app.bin and logs; readback_verification.json; bringup_wire.jsonl; benchmark.log; target3d/accepted_run/{manifest.json,wire.jsonl,samples.jsonl,summary.json}; target3d/{transport.jsonl,metrics.json}; slewos/accepted_run partial raw artifacts; slewos/{transport.jsonl,metrics.json}; reconfigure_diagnostic.jsonl. Compact metrics.csv, audit.json and artifact_hashes.json accompany this report. All failed/unavailable metrics remain null/absent, not fabricated zeros. Target left live on approved image; COM5 released.
