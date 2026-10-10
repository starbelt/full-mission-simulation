# ESP32 ADCS — independent acceptance review

## Verdict

**Approved for the bounded synthetic-sensor hardware-in-the-loop scope.** The independently validated PC physical model and configurations were reused unchanged. The approved v5 firmware is flashed to COM5 and responding; COM7 was not touched. This is not flight qualification, a real-sensor test, or proof that the firmware can never hang.

The parent independently executed `parent_review.py`, rejoined every primary transmitted frame/reply/physical sample/timing record, verified all 80 worker artifact hashes and all 22 immutable v5 source/image pins, reran 65 focused timing/integration tests, and ran the v5 verifier covering eight complete native physical missions plus raw ISR framing/state proofs. Full-length flash readbacks match the approved bootloader and application hashes.

## Hardware results

| Test | Frames | Simulated duration | Final physical target error | Maximum device service |
|---|---:|---:|---:|---:|
| Nontrivial 3-D maneuver | 6,001 | 300 s | 0.007558° | 14.369 ms |
| Complete SlewOS schedule | 53,641 | 2,681.97485 s | 0.028462° | 13.944 ms |
| Gyro-only eclipse interval | 6,001 | 300 s | 0.007558° | 13.952 ms |
| Deliberately biased/noisy sensors | 6,001 | 300 s | 1.010198° | 13.955 ms |

Total: **71,644 exactly-once primary frames**, with zero normal-run drops/overflows, no unexpected reboot, and no manual reset between missions. The three nominal cases meet the <0.1° endpoint criterion. The biased/noisy case deliberately fails nominal accuracy and matches the diagnostic baseline; it is not advertised as nominal success.

The complete dispatch service maximum was **14.368983 ms**, measured by the ESP32 cycle counter and its validated 240 MHz identity. That is below the 50 ms compute budget for 20 Hz. This elapsed measurement includes interrupt/preemption inside the dispatch but excludes ISR framing, queue wait, suffix/TX and transport. Host exchange round-trip latency and individual stage times remain separate in `metrics.csv`.

Primary missions were unpaced lockstep, not full-duration wall-paced 20 Hz tests. A separate **601-frame, 30-second actual 20 Hz** run measured a 30.0005436 s transmit span and passed without losses. Minimum observed main-thread stack headroom was **5,084 bytes**.

## Reliability and lifecycle evidence

- Target main stack is 16,384 bytes with initialized-stack accounting, thread stack info, sentinel and assertions; the final compiler-audited nested ephemeris chain is at least 11,120 bytes, excluding runtime overhead. The earlier 4 KiB budget was insufficient.
- The ESP32-S3 flash path uses the documented full watchdog reset; serial hosts configure DTR/RTS before opening, avoiding an unintended control-line pulse.
- Explicit `XZ` starts a fresh software mission. Ordinary reopen/configuration/profile changes preserve live dynamics. No hardware reset was used to hide a mission failure.
- Internal CR bytes remain separators rather than being deleted: `X\rZ` cannot become destructive `XZ`. NUL/control/prefix-malformed and overlong input is refused, followed by ordinary sample recovery.
- Sixteen retained UART/lifecycle records cover idle/reopen/live updates, ten malformed stimuli, stale/duplicate handling, legacy S2, overlong input, burst overload and explicit fresh-session reset.
- The overload test intentionally caused 125 queue drops: 192 stimuli = 67 unique replies + 125 measured drops. One separately tested overlong frame incremented the overflow counter. Recovery occurred without reboot or clearing counters. These intentional faults are excluded from normal pointing/performance results.
- Final independent **read-only** `XI`/`XH` probe confirmed the exact approved image, advancing uptime 4,808,514 → 4,809,517 ms, unchanged intentional-fault counters 125/1, and stack headroom 5,084 bytes. COM5 was then released.

## Disclosed deviation and operator disposition

The worker performed one additional wheel-bookkeeping replay **after every approved physical, paced and lifecycle run**, temporarily lowering only the device's configured wheel-speed limit to 1 rad/s. That exceeded the parent's ancillary scope guard. The violation and its original evidence remain retained, not deleted or quietly relabelled.

Independent review verified chronological isolation, unchanged approved fixtures/physics/firmware, and restoration of the exact normal configuration:

`XC .35 .15 .0028 200 3.27e-6 1675.5160819145563`

**Operator disposition:** the replay is accepted only as separately disclosed supplemental software-state evidence, not as nominal pointing, speed-limit performance or accepted-plant evidence. Its returned commands mathematically distinguish preserved wheel bookkeeping from explicitly cleared bookkeeping without a hardware reset. The main acceptance/performance conclusions rely solely on the unchanged approved runs. This disposition does not grant permission to change future physical parameters or erase the scope violation.

An earlier auxiliary harness assertion incorrectly expected legacy flags 0 instead of the source-defined valid flags 1. The original failing log/raw summary remains preserved; the assertion was corrected and testing resumed on the same live MCU. This was not a firmware change or a hidden reboot.

## Reproduction and artifacts

Run from the repository root:

```bash
PYTHONPATH=src .venv/Scripts/python.exe results/esp32_adcs/v2/parent_review.py
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_adcs_service_timing.py tests/test_zephyr_adcs_integration.py -q
PYTHONPATH=src .venv/Scripts/python.exe results/zephyr_adcs/v5/verify_artifacts.py
```

`parent_review.py` requires exclusive COM5 access for its final read-only probe. It does not flash, reset, configure the device, or overwrite the worker's report/audit. `parent_readback.json` contains the independently reconstructed metrics and live probe. Complete raw evidence remains under `results/esp32_adcs/v2`; `REPORT.md` preserves the implementer's original review-required handoff and is superseded for verdict purposes by this document.

Image ID: `a1218da36d2248d7b439bdb91e0873726001a62a2ca532b31b5ec1144536780c`.

The host generates synthetic sensors and applies only returned device wheel commands through the accepted physical stand-in. The ESP32 computes references, attitude estimate, profile/control and allocation. Ground truth stays in host scoring. No flight, real mechanical actuator, real physical sensor, unlimited soak, or complete-orbit wall-paced qualification is claimed.
