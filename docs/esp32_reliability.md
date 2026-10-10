# ESP32-S3 ADS reliability investigation

## Confirmed problems and changes

1. **Native USB post-flash reset.** Both flashed images verified, but the application remained silent after the default RTS/core reset. The same image emitted the idle heartbeat and answered an application command after `esptool --after watchdog-reset`. ESP32-S3 native USB-Serial/JTAG can retain the sampled download-mode strap across a core-only USB reset; the full watchdog reset re-samples it. `esp32/flash_zephyr.sh` now explicitly selects watchdog reset. This is S3-specific, not advice for ESP32-C6 or a claim that reset equals power removal.
2. **Insufficient main-thread stack.** Actual Xtensa `-Os -fstack-usage` found main(640) -> references(96) -> igrfFieldGcrs(416) -> igrfFieldItrs(3504) -> legendreSchmidt(4832), at least **9488 bytes before libm/runtime overhead**, against the configured 4096. `CONFIG_MAIN_STACK_SIZE=16384` now has margin. INIT_STACKS, THREAD_STACK_INFO, STACK_SENTINEL and ASSERT are enabled in the ordinary build, not only in a temporary diagnostic overlay. This fixes a real resource defect; it does not establish that every historic transport stall had that cause.
3. **Host acceptance protocol mismatch.** Instrumented Zephyr emits 13 tokens (including two stage cycle counts), while standard HIL acceptance accepted only 11. That falsely classified all real replies as missing/unparseable. `hil_stream.parse_reply` now accepts precisely the two defined shapes and validates the timing fields. A red-then-green regression exercises the actual 13-token reply; malformed trailers still fail.
4. **Unbounded send backlog.** The standard bulk stream sent 1201 frames faster than the ephemeris worker could consume them: only 111 replies arrived, but the firmware continued heartbeating afterward. This distinguishes a queue-overrun/data-loss failure from a dead application. The repaired sender keeps a 16-frame outstanding window, freeing capacity on any matched sequence (out-of-order replies remain supported). The **same 120-second trajectory subsequently passed 1201/1201**, without reflashing or resetting. The standard harness now also configures modem lines before open and bounds writes. Its final-reply timeout starts after sending, not before a potentially lengthy run.

Reference: https://docs.espressif.com/projects/esptool/en/latest/esp32s3/troubleshooting.html

## Executed checks

- Target rebuilt from repository configuration, without temporary diagnostic extra-conf files, then flashed COM5. Both image hashes verified and post-flash heartbeat observed **without operator pressing reset**.
- Resource/flash regressions and relevant existing host firmware tests: **11 passed**.
- HIL protocol/capture/timed acceptance/resource/flash regressions: **48 passed**; focused Ruff checks clean.
- Standard pipeline, independent 120-second trajectory: **1201/1201 replies**, no missing/duplicate/parse or magnitude failures, max Sun/magnetic separation error **0.0042 degrees**, max full attitude error **0.9209 degrees**. Retained before/after captures are `standard_stream.jsonl` (unbounded send, failed) and `bounded_stream.jsonl` (passed).
- Real-board bounded request/reply test: **3030/3030** exchanges over 30 repeats of an independently simulated 10-second ideal-sensor trajectory. Every batch scored exactly-once sequences, flags, finite near-unit quaternion, input-vector magnitudes and full-angle attitude error. Max attitude error **0.9208797 degrees**; the tolerance was 2 degrees, not tuned to hide this residual.
- **10 serial close/reopen cycles**, each followed by a valid S2 frame/reply, passed without resetting. Modem lines were set false on the unopened port before open.
- Onboard medians: ephemeris **11655.47 us**, TRIAD **178.25 us**. Ephemeris observed maximum **11694.84 us**. Cycle frequency verified in generated app configuration (240000000 Hz) and bound Xtensa timer driver's `CCOUNT` implementation, not inferred solely from the CPU label.
- Host USB round-trip median **16 ms**. This includes transport and is not algorithm execution time.

The measurements in this section are the preserved **v1 baseline**, before the
Sun epoch/frame corrections described below; they are not current accuracy.

Artifacts: `results/firmware_reliability/v1/summary.json` and append-only `raw.jsonl`. The first raw record preserves a FAILED test-harness attempt caused by the 11-vs-13-token mismatch; the following 30 batches are the corrected run. The summary refers only to that corrected run.

## Current corrected-reference image (v2)

The audit found two additional deterministic accuracy defects in the shared
firmware Sun model, not merely acceptable low-precision-model error:

- The Unix-to-J2000 offset had the wrong sign. Actual J2000 Unix milliseconds
  (946728000000) returned **21915 days** rather than zero. The conversion now
  subtracts 10957.5 days.
- Meeus longitude/obliquity produce mean equatorial axes **of date**, but the
  code labelled them GCRS. Removing precession fixes the approximately
  **0.35-degree** coordinate-frame error at 2025 (after fixing the epoch).

Eight red/green shared-header host regressions pin independent calendar anchors
and compare the C++ Sun vector with Astropy `get_sun` over four seasons and
multiple years. All pass; the direction threshold is 0.03 degrees. The 2035
reference raises expected ERFA future-leap-second warnings, not a test failure.
All relevant firmware/protocol/resource host tests: **59 passed**.

Rebuilt and flashed COM5; both flash hashes verified. Idle heartbeat and protocol
replies returned with no manual reset. Current physical-device measurements:

| Check | Result |
| --- | --- |
| 120-second independent trajectory, bounded pipeline | 1201/1201 replies, max full attitude error **0.0099 deg** |
| Soak, 30 repeats of a 10-second trajectory | **3030/3030**, every batch passed at **0.1 deg** attitude tolerance |
| Soak max full attitude error | **0.00521413 deg** |
| Strict serial close/reopen and scored round trip | **10/10**, no reset |
| Onboard ephemeris median / observed maximum | **11758.70 / 11800.43 us** |
| Onboard TRIAD median | **174.99 us** |
| USB round-trip median | **16 ms** (not algorithm time) |

Retained v2 raw replies, per-batch acceptance, timing samples and source hashes
are in `results/firmware_reliability/v2/`. The reproducible soak entry point is:

```bash
PYTHONPATH=src .venv/Scripts/python.exe examples/hil_reliability.py \
  --port COM5 --output-dir results/firmware_reliability/v3 \
  --quat-tolerance-deg 0.1
```

Use a fresh output folder; it refuses to mix two runs. These corrected
stateless-ADS measurements are not yet the extended closed-loop port acceptance.

## Overload is not the same as a dead MCU

A separate intentional overload probe in
`results/firmware_reliability/overflow_recovery_v1/` sent **512 valid S2 frames**
(62,464 bytes) in **0.110 s**. It received **69 unique replies**, with **443
missing sequences**. The board subsequently emitted idle heartbeats and answered
application sequence 999999 with flags=1 **without a reset**. No boot banner
appeared during the captured probe.

This directly distinguishes missing replies under an unbounded stream from a
permanently hung MCU. The raw trace alone cannot identify the exact buffer that
dropped each frame, or prove that overload caused the original reset-required
failure. The independent stack and native-USB reset defects remain real;
bounded streaming removes this demonstrated overload failure mode. The extended
port must report real drop counters rather than infer them from missing replies.

## Longer post-fix stability check (v3)

`results/firmware_reliability/v3/` contains 100 raw scored batches and their
summary. All **10,100/10,100** application replies passed, and **25/25**
close/reopen/application-round-trip cycles passed without resetting the board.
The parent independently counted all actual wire replies: each had the defined
13-token shape, flags=1, unique sequence within its batch, finite values, and
nonnegative stage counts. Maximum quaternion norm error was 6.37e-7.

Maximum attitude error was **0.0052141284 deg**. On-device median ephemeris was
**11.758692 ms** and TRIAD **0.174990 ms**; their sum was **11.933681 ms**.
Median transport round-trip was **16 ms**, reported separately from execution.
This repeats one ideal 10-second trajectory; it is an endurance/reopen check,
not extended ADCS acceptance, full-orbit accuracy, or a physical motor test.

## Scope and remaining verification

These results exercise the **existing stateless ephemeris/TRIAD firmware**, not the extended gyro-propagating/controller port. Repeating a short trajectory is a stability test, not full-orbit accuracy validation. No physical sensor drivers or motors were tested. Main-stack runtime high-water reporting is not yet exposed; configuration margin and a live sustained test are not a measured stack watermark. The closed-loop port must expose that diagnostic and rerun hardware acceptance.

PC acceptance `t_1564a1dd` and the corrected native closed-loop harness `t_a4be5194` are approved. The original broad port `t_3ccaa0d1` was archived after its replacement plant/sensor fixtures were rejected; its tested C++ core was retained. The corrected harness reuses the accepted physical plant, independent Sun/IGRF references, and exact C++ core/protocol/reference-provider surrogate. Its full 2681.974849858644-second SlewOS run, nontrivial 3D, eclipse and deliberately biased/noisy cases are native correctness evidence, **not ESP32 performance**. Strict malformed/nonfinite configuration ACK rejection and complete source/executable provenance were independently verified.

Target integration/build is now `t_c44ec5d2`, gated on that accepted harness. This stage cannot flash or access COM5. Operator must independently review the actual target build before flashing the exact artifacts and scoring physical pointing in the same PC plant driven only by ESP32 commands. Runtime stack watermark, RX drop counters and extended on-device stage timings remain to be measured on the port. Actual wheel hardware remains a stand-in and the 200 rad/s2 acceleration cap remains an engineering assumption, not a measured no-stall guarantee.
