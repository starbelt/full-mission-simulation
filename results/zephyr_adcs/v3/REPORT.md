# ADCS fresh-session correction: round-2 build-only handoff (v3)

Task: t_ee164a93. Implementation and verification complete; independent approval required. No serial access, COM5, esptool, flashing or MCU reset. Hardware acceptance remains owned by t_ae7823e4. v1 and the rejected v2 candidate/evidence are preserved.

## Lifecycle decision retained from v2

XZ is an explicit fresh isolated HIL experiment boundary, used only when the host initializes a new physical plant and wheel-board stand-in. It clears estimator attitude/history/fix/rejection/covariance, controller-input validity, sample timing/stale state, wheel bookkeeping, active and pending schedule/reference, selection state and transaction counters in place. Valid configuration/calibration, platform cycle callback and cumulative real uptime/RX/stack health survive. XC and ordinary live plan/reference changes preserve mission dynamics and momentum. Adapter validates XZ before XC. Timestamp regression/duplicate refusal remains strict within a mission, including stale-gap cases; no timestamp rebasing. First gyro-only data after reset cannot invent an absolute fix.

## Independent review defect and correction

Review rejected v2 because the actual UART ISR accepted raw `XZ` + NUL + ` extra\n`, and C-string truncation dispatched a destructive bare XZ. The protocol's exact-command check cannot recover bytes discarded by framing.

The actual target ISR now validates bytes before C-string conversion using src/command_framing.h. Allowed bytes are printable ASCII, tab, ignored CR and terminating LF. NUL, DEL, non-ASCII and other controls mark the whole line invalid. The ISR drains through LF, queues one invalid sentinel, clears assembly flags and accepts the next line. No prefix or suffix is dispatched. Binary invalidity does not masquerade as a length overflow; genuine overflow and queue-failure accounting remains intact. The refusal preserves all mission dynamics and pending transactions.

Both native drivers now consume complete bounded lines with the same predicate rather than allowing fgets/C-string truncation. Native Windows stdin is explicitly binary: Ctrl-Z must not act as EOF and truncate a malformed reset into XZ. The actual-target host driver also uses binary stdin so raw-byte regressions test the real ISR behavior. Source changes relative to v2 are confined to the framing header, actual UART assembly, native test/surrogate framing, tests and documentation. The accepted plant, environment, configuration and controller/core remain unchanged.

## Executed evidence

framing_red_corrected.log: before the correction, 7 failures and 5 passes; embedded-NUL reset destructively accepted by actual ISR+main and protocol driver. framing_red.log is an initial test-authoring run with fixture errors, not the final red evidence.

framing_final.log: 52 passed after correction, including actual ISR plus firmware main dispatch, raw NUL/control/DEL/high-bit/Ctrl-Z bytes, exact bare/whitespace resets, real ephemeris native framing and recovery. Actual malformed-dispatch tests snapshot every byte of Session immediately before and after refusal: estimator, wheel bookkeeping, active reference, pending reference/transaction counters and configuration must be unchanged. A nonzero wheel command is required before refusal, and the next ordinary sample must produce exactly the baseline Y response. Native tests additionally drain over-5000-byte invalid lines once.

verified_tests_final.log: 312 passed, 3 known ERFA future-date warnings, 194.35 seconds. Covers lifecycle, complete persistent missions, malformed commands, live momentum preservation, timestamp refusal, first gyro-only behavior, failed-reset adapter gating, protocol/HIL/actual Zephyr integration/resource budgets/core differential/ephemeris and PC controller/estimator/quaternion/rotor/wheel/SlewOS/fast-plant components. The earlier verified_tests.log is the pre-final 310-test run; use the final log. ruff.log is green; git diff --check passes. No unrestricted full-suite pass is claimed.

run_native.py and native.log: eight complete cases pass, each matching approved accepted-v3 summaries exactly. Fresh processes then all four missions on the SAME persistent process (PID 83784) exercise target3d, SlewOS, eclipse and biased/noisy. Every mission begins at host time zero with validated XZ; no process reset between persistent missions. Final errors in degrees: target3d 0.0075581127842401456; SlewOS 0.02846167315102706; eclipse 0.007558120885474125; biased/noisy 1.0101981109026688. These are correctness surrogates, not embedded performance results.

The incomplete native_pre_binary_stdin directory/log is retained separately: that initial run was deliberately stopped when adding binary Windows stdin, then all final cases were rebuilt and rerun in native/. It is NOT acceptance evidence.

verify_artifacts.py exited 0 and produced verification.json: all eight TX/Y/sample joins, finite fields, sequence completeness, time-zero and exact endpoints, estimate/controller equality, applied wheel torques, reset ACKs, endpoint errors, native binary/source hashes, unchanged accepted physics, target identity material, retained source/images, partition/linker fit and stack chain pass read-back. handoff_sources/ and handoff_hashes.json retain the exact adapter, documentation and regression sources in addition to the full target sources/build.

## Immutable candidate and build/memory audit

Build: ADS_ZEPHYR_STAGE=$HOME/AppData/Local/hermes/cache/scratch/pocket_adcs_t_ee164a93_v3 bash esp32/build_zephyr.sh. Exit 0. Board waveshare_esp32s3_lr1121_hf/esp32s3/procpu. Zephyr 4.4.2; Xtensa GCC 14.3.0 / SDK 1.0.1. Both MCUboot/application generated configs specify 4194304-byte flash. audit_build.py exited 0 and retains complete staged source, build/config/object/ELF/map/image/stack inventory plus 3870 hashed dependencies and actual west module revisions. build_manifest.json pins these inputs and all flash artifacts.

Firmware changes versus approved v1: src/adcs_protocol.h; added src/command_framing.h; zephyr/src/main.cpp. Other approved firmware/model sources remain identical.

Protocol SHA256: 14f5855faf2c3e07ef96460e89ce16d8f6981334f0d0b2aa9ccffb335f6fd200.
Framing SHA256: 55ceddaebaeee0281798c8c87442c70212e53cdea904f7c9252eacc9db8c1600.
Target main SHA256: 05169f2d7724a57416ccacfe7dcc36dd947f5f06341c5c46252858ac03df71f1.
Build ID: db09c6918d21051f77190a77b22df604fd5d2f365be128c43679133c2d0d1fb2.
Expected XI: I 1 db09c6918d21051f77190a77b22df604fd5d2f365be128c43679133c2d0d1fb2 240000000 320 64.

target/build/mcuboot/zephyr/zephyr.bin: 37344 bytes at 0x0; SHA256 4d5212004c034d8b4578a54889b5b6a59bcc322233f8106bebcdb4ee4a3ebcc0.

target/build/zephyr/zephyr/zephyr.signed.bin: 188204 bytes at 0x20000; SHA256 95d3995b9ad1c74c960d3d51e3e10459bd8669ba6f03df9478e9f426514bce1b.

All flash artifacts and audited linker regions fit. Compiler main-thread ephemeris chain lower bound remains 11120 bytes in configured 16384-byte main stack; actual ISR frame is 96 bytes. This is NOT a proven runtime maximum: libm/ABI/interrupt/runtime overhead is excluded. Real new-image stack watermark, RX losses and stage service times still require hardware acceptance. Existing build warnings (legacy float promotions and MCUboot Kconfig assignments) remain; successful does not mean warning-free.

## Independent review and migration gate

Reviewer must inspect actual byte framing and reset lifecycle, independently run regression tests and verify_artifacts.py, and approve this exact retained source/image manifest. The rejected v2 candidate is not approved. Only after independent approval may the operator revise t_ae7823e4 from immutable v1 build ID e6b87a559797bc135e9ec7ad8a2d36acf33f005a7b0a2e302f61bf36925309c4 to the v3 build ID and exact two flash hashes above, together with adapter/source hashes in handoff_hashes.json. No flashing is authorized by this implementation report. The sole hardware owner must verify XI, perform fresh-session hardware acceptance and collect post-run XH. Do not combine v1 firmware with the revised fresh-session adapter.
