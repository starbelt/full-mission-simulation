# ADCS explicit fresh-session correction — build-only review handoff

Task: t_ee164a93. Status: implementation verified; independent review required. No COM5 access, esptool, flashing or MCU reboot was performed. Hardware acceptance remains owned by t_ae7823e4 and must not resume on the revised image until independent approval and operator manifest migration.

## Decision and correction

XZ now denotes an explicit fresh isolated HIL experiment boundary. It is appropriate only when the host creates a fresh physical plant and wheel-board stand-in. It clears estimator attitude/history/fix/rejection/covariance, controller-input validity, sample time/stale state, wheel bookkeeping, active and pending schedules/references, reference-selection state and transaction counts. It resets fields in place, never copying a whole large Session onto the stack. Valid configuration/calibration, platform cycle callback and cumulative real uptime/RX/stack health are preserved.

XC and ordinary XL/XS/XR/XE live updates do not reset mission dynamics or wheel momentum. Exact XZ arity is required: whitespace is allowed; trailing payload or XZ-prefixed variants are refused without modifying state. The accepted adapter sends and validates XZ before XC and stops before sending configuration/sensors on a failed reset.

Timestamp validation now checks the last accepted sample before mutating stale state. This closes duplicate acceptance following a stale/coasted gap without rebasing timestamps or weakening within-mission negative/duplicate/regression refusal. First gyro-only data after reset cannot invent an absolute attitude fix.

Changed implementation files: esp32/src/adcs_protocol.h; examples/hil_closed_loop_run.py. Documentation: docs/esp32_closed_loop.md. Regressions: tests/test_adcs_protocol_host.py; tests/test_hil_accepted_adapter.py; tests/firmware/adcs_protocol_driver.cpp.

## Red/green evidence

persistent_time_red.log reproduces a completed first mission followed by a new time-zero mission in the same process, refused with E EST 2 1 1 before correction. state_red.log records the dynamic-state reset failures. The first persistent_red.log includes an overly strict old stale-flag assertion; that assertion was corrected before recording the actual timestamp red reproduction.

verified_tests.log: 295 passed, 3 ERFA dubious-year warnings, 195.68 seconds. Covers protocol/HIL/Zephyr/resource/core differential/ephemeris/harness/flash-reset and PC controller/estimator/quaternion/rotor/wheel/SlewOS/fast-plant components. Includes malformed reset, every stored segment/reference endpoint, configuration/health preservation, live dynamics preservation, timestamp refusal, first gyro-only behavior, failed-reset adapter gating and complete persistent missions. ruff_final.log: all checks passed for the modified Python implementation/tests. git diff --check passed.

The attempted unrestricted full suite exceeded the foreground tool wall with an incomplete log; no full-suite pass is claimed. native.log records an initial import-path failure; native_rerun.log and native_rerun.exit record the successful run using PYTHONPATH='src;.'.

## Complete native experiments

run_native.py exercised target3d, SlewOS, eclipse and biased/noisy cases first on fresh surrogate processes and then sequentially on one persistent surrogate process (PID recorded in verification.json), without rebooting between missions. All eight complete cases match the approved accepted-v3 summaries exactly; configuration, plant and environmental inputs were reused verbatim. These are native-host exact C++ protocol/core/ephemeris experiments, not embedded performance measurements.

Each target3d/eclipse/biased-noisy case has 6001 frames; SlewOS has 53641 frames. Final errors in degrees, identical in fresh/persistent modes: target3d 0.0075581127842401456; SlewOS 0.02846167315102706; eclipse 0.007558120885474125; biased/noisy 1.0101981109026688.

verify_artifacts.py exited 0 and generated verification.json. It independently joins TX XS2 sequence numbers, RX Y frames and sample rows; checks finite wire fields, controller-input equality to estimator output, applied wheel-torque equality, time-zero through exact endpoint, reset ACKs and endpoint quaternion error. It verifies retained source/image hashes and build identity material. Firmware is compared to the approved Zephyr v1 baseline (the earlier accepted-PC v3 core hash was superseded by that integration); accepted PC physics/environment sources remain unchanged.

## Target and immutable candidate manifest

Build command: ADS_ZEPHYR_STAGE=$HOME/AppData/Local/hermes/cache/scratch/pocket_adcs_t_ee164a93_v2 bash esp32/build_zephyr.sh. Exit 0. Target: waveshare_esp32s3_lr1121_hf/esp32s3/procpu. Complete sources, build outputs/config/map/stack data are retained under target/. v1 was preserved. build_manifest.json records source SHA256s, dependency inventory, resolved/actual west modules, all images, partitions and compiler stack data.

Only firmware source changed versus approved v1: src/adcs_protocol.h.
Protocol SHA256: 14f5855faf2c3e07ef96460e89ce16d8f6981334f0d0b2aa9ccffb335f6fd200.
Candidate build ID: 955fb6ab887b2d0a360a4ff96cba13a8cbbf4986337eba0931e4b4e236f6b3e8.
Expected XI: I 1 955fb6ab887b2d0a360a4ff96cba13a8cbbf4986337eba0931e4b4e236f6b3e8 240000000 320 64.
MCUboot target/build/mcuboot/zephyr/zephyr.bin: 37344 bytes at 0x0; SHA256 0de7e7ab5ae32ea60f97a7f89427ecfa1a6d57a73c75634d69c55d38f4b5a364.
Application target/build/zephyr/zephyr/zephyr.signed.bin: 188204 bytes at 0x20000; SHA256 f91b550d78b8744357acb40ad9b4a1a49939ef3c8ea08c598f2aca3bc021deef.

Both flash artifacts fit their partitions; all audited linker regions fit. Compiler ephemeris call-chain lower bound decreased from 11136 to 11120 bytes because handleLine uses 576 rather than 592 bytes. Main stack remains 16384 bytes. This is not a proven runtime maximum: libm/ABI/runtime overhead is excluded. Runtime stack headroom, RX drops and service time remain unmeasured for this new target and require hardware acceptance. Existing build warnings include legacy float-to-double promotion and unmet MCUboot Kconfig assignments; the build is successful, not warning-free.

## Review and migration gate

Reviewer must independently inspect reset semantics, malformed-command transactional behavior, live momentum preservation and timestamp guard; run regressions and artifact verifier; approve exact source/image hashes. After approval, the operator must revise t_ae7823e4's immutable manifest from old ID e6b87a559797bc135e9ec7ad8a2d36acf33f005a7b0a2e302f61bf36925309c4 to the reviewed v2 candidate and the retained flash artifact hashes above, plus the revised adapter source hashes. Do not combine the v1 image with the v2 adapter expecting the new lifecycle. The sole serial owner must verify XI and post-run XH and rerun physical hardware acceptance. This report does not itself authorize revised flashing.
