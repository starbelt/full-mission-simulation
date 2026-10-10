# CR token-preservation correction — v4 BUILD ONLY

Task t_8fca685e. Implementation and local verification complete. A fresh-context independent reviewer approved the final local correction (independent_review.txt: 70 tests, Ruff/diff, full artifact verifier, six raw ISR replays and independent hash checks). The delegating parent then read that verdict and independently read back all final image/current/retained handoff hashes (parent_readback.json). Board review/operator migration authorization remains separate. No serial port access, esptool, flashing or hardware reset. Hardware t_ae7823e4 stays blocked. v1/v2/v3 evidence is unchanged; v3 is not flash-approved.

## Change and scope

The actual UART ISR and both native command receivers now map every CR byte to one space rather than deleting it. Internal CR therefore cannot turn `X\rZ` into destructive `XZ`, or `.3\r5` into `.35`. Wire validation still rejects NUL, controls, DEL and non-ASCII before C-string conversion, drains malformed/overlong input through LF and recovers on the next line. CR occupies one bounded payload byte; LF is the delimiter. Trailing space/tab is trimmed in the actual ISR and protocol-test receiver so CRLF exact diagnostics and test-only probes remain valid. No internal separator is trimmed.

Firmware delta against immutable v3 is ONLY src/command_framing.h and zephyr/src/main.cpp. adcs_protocol.h, estimator/controller/core, physical plant, environment, configuration, HIL adapter and input ownership are unchanged. Native drivers, regression observer/tests and protocol documentation are updated. source_delta.patch records the complete handoff-source delta relative to v3.

## Red/green and raw proof

framing_red.log was the tests-first pre-fix run, but included a leading-whitespace test that exceeded existing syntax. framing_green.log was a provisional run that exposed Windows CRLF around test-only exact probes; neither is final acceptance evidence.

reproduce_red.py then ran corrected regressions against immutable v3 target/native sources in an isolated scratch copy: framing_red_corrected.log has 15 failures, 16 passes. Actual ISR and native tests reproduce split-reset, split gain, plan numeric and epoch token aliases. The additional crlf_identity_red_corrected.log exposes the provisional v4 trailing-CR identity regression; it was fixed before the final build. crlf_identity_red.log is an insufficient early assertion, not red evidence.

verified_tests_final_raw.log: 339 passed, 3 existing ERFA future-date warnings, 197.09 seconds. Covers actual ISR/main raw bytes, CRLF identity/health/legacy, reset lifecycle, binary rejection/overflow recovery, live wheel preservation, strict timestamps, first gyro-only no-fix, protocol/HIL/Zephyr/resource budgets, core differential, ephemeris and accepted PC components. The ISR fixture uses binary subprocess input so LF and CRLF tests exercise exactly the specified bytes, not Windows newline rewriting. Ruff on both changed Python test files and git diff --check pass. No unrestricted full-suite pass is claimed.

actual_isr_proof.json retains six exact hexadecimal input scripts and complete raw replies from the actual ISR plus firmware main with only transport/kernel stubs. Every malformed dispatch has nonzero live wheel-speed bookkeeping before it, byte-identical entire Session afterward (active/pending plans, estimator, counters, configuration, wheel state), and every Y response including the next ordinary sample exactly matches baseline. The stub cycle values demonstrate invocation only, not embedded timing.

## Complete physical correctness surrogate

native_final.log and native/ contain four complete fresh-process runs plus all four full missions on one persistent native process, beginning each mission at time zero with a validated XZ. Eight summaries exactly match accepted results/hil_accepted/v3. Each target3d/eclipse/biased_noisy case has 6001 frames; each complete SlewOS case has 53641 frames and endpoint 2681.974849858644 seconds. Final physical target errors: target3d 0.0075581127842401456 degrees; SlewOS 0.02846167315102706; eclipse 0.007558120885474125; biased/noisy 1.0101981109026688 (diagnostic, not nominal accuracy).

verify_artifacts.py exits 0. It checks all eight complete TX/Y/sample joins, sequence/time endpoints, physical endpoint score, estimate=controller input, device wheel torque=applied wheel torque, reset ACKs, same persistent PID, accepted physics/source equality, exact source snapshots and signed images, identity material, partition/linker fit, compiler stack chain and the six raw actual-ISR refusals. Native cycles/timing remain unavailable; no hardware performance claim. native.log/native.exit are the canonical successful complete execution (exit 0); native_final.log/native_final.exit retain the same full execution. The failed initial import invocation with no produced cases is preserved separately as native.log_import_failure/native.exit_import_failure.

## Final retained target and exact approval pins

Final unique stage: C:/Users/drago/AppData/Local/hermes/cache/scratch/pocket_adcs_t_8fca685e_v4_final. Actual build_zephyr.sh exits 0. audit_build.py exits 0; complete staged source, generated configuration, objects, ELF/map/image and stack inventory are retained under target/, with 3870 hashed build dependencies and actual west revisions in build_manifest.json. Zephyr 4.4.2 / SDK 1.0.1 / Xtensa GCC 14.3.0; board waveshare_esp32s3_lr1121_hf/esp32s3/procpu; BOTH generated images specify 4194304-byte flash.

Build ID: 8b97f32191cea59a0b8038c41e64ec3be346c5d7921cc7fec3342d88f109e7bc
Expected XI: I 1 8b97f32191cea59a0b8038c41e64ec3be346c5d7921cc7fec3342d88f109e7bc 240000000 320 64
Bootloader target/build/mcuboot/zephyr/zephyr.bin: 37344 bytes at 0x0; SHA256 0c58220e5450f8c9761d99b23adb9c27a8b10a4b508f6fcd74fa0fdb229e4844
Application target/build/zephyr/zephyr/zephyr.signed.bin: 188204 bytes at 0x20000; SHA256 7bedadeeb32f0c9c1da93ff6a15f7a38b2b60f68aa91012b3048dcab95834f33
Framing SHA256: 6ca25a14ad7668dea8b77ae56fcbcbbaa29b7ae267d1c029c2a8905245fbab7f
Target main SHA256: 30d49015d22cb03efa1f96cba3e4b5aae0ab398f768fd65351caf410614ab1a4
Protocol SHA256 (unchanged v3): 14f5855faf2c3e07ef96460e89ce16d8f6981334f0d0b2aa9ccffb335f6fd200
All adapter/docs/test snapshot pins: handoff_hashes.json.

All audited flash partitions and 12 linker regions fit. Application text/data/BSS: 159416/5800/166676 bytes. Actual ISR compiler frame: 96 bytes. Compiler main-thread ephemeris-chain lower bound: 11120 bytes; configured main stack: 16384 bytes. This is not a measured runtime maximum: libm, ABI, interrupts and runtime overhead are excluded. Real new-image stack watermark, RX losses and service times still require authorized hardware acceptance. Existing compiler float-promotion and MCUboot Kconfig warnings remain.

The earlier provisional v4 image before CRLF XI correction is preserved as target_pre_tail_trim and build_manifest.json_pre_tail_trim, with corresponding pre_tail_trim build/proof evidence. It is NOT the final candidate and is not approved. Review only target/ and build_manifest.json.

## Gate

Independent reviewer must run regressions, inspect raw byte framing/state refusal, independently read hashes and run verify_artifacts.py against the FINAL pins above. Parent/operator must then read back those exact sources/images before any later task uses them. This task does not migrate or unblock hardware t_ae7823e4. Total-service timing instrumentation is separate scope and remains outstanding. No commit or push was made.
