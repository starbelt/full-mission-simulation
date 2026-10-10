# v5 complete dispatch service timing — build-only review handoff

Task: t_7230e62b. Baseline: independently approved lifecycle/framing v4.
No serial port, reset, flash, esptool, hardware authorization or acceptance-manifest migration was performed. Hardware task t_ae7823e4 remains blocked. Independent review and operator readback of final pins are required before any separately authorized migration.

## Measurement contract

Target Y now has 26 whitespace-separated tokens: the unchanged 25-token Y plus optional trailing service_cycles. The portable protocol still produces the original shape. Historical Y lacking this field parses as None, never fabricated zero; actual zero is valid. All stage and service counts are exact finite integral uint32, 0..4294967295. Decimal validation precedes reliance on float-rounded counts. Four original stage counters remain separately available.

The outer unsigned k_cycle_get_32 delta starts immediately after successful dequeue, before uptime/health sampling, routing, XS2 numeric parsing, reference-profile work, gyro/absolute estimation, onboard ephemeris, controller, allocation and original Y formatting. It ends immediately after handleLine returns, before suffix append and UART TX. It excludes RX ISR framing, queue wait/dequeue, metric suffix/newline formatting, transport and later idle work. Interrupt/preemption time between reads is included; this is elapsed command service, not pure algorithm CPU time. Unsigned wrap subtraction is valid for spans shorter than 2^32 counter cycles (17.8956970667 seconds at 240 MHz), not arbitrary multiwrap spans.

Append is transactional: format a bounded suffix, validate would-be length, available capacity and termination, then copy. Formatting or append failure emits E SHORT 4 -1 0 rather than a truncated Y. There is no control/plant/state-schema change.

The generated target clock is 240000000 Hz; XI reports sys_clock_hw_cycles_per_sec and its source digest. Build audit pins generated configuration and Xtensa timer driver. Host hardware conversion uses validated XI frequency, not a CPU-clock assumption. Raw/native callback counts are test-only. Native timing sidecars deliberately leave device cycles/times/frequency unavailable.

Host transport timing is separate: perf_counter_ns immediately before send through received reply (including host I/O/decode/receive logging); timing.jsonl labels host_exchange_rtt_ns. It is never reported as algorithm time, substituted for device service, or subtracted to fabricate a device measurement.

## Executed verification

- Tests-first initial red: 26 failures, 9 passes; timing_red.log and timing_red.exit retain evidence. Precision red separately exposes two float-rounding acceptance holes; precision_red.log retains failures. Initial intermediate green attempt failed and is retained, not advertised as final green.
- Final bounded focused suite: 381 passed, 3 known ERFA future-date warnings, 225.98 seconds; focused_bounded.log/.exit. Covers actual-main span/uint32 wrap, profile and parsing inclusion, existing formatting boundary, excluded queue/TX, all suffix lengths/canaries, legacy/new/malformed Y and XI, high-resolution transport separation, protocol/lifecycle/framing, HIL, differential core, estimator/controller, plants, wheels, profile and resource/ephemeris tests.
- Ruff for changed host/test Python: passed, ruff_final.log/.exit. git diff --check passed, diff_check_final.log/.exit.
- Expanded suite including the long Python closed-loop acceptance module hit the tool's 420-second limit without completing; focused_final.log is partial and has no success exit. No unrestricted full-suite claim. Full accepted physical missions were instead executed through actual firmware main+ISR below.
- Six retained raw ISR malformed-CR/NUL cases: byte-identical Session preservation, nonzero wheel bookkeeping, exact next-sample recovery. actual_isr_proof.json and prove_isr.py retain reproducible evidence.
- Eight complete physical missions (target3d, full SlewOS, eclipse, biased/noisy; each fresh and persistent) were rerun through compiled actual main+ISR using native Zephyr stubs. Every summary exactly equals accepted results/hil_accepted/v3. Persistent cases share PID 22652. Complete wire, physical samples, separate host timing and manifests are in native/. Full SlewOS covers 2681.974849858644 seconds and 53641 frames per run; each other case covers 300 seconds and 6001 frames. Native cycle counts are stub artifacts, not device performance.
- Native rerun after final parser guard exited 0. Earlier failed import and pre-guard runs are retained with explicit names, not selected as final evidence.
- verify_artifacts.py executed exit 0: source snapshots/current sources, entire retained target inventory, identity material/build ID, image hashes/partition fit, compiler stack rows, unchanged accepted physics, six raw ISR proofs, full eight-case trace/sequence joins, source and binary pins, exact accepted summaries, host RTT sidecars and unavailable native device metrics all passed.

## Actual target build and resource audit

Fresh unique build stage: pocket_adcs_t_7230e62b_v5. Actual waveshare_esp32s3_lr1121_hf/esp32s3/procpu Xtensa sysbuild exited 0; target_build.log retains output. audit_build.py retains the complete stage under target/, tool versions, dependency hashes, west manifest/project revisions, objects/ELFs/maps/images/configuration/source snapshots and memory/stack evidence. Compared with v4, only zephyr/src/main.cpp changes in firmware source pins; generated application/bootloader configuration and west revisions are unchanged.

Build ID: a1218da36d2248d7b439bdb91e0873726001a62a2ca532b31b5ec1144536780c
Expected XI: I 1 a1218da36d2248d7b439bdb91e0873726001a62a2ca532b31b5ec1144536780c 240000000 320 64

Application zephyr.signed.bin: 253740 bytes, partition 1376256 bytes, address 0x20000
SHA256: c3fafdd8e4baad118fcf6987392d5c3f0f39ba6db5366c70daef4caaf88e37d3
Bootloader zephyr.bin: 37344 bytes, partition 65536 bytes, address 0x0
SHA256: d4977e94efb9af1ab194d60fa4a75857516c0a0d1743402a4228d64d9de71741

All 12 linker regions fit. Application DRAM used 83904 / 384256 bytes. Main stack configured 16384; compiler-reported nested ephemeris stack chain lower bound 11120 bytes, unchanged from v4. Main compiler frame is 1296 bytes. This is NOT runtime stack acceptance: library/ABI/interrupt overhead and actual watermark require hardware XH evidence.

Application text crosses the 64-KiB flash-mapping boundary (65520 -> 65616 bytes), increasing alignment padding by 65536 bytes rather than increasing SRAM usage. Bootloader byte changes normalize exactly to compile-time timestamp/checksum/appended digest changes; both appended hashes verify. image_delta_diagnostic.json retains this diagnostic. A signed filename alone does not claim cryptographic signing.

## Reviewer/operator gate

Read and execute verify_artifacts.py; independently recompute parent_readback.json pins against exact final sources, manifests, images and retained evidence. handoff_hashes.json pins current handoff code/docs and their snapshots in handoff_sources/. Preserve v1-v4; no physical-model edits. Neither this report nor build success approves flashing. After independent approval, the operator must perform exact final readback before any separately authorized hardware acceptance-manifest migration. Hardware timing, RX drops, stack watermark and physical execution remain unmeasured.
