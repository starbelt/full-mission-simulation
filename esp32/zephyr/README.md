# PocketADCS ADS HIL on Zephyr (ESP32-S3)

This firmware receives synthetic body gyro, magnetic and Sun readings, plus navigation position and absolute clock, from `examples/hil_stream.py`. It derives inertial Sun and magnetic references **onboard** and computes attitude using shared `esp32/src/ads_math.h`. The host does not send reference vectors or truth attitude. Legacy S2 remains stateless TRIAD. The separate XC/XL/XS/XE/XR/XS2 protocol now calls the accepted shared gyro-propagating estimator, controller and allocator with the real ephemeris callback. Target integration is build-only pending independent review and later authorized hardware evaluation. It does **not** read physical sensors or run the LoRa firmware.

The target is the local `waveshare_esp32s3_lr1121_hf/esp32s3/procpu` board definition for the N4R2 ESP32-S3 (4 MB flash, 2 MB PSRAM), using Zephyr 4.4.2. Sysbuild provides MCUboot and a signed PROCPU application. Console is native USB-Serial/JTAG at 115200, not an external USB-UART adapter.

## Build, flash, verify (Git Bash on Windows)

Prerequisites: Zephyr v4.4.2 west workspace (`~/zephyrproject`), Zephyr SDK 1.0.1 (`~/zephyr-sdk-1.0.1`), Python 3.13 with west/esptool, dtc and Espressif HAL blobs. Sources are staged outside the space-containing Nextcloud path.

```sh
export ADS_ZEPHYR_STAGE="$HOME/AppData/Local/hermes/cache/scratch/pocket_ads_z"
bash esp32/build_zephyr.sh
bash esp32/flash_zephyr.sh COM5
PYTHONPATH=src .venv/Scripts/python.exe examples/hil_stream.py --port COM5 --duration 10 --rate 10
```

**Enumerate and identify the port before flashing.** COM5 is the ADS ESP32-S3 on this bench. COM7 carries different LoRa firmware when connected and must not be overwritten. Flash offsets are MCUboot `0x0` and signed app `0x20000`; both hashes must verify. The flash script uses **`--after watchdog-reset`** because default native-USB RTS reset can leave this S3 in download mode. Prove application boot by heartbeat plus a valid frame/reply, not by esptool ROM communication. Do not use no-reset or generalize this setting to ESP32-C6.

Configure DTR and RTS **before** opening pySerial. The idle heartbeat is `R -2 0 0 0 0 8 1 0 0 0 -1 -1`. Instrumented frame replies have 13 tokens: `R seq angle rate magNorm sunNorm flags qw qx qy qz ephCycles adsCycles`. Standard acceptance also understands the legacy uninstrumented 11-token reply. The angle field is Sun/magnetic separation, not attitude error; score quaternion against withheld truth with `2*acos(abs(dot))`.

A passing HIL run prints OK and exits zero; missing/duplicate replies, parse failures, bad flags/quaternion/norms or angle/attitude tolerance failures exit nonzero. Duration is **simulation time**; frames are streamed faster than real time. Use bounded pacing for extended measurements rather than an unbounded bulk write.

Main-thread stack is 16384 bytes with runtime sentinel/assert and stack accounting enabled: actual Xtensa ephemeris call-chain budget is at least 9488 bytes before runtime overhead. See [reliability evidence](../../docs/esp32_reliability.md) for the reproduced failures, fixes, measured timing and honest limits.

Host regressions:

```sh
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_ads_math_host.py tests/test_ads_estimate_host.py tests/test_sensor_source_host.py tests/test_hil_stream_capture.py tests/test_hil_timed_acceptance.py tests/test_hil_onboard_bench.py tests/test_zephyr_flash_reset.py tests/test_zephyr_resource_budget.py -q -p no:randomly
```
