# PocketADCS ADS HIL on Zephyr (ESP32-S3)

This is the Zephyr port of the serial sensor-stream attitude-determination firmware. It receives synthetic gyro, magnetic, Sun, and reference vectors from `examples/hil_stream.py`, computes the TRIAD result using the same `esp32/src/ads_math.h` as the Arduino implementation, and returns angle, rates, norms, flags and quaternion. It does **not** read physical sensors or run the LoRa firmware.

On the Waveshare ESP32-S3-LR1121-HF (4 MB embedded flash, 2 MB PSRAM), the Zephyr 4.4.2 `esp32s3_devkitc/esp32s3/procpu` board with the local USB-Serial/JTAG and flash-size overlay runs on the PROCPU. Sysbuild provides MCUboot and the signed app. UART is 115200 baud on the native USB port.

## Build, flash, verify (Git Bash on Windows)

Prerequisites: Zephyr v4.4.2 west workspace (`~/zephyrproject`), Zephyr SDK 1.0.1 (`~/zephyr-sdk-1.0.1`), Python 3.13 with `west` and `esptool`, `dtc` on PATH, and the Espressif HAL blobs installed. Override `ZEPHYR_WORKSPACE`, `ZEPHYR_SDK_INSTALL_DIR`, or `ADS_ZEPHYR_STAGE` to use different locations. The build stages sources at `%LOCALAPPDATA%/Temp/pocket_ads_z` because Zephyr sysbuild cannot handle spaces in the original repository path.

```sh
bash esp32/build_zephyr.sh
bash esp32/flash_zephyr.sh COM5
PYTHONPATH=src .venv/Scripts/python.exe examples/hil_stream.py --port COM5 --duration 120 --rate 10
```

**Check the port before flashing.** `flash_zephyr.sh` overwrites the specified board's existing bootloader and application; COM7 on this bench carries a different Zephyr LoRa image and must not be used for this ADS target. MCUboot is written at `0x0`, the signed application at `0x20000` (4 MB AMP partition map). Esptool verifies both writes. A passing HIL run prints `OK` and exits zero; any missing/duplicate reply, parse error, bad flags or quaternion, or error beyond the default angle/attitude tolerances exits nonzero. The `--duration` flag spans simulation time; frames are transmitted as fast as serial allows rather than in real time.

Host regressions: `PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_ads_math_host.py tests/test_hil_stream_capture.py -q`.
