#!/usr/bin/env bash
# Flash the PocketADCS ADS HIL firmware to a Waveshare ESP32-S3-LR1121-HF.
# Usage: bash esp32/flash_zephyr.sh COM5 [BUILD_DIR]
#
# Two images at the offsets fixed by the board partition table
# (partitions_0x0_amp.dtsi, from the project's Zephyr board definition):
#
#   mcuboot   @ 0x000000  boot_partition   ( 64K)
#   PROCPU    @ 0x020000  slot0_partition  (1344K)
#
# The images carry the board's own flash parameters, so no --flash-mode or
# --flash-size overrides are passed.
#
# This target uses the ESP32-S3's native USB-Serial/JTAG, not an external
# USB/UART bridge. The default RTS reset may reset only the cores and leave
# the sampled download-mode strap latched. A watchdog full-system reset
# re-samples the boot straps and starts the application without an EN press.
# Verified on COM5: same image was silent after hard-reset, then emitted its
# heartbeat and answered a frame after watchdog-reset. Do not use no-reset.
# This choice is S3-specific; it must not be generalized to ESP32-C6.
# https://docs.espressif.com/projects/esptool/en/latest/esp32s3/troubleshooting.html
set -euo pipefail

port="${1:?specify a serial port, e.g. COM5}"
build="${2:-${ADS_ZEPHYR_STAGE:-$HOME/AppData/Local/Temp/pocket_ads_z}/build}"
boot="$build/mcuboot/zephyr/zephyr.bin"
app="$build/zephyr/zephyr/zephyr.signed.bin"
for image in "$boot" "$app"; do
    test -s "$image" || { printf 'missing image: %s\n' "$image" >&2; exit 1; }
done
build_native="$(cd "$build" && pwd -W)"

py -3.13 -m esptool --chip esp32s3 --port "$port" --baud 460800 --after watchdog-reset \
    write-flash 0x0 "$build_native/mcuboot/zephyr/zephyr.bin" \
    0x20000 "$build_native/zephyr/zephyr/zephyr.signed.bin"
