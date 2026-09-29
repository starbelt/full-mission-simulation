#!/usr/bin/env bash
# Flash only the specified ESP32-S3. This replaces its current firmware.
# Usage: bash esp32/flash_zephyr.sh COM5 [BUILD_DIR]
set -euo pipefail

port="${1:?specify a serial port, e.g. COM5}"
build="${2:-${ADS_ZEPHYR_STAGE:-$HOME/AppData/Local/Temp/pocket_ads_z}/build}"
boot="$build/mcuboot/zephyr/zephyr.bin"
app="$build/zephyr/zephyr/zephyr.signed.bin"
for image in "$boot" "$app"; do
    test -s "$image" || { printf 'missing image: %s\n' "$image" >&2; exit 1; }
done
build_native="$(cd "$build" && pwd -W)"
py -3.13 -m esptool --chip esp32s3 --port "$port" --baud 460800 \
    write-flash 0x0 "$build_native/mcuboot/zephyr/zephyr.bin" \
    0x20000 "$build_native/zephyr/zephyr/zephyr.signed.bin"
