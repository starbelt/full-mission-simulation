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
# There is deliberately NO trailing reset (`--after no-reset`). esptool's
# default "Hard resetting via RTS pin" latches this board in the ROM download
# stub (boot:0x0, "waiting for download"), where writes are accepted and
# silently discarded -- which looks exactly like a firmware that ignores its
# input. With the reset suppressed the chip boots the new images on its own
# once esptool closes the port. The AIO PocketQube flash script, which has a
# proven working link on this same board, likewise leaves reset alone. If a
# flash does not take, unplug the board rather than adding another reset.
#
# On this board the PC does not need DTR/RTS pulsed either: the USB console
# runs at DTR=false, RTS=false, which is what the host-side harnesses use.
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
    --after no-reset \
    write-flash 0x0 "$build_native/mcuboot/zephyr/zephyr.bin" \
    0x20000 "$build_native/zephyr/zephyr/zephyr.signed.bin"
