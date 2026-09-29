#!/usr/bin/env bash
# Build the PocketADCS ADS HIL application with Zephyr 4.4.2 on ESP32-S3.
# Usage: bash esp32/build_zephyr.sh
# Override ZEPHYR_WORKSPACE, ZEPHYR_SDK_INSTALL_DIR, or ADS_ZEPHYR_STAGE if needed.
set -euo pipefail

src="$(cd "$(dirname "$0")" && pwd)"
stage="${ADS_ZEPHYR_STAGE:-$HOME/AppData/Local/Temp/pocket_ads_z}"
workspace="${ZEPHYR_WORKSPACE:-$HOME/zephyrproject}"
export ZEPHYR_SDK_INSTALL_DIR="${ZEPHYR_SDK_INSTALL_DIR:-$(cd "$HOME/zephyr-sdk-1.0.1" && pwd -W)}"

# Zephyr's sysbuild CONF_FILE handling splits paths with spaces. Stage the
# minimal sources outside Nextcloud's School - NextCloud directory.
rm -rf "$stage"
# Copy the whole shared-source directory rather than naming files: a new
# header added to esp32/src/ must reach the stage automatically, or the build
# breaks with a "No such file" error that looks like a toolchain problem.
mkdir -p "$stage/src"
cp -r "$src/zephyr" "$stage/zephyr"
cp -r "$src/src/." "$stage/src/"
stage_native="$(cd "$stage" && pwd -W)"

# sysbuild applies the app's board overlay to the APPLICATION image only, so
# MCUboot gets built believing the board is the 8 MB esp32s3_devkitc default
# while the Waveshare ESP32-S3-LR1121-HF actually has 4 MB. MCUboot then derives
# its slot layout from addresses past the end of the physical flash, never
# finds the application, and the board boots to total silence after a flash
# that esptool reports as successful and hash-verified.
#
# The bootloader image is built from ZEPHYR_BASE/bootloader/mcuboot, whose
# boards/ dir is outside this repo, so stage the same overlay there under the
# name Zephyr auto-discovers. The build script must create it, or a clean
# checkout silently reintroduces the 8 MB geometry.
mcuboot_boards="$(cd "$workspace/zephyr" && pwd -W)/../bootloader/mcuboot/boot/zephyr/boards"
if [ -d "$mcuboot_boards" ]; then
    cp "$stage_native/zephyr/boards/esp32s3_devkitc_procpu.overlay" \
       "$mcuboot_boards/esp32s3_devkitc_procpu.overlay"
fi

cd "$workspace"
py -3.13 -m west build -b esp32s3_devkitc/esp32s3/procpu --sysbuild \
    -d "$stage_native/build" -p always "$stage_native/zephyr"
