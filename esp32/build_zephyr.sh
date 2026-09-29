#!/usr/bin/env bash
# Build the PocketADCS ADS HIL application with Zephyr 4.4.2 for the
# Waveshare ESP32-S3-LR1121-HF.
#
# Usage: bash esp32/build_zephyr.sh
# Override ZEPHYR_WORKSPACE, ZEPHYR_SDK_INSTALL_DIR, or ADS_ZEPHYR_STAGE.
#
# The board target is the project's own Zephyr board definition under
# esp32/zephyr/boards/xtensa/waveshare_esp32s3_lr1121_hf/, not the generic
# esp32s3_devkitc. That definition is what makes the board correct rather than
# patched: it includes the N4R2 SoC dtsi (4 MB flash, 2 MB QSPI PSRAM) and a
# 4 MB partition table, so the flash geometry and the mcuboot slot layout are
# right in BOTH the app and the bootloader image. Building against
# esp32s3_devkitc required patching the geometry with a board overlay -- and
# sysbuild does not pass that overlay to the MCUboot image, so the bootloader
# derived its slot layout from addresses past the end of the real flash and the
# board booted to silence. Using the real board target removes that failure
# mode, along with the build step that patched mcuboot's boards/ dir by hand.
set -euo pipefail

src="$(cd "$(dirname "$0")" && pwd)"
stage="${ADS_ZEPHYR_STAGE:-$HOME/AppData/Local/Temp/pocket_ads_z}"
workspace="${ZEPHYR_WORKSPACE:-$HOME/zephyrproject}"
export ZEPHYR_SDK_INSTALL_DIR="${ZEPHYR_SDK_INSTALL_DIR:-$(cd "$HOME/zephyr-sdk-1.0.1" && pwd -W)}"

# dtc is vectored via WinGet on this host; drop this if dtc is already on PATH.
DTC_PATH="/c/Users/drago/AppData/Local/Microsoft/WinGet/Packages/oss-winget.dtc_Microsoft.Winget.Source_8wekyb3d8bbwe/usr/bin"
[ -d "$DTC_PATH" ] && export PATH="$DTC_PATH:$PATH"

# Zephyr's sysbuild CONF_FILE handling splits paths with spaces. Stage the
# minimal sources outside Nextcloud's School - NextCloud directory.
rm -rf "$stage"
mkdir -p "$stage"
# Copy the whole shared-source directory rather than naming files: a new
# header added to esp32/src/ must reach the stage automatically, or the build
# breaks with a "No such file" error that looks like a toolchain problem.
cp -r "$src/zephyr" "$stage/zephyr"
cp -r "$src/src/." "$stage/src/"
stage_native="$(cd "$stage" && pwd -W)"
workspace_native="$(cd "$workspace" && pwd -W)"

cd "$workspace_native"
py -3.13 -m west build \
    -b waveshare_esp32s3_lr1121_hf/esp32s3/procpu --sysbuild \
    -d "$stage_native/build" -p always "$stage_native/zephyr" -- \
    -DBOARD_ROOT="$stage_native/zephyr"

# The board target carries the right geometry in both images, but assert it
# anyway: a board that boots to silence looks exactly like a successful build,
# so this check is what separates the two before the board is ever touched.
for img in mcuboot zephyr; do
    f="$stage/build/$img/zephyr/include/generated/zephyr/autoconf.h"
    [ -f "$f" ] || { printf 'missing autoconf for %s: %s\n' "$img" "$f" >&2; exit 1; }
    size=$(grep -E '^#define CONFIG_FLASH_SIZE ' "$f" | awk '{print $3}')
    printf '%-9s CONFIG_FLASH_SIZE = %s\n' "$img" "$size"
    if [ "$size" != "4194304" ]; then
        printf 'ERROR: %s built for a %s-byte flash; this board has 4 MB\n' \
               "$img" "$size" >&2
        exit 1
    fi
done
