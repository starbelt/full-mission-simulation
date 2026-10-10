"""Exercise the flash command without a real board or esptool side effects."""
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "esp32" / "flash_zephyr.sh"


def test_native_usb_s3_flash_uses_full_system_reset(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    for relative in ("mcuboot/zephyr/zephyr.bin", "zephyr/zephyr/zephyr.signed.bin"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"test fixture, not firmware")
    # Override py in this child shell ONLY. The real script resolves/builds
    # image arguments, but esptool is never executed and no serial port opens.
    result = subprocess.run(
        [bash, "-c", 'py() { printf "%s\\n" "$@"; }; export -f py; '
         'bash "$1" COM5 "$2"', "flash-contract", str(SCRIPT), str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    args = result.stdout.splitlines()
    assert "--after" in args, "native USB hard-reset can leave S3 in download mode"
    assert args[args.index("--after") + 1] == "watchdog-reset"
    assert args[args.index("--chip") + 1] == "esp32s3"
    assert "no-reset" not in args
    assert "write-flash" in args and "0x0" in args and "0x20000" in args
