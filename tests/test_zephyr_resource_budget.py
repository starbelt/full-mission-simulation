"""Guard the target-thread budget, not host-compiler stack estimates.

Xtensa -Os -fstack-usage measured this ephemeris call chain:
main 640 -> references 96 -> igrfFieldGcrs 416 -> igrfFieldItrs 3504
-> legendreSchmidt 4832 = 9488 bytes before libm/ABI/runtime overhead.
This regression is a configuration lower-bound check. Runtime stack high-water
and live liveness tests are still required; this is not proof of stability.
"""
from pathlib import Path

CONF = Path(__file__).resolve().parents[1] / "esp32" / "zephyr" / "prj.conf"


def test_main_stack_fits_measured_xtensa_ephemeris_chain_with_margin():
    settings = dict(
        line.split("=", 1)
        for line in CONF.read_text().splitlines()
        if line.startswith("CONFIG_") and "=" in line
    )
    measured_call_chain_bytes = 640 + 96 + 416 + 3504 + 4832
    minimum_overhead_margin_bytes = 2048
    configured = int(settings["CONFIG_MAIN_STACK_SIZE"])
    assert configured >= measured_call_chain_bytes + minimum_overhead_margin_bytes, (
        f"main stack {configured} cannot fit the measured Xtensa ephemeris "
        f"chain ({measured_call_chain_bytes}) plus runtime margin"
    )


def test_stack_overrun_is_detected_instead_of_silent_memory_corruption():
    settings = dict(
        line.split("=", 1)
        for line in CONF.read_text().splitlines()
        if line.startswith("CONFIG_") and "=" in line
    )
    for option in (
        "CONFIG_INIT_STACKS", "CONFIG_THREAD_STACK_INFO",
        "CONFIG_STACK_SENTINEL", "CONFIG_ASSERT",
    ):
        assert settings.get(option) == "y", f"missing runtime stack safety: {option}"

