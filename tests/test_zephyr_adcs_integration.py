"""Execute the actual Zephyr integration with only kernel/transport stubs."""
import subprocess
from pathlib import Path

import pytest
from test_ads_math_host import HEADER_DIR, _env_for, find_usable_compiler

ROOT = Path(__file__).resolve().parents[1]
STUB = r'''
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <deque>
#include <string>
#include <vector>
#include <stdarg.h>
#include <stdlib.h>
#include <math.h>
static bool timing_probe = false;
static uint32_t counter_value = UINT32_MAX - 100;
static std::string events;
static std::vector<uint32_t> readings;
static int forced_y_length = 0;
static void charge(char event, uint32_t cost) {
    if (timing_probe) { events += event; counter_value += cost; }
}
static int probe_snprintf(char *out, size_t cap, const char *format, ...) {
    charge(format[0] == ' ' ? 'M' : 'F', 37);
    va_list args; va_start(args, format);
    int n = vsnprintf(out, cap, format, args); va_end(args);
    if (format[0] == 'Y' && forced_y_length) {
        n = forced_y_length;
        if (n > 0 && size_t(n) < cap) {
            memset(out, 'x', n); out[0] = 'Y'; out[1] = ' '; out[n] = 0;
        }
    }
    return n;
}
static double probe_strtod(const char *text, char **end) {
    charge('P', 7); return strtod(text, end);
}
static double probe_sqrt(double value) { charge('S', 11); return sqrt(value); }
#define snprintf probe_snprintf
#define strtod probe_strtod
#define sqrt probe_sqrt
struct device {};
static device dev;
struct k_thread { struct { uintptr_t start; size_t size; } stack_info; };
static unsigned char stack_bytes[16384];
static k_thread thread = {{reinterpret_cast<uintptr_t>(stack_bytes), sizeof(stack_bytes)}};
static size_t stack_unused = 0;
static int stack_error = 0;
struct Stop {};
static std::deque<std::string> queued;
static std::string incoming, sent;
static bool queue_fail = false;
static bool native_stream = false;
static void (*before_get)() = nullptr;
#define CONFIG_INIT_STACKS 1
#define CONFIG_THREAD_STACK_INFO 1
#define CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC 240000000
#define ADCS_BUILD_ID "test-source-digest"
#define DT_CHOSEN(x) 0
#define DEVICE_DT_GET(x) (&dev)
#define K_NO_WAIT 0
#define K_MSEC(x) (x)
struct msgq { size_t size; };
#define K_MSGQ_DEFINE(name, size, count, align) static msgq name = {size}
static bool device_is_ready(const device *) { return true; }
static int uart_irq_callback_user_data_set(
    const device *, void (*)(const device *, void *), void *) { return 0; }
static void uart_irq_rx_enable(const device *) {}
static int uart_irq_update(const device *) { return !incoming.empty(); }
static int uart_irq_rx_ready(const device *) { return !incoming.empty(); }
static int uart_fifo_read(const device *, unsigned char *out, int cap) {
    int n = incoming.size() < size_t(cap) ? int(incoming.size()) : cap;
    memcpy(out, incoming.data(), n); incoming.erase(0, n); return n;
}
static void uart_poll_out(const device *, unsigned char c) {
    charge('T', 1000);
    if (native_stream) { putchar(c); if (c == '\n') fflush(stdout); }
    else sent += char(c);
}
static int k_msgq_put(msgq *, const void *line, int) {
    if (queue_fail) return -1;
    queued.push_back(static_cast<const char *>(line)); return 0;
}
static int k_msgq_get(msgq *q, void *line, int) {
    if (before_get) before_get();
    if (queued.empty()) throw Stop();
    charge('Q', 50000);
    strncpy(static_cast<char *>(line), queued.front().c_str(), q->size);
    queued.pop_front(); return 0;
}
static uint32_t k_cycle_get_32() {
    counter_value += 100;
    if (timing_probe) { events += 'C'; readings.push_back(counter_value); }
    return counter_value;
}
static int64_t k_uptime_get() { charge('U', 23); return 12345; }
static uint32_t sys_clock_hw_cycles_per_sec() { return 240000000; }
static k_thread *k_current_get() { return &thread; }
static int k_thread_stack_space_get(const k_thread *, size_t *out) {
    charge('H', 19); *out = stack_unused; return stack_error;
}
typedef long atomic_t;
template<typename T> static T atomic_get(const T *value) { return *value; }
template<typename T> static void atomic_set(T *value, int n) { *value = T(n); }
template<typename T> static T atomic_inc(T *value) { return (*value)++; }
'''


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    directory = tmp_path_factory.mktemp("zephyr_main")
    compiler = find_usable_compiler(directory)
    if compiler is None:
        pytest.skip("no usable host compiler")
    (directory / "zephyr_stub.h").write_text(STUB)
    for name in ("kernel.h", "device.h", "drivers/uart.h", "devicetree.h",
                 "sys/printk.h", "sys/atomic.h"):
        path = directory / "zephyr" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#include "zephyr_stub.h"\n')
    binary = directory / "target.exe"
    subprocess.run([str(compiler), "-std=c++11", "-O2", "-Wall", "-Wextra",
                    "-Werror", "-Wno-unused-function", "-I", str(directory),
                    "-I", str(HEADER_DIR),
                    str(ROOT / "tests/firmware/zephyr_main_driver.cpp"),
                    "-o", str(binary), "-lm"], env=_env_for(compiler),
                   capture_output=True, text=True, check=True, timeout=120)
    def run(mode, script=None):
        result = subprocess.run([str(binary), mode], env=_env_for(compiler),
                                capture_output=True,
                                input=script.encode("latin1") if script is not None else None,
                                check=True, timeout=10)
        return result.stdout.decode("latin1").splitlines()
    return run


def test_actual_isr_bound_overflow_recovery_and_success_not_drop(target):
    assert target("rx") == ["3 0 1", "319 a", "1 !", "2 X"]


def test_actual_isr_counts_only_failed_queue_puts(target):
    assert target("queue") == ["2 1"]


def test_actual_health_api_preserves_byte_count_and_reports_unavailable(target):
    assert target("health") == ["1234 -1"]


def test_actual_main_reports_measured_health_identity_and_preserves_legacy(target):
    lines = target("dispatch")
    assert lines[1] == "H 0 12345 7 3 4321 0"
    assert lines[3] == "H 0 12345 7 3 4321 0"
    assert lines[4] == "I 1 test-source-digest 240000000 320 64"
    fields = lines[5].split()
    assert fields[:2] == ["R", "1"]
    assert len(fields) == 13
    assert int(fields[-2]) > 0 and int(fields[-1]) > 0


def test_actual_main_uses_exact_protocol_and_real_ephemeris_with_wrap_safe_cycles(target, tmp_path):
    from examples import hil_closed_loop_run as hil
    binary = hil.build_offline(tmp_path / "surrogate")
    script = "\n".join([
        "XC .35 .15 .0028 200 3.27e-6 1675.5160819145563",
        "XL 0 0", "XR 1 0 0 0 .9393727128473789 .10391372781674944 "
        ".27710327417799857 .17318954636124906 300",
        "XE",
        "XS2 1 1735689600000 0 0 0 0 .00002 .00001 .00003 1 0 0 6878137 0 0",
        "XS2 2 1735689600050 .05 .01 .02 .03 .00002 .00001 .00003 0 0 0 6878137 0 0",
    ]) + "\n"
    native = subprocess.run([str(binary)], input=script, env=_env_for(binary),
                            text=True, capture_output=True, check=True).stdout.splitlines()
    actual = target("script", script)[1:]  # startup banner is not a reply
    assert len(actual) == len(native)
    for observed, expected in zip(actual, native, strict=True):
        if expected.startswith("Y "):
            assert observed.split()[:-5] == expected.split()[:-4]
            assert observed.split()[-5:-1] == ["100"] * 4
            assert observed.split()[-1] == "700"  # kernel test counter, NOT device timing
        else:
            assert observed == expected


ISR_SETUP = (
    "XC .35 .15 .0028 200 3.27e-6 1675.5160819145563\n"
    "XL 0 0\nXR 1 0 0 0 .9393727128473789 .10391372781674944 "
    ".27710327417799857 .17318954636124906 300\nXE\n"
    "XS2 1 1735689600000 0 0 0 0 .00002 .00001 .00003 1 0 0 6878137 0 0\n"
    "XS2 2 1735689600050 .05 .01 .02 .03 .00002 .00001 .00003 1 0 0 6878137 0 0\n"
    "XL .2 0\nXR 1 0 0 0 1 0 0 0 20\n"
)
ISR_NEXT = "XS2 3 1735689600100 .1 .01 .02 .03 .00002 .00001 .00003 1 0 0 6878137 0 0\n"


@pytest.mark.parametrize(
    "invalid", ["XZ\x00 extra", "XZ\x01", "XZ\x1a", "XZ\x7f", "XZ\x80", "\x00XZ",
                "X\rZ", "X\r\rZ", "XC .3\r5 .15 .0028 200 3.27e-6 1675.5160819145563",
                "XL .\r2 0", ISR_NEXT.rstrip("\n").replace("1735689600100", "1735689600\r100")]
)
def test_actual_isr_malformed_binary_reset_is_transactional_and_recovers(target, invalid):
    baseline = target("isr-script", ISR_SETUP + ISR_NEXT)
    observed = target("isr-script", ISR_SETUP + invalid + "\n" + ISR_NEXT)
    assert "TRANSACTION 1" in observed
    assert "MOMENTUM 1" in observed
    assert "TRANSACTION 0" not in observed
    refusals = [line for line in observed if line.startswith("E ") and line.split()[2] == "4"]
    assert len(refusals) == 1
    assert refusals[0].split()[-1] == "1"  # absolute fix is still live
    assert [line for line in observed if line.startswith("Y ")] == [
        line for line in baseline if line.startswith("Y ")
    ]
    assert any(float(v) != 0 for v in baseline[6].split()[15:18])


@pytest.mark.parametrize("reset", ["XZ", "XZ\r", "XZ \t\r", "XZ\r\r"])
def test_actual_isr_valid_reset_still_clears_and_accepts_fresh_time_zero(target, reset):
    lines = target("isr-reset", ISR_SETUP + reset +
                   "\nXH\nXL 0 0\nXR 1 0 0 0 1 0 0 0 20\nXE\n" +
                   ISR_NEXT.replace(" 3 1735689600100 .1 ", " 3 1735689600100 0 "))
    assert "E XZ 0 -1 0" in lines
    assert "H 0 12345 0 0 0 0" in lines
    assert lines[-1].startswith("Y 0 3 1 ")


def test_actual_isr_crlf_identity_health_and_legacy_match_lf(target):
    script = ("XI\nXH\nS2 1 1735689600000 0 0 0 .00002 .00001 .00003 "
              "1 0 0 6878137 0 0\n")
    baseline = target("isr-reset", script)
    assert "I 1 test-source-digest 240000000 320 64" in baseline
    assert target("isr-reset", script.replace("\n", "\r\n")) == baseline


def test_actual_isr_cr_separators_and_crlf_keep_valid_syntax(target):
    baseline = target("isr-reset", ISR_SETUP + ISR_NEXT)
    script = (ISR_SETUP + ISR_NEXT).replace(" ", "\r").replace("\n", "\r\n")
    assert target("isr-reset", script) == baseline


def test_native_ephemeris_framing_refuses_binary_and_overlong_lines_once(tmp_path):
    from examples import hil_closed_loop_run as hil

    binary = hil.build_offline(tmp_path / "native")
    def run(script):
        return subprocess.run([str(binary)], input=script, env=_env_for(binary),
                              text=True, encoding="latin1", capture_output=True,
                              check=True).stdout.splitlines()

    baseline = run(ISR_SETUP + ISR_NEXT)
    assert run((ISR_SETUP + ISR_NEXT).replace(" ", "\r").replace("\n", "\r\n")) == baseline
    for reset in ("XZ", "XZ\r", "XZ \t\r", "XZ\r\r"):
        assert run(ISR_SETUP + reset + "\n")[-1] == "E XZ 0 -1 0"
    for invalid in ("XZ\x00 extra", "XZ\x01", "XZ\x1a", "XZ\x7f", "XZ\x80", "\x00XZ",
                    "XZ\x00" + "x" * 5000, "XZ" + " " * 5000,
                    "X\rZ", "X\r\rZ", "XC .3\r5 .15 .0028 200 3.27e-6 1675.5160819145563",
                    "XL .\r2 0", ISR_NEXT.rstrip("\n").replace("1735689600100", "1735689600\r100")):
        lines = run(ISR_SETUP + invalid + "\n" + ISR_NEXT)
        assert len(lines) == len(baseline) + 1
        refusal = lines[len(ISR_SETUP.splitlines())].split()
        assert refusal[0] == "E" and refusal[2] == "4" and refusal[-1] == "1"
        assert [line for line in lines if line.startswith("Y ")] == [
            line for line in baseline if line.startswith("Y ")
        ]


def test_large_session_and_transaction_are_not_automatic_stack_storage():
    main = (ROOT / "esp32/zephyr/src/main.cpp").read_text()
    protocol = (ROOT / "esp32/src/adcs_protocol.h").read_text()
    assert "static adcs::Session session;" in main
    assert "\n    adcs::Session session;" not in main
    assert "Schedule candidate = session.pending;" not in protocol
