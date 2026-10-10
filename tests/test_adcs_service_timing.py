"""Service-span and host transport timing contract; no hardware claims."""
import json

import pytest
from test_hil_closed_loop_harness import y_line
from test_zephyr_adcs_integration import target as target  # shared actual-main fixture

from examples import hil_closed_loop_run as hil
from examples.hil_closed_loop import ProtocolError, parse_y
from pocketadcs_sim.adcs.closed_loop import ClosedLoopConfig

SCRIPT = (
    "XC .35 .15 .0028 200 3.27e-6 1675.5160819145563\n"
    "XL 0 0\nXR 1 0 0 0 .9393727128473789 .10391372781674944 "
    ".27710327417799857 .17318954636124906 300\nXE\n"
    "XS2 1 1735689600000 150 .01 .02 .03 .00002 .00001 .00003 1 0 0 6878137 0 0\n"
)


def test_actual_main_complete_service_span_wraps_and_includes_dispatch_format(target):
    lines = target("service", SCRIPT)
    reply = parse_y(next(line for line in lines if line.startswith("Y ")))
    events = next(line[7:] for line in lines if line.startswith("EVENTS "))
    readings = [int(v) for v in lines[-1].split()[1:]]
    assert events.startswith("QCUH"), "start after dequeue, before health and dispatch"
    assert len(readings) == 8, "outer span plus six original stage counter reads"
    assert readings[0] > readings[-1], "controlled uint32 wrap is actually exercised"
    assert events.count("P") == 15, "all XS2 numeric parsing is inside the outer span"
    before_eph = events.split("C")[1]
    assert "P" in before_eph and before_eph.count("S") == 3, (
        "profile normalizations before stage timing")
    assert "FCMT" in events, "finish after original formatting, before metric formatting/TX"
    assert reply.service_cycles == (readings[-1] - readings[0]) % 2**32
    pairs = ((1, 2), (2, 3), (4, 5), (5, 6))
    assert reply.cycles.tolist() == [(readings[b] - readings[a]) % 2**32 for a, b in pairs]
    assert reply.service_cycles > sum(reply.cycles)
    assert reply.service_cycles < 50000, "queue wait and UART are outside service"


@pytest.mark.parametrize("mode,expected", [
    ("service-fit", "Y "), ("service-short", "E SHORT 4 -1 0"),
    ("service-format-overflow", "E SHORT 4 -1 0"),
])
def test_actual_main_service_append_never_transmits_truncated_y(target, mode, expected):
    lines = target(mode, SCRIPT.replace(" 150 ", " 0 "))
    assert lines[-3].startswith(expected)
    if mode == "service-fit":
        assert len(lines[-3]) == 501 + len(lines[-3].split()[-1]) < 512
    else:
        assert not any(line.startswith("Y ") for line in lines)


def test_actual_main_streaming_driver_matches_batch_dispatch(target):
    assert target("stdio", SCRIPT) == target("script", SCRIPT)


def test_actual_append_all_uint32_lengths_caps_and_canaries(target):
    assert target("append-bounds") == ["APPEND 68"]


def test_historical_y_has_unavailable_service_not_zero():
    reply = parse_y(y_line())
    assert reply.service_cycles is None
    assert reply.cycle_times_us(240000000) == {
        "stages": {"ephemeris": 100 / 240, "estimator": 200 / 240,
                   "controller": 50 / 240, "allocation": 30 / 240},
        "service": None,
    }


@pytest.mark.parametrize("cycles", [0, 1, 4294967295])
def test_new_y_service_uint32_and_xi_frequency_conversion(cycles):
    reply = parse_y(y_line() + f" {cycles}")
    assert reply.service_cycles == cycles
    assert reply.cycle_times_us(80000000)["service"] == cycles / 80


@pytest.mark.parametrize("suffix", ["-1", "4294967296", "1.5", "nan", "inf", "abc",
                                     "0 1", "1e999", "4294967295.00000001",
                                     "1.00000000000000001", "-0.00000000000000000001",
                                     "1e-999999999999999999999999"])
def test_new_y_rejects_malformed_service(suffix):
    with pytest.raises(ProtocolError):
        parse_y(y_line() + " " + suffix)


@pytest.mark.parametrize("index", [20, 21, 22, 23])
def test_stage_cycles_are_uint32_not_unbounded_numbers(index):
    with pytest.raises(ProtocolError):
        parse_y(y_line(**{f"f{index}": 4294967296}))
    assert parse_y(y_line(**{f"f{index}": 4294967295})).cycles[index - 20] == 4294967295


@pytest.mark.parametrize("hz", [0, -1, 1.5, float("nan"), float("inf"), 4294967296])
def test_cycle_conversion_requires_valid_counter_frequency(hz):
    with pytest.raises(ProtocolError):
        parse_y(y_line()).cycle_times_us(hz)


def test_identity_clock_parser_uses_reported_not_assumed_clock():
    assert hil.parse_identity("I 1 " + "a" * 64 + " 80000000 320 64")["counter_hz"] == 80000000


@pytest.mark.parametrize("tail", ["0 320 64", "nan 320 64", "240000000.5 320 64",
                                   "4294967296 320 64", "240000000 320", "240000000 320 64 0"])
def test_identity_clock_rejects_malformed_shapes_and_frequency(tail):
    with pytest.raises(ProtocolError):
        hil.parse_identity("I 1 " + "a" * 64 + " " + tail)


@pytest.mark.parametrize("device_timing", [False, True])
def test_adapter_separates_high_resolution_exchange_rtt_and_device_service(tmp_path, monkeypatch,
                                                                          device_timing):
    binary = hil.build_offline(tmp_path / "build")
    ticks = iter(range(100, 100000, 321))
    monkeypatch.setattr(hil.time, "perf_counter_ns", lambda: next(ticks))
    class TestLink(hil.StdioLink):
        def send(self, line):
            if line == "XI":
                self.lines.put("I 1 " + "a" * 64 + " 80000000 320 64")
            else:
                super().send(line)

        def read_reply(self, deadline):
            line = super().read_reply(deadline)
            if device_timing and line and line.startswith("Y "):
                line += " 0"
            return line
    TestLink.device_timing = device_timing  # test-only fake XI, never device evidence
    with TestLink(binary) as link:
        hil.run_loop(link, ClosedLoopConfig(duration_s=.1), tmp_path / "run")
    records = [json.loads(row) for row in (tmp_path / "run/timing.jsonl").read_text().splitlines()]
    assert all(row["host_exchange_rtt_ns"] == 321 for row in records)
    frames = [row for row in records if row["command"] == "XS2"]
    assert len(frames) == 3
    for frame in frames:
        assert frame["service_cycles"] == (0 if device_timing else None)
        assert frame["service_us"] == (0 if device_timing else None)
        assert frame["counter_hz"] == (80000000 if device_timing else None)
        assert (frame["stage_us"] is None) is (not device_timing)
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["timing"]["host_clock"] == "perf_counter_ns"
    assert manifest["timing"]["algorithm_time_from_rtt"] is False
