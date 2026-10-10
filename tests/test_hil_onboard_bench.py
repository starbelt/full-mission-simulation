"""The core-0 benchmark must not confuse a response field or lose serial lines."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from hil_onboard_bench import exchange_frames, open_board_port, parse_reply


def test_serial_control_lines_are_set_before_open(monkeypatch):
    class Port:
        def __init__(self):
            self.port = None
            self.dtr = True
            self.rts = True
            self.opened = False

        def open(self):
            assert self.dtr is False and self.rts is False
            assert self.port == "COM5"
            self.opened = True

    port = Port()
    monkeypatch.setattr("hil_onboard_bench.serial.Serial", lambda: port)
    assert open_board_port("COM5") is port
    assert port.opened



class FakePort:
    def __init__(self):
        self.lines = []
        self.writes = []

    def write(self, data):
        self.writes.append(data)
        seq = int(data.split()[1])
        self.lines.extend([
            b"R -2 0 0 0 0 8 1 0 0 0 -1 -1\n",
            f"R {seq} 100 0.01 2e-05 1 1 1 0 0 0 2400 24\n".encode(),
        ])
        return len(data)

    def flush(self):
        pass

    def readline(self):
        return self.lines.pop(0) if self.lines else b""


class FakeFrame:
    def __init__(self, seq):
        self.seq = seq

    def to_line(self):
        return f"S2 {self.seq} 1735689600000"


def test_exchange_matches_each_reply_and_ignores_idle():
    port = FakePort()
    replies, latencies = exchange_frames(port, [FakeFrame(4), FakeFrame(5)], 2)
    assert sorted(replies) == [4, 5]
    assert all(v["cyc_eph"] == 2400 for v in replies.values())
    assert len(port.writes) == 2
    assert len(latencies) == 2
    assert all(v >= 0 for v in latencies.values())



def test_timed_reply_preserves_both_cycle_counts():
    line = "R 7 121.16 0.03 2.37e-05 1.0000 1 0.9 -0.1 0.3 -0.2 2898664 39140"
    reply = parse_reply(line)
    assert reply is not None
    assert reply["seq"] == 7
    assert reply["q"] == [0.9, -0.1, 0.3, -0.2]
    assert reply["cyc_eph"] == 2898664
    assert reply["cyc_ads"] == 39140


def test_incomplete_timing_reply_is_rejected():
    assert parse_reply("R 7 121.16 0.03 2.37e-05 1.0 1 0.9 -0.1 0.3 -0.2 2898664") is None


def test_idle_reply_is_not_a_measurement():
    reply = parse_reply("R -2 0 0 0 0 8 1 0 0 0 -1 -1")
    assert reply["seq"] == -2
    assert reply["cyc_eph"] == -1
    assert reply["cyc_ads"] == -1
