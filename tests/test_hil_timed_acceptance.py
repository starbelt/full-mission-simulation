"""Acceptance must understand the instrumented Zephyr reply and open safely."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import hil_stream as H  # noqa: E402


def test_instrumented_zephyr_reply_is_scored_not_discarded():
    frame = H.Frame(0, 0.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
    reply = "R 0 90 0 1 1 1 1 0 0 0 2898664 39140"
    verdict = H.evaluate_replies([frame], [{"raw": reply}], 0.1, 0.1)
    assert verdict.ok, verdict.reasons
    parsed = H.parse_reply(reply)
    assert parsed["quaternion"] == (1.0, 0.0, 0.0, 0.0)
    assert parsed["ephemeris_cycles"] == 2898664
    assert parsed["attitude_cycles"] == 39140


@pytest.mark.parametrize("suffix", ["24", "24 garbage", "24 240 123", "24 2.5"])
def test_incomplete_or_invalid_timing_trailer_fails(suffix):
    with pytest.raises(ValueError):
        H.parse_reply("R 0 90 0 1 1 1 1 0 0 0 " + suffix)


def test_standard_stream_configures_modem_lines_before_open(monkeypatch):
    import serial

    class Port:
        dtr = True
        rts = True
        in_waiting = 0
        opened = False
        closed = False

        def open(self):
            assert self.dtr is False and self.rts is False
            assert self.port == "COM5"
            assert self.write_timeout is not None
            self.opened = True

        def close(self):
            self.closed = True

        def reset_input_buffer(self):
            assert self.opened

        def readline(self):
            return b""

    port = Port()

    def factory(*args, **kwargs):
        assert not args and not kwargs, "immediate Serial(port) can pulse native USB reset"
        return port

    monkeypatch.setattr(serial, "Serial", factory)
    assert H.stream("COM5", [], 0.01) == []
    assert port.opened and port.closed


def test_pipeline_bounds_backlog_without_forcing_reply_order(monkeypatch):
    from test_hil_stream_capture import FakeSerial, frame, response, run

    class FiniteDelayedPort(FakeSerial):
        def __init__(self):
            super().__init__()
            self.pending = []
            self.max_backlog = 0

        @property
        def in_waiting(self):
            # Replies become readable only when the host actually waits.
            return 0

        def write(self, data):
            seq = int(data.split()[1])
            self.pending.append(seq)
            self.max_backlog = max(self.max_backlog, len(self.pending))
            assert len(self.pending) <= 16, "unbounded sender overran target queue"
            self.writes.append(data)
            return len(data)

        def readline(self):
            if self.pending:
                # Deliberately out of order: any reply can free a window slot.
                return (response(self.pending.pop()) + "\n").encode()
            return b""

    port = FiniteDelayedPort()
    replies = run(monkeypatch, port, [frame(i) for i in range(40)], timeout=0.4)
    assert port.max_backlog == 16
    assert sorted(H.parse_reply(r["raw"])["seq"] for r in replies) == list(range(40))
    assert H.parse_reply(replies[0]["raw"])["seq"] == 15


def test_final_reply_timeout_starts_after_sending_not_before(monkeypatch):
    from test_hil_stream_capture import FakeSerial, frame, response, run

    now = [0.0]
    monkeypatch.setattr(H.time, "monotonic", lambda: now[0])

    class SlowSender(FakeSerial):
        @property
        def in_waiting(self):
            return 0

        def write(self, data):
            now[0] += 1.0
            return super().write(data)

        def readline(self):
            now[0] += 0.01
            return super().readline()

    port = SlowSender(by_send={i: [response(i)] for i in range(3)})
    replies = run(monkeypatch, port, [frame(i) for i in range(3)], timeout=0.1)
    assert [H.parse_reply(r["raw"])["seq"] for r in replies] == [0, 1, 2]
