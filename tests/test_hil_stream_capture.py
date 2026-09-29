"""Fake-serial checks for append-only, sequence-keyed HIL capture."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import hil_stream  # noqa: E402


def frame(seq):
    return hil_stream.Frame(
        seq, float(seq), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )


def frame_small_mag(seq, mag=(1.0e-5, 0.0, 0.0)):
    """Frame whose ORIGINAL magnetometer magnitude is ~1e-5 T, like real HIL."""
    return hil_stream.Frame(
        seq, float(seq), (0.0, 0.0, 0.0), mag,
        (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )


def r(seq, angle="90", rate="0", mnorm="1", snorm="1", flags="1",
      qw="1", qx="0", qy="0", qz="0"):
    return f"R {seq} {angle} {rate} {mnorm} {snorm} {flags} {qw} {qx} {qy} {qz}"


def response(seq):
    return f"R {seq} 90 0 1 1 1 1 0 0 0"


class FakeSerial:
    def __init__(self, by_send=None, delayed=None, fail_on=None):
        self.by_send = by_send or {}
        self.delayed = list(delayed or [])
        self.fail_on = fail_on
        self.queue = []
        self.writes = []
        self.closed = False

    def reset_input_buffer(self):
        pass

    def write(self, data):
        seq = int(data.decode().split()[1])
        if seq == self.fail_on:
            raise OSError("write failed")
        self.writes.append(data)
        self.queue.extend(self.by_send.get(seq, []))
        return len(data)

    def flush(self):
        pass

    @property
    def in_waiting(self):
        return len(self.queue)

    def readline(self):
        if self.queue:
            return (self.queue.pop(0) + "\n").encode()
        if self.delayed:
            return (self.delayed.pop(0) + "\n").encode()
        return b""

    def close(self):
        self.closed = True


def run(monkeypatch, serial, frames, capture=None, timeout=0.03):
    import serial as serial_module

    monkeypatch.setattr(serial_module, "Serial", lambda *args, **kwargs: serial)
    return hil_stream.stream("FAKE", frames, timeout, capture_path=capture)


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_out_of_order_delayed_replies_are_joinable_by_seq(monkeypatch, tmp_path):
    frames = [frame(i) for i in range(3)]
    serial = FakeSerial(delayed=[response(2), response(0), response(1)])
    path = tmp_path / "capture.jsonl"
    run(monkeypatch, serial, iter(frames), str(path))
    rows = records(path)
    sends = {row["seq"]: row for row in rows if row["kind"] == "send"}
    replies = [row for row in rows if row["kind"] == "reply"]
    assert [row["seq"] for row in replies] == [2, 0, 1]
    assert set(sends) == {0, 1, 2}
    for row in replies:
        seq = row["seq"]
        assert row["parsed"]["seq"] == seq
        assert row["raw"] == response(seq)
        assert row["latency_s"] >= 0
        assert sends[seq]["line"] == frames[seq].to_line()
        assert sends[seq]["truth_quaternion"] == pytest.approx(
            frames[seq].truth_quaternion()
        )
    assert serial.closed


def test_malformed_idle_unmatched_and_duplicate(monkeypatch, tmp_path):
    path = tmp_path / "capture.jsonl"
    serial = FakeSerial(by_send={0: ["R 0 partial", response(-2), response(99),
                                     response(0), response(0)]})
    run(monkeypatch, serial, [frame(0)], str(path))
    assert [row["kind"] for row in records(path)] == [
        "send", "parse_error", "idle", "unmatched", "reply", "unmatched"
    ]
    assert serial.closed


def test_no_capture_and_failed_write_never_records_send(monkeypatch, tmp_path):
    serial = FakeSerial(by_send={0: [response(0)]})
    assert run(monkeypatch, serial, [frame(0)]) == [{"raw": response(0)}]
    assert serial.closed
    path = tmp_path / "capture.jsonl"
    serial = FakeSerial(fail_on=0)
    with pytest.raises(OSError, match="write failed"):
        run(monkeypatch, serial, [frame(0)], str(path))
    assert records(path) == []
    assert serial.closed


def test_main_exits_nonzero_on_mismatching_reply(monkeypatch):
    # Quaternion (0,1,0,0) is 90 deg off the identity truth quaternion of
    # frame(0) -> attitude check must fail -> MISMATCH -> nonzero exit.
    mismatch = "R 0 90 0 1 1 1 0 1 0 0"
    monkeypatch.setattr(hil_stream, "build_frames", lambda *_: [frame(0)])
    monkeypatch.setattr(hil_stream, "stream", lambda *a, **k: [{"raw": mismatch}])
    monkeypatch.setattr(sys, "argv", ["hil_stream"])
    with pytest.raises(SystemExit) as exc:
        hil_stream.main()
    assert exc.value.code == hil_stream.EXIT_MISMATCH
    assert exc.value.code != 0


def test_main_exits_zero_on_matching_reply(monkeypatch):
    monkeypatch.setattr(hil_stream, "build_frames", lambda *_: [frame(0)])
    monkeypatch.setattr(hil_stream, "stream", lambda *a, **k: [{"raw": response(0)}])
    monkeypatch.setattr(sys, "argv", ["hil_stream"])
    with pytest.raises(SystemExit) as exc:
        hil_stream.main()
    assert exc.value.code == hil_stream.EXIT_OK


def test_attitude_error_is_full_rotation_angle():
    q = (math.cos(math.radians(5)), math.sin(math.radians(5)), 0.0, 0.0)
    error = hil_stream._quat_error_deg(q, frame(0).truth_quaternion())
    assert error == pytest.approx(10.0, abs=1e-5)


def test_nonunit_quaternion_is_rejected():
    reply = [{"raw": "R 0 90 0 1 1 1 2 0 0 0"}]
    verdict = hil_stream.evaluate_replies([frame(0)], reply, 0.1, 2.0)
    assert not verdict.ok
    assert verdict.bad_quat_seqs == (0,)


def test_wrong_mag_norm_is_rejected():
    # frame(0) has |mag| = 1.0, |sun| = 1.0, |gyro| = 0.0; a reply that
    # reports the normalized/nonsense norms the firmware regressed to, or a
    # wrong rate, must not pass a finiteness-only check.
    wrong = "R 0 90 999 999 999 1 1 0 0 0"
    verdict = hil_stream.evaluate_replies([frame(0)], [{"raw": wrong}], 0.1, 2.0)
    assert not verdict.ok
    assert verdict.bad_norm_seqs == (0,)


def test_wrong_rate_is_rejected():
    # frame(0) is a zero-rate frame; a reply reporting 999 rad/s must fail.
    wrong = "R 0 90 999 1 1 1 1 0 0 0"
    verdict = hil_stream.evaluate_replies([frame(0)], [{"raw": wrong}], 0.1, 2.0)
    assert verdict.bad_norm_seqs == (0,)  # rate is covered by the same check
    assert not verdict.ok


def test_firmware_parse_error_reply_fails_acceptance():
    # A well-formed R -1 (flags=4) means the firmware could not parse a
    # frame. It must fail the run even when every seq also has a good reply.
    replies = [{"raw": response(0)}, {"raw": "R -1 0 0 0 0 4 1 0 0 0"}]
    verdict = hil_stream.evaluate_replies([frame(0)], replies, 0.1, 2.0)
    assert not verdict.ok
    assert any("parse" in reason for reason in verdict.reasons)


def test_duplicate_reply_after_all_answered_is_detected(monkeypatch, tmp_path):
    # stream() must drain the port after the last expected seq is answered,
    # otherwise a firmware that answers twice is scored as exactly-once.
    serial = FakeSerial(by_send={0: [response(0), response(0)]})
    path = tmp_path / "capture.jsonl"
    replies = run(monkeypatch, serial, [frame(0)], str(path), timeout=0.2)
    rows = records(path)
    assert [row["kind"] for row in rows] == ["send", "reply", "unmatched"]
    verdict = hil_stream.evaluate_replies([frame(0)], replies, 0.1, 2.0)
    assert not verdict.ok
    assert verdict.duplicate_seqs == (0,)


def test_capture_appends_and_dry_does_not_open_serial(monkeypatch, tmp_path):
    path = tmp_path / "capture.jsonl"
    for _ in range(2):
        run(monkeypatch, FakeSerial(by_send={0: [response(0)]}), [frame(0)], str(path))
    assert [row["kind"] for row in records(path)] == ["send", "reply", "send", "reply"]
    monkeypatch.setattr(hil_stream, "build_frames", lambda *_: [frame(0)])
    monkeypatch.setattr(sys, "argv", ["hil_stream", "--dry", "--capture", str(tmp_path / "dry")])
    hil_stream.main()
    assert not (tmp_path / "dry").exists()


# ---------------------------------------------------------------------------
# Acceptance (evaluate_replies) checks over fake-serial streams: the verdict
# must fail the run for missing, duplicate, malformed, nonfinite, bad-flag,
# and magnitude-corrupted replies -- not just log them.
# ---------------------------------------------------------------------------


def evaluate_via_serial(monkeypatch, by_send, frames):
    replies = run(monkeypatch, FakeSerial(by_send=by_send), iter(frames), timeout=0.02)
    return hil_stream.evaluate_replies(frames, replies, 0.1, 2.0)


def test_missing_reply_fails_acceptance(monkeypatch):
    verdict = evaluate_via_serial(monkeypatch, {0: [response(0)]}, [frame(0), frame(1)])
    assert not verdict.ok
    assert verdict.missing_seqs == (1,)
    assert any("missing" in reason for reason in verdict.reasons)


def test_duplicate_reply_fails_acceptance(monkeypatch):
    verdict = evaluate_via_serial(monkeypatch, {0: [response(0), response(0)]},
                                  [frame(0), frame(1)])
    assert not verdict.ok
    assert verdict.duplicate_seqs == (0,)
    assert any("duplicate" in reason for reason in verdict.reasons)


def test_malformed_reply_fails_acceptance(monkeypatch):
    verdict = evaluate_via_serial(monkeypatch, {0: ["R 0 90 0 1"]}, [frame(0)])
    assert not verdict.ok
    assert verdict.parse_error_count == 1
    assert verdict.missing_seqs == (0,)
    assert any("unparseable" in reason for reason in verdict.reasons)


def test_nonfinite_reply_fails_acceptance(monkeypatch):
    verdict = evaluate_via_serial(monkeypatch, {0: [r(0, qw="inf")]}, [frame(0)])
    assert not verdict.ok
    assert verdict.nonfinite_seqs == (0,)
    assert any("non-finite" in reason for reason in verdict.reasons)


def test_bad_flags_fail_acceptance(monkeypatch):
    by_send = {
        0: [r(0, flags="0")],     # valid bit never set
        1: [r(1, flags="16")],    # degenerate TRIAD
        2: [r(2, flags="4")],     # firmware parse error
    }
    verdict = evaluate_via_serial(monkeypatch, by_send, [frame(0), frame(1), frame(2)])
    assert not verdict.ok
    assert verdict.bad_flag_seqs == (0, 1, 2)
    assert any("flags" in reason for reason in verdict.reasons)


def test_old_mag_norm_one_is_rejected_against_original_value(monkeypatch):
    # frame_small_mag(0) sent the original ~1e-5 T field. A healthy seq,
    # valid flags, unit quaternion, but the legacy mag_norm = 1.0 reply must
    # be rejected: 1.0 is not the magnitude the harness sent.
    verdict = evaluate_via_serial(monkeypatch, {0: [r(0, mnorm="1.0")]},
                                  [frame_small_mag(0)])
    assert not verdict.ok
    assert verdict.bad_norm_seqs == (0,)
    assert any("magnitude" in reason for reason in verdict.reasons)
    assert verdict.max_quat_err == 0.0  # never reached: magnitude check first


def test_correct_original_mag_norm_passes(monkeypatch):
    # The same reply shape with the ORIGINAL magnetometer magnitude is
    # accepted end to end.
    verdict = evaluate_via_serial(monkeypatch, {0: [r(0, mnorm="1e-05")]},
                                  [frame_small_mag(0)])
    assert verdict.ok, verdict.reasons
    assert verdict.max_quat_err == pytest.approx(0.0, abs=1e-5)


def test_corrupted_magnitudes_are_caught(monkeypatch):
    # mag_norm off by 2x (not the 1.0 regression, but still corrupted), and
    # rate_mag corrupted against the zero gyro that was sent.
    verdict = evaluate_via_serial(
        monkeypatch,
        {0: [r(0, mnorm="2e-05")], 1: [r(1, rate="5")]},
        [frame_small_mag(0), frame(1)],
    )
    assert not verdict.ok
    assert verdict.bad_norm_seqs == (0, 1)


def test_quaternion_magnitude_corruption_is_rejected(monkeypatch):
    # Near-unit is required: norm 1.5 is not a unit quaternion.
    verdict = evaluate_via_serial(monkeypatch, {0: [r(0, qw="1.5")]}, [frame(0)])
    assert not verdict.ok
    assert verdict.bad_quat_seqs == (0,)
