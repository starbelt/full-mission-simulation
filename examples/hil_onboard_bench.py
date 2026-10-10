"""Measure the ADS algorithm AS IT RUNS on the ESP32-S3 core 0.

Everything reported here comes off the board. The Python simulation is only
the frame source -- it stands in for the real sensor drivers and contributes
no performance number of its own.

Two measurements, kept separate because they answer different questions:

  execution time  the firmware's own cycle counter, around the ephemeris and
                  around the attitude solution. This is the algorithm's real
                  cost on a 240 MHz core, which is the number that decides
                  whether the ADS can keep up with the sensors.

  end-to-end      host write to reply read, which includes the USB-serial
                  transport. That is a property of the test rig, not the
                  algorithm, so it is reported but never presented as the
                  algorithm's latency.

The attitude error uses 2*acos(|dot|). The arccos-of-trace form is the half
angle and halves every error, which manufactures false passes.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import serial

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))
sys.path.insert(0, str(REPO / "src"))
import hil_stream as H  # noqa: E402

# The board's core clock. k_cycle_get_32 counts at this rate, so cycles / this
# is microseconds. The S3 runs PROCPU at 240 MHz.
CORE_HZ = 240_000_000.0

# Bit 4 of the flags word is set when TRIAD is degenerate; those frames have
# no attitude to compare and must not be scored as passes.
FLAG_DEGENERATE = 16


def parse_reply(line):
    """Parse a 13-token R line, or None if it is not one."""
    q = line.split()
    if len(q) != 13 or q[0] != "R":
        return None
    try:
        return {
            "seq": int(q[1]),
            "angle_deg": float(q[2]),
            "rate": float(q[3]),
            "mag_norm": float(q[4]),
            "sun_norm": float(q[5]),
            "flags": int(q[6]),
            "q": [float(x) for x in q[7:11]],
            "cyc_eph": int(q[11]),
            "cyc_ads": int(q[12]),
        }
    except ValueError:
        return None


def angle_error_deg(q_est, q_truth):
    """2*acos(|dot|) between two w-first quaternions.

    The abs makes it invariant to the double-cover sign, and the factor of 2
    is what makes it the full angle rather than the half angle.
    """
    a = np.asarray(q_est, dtype=float)
    b = np.asarray(q_truth, dtype=float)
    d = abs(float(a @ b) / (float(np.linalg.norm(a) * np.linalg.norm(b))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, max(-1.0, d)))))


def exchange_frames(port, frames, timeout_s):
    """Send one frame at a time and match complete newline-delimited replies.

    Bounded outstanding data avoids filling the USB-Serial/JTAG RX FIFO.
    Reading with readline also preserves lines split across USB packets.
    """
    replies = {}
    latencies = {}
    deadline = time.monotonic() + timeout_s
    for frame in frames:
        sent = time.monotonic()
        port.write((frame.to_line() + "\n").encode())
        port.flush()
        while time.monotonic() < deadline:
            raw = port.readline()
            if not raw:
                continue
            reply = parse_reply(raw.decode(errors="replace").strip())
            if reply is None or reply["seq"] < 0:
                continue
            seq = reply["seq"]
            if seq not in replies:
                replies[seq] = reply
                latencies[seq] = (time.monotonic() - sent) * 1e3
            if seq == frame.seq:
                break
        else:
            break
    return replies, latencies


def open_board_port(port):
    """Set modem line state *before* Windows opens the USB serial device."""
    link = serial.Serial()
    link.port = port
    link.baudrate = 115200
    link.timeout = 0.2
    link.write_timeout = 10
    link.xonxoff = False
    link.rtscts = False
    link.dsrdtr = False
    link.dtr = False
    link.rts = False
    link.open()
    return link


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM5"
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    sensors = sys.argv[3] if len(sys.argv) > 3 else "hardware"
    timeout_s = 90

    frames = list(H.build_frames(n_frames / 10.0, 10.0, sensors, 7))
    print(f"{len(frames)} frames, sensors={sensors}, port={port}")

    s = open_board_port(port)
    s.reset_input_buffer()

    # Prove the board is live before trusting anything it says.
    t0 = time.monotonic()
    banner = b""
    while time.monotonic() - t0 < 3.0:
        banner += s.read(256) or b""
    if b"ADS Zephyr ADS firmware up" not in banner:
        print("NOTE: no banner (it is one-shot, so this is expected on a "
              "board that booted before the port opened)")
    else:
        print("banner seen: firmware is up")

    replies, latencies = exchange_frames(s, frames, timeout_s)
    s.close()

    got = len(replies)
    print(f"replies {got}/{len(frames)}")

    errors, eph_us, ads_us, rtt_ms, degenerate = [], [], [], [], 0
    for f in frames:
        r = replies.get(f.seq)
        if r is None:
            continue
        if r["flags"] & FLAG_DEGENERATE:
            degenerate += 1
            continue
        errors.append(angle_error_deg(r["q"], f.truth_quaternion()))
        if r["cyc_eph"] >= 0:
            eph_us.append(r["cyc_eph"] / CORE_HZ * 1e6)
        if r["cyc_ads"] >= 0:
            ads_us.append(r["cyc_ads"] / CORE_HZ * 1e6)
        rtt_ms.append(latencies[f.seq])

    if not errors:
        print("no scored frames -- board did not answer")
        return 1

    tot = [a + b for a, b in zip(eph_us, ads_us, strict=True)] if eph_us and ads_us else []
    print()
    print("ON-DEVICE, core 0 @ 240 MHz")
    print(f"  ephemeris        median {statistics.median(eph_us):8.1f} us")
    if ads_us:
        print(f"  attitude (TRIAD) median {statistics.median(ads_us):8.1f} us")
    if tot:
        print(f"  total per sample median {statistics.median(tot):8.1f} us")
        print(f"  ceiling          {1e6 / statistics.median(tot):,.0f} samples/s")
    print()
    print(f"  attitude error vs truth: median {statistics.median(errors):.3f} deg"
          f"  p95 {sorted(errors)[int(0.95 * len(errors))]:.3f}"
          f"  max {max(errors):.3f}")
    print(f"  degenerate (TRIAD) frames excluded: {degenerate}")
    if rtt_ms:
        print(f"  host->board->host (transport, NOT the algorithm): "
              f"{statistics.median(rtt_ms):.0f} ms median")

    dest = REPO / "results" / "hil"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "onboard_accuracy.json").write_text(json.dumps({
        "sensors": sensors, "frames_sent": len(frames), "replies": got,
        "scored": len(errors), "degenerate": degenerate,
        "error_deg": errors,
        "core_hz": CORE_HZ,
    }, indent=2), encoding="utf-8")
    (dest / "onboard_latency.json").write_text(json.dumps({
        "core_hz": CORE_HZ,
        "us_per_sample": tot,
        "ephemeris_us": eph_us,
        "attitude_us": ads_us,
        "transport_rtt_ms_median": statistics.median(rtt_ms) if rtt_ms else None,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}/onboard_accuracy.json and onboard_latency.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
