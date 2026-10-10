"""Bounded real-board ADS soak/reopen test; no reset and no truth on the wire.

Run from the repository root with PYTHONPATH=src. Uses versioned output folders
and refuses to mix two soak runs. Instrumented firmware must report CPU cycles;
confirm --clock-hz against the generated Zephyr configuration/timer driver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import hil_onboard_bench as bench
import hil_stream as stream


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--reopens", type=int, default=10)
    parser.add_argument("--clock-hz", type=int, default=240_000_000)
    parser.add_argument("--quat-tolerance-deg", type=float, default=0.1)
    args = parser.parse_args()
    if args.batches <= 0 or args.reopens < 0 or args.clock_hz <= 0:
        parser.error("batches/clock must be positive and reopens nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    if raw_path.exists():
        parser.error("raw.jsonl exists: select a fresh versioned output folder")
    frames = stream.build_frames(10.0, 10.0, "ideal", 7)
    records = []
    us_per_cycle = 1_000_000.0 / args.clock_hz
    print(f"Built {len(frames)} independent orbit/sensor frames", flush=True)
    with bench.open_board_port(args.port) as link:
        stop = time.monotonic() + 4
        idle = bytearray()
        while time.monotonic() < stop:
            idle.extend(link.read(1024))
        if b"R -2 " not in idle:
            raise RuntimeError(f"idle heartbeat not observed: {bytes(idle)!r}")
        print("Idle heartbeat verified", flush=True)
        for batch in range(args.batches):
            raw = []
            timings = []
            for frame in frames:
                sent = time.monotonic()
                link.write((frame.to_line() + "\n").encode())
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    text = link.readline().decode(errors="replace").strip()
                    if text.startswith("R ") and not text.startswith("R -2 "):
                        raw.append({"raw": text})
                        reply = bench.parse_reply(text)
                        if reply and reply["seq"] == frame.seq:
                            if reply["cyc_eph"] < 0 or reply["cyc_ads"] < 0:
                                raise RuntimeError("missing onboard cycle instrumentation")
                            timings.append({
                                "eph_us": reply["cyc_eph"] * us_per_cycle,
                                "ads_us": reply["cyc_ads"] * us_per_cycle,
                                "rtt_ms": (time.monotonic() - sent) * 1000,
                            })
                            break
                else:
                    raise RuntimeError(f"timeout: batch {batch}, seq {frame.seq}")
            deadline = time.monotonic() + 0.25
            while time.monotonic() < deadline:
                text = link.readline().decode(errors="replace").strip()
                if text.startswith("R ") and not text.startswith("R -2 "):
                    raw.append({"raw": text})
            verdict = stream.evaluate_replies(frames, raw, 0.1, args.quat_tolerance_deg)
            record = {"batch": batch, "verdict": asdict(verdict), "timings": timings,
                      "replies": raw}
            records.append(record)
            with raw_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            print(f"batch {batch}: {verdict.replies} replies; OK={verdict.ok}; "
                  f"max attitude error={verdict.max_quat_err:.8f} deg", flush=True)
            if not verdict.ok:
                raise RuntimeError(verdict.reasons)
    # Close/open the actual Windows USB device repeatedly, without esptool,
    # DTR/RTS pulses, physical reset or power removal.
    for attempt in range(args.reopens):
        with bench.open_board_port(args.port) as link:
            link.write((frames[0].to_line() + "\n").encode())
            raw = []
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                text = link.readline().decode(errors="replace").strip()
                if text.startswith("R ") and not text.startswith("R -2 "):
                    raw.append({"raw": text})
            verdict = stream.evaluate_replies([frames[0]], raw, 0.1,
                                             args.quat_tolerance_deg)
            if not verdict.ok:
                raise RuntimeError(f"reopen {attempt}: {verdict.reasons}")
    print(f"{args.reopens} reopen/round-trip cycles passed without reset", flush=True)
    timings = [t for r in records for t in r["timings"]]
    root = Path(__file__).resolve().parents[1]
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                     for name in ("esp32/zephyr/src/main.cpp", "esp32/zephyr/prj.conf",
                                  "esp32/src/reference_provider.h", "esp32/src/igrf.h")}
    summary = {
        "port": args.port, "batches": len(records),
        "frames_sent": len(records) * len(frames),
        "replies": sum(r["verdict"]["replies"] for r in records),
        "max_attitude_error_deg": max(r["verdict"]["max_quat_err"] for r in records),
        "median_ephemeris_us": statistics.median(t["eph_us"] for t in timings),
        "median_triad_us": statistics.median(t["ads_us"] for t in timings),
        "max_ephemeris_us": max(t["eph_us"] for t in timings),
        "median_transport_rtt_ms": statistics.median(t["rtt_ms"] for t in timings),
        "serial_reopens_without_reset": args.reopens,
        "all_batches_pass": all(r["verdict"]["ok"] for r in records),
        "quaternion_tolerance_deg": args.quat_tolerance_deg,
        "clock_hz": args.clock_hz, "source_sha256": source_hashes,
        "inputs": {"sensors": "ideal", "seed": 7, "duration_s": 10, "rate_hz": 10},
        "scope": "Existing stateless S2 ephemeris/TRIAD firmware. Repeated 10-second "
                 "trajectory; not full-orbit accuracy or extended ADCS acceptance.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                                 encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
