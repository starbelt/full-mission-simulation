"""Firmware attitude error over real HIL frames, per-sample.

Runs the exact onboard sequence (HilStreamSource::parse ->
EphemerisReferenceProvider::references -> adsCompute) over frames from an
increasingly long window, and records the per-sample attitude error against
the Python truth quaternion so the error distribution can be plotted rather
than just quoted as a max.

The error metric is 2*acos(|dot|): the plain arccos of the trace relation is
the half angle and halves every error, which manufactures false passes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\drago\Nextcloud\School - NextCloud\Research\pocketadcs-sim")
sys.path.insert(0, str(REPO / "examples"))
sys.path.insert(0, str(REPO / "src"))
import hil_stream as H  # noqa: E402

CPP = r"""
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "ads_math.h"
#include "sensor_source.h"
#include "reference_provider.h"

int main(int argc, char** argv) {
  FILE* fh = fopen(argv[1], "r");
  if (!fh) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  // One chunk at a time, passed in as argv[2..3] (start, count): passing every
  // frame on the command line overflows Windows' 32k limit, and a fixed buffer
  // silently drops the tail (which is how 6001 frames once came back as 4000).
  static char lines[2048][192];
  int n = 0;
  int want = (argc > 3) ? atoi(argv[3]) : 2048;
  while (n < want && fgets(lines[n], 192, fh)) {
    size_t L = strlen(lines[n]);
    while (L && (lines[n][L-1] == '\n' || lines[n][L-1] == '\r')) lines[n][--L] = 0;
    if (L) n++;
  }
  fclose(fh);
  for (int i = 0; i < n; i++) {
    HilStreamSource src; SensorSample s;
    if (!src.parse(lines[i], s)) { printf("P BADPARSE\n"); continue; }
    EphemerisReferenceProvider eph;
    eph.references(s.epoch_ms, s.pos_gcrs_m, s);
    AdsFrameResult r;
    adsCompute(s.gyro_rad_s, s.mag_ut, s.sun_unit,
               s.ref_sun_unit, s.ref_mag_unit, &r);
    printf("R %d %u %.9f %.9f %d %.9f %.9f %.9f %.9f\n",
           (int)s.seq, (unsigned)s.flags, r.angle_deg, r.rate,
           (int)r.valid, r.q[0], r.q[1], r.q[2], r.q[3]);
  }
  return 0;
}
"""


def angle_between_deg(q_est_wxyz, q_truth_wxyz):
    """2*acos(|dot|) -- the full angle, sign-ambiguity invariant."""
    a = np.asarray(q_est_wxyz, dtype=float)
    b = np.asarray(q_truth_wxyz, dtype=float)
    d = abs(float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0))))


if __name__ == "__main__":
    frames = list(H.build_frames(600.0, 10.0, "hardware", 7))
    td = Path(tempfile.mkdtemp(prefix="fwerr_"))
    (td / "e.cpp").write_text(CPP, encoding="ascii")
    sys.path.insert(0, str(REPO / "tests"))
    from test_ads_math_host import _env_for, find_usable_compiler

    cxx = find_usable_compiler(td)
    r = subprocess.run([str(cxx), "-std=c++17", "-O2",
                        "-I", str(REPO / "esp32" / "src"),
                        str(td / "e.cpp"), "-o", str(td / "e.exe")],
                       capture_output=True, text=True, env=_env_for(cxx),
                       timeout=180)
    if r.returncode != 0:
        print("COMPILE FAILED:\n", r.stderr[:2000])
        raise SystemExit(1)

    lines = [f.to_line() for f in frames]
    CHUNK = 2000
    rows = {}
    for start in range(0, len(lines), CHUNK):
        chunk = lines[start:start + CHUNK]
        ff = td / "frames.txt"
        ff.write_text("\n".join(chunk) + "\n", encoding="ascii")
        out = subprocess.run(
            [str(td / "e.exe"), str(ff), str(len(chunk))],
            capture_output=True, text=True,
            env=_env_for(td / "e.exe"), timeout=300)
        for ln in out.stdout.splitlines():
            q = ln.split()
            if q[0] == "R":
                rows[int(q[1])] = q
        if out.returncode != 0:
            print(f"chunk at {start} rc={out.returncode}: {out.stderr[:200]}")

    errs, invalid, by_sensor = [], 0, []
    for f in frames:
        q = rows.get(f.seq)
        if q is None:
            continue
        if int(q[5]) == 0:
            invalid += 1
            continue
        errs.append(angle_between_deg([float(x) for x in q[6:10]],
                                      f.truth_quaternion()))

    errs = np.array(errs)
    print(f"frames {len(frames)}  replied {len(rows)}  degenerate {invalid}")
    print(f"attitude error vs truth: min {errs.min():.4f}  median "
          f"{np.median(errs):.4f}  mean {errs.mean():.4f}  p95 "
          f"{np.percentile(errs, 95):.4f}  max {errs.max():.4f} deg")

    dest = REPO / "results" / "timing"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "firmware_vs_truth.json").write_text(json.dumps({
        "error_deg": errs.tolist(),
        "frames": len(frames),
        "replies": len(rows),
        "degenerate": invalid,
        "median_deg": float(np.median(errs)),
        "max_deg": float(errs.max()),
    }, indent=2), encoding="utf-8")
    print("wrote", dest / "firmware_vs_truth.json")
