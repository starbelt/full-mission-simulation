# Root cause: ESP32 serial magnitude corruption (ADS-1)

## Symptom
HIL frames over the serial protocol `R seq angle rate mag_norm sun_norm flags qw qx qy qz`
reported `mag_norm = 1.0` (and `sun_norm = 1.0`) on every frame, instead of the original
magnitudes of the raw sensor vectors (~1e-5 T magnetometer field, ~1.0 sun vector in the
study units). Downstream consumers could not distinguish "sensor read zero" from a healthy
reading, and the magnetometer reading was effectively invisible.

## Root cause
`triadDcm(v1, v2, w1, w2, A)` in `esp32/src/ads_math.h` normalized its four input arrays
**in place** before building the TRIAD DCM. `adsCompute` (same file) then emitted
`vecNorm(in.mag)` — and a `sunNorm` cached *after* the TRIAD step — so both vectors had
already been overwritten with unit vectors. `mag_norm`/`sun_norm` therefore always read 1.0
rather than the original magnitudes. The corruption was introduced into the shared header
that both the ESP32-S3 firmware and the host regression tests compile, so the on-board and
host behavior matched in the bug.

## Fix (commit 8db6452, 2026-09-29, branch main)
- `triadDcm` now copies each input into a local array (`nv1..nw2`) and normalizes the
  copies; inputs are never mutated. The header documents the old in-place bug.
- `adsCompute` caches `sun_norm` **before** the TRIAD step and reports the ORIGINAL
  magnitudes in `mag_norm` / `sun_norm` (fields documented as such in the result struct).
- Protocol unchanged: `R seq angle rate mag_norm sun_norm flags qw qx qy qz`, exactly as
  the existing HIL harness parses it.

## Verification
- Host regression tests compiled against the exact firmware header:
  `tests/test_ads_math_host.py`, `tests/test_ads_estimate_host.py`,
  `tests/test_hil_onboard_bench.py` — 13 passed.
- Full repo suite: `pytest -q` = 225 passed (re-run 2026-09-30).
- `pio run -e esp32s3_ads` = SUCCESS (RAM 5.8%, Flash 20.9%).
- RED→GREEN confirmed for the magnitude tests: failing on the pre-fix header, passing after.

## Out of scope
The separate seq>=1 live quaternion divergence (~9 deg against truth on the board) is a
distinct issue and is NOT addressed by this fix; it is tracked separately.
