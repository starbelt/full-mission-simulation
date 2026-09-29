# ADS-2 hardware sensor models

Simulation models that reproduce the documented behavior of the selected
ADS-2 flight hardware, layered on the ideal body-frame truth. This is a
first slice: **only the MEMSIC MMC5983MA magnetometer is implemented and
tested** (`src/pocketadcs_sim/sensors/hardware_models.py`,
`tests/test_hardware_sensor_models.py`). The TDK ICM-40609-D and the six
Vishay VEMD8080 photodiodes are deliberately deferred to later bounded
cards (see Deferrals below); no values for them are modeled or assumed here.

The ideal sensor models (`sensors/base.py`, `gyroscope.py`,
`magnetometer.py`, `sun_sensor.py`) and their public API are untouched.

## MMC5983MA model

Class: `MMC5983MAModel` in `pocketadcs_sim.sensors.hardware_models`.

- Input: body-frame magnetic field in **Tesla** (from the truth history,
  `truth.magnetic_field_body_t`).
- Output: (N, 3) array in **Tesla**, sampled at `odr_hz`.
- Signal chain: truth field -> additive Gaussian noise (RMS configurable,
  zero = deterministic zero-noise mode) -> per-axis digital quantization
  to the documented unsigned code range (full-scale saturation) -> additive
  bias in counts -> dequantization.
- Determinism: a fixed `seed` replays the same noise stream on every call
  (re-seeded per call; output depends only on seed + input array).

### Documented spec values used (source-verified)

Source: MEMSIC **MMC5983MA datasheet, Rev A (2019-04-03)**, primary
`https://www.memsic.com/Public/Uploads/uploadfile/files/20220119/MMC5983MADatasheetRevA.pdf`
(MEMSIC Semiconductor Co., Ltd.).

Verification note: values below were read from the datasheet's FEATURES
list and electrical spec table as mirrored at the URLs below (identical
Rev A document, cross-checked across four independent copies). Exact page
numbers are **not verified** — the PDF text layer could not be paged
during this workstream; the feature-list/spec-table values matched in all
four copies.

Mirrors of the same Rev A document:
- `https://cdn.sparkfun.com/assets/a/b/7/7/2/19921-09102019_MMC5983MA_Datasheet_Rev_A-1635338.pdf`
- `https://hallaweb.jlab.org/tech/Detectors/public%5Fhtml/manuals/chip%5Fspecs/M-Z/memsic/MMC5983MA_RevA_4-3-19.pdf`
- `https://datasheet.sic-components.com/MMC5983MA_RevA_4-3-19.pdf`

| Quantity | Documented value | Mode |
| --- | --- | --- |
| Full-scale range per axis | ±8 G | all modes |
| Sensitivity | 16384 counts/G | 18-bit operation |
| Sensitivity | 4096 counts/G | 16-bit operation |
| Null-field output | 131072 counts | 18-bit operation |
| Null-field output | 32768 counts | 16-bit operation |
| LSB resolution | 0.0625 mG | 18-bit |
| LSB resolution | 0.25 mG | 16-bit |
| Max output data rate | 1000 Hz | BW=11, CM_Freq=111 |
| ODR (BW=00/01/10/11) | 50 / 100 / 225 / 580 Hz | — |
| Total RMS noise (BW=00/01/10/11) | 0.4 / 0.6 / 0.8 / 1.2 mG | 18-bit |
| Output format | unsigned, 18-bit (Xout[17:0]) / 16-bit (Xout[17:2]) per axis | — |

The unsigned, offset-based transfer function (code = null offset +
field × counts/G, clipped to the unsigned range) is the format documented
in the datasheet register map and spec table; it is **not** a signed
two's-complement code.

### Engineering assumptions (NOT datasheet figures)

- Noise is modeled as per-sample, per-axis zero-mean Gaussian with
  configurable RMS (datasheet specifies *total* RMS noise at each ODR; the
  per-sample Gaussian is an engineering approximation, and cross-axis
  correlation is not modeled).
- Quantization rounds to the nearest count (nearest-integer transfer).
- `bias_counts` (default 0) represents a user-specified DC offset in
  counts; the datasheet's temperature-dependent null-field drift and the
  SET/RESET degauss function are **not** modeled.
- Any `odr_hz` in (0, 1000] Hz is accepted; the BW-band ODRs are exposed
  through `MMC5983_BANDS` for configuration reference.
- Sensitivity accuracy (±5 % over ±8 G), sensitivity change over
  temperature, heading accuracy, repeatability error, and the on-chip
  temperature sensor are documented properties **not** modeled here.

## Deferrals (no values assumed in this slice)

- **TDK ICM-40609-D gyro + accelerometer.** TDK InvenSense part (TDK
  product page:
  `https://www.invensense.tdk.com/en-us/products/6-axis/icm-40609-d`).
  Candidate datasheet: TDK DS-000330 v1.2 (manufacturer PDF URL from the
  task brief; not extracted during this slice). A later bounded card will
  model ranges, ODR, noise, bias/drift with cited values. Note for that
  card: the accelerometer reads *apparent* acceleration (specific force
  with gravity reaction in orbit); it must not be presented as inertial
  acceleration unless the truth supplies specific force.
- **Vishay VEMD8080 × 6 body-frame sun sensor array.** Datasheet URL from
  the task brief: `https://www.vishay.com/docs/84565/vemd8080.pdf` (to be
  verified). A later bounded card will define six face normals, a
  calibrated/normalized per-channel response with an explicit external
  electronics scale, eclipse/no-light behavior (zero or dark-current floor),
  and keep the raw six channels — never an implicit direct 3-axis vector
  without a stated reconstruction/visibility algorithm. Do not multiply the
  950 nm peak responsivity by the broadband solar constant (spectral
  weighting required).

## Test coverage (tests/test_hardware_sensor_models.py)

- Null field -> null offset code, zero Tesla output.
- Full-scale clipping at ±8 G FSR and beyond.
- Unit conversions (G <-> T) and quantize/dequantize round-trip within
  half an LSB.
- Seeded determinism (same seed == identical output, different seed !=).
- Noise scale matches configured RMS (within factor 2 on 200k samples).
- Deterministic zero-noise mode is exact and repeatable.
- Bias in counts shifts output by exactly bias × LSB.
- Configuration errors (ODR, bits, noise, bias, malformed field array) all
  raise `ValueError`.
- Integration: `measure(truth)` samples the truth environment at the ODR
  and stays within half an LSB of the ideal truth (LEO field < FSR).

## Integration plan (not implemented yet)

1. Later cards add the ICM-40609-D and VEMD8080-array models in
   `hardware_models.py` with the same "documented value vs assumption"
   separation.
2. Once the idealized ESP32 firmware workstream (task t_5f83af93) is
   complete, the hardware models can be exposed alongside the ideal models
   (e.g., a selectable sensor backend in the sensing layer) so
   attitude-determination studies can run on hardware-faithful readings.
3. Nothing here is committed or flashed; no existing ideal model, export,
   or shared test file was modified.
