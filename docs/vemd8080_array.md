# VEMD8080 six-photodiode sun array model

A hardware-faithful simulation of the six Vishay **VEMD8080** silicon PIN
photodiodes selected for the ADS-2 body-frame sun sensor array. This is a
bounded, self-contained slice in new files only:

- `src/pocketadcs_sim/sensors/vemd8080_array.py` — `VEMD8080ArrayModel`
- `tests/test_vemd8080_array.py`
- `docs/vemd8080_array.md` (this file)

Nothing existing was modified. The ideal sun sensor
(`sensors/sun_sensor.py`) and the hardware magnetometer
(`sensors/hardware_models.py`) are untouched.

## What the model does

Six raw body-frame channels, one photodiode per cube face, with explicit
outward normals **+X, -X, +Y, -Y, +Z, -Z**. Per channel, per sample:

```
visible = max(0, n_channel . sun_body)     # cosine visibility
signal  = visible * shadow_factor * scale  # eclipse + per-channel scale
signal  = signal + dark_current            # floor (0 disables)
signal  = signal + noise                   # seeded Gaussian (0 = exact)
reading = clip(signal, 0, full_scale)      # saturation
```

- Input: body-frame unit sun direction (from
  `truth.sun_direction_body`) and the truth eclipse shadow factor
  (`truth.illumination_fraction`, Basilisk: 1 = full light, 0 = umbra).
- Output: **(N, 6) raw channel readings**, never a 3-axis sun vector.
- `scale` is per-channel and configurable (calibrated photocurrent in A,
  or a normalized 0..1 electronics scale). Default is all-ones.
- `noise_rms_a = 0` (default) is the deterministic zero-noise mode; a
  fixed `seed` replays the identical noise stream on every call.

## Verified datasheet values (documented spec)

Source: Vishay **VEMD8080 datasheet, Document Number 84565, Rev. 1.1,
05-Sep-2023**, primary
`https://www.vishay.com/docs/84565/vemd8080.pdf` (fetched and read
directly; 7-page PDF, text layer extracted). Values below were read from
the PRODUCT SUMMARY and BASIC CHARACTERISTICS tables.

| Quantity | Documented value | Used in model |
| --- | --- | --- |
| Wavelength of peak sensitivity | 850 nm | constant only (`VEMD8080_PEAK_WAVELENGTH_NM`) |
| Spectral bandwidth | 350 to 1100 nm | constant only |
| Angle of half sensitivity | ±65 deg | constant only (angular roll-off not modeled) |
| Reverse dark current | 0.2 nA typ / 10 nA max at VR = 10 V, E = 0 | default `dark_current_a` = 0.2 nA typ |
| Operating temperature | -40 to +85 C | not modeled (no temperature in signal chain) |
| Radiant sensitive area | 4.6 mm² | not modeled (scale absorbs geometry) |
| Reverse voltage (absolute max) | 20 V | not modeled |

**Verification note (Rev. 1.1, 05-Sep-2023, re-verified):** the dark-current
row and the peak-sensitivity row were read off the BASIC CHARACTERISTICS
table on page 2. Two earlier values in this file were **wrong and have been
corrected**: dark current was recorded as 10 nA typ / 20 nA max (the real
spec is 0.2 nA typ / 10 nA max — a 50x error in the eclipse floor) and peak
sensitivity as 950 nm (the 950 nm in the document is the *test condition*
wavelength for the Voc/Isc/TK rows, not the peak; the peak is 850 nm, the
same wavelength the responsivity is specified at).

The text layer of this PDF is column-scrambled by `pdftotext` — values appear
in a separate block from their parameter names, which is what allowed the
original misread. Extract page 2 with `pdftotext -layout -f 2 -l 2` and align
by hand before trusting any figure here. Distributor pages (e.g. "1 nA" or
"10 nA max" variants, 4.5 mm² vs 4.6 mm² area) disagree and are **not** used.

Note the datasheet's *reverse light current* rows (23/28/33 µA at 850 nm,
3.4/4.4/5.3 µA at 525 nm) are a real two-point spectral responsivity
calibration — roughly 2.8 mA/W at 850 nm. That is enough to derive a physical
`scale` and weight the solar spectrum; the per-channel `scale` is currently
left configurable rather than derived, and should be revisited if photocurrent
fidelity becomes load-bearing.

## Engineering assumptions (NOT datasheet figures)

These are modeling choices, explicitly separated from the verified
specs above:

- **Cosine visibility** `max(0, n . sun)` is the standard ideal
  detector convention. The datasheet's ±65 deg angle-of-half-sensitivity
  (an angular roll-off of the response) is **not** modeled; the true
  angular response would taper the visible channels beyond ~65 deg.
- **Linear response** with a single per-channel scale. The datasheet
  plots relative light current vs irradiance (Fig. 3) showing the
  expected near-linearity; temperature dependence of the photocurrent
  (Fig. 2) is not modeled.
- **Eclipse = hard shadow factor** (0 or 1, or the truth fractional
  value). Penumbra/umbra partial shading is taken from the truth model,
  not derived here.
- **Gaussian noise** per sample per channel (RMS configurable, zero by
  default). The datasheet specifies no readout-noise figure (that is a
  property of the ADS-2 electronics, not the diode); the dark current
  itself contributes shot noise in reality, which is lumped into the
  configurable noise term.
- **Dark-current floor** added on every sample (including eclipse),
  defaulting to the 10 nA typical.
- **Clipping** at `full_scale_a` (default 1.0) represents the ADC /
  electronics full scale, not a diode limit.
- **Cube face layout** (+/-X, +/-Y, +/-Z) is the ADS-2 assembly choice;
  the datasheet does not prescribe a cube array.
- **Scale is left configurable** on purpose: the 950 nm peak
  responsivity must NOT be multiplied by the broadband solar constant
  (proper spectral weighting of the solar spectrum against the diode's
  350-1100 nm response is required, and the diode + electronics gain is
  unknown until calibrated). A calibrated per-channel gain/current or a
  normalized electronics scale is the intended way to set it.

## What this model does NOT do

- It does **not** output a 3-axis sun vector. Deriving one from the six
  channels requires an explicit reconstruction/visibility algorithm
  (e.g. weighted sum of visible faces, gated on a minimum visible-face
  count to avoid eclipse ambiguity), which is out of scope for this
  slice and must be specified and tested in a later bounded card.
- It does not model the diode's transit response (rise/fall ~70-100 ns),
  capacitance (40-47 pF), or the ADC sampling/quantization of the
  electronics.
- It does not model temperature effects.

## Test coverage (tests/test_vemd8080_array.py)

- Face geometry: unit sun along each axis lights exactly one face;
  oblique sun gives cosine visibility on the lit faces and exactly zero
  on the others.
- Eclipse: total shadow reads only the dark-current floor; zero floor
  reads exactly zero; partial shadow scales the visible channels.
- Per-channel scale applies independently.
- Full-scale saturation clips lit channels and the dark floor.
- Noise: seeded determinism (same seed == identical, different seed
  !=), configured RMS tracks the noise scale, dark channels show
  half-wave-rectified positive leakage at the 0 floor.
- Zero-noise mode is exact and repeatable.
- Invalid configs (rate, scale shape/sign/NaN, dark current, noise,
  full scale) and invalid inputs (bad sun shape, non-finite sun, shadow
  length/range) all raise `ValueError`.
- Integration: `measure(truth)` samples the truth environment, stays in
  [0, full_scale], lights at least one face in full light, and reads
  exactly the floor during shadow.

## Limitations (honest notes)

- The datasheet PDF text layer interleaves table columns, so a few
  spec rows (e.g. rise/fall times per load, VO/TKVO) could not be paged
  unambiguously; only the rows used above were relied on, and each was
  cross-checked against the document's product summary.
- Distributor mirrors of the same part show small discrepancies
  (dark current max 10 vs 20 nA, sensitive area 4.5 vs 4.6 mm²); the
  primary Vishay PDF was used as the sole authority.
- No real ADS-2 electronics calibration exists yet, so the default scale
  (all-ones, normalized) is a placeholder, not a physical current.
