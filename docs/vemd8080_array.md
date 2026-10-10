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
c        = n_channel . sun_body              # cosine of incidence
response = c ** q + t * (1 - c)   (c >= 0)   # 0 behind the face
signal   = response * shadow_face * scale    # per-face eclipse + scale
signal  += dark_current                      # per-channel floor
signal  += Gaussian noise (optional)         # seeded (0 => off)
signal  += shot noise (optional)             # std ~ sqrt(I)
reading  = clip(signal, 0, full_scale)       # saturation
reading  = quantized onto the 12-bit ADC grid
```

- `q` is solved so the response is exactly 0.5 at the datasheet's ±65 deg
  half-sensitivity angle (see below); `t` is the reflection-tail fraction.
- Input: body-frame **unit** sun direction (from
  `truth.sun_direction_body`; rows whose norm differs from 1 by more than
  1e-6 raise `ValueError`) and a per-face shadow factor in [0, 1]
  (1 = full light, 0 = total eclipse). `measure` derives the per-face
  factor from the truth: the eclipse factor
  (`truth.illumination_fraction`, Basilisk: 1 = full light, 0 = umbra)
  times a body-shadow indicator — a face whose normal points away from
  the sun is occluded by the body.
- Output: **(N, 6) raw channel readings**, never a 3-axis sun vector.
  Deriving one is `sensors/sun_reconstruction.py`, which inverts the
  response law (both derive from the same constants) and solves a
  clipped least-squares direction.
- `scale` is per-channel and configurable (calibrated photocurrent in A,
  or a normalized electronics scale). Default is `_FULL_SCALE_HEADROOM`
  (0.9) of the full scale on every channel, so a realistic microamp range
  does not pin every lit channel at the clip ceiling.
- `dark_current_a` is per-channel: a scalar (the nominal part) or a (6,)
  array for the per-unit dark-current spread of a real build.
- `noise_rms_a = 0` (default) is the deterministic zero-noise mode; a
  fixed `seed` replays the identical noise stream on every call.
- `shot_noise` (off by default) adds the diode's physical shot noise: a
  zero-mean Gaussian with std `sqrt(2 e B I)` where `I` is the
  deterministic signal current and `B` the Nyquist bandwidth of the
  sample rate. `noise_rms_a` remains the electronics' readout floor.
- ADC quantization rounds the clipped signal onto a 12-bit grid
  (the ESP32-S3's on-chip ADC, which is what the HIL harness models) as
  the last stage of the chain — matching the other hardware models in
  this package, which also quantize.

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
| Angle of half sensitivity | ±65 deg | response exponent anchored to 0.5 at 65 deg |
| Reverse dark current | 0.2 nA typ / 10 nA max at VR = 10 V, E = 0 | default `dark_current_a` = 0.2 nA typ (per-channel; a (6,) array models the spread between 0.2 and 10 nA) |
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

- **Angular response = power law plus reflection tail.**
  `response = c ** q + t * (1 - c)` for `c >= 0` and exactly 0 behind
  the face, where `q = ln(0.5 - t(1 - cos 65°)) / ln(cos 65°)` so the
  datasheet's half-sensitivity point (0.5 at ±65 deg) is exact by
  construction. The power-law *shape* and the tail *shape* interpolate
  the datasheet's two anchors; the anchors are the datasheet figures.
  A pure `clip(cos, 0, 1)` reads 0.423 at 65 deg (~15% under the
  datasheet's own half-sensitivity point) and a hard zero at 90 deg,
  where the datasheet plots a real tail (package and PCB reflections).
  The tail fraction `VEMD8080_REFLECTION_TAIL_FRACTION` = 0.05 is a
  representative value, not a datasheet figure. The law is strictly
  monotone on [0, 1], so the sun reconstruction inverts it
  numerically (a vectorized bisection to machine precision); both
  sides derive from the same constants and cannot drift apart.
- **Linear response** with a single per-channel scale (the
  datasheet's Fig. 3 shows near-linearity; temperature dependence,
  Fig. 2, is not modeled). Passing six distinct `scale` values models
  the per-unit gain spread of a real build (datasheet tolerance
  23/28/33 µA, ±18%).
- **Eclipse = per-face shadow factor** in [0, 1]. A scalar factor (or
  the truth's single `illumination_fraction`) applies to all six
  faces; an (N, 6) factor expresses the real array's per-face
  occlusion — the body shadows the anti-sun face during a backside
  pass, which a single global scalar cannot. `measure` combines the
  truth eclipse factor with the cube's own body-shadow indicator.
  Penumbra/umbra partial shading is taken from the truth model, not
  derived here.
- **Gaussian readout noise** per sample per channel (RMS configurable,
  zero by default) models the electronics' readout floor. The
  datasheet specifies no readout-noise figure (a property of the ADS-2
  electronics, not the diode).
- **Shot noise** (opt-in, off by default) models the diode's own
  Poisson current: a zero-mean Gaussian with std `sqrt(2 e B I)`, `I`
  the deterministic signal current, `B` the single-pole bandwidth
  (the Nyquist of the sample rate). At the array's low bandwidth the
  readout floor usually dominates, which is why it is a separate term
  rather than the default.
- **Dark-current floor** added on every sample (including eclipse),
  defaulting to the 0.2 nA typ. Per-channel: a scalar (nominal) or a
  (6,) array (per-unit spread within the 0.2 nA typ / 10 nA max range).
- **Clipping** at `full_scale_a` (default 1e-5 A, a transimpedance
  front-end range matched to the diode's microamp photocurrent)
  represents the ADC/electronics full scale, not a diode limit.
- **ADC quantization** onto a 12-bit grid (the ESP32-S3's on-chip ADC,
  which the HIL harness models) as the last stage of the chain. The
  datasheet specifies no ADC of its own; 12 bits is an engineering
  choice tied to the readout hardware, not a part parameter.
- **Cube face layout** (+/-X, +/-Y, +/-Z) is the ADS-2 assembly choice;
  the datasheet does not prescribe a cube array.
- **Scale is left configurable** on purpose: the 850 nm peak
  responsivity must NOT be multiplied by the broadband solar constant
  (proper spectral weighting of the solar spectrum against the diode's
  350-1100 nm response is required, and the diode + electronics gain is
  unknown until calibrated). A calibrated per-channel gain/current or a
  normalized electronics scale is the intended way to set it.

## What this model does NOT do

- It does **not** output a 3-axis sun vector. Deriving one from the six
  channels is `sensors/sun_reconstruction.py` (a clipped least-squares
  solve over the lit faces, gated on a minimum/maximum lit-face count
  and a per-sample lit-floor to avoid eclipse ambiguity).
- It does not model the diode's transit response (rise/fall ~70-100 ns)
  or its capacitance (40-47 pF).
- It does not model temperature effects.

## Test coverage (tests/test_vemd8080_array.py, tests/test_sun_reconstruction.py)

- Face geometry: unit sun along an axis lights exactly that face at its
  scale, reads 0 on the anti-sun face, and reads the reflection tail on
  the edge-on faces (not the old hard zero); oblique sun gives the
  anchored power-law-plus-tail response on the lit faces.
- Angular response (datasheet anchors): 0.5 at the 65 deg
  half-sensitivity angle (exact by construction), 1.0 at normal
  incidence, a finite tail at 90 deg (not a hard cliff), monotone decay
  into the horizon, and exactly 0 beyond it.
- Eclipse: total shadow reads only the dark-current floor; zero floor
  reads exactly zero; partial shadow scales the visible channels; a
  per-face (N, 6) shadow factors each face independently (body shadow
  of the anti-sun face) and wrong shapes are rejected.
- Per-channel scale applies independently.
- Per-channel dark current: a (6,) array applies each floor
  independently (per-unit spread); a scalar broadcasts; the stored
  attribute is (6,).
- Full-scale saturation clips lit channels and the dark floor.
- Noise: seeded determinism (same seed == identical, different seed
  !=), configured RMS tracks the noise scale, dark channels show
  half-wave-rectified positive leakage at the 0 floor, and the noise
  survives the ADC at a realistic full scale.
- Shot noise: opt-in, scales with the signal current (a lit channel
  shows more absolute spread than the same channel at a quarter signal),
  and the deterministic mode is unchanged when it is off.
- Zero-noise mode is exact and repeatable, code for code.
- ADC quantization: a 2001-angle sweep collapses to a bounded number of
  distinct values, every output lies on the 12-bit grid, full scale and
  zero are exact, and the bit count is validated (default 12).
- Invalid configs (rate, scale shape/sign/NaN, dark current, noise,
  full scale, adc_bits) and invalid inputs (bad sun shape, non-unit sun
  rows, non-finite sun, shadow length/range) all raise `ValueError`.
- Reconstruction: a perfect (zero-noise, zero-dark-current) array
  recovers the sun direction to ADC-limited accuracy (the
  response-law inversion is exact); the noise-free bound is
  ~1e-2 deg (worst-case 12-bit quantization), and a specific oblique
  three-face geometry sits at 0.010 deg — a quantization limit, not a
  reconstruction bias (the unquantized error is 0.0 deg).
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
- The 90 deg reflection-tail fraction (0.05) is a representative
  engineering value interpolating the datasheet's Fig. 6, not a
  figure read off the document; the 65 deg half-sensitivity anchor is
  the datasheet's, and the exponent is derived from it.
- No real ADS-2 electronics calibration exists yet, so the default
  scale (0.9 of the full scale on every channel) is a nominal
  matched-unit case, not a physical current; per-unit spread is modeled
  by passing a (6,) `scale` and a (6,) `dark_current_a`.
