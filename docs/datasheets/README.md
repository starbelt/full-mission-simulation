# Datasheets

Vendor datasheets for every part whose published specification is used as a
modelling input. They are vendored here so the numbers a sensor or rotor model
depends on can be checked offline, against the same revision the code cites.

| File | Part | Revision | Size | Pages | SHA-256 |
|---|---|---|---:|---:|---|
| [`icm-40609-d_ds-000330_v1.2.pdf`](icm-40609-d_ds-000330_v1.2.pdf) | TDK InvenSense ICM-40609-D 6-axis IMU | DS-000330 Rev 1.2 (2022-06-07) | 4,250,288 B | 86 | `c457a8ed1b946c46c7bdb81f6a94dee4b12c0f3f6431d9199cd2744c0a960dd8` |
| [`MMC5983MADatasheetRevA.pdf`](MMC5983MADatasheetRevA.pdf) | MEMSIC MMC5983MA 3-axis magnetometer | Rev A (2019-04-03) | 1,183,817 B | 21 | `d82083b922c2421757035ceb064e6f316a8f8c20b4593277c13ef750115731f7` |
| [`TGR-29-12-Datasheet-Rev-B.pdf`](TGR-29-12-Datasheet-Rev-B.pdf) | TGR 29-12 reaction-wheel micro motor | Rev B | 340,665 B | 3 | `e227a5d0f5b2e462d20dbdd56a5d898ec3f3059c31f640e2d56a1f191446ae22` |
| [`vemd8080.pdf`](vemd8080.pdf) | Vishay VEMD8080 silicon PIN photodiode | Document 84565 Rev 1.1 (2023-09-05) | 160,201 B | 7 | `9fc7401a0d6d420c21e5d1f69b2930894378d65486fd27d8fbb320af1ad998c6` |

Verify a file before trusting it:

```bash
sha256sum docs/datasheets/*.pdf
```

## Where each one is used

| Part | Model | Notes |
|---|---|---|
| ICM-40609-D | `src/pocketadcs_sim/sensors/icm40609d.py` | see [../icm40609d.md](../icm40609d.md) |
| MMC5983MA | `src/pocketadcs_sim/sensors/hardware_models.py` | see [../hardware_sensor_models.md](../hardware_sensor_models.md) |
| VEMD8080 | `src/pocketadcs_sim/sensors/vemd8080_array.py`, `sun_reconstruction.py` | see [../vemd8080_array.md](../vemd8080_array.md) |
| TGR 29-12 | `src/pocketadcs_sim/adcs/rotors.py` | see [../adcs_closed_loop.md](../adcs_closed_loop.md) |

## Provenance

Retrieved 2026-10-01 from the vendors' own distribution points, except where a
mirror is noted:

- VEMD8080 — `https://www.vishay.com/docs/84565/vemd8080.pdf`
- MMC5983MA — `https://www.memsic.com/Public/Uploads/uploadfile/files/20220119/MMC5983MADatasheetRevA.pdf`
- TGR 29-12 — `https://www.thingap.com/wp-content/uploads/2023/02/TGR-29-12-Datasheet-Rev-B.pdf`
- ICM-40609-D — `https://www.endrich.com/Datenbl%C3%A4tter/Sensoren/Inertialsensoren/ICM-40609-D.pdf`
  (byte-identical copy of DS-000330 Rev 1.2; the TDK origin at
  `product.tdk.com/.../ds-000330_icm-40609-d_v1.2.pdf` returned HTTP 403 to
  non-interactive fetches, so the mirror is vendored instead and the document
  number and revision are carried in the filename)

These are vendor documents redistributed for engineering reference. They remain
the property of their respective publishers.
