# Technical documentation

Design notes, model assumptions, reference frames, sensor models, the orbital
environment, and validation methodology for PocketADCS.

| Document | Contents |
|---|---|
| [architecture.md](architecture.md) | Library reuse boundary, ownership, verification policy |
| [simulation-requirements.md](simulation-requirements.md) | Requirements |
| [adcs_closed_loop.md](adcs_closed_loop.md) | PC closed-loop ADCS: design, conventions, measured results, test coverage |
| [esp32_closed_loop.md](esp32_closed_loop.md) | Closed-loop firmware: protocol, mission lifecycle, timing, safety behaviour |
| [esp32_reliability.md](esp32_reliability.md) | Firmware reliability findings, corrections and measured soak results |
| [hardware_sensor_models.md](hardware_sensor_models.md) | MMC5983MA magnetometer model and verified datasheet values |
| [icm40609d.md](icm40609d.md) | ICM-40609-D IMU model and verified datasheet values |
| [vemd8080_array.md](vemd8080_array.md) | Six-photodiode VEMD8080 sun array model and reconstruction |
| [datasheets/](datasheets/README.md) | Vendored part datasheets with sizes and SHA-256 digests |

Evidence produced by the harnesses lives under [`../results/`](../results), in
versioned directories that are append-only and never overwritten.
