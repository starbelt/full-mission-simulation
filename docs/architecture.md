# Simulation architecture

## Guiding decision

This project is an integration and verification harness, not a new astrodynamics simulator. Mission physics, reference frames, geodesy, environmental models, and rigid-body dynamics should come from established libraries wherever a suitable implementation exists. Project-owned code should primarily configure those models, connect sensor and flight-software interfaces, represent the actual hardware, and calculate verification metrics.

## Selected foundation

- **Basilisk is the preferred truth-simulation framework for orbit, six-degree-of-freedom spacecraft dynamics, effectors, disturbances, Monte Carlo runs, and future hardware-in-the-loop work.** It is explicitly designed for spacecraft dynamics, flight-software integration, Monte Carlo simulation, and hardware-in-the-loop operation.[1]
- **Astropy/ERFA supplies time, WGS84 geodesy, and terrestrial/inertial frame transformations.** `EarthLocation` delegates geodetic/geocentric conversion to ERFA and integrates with the ITRS/GCRS transformation graph.[2]
- **Astropy IERS data supplies Earth-orientation parameters** used by terrestrial-to-celestial transformations; production runs must record the IERS data version and must not silently accept degraded accuracy.[3]
- **SciPy Rotation will supply quaternion and rotation operations** at interfaces where Basilisk types are not used directly.
- **PyMap3D supplies WGS84 local-horizon geometry** for independently testable target visibility and azimuth/elevation calculations.
- **An official IGRF implementation or a validated wrapper around official coefficients will supply the main geomagnetic field.** A package will be selected only after numerical comparison against published reference values.

## Ownership boundary

### Reuse from validated libraries

- time scales, leap seconds, and Earth orientation
- WGS84 geodetic conversion
- inertial/Earth-fixed transformations
- orbit and rigid-body propagation
- gravity, ephemerides, eclipse, and environmental models
- quaternion/rotation primitives
- geomagnetic reference-field evaluation

### Implement in this project

- PocketQube mass properties and hardware configuration
- measured sensor error models and calibration parameters
- interfaces matching the embedded ADS implementation
- target selection and mission-mode logic
- hardware actuator characteristics
- test scenarios, acceptance limits, fault injection, and Monte Carlo analysis
- traceability from hardware requirements to simulation evidence

## Verification policy

1. Pin all model and data versions in `uv.lock` or scenario metadata.
2. Store frame, epoch, units, and quaternion convention at every interface.
3. Compare every library integration against an independent published case or reference value.
4. Keep truth models separate from flight-software models to avoid common-mode errors.
5. Record random seeds and complete scenario configuration.
6. Add measured hardware parameters and hardware-in-the-loop tests as they become available.
7. Treat simulation success as evidence, not proof; correlate the model with bench, Helmholtz-cage, rate-table, thermal-vacuum, and integrated hardware tests.

## Initial implementation

The first geometry module deliberately contains only integration code: an Earth target is represented by Astropy `EarthLocation`, GCRS target vectors use Astropy frame transformations, and target visibility uses PyMap3D WGS84 local-horizon geometry. No local Earth-rotation, ellipsoid, or quaternion implementation is introduced.

## Sources

[1] https://avslab.github.io/basilisk — Basilisk Astrodynamics Simulation Framework
[2] https://docs.astropy.org/en/stable/api/astropy.coordinates.EarthLocation.html — Astropy EarthLocation
[3] https://docs.astropy.org/en/stable/utils/iers.html — Astropy IERS data
