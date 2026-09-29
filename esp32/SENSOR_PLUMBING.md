# Plumbing real sensors into the Zephyr firmware

`esp32/src/sensor_source.h` defines the seam. Everything below it is already
written and tested; everything above it is your job when the hardware lands.

## The one-line swap

`esp32/zephyr/src/main.cpp` selects a source in a single declaration:

```cpp
HilStreamSource source;        // today: ASCII frames over USB serial
// DriverSource source;         // tomorrow: I2C + ADC
```

`estimate()` below it takes a `SensorSample` and nothing else. If you find
yourself editing `estimate()` to add a sensor, the abstraction has been
leaked — put the change in a `SensorSource` instead.

## Writing a driver

Implement three things. This header has **no Zephyr includes on purpose**, so
your driver should live in its own `.cpp` under `esp32/zephyr/src/` and be
free to include Zephyr.

```cpp
class DriverSource : public SensorSource {
public:
    const char *name() const override { return "i2c-drivers"; }

    bool read(SensorSample &out) override {
        if (!gyro_ok_ || !mag_ok_ || !sun_ok_) return false;  // no data
        out = SensorSample{};
        // Convert to SI at this boundary. Raw counts must not escape.
        out.gyro_rad_s[0] = gyro_counts_[0] * GYRO_LSB_RAD_S;  // 16.384 LSB/dps
        out.mag_ut[0]     = mag_counts_[0]  * MAG_LSB_UT;
        out.sun_unit[0]   = sun_reconstruct(...);
        out.flags = kSensorGyro | kSensorMag | kSensorSun;
        out.seq = ++seq_;
        return true;
    }
};
```

Three rules the interface exists to enforce:

1. **Return `false` rather than zeros.** A zero gyro is a real reading (no
   rotation). A zero sun vector means eclipse. Conflating "no data" with
   either produces a confident wrong attitude instead of a visible failure.
   `tests/firmware/test_sensor_source.cpp` pins this.

2. **Set only the flags you can back up.** Bits are independent on purpose —
   the magnetometer is the likeliest thing to be absent on a bench, and
   losing it should not blind the gyro.

3. **Convert to SI in the driver.** rad/s, microtesla, unit vectors. The
   estimator must never re-derive a datasheet scale factor from a device
   that may change. Cross-check the constants against the host models, which
   already encode them:
   - ICM-40609-D gyro: **32768/FSR = 16.384 counts/dps**, *not* the datasheet's
     rounded 16.4 (ArduPilot's driver uses 16.384; see `icm40609d.py`)
   - ICM-40609-D accel: 12-bit, 100 µg/√Hz density
   - MMC5983MA: counts → µT per its own scale
   - VEMD8080: six raw photocurrents → run `sun_reconstruction` (the C++
     equivalent of `src/pocketadcs_sim/sensors/sun_reconstruction.py`) to get
     a unit vector. Note the `cos ** 0.8048` angular law and the peak-relative
     lit-face test at 20% — both are load-bearing and both were bugs.

## The reference vectors are a separate job

`esp32/src/reference_provider.h`. TRIAD needs refSun/refMag in the inertial
frame; no sensor bus can supply them. Today the HIL harness transmits them,
which is why the firmware appears to work — it is being handed the answer.
`NoReferenceProvider` reports "no reference" rather than fabricating one,
because TRIAD against invented vectors yields a plausible, wrong attitude.

A real build needs sun position from time plus a magnetic model (IGRF or a
dipole). Until then the firmware should propagate from the gyro and flag
`kSensorRefSun`/`kSensorRefMag` clear.

## Testing without hardware

The header is host-compilable, so the seam is testable before the drivers
exist — which is the only time it is cheap to test:

```sh
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_sensor_source_host.py -q
```

Add a case to `tests/firmware/test_sensor_source.cpp` for anything you rely
on. The fixture compiles with `-Wall -Wextra -Werror`.

Then build and run on hardware as usual:

```sh
bash esp32/build_zephyr.sh
bash esp32/flash_zephyr.sh COM5
PYTHONPATH=src .venv/Scripts/python.exe examples/hil_stream.py --port COM5 --duration 120 --rate 10
```

## Not solved by this abstraction

State. `adsCompute` is still a pure per-frame TRIAD call: no gyro
integration, no filter, no covariance. The host study shows why that matters
— the sun array degrades to 3–15° error in deep partial eclipse, and with no
propagation there is nothing to fall back on. That is the next substantial
piece of firmware work, and it is independent of the sensor plumbing.
