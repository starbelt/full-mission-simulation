// Sensor-source abstraction for the PocketADCS ESP32-S3 ADS firmware.
//
// WHY THIS EXISTS
// ---------------
// The Zephyr firmware today gets every reading as ASCII over serial from
// examples/hil_stream.py -- including the attitude *reference* vectors, so
// it never actually determines attitude; it is handed one. The goal is to
// replace that with real I2C/ADC drivers for the ICM-40609-D, MMC5983MA and
// VEMD8080 array, plus an ephemeris for the references.
//
// To make that swap a configuration choice rather than a rewrite, the
// estimator must depend on an interface, not on a transport. This header is
// that interface.
//
// DESIGN CONSTRAINTS
// ------------------
// 1. **No Zephyr includes.** The seam has to compile on the host so it can
//    be tested before any driver exists. Driver implementations live in
//    separate translation units that *may* include Zephyr; this header may
//    not. tests/firmware/test_sensor_source.cpp exercises everything here.
//
// 2. **Explicit validity, never implicit zeros.** Every field carries its
//    own good/bad bit. A bus that is not populated, or a driver that has not
//    been written yet, must report "no data" rather than fill zeros -- a
//    zero gyro is a real reading (no rotation) and a zero sun vector means
//    eclipse. Conflating them is how a missing driver becomes a silent
//    physics bug instead of an obvious failure.
//
// 3. **Units are fixed at the boundary, in SI.** Radians/s, microtesla,
//    unit vectors. Raw counts never cross this interface; the conversion
//    belongs in the driver, so the estimator is never re-deriving a
//    datasheet scale factor from a device that may change.
//
// 4. **The reference vectors are modelled as a separate concern.** Real
//    attitude determination needs refSun/refMag from an ephemeris, not from
//    the sensor bus. They travel in the same struct because the estimator
//    consumes them together, but they carry their own flag so a driver-based
//    source does not have to fabricate them.

#ifndef SENSOR_SOURCE_H
#define SENSOR_SOURCE_H

#include <cstdint>
#include <stdio.h>

// Field-validity bits. A set bit means "trust this field"; a clear bit means
// the value is meaningless and the estimator must fall back or mark the
// sample invalid. Bits are independent on purpose: the magnetometer is the
// most likely to be absent on a bench, and losing it should not blind the
// gyro.
enum SensorField : uint8_t {
    kSensorNone  = 0,
    kSensorGyro  = 1u << 0,
    kSensorMag   = 1u << 1,
    kSensorSun   = 1u << 2,
    kSensorRefSun = 1u << 3,
    kSensorRefMag = 1u << 4,
    kSensorAll   = kSensorGyro | kSensorMag | kSensorSun | kSensorRefSun | kSensorRefMag,
};

// One coherent sample. All three vectors are sampled as close to
// simultaneously as the hardware allows, because TRIAD is a
// cross-product-difference algorithm and its accuracy degrades if the
// vectors are not time-coherent.
//
// pos_gcrs_m and epoch_ms are the spacecraft's own state, not a sensor
// reading: a real vehicle knows where it is (GPS or orbit propagation) and
// what time it is. They are what the onboard ephemeris consumes to derive the
// inertial reference vectors. The host must NOT transmit pre-computed
// reference vectors -- doing so moves the attitude determination off the
// flight computer, which is the whole point of running it on the ESP32.
struct SensorSample {
    float gyro_rad_s[3];   // body angular rate, rad/s
    float mag_ut[3];       // body magnetic field, microtesla
    float sun_unit[3];     // body-frame unit sun direction
    double pos_gcrs_m[3];  // spacecraft position in GCRS, metres
    int64_t epoch_ms;      // absolute time, Unix milliseconds
    float ref_sun_unit[3]; // inertial-frame unit sun direction  (derived onboard)
    float ref_mag_unit[3]; // inertial-frame unit magnetic field  (derived onboard)
    uint8_t flags;         // bitwise-or of SensorField
    int32_t seq;           // monotonic sample counter, for gap detection
};

// The interface every reading path implements.
//
// read() returns true only if it produced a sample. On false the out
// parameter is left untouched -- implementations must not zero it, because
// the caller may be holding a previous good sample it needs to keep
// propagating from.
class SensorSource {
public:
    virtual ~SensorSource() = default;
    virtual bool read(SensorSample &out) = 0;

    // Human-readable name for logs and the HIL banner. Must not allocate.
    virtual const char *name() const = 0;
};

// ---------------------------------------------------------------------------
// Serial HIL source: the current production path.
//
// Parses the S2 frame: time/seq, gyro[3], mag[3], sun[3], pos_gcrs_m[3].
//
// Note what is NOT in it: the inertial reference vectors. In the previous
// (v1) protocol the host transmitted refSun and refMag, which meant the host
// performed the attitude determination and the firmware merely returned a
// rotation matrix it had been handed -- a 0.005 deg "result" that measured
// nothing about the firmware. V2 sends only what a real spacecraft knows
// about itself, and the reference vectors are derived onboard by the
// ephemeris. Any accuracy figure produced from v1 was not a measurement of
// the ESP32's ADS.
// ---------------------------------------------------------------------------
class HilStreamSource : public SensorSource {
public:
    // Parse a whitespace-separated S2 line. Returns false on a malformed
    // frame, leaving `out` untouched. Positions are metres and can reach
    // 6.9e6, so they are double-precision: a float32 has ~0.5 m resolution
    // at LEO radius, which is visible as a magnetic-reference error.
    bool parse(const char *line, SensorSample &out) {
        char type[8];
        long long seq = 0, epoch_ms = 0;
        double p[3];
        if (sscanf(line, "%7s %lld %lld %f %f %f %f %f %f %f %f %f %lf %lf %lf",
                   type, &seq, &epoch_ms, &out.gyro_rad_s[0], &out.gyro_rad_s[1],
                   &out.gyro_rad_s[2], &out.mag_ut[0], &out.mag_ut[1],
                   &out.mag_ut[2], &out.sun_unit[0], &out.sun_unit[1],
                   &out.sun_unit[2], &p[0], &p[1], &p[2]) != 15) {
            return false;
        }
        if (type[0] != 'S' || type[1] != '2') return false;
        out.seq = static_cast<int32_t>(seq);
        out.epoch_ms = static_cast<int64_t>(epoch_ms);
        for (int i = 0; i < 3; ++i) out.pos_gcrs_m[i] = p[i];
        // Measured vectors only. The reference flags are cleared here and set
        // later, and only if the onboard ephemeris actually produced them.
        out.flags = kSensorGyro | kSensorMag | kSensorSun;
        return true;
    }

    bool read(SensorSample &) override { return false; }
    const char *name() const override { return "hil-serial"; }
};

#endif  // SENSOR_SOURCE_H
