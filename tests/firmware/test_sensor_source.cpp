// Host regression for the firmware sensor-source abstraction.
//
// The ESP32-S3 Zephyr firmware must be able to take its sensor readings from
// EITHER the serial HIL stream (today) OR real I2C/ADC drivers (not yet
// written), without the estimator knowing which. This header is deliberately
// free of any Zephyr include so the seam can be compiled and exercised on
// the host, which is the only way to test it before the drivers exist.
//
// What this fixture pins down:
//   * a source fills a SensorSample and reports which fields are trustworthy,
//   * the S2 frame parses position (not pre-computed reference vectors),
//   * a v1 frame is REJECTED -- the old protocol let the host supply the
//     inertial references, which moved the attitude determination off the
//     flight computer,
//   * malformed frames leave the sample untouched rather than zero-filling,
//   * a source that has no data yet says so instead of returning zeros,
//   * the estimator consumes a sample identically regardless of its origin.

#include <cmath>
#include <cstdio>
#include <cstring>

#include "sensor_source.h"

static int failures = 0;

static void check(bool ok, const char *what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

static bool near(float a, float b, float tol = 1e-6f) {
    return std::fabs(a - b) <= tol;
}

// A stand-in for a future I2C/ADC driver: fills a sample from fixed values
// and reports every measured field good. Exercises the same interface the
// real drivers will implement.
class FakeDriverSource : public SensorSource {
public:
    const char *name() const override { return "fake-driver"; }
    bool read(SensorSample &out) override {
        out = SensorSample{};
        out.gyro_rad_s[0] = 0.01f;
        out.gyro_rad_s[1] = -0.02f;
        out.gyro_rad_s[2] = 0.03f;
        out.mag_ut[0] = 20.0f;
        out.mag_ut[1] = -5.0f;
        out.mag_ut[2] = 40.0f;
        out.sun_unit[0] = 0.0f;
        out.sun_unit[1] = 0.0f;
        out.sun_unit[2] = 1.0f;
        out.pos_gcrs_m[0] = 6878137.0;
        out.epoch_ms = 1735689600000LL;
        out.flags = kSensorGyro | kSensorMag | kSensorSun;
        out.seq = 7;
        return true;
    }
};

// A source with nothing to report yet -- what an unpopulated bus looks like.
class EmptySource : public SensorSource {
public:
    bool read(SensorSample &) override { return false; }
    const char *name() const override { return "empty"; }
};

int main() {
    // --- the interface is usable polymorphically -------------------------
    FakeDriverSource driver;
    SensorSource *as_interface = &driver;
    SensorSample s{};
    check(as_interface->read(s), "driver source reports data");
    check(near(s.gyro_rad_s[2], 0.03f), "gyro passthrough");
    check(near(s.mag_ut[2], 40.0f), "mag passthrough");
    check(s.flags & kSensorSun, "sun flag set");

    // --- an empty source must NOT fabricate zeros -------------------------
    EmptySource empty;
    SensorSample e{};
    e.flags = kSensorGyro | kSensorMag | kSensorSun;  // poison
    check(!empty.read(e), "empty source reports no data");
    check(e.flags == (kSensorGyro | kSensorMag | kSensorSun),
          "empty source leaves the sample untouched (no silent zero-fill)");

    // --- a partially-populated bus reports per-field validity ------------
    class MagOnlySource : public SensorSource {
    public:
        const char *name() const override { return "mag-only"; }
        bool read(SensorSample &out) override {
            out = SensorSample{};
            out.mag_ut[0] = 1.0f;
            out.flags = kSensorMag;
            return true;
        }
    };
    MagOnlySource mag_only;
    SensorSample m{};
    check(mag_only.read(m), "mag-only source reports data");
    check(m.flags == kSensorMag, "only the mag field is flagged good");
    check(!(m.flags & kSensorSun), "sun explicitly not flagged good");

    // --- S2 frame parsing ------------------------------------------------
    HilStreamSource hil;
    // type, seq, epoch_ms, gyro[3], mag[3], sun[3], pos_gcrs_m[3]
    char line[256];
    std::snprintf(line, sizeof(line),
                  "S2 42 1735689600000 0.1 -0.2 0.3 10 20 30 "
                  "0 0 1 6878137.000 0.000 0.000");
    SensorSample h{};
    check(hil.parse(line, h), "S2 frame parses");
    check(h.seq == 42, "sequence number preserved");
    check(h.epoch_ms == 1735689600000LL, "epoch preserved");
    check(near(h.gyro_rad_s[1], -0.2f), "gyro preserved");
    check(near(h.mag_ut[2], 30.0f), "mag preserved");
    check(near(h.sun_unit[2], 1.0f), "sun preserved");
    // Position must keep sub-metre precision: float32 at LEO radius has
    // ~0.5 m resolution, which shows up directly as magnetic-reference error.
    check(h.pos_gcrs_m[0] > 6878136.9 && h.pos_gcrs_m[0] < 6878137.1,
          "position parsed in double precision");
    // The critical property: the frame must NOT mark the references good.
    check(!(h.flags & kSensorRefSun), "S2 frame does not supply refSun");
    check(!(h.flags & kSensorRefMag), "S2 frame does not supply refMag");
    check(h.flags == (kSensorGyro | kSensorMag | kSensorSun),
          "S2 frame flags only the measured vectors");

    // --- the old v1 protocol must be rejected ----------------------------
    // v1 carried the inertial reference vectors. Accepting it would let the
    // host perform the attitude determination, which is precisely the thing
    // that must happen on the ESP32.
    char v1[256];
    std::snprintf(v1, sizeof(v1),
                  "S 42 0.000 0.1 -0.2 0.3 10 20 30 0 0 1 0 0 1 1 0 0");
    SensorSample r{};
    check(!hil.parse(v1, r), "legacy v1 frame is rejected");

    // --- malformed frames leave the sample untouched --------------------
    SensorSample guarded{};
    guarded.mag_ut[2] = 123.0f;
    guarded.flags = kSensorMag;
    char truncated[] = "S2 42 1735689600000 0.1 -0.2";
    check(!hil.parse(truncated, guarded), "truncated frame is rejected");
    check(guarded.mag_ut[2] == 123.0f, "rejected frame did not clobber the sample");
    check(guarded.flags == kSensorMag, "rejected frame left flags untouched");

    // --- the estimator sees the same shape from either source -----------
    // This is the property the abstraction exists for: swapping the source
    // must not change what the estimator is handed.
    SensorSample from_driver{}, from_hil{};
    FakeDriverSource d2;
    d2.read(from_driver);
    HilStreamSource h2;
    char l2[256];
    std::snprintf(l2, sizeof(l2),
                  "S2 1 1735689600000 0.01 -0.02 0.03 20 -5 40 0 0 1 "
                  "6878137.000 0.000 0.000");
    h2.parse(l2, from_hil);
    check((from_driver.flags & kSensorGyro) && (from_hil.flags & kSensorGyro),
          "both sources present a gyro field to the estimator");
    check((from_driver.flags & kSensorSun) && (from_hil.flags & kSensorSun),
          "both sources present a sun field to the estimator");
    check(from_driver.pos_gcrs_m[0] == from_hil.pos_gcrs_m[0],
          "both sources present a position for the onboard ephemeris");

    if (failures == 0) {
        std::printf("sensor_source: all checks passed\n");
        return 0;
    }
    std::printf("sensor_source: %d check(s) failed\n", failures);
    return 1;
}
