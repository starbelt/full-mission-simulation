// Onboard reference-vector computation.
//
// WHY THIS FILE EXISTS
// --------------------
// TRIAD needs two body-frame vectors (measured by the sensors) and the same
// two vectors in the INERTIAL frame (known, not measured). Historically the
// HIL harness computed the inertial pair and transmitted it alongside the
// sensor data, which meant the host was doing the attitude determination and
// the ESP32 was just echoing a rotation matrix back. That is not an ADS.
//
// This file computes the inertial references ONBOARD from the spacecraft's
// position and the clock. That is exactly what a flight ADS does: position
// from GPS or orbit propagation, time from the onboard clock, and the
// environmental models below.
//
// MODELS
// ------
// Sun: low-precision solar ephemeris from the Julian date (Astronomical
// Algorithms, Meeus ch. 25). Accurate to roughly 0.01 deg over 1950-2050 --
// four orders of magnitude better than the sensor noise, so it is not the
// limiting term.
//
// Mag: tilted dipole. A full IGRF-14 is ~200 coefficients per epoch and is
// the right choice eventually, but a dipole is the standard first-order
// model for LEO and this is what the model is calibrated against. See
// magModelDipoleDeg(). The host truth uses IGRF-14, so a residual angle
// between this and the host is EXPECTED and is a real, measured part of the
// accuracy budget rather than a bug to be tuned away. Swap in IGRF when the
// coefficient tables are worth the flash.

#ifndef REFERENCE_PROVIDER_H
#define REFERENCE_PROVIDER_H

#include <math.h>
#include <stdint.h>

#include "igrf.h"
#include "sensor_source.h"

// The Zephyr/newlib toolchain does not define M_PI under C++ unless a
// feature-test macro is set, and we build freestanding. Define it locally
// rather than relying on the toolchain.
#ifndef POCKETADCS_PI
#define POCKETADCS_PI 3.14159265358979323846
#endif

// Days since the J2000.0 epoch.
static inline double days_since_j2000(int64_t epoch_ms) {
    // J2000 is 10957.5 days AFTER Unix epoch: subtract the offset.
    return static_cast<double>(epoch_ms) / 86400000.0 - 10957.5;
}

// Low-precision solar position (Meeus, Astronomical Algorithms ch. 25).
// Returns the unit GCRS direction TO the Sun.
static inline void sunDirectionGcrs(int64_t epoch_ms, float out[3]) {
    const double n = days_since_j2000(epoch_ms);

    // Mean longitude and mean anomaly of the Sun, degrees.
    const double L = fmod(280.460 + 0.9856474 * n, 360.0);
    const double g = fmod(357.528 + 0.9856003 * n, 360.0);

    const double g_rad = g * POCKETADCS_PI / 180.0;
    const double lambda = L + 1.915 * sin(g_rad) + 0.020 * sin(2.0 * g_rad);
    const double lambda_rad = lambda * POCKETADCS_PI / 180.0;
    // Obliquity of the ecliptic, with the short-period nutation term.
    const double eps = (23.439 - 0.0000004 * n) * POCKETADCS_PI / 180.0;

    // Meeus' ecliptic longitude/obliquity give mean equatorial axes OF DATE,
    // not GCRS. Remove precession with the same verified rotation primitives
    // as the magnetic-reference model. Merely labelling these axes GCRS costs
    // ~0.35 degrees in 2025 and grows with distance from J2000.
    const double equatorial_date[3] = {
        cos(lambda_rad), cos(eps) * sin(lambda_rad), sin(eps) * sin(lambda_rad)
    };
    const double jd = n + 2451545.0;
    double rz1[9], ry[9], rz2[9], tmp[9], p[9];
    pocketadcs_igrf::mat3RotZ(-pocketadcs_igrf::precessionZ(jd), rz1);
    pocketadcs_igrf::mat3RotY(pocketadcs_igrf::precessionTheta(jd), ry);
    pocketadcs_igrf::mat3RotZ(-pocketadcs_igrf::precessionZeta(jd), rz2);
    pocketadcs_igrf::mat3Mul(rz1, ry, tmp);
    pocketadcs_igrf::mat3Mul(tmp, rz2, p);
    for (int i = 0; i < 3; ++i) {
        out[i] = static_cast<float>(p[i] * equatorial_date[0] +
                                   p[3 + i] * equatorial_date[1] +
                                   p[6 + i] * equatorial_date[2]);
    }
}

// The interface. Separate from SensorSource because the reference vectors
// are not measurements: no sensor bus can produce them.
class ReferenceProvider {
public:
    virtual ~ReferenceProvider() = default;

    // Fills ref_sun_unit / ref_mag_unit for the given time and position.
    // Returns false if no reference is available, in which case the caller
    // must fall back to propagation and MUST NOT use stale or zero values.
    virtual bool references(int64_t epoch_ms, const double pos_gcrs_m[3],
                            SensorSample &out) const = 0;
    virtual const char *name() const = 0;
};

// Reports "no reference" rather than fabricating one. Used in the build that
// has no ephemeris wired up, so a missing model is visible instead of
// silently producing a confident wrong attitude.
class NoReferenceProvider : public ReferenceProvider {
public:
    bool references(int64_t, const double[3], SensorSample &) const override {
        return false;
    }
    const char *name() const override { return "none (no ephemeris)"; }
};

// The production provider. Derives both references from the vehicle's own
// position and clock. The magnetic field uses the full IGRF-14 spherical
// harmonic expansion in igrf.h -- NOT a dipole approximation, which is wrong
// by ~10 deg at LEO and would rotate the attitude answer directly, since
// TRIAD compares the magnetic direction against the sun direction.
class EphemerisReferenceProvider : public ReferenceProvider {
public:
    bool references(int64_t epoch_ms, const double pos_gcrs_m[3],
                    SensorSample &out) const override {
        sunDirectionGcrs(epoch_ms, out.ref_sun_unit);
        if (igrfFieldUnitGcrs(epoch_ms, pos_gcrs_m, out.ref_mag_unit) != 0) {
            // The model failed. Leave the reference flags clear so the caller
            // propagates rather than trusting an undefined direction.
            return false;
        }
        out.flags |= kSensorRefSun | kSensorRefMag;
        return true;
    }

    const char *name() const override { return "ephemeris (IGRF-14 + Meeus)"; }
};

#endif  // REFERENCE_PROVIDER_H
