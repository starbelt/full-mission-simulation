// Validate measurements and onboard reference vectors before invoking TRIAD.
// Shared by Zephyr core 0 and the host-side regression test.
#pragma once

#include "ads_math.h"
#include "sensor_source.h"

static void estimateSample(const SensorSample &s, AdsFrameResult &result,
                           unsigned &flags) {
    flags = 1u;
    if (!(s.flags & kSensorGyro)) flags |= 32u;
    if (!(s.flags & kSensorMag)) flags |= 64u;
    if (!(s.flags & kSensorSun)) flags |= 128u;
    if (!(s.flags & kSensorRefSun) || !(s.flags & kSensorRefMag)) flags |= 256u;
    if ((flags & (32u | 64u | 128u | 256u)) != 0u) {
        // A failed ephemeris leaves ref_mag_unit undefined. Never call TRIAD
        // with uninitialized references or score its accidental quaternion.
        result.angle_deg = angleBetween(s.mag_ut, s.sun_unit);
        result.rate = vecNorm(s.gyro_rad_s);
        result.mag_norm = vecNorm(s.mag_ut);
        result.sun_norm = vecNorm(s.sun_unit);
        result.q[0] = 1.0f;
        result.q[1] = result.q[2] = result.q[3] = 0.0f;
        result.valid = 0;
    } else {
        adsCompute(s.gyro_rad_s, s.mag_ut, s.sun_unit, s.ref_sun_unit,
                   s.ref_mag_unit, &result);
    }
    if (result.sun_norm < 0.01f) flags |= 2u;
    if (!result.valid) flags |= 16u;
}
