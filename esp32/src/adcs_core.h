// PocketADCS closed-loop ADCS core for the ESP32-S3 flight computer.
//
// This header is the port of the reviewed PC prototype
// (src/pocketadcs_sim/adcs/{quaternions,estimator,controller,wheel_board,
// slewos_adapter}.py, docs/adcs_closed_loop.md) to a freestanding, fixed-storage
// C++ form. It computes, on the device:
//
//   * attitude by gyro propagation with TRIAD absolute fixes,
//   * the planned reference angle along a real SlewOS/FA01 schedule,
//   * a PD body torque from the ESTIMATED attitude and the measured rate,
//   * three orthogonal wheel commands with the board's real limits.
//
// WHAT MUST NOT HAPPEN HERE
// ------------------------
// The header has no parameter, field or method through which a truth attitude
// or a truth rate could enter. It is handed a gyro rate, two body-frame vector
// measurements, the vehicle's own position/clock (which the onboard ephemeris
// turns into reference vectors), and a schedule. That is the whole
// truth-ownership rule, and it is enforced by the interface rather than by
// convention -- the same structural argument the PC core makes, preserved
// through the port. A green differential run means the device computes the
// answer; it can never mean the device was handed one.
//
// HOST COMPATIBILITY (why there is no <cmath> or <cstdint>)
// ---------------------------------------------------------
// The Xtensa toolchain has no libstdc++ headers, so this file includes only
// <math.h>, <stdio.h> and <stdint.h> and defines its own M_PI. It is compiled
// unmodified by the host regression test (tests/firmware/adcs_core_driver.cpp,
// driven by tests/test_adcs_core_differential.py) with -Wall -Wextra -Werror,
// so anything host-legal but target-illegal is caught before the board is
// touched. That is also why every buffer below has a fixed capacity: no
// dynamic allocation means no unbounded memory path on the device.

#ifndef ADCS_CORE_H
#define ADCS_CORE_H

#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace adcs {

// Error codes. The estimator and controller return one of these and leave all
// state untouched when they return anything but kOk -- a rejected sample must
// not half-advance the estimate.
enum : int32_t {
    kOk = 0,
    kErrNonFinite = 1,
    kErrTimeRegression = 2,
    kErrZeroDt = 3,
    kErrBadArgument = 4,
};

// Why the most recent absolute fix was not applied. Reported so a host can
// distinguish "coasting through an eclipse" from "the fix is silently broken".
enum : int32_t {
    kRejectNone = 0,
    kRejectSunInvalid = 1,
    kRejectMagInvalid = 2,
    kRejectParallelPair = 3,
    kRejectTriadFailed = 4,
};

// Hard resource bounds. These are the device's whole schedule capacity; a
// larger plan is refused rather than truncated, because a truncated maneuver
// would report a tiny final error while leaving the vehicle short of the
// planned attitude.
enum : int32_t { kMaxSegments = 64 };
// Longest plan the device will accept, in seconds (24 h).
const double kMaxScheduleDurationS = 86400.0;

// Below this |cos| the two reference vectors are too close to define attitude.
const double kParallelTolerance = 1.0 - 1e-9;

struct Quat {
    double w, x, y, z;
};

struct Vec3 {
    double v[3];
};

struct Config {
    double proportional_gain;    // rad -> N.m
    double derivative_gain;      // N.m.s -> N.m
    double max_torque_nm;        // continuous vacuum torque, the clip limit
    double acceleration_cap_rad_s2;  // engineering parameter, NOT a datasheet claim
    double rotor_inertia_kg_m2;
    double max_speed_rad_s;
};

struct AbsoluteUpdate {
    double sun_body[3];
    double magnetic_body[3];
    double sun_ref[3];
    double magnetic_ref[3];
    bool sun_valid;
    bool magnetic_valid;
};

inline bool finite3(const double v[3]) {
    for (int i = 0; i < 3; ++i) {
        if (!isfinite(v[i])) return false;
    }
    return true;
}

inline bool finiteQ(const Quat &q) {
    return isfinite(q.w) && isfinite(q.x) && isfinite(q.y) && isfinite(q.z);
}

inline double dot3(const double a[3], const double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double norm3(const double v[3]) { return sqrt(dot3(v, v)); }

inline void cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

inline void unit3(const double v[3], double out[3]) {
    const double n = norm3(v);
    if (n > 0.0) {
        out[0] = v[0] / n;
        out[1] = v[1] / n;
        out[2] = v[2] / n;
    } else {
        out[0] = out[1] = out[2] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Quaternion algebra -- conventions copied from the accepted PC module
// (pocketadcs_sim/adcs/quaternions.py), whose docstring states them once:
//
//   * scalar-first Hamilton quaternions [w, x, y, z];
//   * q denotes C_BN, the DCM mapping INERTIAL components to BODY components,
//     i.e. the transpose of the active rotation matrix;
//   * to_dcm(a (x) b) == to_dcm(b) @ to_dcm(a), so quaternion products compose
//     in the opposite order to matrix products in this convention;
//   * every function is sign-invariant in its quaternion input.
// ---------------------------------------------------------------------------

inline Quat quatIdentity() {
    const Quat q = {1.0, 0.0, 0.0, 0.0};
    return q;
}

inline Quat quatNormalized(double w, double x, double y, double z) {
    const double n = sqrt(w * w + x * x + y * y + z * z);
    Quat q = {w / n, x / n, y / n, z / n};
    return q;
}

// Unnormalized Hamilton product; the caller normalizes.
inline Quat quatRawMul(const Quat &a, const Quat &b) {
    Quat r;
    r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    r.x = a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y;
    r.y = a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z;
    r.z = a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x;
    return r;
}

inline Quat quatMul(const Quat &a, const Quat &b) {
    const Quat raw = quatRawMul(a, b);
    return quatNormalized(raw.w, raw.x, raw.y, raw.z);
}

inline Quat quatConjugate(const Quat &q) {
    Quat c = {q.w, -q.x, -q.y, -q.z};
    return c;
}

inline Quat quatFromAxisAngle(const double axis[3], double angle_rad) {
    const double n = norm3(axis);
    const double half = 0.5 * angle_rad;
    const double s = sin(half);
    return quatNormalized(cos(half), s * axis[0] / n, s * axis[1] / n, s * axis[2] / n);
}

inline Quat quatFromMrp(const double sigma[3]) {
    const double m2 = sigma[0] * sigma[0] + sigma[1] * sigma[1] + sigma[2] * sigma[2];
    return quatNormalized(1.0 - m2, 2.0 * sigma[0], 2.0 * sigma[1], 2.0 * sigma[2]);
}

// MRP of a C_BN quaternion, scalar part forced non-negative so the result is a
// function of the rotation alone. Matches scipy's convention, which the PC
// module also matches.
inline void quatToMrp(const Quat &q_in, double sigma[3]) {
    Quat q = q_in;
    if (q.w < 0.0) {
        q.w = -q.w;
        q.x = -q.x;
        q.y = -q.y;
        q.z = -q.z;
    }
    const double d = 1.0 + q.w;
    sigma[0] = q.x / d;
    sigma[1] = q.y / d;
    sigma[2] = q.z / d;
}

// The rotation from the actual body frame to the desired one, scalar part
// non-negative: its vector part is the increment still to be applied, in
// desired body axes.
inline Quat quatError(const Quat &q_desired, const Quat &q_actual) {
    Quat e = quatRawMul(q_desired, quatConjugate(q_actual));
    if (e.w < 0.0) {
        e.w = -e.w;
        e.x = -e.x;
        e.y = -e.y;
        e.z = -e.z;
    }
    return quatNormalized(e.w, e.x, e.y, e.z);
}

inline double quatAngleRad(const Quat &q_in) {
    double w = q_in.w < 0.0 ? -q_in.w : q_in.w;
    if (w > 1.0) w = 1.0;
    return 2.0 * acos(w);
}

// Propagate C_BN under a constant body rate for dt, using the exact exponential
// so a constant rate reproduces a known-axis rotation to machine precision and
// splitting the interval changes nothing.
//
// Sign/ordering: because C_BN maps inertial to body components, the increment
// multiplies on the RIGHT of the quaternion. The opposite order coincides at
// the identity attitude and then silently mirrors every update once the
// vehicle is slewing, which is why the differential case starts away from
// identity.
inline Quat quatPropagate(const Quat &q, const double omega[3], double dt_s) {
    if (!isfinite(dt_s) || dt_s <= 0.0) return q;  // caller checks and refuses
    const double angle = norm3(omega) * dt_s;
    if (angle == 0.0) return q;
    const Quat delta = quatFromAxisAngle(omega, angle);
    return quatMul(q, delta);
}

// ---------------------------------------------------------------------------
// TRIAD, in double precision, ported from ahrs.filters.TRIAD exactly as the
// firmware's float triadDcm already is (see esp32/src/ads_math.h):
//   Mb = [w1 s2 s3], Mr = [v1 r2 r3] are COLUMN matrices, A = Mb Mr^T.
// A is C_BN (reference -> body).
// ---------------------------------------------------------------------------
inline bool triadDcm(const double v1[3], const double v2[3], const double w1[3],
                     const double w2[3], double A[3][3]) {
    double nv1[3], nv2[3], nw1[3], nw2[3];
    unit3(v1, nv1);
    unit3(v2, nv2);
    unit3(w1, nw1);
    unit3(w2, nw2);
    if (norm3(nv1) == 0.0 || norm3(nv2) == 0.0 || norm3(nw1) == 0.0 || norm3(nw2) == 0.0) {
        return false;
    }
    double r2[3], r3[3], s2[3], s3[3];
    cross3(nv1, nv2, r2);
    if (norm3(r2) < 1e-9) return false;  // reference vectors parallel
    unit3(r2, r2);
    cross3(nv1, r2, r3);
    unit3(r3, r3);
    cross3(nw1, nw2, s2);
    if (norm3(s2) < 1e-9) return false;  // body vectors parallel
    unit3(s2, s2);
    cross3(nw1, s2, s3);
    unit3(s3, s3);
    const double *body[3] = {nw1, s2, s3};
    const double *reference[3] = {nv1, r2, r3};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int k = 0; k < 3; ++k) acc += body[k][i] * reference[k][j];
            A[i][j] = acc;
        }
    }
    return true;
}

// Quaternion of a C_BN matrix. to_dcm returns the transpose of the active
// rotation, so this recovers the quaternion of A^T; passing the inverse in by
// mistake would negate the vector part and silently mirror every estimate.
inline Quat quatFromCbn(const double A[3][3]) {
    const double c[3][3] = {{A[0][0], A[1][0], A[2][0]},
                            {A[0][1], A[1][1], A[2][1]},
                            {A[0][2], A[1][2], A[2][2]}};
    const double trace = c[0][0] + c[1][1] + c[2][2];
    if (trace > 0.0) {
        const double scale = 2.0 * sqrt(1.0 + trace);
        return quatNormalized(0.25 * scale, (c[2][1] - c[1][2]) / scale,
                              (c[0][2] - c[2][0]) / scale, (c[1][0] - c[0][1]) / scale);
    }
    // trace <= 0: the trace formulas lose all precision, so drive the largest
    // diagonal term. The scalar part is a DIFFERENCE of the two off-diagonal
    // terms while the other two are SUMS; using differences throughout returns
    // the axis unit vector instead of the rotation.
    int index = 0;
    if (c[1][1] > c[index][index]) index = 1;
    if (c[2][2] > c[index][index]) index = 2;
    const int j = (index + 1) % 3;
    const int k = (index + 2) % 3;
    double scale;
    if (index == 0) {
        scale = 2.0 * sqrt(1.0 + c[0][0] - c[1][1] - c[2][2]);
    } else if (index == 1) {
        scale = 2.0 * sqrt(1.0 - c[0][0] + c[1][1] - c[2][2]);
    } else {
        scale = 2.0 * sqrt(1.0 - c[0][0] - c[1][1] + c[2][2]);
    }
    double v[4] = {0.0, 0.0, 0.0, 0.0};
    v[0] = (c[j][k] - c[k][j]) / scale;
    v[1 + index] = 0.25 * scale;
    v[1 + j] = (c[index][j] + c[j][index]) / scale;
    v[1 + k] = (c[index][k] + c[k][index]) / scale;
    return quatNormalized(v[0], v[1], v[2], v[3]);
}

// ---------------------------------------------------------------------------
// Estimator: gyro propagation plus TRIAD absolute fixes.
//
// The fix is applied AFTER propagation, at the new timestamp, so the estimate
// is correct *at* the sample time rather than one step behind it.
// ---------------------------------------------------------------------------
struct Estimator {
    Quat attitude;
    double last_time_s;
    bool has_last_time;
    bool has_absolute_fix;
    int32_t last_reject;
    double covariance_trace;

    void reset() {
        attitude = quatIdentity();
        last_time_s = 0.0;
        has_last_time = false;
        has_absolute_fix = false;
        last_reject = kRejectNone;
        covariance_trace = 0.0;
    }

    Estimator() { reset(); }

    // Advance to time_s. Returns kOk, or an error with the state UNCHANGED.
    int32_t propagate(double time_s, const double gyro[3]) {
        if (!isfinite(time_s) || !finite3(gyro)) return kErrNonFinite;
        if (has_last_time) {
            const double delta = time_s - last_time_s;
            if (delta < 0.0) return kErrTimeRegression;
            if (delta == 0.0) return kErrZeroDt;
            attitude = quatPropagate(attitude, gyro, delta);
        }
        last_time_s = time_s;
        has_last_time = true;
        return kOk;
    }

    // Apply a TRIAD fix if the measurements are usable, else coast on the gyro
    // and record why. A zero-length vector is "no data" (eclipse), never a real
    // direction, and a parallel pair makes TRIAD singular.
    int32_t applyAbsolute(const AbsoluteUpdate &update) {
        last_reject = kRejectNone;
        if (!finite3(update.sun_body) || !finite3(update.magnetic_body) ||
            !finite3(update.sun_ref) || !finite3(update.magnetic_ref)) {
            last_reject = kRejectTriadFailed;
            return kRejectTriadFailed;
        }
        const double sun_norm = norm3(update.sun_body);
        const double magnetic_norm = norm3(update.magnetic_body);
        const double sun_ref_norm = norm3(update.sun_ref);
        const double magnetic_ref_norm = norm3(update.magnetic_ref);
        if (!update.sun_valid || sun_norm <= 0.0 || sun_ref_norm <= 0.0) {
            last_reject = kRejectSunInvalid;
            return last_reject;
        }
        if (!update.magnetic_valid || magnetic_norm <= 0.0 || magnetic_ref_norm <= 0.0) {
            last_reject = kRejectMagInvalid;
            return last_reject;
        }
        double unit_sun[3], unit_mag[3], unit_sun_ref[3], unit_mag_ref[3];
        unit3(update.sun_body, unit_sun);
        unit3(update.magnetic_body, unit_mag);
        unit3(update.sun_ref, unit_sun_ref);
        unit3(update.magnetic_ref, unit_mag_ref);
        if (fabs(dot3(unit_sun_ref, unit_mag_ref)) >= kParallelTolerance ||
            fabs(dot3(unit_sun, unit_mag)) >= kParallelTolerance) {
            last_reject = kRejectParallelPair;
            return last_reject;
        }
        double A[3][3];
        if (!triadDcm(unit_sun_ref, unit_mag_ref, unit_sun, unit_mag, A)) {
            last_reject = kRejectTriadFailed;
            return last_reject;
        }
        attitude = quatFromCbn(A);
        has_absolute_fix = true;
        // The only disagreement the estimator can measure without truth: how far
        // the two measured vectors were from collinear.
        covariance_trace = 1.0 - fabs(dot3(unit_sun, unit_mag));
        return kRejectNone;
    }
};

// ---------------------------------------------------------------------------
// Controller: PD on the attitude error, fed the ESTIMATE and the measured rate.
//
//   tau = Kp * e - Kd * omega
//
// A positive error component means the body must rotate about that axis to
// reach the target, so the commanded torque about it is positive. A mirrored
// controller still settles, just the wrong way.
// ---------------------------------------------------------------------------
inline int32_t controllerTorque(const Config &config, const Quat &estimate,
                                const double rate[3], const Quat &desired,
                                double torque[3]) {
    if (!finiteQ(estimate) || !finiteQ(desired) || !finite3(rate)) return kErrNonFinite;
    if (!(config.proportional_gain > 0.0) || !(config.derivative_gain > 0.0) ||
        !(config.max_torque_nm > 0.0) || !(config.rotor_inertia_kg_m2 > 0.0) ||
        !(config.acceleration_cap_rad_s2 > 0.0) || !(config.max_speed_rad_s > 0.0)) {
        return kErrBadArgument;
    }
    double sigma[3];
    quatToMrp(quatError(desired, estimate), sigma);
    for (int i = 0; i < 3; ++i) {
        const double value = config.proportional_gain * sigma[i] - config.derivative_gain * rate[i];
        torque[i] = value > config.max_torque_nm
                        ? config.max_torque_nm
                        : (value < -config.max_torque_nm ? -config.max_torque_nm : value);
    }
    return kOk;
}

// ---------------------------------------------------------------------------
// Wheel allocation: three orthogonal body-axis wheels under perfect commutation,
// so wheel i produces torque about body axis i and nothing else. The physical
// sign is u = -(A^T tau): a wheel spinning positive about +Z applies a negative
// torque to the body.
//
// Two limits, in this order, matching the PC WheelBoard:
//   1. commanded acceleration cap, |u| <= cap * J  (an engineering parameter,
//      not a datasheet quantity and not a stall model);
//   2. speed bound -- a wheel at max speed refuses torque that pushes further
//      out but may still be braked. Clipping both ways would freeze a wheel
//      that is overspeeding and could never recover.
// `saturated` is set when either limit changed what was delivered.
// ---------------------------------------------------------------------------
struct AllocationResult {
    double torque[3];
    bool saturated;
};

inline int32_t allocateWheelTorque(const Config &config, const double body_torque[3],
                                   const double wheel_speed[3], double dt_s,
                                   AllocationResult *out) {
    if (!finite3(body_torque)) return kErrNonFinite;
    if (!(config.rotor_inertia_kg_m2 > 0.0) || !(config.acceleration_cap_rad_s2 > 0.0) ||
        !(config.max_speed_rad_s > 0.0)) {
        return kErrBadArgument;
    }
    if (wheel_speed != NULL) {
        if (!finite3(wheel_speed) || !isfinite(dt_s) || dt_s <= 0.0) return kErrNonFinite;
    }
    const double cap_torque = config.acceleration_cap_rad_s2 * config.rotor_inertia_kg_m2;
    bool saturated = false;
    for (int i = 0; i < 3; ++i) {
        const double requested = -body_torque[i];
        double applied = requested;
        if (applied > cap_torque) {
            applied = cap_torque;
        } else if (applied < -cap_torque) {
            applied = -cap_torque;
        }
        if (applied != requested) saturated = true;
        if (wheel_speed != NULL) {
            const double projected =
                wheel_speed[i] + applied / config.rotor_inertia_kg_m2 * dt_s;
            if (fabs(projected) > config.max_speed_rad_s && applied * wheel_speed[i] > 0.0) {
                applied = 0.0;
                saturated = true;
            }
        }
        out->torque[i] = applied;
    }
    out->saturated = saturated;
    return kOk;
}

// ---------------------------------------------------------------------------
// Schedule storage: a real SlewOS/FA01 plan, fixed capacity.
//
// Each segment is one planner move. The angle law is the planner's own
// symmetric accel/coast/decel trapezoid, normalised by the EFFECTIVE time
// V = duration - (accel + decel)/2, not by the raw duration: the distance
// covered is v_peak * V, so dividing by the duration rescales the whole move,
// makes the profile discontinuous at the coast/decel junction and implies an
// acceleration the planner never commanded. For a long coast V and duration
// differ by ~1e-6; for a triangular move they differ by 50%.
// ---------------------------------------------------------------------------
struct Segment {
    int32_t objective_id;
    double start_angle_rad;
    double target_angle_rad;
    double duration_s;
    double start_time_s;
    double accel_s;
    double coast_s;
    double decel_s;
};

struct Schedule {
    double initial_angle_rad;
    Segment segments[kMaxSegments];
    int32_t count;
    bool loaded;

    void clear() {
        initial_angle_rad = 0.0;
        count = 0;
        loaded = false;
        for (int32_t i = 0; i < kMaxSegments; ++i) {
            segments[i] = Segment();
        }
    }

    Schedule() { clear(); }

    double totalDurationS() const {
        if (count == 0) return 0.0;
        const Segment &last = segments[count - 1];
        return last.start_time_s + last.duration_s;
    }

    double finalAngleRad() const {
        if (count == 0) return initial_angle_rad;
        return segments[count - 1].target_angle_rad;
    }
};

// Commanded angle inside one segment. Returns false for a degenerate profile
// (no effective time to move in), whose commanded angle is undefined.
inline bool segmentAngleAt(const Segment &s, double t_s, double *out) {
    const double local = t_s - s.start_time_s;
    if (local <= 0.0) {
        *out = s.start_angle_rad;
        return true;
    }
    if (local >= s.duration_s) {
        *out = s.target_angle_rad;
        return true;
    }
    const double effective = s.duration_s - 0.5 * (s.accel_s + s.decel_s);
    if (effective <= 0.0) return false;
    double fraction;
    if (local < s.accel_s) {
        fraction = s.accel_s > 0.0 ? 0.5 * local * local / (s.accel_s * effective) : 0.0;
    } else if (local < s.accel_s + s.coast_s) {
        fraction = (0.5 * s.accel_s + (local - s.accel_s)) / effective;
    } else {
        const double remaining = s.duration_s - local;
        fraction = s.decel_s > 0.0 ? 1.0 - 0.5 * remaining * remaining / (s.decel_s * effective) : 1.0;
    }
    if (fraction < 0.0) fraction = 0.0;
    if (fraction > 1.0) fraction = 1.0;
    *out = s.start_angle_rad + fraction * (s.target_angle_rad - s.start_angle_rad);
    return true;
}

// Planned angle at an absolute time.
inline bool scheduleAngleAt(const Schedule &plan, double t_s, double *out) {
    if (!plan.loaded) return false;
    if (plan.count == 0 || t_s <= 0.0) {
        *out = plan.initial_angle_rad;
        return true;
    }
    if (t_s >= plan.totalDurationS()) {
        *out = plan.finalAngleRad();
        return true;
    }
    for (int32_t i = 0; i < plan.count; ++i) {
        if (t_s <= plan.segments[i].start_time_s + plan.segments[i].duration_s) {
            return segmentAngleAt(plan.segments[i], t_s, out);
        }
    }
    *out = plan.finalAngleRad();
    return true;
}

// Finish loading: every segment must be ordered, finite and inside the
// duration bound, and the plan must fit in kMaxSegments. A plan that does not
// fit is refused, never truncated.
inline int32_t finalizeSchedule(Schedule *plan) {
    if (!isfinite(plan->initial_angle_rad)) return kErrNonFinite;
    if (plan->count <= 0 || plan->count > kMaxSegments) return kErrBadArgument;
    double clock = 0.0;
    for (int32_t i = 0; i < plan->count; ++i) {
        const Segment &s = plan->segments[i];
        if (!isfinite(s.start_angle_rad) || !isfinite(s.target_angle_rad) ||
            !isfinite(s.duration_s) || !isfinite(s.start_time_s) || !isfinite(s.accel_s) ||
            !isfinite(s.coast_s) || !isfinite(s.decel_s)) {
            return kErrNonFinite;
        }
        if (!(s.duration_s > 0.0) || s.accel_s < 0.0 || s.coast_s < 0.0 || s.decel_s < 0.0) {
            return kErrBadArgument;
        }
        if (s.start_time_s != clock) return kErrBadArgument;
        clock += s.duration_s;
        if (clock > kMaxScheduleDurationS) return kErrBadArgument;
    }
    plan->loaded = true;
    return kOk;
}

// ---------------------------------------------------------------------------
// Externally supplied 3D reference profile.
//
// This is NOT planner output. The implemented FA01 engine is single-axis
// rest-to-rest (see PlannerScope in the PC adapter), so multi-axis pointing is
// exercised with a smooth profile supplied from outside and recorded as such.
// Endpoints are exact: start at t=0, target at t=total.
// ---------------------------------------------------------------------------
struct ReferenceProfile {
    Quat start;
    Quat target;
    double total_s;
    bool loaded;

    void clear() {
        start = quatIdentity();
        target = quatIdentity();
        total_s = 0.0;
        loaded = false;
    }

    ReferenceProfile() { clear(); }
};

inline int32_t initReferenceProfile(ReferenceProfile *profile, const Quat &start,
                                    const Quat &target, double total_s) {
    if (!finiteQ(start) || !finiteQ(target) || !isfinite(total_s)) return kErrNonFinite;
    profile->start = start;
    profile->target = target;
    profile->total_s = total_s;
    profile->loaded = true;
    return kOk;
}

inline int32_t referenceAttitudeAt(const ReferenceProfile &profile, double t_s, Quat *out) {
    if (!profile.loaded || !finiteQ(profile.start) || !finiteQ(profile.target)) return kErrBadArgument;
    if (!isfinite(t_s)) return kErrNonFinite;
    if (!(profile.total_s > 0.0)) {
        *out = profile.start;
        return kOk;
    }
    double fraction = t_s / profile.total_s;
    if (fraction < 0.0) fraction = 0.0;
    if (fraction > 1.0) fraction = 1.0;
    // Interpolate the error vector, not the quaternion: it is the
    // well-conditioned form near the identity, and smoothstep easing gives zero
    // rate and acceleration at both ends. The increment multiplies on the LEFT,
    // which is where error_quaternion(target, start) already lives in the C_BN
    // convention; on the right it coincides only at the identity attitude.
    const double eased = fraction * fraction * (3.0 - 2.0 * fraction);
    double sigma[3];
    quatToMrp(quatError(profile.target, profile.start), sigma);
    const double scaled[3] = {eased * sigma[0], eased * sigma[1], eased * sigma[2]};
    *out = quatMul(quatFromMrp(scaled), profile.start);
    return kOk;
}

}  // namespace adcs

#endif  // ADCS_CORE_H
