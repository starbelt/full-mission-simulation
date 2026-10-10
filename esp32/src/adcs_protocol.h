// PocketADCS closed-loop application protocol, version 1 (the "X" family).
//
// WHY A NEW PROTOCOL RATHER THAN A NEW S2
// ---------------------------------------
// The stateless S2/R exchange is already passing an on-board regression suite
// (tests/test_hil_stream_capture.py, tests/test_hil_timed_acceptance.py) and an
// operator soak on COM5. Reusing its frame tag for a *stateful* closed-loop
// session would make those results describe different behaviour than the name
// implies, which is how an invalidated rig stays in the lab looking calibrated.
// So this is a separate, explicitly versioned protocol; S2/R is parsed here too
// and still answers exactly as before.
//
// WHAT THE HOST MAY SEND
// ----------------------
// Only what a real vehicle knows about itself:
//
//   seq, absolute clock, body gyro, body magnetic and sun vectors, and its own
//   position. The inertial REFERENCE vectors are derived onboard from position
//   and time through the ephemeris. Truth attitude, truth rate and
//   precomputed references are not part of this protocol and have no field to
//   travel in.
//
//   Commands carry configuration and a schedule -- gains, limits, planner
//   segments, an externally supplied reference profile. A schedule is a plan
//   the vehicle was told to fly, not the state it is in.
//
// TRANSACTIONAL PARSING
// ---------------------
// A line is parsed into a scratch buffer and only committed once it is known
// good. A malformed line is refused with the estimate, the plan, the
// configuration and the counters all untouched. Partial application is the
// defect this exists to prevent: a run that reports a small final error for a
// state the device never actually held is worse than a loud failure.
//
// HOST COMPATIBILITY
// ------------------
// No Zephyr and no C++ standard library: only <math.h>, <stdio.h> and
// <stdint.h>. The cycle-count source is injected as a function pointer so the
// host test can supply known per-stage costs and the firmware supplies
// k_cycle_get_32(). Nothing in this header allocates.

#ifndef ADCS_PROTOCOL_H
#define ADCS_PROTOCOL_H

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "adcs_core.h"

namespace adcs {

// Longest accepted input line. Anything longer is refused, not truncated: a
// truncated frame parses as a valid frame with different numbers.
enum : int32_t { kMaxLine = 320 };
// Longest formatted reply.
enum : int32_t { kMaxReply = 512 };
// Samples retained for the control-input echo of the most recent frame only;
// the protocol is stateless beyond the estimator, the plan and the counters.
enum : int32_t { kPlanLinesPerSegment = 1 };

// Reference vectors the device derives itself when the caller supplies a
// position but the host test needs deterministic references. In the firmware
// these come from EphemerisReferenceProvider; the host driver injects a fixed
// pair so the differential test does not depend on the IGRF tables.
struct ReferencePair {
    double sun[3];
    double magnetic[3];
    bool valid;
};

// Injected cycle counter. Returns the hardware cycle count (Zephyr) or a known
// per-stage cost (host test). Never null.
typedef uint32_t (*CycleFn)();

// Mutable session state. Every field is either configuration the host chose or
// computed state; none of it can carry a truth attitude in, because nothing
// here has a parameter for one.
struct Session {
    Config config;
    Estimator estimator;
    Schedule schedule;
    ReferenceProfile reference;
    // The externally supplied 3D reference, when one is loaded. Exactly one of
    // `schedule` and `reference` is authoritative; `use_reference` says which.
    bool use_reference;
    // Wheel speeds are the device's own open-loop bookkeeping of what it
    // commanded, not sensed speeds. There is no motor hardware; the host's
    // wheel-board stand-in holds the real rotor state.
    double wheel_speed[3];
    // Control input echoed back: the attitude the torque was computed from.
    Quat controller_input;
    bool controller_input_valid;
    // Absolute time of the last accepted sample, and the gap beyond which the
    // link is declared stale.
    double last_sample_s;
    bool has_last_sample;
    double stale_gap_s;
    bool stale;
    // Diagnostics.
    uint32_t uptime_ms;
    uint32_t rx_drops;
    uint32_t rx_overflows;
    int32_t stack_headroom_bytes;
    // Plan loading is a two-step transaction: XL opens, XS* appends, XE
    // commits. Until XE, the live plan is untouched.
    Schedule pending;
    int32_t pending_count;
    int32_t pending_declared;
    bool plan_open;
    ReferenceProfile pending_reference;
    bool reference_open;
    CycleFn cycle;

    void configureDefaults() {
        // TGR 29-12 triad; the host overrides these with XC.
        config.proportional_gain = 0.35;
        config.derivative_gain = 0.15;
        config.max_torque_nm = 0.0028;
        config.acceleration_cap_rad_s2 = 200.0;
        config.rotor_inertia_kg_m2 = 3.27e-6;
        config.max_speed_rad_s = 1675.5160819145563;
        wheel_speed[0] = wheel_speed[1] = wheel_speed[2] = 0.0;
        stale_gap_s = 2.0;
        reference.clear();
        use_reference = false;
        controller_input = quatIdentity();
        controller_input_valid = false;
        last_sample_s = 0.0;
        has_last_sample = false;
        stale = false;
        uptime_ms = 0;
        rx_drops = 0;
        rx_overflows = 0;
        stack_headroom_bytes = 0;
        pending.clear();
        pending_count = 0;
        pending_declared = 0;
        plan_open = false;
        pending_reference.clear();
        reference_open = false;
    }

    void reset(CycleFn cycle_source, int32_t headroom_bytes) {
        configureDefaults();
        config_set_unconfigured();
        estimator.reset();
        schedule.clear();
        cycle = cycle_source;
        stack_headroom_bytes = headroom_bytes;
    }

    // Explicit isolated HIL mission boundary, NOT live reconfiguration.
    // The host simultaneously creates a fresh plant and wheel-board stand-in.
    // Leave validated gains/limits, calibration outside Session, the platform
    // cycle source and cumulative real health alone. Reset in place: Session
    // contains two large schedules and must never be copied onto the stack.
    void resetMission() {
        estimator.reset();
        schedule.clear();
        reference.clear();
        use_reference = false;
        wheel_speed[0] = wheel_speed[1] = wheel_speed[2] = 0.0;
        controller_input = quatIdentity();
        controller_input_valid = false;
        last_sample_s = 0.0;
        has_last_sample = false;
        stale = false;
        pending.clear();
        pending_count = 0;
        pending_declared = 0;
        plan_open = false;
        pending_reference.clear();
        reference_open = false;
    }

    void config_set_unconfigured() {
        // A session with no explicit configuration is not allowed to fly: the
        // gains are marked invalid so a sample is refused rather than executed
        // with numbers nobody chose.
        config.proportional_gain = 0.0;
    }

    bool configured() const { return config.proportional_gain > 0.0; }
};

// ---------------------------------------------------------------------------
// Bounded numeric field reader.
//
// A field-count mismatch must be a refusal, never a partially filled vector: a
// shifted field is precisely the defect that once made the FIRST case in a
// capture file work and every later one fail.
// ---------------------------------------------------------------------------
struct FieldReader {
    const char *cursor;
    int count;

    explicit FieldReader(const char *text) : cursor(text), count(0) {}

    bool next(double *out) {
        while (*cursor == ' ' || *cursor == '\t') ++cursor;
        if (*cursor == '\0' || *cursor == '\n' || *cursor == '\r') return false;
        char *end = NULL;
        const double value = strtod(cursor, &end);
        if (end == cursor) return false;
        // Reject a bare "nan"/"inf" outright rather than letting a non-finite
        // value reach the estimator and be refused there for an unclear reason.
        if (!isfinite(value)) return false;
        cursor = end;
        ++count;
        *out = value;
        return true;
    }

    bool readVec(double *out, int32_t n) {
        for (int32_t i = 0; i < n; ++i) {
            if (!next(&out[i])) return false;
        }
        return true;
    }

    bool atEnd() {
        while (*cursor == ' ' || *cursor == '\t') ++cursor;
        return *cursor == '\0' || *cursor == '\n' || *cursor == '\r';
    }
};

inline bool readQuatField(FieldReader *reader, Quat *q) {
    double v[4];
    if (!reader->readVec(v, 4)) return false;
    q->w = v[0];
    q->x = v[1];
    q->y = v[2];
    q->z = v[3];
    return true;
}

inline void writeQuat(char *out, const Quat &q) {
    sprintf(out, " %.9f %.9f %.9f %.9f", q.w, q.x, q.y, q.z);
}

// ---------------------------------------------------------------------------
// Reply formatting
// ---------------------------------------------------------------------------

// A closed-loop frame reply. Field order is fixed and documented in
// docs/esp32_closed_loop.md; the HIL harness parses exactly this.
//
//   Y <status> <seq> <has_fix> <q[4]> <ctrl_q[4]> <tau[3]> <u[3]>
//     <saturated> <stale> <reject> <cyc_eph> <cyc_est> <cyc_ctrl> <cyc_alloc>
inline int32_t formatFrame(const Session &session, int32_t status, int32_t seq,
                           const Quat &estimate, const Quat &controller_input,
                           const double torque[3], const double wheel[3], bool saturated,
                           bool stale, int32_t reject, const uint32_t cycles[4],
                           char *reply, int32_t capacity) {
    const int written = snprintf(
        reply, static_cast<size_t>(capacity),
        "Y %d %d %d %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f "
        "%.9f %.9f %.9f %d %d %d %u %u %u %u",
        static_cast<int>(status), static_cast<int>(seq),
        session.estimator.has_absolute_fix ? 1 : 0, estimate.w, estimate.x, estimate.y,
        estimate.z, controller_input.w, controller_input.x, controller_input.y,
        controller_input.z, torque[0], torque[1], torque[2], wheel[0], wheel[1], wheel[2],
        saturated ? 1 : 0, stale ? 1 : 0, static_cast<int>(reject), cycles[0], cycles[1],
        cycles[2], cycles[3]);
    if (written <= 0 || written >= capacity) return 0;  // never emit a partial line
    return written;
}

// A short refusal: the type, a status, and the current state so a caller can
// prove nothing was advanced. No measurement fields, because none happened.
inline int32_t formatRefusal(const Session &session, const char *type, int32_t status,
                             int32_t seq, char *reply, int32_t capacity) {
    const int written = snprintf(reply, static_cast<size_t>(capacity), "E %s %d %d %d",
                                 type, static_cast<int>(status), static_cast<int>(seq),
                                 session.estimator.has_absolute_fix ? 1 : 0);
    if (written <= 0 || written >= capacity) return 0;
    return written;
}

inline int32_t formatHealth(const Session &session, char *reply, int32_t capacity) {
    const int written = snprintf(
        reply, static_cast<size_t>(capacity), "H %d %u %u %u %d %d", 0, session.uptime_ms,
        session.rx_drops, session.rx_overflows, static_cast<int>(session.stack_headroom_bytes),
        session.estimator.has_absolute_fix ? 1 : 0);
    if (written <= 0 || written >= capacity) return 0;
    return written;
}

// ---------------------------------------------------------------------------
// The reference provider seam.
//
// The firmware passes a provider that derives the reference vectors from
// position and clock. It is a function pointer rather than a virtual class so
// this header stays freestanding; the host driver passes a fixed pair.
// ---------------------------------------------------------------------------
typedef bool (*ReferenceFn)(const Session &session, int64_t epoch_ms, const double pos[3],
                            AbsoluteUpdate *out);

struct CallContext {
    Session *session;
    ReferenceFn references;
};

inline bool defaultReferences(const Session &, int64_t, const double[3], AbsoluteUpdate *out) {
    (void)out;
    return false;
}

// Desired attitude -> controller -> allocation -> reply. Declared before
// runControlCycle because the stale-gap path returns through it directly.
inline int32_t finishFrame(CallContext *context, double t_s, const double gyro[3],
                           const Quat &desired, int32_t seq, uint32_t cycles[4],
                           int32_t status_in, char *reply, int32_t capacity);

// ---------------------------------------------------------------------------
// One closed-loop control cycle.
//
// Stage order and what each stage owns:
//
//   1. estimator   -- propagate, then TRIAD fix. Owns the estimate.
//   2. reference   -- planned or externally supplied desired attitude. Never
//                     the truth attitude, only a plan or profile.
//   3. controller  -- PD torque from the ESTIMATE and the measured gyro.
//   4. allocation  -- three orthogonal wheel commands with the board's limits.
//
// The controller's input is copied here, on the way in, so the reply can carry
// the value the torque was really computed from. Reading the caller's intent
// instead would be the same defect as a hardcoded truth-leakage flag.
// ---------------------------------------------------------------------------
inline int32_t runControlCycle(CallContext *context, double t_s, int64_t epoch_ms,
                               const double pos[3], const double gyro[3],
                               const double magnetic[3], const double sun[3], int32_t seq,
                               char *reply, int32_t capacity) {
    Session &session = *context->session;
    uint32_t cycles[4] = {0u, 0u, 0u, 0u};

    // Validate against the last ACCEPTED sample, including a stale/coasted
    // sample, before changing stale flags or touching the estimator. Checking
    // only propagate() allowed duplicates after a gap and mutated stale state
    // on refused regressions. A fresh t0 is legal only after explicit XZ.
    if (session.has_last_sample && t_s <= session.last_sample_s) {
        return formatRefusal(session, "EST",
                             t_s == session.last_sample_s ? kErrZeroDt : kErrTimeRegression,
                             seq, reply, capacity);
    }

    // A zero-length sun vector is "no data" (eclipse), never a real direction,
    // and must not share an encoding with "pointing nowhere".
    AbsoluteUpdate update;
    update.sun_body[0] = sun[0];
    update.sun_body[1] = sun[1];
    update.sun_body[2] = sun[2];
    update.magnetic_body[0] = magnetic[0];
    update.magnetic_body[1] = magnetic[1];
    update.magnetic_body[2] = magnetic[2];
    update.sun_valid = (sun[0] != 0.0 || sun[1] != 0.0 || sun[2] != 0.0);
    update.magnetic_valid = (magnetic[0] != 0.0 || magnetic[1] != 0.0 || magnetic[2] != 0.0);
    update.sun_ref[0] = update.sun_ref[1] = update.sun_ref[2] = 0.0;
    update.magnetic_ref[0] = update.magnetic_ref[1] = update.magnetic_ref[2] = 0.0;

    // --- 0. Desired attitude from the plan or the supplied profile --------
    // Computed before the estimator because both the normal and the stale path
    // need it, and a stale frame must still be a complete reply.
    Quat desired = quatIdentity();
    if (session.use_reference) {
        if (referenceAttitudeAt(session.reference, t_s, &desired) != kOk) {
            return formatRefusal(session, "REF", kErrBadArgument, seq, reply, capacity);
        }
    } else {
        double angle = 0.0;
        if (!session.schedule.loaded) {
            return formatRefusal(session, "PLAN", kErrBadArgument, seq, reply, capacity);
        }
        if (!scheduleAngleAt(session.schedule, t_s, &angle)) {
            return formatRefusal(session, "PLAN", kErrBadArgument, seq, reply, capacity);
        }
        const double axis[3] = {0.0, 0.0, 1.0};
        desired = quatFromAxisAngle(axis, angle);
    }

    // --- 1. Ephemeris: position + clock -> inertial reference vectors -----
    const uint32_t tEph0 = session.cycle();
    const bool have_references = context->references(session, epoch_ms, pos, &update);
    const uint32_t t1 = session.cycle();
    cycles[0] = t1 - tEph0;
    if (!have_references) {
        // The ephemeris failed: leave the estimate alone rather than feed TRIAD
        // references that were never derived. Without both references TRIAD is
        // meaningless, so the sample coasts and the reply says so.
        update.sun_valid = false;
        update.magnetic_valid = false;
    }

    // --- 1. Estimator -----------------------------------------------------
    // A gap longer than the stale threshold is reported, and NOT integrated.
    // Propagating across it multiplies a stale, possibly biased gyro by tens of
    // seconds and produces a large confident error; a TRIAD fix that happens to
    // land afterwards hides that, so the defect survives into a run that looks
    // fine. Coasting is the safe reading of "I do not know what happened", and
    // the reply says so.
    const bool gap = session.has_last_sample &&
                     ((t_s - session.last_sample_s) > session.stale_gap_s);
    session.stale = gap;
    if (gap) {
        session.last_sample_s = t_s;
        // A fix is absolute, not incremental, so a fresh measurement is still
        // worth applying even after a gap -- but it is not enough to *drive*
        // with, which is why actuation stays suppressed below.
        if (have_references) {
            session.estimator.applyAbsolute(update);
        } else {
            session.estimator.last_reject = kRejectSunInvalid;
        }
        const uint32_t tGap = session.cycle();
        cycles[1] = tGap - t1;
        session.last_sample_s = t_s;
        return finishFrame(context, t_s, gyro, desired, seq, cycles, 0, reply, capacity);
    }

    const int32_t propagated = session.estimator.propagate(t_s, gyro);
    if (propagated != kOk) {
        return formatRefusal(session, "EST", propagated, seq, reply, capacity);
    }
    if (have_references) {
        session.estimator.applyAbsolute(update);
    } else {
        session.estimator.last_reject = kRejectSunInvalid;
    }
    const uint32_t t2 = session.cycle();
    cycles[1] = t2 - t1;

    session.last_sample_s = t_s;
    session.has_last_sample = true;

    return finishFrame(context, t_s, gyro, desired, seq, cycles, 0, reply, capacity);
}

// Desired attitude -> controller -> allocation -> reply, shared by the normal
// and stale paths so a stale frame is still a complete, seq-linked reply with
// every field present. A short reply here would be indistinguishable from a
// dropped one.
inline int32_t finishFrame(CallContext *context, double t_s, const double gyro[3],
                           const Quat &desired, int32_t seq, uint32_t cycles[4],
                           int32_t status_in, char *reply, int32_t capacity) {
    Session &session = *context->session;
    (void)t_s;
    (void)status_in;

    // --- 3. Controller, on the ESTIMATE only ------------------------------
    const uint32_t t3 = session.cycle();
    session.controller_input = session.estimator.attitude;
    session.controller_input_valid = true;
    double torque[3] = {0.0, 0.0, 0.0};
    // Before the first absolute fix there is no observed attitude, and a stale
    // link cannot be trusted: command nothing rather than act on a guess.
    const bool may_actuate = session.estimator.has_absolute_fix && !session.stale;
    int32_t status = kOk;
    if (!may_actuate) {
        torque[0] = torque[1] = torque[2] = 0.0;
        status = session.stale ? kErrTimeRegression : kErrBadArgument;
    } else {
        status = controllerTorque(session.config, session.estimator.attitude, gyro, desired,
                                  torque);
    }
    const uint32_t t4 = session.cycle();
    cycles[2] = t4 - t3;

    // --- 4. Allocation ----------------------------------------------------
    AllocationResult allocation;
    allocation.torque[0] = allocation.torque[1] = allocation.torque[2] = 0.0;
    allocation.saturated = false;
    if (may_actuate) {
        allocateWheelTorque(session.config, torque, session.wheel_speed, 0.05, &allocation);
        // Bookkeeping only: this is what the device commanded, integrated
        // open-loop. It is not a sensed wheel speed -- there is no motor
        // hardware, and the host's board stand-in holds the real rotor state.
        // Rotor momentum is preserved across a suppressed cycle, never reset:
        // zero-filling wheel state on a dropout turns a comms gap into a full
        // attitude excursion because the stored momentum silently disappears.
        for (int i = 0; i < 3; ++i) {
            session.wheel_speed[i] +=
                allocation.torque[i] / session.config.rotor_inertia_kg_m2 * 0.05;
        }
    }
    const uint32_t t5 = session.cycle();
    cycles[3] = t5 - t4;

    return formatFrame(session, status, seq, session.estimator.attitude,
                       session.controller_input, torque, allocation.torque,
                       allocation.saturated, session.stale, session.estimator.last_reject,
                       cycles, reply, capacity);
}

// ---------------------------------------------------------------------------
// Line dispatch.
//
// Every branch is transactional: it parses into locals, and commits to session
// state only once the whole line is known good. A refusal changes nothing but
// the reply. The alternative -- half-applying a command -- produces a run that
// reports a small final error for a state the device never held.
// ---------------------------------------------------------------------------

// The legacy stateless frame, unchanged. Parsed here so the old regression
// suite keeps describing the same behaviour after this protocol is added.
inline bool parseLegacyS2(const char *text, int32_t *seq, int64_t *epoch_ms, double gyro[3],
                          double magnetic[3], double sun[3], double pos[3]) {
    if (text[0] != 'S' || text[1] != '2' || text[2] != ' ') return false;
    FieldReader reader(text + 3);
    double seq_value = 0.0;
    double epoch_value = 0.0;
    if (!reader.next(&seq_value) || !reader.next(&epoch_value)) return false;
    if (!reader.readVec(gyro, 3) || !reader.readVec(magnetic, 3) || !reader.readVec(sun, 3) ||
        !reader.readVec(pos, 3)) {
        return false;
    }
    if (!reader.atEnd()) return false;
    *seq = static_cast<int32_t>(seq_value);
    *epoch_ms = static_cast<int64_t>(epoch_value);
    return true;
}

inline int32_t handleLine(CallContext *context, const char *line, char *reply,
                          int32_t capacity) {
    Session &session = *context->session;
    if (line[0] == '\0') {
        // An empty line is a keepalive, not a reset and not an error.
        return formatHealth(session, reply, capacity);
    }

    // --- legacy stateless S2 ---------------------------------------------
    if (line[0] == 'S') {
        int32_t seq = 0;
        int64_t epoch_ms = 0;
        double gyro[3], magnetic[3], sun[3], pos[3];
        if (!parseLegacyS2(line, &seq, &epoch_ms, gyro, magnetic, sun, pos)) {
            return formatRefusal(session, "S2", kErrBadArgument, -1, reply, capacity);
        }
        // The legacy path stays stateless: it reports the attitude the
        // onboard ephemeris and TRIAD produce for this one frame, with no
        // propagation and no control. It is not the closed loop.
        AbsoluteUpdate update;
        update.sun_body[0] = sun[0];
        update.sun_body[1] = sun[1];
        update.sun_body[2] = sun[2];
        update.magnetic_body[0] = magnetic[0];
        update.magnetic_body[1] = magnetic[1];
        update.magnetic_body[2] = magnetic[2];
        update.sun_ref[0] = update.sun_ref[1] = update.sun_ref[2] = 0.0;
        update.magnetic_ref[0] = update.magnetic_ref[1] = update.magnetic_ref[2] = 0.0;
        update.sun_valid = (sun[0] != 0.0 || sun[1] != 0.0 || sun[2] != 0.0);
        update.magnetic_valid =
            (magnetic[0] != 0.0 || magnetic[1] != 0.0 || magnetic[2] != 0.0);
        const bool have_references =
            context->references(session, epoch_ms, pos, &update);
        if (!have_references) {
            update.sun_valid = false;
            update.magnetic_valid = false;
        }
        Estimator scratch;
        scratch.reset();
        scratch.applyAbsolute(update);
        const double zero[3] = {0.0, 0.0, 0.0};
        const Quat identity = quatIdentity();
        uint32_t cycles[4] = {0u, 0u, 0u, 0u};
        return formatFrame(session, scratch.has_absolute_fix ? kOk : kErrBadArgument, seq,
                           scratch.attitude, identity, zero, zero, false, false,
                           scratch.last_reject, cycles, reply, capacity);
    }

    // --- closed-loop session commands ------------------------------------
    const char *rest = line + 1;
    if (line[0] == 'X') {
        if (rest[0] == 'C' && rest[1] == ' ') {
            // XC kp kd max_torque accel_cap inertia max_speed
            FieldReader reader(rest + 2);
            double values[6];
            if (!reader.readVec(values, 6) || !reader.atEnd()) {
                return formatRefusal(session, "XC", kErrBadArgument, -1, reply, capacity);
            }
            Config candidate = session.config;
            candidate.proportional_gain = values[0];
            candidate.derivative_gain = values[1];
            candidate.max_torque_nm = values[2];
            candidate.acceleration_cap_rad_s2 = values[3];
            candidate.rotor_inertia_kg_m2 = values[4];
            candidate.max_speed_rad_s = values[5];
            // Refuse an invalid configuration outright rather than falling back
            // to a default nobody chose.
            if (!(candidate.proportional_gain > 0.0) || !(candidate.derivative_gain > 0.0) ||
                !(candidate.max_torque_nm > 0.0) ||
                !(candidate.acceleration_cap_rad_s2 > 0.0) ||
                !(candidate.rotor_inertia_kg_m2 > 0.0) ||
                !(candidate.max_speed_rad_s > 0.0)) {
                return formatRefusal(session, "XC", kErrBadArgument, -1, reply, capacity);
            }
            session.config = candidate;
            return formatRefusal(session, "XC", kOk, -1, reply, capacity);
        }
        if (rest[0] == 'L' && rest[1] == ' ') {
            // XL initial_angle declared_segments -- opens a plan transaction.
            FieldReader reader(rest + 2);
            double initial = 0.0;
            double count = 0.0;
            if (!reader.next(&initial) || !reader.next(&count) || !reader.atEnd()) {
                return formatRefusal(session, "XL", kErrBadArgument, -1, reply, capacity);
            }
            // A count of 0 is legal and means "this transaction carries a
            // reference profile, not a plan": XR supplies no segments, and XE
            // commits whichever of the two the transaction actually collected.
            // Refusing count 0 here would make the documented 3D pointing
            // sequence unreachable -- a path that looked implemented and was
            // not. A count beyond capacity is still refused, since it cannot
            // ever be satisfied.
            if (count < 0.0 || count > static_cast<double>(kMaxSegments) ||
                count != floor(count)) {
                return formatRefusal(session, "XL", kErrBadArgument, -1, reply, capacity);
            }
            session.pending.clear();
            session.pending.initial_angle_rad = initial;
            session.pending_count = 0;
            session.pending_declared = static_cast<int32_t>(count);
            session.plan_open = true;
            return formatRefusal(session, "XL", kOk, -1, reply, capacity);
        }
        if (rest[0] == 'S' && rest[1] == ' ') {
            if (!session.plan_open) {
                return formatRefusal(session, "XS", kErrBadArgument, -1, reply, capacity);
            }
            // XS objective start target duration accel coast decel
            // (start_time is implicit: segments are contiguous by construction,
            // which is also what makes a mis-ordered plan detectable at XE.)
            FieldReader reader(rest + 2);
            double values[7];
            if (!reader.readVec(values, 7) || !reader.atEnd()) {
                return formatRefusal(session, "XS", kErrBadArgument, -1, reply, capacity);
            }
            if (session.pending_count >= static_cast<int32_t>(kMaxSegments)) {
                return formatRefusal(session, "XS", kErrBadArgument, -1, reply, capacity);
            }
            Segment &segment = session.pending.segments[session.pending_count];
            segment.objective_id = static_cast<int32_t>(values[0]);
            segment.start_angle_rad = values[1];
            segment.target_angle_rad = values[2];
            segment.duration_s = values[3];
            segment.accel_s = values[4];
            segment.coast_s = values[5];
            segment.decel_s = values[6];
            double clock = 0.0;
            for (int32_t i = 0; i < session.pending_count; ++i) {
                clock += session.pending.segments[i].duration_s;
            }
            segment.start_time_s = clock;
            ++session.pending_count;
            return formatRefusal(session, "XS", kOk, -1, reply, capacity);
        }
        if (rest[0] == 'E') {
            // XE commits the pending plan, or closes the pending 3D reference.
            if (session.reference_open) {
                if (session.pending_count < 0) {
                    return formatRefusal(session, "XE", kErrBadArgument, -1, reply, capacity);
                }
                session.reference = session.pending_reference;
                session.use_reference = true;
                session.reference_open = false;
                return formatRefusal(session, "XE", kOk, -1, reply, capacity);
            }
            if (!session.plan_open) {
                return formatRefusal(session, "XE", kErrBadArgument, -1, reply, capacity);
            }
            // A zero-segment transaction that collected no profile commits
            // nothing, and must not silently clear the live plan. Committing a
            // zero-segment plan would leave the device with no desired attitude,
            // which reads downstream as "everything is on target".
            if (session.pending_declared == 0 && !session.reference_open) {
                return formatRefusal(session, "XE", kErrBadArgument, -1, reply, capacity);
            }
            // Validate the static pending store in place; never copy a 64-
            // segment schedule onto the main stack. The live plan is untouched
            // until validation succeeds.
            if (session.pending_count != session.pending_declared) {
                return formatRefusal(session, "XE", kErrBadArgument, -1, reply, capacity);
            }
            const int32_t saved_count = session.pending.count;
            const bool saved_loaded = session.pending.loaded;
            session.pending.count = session.pending_count;
            const int32_t validated = finalizeSchedule(&session.pending);
            if (validated == kOk) session.schedule = session.pending;
            // Match the original scratch-copy transaction: validation (whether
            // accepted or refused) never changes the pending metadata.
            session.pending.count = saved_count;
            session.pending.loaded = saved_loaded;
            if (validated != kOk) {
                return formatRefusal(session, "XE", kErrBadArgument, -1, reply, capacity);
            }
            session.use_reference = false;
            session.plan_open = false;
            return formatRefusal(session, "XE", kOk, -1, reply, capacity);
        }
        if (rest[0] == 'R' && rest[1] == ' ') {
            // XR qStart[4] qTarget[4] total_s -- an EXTERNALLY SUPPLIED 3D
            // reference profile. This is not planner output: the implemented
            // FA01 engine is single-axis rest-to-rest, so multi-axis pointing
            // is exercised with a profile supplied from outside and recorded
            // as such.
            FieldReader reader(rest + 2);
            double values[9];
            if (!reader.readVec(values, 9) || !reader.atEnd()) {
                return formatRefusal(session, "XR", kErrBadArgument, -1, reply, capacity);
            }
            ReferenceProfile candidate;
            Quat start, target;
            start.w = values[0]; start.x = values[1];
            start.y = values[2]; start.z = values[3];
            target.w = values[4]; target.x = values[5];
            target.y = values[6]; target.z = values[7];
            if (initReferenceProfile(&candidate, start, target, values[8]) != kOk) {
                return formatRefusal(session, "XR", kErrBadArgument, -1, reply, capacity);
            }
            session.pending_reference = candidate;
            session.pending_count = 0;
            session.reference_open = true;
            return formatRefusal(session, "XR", kOk, -1, reply, capacity);
        }
        if (rest[0] == 'S' && rest[1] == '2' && rest[2] == ' ') {
            // XS2 is the closed-loop sensor frame. It is a distinct tag from XS
            // (plan segment) so a mis-typed plan line cannot be read as data.
            // XS2 seq epoch_ms t_s gyro[3] mag[3] sun[3] pos[3] = 15 numbers.
            FieldReader reader(rest + 3);
            double values[15];
            if (!reader.readVec(values, 15) || !reader.atEnd()) {
                return formatRefusal(session, "XS2", kErrBadArgument, -1, reply, capacity);
            }
            if (!session.configured()) {
                return formatRefusal(session, "XS2", kErrBadArgument,
                                     static_cast<int32_t>(values[0]), reply, capacity);
            }
            const double pos[3] = {values[12], values[13], values[14]};
            const double magnetic[3] = {values[6], values[7], values[8]};
            const double sun[3] = {values[9], values[10], values[11]};
            const double gyro[3] = {values[3], values[4], values[5]};
            const int64_t epoch_ms = static_cast<int64_t>(values[1]);
            return runControlCycle(context, values[2], epoch_ms, pos, gyro, magnetic, sun,
                                   static_cast<int32_t>(values[0]), reply, capacity);
        }
        if (rest[0] == 'D' && rest[1] == ' ') {
            // XD drops overflows -- the host reports what its side lost, so a
            // clean verdict cannot hide a lossy link.
            FieldReader reader(rest + 2);
            double drops = 0.0;
            double overflows = 0.0;
            if (!reader.next(&drops) || !reader.next(&overflows) || !reader.atEnd()) {
                return formatRefusal(session, "XD", kErrBadArgument, -1, reply, capacity);
            }
            if (drops < 0.0 || overflows < 0.0) {
                return formatRefusal(session, "XD", kErrBadArgument, -1, reply, capacity);
            }
            session.rx_drops = static_cast<uint32_t>(drops);
            session.rx_overflows = static_cast<uint32_t>(overflows);
            return formatRefusal(session, "XD", kOk, -1, reply, capacity);
        }
        if (rest[0] == 'O' && rest[1] == ' ') {
            FieldReader reader(rest + 2);
            double overflows = 0.0;
            if (!reader.next(&overflows) || !reader.atEnd() || overflows < 0.0) {
                return formatRefusal(session, "XO", kErrBadArgument, -1, reply, capacity);
            }
            session.rx_overflows = static_cast<uint32_t>(overflows);
            return formatRefusal(session, "XO", kOk, -1, reply, capacity);
        }
        if (rest[0] == 'H') {
            return formatHealth(session, reply, capacity);
        }
        if (rest[0] == 'G' && rest[1] == 'S' && rest[2] == ' ') {
            // XGS <uptime_ms> -- the device's own boot/health clock.
            FieldReader reader(rest + 3);
            double uptime = 0.0;
            if (!reader.next(&uptime) || !reader.atEnd() || uptime < 0.0) {
                return formatRefusal(session, "XGS", kErrBadArgument, -1, reply, capacity);
            }
            session.uptime_ms = static_cast<uint32_t>(uptime);
            return formatHealth(session, reply, capacity);
        }
        if (rest[0] == 'Z') {
            // XZ is a validated, explicit fresh isolated HIL mission reset.
            // Never call it for live gain/plan/reference changes.
            FieldReader reader(rest + 1);
            if (!reader.atEnd()) {
                return formatRefusal(session, "XZ", kErrBadArgument, -1, reply, capacity);
            }
            session.resetMission();
            return formatRefusal(session, "XZ", kOk, -1, reply, capacity);
        }
    }
    return formatRefusal(session, "?", kErrBadArgument, -1, reply, capacity);
}

}  // namespace adcs

#endif  // ADCS_PROTOCOL_H
