// Host entrypoint for the differential ADCS core test.
//
// Compiles esp32/src/adcs_core.h -- the exact header the Zephyr application
// compiles -- with a host C++ compiler and drives it from a line script, so
// tests/test_adcs_core_differential.py can compare the device's arithmetic
// against the accepted Python core sample by sample.
//
// One reply line per command, all numeric, in the command's own order. Nothing
// here is part of the firmware: it exists so a convention error in the core
// fails on the host instead of as a plausible-looking HIL run.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "adcs_core.h"

namespace {

// A command line is a keyword followed by whitespace-separated numbers. Reading
// them positionally (rather than with a per-command sscanf format) keeps every
// command in this file to a bounds check and a call, and a wrong field count
// becomes a refusal instead of a silently shifted vector -- which is exactly
// the class of bug that produced a whole debug cycle elsewhere in this repo.
const int kMaxArgs = 32;

int splitArgs(const char *text, double *values, int capacity) {
    int count = 0;
    const char *cursor = text;
    while (*cursor != '\0' && *cursor != '\n' && count < capacity) {
        char *end = NULL;
        const double value = strtod(cursor, &end);
        if (end == cursor) {
            ++cursor;  // skip a non-numeric byte; the count check catches it
            continue;
        }
        values[count++] = value;
        cursor = end;
    }
    return count;
}

double argsAt(const double *values, int index) { return values[index]; }

void quatFromArgs(const double *values, int index, adcs::Quat *q) {
    q->w = values[index];
    q->x = values[index + 1];
    q->y = values[index + 2];
    q->z = values[index + 3];
}

void printQuat(const adcs::Quat &q) {
    printf(" %.17g %.17g %.17g %.17g", q.w, q.x, q.y, q.z);
}

void printStatus(int32_t status) { printf(" %d", static_cast<int>(status)); }

adcs::Config g_config;
adcs::Estimator g_estimator;
adcs::Schedule g_schedule;
adcs::ReferenceProfile g_reference;
int32_t g_declared_segments = 0;
int32_t g_stored_segments = 0;

void defaultConfig() {
    // TGR 29-12 triad values, matching pocketadcs_sim.adcs.rotors.
    g_config.proportional_gain = 0.35;
    g_config.derivative_gain = 0.15;
    g_config.max_torque_nm = 0.0028;
    g_config.acceleration_cap_rad_s2 = 200.0;
    g_config.rotor_inertia_kg_m2 = 3.27e-6;
    g_config.max_speed_rad_s = 16000.0 * 2.0 * 3.14159265358979323846 / 60.0;
}

// A refused sample echoes the estimate so the caller can prove it did not move.
void printRefused(int32_t status) {
    printStatus(status);
    printQuat(g_estimator.attitude);
    printf("\n");
}

void printEstimate() {
    printStatus(adcs::kOk);
    printQuat(g_estimator.attitude);
    printf(" %d %.17g\n", g_estimator.has_absolute_fix ? 1 : 0, g_estimator.covariance_trace);
}

}  // namespace

int main() {
    defaultConfig();
    char line[4096];
    while (fgets(line, sizeof(line), stdin) != NULL) {
        char op[16] = {0};
        if (sscanf(line, "%15s", op) != 1) continue;
        const char *rest = strchr(line, ' ');
        while (rest != NULL && *rest == ' ') ++rest;
        if (rest == NULL) rest = line + strlen(line);

        double a[kMaxArgs];
        const int n = splitArgs(rest, a, kMaxArgs);

        if (strcmp(op, "CFG") == 0) {
            if (n != 6) {
                printStatus(adcs::kErrBadArgument);
                printf("\n");
                continue;
            }
            g_config.proportional_gain = argsAt(a, 0);
            g_config.derivative_gain = argsAt(a, 1);
            g_config.max_torque_nm = argsAt(a, 2);
            g_config.acceleration_cap_rad_s2 = argsAt(a, 3);
            g_config.rotor_inertia_kg_m2 = argsAt(a, 4);
            g_config.max_speed_rad_s = argsAt(a, 5);
            printStatus(adcs::kOk);
            printf("\n");

        } else if (strcmp(op, "RESET") == 0) {
            g_estimator.reset();
            g_schedule.clear();
            g_reference.clear();
            g_declared_segments = 0;
            g_stored_segments = 0;
            printStatus(adcs::kOk);
            printf("\n");

        } else if (strcmp(op, "SETQ") == 0) {
            // Seed the estimate directly, so a propagation case can start away
            // from the identity attitude where left/right product order differ.
            if (n != 4) {
                printRefused(adcs::kErrBadArgument);
                continue;
            }
            adcs::Quat q;
            quatFromArgs(a, 0, &q);
            if (!adcs::finiteQ(q)) {
                printRefused(adcs::kErrNonFinite);
                continue;
            }
            g_estimator.attitude = adcs::quatNormalized(q.w, q.x, q.y, q.z);
            g_estimator.has_last_time = false;
            printStatus(adcs::kOk);
            printQuat(g_estimator.attitude);
            printf("\n");

        } else if (strcmp(op, "PROP") == 0) {
            // 4 args: omega[3], dt. Propagation of the stored estimate, so a
            // refused dt is observable as an unchanged quaternion.
            if (n != 4) {
                printRefused(adcs::kErrBadArgument);
                continue;
            }
            double omega[3] = {a[0], a[1], a[2]};
            const double dt = a[3];
            if (!isfinite(dt) || !adcs::finite3(omega)) {
                printRefused(adcs::kErrNonFinite);
                continue;
            }
            if (dt == 0.0) {
                printRefused(adcs::kErrZeroDt);
                continue;
            }
            g_estimator.attitude = adcs::quatPropagate(g_estimator.attitude, omega, dt);
            printStatus(adcs::kOk);
            printQuat(g_estimator.attitude);
            printf("\n");

        } else if (strcmp(op, "EST") == 0) {
            // t gyro[3] flags sun[3] mag[3] refSun[3] refMag[3] = 17 numbers.
            if (n != 17) {
                printRefused(adcs::kErrBadArgument);
                continue;
            }
            adcs::AbsoluteUpdate update;
            update.sun_body[0] = a[5];
            update.sun_body[1] = a[6];
            update.sun_body[2] = a[7];
            update.magnetic_body[0] = a[8];
            update.magnetic_body[1] = a[9];
            update.magnetic_body[2] = a[10];
            update.sun_ref[0] = a[11];
            update.sun_ref[1] = a[12];
            update.sun_ref[2] = a[13];
            update.magnetic_ref[0] = a[14];
            update.magnetic_ref[1] = a[15];
            update.magnetic_ref[2] = a[16];
            const int flags = static_cast<int>(a[4]);
            update.sun_valid = (flags & 1) != 0;
            update.magnetic_valid = (flags & 2) != 0;
            double gyro[3] = {a[1], a[2], a[3]};
            // Propagate first: nothing is advanced unless this succeeds, so a
            // refused sample leaves the estimator exactly as it was.
            const int32_t status = g_estimator.propagate(a[0], gyro);
            if (status != adcs::kOk) {
                printRefused(status);
                continue;
            }
            g_estimator.applyAbsolute(update);
            printEstimate();

        } else if (strcmp(op, "STATE") == 0) {
            printf(" %d\n", g_estimator.has_absolute_fix ? 1 : 0);

        } else if (strcmp(op, "REJECT") == 0) {
            printf(" %d\n", static_cast<int>(g_estimator.last_reject));

        } else if (strcmp(op, "CTRL") == 0) {
            // qEst[4] rate[3] qDes[4] = 11 numbers.
            if (n != 11) {
                printStatus(adcs::kErrBadArgument);
                printf("\n");
                continue;
            }
            adcs::Quat estimate, desired;
            quatFromArgs(a, 0, &estimate);
            double rate[3] = {a[4], a[5], a[6]};
            quatFromArgs(a, 7, &desired);
            double torque[3] = {0.0, 0.0, 0.0};
            const int32_t status =
                adcs::controllerTorque(g_config, estimate, rate, desired, torque);
            printStatus(status);
            printf(" %.17g %.17g %.17g\n", torque[0], torque[1], torque[2]);

        } else if (strcmp(op, "WHEEL") == 0) {
            // body[3] or body[3] speed[3] dt.
            if (n != 3 && n != 7) {
                printStatus(adcs::kErrBadArgument);
                printf("\n");
                continue;
            }
            const double body[3] = {a[0], a[1], a[2]};
            double speed[3] = {0.0, 0.0, 0.0};
            double dt = 0.05;
            const bool with_speed = n == 7;
            if (with_speed) {
                speed[0] = a[3];
                speed[1] = a[4];
                speed[2] = a[5];
                dt = a[6];
            }
            adcs::AllocationResult result;
            result.torque[0] = result.torque[1] = result.torque[2] = 0.0;
            result.saturated = false;
            const int32_t status = adcs::allocateWheelTorque(
                g_config, body, with_speed ? speed : NULL, dt, &result);
            printStatus(status);
            printf(" %.17g %.17g %.17g %d\n", result.torque[0], result.torque[1], result.torque[2],
                   result.saturated ? 1 : 0);

        } else if (strcmp(op, "SCHED") == 0) {
            // initial_angle, declared segment count.
            if (n != 2) {
                printStatus(adcs::kErrBadArgument);
                printf("\n");
                continue;
            }
            g_schedule.clear();
            g_schedule.initial_angle_rad = a[0];
            g_declared_segments = static_cast<int32_t>(a[1]);
            g_stored_segments = 0;
            printStatus(adcs::kOk);
            printf("\n");

        } else if (strcmp(op, "SEG") == 0) {
            // objective, start, target, duration, start_time, accel, coast, decel.
            if (n != 8) {
                printStatus(adcs::kErrBadArgument);
                printf("\n");
                continue;
            }
            adcs::Segment segment;
            segment.objective_id = static_cast<int32_t>(a[0]);
            segment.start_angle_rad = a[1];
            segment.target_angle_rad = a[2];
            segment.duration_s = a[3];
            segment.start_time_s = a[4];
            segment.accel_s = a[5];
            segment.coast_s = a[6];
            segment.decel_s = a[7];
            if (g_stored_segments < adcs::kMaxSegments) {
                g_schedule.segments[g_stored_segments] = segment;
            }
            ++g_stored_segments;
            printStatus(adcs::kOk);
            printf("\n");

        } else if (strcmp(op, "SCHEDOK") == 0) {
            // The DECLARED count is what finalization judges, so a plan larger
            // than the store is refused rather than silently truncated.
            g_schedule.count = g_declared_segments;
            const int32_t status = adcs::finalizeSchedule(&g_schedule);
            printStatus(status);
            printf(" %.17g %.17g\n", g_schedule.totalDurationS(), g_schedule.finalAngleRad());

        } else if (strcmp(op, "ANGLE") == 0) {
            if (n != 1) {
                printStatus(adcs::kErrBadArgument);
                printf("\n");
                continue;
            }
            double angle = 0.0;
            const bool ok = adcs::scheduleAngleAt(g_schedule, a[0], &angle);
            printStatus(ok ? adcs::kOk : adcs::kErrBadArgument);
            printf(" %.17g\n", angle);

        } else if (strcmp(op, "TOTAL") == 0) {
            printf(" %.17g %.17g\n", g_schedule.totalDurationS(), g_schedule.finalAngleRad());

        } else if (strcmp(op, "FINAL") == 0) {
            printf(" %.17g\n", g_schedule.finalAngleRad());

        } else if (strcmp(op, "REFINIT") == 0) {
            // qStart[4] qTarget[4] total_s.
            if (n != 9) {
                printStatus(adcs::kErrBadArgument);
                printf("\n");
                continue;
            }
            adcs::Quat start, target;
            quatFromArgs(a, 0, &start);
            quatFromArgs(a, 4, &target);
            printStatus(adcs::initReferenceProfile(&g_reference, start, target, a[8]));
            printf("\n");

        } else if (strcmp(op, "REF") == 0) {
            if (n != 1) {
                printStatus(adcs::kErrBadArgument);
                printf("\n");
                continue;
            }
            adcs::Quat out = adcs::quatIdentity();
            const int32_t status = adcs::referenceAttitudeAt(g_reference, a[0], &out);
            printStatus(status);
            printQuat(out);
            printf("\n");

        } else if (strcmp(op, "PLANREF") == 0) {
            // 0 = externally supplied reference, explicitly not planner output.
            printf(" 0\n");

        } else {
            printStatus(adcs::kErrBadArgument);
            printf("\n");
        }
    }
    return 0;
}
