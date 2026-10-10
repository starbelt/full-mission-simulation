// Host entrypoint for the closed-loop protocol test.
//
// Compiles esp32/src/adcs_protocol.h -- the exact header the Zephyr
// application compiles -- and feeds it a line script. Every reply is echoed as
// pure whitespace-separated numbers, with a leading record type, so the
// pytest side compares fields rather than re-parsing text:
//
//   <type> <fields...>
//
// where <type> is 0 for a Y frame, 1 for an E refusal and 2 for an H health
// line. The Y field order is the one documented in docs/esp32_closed_loop.md.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "adcs_protocol.h"
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include "command_framing.h"

namespace {

// Injected cycle source. Real costs, distinct per stage, so a test can tell a
// genuinely separate measurement from one lumped number copied four times.
uint32_t g_now = 0u;
int g_stage = 0;

uint32_t hostCycle() {
    // Deterministic, monotonically increasing, and different per stage.
    g_now += 100u + 37u * static_cast<uint32_t>(g_stage);
    ++g_stage;
    return g_now;
}

// A fixed reference pair, standing in for the onboard ephemeris. The protocol
// test is about framing, parsing and transactional state; the ephemeris itself
// is covered separately by tests/test_sun_ephemeris_host.py and by the
// firmware's own ephemeris stage timing.
bool hostReferences(const adcs::Session &, int64_t, const double[3], adcs::AbsoluteUpdate *out) {
    out->sun_ref[0] = 1.0;
    out->sun_ref[1] = 0.1;
    out->sun_ref[2] = -0.2;
    out->magnetic_ref[0] = 0.2;
    out->magnetic_ref[1] = 1.0;
    out->magnetic_ref[2] = 0.3;
    return true;
}

void echoReply(const char *text) {
    const char *cursor = text;
    int type = 2;
    if (text[0] == 'Y') {
        type = 0;
        cursor = text + 1;
    } else if (text[0] == 'E') {
        type = 1;
        // "E <tag> <status> <seq> <has_fix>" -- the tag is a word, so skip it.
        cursor = strchr(text, ' ');
        if (cursor != NULL) cursor = strchr(cursor + 1, ' ');
    } else if (text[0] == 'H') {
        type = 2;
        cursor = text + 1;
    }
    printf("%d", type);
    while (*cursor != '\0') {
        if (*cursor == ' ' || *cursor == '\n' || *cursor == '\r') {
            ++cursor;
            continue;
        }
        char *end = NULL;
        const double value = strtod(cursor, &end);
        if (end == cursor) {
            ++cursor;
            continue;
        }
        printf(" %.17g", value);
        cursor = end;
    }
    printf("\n");
}

}  // namespace

int main() {
#ifdef _WIN32
    // Text-mode stdin treats Ctrl-Z as EOF and can truncate into bare XZ.
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    adcs::Session session;
    // 4096 bytes of headroom is a plausible non-zero figure; the protocol only
    // has to carry the number the firmware measures.
    session.reset(&hostCycle, 4096);
    adcs::CallContext context;
    context.session = &session;
    context.references = &hostReferences;

    char reply[adcs::kMaxReply];
    // Read a line without fgets, which would split an over-long line into two
    // replies instead of refusing it once. A line longer than the device's
    // buffer is consumed whole and refused: truncating it would produce a
    // syntactically valid frame carrying different numbers.
    char line[adcs::kMaxLine];
    for (;;) {
        int c = getchar();
        if (c == EOF) break;
        size_t length = 0;
        bool overlong = false;
        bool invalid = false;
        while (c != EOF && c != '\n') {
            if (!adcs::commandByteValid(static_cast<unsigned char>(c))) invalid = true;
            if (!invalid) {
                if (length + 1 >= static_cast<size_t>(adcs::kMaxLine)) {
                    overlong = true;
                } else {
                    line[length++] = adcs::commandTextByte(static_cast<unsigned char>(c));
                }
            }
            c = getchar();
        }
        if (c == EOF && length == 0 && !overlong && !invalid) break;
        // Windows text callers send CRLF even though stdin is binary. Trim
        // trailing whitespace only; internal CR separators must survive.
        while (length && (line[length - 1] == ' ' || line[length - 1] == '\t')) --length;
        line[length] = '\0';
        if (overlong || invalid) {
            adcs::formatRefusal(session, invalid ? "LINE" : "LONG",
                                adcs::kErrBadArgument, -1, reply, sizeof(reply));
            echoReply(reply);
            if (c == EOF) break;
            continue;
        }
        if (strcmp(line, "TEST_STORES_CLEAR") == 0) {
            bool clear = true;
            const adcs::Schedule *plans[] = {&session.schedule, &session.pending};
            for (int i = 0; i < 2; ++i) {
                clear = clear && !plans[i]->loaded && plans[i]->count == 0 &&
                        plans[i]->initial_angle_rad == 0.0;
                for (int j = 0; j < adcs::kMaxSegments; ++j) {
                    const adcs::Segment &s = plans[i]->segments[j];
                    clear = clear && s.objective_id == 0 && s.start_angle_rad == 0.0 &&
                            s.target_angle_rad == 0.0 && s.duration_s == 0.0 &&
                            s.start_time_s == 0.0 && s.accel_s == 0.0 &&
                            s.coast_s == 0.0 && s.decel_s == 0.0;
                }
            }
            const adcs::ReferenceProfile *refs[] = {&session.reference,
                                                   &session.pending_reference};
            for (int i = 0; i < 2; ++i) {
                const adcs::ReferenceProfile &r = *refs[i];
                clear = clear && !r.loaded && r.total_s == 0.0 &&
                        r.start.w == 1.0 && r.start.x == 0.0 &&
                        r.start.y == 0.0 && r.start.z == 0.0 &&
                        r.target.w == 1.0 && r.target.x == 0.0 &&
                        r.target.y == 0.0 && r.target.z == 0.0;
            }
            printf("7 %d\n", clear ? 1 : 0);
            continue;
        }
        if (strcmp(line, "TEST_DYNAMIC") == 0) {
            const adcs::Estimator &e = session.estimator;
            printf("4 %.17g %.17g %.17g %.17g %.17g %d %d %d %.17g "
                   "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %.17g %d %d\n",
                   e.attitude.w, e.attitude.x, e.attitude.y, e.attitude.z,
                   e.last_time_s, int(e.has_last_time), int(e.has_absolute_fix),
                   int(e.last_reject), e.covariance_trace,
                   session.wheel_speed[0], session.wheel_speed[1], session.wheel_speed[2],
                   session.controller_input.w, session.controller_input.x,
                   session.controller_input.y, session.controller_input.z,
                   int(session.controller_input_valid), session.last_sample_s,
                   int(session.has_last_sample), int(session.stale));
            continue;
        }
        if (strcmp(line, "TEST_PLANS") == 0) {
            printf("5 %d %.17g %d %d %.17g %d %d %d %d %d %.17g %d\n",
                   int(session.schedule.loaded), session.schedule.initial_angle_rad,
                   int(session.schedule.count), int(session.reference.loaded),
                   session.reference.total_s, int(session.use_reference),
                   int(session.pending.loaded), int(session.pending_count),
                   int(session.pending_declared), int(session.plan_open),
                   session.pending_reference.total_s, int(session.reference_open));
            continue;
        }
        if (strcmp(line, "TEST_CONFIG") == 0) {
            printf("6 %.17g %.17g %.17g %.17g %.17g %.17g %.17g %d\n",
                   session.config.proportional_gain, session.config.derivative_gain,
                   session.config.max_torque_nm, session.config.acceleration_cap_rad_s2,
                   session.config.rotor_inertia_kg_m2, session.config.max_speed_rad_s,
                   session.stale_gap_s, session.cycle == &hostCycle ? 1 : 0);
            continue;
        }
        if (strcmp(line, "TEST_PENDING") == 0) {
            printf("3 %d %d\n", int(session.pending.count), session.pending.loaded ? 1 : 0);
            continue;
        }
        if (adcs::handleLine(&context, line, reply, adcs::kMaxReply) <= 0) {
            echoReply("E SHORT 4 -1 0");
        } else {
            echoReply(reply);
        }
        if (c == EOF) break;
    }
    return 0;
}
