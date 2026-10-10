// PocketADCS Zephyr ESP32-S3 ADS application. USB-Serial/JTAG at 115200.
//
// Two protocols share the port. Both are answered; neither is repurposed.
//
//   Legacy, stateless (unchanged):
//     S2: seq epoch_ms gyro[3] mag[3] sun[3] pos_gcrs_m[3]
//     R:  seq angle_deg rate mag_norm sun_norm flags quaternion[4] cycles[2]
//     The attitude determination happens HERE, on the flight computer. The host
//     simulates the orbit and the sensor hardware and transmits what the
//     vehicle legitimately knows about itself; it does not transmit the
//     inertial reference vectors, because supplying those would make the host
//     the ADS.
//
//   Closed loop, stateful, version 1 (the "X" family):
//     XC  kp kd max_torque accel_cap inertia max_speed     configure
//     XL / XS / XE                                          load a plan (or a
//                                                          supplied 3D profile)
//     XS2 seq epoch_ms t_s gyro[3] mag[3] sun[3] pos[3]    one control frame
//     Y   status seq has_fix q[4] ctrl_q[4] tau[3] u[3]
//         saturated stale reject cyc[4]
//     XD / XO / XGS / XH                                    diagnostics
//     XZ                                                    clear the plan
//     See esp32/src/adcs_protocol.h and docs/esp32_closed_loop.md.
//
// WHY A SEPARATE PROTOCOL: the stateless S2/R exchange already passes a
// firmware regression suite and an operator soak on COM5. Reusing its tag for
// a stateful session would make those results describe behaviour their names do
// not imply, which is how an invalidated rig stays in the lab looking
// calibrated.
//
// WHAT THE HOST MAY NOT SEND: truth attitude, truth rate, or precomputed
// reference vectors. They have no field to travel in. Commands carry gains,
// limits and a schedule -- a plan the vehicle was told to fly, not the state it
// is in.
#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/devicetree.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/atomic.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "ads_estimate.h"
#include "sensor_source.h"
#include "reference_provider.h"
#include "adcs_core.h"
#include "adcs_protocol.h"
#include "command_framing.h"

#define LINE_SIZE adcs::kMaxLine
// The closed-loop reply carries four quaternions, three torques and four cycle
// counts, so it needs more room than a 192-byte sensor line.
#define REPLY_SIZE 512
K_MSGQ_DEFINE(rx_lines, LINE_SIZE, 64, 4);
static const struct device *const serial = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
static char rx_line[LINE_SIZE];
static size_t rx_len;
static bool rx_overflow;
static bool rx_invalid;
// A line the RX path could not store is a DROPPED line, not a discarded one:
// counted so a run that quietly lost frames cannot report a clean verdict.
static atomic_t rx_drops;
static atomic_t rx_line_overflows;
// Committed and pending 64-segment stores belong in BSS, not the thread stack.
static adcs::Session session;

// Append only after the complete dispatch/old Y formatting span has ended.
// snprintf returns a would-be length: never use it as an unchecked offset.
// Format into a separate bounded suffix, then copy atomically after all checks.
static bool append_service_cycles(char *reply, size_t capacity, int32_t written,
                                  uint32_t cycles) {
    if (written <= 0 || static_cast<size_t>(written) >= capacity) return false;
    const size_t used = static_cast<size_t>(written);
    if (reply[used] != '\0') return false;
    char suffix[12];  // space + UINT32_MAX decimal + NUL
    const int n = snprintf(suffix, sizeof(suffix), " %u", static_cast<unsigned>(cycles));
    if (n <= 0 || static_cast<size_t>(n) >= sizeof(suffix) ||
        static_cast<size_t>(n) >= capacity - used) return false;
    memcpy(reply + used, suffix, static_cast<size_t>(n) + 1);
    return true;
}

static void send_line(const char *s) {
    while (*s) uart_poll_out(serial, (unsigned char)*s++);
}

// Cycle counts from the last frame, so the host can compute real on-device
// execution times. -1 when no frame has been processed yet (the idle
// heartbeat, which is not a measurement).
static int32_t g_cyc_eph = -1;
static int32_t g_cyc_ads = -1;

// Cycle counts for the closed-loop stages. This is the Xtensa hardware cycle
// counter (CCOUNT) through Zephyr's k_cycle_get_32, not a CPU-frequency
// setting: claiming a clock that was never verified is how a timing claim goes
// wrong by a factor of the core clock.
static uint32_t host_cycles() { return k_cycle_get_32(); }

// The onboard ephemeris wrapped as the protocol's reference seam. This is what
// turns position + clock into the inertial reference vectors, which is the part
// that must not be done on the host.
static EphemerisReferenceProvider g_ephemeris;

static bool onboard_references(const adcs::Session &, int64_t epoch_ms, const double pos[3],
                               adcs::AbsoluteUpdate *out) {
    // EphemerisReferenceProvider::references writes the reference vectors into
    // a SensorSample and ORs its own flags in, so the flags -- not the vector
    // contents -- are the authority on whether a reference was produced. A
    // failed model leaves the vectors undefined, and a TRIAD fix built on them
    // would be a confident wrong answer.
    SensorSample derived = {};
    derived.epoch_ms = epoch_ms;
    for (int i = 0; i < 3; ++i) derived.pos_gcrs_m[i] = pos[i];
    derived.flags = 0;
    if (!g_ephemeris.references(epoch_ms, derived.pos_gcrs_m, derived)) return false;
    for (int i = 0; i < 3; ++i) {
        out->sun_ref[i] = derived.ref_sun_unit[i];
        out->magnetic_ref[i] = derived.ref_mag_unit[i];
    }
    return true;
}

static void emit(int seq, const AdsFrameResult &r, unsigned flags,
                uint32_t eph_cycles, uint32_t ads_cycles) {
    char out[224];
    g_cyc_eph = (int32_t)eph_cycles;
    g_cyc_ads = (int32_t)ads_cycles;
    int n = snprintf(out, sizeof(out),
        "R %d %.6f %.6f %.6e %.4f %u %.6f %.6f %.6f %.6f %d %d\n",
        seq, r.angle_deg, r.rate, r.mag_norm, r.sun_norm, flags,
        r.q[0], r.q[1], r.q[2], r.q[3], g_cyc_eph, g_cyc_ads);
    if (n > 0 && n < (int)sizeof(out)) send_line(out);
}

static void error_reply() {
    const AdsFrameResult empty = {0, 0, 0, 0, {1, 0, 0, 0}, 0};
    g_cyc_eph = -1;
    g_cyc_ads = -1;
    emit(-1, empty, 4, 0, 0);
}

// Stack headroom of the main thread, so stack exhaustion is observable rather
// than inferred from a crash. Requires CONFIG_INIT_STACKS and
// CONFIG_THREAD_STACK_INFO, both of which prj.conf keeps enabled.
static int32_t main_stack_headroom(void) {
#if defined(CONFIG_INIT_STACKS) && defined(CONFIG_THREAD_STACK_INFO)
    size_t unused = 0;
    if (k_thread_stack_space_get(k_current_get(), &unused) != 0) return -1;
    return (int32_t)unused;
#else
    return -1;
#endif
}

static void uart_rx_isr(const struct device *dev, void *) {
    // Short by construction: bytes in, bytes out, no floating math, no control
    // law, no blocking. All of that happens on the main thread.
    while (uart_irq_update(dev) && uart_irq_rx_ready(dev)) {
        unsigned char chunk[64];
        int count = uart_fifo_read(dev, chunk, sizeof(chunk));
        for (int i = 0; i < count; ++i) {
            char c = adcs::commandTextByte(chunk[i]);
            if (c == '\n') {
                // CRLF/trailing CR must not change exact diagnostics (XI).
                // Trim only the tail; internal CR separators remain spaces.
                while (rx_len && (rx_line[rx_len - 1] == ' ' ||
                                  rx_line[rx_len - 1] == '\t')) --rx_len;
                if (rx_overflow || rx_invalid) {
                    // Consume/refuse the whole malformed line once. Never let
                    // binary payload truncate an XZ into a destructive reset.
                    static const char invalid[LINE_SIZE] = "!";
                    if (k_msgq_put(&rx_lines, invalid, K_NO_WAIT) != 0) atomic_inc(&rx_drops);
                    if (rx_overflow) atomic_inc(&rx_line_overflows);
                } else if (rx_len) {
                    rx_line[rx_len] = '\0';
                    // Queue full is DETECTED, not silently discarded: a line that
                    // vanishes here would show up as a missing seq on the host
                    // and read as a firmware that skipped a measurement.
                    if (k_msgq_put(&rx_lines, rx_line, K_NO_WAIT) != 0) atomic_inc(&rx_drops);
                }
                rx_len = 0;
                rx_overflow = false;
                rx_invalid = false;
            } else if (!adcs::commandByteValid(chunk[i])) {
                rx_invalid = true;
            } else if (rx_invalid) {
                // Drain to LF without storing or dispatching any suffix.
                continue;
            } else if (rx_len < sizeof(rx_line) - 1 && !rx_overflow) {
                rx_line[rx_len++] = c;
            } else {
                rx_overflow = true;
            }
        }
    }
}

int main() {
    if (!device_is_ready(serial)) return -1;
    if (uart_irq_callback_user_data_set(serial, uart_rx_isr, nullptr) != 0) return -2;
    uart_irq_rx_enable(serial);

    // The one line that chooses where readings come from. Swapping in real
    // I2C/ADC drivers is a change here and nowhere else.
    HilStreamSource source;

    session.reset(&host_cycles, main_stack_headroom());
    adcs::CallContext context;
    context.session = &session;
    context.references = &onboard_references;

    send_line("ADS Zephyr ADS firmware up (onboard ephemeris, closed-loop v1)\n");

    int64_t last_rx = k_uptime_get();
    char line[LINE_SIZE];
    char reply[REPLY_SIZE];
    for (;;) {
        if (k_msgq_get(&rx_lines, line, K_MSEC(100)) == 0) {
            // Full queued-command service starts AFTER dequeue, before health
            // sampling, routing, parsing and all closed-loop work. ISR framing
            // and queue wait are excluded. Unsigned subtraction permits one
            // CCOUNT wrap; the interval must be shorter than 2^32 cycles.
            const uint32_t service_start = k_cycle_get_32();
            last_rx = k_uptime_get();
            session.stack_headroom_bytes = main_stack_headroom();
            session.rx_drops = (uint32_t)atomic_get(&rx_drops);
            session.rx_overflows = (uint32_t)atomic_get(&rx_line_overflows);
            session.uptime_ms = (uint32_t)last_rx;

            if (line[0] == 'S' && line[1] == '2') {
                // --- legacy stateless path, unchanged ---------------------
                SensorSample sample;
                if (!source.parse(line, sample)) {
                    error_reply();
                    continue;
                }
                // Reference-vector generation and TRIAD are timed separately
                // on core 0. Parsing and USB transport are outside these
                // intervals.
                uint32_t t0 = k_cycle_get_32();

                // Derive the reference vectors onboard, from the vehicle's own
                // position and time. The flags are only set if this succeeds.
                g_ephemeris.references(sample.epoch_ms, sample.pos_gcrs_m, sample);
                uint32_t t1 = k_cycle_get_32();

                AdsFrameResult result;
                unsigned flags;
                estimateSample(sample, result, flags);
                uint32_t t2 = k_cycle_get_32();

                emit((int)sample.seq, result, flags, t1 - t0, t2 - t1);
                continue;
            }

            // Machine-readable identity is explicit, not an unsolicited line
            // that could masquerade as the ACK to a control command.
            if (strcmp(line, "XI") == 0) {
                snprintf(reply, sizeof(reply), "I 1 %s %u %d %d\n", ADCS_BUILD_ID,
                         (unsigned)sys_clock_hw_cycles_per_sec(),
                         adcs::kMaxLine, adcs::kMaxSegments);
                send_line(reply);
                continue;
            }
            // Device health is measured here. Host diagnostic setters remain
            // available in the portable protocol tests but cannot forge the
            // hardware counters or boot clock on the target.
            if (strncmp(line, "XD ", 3) == 0 || strncmp(line, "XO ", 3) == 0 ||
                strncmp(line, "XGS ", 4) == 0) {
                adcs::formatRefusal(session, "HEALTH", adcs::kErrBadArgument,
                                    -1, reply, sizeof(reply));
                send_line(reply);
                send_line("\n");
                continue;
            }

            // --- closed-loop stateful path -------------------------------
            const int32_t written = adcs::handleLine(&context, line, reply, REPLY_SIZE);
            // Includes original reply formatting; excludes metric append,
            // newline formatting, TX and subsequent idle/heartbeat work.
            const uint32_t service_cycles = k_cycle_get_32() - service_start;
            const bool is_frame = written > 0 && reply[0] == 'Y' && reply[1] == ' ';
            if (written > 0 && (!is_frame ||
                append_service_cycles(reply, sizeof(reply), written, service_cycles))) {
                send_line(reply);
                send_line("\n");
            } else {
                // A reply that would not fit is dropped rather than truncated:
                // a partial line parses as a different, valid-looking frame.
                send_line("E SHORT 4 -1 0\n");
            }
        }

        if (k_uptime_get() - last_rx >= 3000) {
            // Same 12-field shape as a real reply so the host parser never has
            // to special-case the heartbeat; -1 marks "no measurement".
            send_line("R -2 0 0 0 0 8 1 0 0 0 -1 -1\n");
            last_rx = k_uptime_get();
        }
    }
    return 0;
}
