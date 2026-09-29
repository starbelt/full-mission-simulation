// PocketADCS Zephyr ESP32-S3 ADS application. USB-Serial/JTAG at 115200.
//
// S2: seq epoch_ms gyro[3] mag[3] sun[3] pos_gcrs_m[3]
// R:  seq angle_deg rate mag_norm sun_norm flags quaternion[4]
//
// The attitude determination happens HERE, on the flight computer. The host
// simulates the orbit and the sensor hardware and transmits what the vehicle
// legitimately knows about itself; it does not transmit the inertial
// reference vectors, because supplying those would make the host the ADS.
#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/devicetree.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "ads_math.h"
#include "sensor_source.h"
#include "reference_provider.h"

#define LINE_SIZE 192
K_MSGQ_DEFINE(rx_lines, LINE_SIZE, 64, 4);
static const struct device *const serial = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
static char rx_line[LINE_SIZE];
static size_t rx_len;
static bool rx_overflow;

static void send_line(const char *s) {
    while (*s) uart_poll_out(serial, (unsigned char)*s++);
}

// The estimator core. Takes a SensorSample and nothing else, so it is
// identical whether the sample came from the serial HIL stream or from real
// I2C/ADC drivers. Swapping SensorSource implementations must not require
// touching anything below this line.
static void estimate(const SensorSample &s, AdsFrameResult &result,
                     unsigned &flags) {
    adsCompute(s.gyro_rad_s, s.mag_ut, s.sun_unit, s.ref_sun_unit,
               s.ref_mag_unit, &result);
    flags = 1u;
    // Field-level validity, surfaced to the host instead of being folded
    // into a single pass/fail. A missing magnetometer should be visible as
    // a missing magnetometer, not as a quietly wrong attitude.
    if (!(s.flags & kSensorGyro)) flags |= 32u;
    if (!(s.flags & kSensorMag)) flags |= 64u;
    if (!(s.flags & kSensorSun)) flags |= 128u;
    // The references are a separate concern from the sensor bus. The HIL
    // stream supplies them; real drivers cannot, until an ephemeris exists.
    // Flagging their absence is what keeps that gap honest.
    if (!(s.flags & kSensorRefSun) || !(s.flags & kSensorRefMag)) flags |= 256u;
    if (result.sun_norm < 0.01f) flags |= 2u;
    if (!result.valid) flags |= 16u;
}

static void emit(int seq, const AdsFrameResult &r, unsigned flags) {
    char out[192];
    int n = snprintf(out, sizeof(out),
        "R %d %.6f %.6f %.6e %.4f %u %.6f %.6f %.6f %.6f\n",
        seq, r.angle_deg, r.rate, r.mag_norm, r.sun_norm, flags,
        r.q[0], r.q[1], r.q[2], r.q[3]);
    if (n > 0 && n < (int)sizeof(out)) send_line(out);
}

static void error_reply() {
    const AdsFrameResult empty = {0, 0, 0, 0, {1, 0, 0, 0}, 0};
    emit(-1, empty, 4);
}

static void uart_rx_isr(const struct device *dev, void *) {
    while (uart_irq_update(dev) && uart_irq_rx_ready(dev)) {
        unsigned char chunk[64];
        int count = uart_fifo_read(dev, chunk, sizeof(chunk));
        for (int i = 0; i < count; ++i) {
            char c = (char)chunk[i];
            if (c == '\r') continue;
            if (c == '\n') {
                if (rx_overflow) {
                    static const char invalid[LINE_SIZE] = "!";
                    (void)k_msgq_put(&rx_lines, invalid, K_NO_WAIT);
                } else if (rx_len) {
                    rx_line[rx_len] = '\0';
                    (void)k_msgq_put(&rx_lines, rx_line, K_NO_WAIT);
                }
                rx_len = 0;
                rx_overflow = false;
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

    // The onboard ephemeris. This is what turns position + clock into the
    // inertial reference vectors, which is the part that must not be done on
    // the host.
    EphemerisReferenceProvider ephemeris;

    send_line("ADS Zephyr ADS firmware up (onboard ephemeris)\n");

    int64_t last_rx = k_uptime_get();
    char line[LINE_SIZE];
    for (;;) {
        if (k_msgq_get(&rx_lines, line, K_MSEC(100)) == 0) {
            last_rx = k_uptime_get();
            SensorSample sample;
            if (!source.parse(line, sample)) {
                error_reply();
                continue;
            }
            // Derive the reference vectors onboard, from the vehicle's own
            // position and time. The flags are only set if this succeeds.
            ephemeris.references(sample.epoch_ms, sample.pos_gcrs_m, sample);

            AdsFrameResult result;
            unsigned flags;
            estimate(sample, result, flags);
            emit((int)sample.seq, result, flags);
        }

        if (k_uptime_get() - last_rx >= 3000) {
            send_line("R -2 0 0 0 0 8 1 0 0 0\n");
            last_rx = k_uptime_get();
        }
    }
    return 0;
}
