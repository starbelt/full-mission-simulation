// PocketADCS Attitude Determination System (ADS) firmware for the ESP32-S3 HIL rig.
//
// Architecture:
//   * The Arduino loop thread parses ASCII sensor frames off USB serial.
//   * Parsed frames are pushed onto a FreeRTOS queue.
//   * A dedicated ADS task, pinned to CORE 0 with xTaskCreatePinnedToCore
//     (the installed core, Arduino 3.20017 / ESP-IDF 4.4.7, has no
//     xTaskCreateAffinitySet / configUSE_CORE_AFFINITY; xTaskCreatePinnedToCore
//     with xCoreID=0 is the verified API in its FreeRTOS task.h), consumes the
//     queue and computes per-frame attitude: TRIAD DCM + quaternion + flags.
//
// TRIAD: Black 1964 / Lerner 1978 eqs. 12-39..12-45, and the Chiaverini
// DCM->quaternion conversion, both ported from the reference implementation
// ahrs.filters.triad (https://ahrs.readthedocs.io/en/latest/filters/triad.html).
// Do not re-derive the algebra here.
//
// Protocol (newline-terminated ASCII):
//   S <seq> <time_s> <gx> <gy> <gz> <mx> <my> <mz> <sx> <sy> <sz>
//     <rx> <ry> <rz> <hx> <hy> <hz>
//   gx..gz gyro; mx..mz body MAG; sx..sy sz body Sun;
//   rx..rz reference Sun unit vector, hx..hz reference MAG unit vector.
// Reply:
//   R <seq> <angle_deg> <rate_mag> <mag_norm> <sun_norm> <flags> <qw qx qy qz>
// flags bit0=valid frame, bit1=sun zero (eclipse), bit2=parse error,
//      bit3=idle tick, bit4=degenerate TRIAD (parallel / zero vectors).
// Also: "P" prints stats, "T" emits one synthetic identity frame,
// "C" asks the ADS task to report its core id again.

#include <Arduino.h>
#include <cmath>
#include <cstdio>

#include "ads_math.h"  // pure ADS math + per-frame compute path (host-tested)

static uint32_t rxLines = 0, rxGood = 0, rxBad = 0, idleTicks = 0, processed = 0;
static uint32_t lastRxMs = 0;

enum AdsKind : uint8_t {
  ADS_FRAME = 0,
  ADS_CORE_QUERY = 1,
};

struct AdsInput {
  uint8_t kind;
  int seq;
  float gyro[3];
  float mag[3];
  float sun[3];
  float refSun[3];
  float refMag[3];
};

static const UBaseType_t ADS_QUEUE_LEN = 24;
static QueueHandle_t adsQueue;
static TaskHandle_t adsHandle = nullptr;

// The pure ADS math (vecNorm, angleBetween, normalize, cross, triadDcm,
// dcmToQuaternion) and the per-frame compute path (adsCompute) live in
// ads_math.h, shared verbatim with the host-compiled regression test.

static void emit(const int seq, const float angle, const float rate,
                 const float magNorm, const float sunNorm, uint8_t flags,
                 const float *q) {
  Serial.printf("R %d %.6f %.6f %.6e %.4f %u %.6f %.6f %.6f %.6f\n",
                seq, angle, rate, magNorm, sunNorm, flags, q[0], q[1], q[2], q[3]);
}

// ADS task: pinned to core 0 at creation; reports its core id at boot and on
// every 'C' command so the harness can confirm it really runs on core 0.
static void adsTask(void *arg) {
  Serial.printf("ADS task online on core %d\n", (int)xPortGetCoreID());
  for (;;) {
    AdsInput in;
    if (xQueueReceive(adsQueue, &in, portMAX_DELAY) != pdTRUE) continue;
    if (in.kind == ADS_CORE_QUERY) {
      Serial.printf("C core=%d processed=%u queued=%u stack=%u\n",
                    (int)xPortGetCoreID(), processed,
                    (unsigned)uxQueueMessagesWaiting(adsQueue),
                    (unsigned)uxTaskGetStackHighWaterMark(adsHandle));
      continue;
    }
    AdsFrameResult r;
    adsCompute(in.gyro, in.mag, in.sun, in.refSun, in.refMag, &r);
    uint8_t flags = 1 | (r.sun_norm < 0.01f ? 2 : 0) | (r.valid ? 0 : 16);
    processed++;
    emit(in.seq, r.angle_deg, r.rate, r.mag_norm, r.sun_norm, flags, r.q);
  }
}

static void handleLine(const char *line) {
  rxLines++;
  if (line[0] == 'P') {
    Serial.printf("P %u %u %u %u %u %u %u\n", rxLines, rxGood, rxBad, idleTicks,
                  processed, (unsigned)uxQueueMessagesWaiting(adsQueue),
                  (unsigned)uxTaskGetStackHighWaterMark(adsHandle));
    return;
  }
  if (line[0] == 'C') {
    AdsInput cmd = {};
    cmd.kind = ADS_CORE_QUERY;
    if (xQueueSend(adsQueue, &cmd, 0) != pdTRUE) {
      Serial.printf("C core=%d (queue full)\n", (int)xPortGetCoreID());
    }
    return;
  }
  if (line[0] == 'T') {
    // Identity case: reference vectors equal the body vectors.
    AdsInput in = {};
    in.kind = ADS_FRAME;
    in.seq = 0;
    float gyro[3] = {0.01f, -0.02f, 0.03f};
    float mag[3] = {1e-5f, 2e-5f, 3e-5f};
    float sun[3] = {0.6f, 0.8f, 0.0f};
    for (int i = 0; i < 3; i++) {
      in.gyro[i] = gyro[i];
      in.mag[i] = mag[i];
      in.sun[i] = sun[i];
      in.refSun[i] = sun[i];
      in.refMag[i] = mag[i];
    }
    if (xQueueSend(adsQueue, &in, 100) != pdTRUE) {
      Serial.printf("R 0 0 0 0 0 4 1 0 0 0\n");
    }
    return;
  }
  int seq = -1;
  float time = 0;
  AdsInput in = {};
  // Robust token parser: S <seq> <time> gx..gz mx..mz sx..sz rx..rz hx..hz.
  // newlib sscanf faulted on truncated lines, so parse tokens by hand.
  const char *p = line + 1;
  float vals[17];
  int n = 0;
  while (n < 17) {
    while (*p == ' ') p++;
    if (*p == '\0') break;
    char *end = nullptr;
    vals[n] = strtof(p, &end);
    if (end == p) break;  // not a number: malformed frame
    n++;
    p = end;
  }
  if (n != 17) {
    rxBad++;
    float q[4] = {0, 0, 0, 0};
    emit(-1, 0, 0, 0, 0, 4, q);
    return;
  }
  seq = (int)vals[0];
  time = vals[1];
  for (int i = 0; i < 3; i++) {
    in.gyro[i] = vals[2 + i];
    in.mag[i] = vals[5 + i];
    in.sun[i] = vals[8 + i];
    in.refSun[i] = vals[11 + i];
    in.refMag[i] = vals[14 + i];
  }
  rxGood++;
  in.kind = ADS_FRAME;
  in.seq = seq;
  if (xQueueSend(adsQueue, &in, 100) != pdTRUE) {
    float q[4] = {0, 0, 0, 0};
    emit(seq, 0, 0, 0, 0, 4, q);
  }
}

void setup() {
  Serial.setRxBufferSize(4096);
  Serial.begin(115200);
  adsQueue = xQueueCreate(ADS_QUEUE_LEN, sizeof(AdsInput));
  if (adsQueue == nullptr) {
    Serial.println("ADS queue create failed");
    return;
  }
  if (xTaskCreatePinnedToCore(adsTask, "ads", 4096, nullptr, 3, &adsHandle, 0) != pdTRUE) {
    Serial.println("ADS task create failed");
  }
  Serial.println("ADS HIL firmware up (S/R protocol + TRIAD, queue to core0 task)");
}

void loop() {
  static char buf[160];
  static int idx = 0;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      buf[idx < 159 ? idx : 159] = '\0';
      idx = 0;
      lastRxMs = millis();
      handleLine(buf);
    } else if (idx < 159) {
      buf[idx++] = c;
    }
  }
  uint32_t now = millis();
  if (lastRxMs == 0 || now - lastRxMs > 3000) {
    idleTicks++;
    Serial.printf("R -2 0 0 0 0 8 1 0 0 0\n");
    lastRxMs = now;
  }
}
