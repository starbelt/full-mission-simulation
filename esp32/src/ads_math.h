// PocketADCS ADS firmware: pure attitude math + per-frame compute path,
// shared verbatim by
//   * the ESP32-S3 HIL firmware (esp32/src/main.cpp), and
//   * the host-compiled regression test (tests/firmware/test_ads_math.cpp).
// This header must stay Arduino/FreeRTOS-free so both targets compile it
// unmodified.

#pragma once

#include <math.h>

static float vecNorm(const float *v) {
  return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static float angleBetween(const float *a, const float *b) {
  float na = vecNorm(a), nb = vecNorm(b);
  if (na == 0.0f || nb == 0.0f) return -1.0f;
  float dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
  if (dot > 1.0f) dot = 1.0f;
  if (dot < -1.0f) dot = -1.0f;
  return acosf(dot) * 57.295779513082325f; // rad -> deg
}

static void normalize(float *v) {
  float n = vecNorm(v);
  if (n > 0.0f) {
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
  }
}

static void cross(const float *a, const float *b, float *out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

// TRIAD DCM (reference -> body), row-major A[i][j]. Ported exactly from
// ahrs.filters.triad.TRIAD.estimate (Black 1964 / Lerner 1978 eqs 12-39..12-45):
// Mb = [w1|s2|s3] and Mr = [v1|r2|r3] are COLUMN matrices; A = Mb @ Mr^T (12-45).
//
// Does NOT modify its input vectors: it normalizes local copies. The original
// version normalized its four arguments in place, so the caller's
// vecNorm(mag) afterwards reported mag_norm = 1.0 instead of the original
// ~1e-5 T magnetometer magnitude on every frame (confirmed in captured HIL
// replies). This is the regression the host test guards.
//
// Returns false on degenerate input (parallel or zero vectors).
static bool triadDcm(const float v1[3], const float v2[3], const float w1[3],
                     const float w2[3], float A[3][3]) {
  float nv1[3] = {v1[0], v1[1], v1[2]};
  float nv2[3] = {v2[0], v2[1], v2[2]};
  float nw1[3] = {w1[0], w1[1], w1[2]};
  float nw2[3] = {w2[0], w2[1], w2[2]};
  normalize(nv1);
  normalize(nv2);
  normalize(nw1);
  normalize(nw2);
  float r2[3], r3[3], s2[3], s3[3];
  cross(nv1, nv2, r2);
  if (vecNorm(r2) < 1e-9f) return false;  // reference vectors parallel
  normalize(r2);
  cross(nv1, r2, r3);
  normalize(r3);  // eq. 12-41/12-42 triad
  cross(nw1, nw2, s2);
  if (vecNorm(s2) < 1e-9f) return false;  // body vectors parallel
  normalize(s2);
  cross(nw1, s2, s3);
  normalize(s3);  // eq. 12-39b/c
  const float *body[3] = {nw1, s2, s3};  // Mb columns
  const float *ref[3] = {nv1, r2, r3};   // Mr columns
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      float acc = 0.0f;
      for (int k = 0; k < 3; k++) acc += body[k][i] * ref[k][j];
      A[i][j] = acc;
    }
  return true;
}

static float signf(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }

// sqrt() guarded against float32 round-off making a Chiaverini argument
// slightly negative (identity DCM: 1-1-1+1 computed in float).
static float sqrtNonneg(float x) { return x > 0.0f ? sqrtf(x) : 0.0f; }

// Half-turn fallback for dcmToQuaternion. For an exact 180 deg rotation the
// scalar part is 0 and the skew terms vanish, so the trace branch below
// yields the all-zero quaternion. A half-turn is A = 2*u*u^T - I for a unit
// axis u: A[j][k] = 2*u_j*u_k and A[k][k] + 1 = 2*u_k^2. Solve on the column
// with the largest diagonal term (best conditioned) and leave q[0] = 0.
static void dcmToQuaternionHalfturn(const float A[3][3], float q[4]) {
  int k = 0;
  float best = A[0][0] + 1.0f;
  for (int i = 1; i < 3; i++) {
    float v = A[i][i] + 1.0f;
    if (v > best) {
      best = v;
      k = i;
    }
  }
  float axis[3] = {0.0f, 0.0f, 0.0f};
  axis[k] = signf(best) * sqrtNonneg(best * 0.5f);
  for (int j = 0; j < 3; j++) {
    if (j != k && fabsf(axis[k]) > 1e-6f) axis[j] = A[j][k] / (2.0f * axis[k]);
  }
  q[0] = 0.0f;
  q[1] = axis[0];
  q[2] = axis[1];
  q[3] = axis[2];
}

// Chiaverini quaternion from DCM (ahrs common.orientation.chiaverini).
static void dcmToQuaternion(const float A[3][3], float q[4]) {
  q[0] = 0.5f * sqrtNonneg(A[0][0] + A[1][1] + A[2][2] + 1.0f);
  q[1] = 0.5f * signf(A[2][1] - A[1][2]) *
         sqrtNonneg(A[0][0] - A[1][1] - A[2][2] + 1.0f);
  q[2] = 0.5f * signf(A[0][2] - A[2][0]) *
         sqrtNonneg(A[1][1] - A[2][2] - A[0][0] + 1.0f);
  q[3] = 0.5f * signf(A[1][0] - A[0][1]) *
         sqrtNonneg(A[2][2] - A[0][0] - A[1][1] + 1.0f);
  float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n <= 1e-6f) {  // half turn: no scalar part to recover from the trace
    dcmToQuaternionHalfturn(A, q);
    n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  }
  if (n > 0.0f)
    for (int i = 0; i < 4; i++) q[i] /= n;
}

// Per-frame ADS result: the exact fields the firmware emits in one
// "R" reply line.
struct AdsFrameResult {
  float angle_deg;  // angleBetween(mag, sun)
  float rate;       // vecNorm(gyro)
  float mag_norm;   // vecNorm(mag)  -- must be the ORIGINAL magnitude
  float sun_norm;   // vecNorm(sun)  -- cached before the compute path
  float q[4];       // quaternion (identity if degenerate)
  int valid;        // 1 = TRIAD succeeded, 0 = degenerate
};

// Extracted per-frame compute path -- exactly what the firmware's adsTask
// computes and emits. `mag`/`sun` are mutable (they are the firmware's
// AdsInput arrays) but this function must NOT rely on or corrupt them:
// after it returns, vecNorm(mag) must still equal the original magnitude.
// sun_norm is cached before the TRIAD step, matching the firmware, so a
// future regression that mutates sun cannot silently change the reported
// norm.
// ``mag`` and ``sun`` are declared non-const purely as a historical artifact
// of the in-place-normalization regression documented above; the body has
// not written to them since. They are const here so callers with read-only
// samples do not need a const_cast, and so the compiler enforces it.
static void adsCompute(const float gyro[3], const float mag[3],
                       const float sun[3], const float refSun[3],
                       const float refMag[3], AdsFrameResult *out) {
  float sunNorm = vecNorm(sun);  // cached BEFORE the compute path
  float q[4] = {1, 0, 0, 0};
  float A[3][3];
  int valid = 1;
  if (!triadDcm(refSun, refMag, sun, mag, A)) {
    valid = 0;  // degenerate TRIAD: keep identity quaternion
  } else {
    dcmToQuaternion(A, q);
  }
  out->angle_deg = angleBetween(mag, sun);
  out->rate = vecNorm(gyro);
  out->mag_norm = vecNorm(mag);  // must reflect the original magnitude
  out->sun_norm = sunNorm;
  for (int i = 0; i < 4; i++) out->q[i] = q[i];
  out->valid = valid;
}
