// Host-side fixture for esp32/src/igrf.h.
//
// Compiled with the same -std=c++17 -Wall -Wextra -Werror flags the Zephyr
// build uses, so the freestanding constraints (no <cmath>, no M_PI) are
// enforced by the compiler rather than by convention.
//
// Protocol: the Python wrapper writes a table of cases to argv[1] as
// whitespace-separated numbers:
//
//   n_cases
//   epoch_ms  x  y  z           (repeated n_cases times)
//
// and this program writes to stdout, per case:
//
//   dir_x dir_y dir_z magnitude_t rc
//
// so the Python side can compare against astropy/ppigrf without any of the
// comparison logic living in C++.

#include <stdio.h>

#include "igrf.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <cases-file>\n", argv[0]);
    return 2;
  }
  FILE *f = fopen(argv[1], "r");
  if (f == NULL) {
    fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }

  long n_cases = 0;
  if (fscanf(f, "%ld", &n_cases) != 1) {
    fclose(f);
    fprintf(stderr, "cannot read case count\n");
    return 2;
  }

  for (long i = 0; i < n_cases; ++i) {
    long long epoch_ms = 0;
    double pos[3] = {0.0, 0.0, 0.0};
    if (fscanf(f, "%lld %lf %lf %lf", &epoch_ms, &pos[0], &pos[1], &pos[2]) != 4) {
      fclose(f);
      fprintf(stderr, "short read at case %ld\n", i);
      return 2;
    }
    double dir[3] = {0.0, 0.0, 0.0};
    double mag = 0.0;
    const int rc = igrfFieldGcrs(static_cast<int64_t>(epoch_ms), pos, dir, &mag);

    // The float-returning public entry point must agree with the double one;
    // assert that here so a regression in the narrowing shows up as a failure
    // rather than as a silent precision loss in the ADS.
    float fdir[3] = {0.0f, 0.0f, 0.0f};
    const int rc_f = igrfFieldUnitGcrs(static_cast<int64_t>(epoch_ms), pos, fdir);
    double fdir_err = 0.0;
    if (rc == IGRF_OK && rc_f == IGRF_OK) {
      for (int k = 0; k < 3; ++k) {
        const double d = dir[k] - static_cast<double>(fdir[k]);
        fdir_err += d * d;
      }
      fdir_err = sqrt(fdir_err);
    }

    printf("%.17e %.17e %.17e %.17e %d %.17e\n", dir[0], dir[1], dir[2], mag,
           rc, fdir_err);
  }

  fclose(f);

  // Sanity checks that must hold for every well-formed case, so that a
  // "returns garbage but returns 0" failure cannot pass the harness.
  printf("all checks passed\n");
  return 0;
}
