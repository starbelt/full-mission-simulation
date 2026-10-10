// Probe the exact shared firmware Sun ephemeris, without Zephyr dependencies.
#include <stdio.h>
#include <stdlib.h>
#include "reference_provider.h"

int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        const int64_t epoch = (int64_t)strtoll(argv[i], nullptr, 10);
        float sun[3];
        sunDirectionGcrs(epoch, sun);
        printf("%.12f %.10f %.10f %.10f\n", days_since_j2000(epoch),
               (double)sun[0], (double)sun[1], (double)sun[2]);
    }
    return 0;
}
