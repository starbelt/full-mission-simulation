// Host entrypoint exercises the same parser/formatter/ADS math as Zephyr.
#include "protocol.h"
#include <cstdio>

int main() {
  char line[512];
  char reply[256];
  while (fgets(line, sizeof(line), stdin)) {
    if (!ads_reply(line, reply, sizeof(reply))) return 2;
    fputs(reply, stdout);
  }
  return 0;
}
