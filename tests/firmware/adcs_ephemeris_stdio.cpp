// Native correctness surrogate: exact protocol/core and actual reference provider.
// No hardware counters exist here. Zero cycles mean unavailable, not free work.
#include "adcs_protocol.h"
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include "command_framing.h"
#include "reference_provider.h"

static uint32_t unavailableCycles() { return 0; }
static bool references(const adcs::Session &, int64_t epoch, const double pos[3],
                       adcs::AbsoluteUpdate *out) {
    SensorSample sample = {};
    EphemerisReferenceProvider provider;
    if (!provider.references(epoch, pos, sample)) return false;
    for (int i = 0; i < 3; ++i) {
        out->sun_ref[i] = sample.ref_sun_unit[i];
        out->magnetic_ref[i] = sample.ref_mag_unit[i];
    }
    return true;
}

int main() {
#ifdef _WIN32
    // Text-mode stdin treats Ctrl-Z as EOF and can truncate into bare XZ.
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    adcs::Session session;
    session.reset(unavailableCycles, 0);
    adcs::CallContext context = {&session, references};
    char line[adcs::kMaxLine], reply[adcs::kMaxReply];
    for (;;) {
        int c = getchar();
        if (c == EOF) break;
        size_t n = 0;
        bool invalid = false;
        while (c != EOF && c != '\n') {
            if (!adcs::commandByteValid(static_cast<unsigned char>(c))) invalid = true;
            if (!invalid) {
                if (n + 1 >= sizeof(line)) invalid = true;
                else line[n++] = adcs::commandTextByte(static_cast<unsigned char>(c));
            }
            c = getchar();
        }
        line[n] = '\0';
        if (invalid) {
            adcs::formatRefusal(session, "LINE", adcs::kErrBadArgument, -1,
                                reply, sizeof(reply));
        } else if (!adcs::handleLine(&context, line, reply, sizeof(reply))) {
            return 2;
        }
        puts(reply);
        fflush(stdout);
        if (c == EOF) break;
    }
    return ferror(stdin) ? 1 : 0;
}
