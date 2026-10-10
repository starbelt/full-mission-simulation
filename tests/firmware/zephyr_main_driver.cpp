// Host tests execute the actual target ISR, health sampler and main dispatch.
// Only Zephyr/hardware facilities are stubbed, never the ADCS or ephemeris.
#include "zephyr_stub.h"
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#define main firmware_main
#include "../../esp32/zephyr/src/main.cpp"
#undef main

// Snapshot every byte of mission state around malformed dispatch, not merely
// selected diagnostics. Static test storage avoids copying Session on stack.
static unsigned char before_invalid[sizeof(adcs::Session)];
static bool checking_invalid;
static void check_invalid_transaction() {
    if (checking_invalid) {
        sent += memcmp(before_invalid, &session, sizeof(session)) == 0
                    ? "TRANSACTION 1\n" : "TRANSACTION 0\n";
    }
    // isr-script regressions place one malformed command before the final
    // ordinary sample. Snapshot its dispatch even if broken framing merges it
    // into an apparently valid command; exclude legitimate setup commands.
    checking_invalid = queued.size() == 2 && queued.front().find("XR ") != 0;
    if (checking_invalid) {
        memcpy(before_invalid, &session, sizeof(session));
        sent += (session.wheel_speed[0] != 0 || session.wheel_speed[1] != 0 ||
                 session.wheel_speed[2] != 0) ? "MOMENTUM 1\n" : "MOMENTUM 0\n";
    }
}

static void begin_timing_sample() {
    if (!queued.empty() && queued.front().find("XS2 ") == 0) {
        timing_probe = true;
        counter_value = UINT32_MAX - 50200;
        events.clear(); readings.clear();
    }
}

static void stream_receive() {
    while (queued.empty()) {
        int c;
        while ((c = getchar()) != EOF) { incoming += char(c); if (c == '\n') break; }
        if (incoming.empty()) throw Stop();
        uart_rx_isr(serial, nullptr);
    }
}

int main(int argc, char **argv) {
#ifdef _WIN32
    // Text-mode stdin treats Ctrl-Z as EOF and can truncate into bare XZ.
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    if (argc != 2) return 2;
    std::string mode(argv[1]);
    if (mode == "stdio") {
        native_stream = true;
        before_get = stream_receive;
        try { firmware_main(); } catch (const Stop &) {}
    } else if (mode == "rx") {
        incoming = std::string(319, 'a') + "\n" + std::string(320, 'b') + "\nXH\n";
        uart_rx_isr(serial, nullptr);
        printf("%u %u %u\n", unsigned(queued.size()), unsigned(atomic_get(&rx_drops)),
               unsigned(atomic_get(&rx_line_overflows)));
        for (const auto &s : queued) printf("%u %c\n", unsigned(s.size()), s[0]);
    } else if (mode == "queue") {
        queue_fail = true;
        incoming = "XH\n" + std::string(320, 'b') + "\n";
        uart_rx_isr(serial, nullptr);
        printf("%u %u\n", unsigned(atomic_get(&rx_drops)),
               unsigned(atomic_get(&rx_line_overflows)));
    } else if (mode == "health") {
        stack_unused = 1234;
        printf("%d ", int(main_stack_headroom()));
        stack_error = -5;
        printf("%d\n", int(main_stack_headroom()));
    } else if (mode == "isr-script" || mode == "isr-reset") {
        int c;
        while ((c = getchar()) != EOF) incoming += char(c);
        uart_rx_isr(serial, nullptr);
        if (mode == "isr-script") before_get = check_invalid_transaction;
        try { firmware_main(); } catch (const Stop &) {}
        printf("%s", sent.c_str());
    } else if (mode == "script" || mode == "service" || mode == "service-fit" ||
               mode == "service-short" || mode == "service-format-overflow") {
        if (mode != "script") before_get = begin_timing_sample;
        if (mode == "service-fit") forced_y_length = 500;
        if (mode == "service-short") forced_y_length = 511;
        if (mode == "service-format-overflow") forced_y_length = 512;
        char line[adcs::kMaxLine];
        while (fgets(line, sizeof(line), stdin)) {
            line[strcspn(line, "\r\n")] = '\0';
            queued.push_back(line);
        }
        try { firmware_main(); } catch (const Stop &) {}
        printf("%s", sent.c_str());
        if (mode != "script") {
            printf("EVENTS %s\nREADINGS", events.c_str());
            for (uint32_t value : readings) printf(" %u", unsigned(value));
            printf("\n");
        }
    } else if (mode == "append-bounds") {
        // Canary and transactional failure tests on the actual target helper.
        const uint32_t values[] = {0u, 1u, UINT32_MAX};
        unsigned checks = 0;
        for (uint32_t value : values) {
            char suffix[12];
            const int n = snprintf(suffix, sizeof(suffix), " %u", unsigned(value));
            for (size_t capacity = 0; capacity <= 20; ++capacity) {
                char buffer[24], original[24];
                memset(buffer, '#', sizeof(buffer)); memcpy(buffer + 2, "Y ", 3);
                memcpy(original, buffer, sizeof(buffer));
                const bool fit = capacity > size_t(2 + n);
                const bool ok = append_service_cycles(buffer + 2, capacity, 2, value);
                if (ok != fit || buffer[0] != '#' || buffer[1] != '#') return 4;
                if (!ok && memcmp(buffer, original, sizeof(buffer))) return 5;
                if (ok && (strcmp(buffer + 4, suffix) || buffer[5 + n] != '#')) return 6;
                ++checks;
            }
        }
        for (int32_t length : {-1, 0, 20, INT32_MAX}) {
            char buffer[24], original[24];
            memset(buffer, '#', sizeof(buffer)); memcpy(buffer, "Y ", 3);
            memcpy(original, buffer, sizeof(buffer));
            if (append_service_cycles(buffer, 20, length, UINT32_MAX) ||
                memcmp(buffer, original, sizeof(buffer))) return 7;
            ++checks;
        }
        char missing_nul[8] = {'Y', ' ', 'x', '#', '#', '#', '#', '#'};
        if (append_service_cycles(missing_nul, sizeof(missing_nul), 2, 0)) return 8;
        printf("APPEND %u\n", checks + 1);
    } else if (mode == "dispatch") {
        stack_unused = 4321;
        atomic_set(&rx_drops, 7);
        atomic_set(&rx_line_overflows, 3);
        queued.push_back("XH");
        queued.push_back("XD 0 0");  // cannot erase measured counters
        queued.push_back("XH");
        queued.push_back("XI");
        queued.push_back("S2 1 1735689600000 0 0 0 0.00002 0.00001 0.00003 1 0 0 6878137 0 0");
        try { firmware_main(); } catch (const Stop &) {}
        printf("%s", sent.c_str());
    } else return 3;
    return 0;
}
