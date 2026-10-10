# ESP32 v5 hardware verification — REVIEW REQUIRED

Full acceptance withheld: an additional wheel-state diagnostic temporarily changed device XC speed cap to1rad/s outside parent comment103 scope. All main physical runs retain exact approved fixtures; paced/lifecycle runs finished before the cap diagnostic and were not changed by it; normal configuration was restored without reset. The extra diagnostic is NOT approved acceptance evidence. Independent review/operator disposition required.

I 1 a1218da36d2248d7b439bdb91e0873726001a62a2ca532b31b5ec1144536780c 240000000 320 64

Exact retained image hashes independently read back;22/22 approved post-review pins unchanged. Every primary sample is uniquely joined TX/Y/sample/timing; device estimate equals controller input and only actual device wheel commands drive accepted plant. No resets between missions; explicit fresh XZ, live uptime advancement, normal drops/overflows zero.

|Case|Frames|Simulated s|Wall s|Physical final error deg|Service p50/p95/p99/max ms|Min stack bytes|
|---|---:|---:|---:|---:|---|---:|
|target3d|6001|300.0|175.377|0.007558113|13.907858/13.930483/13.937433/14.368983|5084|
|slewos|53641|2681.974849858644|1539.009|0.028461673|13.827833/13.904971/13.917631/13.944433|5084|
|eclipse|6001|300.0|171.474|0.007558121|13.891208/13.926508/13.934096/13.952333|5084|
|biased_noisy|6001|300.0|171.555|1.010198111|13.890071/13.926421/13.934583/13.955008|5084|

Biased/noisy case is diagnostic, NOT a nominal <0.1deg result. Baseline comparison is retained in audit.json. Four individual stages and QPC host exchange RTT percentiles are in metrics.csv; stage sum is NOT full service. Actual service maximum is below50ms20Hz compute budget.

Separate601-frame/30s20Hz wall-paced auxiliary loop TX span 30.000544s. Auxiliary30s profile is explicitly separate from full primary fixtures. No primary trajectory shortened.

Lifecycle tests:
- idle,reopen,live XC and profile: PASS
- stale-gap,duplicate,ordinary recovery: PASS
- legacy stateless S2 compatibility with live state: PASS
- overlong recoverable refusal: PASS
- bounded burst overload: PASS;stimuli192, replies67, drop delta125, overflow delta0
- explicit XZ fresh-session reset: PASS

Discriminating wheel-state replay diagnostic: XC preserved measured device bookkeeping at nonzero momentum; configured1rad/s speed cap suppressed all3 outward commands. Explicit XZ preserved uptime/counters but cleared wheel state: identical body torque then returned3 nonzero wheel commands. This used retained legitimate raw sensor frames, not host truth/reference injection or a physical-pointing claim. Normal accepted configuration restored without reset; final exact XI and health in final_status.json.

Original summary/log/raw retained: legacy valid flags1 was incorrectly asserted0. Resume used same MCU live state, no hardware reset. Flags1 checked against pinned source and actual reply.
Original failing auxiliary summary/raw/log remain retained; no failure was removed and no device reset was used to continue.

-10 individually retained malformed CR/NUL/XZ-prefixed stimuli refused; each followed by old-time refusal and ordinary sample recovery.
-Overload counters never cleared; overload kept outside normal pointing evidence.

- Synthetic sensors/host accepted physical stand-in, not flight or real physical sensor qualification
- Primary fixtures run unpaced lockstep;20Hz wall pacing demonstrated separately for30s
- CCOUNT measures elapsed complete dispatch with preemption, not pure CPU; excludes ISR/queue wait/suffix/TX
- Health reports minimum observed main-thread stack headroom, not worst case across all flight workloads

Independent review required. Reproduce offline audit: PYTHONPATH=src .venv/Scripts/python.exe results/esp32_adcs/v2/verify.py
