"""Accuracy and timing plots for the PocketADCS sensor-fidelity studies.

Reads the study JSON in results/sensor_fidelity/ and the timing JSON in
results/timing/, and writes both accuracy and execution-time figures.

Two rules that keep these figures honest, both learned the hard way here:

- Never plot a noise variant whose valid-sample count has collapsed. A high
  error with a collapsed valid count is a rejected-sample artifact, not a
  sensor result, so the valid fraction goes on the same figure as the error.
- The firmware's attitude error must use 2*acos(|dot|), not the plain arccos
  of the trace: the half-angle halves every error and manufactures false
  passes. The study code already does this; these plots only re-present it.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

INK = "#1a1a1a"
ACCENT = "#2f6f9f"
WARN = "#c1543a"
MUTED = "#8a8a8a"


def load(tag):
    p = REPO / "results" / "sensor_fidelity" / tag
    manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for f in sorted(p.glob("rate*_seed*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for run in d["runs"]:
            run["_seed"] = d["config"]["seed"]
            run["_rate"] = d["config"]["rate_hz"]
            rows.append(run)
    return manifest, rows


def collect(rows, variant):
    sel = [r for r in rows if r["variant_name"] == variant]
    if not sel:
        return None
    return {
        "rms": np.array([r["angle_rms_deg"] for r in sel]),
        "p95": np.array([r["angle_p95_deg"] for r in sel]),
        "max": np.array([r["angle_max_deg"] for r in sel]),
        "valid": np.array([r["valid_samples"] / max(1, r["samples"]) for r in sel]),
        "per_sample_us": np.array([r.get("per_sample_us", np.nan) for r in sel]),
        "n": len(sel),
    }


# ---------------------------------------------------------------- accuracy ---
def fig_ablation(rows, tag):
    """Where the error comes from: one bar per ablation variant, with the
    valid fraction beneath it so a collapsed-valid artifact cannot hide."""
    order = [r["variant_name"] for r in rows
             if r["_seed"] == rows[0]["_seed"]]
    order = list(dict.fromkeys(order))
    stats = [(v, collect(rows, v)) for v in order]
    stats = [(v, s) for v, s in stats if s]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    labels = [v.replace("_", "\n") for v, _ in stats]
    rms = [s["rms"].mean() for _, s in stats]
    spread = [s["rms"].std() for _, s in stats]
    p95 = [s["p95"].mean() for _, s in stats]
    valid = [s["valid"].mean() * 100 for _, s in stats]

    # Lollipops, not bars: a bar on a log axis has no meaningful zero, so the
    # bar length reads as an arbitrary distance above the axis floor. A stem
    # from an explicit floor is honest on a log scale.
    x = np.arange(len(stats))
    floor = 1e-2
    for xi, (m, sd, _hi) in enumerate(zip(rms, spread, p95, strict=True)):
        ax.vlines(xi, floor, m, color=ACCENT, lw=2.2, alpha=0.55)
        ax.vlines(xi, m - sd, m + sd, color=INK, lw=1.1)
        ax.plot([xi - 0.13, xi + 0.13], [m + sd] * 2, color=INK, lw=1.1)
        ax.plot(xi, m, "o", ms=7, color=ACCENT, zorder=3)
    ax.plot([], [], "o", ms=7, color=ACCENT, label="RMS attitude error")
    ax.plot([], [], "o", ms=4.5, color=MUTED, label="95th percentile")
    for xi, (m, hi) in enumerate(zip(rms, p95, strict=True)):
        ax.plot(xi, hi, "D", ms=4.5, color=MUTED, zorder=3)
        ax.annotate(f"{m:.3f}", (xi + 0.17, m), fontsize=7.5, color=INK,
                    va="center")
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.6, max(rms) * 4)
    ax.set_ylabel("attitude error (deg)")
    ax.set_title("Attitude error budget by ablation variant\n"
                 f"{len(stats[0][1]['rms'])} seeds, one 5400 s orbit, 10 Hz; "
                 "bars are means, whiskers +/-1 sd",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax2.bar(x, valid, color=[WARN if v < 50 else MUTED for v in valid],
            width=0.62)
    for xi, v in enumerate(valid):
        ax2.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points",
                     xytext=(0, 2), ha="center", fontsize=7, color=INK)
    ax2.axhline(50, color=WARN, lw=0.8, ls="--", alpha=0.7)
    ax2.set_ylabel("valid\nsamples %")
    ax2.set_ylim(0, 108)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("ablation variant (one noise source each; last row = all three)")

    for xi, (_v, s) in enumerate(stats):
        if s["valid"].mean() * 100 < 50:
            ax2.annotate("rejected-sample\nartifact", (xi, 8), ha="center",
                         fontsize=6.5, color=WARN, fontweight="bold")

    fig.tight_layout()
    p = OUT / f"accuracy_ablation_{tag}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_seed_spread(rows, tag):
    """Per-seed spread of the dominant variant: is the error stable?"""
    worst = max({r["variant_name"] for r in rows},
                key=lambda v: (collect(rows, v) or {"rms": np.zeros(1)})["rms"].mean())
    s = collect(rows, worst)
    if not s:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    seeds = [r["_seed"] for r in rows if r["variant_name"] == worst]
    ax.bar(seeds, s["rms"], color=ACCENT, width=0.6)
    ax.axhline(s["rms"].mean(), color=WARN, ls="--", lw=1,
               label=f"mean {s['rms'].mean():.3f} deg")
    for xi, v in enumerate(s["rms"]):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7.5)
    ax.set_xlabel("seed")
    ax.set_ylabel("RMS attitude error (deg)")
    ax.set_title(f"Seed-to-seed spread, {worst}\n"
                 f"sd {s['rms'].std():.4f} deg over {s['n']} seeds",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = OUT / f"accuracy_seedspread_{worst}_{tag}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ timing ---
def fig_timing():
    p = REPO / "results" / "timing" / "firmware_chain_desktop.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    fw = d.get("firmware_chain_desktop")
    if not fw:
        return []

    eph = fw["parse_plus_ephemeris_us"]
    ads = fw["ads_us"]
    tot = fw["total_us"]
    other = max(0.0, tot - eph - ads)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9.0, 3.6), gridspec_kw={"width_ratios": [1, 1.15]})

    # Panel 1: where the time goes.
    parts = [eph, ads, other]
    names = ["parse +\nephemeris", "TRIAD\n(adsCompute)", "other"]
    colors = [ACCENT, "#7aa6c2", MUTED]
    if other < tot * 0.005:
        parts = parts[:2]
        names, colors = names[:2], colors[:2]
    bars = ax.bar(names, parts, color=colors, width=0.55)
    for b, v in zip(bars, parts, strict=True):
        ax.annotate(f"{v:.2f} us\n({v / tot * 100:.1f}%)",
                    (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=7.5, color=INK)
    ax.set_ylabel("microseconds per sample")
    ax.set_ylim(0, max(parts) * 1.35)
    ax.set_title("Where the per-sample time goes\n"
                 "firmware C++ ALGORITHM on a desktop CPU, 80k samples\n"
                 "(NOT measured on the ESP32-S3 -- see caveat)",
                 fontsize=9, loc="left")

    # Panel 2: CPU load against sample rate.
    r_cross = 1e6 / fw["total_us"]
    rates = np.unique(np.concatenate([
        np.array([1, 5, 10, 20, 50, 100, 200, 500]),
        np.array([r_cross / 2, r_cross, r_cross * 2]),
    ]))
    load = tot * 1e-6 * rates * 100
    ax2.plot(rates, load, "-o", color=ACCENT, ms=4, lw=1.4)
    ax2.axhline(100, color=WARN, ls="--", lw=1,
                label="100% of one core")
    # Extrapolate the crossover honestly: only claim it if it falls in range.
    r_cross = 1e6 / tot  # Hz at which one core is exactly saturated
    r_cross = float(r_cross)
    if rates.max() < r_cross:
        ax2.plot([r_cross], [100.0], "*", ms=11, color=WARN, zorder=4)
        ax2.annotate(f"1 core saturates at {r_cross / 1000:.0f} kHz\n"
                     f"({tot * 1e3:.1f} us/sample)",
                     (r_cross, 100.0), textcoords="offset points",
                     xytext=(-6, -34), ha="right", fontsize=7.5, color=INK)
    else:
        ax2.axvline(r_cross, color=MUTED, lw=0.8, ls=":")
        ax2.annotate(f"saturates 1 core at {r_cross / 1000:.0f} kHz",
                     (r_cross, 100), textcoords="offset points", xytext=(-6, -30),
                     ha="right", fontsize=7.5, color=INK)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("sensor sample rate (Hz)")
    ax2.set_ylabel("CPU load (% of one core)")
    ax2.set_title("Algorithm cost vs sample rate\n"
                  f"all sensor ODRs are far below the {r_cross / 1000:.0f} kHz "
                  "saturation point",
                  fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")

    fig.text(0.5, -0.035,
             "CAVEAT: desktop CPU running the board's algorithm. The ESP32-S3 "
             "runs 240-480 MHz, so its per-sample cost will be higher.\n"
             "The 12.98 us is a lower bound on the board, not a measurement of "
             "it. The study's ~160 us/sample times a different implementation "
             "(Python TriadEstimator), not a wider region.",
             ha="center", va="top", fontsize=7, color=WARN, style="italic")
    fig.tight_layout()
    p = OUT / "timing_execution_cost.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return [p]


# ------------------------------------------------------- study vs firmware ---
def fig_firmware_vs_study():
    """The end-to-end claim in one figure: what the hardware chain costs in
    accuracy, against the pure-study variants."""
    p = REPO / "results" / "timing" / "firmware_vs_truth.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    fw = d["error_deg"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.hist(fw, bins=28, color=ACCENT, alpha=0.85,
            label=f"firmware, {len(fw)} frames")
    deg = d.get("degenerate", 0)
    tot = d.get("frames", len(fw) + deg)
    if deg:
        ax.annotate(f"{deg} of {tot} frames degenerate (TRIAD), excluded",
                    (0.98, 0.96), xycoords="axes fraction", ha="right",
                    fontsize=7, color=WARN)
    ax.axvline(np.median(fw), color=WARN, ls="--", lw=1.2,
               label=f"median {np.median(fw):.3f} deg")
    ax.set_xlabel("firmware attitude error vs truth (deg)")
    ax.set_ylabel("frames")
    ax.set_title("Firmware attitude error, real HIL frames\n"
                 "run through parse -> ephemeris -> adsCompute",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = OUT / "accuracy_firmware_vs_truth.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "plot_orbit_r10"
    made = []
    if (REPO / "results" / "sensor_fidelity" / tag / "manifest.json").exists():
        manifest, rows = load(tag)
        made.append(fig_ablation(rows, tag))
        made.append(fig_seed_spread(rows, tag))
    made += fig_timing()
    fw = fig_firmware_vs_study()
    if fw:
        made.append(fw)
    made = [m for m in made if m]
    for m in made:
        print("wrote", m.relative_to(REPO))
