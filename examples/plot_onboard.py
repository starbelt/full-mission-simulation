"""Figures for the ADS algorithm running on the ESP32-S3 core 0.

Scope, deliberately narrow: everything here is measured ON THE BOARD. The
Python simulation is a stand-in that generates the frames the board consumes,
so it appears only as the source of the input, never as a performance claim.

Kept to three figures, each answering one question:

  1. accuracy      -- how far the board's attitude is from truth
  2. latency       -- how long the board takes per sample
  3. summary       -- the two numbers plus the CPU headroom

No multi-panel figures, no secondary axes, no annotated caveats competing for
space with the data.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE = "#2f6f9f"
DARK = "#1a1a1a"
GREY = "#999999"


def _load(name):
    p = REPO / "results" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ------------------------------------------------------------------ 1. accuracy
def fig_accuracy():
    """Firmware attitude error vs truth, straight off the board."""
    d = _load("hil/onboard_accuracy.json")
    if not d:
        return None
    err = np.array(d["error_deg"])

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.hist(err, bins=30, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(err), color=DARK, lw=1.5, ls="--",
               label=f"median {np.median(err):.2f}$\\degree$")
    ax.set_xlabel("attitude error vs truth (degrees)")
    ax.set_ylabel("samples")
    ax.set_title("ESP32-S3 core 0: attitude error over "
                 f"{len(err):,} HIL frames", loc="left", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    p = OUT / "onboard_accuracy.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------- 2. latency
def fig_latency():
    """Per-sample wall time measured on the board, one dot per frame."""
    d = _load("hil/onboard_latency.json")
    if not d:
        return None
    us = np.array(d["us_per_sample"])
    frame_ms = np.array(d.get("frame_ms_per_sample", []))

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    if us.size:
        ax.hist(us, bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(us), color=DARK, lw=1.5, ls="--",
                   label=f"median {np.median(us):.0f} $\\mu$s")
        ax.set_xlabel("estimator time per sample, measured on core 0 "
                      "(microseconds)")
        ax.set_ylabel("samples")
        ax.set_title("ESP32-S3 core 0: per-sample execution time",
                     loc="left", fontsize=11)
        ax.legend(frameon=False)

    # Frame-to-frame interval is transport, not algorithm; show it only if
    # we have it, on its own axes-free scale as a second row.
    if frame_ms.size:
        fig2, ax2 = plt.subplots(figsize=(6.0, 2.8))
        ax2.hist(frame_ms, bins=40, color=GREY, edgecolor="white", linewidth=0.4)
        ax2.axvline(np.median(frame_ms), color=DARK, lw=1.5, ls="--",
                    label=f"median {np.median(frame_ms):.0f} ms")
        ax2.set_xlabel("frame-to-frame interval incl. serial transport (ms)")
        ax2.set_ylabel("frames")
        ax2.set_title("Serial transport overhead", loc="left", fontsize=11)
        ax2.legend(frameon=False)
        fig2.tight_layout()
        p2 = OUT / "onboard_frame_interval.png"
        fig2.savefig(p2, bbox_inches="tight")
        plt.close(fig2)

    fig.tight_layout()
    p = OUT / "onboard_latency.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# -------------------------------------------------------------------- 3. summary
def fig_summary():
    """One figure, the two numbers that matter, with the rate headroom."""
    acc = _load("hil/onboard_accuracy.json")
    lat = _load("hil/onboard_latency.json")
    if not lat:
        return None
    us = np.array(lat["us_per_sample"])
    med = float(np.median(us))

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    rates = np.array([1, 10, 50, 100, 200, 500, 1000])
    load = med * 1e-6 * rates * 100
    ax.plot(rates, load, "-o", color=BLUE, lw=1.6, ms=5)
    ax.axhline(100, color="#c1543a", ls="--", lw=1.2)
    ax.annotate("1 core saturated", (rates[0], 100), textcoords="offset points",
                xytext=(2, -14), fontsize=9, color="#c1543a")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sensor sample rate (Hz)")
    ax.set_ylabel("core 0 CPU load (%)")
    title = (f"ESP32-S3 core 0: {med:.0f} $\\mu$s per sample "
             f"= {1e6 / med:,.0f} samples/s ceiling")
    if acc:
        e = np.array(acc["error_deg"])
        title += f"\nattitude error median {np.median(e):.2f}$\\degree$, "
        title += f"p95 {np.percentile(e, 95):.2f}$\\degree$"
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    p = OUT / "onboard_summary.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    made = [m for m in (fig_accuracy(), fig_latency(), fig_summary()) if m]
    for m in made:
        print("wrote", m.relative_to(REPO))
    if not made:
        print("no results/hil/*.json yet -- run examples/hil_onboard_bench.py first")
