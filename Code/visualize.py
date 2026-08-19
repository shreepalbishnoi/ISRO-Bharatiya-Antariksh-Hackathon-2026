"""
visualize.py
--------------
Generates the diagnostic figure required by the PS:
  - Raw vs detrended light curve, with the detected transit times marked.
  - BLS periodogram (shows the significance of the period detection).
  - Phase-folded light curve with the best-fit transit model overlaid.
  - Bar chart of classification confidence across all four categories.

All four panels are combined into a single figure per target/candidate so
it can be dropped directly into the report.
"""

import matplotlib.pyplot as plt
import numpy as np

from config import CLASSES


def plot_summary(time, flux_raw, flux_flat, periodogram, candidate: dict,
                  params: dict, classification: dict, target_name: str = "",
                  save_path: str = None):
    """
    Build the 2x2 diagnostic summary plot.
    `periodogram` is the object returned by detection.run_bls (has .period,
    .power attributes); pass None to skip that panel.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Exoplanet Detection Pipeline Summary — {target_name}",
                 fontsize=14, fontweight="bold")

    # --- Panel 1: raw + detrended light curve with transit markers ---
    ax = axes[0, 0]
    ax.plot(time, flux_raw, ".", color="lightgray", ms=2, label="Raw (normalized)")
    ax.plot(time, flux_flat, ".", color="steelblue", ms=2, label="Detrended")
    if "transit_times" in candidate and len(candidate["transit_times"]) > 0:
        for tt in candidate["transit_times"]:
            ax.axvline(tt, color="crimson", alpha=0.4, lw=1)
    ax.set_xlabel("Time (BJD - 2457000)")
    ax.set_ylabel("Relative flux")
    ax.set_title("Light curve (red lines = detected transit epochs)")
    ax.legend(loc="lower left", fontsize=8)

    # --- Panel 2: BLS periodogram ---
    ax = axes[0, 1]
    if periodogram is not None:
        ax.plot(periodogram.period, periodogram.power, color="darkorange", lw=0.8)
        ax.axvline(candidate["period_days"], color="crimson", ls="--",
                    label=f"P = {candidate['period_days']:.4f} d")
        ax.set_xlabel("Trial period (days)")
        ax.set_ylabel("BLS power (SNR objective)")
        ax.set_title("BLS periodogram")
        ax.legend(fontsize=8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Periodogram not available", ha="center")

    # --- Panel 3: phase-folded light curve with model fit ---
    ax = axes[1, 0]
    if "phase_fit" in params:
        ax.plot(params["phase_fit"] * 24, params["flux_fit"], ".", color="gray",
                ms=3, alpha=0.5, label="Phase-folded data")
        ax.plot(params["model_phase"] * 24, params["model_flux"], color="crimson",
                lw=2, label="Best-fit trapezoid model")
        ax.set_xlabel("Hours from mid-transit")
    else:
        phase = ((time - candidate["t0"] + 0.5 * candidate["period_days"])
                  % candidate["period_days"]) - 0.5 * candidate["period_days"]
        ax.plot(phase, flux_flat, ".", color="gray", ms=2)
        ax.set_xlabel("Phase (days)")
    ax.set_ylabel("Relative flux")
    ax.set_title(f"Phase-folded transit (depth = {params.get('depth_ppm', candidate['depth']*1e6):.0f} ppm)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)

    # --- Panel 4: classification confidence ---
    ax = axes[1, 1]
    probs = classification.get("class_probabilities", {})
    classes = [c for c in CLASSES if c in probs]
    values = [probs[c] for c in classes]
    colors = ["#2a9d8f" if c == classification.get("predicted_class") else "#adb5bd"
              for c in classes]
    bars = ax.bar(classes, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Class probability")
    ax.set_title(f"Classification: {classification.get('predicted_class', 'N/A')} "
                 f"(confidence={classification.get('confidence', 0):.2f})\n"
                 f"Detection SNR = {candidate.get('snr', np.nan):.1f}")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
