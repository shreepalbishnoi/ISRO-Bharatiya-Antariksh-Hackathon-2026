"""
detection.py
-------------
Searches the flattened light curve for periodic dip signals using the
Box Least Squares (BLS) algorithm (Kovacs et al. 2002), implemented in
astropy.timeseries. BLS is the standard method used by the TESS/Kepler
pipelines to flag Threshold Crossing Events (TCEs).

For each candidate period found, we:
  - Record period, duration, depth, epoch (t0), and the BLS power.
  - Compute a Signal-to-Noise Ratio (SNR) of the event, used both as a
    detection threshold and as the "significance level" requested in the PS.
  - Optionally search for additional, weaker periodic signals after masking
    the strongest one (iterative BLS), to catch multi-planet systems.
"""

import logging

import numpy as np
from astropy.timeseries import BoxLeastSquares

from config import (MAX_DURATION_HOURS, MAX_PERIOD_DAYS, MIN_DURATION_HOURS,
                     MIN_PERIOD_DAYS, N_DURATIONS, N_PERIODS,
                     SNR_DETECTION_THRESHOLD)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_bls(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray = None,
            min_period=MIN_PERIOD_DAYS, max_period=MAX_PERIOD_DAYS,
            n_periods=N_PERIODS, min_dur_hr=MIN_DURATION_HOURS,
            max_dur_hr=MAX_DURATION_HOURS, n_durations=N_DURATIONS):
    """
    Run the BLS periodogram over a grid of periods and durations.
    Returns the BLS object and its periodogram result (for plotting) plus
    the best-fit (period, duration, t0, depth, power).
    """
    durations = np.linspace(min_dur_hr, max_dur_hr, n_durations) / 24.0  # hours -> days
    periods = np.linspace(min_period, max_period, n_periods)

    bls = BoxLeastSquares(time, flux, dy=flux_err)
    periodogram = bls.power(periods, durations, objective="snr")

    best_idx = np.argmax(periodogram.power)
    best_period = periodogram.period[best_idx]
    best_duration = periodogram.duration[best_idx]
    best_t0 = periodogram.transit_time[best_idx]
    best_depth = periodogram.depth[best_idx]
    best_power = periodogram.power[best_idx]

    stats = bls.compute_stats(best_period, best_duration, best_t0)
    depth_val, depth_err = stats["depth"]
    depth_odd_val, depth_odd_err = stats["depth_odd"]
    depth_even_val, depth_even_err = stats["depth_even"]

    result = {
        "period_days": float(best_period),
        "duration_days": float(best_duration),
        "t0": float(best_t0),
        "depth": float(best_depth),
        "depth_err": float(depth_err),
        "bls_power": float(best_power),
        "depth_snr": float(depth_val / depth_err) if depth_err else np.nan,
        "transit_times": np.asarray(stats["transit_times"]),
        "depth_odd": float(depth_odd_val),
        "depth_even": float(depth_even_val),
        "depth_odd_err": float(depth_odd_err),
        "depth_even_err": float(depth_even_err),
    }
    return bls, periodogram, result


def estimate_snr(flux: np.ndarray, depth: float, duration_days: float,
                  period_days: float, cadence_days: float) -> float:
    """
    Signal-to-noise ratio of the detected event, following the standard
    "transit SNR" definition used by Kepler/TESS pipelines:
        SNR = depth / sigma_per_point * sqrt(N_in_transit_total)
    where N_in_transit_total accounts for all transits observed in the
    baseline, not just one.
    """
    sigma = np.nanstd(flux)
    n_points_per_transit = max(duration_days / cadence_days, 1)
    baseline_days = len(flux) * cadence_days
    n_transits_observed = max(baseline_days / period_days, 1)
    n_in_transit_total = n_points_per_transit * n_transits_observed
    snr = depth / sigma * np.sqrt(n_in_transit_total)
    return float(snr)


def detect_candidates(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray = None,
                       max_candidates: int = 3, mask_width_factor: float = 2.0):
    """
    Iteratively run BLS, recording each significant candidate and masking
    its in-transit points before searching for the next (handles
    multi-planet systems / multiple periodic signals).
    Returns a list of candidate dicts, each augmented with an SNR estimate
    and a boolean `significant` flag.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float).copy()
    cadence_days = np.median(np.diff(np.sort(time)))

    candidates = []
    mask = np.ones_like(flux, dtype=bool)

    for i in range(max_candidates):
        if mask.sum() < 50:
            break
        t_sub, f_sub = time[mask], flux[mask]
        fe_sub = flux_err[mask] if flux_err is not None else None
        try:
            bls, periodogram, result = run_bls(t_sub, f_sub, fe_sub)
        except Exception as exc:  # noqa: BLE001
            log.warning("BLS failed on iteration %d: %s", i, exc)
            break

        snr = estimate_snr(f_sub, result["depth"], result["duration_days"],
                            result["period_days"], cadence_days)
        result["snr"] = snr
        result["significant"] = snr >= SNR_DETECTION_THRESHOLD
        candidates.append(result)

        if not result["significant"]:
            break  # weaker signals beyond this are treated as noise

        # mask out this transit's in-transit points (within the full series)
        phase = ((time - result["t0"] + 0.5 * result["period_days"])
                  % result["period_days"]) - 0.5 * result["period_days"]
        in_transit = np.abs(phase) < (mask_width_factor * result["duration_days"] / 2.0)
        mask &= ~in_transit

    return candidates
