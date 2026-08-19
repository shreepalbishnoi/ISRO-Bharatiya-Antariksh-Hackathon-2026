"""
preprocessing.py
-----------------
Cleans and detrends raw TESS light curves so periodic dip-search algorithms
are not confused by instrumental systematics, long-term stellar variability,
or outliers (cosmic-ray hits, momentum-dump artifacts, etc.)

Pipeline order (standard for TESS SPOC products):
  1. Remove NaNs / non-finite flux.
  2. Remove quality-flagged cadences (using TESS QUALITY bitmask).
  3. Iterative sigma-clipping to remove outliers (flares, cosmic rays).
  4. Normalize flux to relative units (median = 1).
  5. Detrend / flatten using a Savitzky-Golay filter (removes stellar
     rotation signals & instrumental drift while preserving short transits).
"""

import logging

import numpy as np
from astropy.stats import sigma_clip
from astropy.timeseries import TimeSeries
import lightkurve as lk

from config import (FLATTEN_POLYORDER, FLATTEN_WINDOW_LENGTH,
                     SIGMA_CLIP_LOWER, SIGMA_CLIP_UPPER)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def remove_bad_quality(lc: lk.LightCurve) -> lk.LightCurve:
    """Drop cadences flagged by the TESS pipeline's QUALITY bitmask."""
    if "quality" in lc.colnames:
        lc = lc[lc["quality"] == 0]
    return lc


def clean_light_curve(lc: lk.LightCurve) -> lk.LightCurve:
    """Remove NaNs and sigma-clip outliers."""
    lc = lc.remove_nans()
    lc = remove_bad_quality(lc)
    flux = np.asarray(lc.flux.value, dtype=float)
    mask = sigma_clip(flux, sigma_lower=SIGMA_CLIP_LOWER,
                       sigma_upper=SIGMA_CLIP_UPPER, maxiters=5).mask
    lc = lc[~mask]
    return lc


def normalize(lc: lk.LightCurve) -> lk.LightCurve:
    """Normalize flux so the median equals 1 (standard for relative depth)."""
    return lc.normalize()


def flatten_light_curve(lc: lk.LightCurve, window_length: int = FLATTEN_WINDOW_LENGTH,
                         polyorder: int = FLATTEN_POLYORDER):
    """
    Remove long-term trends (stellar rotation, instrument drift) with a
    Savitzky-Golay filter while masking out-of-trend transit-like dips so
    the transit itself is not flattened away.
    Returns (flattened_lc, trend_lc).
    """
    # lightkurve's .flatten() internally masks outliers iteratively (biweight)
    flat_lc, trend_lc = lc.flatten(window_length=window_length,
                                    polyorder=polyorder, return_trend=True)
    return flat_lc, trend_lc


def preprocess(lc: lk.LightCurve) -> dict:
    """
    Full preprocessing pipeline. Returns a dict with intermediate products
    so each stage can be inspected / plotted for the report.
    """
    log.info("Preprocessing light curve with %d points", len(lc.flux))
    raw = lc
    cleaned = clean_light_curve(raw)
    normalized = normalize(cleaned)
    flat, trend = flatten_light_curve(normalized)

    # Estimate per-point photometric noise (used later for SNR calculations)
    noise_ppm = np.nanstd(flat.flux.value) * 1e6

    return {
        "raw": raw,
        "cleaned": cleaned,
        "normalized": normalized,
        "trend": trend,
        "flat": flat,
        "noise_ppm": noise_ppm,
    }
