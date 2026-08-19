"""
features.py
-------------
Extracts diagnostic features from a BLS candidate + phase-folded light
curve that distinguish:
  - transit          : planet crossing the star (deep, U/V shaped, no
                        secondary eclipse, odd/even depths equal)
  - eclipsing_binary  : stellar companion (deep, often V-shaped, frequently
                         shows a secondary eclipse, odd/even depth or
                         duration asymmetry possible for eccentric/grazing
                         orbits)
  - blend             : signal diluted/contaminated by a nearby eclipsing
                         source in the photometric aperture (shallow,
                         odd/even depth mismatch from blended period
                         aliasing, centroid shifts -- approximated here via
                         depth-duration inconsistency since centroid data
                         may be unavailable)
  - other             : starspots / rotational modulation / non-periodic
                         residual variability (shallow power, poor
                         box-shape fit, low SNR, not strictly periodic dip)

These engineered features feed the RandomForest classifier in classifier.py.
The feature set is intentionally compact and physically motivated so the
model remains interpretable, which is important for a hackathon report.
"""

import numpy as np


def phase_fold(time, flux, period, t0):
    """Phase-fold time onto a single period, centered on the transit."""
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    order = np.argsort(phase)
    return phase[order], flux[order]


def _secondary_eclipse_depth(phase, flux, duration, period):
    """
    Look for a dip near phase = +/- period/2 (secondary eclipse location for
    a circular orbit) which is a strong eclipsing-binary indicator and is
    not expected for planetary transits (negligible secondary in optical
    TESS band for most planets).
    """
    half = period / 2.0
    in_secondary = (np.abs(np.abs(phase) - half) < duration)
    if in_secondary.sum() < 5:
        return 0.0
    baseline = np.nanmedian(flux[~in_secondary])
    secondary_flux = np.nanmedian(flux[in_secondary])
    return float(baseline - secondary_flux)


def _shape_metric(phase, flux, duration):
    """
    V-shape vs U-shape metric: compares flux at the transit center vs at
    the transit edges (ingress/egress). Eclipsing binaries / grazing events
    tend to be V-shaped (no flat bottom); planetary transits are typically
    flat-bottomed (U-shaped) unless grazing.
    Returns ratio in [0, 1]; values near 1 => flat-bottomed (U), values
    near 0 => sharply pointed (V).
    """
    half_dur = duration / 2.0
    center_mask = np.abs(phase) < 0.2 * half_dur
    edge_mask = (np.abs(phase) > 0.6 * half_dur) & (np.abs(phase) < half_dur)
    if center_mask.sum() < 3 or edge_mask.sum() < 3:
        return np.nan
    center_depth = np.nanmedian(flux[center_mask])
    edge_depth = np.nanmedian(flux[edge_mask])
    out_of_transit = np.nanmedian(flux[np.abs(phase) > half_dur * 1.5])
    if out_of_transit == edge_depth:
        return np.nan
    ratio = (out_of_transit - center_depth) / (out_of_transit - edge_depth + 1e-12)
    return float(np.clip(ratio, 0, 2))


def extract_features(time, flux, candidate: dict) -> dict:
    """
    Build the feature vector for one BLS candidate.
    `candidate` is one entry returned by detection.detect_candidates().
    """
    period = candidate["period_days"]
    duration = candidate["duration_days"]
    t0 = candidate["t0"]
    depth = candidate["depth"]

    phase, flux_sorted = phase_fold(np.asarray(time), np.asarray(flux), period, t0)

    odd_depth = candidate.get("depth_odd", np.nan)
    even_depth = candidate.get("depth_even", np.nan)
    odd_even_mismatch = (abs(odd_depth - even_depth) /
                          (max(abs(odd_depth), abs(even_depth)) + 1e-8)
                          if np.isfinite(odd_depth) and np.isfinite(even_depth) else np.nan)
    if np.isfinite(odd_even_mismatch):
        odd_even_mismatch = float(np.clip(odd_even_mismatch, 0, 5))

    secondary_depth = _secondary_eclipse_depth(phase, flux_sorted, duration, period)
    secondary_ratio = secondary_depth / depth if depth > 0 else np.nan

    shape = _shape_metric(phase, flux_sorted, duration)

    duration_period_ratio = duration / period

    feats = {
        "period_days": period,
        "duration_hours": duration * 24.0,
        "depth_ppm": depth * 1e6,
        "snr": candidate.get("snr", np.nan),
        "bls_power": candidate.get("bls_power", np.nan),
        "odd_even_mismatch": odd_even_mismatch,
        "secondary_eclipse_ratio": secondary_ratio,
        "shape_metric": shape,           # ~1 = U-shaped (planet-like), ~0 = V-shaped (EB-like)
        "duration_period_ratio": duration_period_ratio,
        "depth": depth,
    }
    return feats
