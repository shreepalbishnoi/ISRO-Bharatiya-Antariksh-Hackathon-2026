"""
pipeline.py
------------
End-to-end orchestrator: raw light curve -> preprocessing -> BLS detection
-> feature extraction -> classification -> parameter estimation (if
transit) -> visualization -> structured JSON report.

Usage (single target, real TESS data):

    import lightkurve as lk
    from pipeline import run_pipeline
    from classifier import build_pipeline, train_classifier, make_synthetic_training_set

    lc = lk.search_lightcurve("TIC 261136679", mission="TESS", author="SPOC")\
           .download_all().stitch()
    clf, _ = train_classifier(make_synthetic_training_set())
    report = run_pipeline(lc.time.value, lc.flux.value, clf, target_name="TOI-700")

See demo_synthetic.py for a fully offline, runnable example.
"""

import json
import logging

import numpy as np

from preprocessing import preprocess
import lightkurve as lk

from detection import detect_candidates, run_bls
from features import extract_features
from classifier import classify_candidate
from parameter_estimation import estimate_parameters
from visualize import plot_summary

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_pipeline(time, flux, trained_clf, target_name: str = "target",
                  max_candidates: int = 3, make_plots: bool = True,
                  plot_dir: str = ".", flux_err=None):
    """
    Run the full detection + classification + parameter-estimation pipeline
    on a single light curve (already-loaded time/flux arrays, e.g. from a
    lightkurve LightCurve object: lc.time.value, lc.flux.value).

    Returns a JSON-serializable report dict (one entry per detected
    candidate) and, if make_plots=True, writes one summary PNG per
    candidate to plot_dir.
    """
    # Wrap raw arrays in a minimal lightkurve LightCurve so we can reuse
    # the standard preprocessing routines (sigma-clip, flatten, normalize).
    lc = lk.LightCurve(time=time, flux=flux)
    if flux_err is not None:
        lc["flux_err"] = flux_err

    stages = preprocess(lc)
    flat = stages["flat"]
    t = np.asarray(flat.time.value, dtype=float)
    f = np.asarray(flat.flux.value, dtype=float)
    fe = None
    if "flux_err" in flat.colnames:
        fe_arr = np.asarray(flat["flux_err"], dtype=float)
        if np.isfinite(fe_arr).any() and not np.isnan(fe_arr).all():
            fe = fe_arr

    candidates = detect_candidates(t, f, fe, max_candidates=max_candidates)

    report = {"target": target_name, "n_candidates": len(candidates), "candidates": []}

    for i, cand in enumerate(candidates):
        feats = extract_features(t, f, cand)
        classification = classify_candidate(trained_clf, feats)

        entry = {
            "candidate_index": i,
            "period_days": cand["period_days"],
            "duration_hours": cand["duration_days"] * 24,
            "depth_ppm": cand["depth"] * 1e6,
            "snr": cand["snr"],
            "significant": cand["significant"],
            "features": feats,
            "classification": classification,
        }

        params = {}
        if classification["predicted_class"] == "transit":
            params = estimate_parameters(t, f, cand)
            entry["refined_parameters"] = {
                k: v for k, v in params.items()
                if k not in ("phase_fit", "flux_fit", "model_phase", "model_flux")
            }
        else:
            entry["refined_parameters"] = None

        if make_plots:
            try:
                _, periodogram, _ = run_bls(t, f, fe)
            except Exception:  # noqa: BLE001
                periodogram = None
            fig = plot_summary(t, np.asarray(stages["normalized"].flux.value), f,
                                periodogram, cand, params if params else cand,
                                classification, target_name=f"{target_name} (cand {i})",
                                save_path=f"{plot_dir}/{target_name}_candidate{i}.png")
            import matplotlib.pyplot as plt
            plt.close(fig)

        report["candidates"].append(entry)

    return report


def report_to_json(report: dict, path: str):
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=_default)
    log.info("Wrote report to %s", path)
