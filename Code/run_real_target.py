"""
run_real_target.py
------------------
Fetches real TESS light curve data online using data_acquisition.py,
runs it through the exoplanet detection pipeline, and writes output
reports and diagnostic plots.
"""

import os
import logging
import joblib
from pathlib import Path
from data_acquisition import search_target
from pipeline import run_pipeline, report_to_json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    # 1. Ensure outputs directory exists
    output_dir = Path("demo_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load trained classifier model
    model_path = Path("models/signal_classifier.joblib")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained classifier not found at {model_path}. Please run train_classifier.py first."
        )
    log.info("Loading classifier model from %s", model_path)
    clf = joblib.load(model_path)

    # 3. Fetch real TESS light curve data using data_acquisition.py
    # We will use TIC 261136679 (TOI-700), a known exoplanet host star.
    target_id = "261136679"
    target_name = "TOI-700"
    log.info("Searching TESS light curve products for TIC %s (%s)...", target_id, target_name)
    try:
        search_result = search_target(target_id)
        if len(search_result) == 0:
            log.error("No light curves found for TIC %s", target_id)
            return
        
        # Download only the first available sector to keep execution fast and avoid 2.4 million points
        log.info("Downloading the first available sector: %s", search_result[0])
        lc = search_result[0].download()
        log.info("Successfully downloaded light curve with %d points.", len(lc))
    except Exception as exc:
        log.error("Failed to acquire data from MAST: %s", exc)
        return

    # 4. Execute the pipeline
    log.info("Running exoplanet detection pipeline on the light curve...")
    try:
        # Pass time, flux, and optionally flux_err
        time = lc.time.value
        flux = lc.flux.value
        flux_err = lc.flux_err.value if hasattr(lc, "flux_err") else None
        
        report = run_pipeline(
            time=time,
            flux=flux,
            trained_clf=clf,
            target_name=target_name,
            max_candidates=3,
            make_plots=True,
            plot_dir=str(output_dir),
            flux_err=flux_err
        )
        
        # 5. Save the report to JSON
        report_path = output_dir / f"{target_name.lower()}_report.json"
        report_to_json(report, str(report_path))
        log.info("Pipeline executed successfully! Output report written to %s", report_path)
        log.info("Diagnostic plots saved under %s/", output_dir)
    except Exception as exc:
        log.error("Pipeline execution failed: %s", exc, exc_info=True)

if __name__ == "__main__":
    main()
