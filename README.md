# AI-Enabled Exoplanet Detection Pipeline (ISRO Antariksh Hackathon)

An end-to-end pipeline that ingests raw TESS light curves, cleans/detrends
them, searches for periodic transit-like dips (BLS), classifies each
detection (transit / eclipsing binary / blend / other), estimates transit
parameters with uncertainties, and produces a single diagnostic
visualization with a classification confidence score.

## Pipeline stages

| Stage | Module | Method |
|---|---|---|
| 1. Data acquisition | `data_acquisition.py` | MAST query via `lightkurve` (single target & sector-bulk) |
| 2. Preprocessing | `preprocessing.py` | quality-flag removal, sigma-clipping, normalization, Savitzky-Golay flattening |
| 3. Detection | `detection.py` | Box Least Squares (BLS) periodogram, iterative multi-signal search, SNR significance |
| 4. Feature extraction | `features.py` | shape (U/V), odd-even depth mismatch, secondary eclipse depth, duration/period ratio |
| 5. Classification | `classifier.py` | RandomForest (4 classes), trained on curated catalog or synthetic fallback, outputs class probabilities |
| 6. Parameter estimation | `parameter_estimation.py` | trapezoid transit-model fit (`scipy.optimize.curve_fit`) + bootstrap uncertainty |
| 7. Visualization | `visualize.py` | 2x2 summary figure: raw/detrended LC, BLS periodogram, phase-fold + model, confidence bars |
| 8. Orchestration | `pipeline.py` | ties stages 2-7 together, emits JSON report per target |

## Quick start

```bash
pip install -r requirements.txt

# 1. Fully offline sanity check / demo (no internet required):
python demo_synthetic.py
# -> writes plots + JSON reports to ./demo_outputs/

# 2. Train the classifier on the hackathon-supplied curated catalog
#    (CSV must have the feature columns listed in classifier.FEATURE_COLUMNS + 'label'):
python train_classifier.py path/to/curated_catalog.csv
# (omit the path to use a synthetic fallback for prototyping)

# 3. Run on a real TESS target (requires internet access to MAST):
python -c "
import lightkurve as lk, joblib
from pipeline import run_pipeline, report_to_json

clf = joblib.load('models/signal_classifier.joblib')
lc = lk.search_lightcurve('TIC 261136679', mission='TESS', author='SPOC').download_all().stitch()
report = run_pipeline(lc.time.value, lc.flux.value, clf, target_name='TOI-700')
report_to_json(report, 'toi700_report.json')
"
```

## Scaling to a full sector (~20-30k targets)

`data_acquisition.list_sector_targets()` enumerates a sector's TIC IDs via
MAST; `data_acquisition.bulk_download()` then fetches and lightly cleans
each one. For genuinely processing tens of thousands of light curves,
run `pipeline.run_pipeline()` in a loop/`multiprocessing.Pool`, persisting
each target's JSON report — full-sector FITS download is multi-terabyte
and is best done via MAST's bulk-download curl scripts or the TESS-SPOC /
QLP HLSP light-curve archives rather than per-target API calls.

## Key design choices (see report.md for full methodology)

- **BLS over deep learning for detection**: small, interpretable feature
  set; physically grounded; works without massive labeled training data,
  appropriate given the hackathon's curated-catalog size.
- **RandomForest over neural classifier**: outputs calibrated-enough class
  probabilities directly (used as "confidence"), is robust to the
  modest/imbalanced catalogs typical of TESS false-positive tables, and
  is fully interpretable (feature importances) for the report.
- **Trapezoid + bootstrap over MCMC**: keeps the parameter-estimation step
  fast and dependency-light while still producing defensible, empirical
  (non-Gaussian-assumption) uncertainties. An optional `batman`-based
  Mandel-Agol fit hook is provided (`parameter_estimation.fit_with_batman`)
  for users who want full physical parameters (Rp/Rs, a/Rs, inclination).

## Validation

`demo_synthetic.py` generates three synthetic 27-day, 2-minute-cadence
light curves (transit, eclipsing binary, pure stellar variability + noise)
and confirms the full pipeline recovers the correct period/depth and
assigns the correct class with reasonable confidence in each case (see
`demo_outputs/`).
