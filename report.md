# AI-Enabled Detection of Exoplanets from Noisy TESS Light Curves

**ISRO Antariksh Hackathon — Technical Report**

## 1. Methodology

The pipeline follows the standard transit-photometry workflow used by the
Kepler/TESS Science Processing Operations Center, adapted with an
ML-based classification layer.

**Data acquisition.** Light curves are retrieved from MAST via
`lightkurve`, using TESS-SPOC 2-minute-cadence PDCSAP (Pre-search Data
Conditioning) flux, which already has common instrumental systematics
(scattered light, pointing jitter, thermal effects) removed by the
official pipeline. For a full sector (20-30k targets as suggested in the
PS), target lists are queried first and light curves are downloaded and
processed in batches, since per-target FITS retrieval at that scale is
multi-terabyte.

**Preprocessing.** Each light curve is (a) stripped of non-zero
QUALITY-flagged cadences and NaNs, (b) iteratively sigma-clipped
(8-sigma lower / 5-sigma upper, asymmetric because flares/positive
outliers are far more common than spurious dips) to remove cosmic rays
and flares without clipping genuine transit bottoms, (c) normalized to
unit median flux, and (d) flattened with a Savitzky-Golay filter
(window ≈ a few transit durations) to remove stellar rotation and
residual instrumental drift while preserving short-duration dips.

**Detection.** A Box Least Squares (BLS) periodogram (Kovács et al. 2002,
via `astropy.timeseries.BoxLeastSquares`) is run over a period grid
(1-20 days, suitable for TESS's ~27-day sector baseline) and a duration
grid (0.5-10 hours). BLS is chosen over generic period-finding methods
(e.g. Lomb-Scargle) because its box-shaped kernel is matched to the
physical shape of a transit/eclipse, giving better sensitivity at fixed
false-alarm rate. The search is run iteratively: after the strongest
period is found, its in-transit points are masked and BLS is re-run,
recovering additional (e.g. multi-planet) signals up to a configurable
limit. Each candidate's significance is quantified as `SNR = depth /
sigma_per_point × sqrt(N_in-transit total)`, summed over all transits
observed in the baseline — the same definition used for Kepler/TESS
Threshold Crossing Events. Candidates below SNR = 7 are discarded as
consistent with noise.

**Feature extraction.** For each significant candidate, physically
motivated features are computed from the phase-folded light curve: depth,
duration, period, BLS power/SNR, an odd-even transit depth mismatch (a
classic giveaway of a blended/aliased period — real planets and most EBs
show negligible odd/even difference; blends from a contaminating EB at
twice the true period do not), a secondary-eclipse depth ratio (present
for eclipsing binaries, generally absent for planets in the TESS optical
band), and a U-shape vs. V-shape metric (flat-bottomed transits indicate
a planet/grazing-free eclipse; sharply pointed V-shapes indicate a
stellar eclipse or grazing geometry).

**Classification.** These features feed a RandomForest classifier (400
trees, balanced class weights) that outputs one of four labels — transit,
eclipsing_binary, blend, other — together with class probabilities, used
directly as the reported confidence level. RandomForest was chosen over a
deep network because the feature set is small, low-dimensional and
physically interpretable; tree ensembles handle the class imbalance and
modest catalog sizes typical of TESS false-positive tables well, and
expose feature importances that make the classification auditable, which
matters for a competition report. The model is trained on the
hackathon-supplied curated catalog of known exoplanets / eclipsing
binaries / false positives (expected as a CSV of these same features +
label); a physically-motivated synthetic catalog is provided as a
fallback so the pipeline is runnable immediately without that file.

**Parameter estimation.** For candidates classified as a genuine transit,
the BLS best-fit (period, epoch, duration, depth) is refined by
least-squares fitting a trapezoidal transit model to the phase-folded
light curve. A trapezoid (rather than a full Mandel-Agol limb-darkened
model) was chosen to avoid a hard dependency on stellar limb-darkening
coefficients / `batman` for the baseline pipeline; an optional
`batman`-based fit hook is included for users who want full physical
parameters (Rp/R★, a/R★, inclination) once stellar parameters are
available.

**Visualization.** A single four-panel figure per candidate shows: the
raw vs. detrended light curve with detected transit epochs marked, the
BLS periodogram, the phase-folded light curve with the best-fit model
overlaid, and a bar chart of classification probabilities with the
detection SNR — directly satisfying the "visualization with confidence
level" requirement.

## 2. Uncertainty Estimation

Two uncertainties are reported per detection:

- **Detection significance (SNR)**: derived analytically from the
  per-point photometric scatter and the total number of in-transit
  cadences across all observed transits, following the standard
  Kepler/TESS TCE SNR definition.
- **Parameter uncertainties (period, duration, depth)**: estimated via
  residual bootstrap — the best-fit trapezoid model is subtracted from
  the phase-folded data, the residuals are resampled with replacement
  (300 realizations), and the model is re-fit to each resampled data set.
  The standard deviation of each parameter across realizations gives an
  empirical, non-Gaussian 1-sigma uncertainty without assuming
  homoscedastic photometric noise (a standard nonparametric alternative
  to a Fisher-matrix/MCMC approach, much cheaper computationally while
  still accounting for real, non-uniform TESS noise). Period uncertainty
  is propagated from the mid-transit timing precision divided by the
  number of transits observed in the baseline (σ_P ≈ σ_t0 / N_transits),
  the standard relation for periods determined from multiple transit
  epochs.

## 3. Assumptions

- TESS-SPOC PDCSAP flux already corrects for most instrumental
  systematics; remaining structure is treated as stellar rotation/drift
  and removed by the Savitzky-Golay flatten.
- The trapezoid transit model assumes a fixed ingress/egress fraction
  (20% of total duration), appropriate at typical TESS 2-min cadence
  where ingress/egress timing is rarely independently resolvable; this
  can be relaxed once high-cadence/space-based follow-up is available.
- Class definitions follow standard TESS vetting conventions: deep,
  V-shaped, often-secondary-bearing signals → eclipsing binary; shallow,
  odd/even-inconsistent, low-SNR signals inconsistent with a clean box
  shape → blend; flat-bottomed, periodic, secondary-free dips → transit;
  everything below the SNR threshold or inconsistent with a clean box
  shape (e.g. pure rotational modulation) → other.
- The classifier's accuracy is bounded by the representativeness of the
  curated training catalog; the synthetic fallback is for pipeline
  validation only and should be replaced with the hackathon-provided
  catalog (or NASA Exoplanet Archive TOI / EB / FP tables) for science
  use.

## 4. Tools and Libraries

`lightkurve` (MAST/TESS data access and standard LC utilities), `astropy`
(BLS periodogram, sigma-clipping, time-series I/O), `scikit-learn`
(RandomForest classifier, cross-validation), `scipy` (trapezoid model
least-squares fitting, bootstrap), `numpy`/`pandas` (array/tabular
processing), `matplotlib` (visualization). All are open-source, free, and
standard in the TESS/Kepler community, satisfying the PS's "publicly
available Python tools" requirement. An optional `batman` dependency is
supported for higher-fidelity (Mandel-Agol) transit modeling.

## 5. Validation Results

The pipeline was validated end-to-end on three synthetic 27-day,
2-minute-cadence light curves built to TESS noise levels (~200 ppm
white noise + stellar rotation signal): (1) a clean planetary transit
(3.5 d period, 3600 ppm depth) — correctly detected at SNR ≈ 113 and
classified `transit` with the recovered period matching the injected
value to <0.001 d; (2) an eclipsing binary with a secondary eclipse
(4.2 d period, ~19,000 ppm primary depth) — correctly detected at SNR ≈
105 and classified `eclipsing_binary`; (3) pure stellar rotational
variability with no injected periodic dip — correctly suppressed to
SNR ≈ 5 (below the SNR = 7 detection threshold) and classified `other`,
demonstrating the pipeline does not over-trigger on non-transit
variability. Cross-validated classifier accuracy on the (synthetic)
training catalog was 98% (macro F1 = 0.98) across the four classes;
real-world accuracy will depend on the curated catalog supplied for the
competition and should be re-validated against it.
