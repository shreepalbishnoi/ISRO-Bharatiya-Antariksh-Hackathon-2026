"""
classifier.py
--------------
Supervised classifier that categorizes a detected periodic-dip candidate
into one of: transit, eclipsing_binary, blend, other.

Training data: the PS states a "curated dataset ... for already known
exoplanets, false positives, and eclipsing binaries" will be supplied.
This module expects that catalog as a CSV of light-curve-derived features
(see features.py) + a `label` column, e.g. built from:
  - NASA Exoplanet Archive TOI table  (disposition: CP/KP -> transit)
  - TESS False Positive catalogs       (disposition: EB -> eclipsing_binary,
                                          O  (blend/contamination) -> blend)
  - Variable-star catalogs / non-periodic residuals -> other

If no curated dataset is supplied (e.g. for a fast prototype/demo), a
physically-motivated synthetic training set is generated instead, using
the same feature distributions implied by the features.py definitions.

Model: RandomForestClassifier - chosen over deep nets because (a) the
feature set is small & physically interpretable, (b) tree ensembles are
robust to the modest, imbalanced training catalogs typical of TESS FP
tables, and (c) it directly outputs class probabilities, used downstream
as the "confidence level of the detected signal" required by the PS.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import CLASSES, RANDOM_STATE

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "period_days", "duration_hours", "depth_ppm", "snr", "bls_power",
    "odd_even_mismatch", "secondary_eclipse_ratio", "shape_metric",
    "duration_period_ratio",
]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def load_curated_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load a curated catalog of features + labels, as supplied for the
    hackathon. Expected columns: FEATURE_COLUMNS + 'label'.
    """
    df = pd.read_csv(csv_path)
    missing = set(FEATURE_COLUMNS + ["label"]) - set(df.columns)
    if missing:
        raise ValueError(f"Curated dataset missing required columns: {missing}")
    return df


def make_synthetic_training_set(n_per_class: int = 300, seed: int = RANDOM_STATE
                                 ) -> pd.DataFrame:
    """
    Generates a labeled feature table consistent with the physical
    definitions in features.py, for bootstrapping / unit-testing the
    classifier when no curated catalog is available yet.
    """
    rng = np.random.default_rng(seed)
    rows = []

    def add(label, n, **dists):
        for _ in range(n):
            row = {k: float(f(rng)) for k, f in dists.items()}
            row["label"] = label
            rows.append(row)

    add("transit", n_per_class,
        period_days=lambda r: r.uniform(0.5, 20),
        duration_hours=lambda r: r.uniform(1, 6),
        depth_ppm=lambda r: r.uniform(100, 5000),
        snr=lambda r: r.uniform(7, 40),
        bls_power=lambda r: r.uniform(0.05, 0.3),
        odd_even_mismatch=lambda r: np.abs(r.normal(0.02, 0.02)),
        secondary_eclipse_ratio=lambda r: np.abs(r.normal(0.0, 0.02)),
        shape_metric=lambda r: r.uniform(0.7, 1.0),
        duration_period_ratio=lambda r: r.uniform(0.01, 0.08))

    add("eclipsing_binary", n_per_class,
        period_days=lambda r: r.uniform(0.3, 15),
        duration_hours=lambda r: r.uniform(1, 10),
        depth_ppm=lambda r: r.uniform(3000, 100000),
        snr=lambda r: r.uniform(10, 80),
        bls_power=lambda r: r.uniform(0.1, 0.5),
        odd_even_mismatch=lambda r: np.abs(r.normal(0.05, 0.05)),
        secondary_eclipse_ratio=lambda r: np.abs(r.normal(0.3, 0.15)),
        shape_metric=lambda r: r.uniform(0.1, 0.5),
        duration_period_ratio=lambda r: r.uniform(0.02, 0.15))

    add("blend", n_per_class,
        period_days=lambda r: r.uniform(0.3, 15),
        duration_hours=lambda r: r.uniform(0.5, 8),
        depth_ppm=lambda r: r.uniform(50, 2000),
        snr=lambda r: r.uniform(5, 15),
        bls_power=lambda r: r.uniform(0.02, 0.15),
        odd_even_mismatch=lambda r: np.abs(r.normal(0.25, 0.15)),
        secondary_eclipse_ratio=lambda r: np.abs(r.normal(0.15, 0.15)),
        shape_metric=lambda r: r.uniform(0.2, 0.7),
        duration_period_ratio=lambda r: r.uniform(0.01, 0.1))

    add("other", n_per_class,
        period_days=lambda r: r.uniform(0.5, 20),
        duration_hours=lambda r: r.uniform(0.5, 10),
        depth_ppm=lambda r: r.uniform(50, 1500),
        snr=lambda r: r.uniform(3, 9),
        bls_power=lambda r: r.uniform(0.01, 0.08),
        odd_even_mismatch=lambda r: np.abs(r.normal(0.3, 0.2)),
        secondary_eclipse_ratio=lambda r: np.abs(r.normal(0.1, 0.1)),
        shape_metric=lambda r: r.uniform(0.0, 1.0),
        duration_period_ratio=lambda r: r.uniform(0.005, 0.2))

    return pd.DataFrame(rows)


def train_classifier(df: pd.DataFrame, model_out: str = None):
    """Trains the RandomForest, reports cross-validated metrics, optionally saves."""
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(pipe, X, y, cv=cv)

    log.info("5-fold cross-validated performance:\n%s",
              classification_report(y, y_pred, labels=CLASSES))
    cm = confusion_matrix(y, y_pred, labels=CLASSES)
    log.info("Confusion matrix (rows=true, cols=pred), classes=%s:\n%s", CLASSES, cm)

    pipe.fit(X, y)  # fit final model on all data
    if model_out:
        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_out)
        log.info("Saved trained model to %s", model_out)
    return pipe, {"confusion_matrix": cm.tolist(),
                  "report": classification_report(y, y_pred, labels=CLASSES, output_dict=True)}


def classify_candidate(pipe: Pipeline, feats: dict):
    """
    Predict the class + confidence (max class probability) for one
    candidate's feature dict (as produced by features.extract_features).
    """
    X = pd.DataFrame([{k: feats.get(k, np.nan) for k in FEATURE_COLUMNS}])
    X = X.replace([np.inf, -np.inf], np.nan)  # SimpleImputer can't handle inf
    proba = pipe.predict_proba(X)[0]
    classes = pipe.named_steps["clf"].classes_
    pred_idx = int(np.argmax(proba))
    result = {
        "predicted_class": classes[pred_idx],
        "confidence": float(proba[pred_idx]),
        "class_probabilities": {c: float(p) for c, p in zip(classes, proba)},
    }
    return result
