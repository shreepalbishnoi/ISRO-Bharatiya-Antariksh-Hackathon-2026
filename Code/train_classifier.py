"""
train_classifier.py
---------------------
Standalone script to train and persist the RandomForest classifier.

If a curated CSV (as supplied for the hackathon, with feature columns +
'label') is available, point CURATED_CSV to it. Otherwise this falls back
to a synthetic-but-physically-motivated training set so the pipeline is
runnable end-to-end immediately.

Run:  python train_classifier.py [path_to_curated_csv]
"""

import sys

from classifier import (load_curated_dataset, make_synthetic_training_set,
                         train_classifier)

MODEL_OUT = "models/signal_classifier.joblib"


def main():
    if len(sys.argv) > 1:
        df = load_curated_dataset(sys.argv[1])
        print(f"Loaded curated training set: {len(df)} rows from {sys.argv[1]}")
    else:
        print("No curated dataset provided -- using synthetic training set "
              "(replace with the hackathon-supplied catalog of known "
              "exoplanets / EBs / false positives for production use).")
        df = make_synthetic_training_set(n_per_class=400)

    pipe, metrics = train_classifier(df, model_out=MODEL_OUT)
    print("Training complete. Model saved to:", MODEL_OUT)


if __name__ == "__main__":
    main()
