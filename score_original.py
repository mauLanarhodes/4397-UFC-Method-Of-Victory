"""Score the published model on the same 2025 bouts as accuracy_experiment.py.

The published model requires pandas 3.x and scikit-learn 1.9.1 to deserialize.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from accuracy_experiment import ROOT, OUTPUT, load_rows, metrics
from predict import legacy_features
from train import CLASSES


def main(rows):
    data = rows.loc[(rows.date.dt.year == 2025) & rows.label.isin(CLASSES)].copy()
    bundle = joblib.load(ROOT / "final_model.joblib")
    # Use the published encoder, profiles and 15 fitted estimators.
    X = legacy_features(data, None, bundle)
    prob = np.mean([np.mean([m.predict_proba(X) for m in members], axis=0)
                    for members in bundle["members"].values()], axis=0)
    pred = prob.argmax(axis=1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = data[["date", "R_fighter", "B_fighter", "label"]].copy()
    out["original"] = np.array(CLASSES)[pred]
    out.to_csv(OUTPUT / "original_2025.csv", index=False)
    print(metrics(data.label.map(CLASSES.index).to_numpy(), pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-rows", help="optional trusted local feature cache")
    args = parser.parse_args()
    main(pd.read_pickle(args.cached_rows) if args.cached_rows else load_rows())
