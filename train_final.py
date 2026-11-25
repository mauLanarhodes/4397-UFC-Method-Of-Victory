"""
train_final.py — train and save the final model used by predict_multimodal().

Final model: average of CatBoost + LightGBM + XGBoost, each averaged over 5 seeds,
trained on all three modalities for every fight from 2010 to March 2021.
Saves models/final_model.joblib with the models, the feature column list, the
fitted text encoder and the training profiles.

Usage: python train_final.py     (about 5 minutes)
"""
import os
import warnings

import joblib
import pandas as pd

from compare_models import fit, make
from text_features import TextEncoder, fight_text_features, load_profiles
from train import CLASSES, SEEDS, TRAIN_FROM, history_features, tabular_features

warnings.filterwarnings("ignore")
MEMBERS = ["CatBoost", "LightGBM", "XGBoost (current)"]


def main():
    d = pd.read_csv("data/fights.csv", parse_dates=["date"])
    d = d[d["date"].dt.year >= TRAIN_FROM].reset_index(drop=True)
    y = d["label"].map({c: i for i, c in enumerate(CLASSES)}).values

    p = load_profiles()
    enc = TextEncoder().fit(p)
    X = pd.concat([tabular_features(d), history_features(d), fight_text_features(d, enc.transform(p))], axis=1)
    print(f"Training on {len(d)} fights (2010 to March 2021), {X.shape[1]} features")

    members = {}
    for name in MEMBERS:
        members[name] = [fit(make(name, s), X, y, name) for s in SEEDS]
        print(f"  trained {name} x {len(SEEDS)} seeds")

    raw_prof = pd.read_csv("data/fighter_profiles.csv").drop_duplicates("fighter_name").set_index("fighter_name")
    os.makedirs("models", exist_ok=True)
    joblib.dump({"members": members, "columns": list(X.columns), "encoder": enc,
                 "profiles": raw_prof[["wiki_text", "ufc_text"]], "classes": CLASSES,
                 "trained_on": "UFC fights 2010 to March 2021"}, "models/final_model.joblib", compress=3)
    print("Saved models/final_model.joblib")


if __name__ == "__main__":
    main()
