"""Run from the repository root: python accuracy_experiment.py

Train through 2023, select on 2024 by accuracy (macro F1 breaks ties),
then evaluate the frozen choices on 2025. Output goes in results/accuracy/.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from accuracy_features import model_features
from build_personal_set import build
from compare_models import fit, make
from train import CLASSES, SEEDS

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/accuracy"
FAMILIES = ("CatBoost", "LightGBM", "XGBoost (current)")


def load_rows():
    raw = pd.read_csv(ROOT / "raw_fights_greco.csv", sep=";")
    dates = pd.to_datetime(raw.date, format="mixed")
    return build(raw.loc[dates.dt.year <= 2025], pd.read_csv(ROOT / "fighter_details_greco.csv"))


def metrics(y, pred):
    return dict(n=len(y), correct=int((y == pred).sum()), accuracy=accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, labels=range(3), average="macro", zero_division=0),
                submission_recall=recall_score(y, pred, labels=[2], average=None, zero_division=0)[0])


def train_one(family, seed, balanced, X, y):
    model = make(family, seed)
    if family == "CatBoost":
        model.set_params(allow_writing_files=False, auto_class_weights="Balanced" if balanced else "None")
    return fit(model, X, y, family)


def main(rows):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    d = rows.loc[rows.date.dt.year.between(2010, 2025) & rows.label.isin(CLASSES)].copy()
    data = {name: d.loc[mask] for name, mask in {
        "train": d.date.dt.year < 2024, "validation": d.date.dt.year == 2024,
        "test": d.date.dt.year == 2025}.items()}
    y = {k: v.label.map(CLASSES.index).to_numpy() for k, v in data.items()}
    majority = int(np.bincount(y["train"]).argmax())
    features = {}
    for kind in ("stats", "matchups"):
        X = model_features(data["train"], kind)
        features[kind] = {"train": X, **{k: model_features(data[k], kind, X.columns)
                                         for k in ("validation", "test")}}
    # Fit each shared member once. All candidates and the tie rule are fixed in advance.
    jobs = [(family, seed, True, "stats") for family in FAMILIES for seed in SEEDS]
    jobs += [("CatBoost", 1, False, "stats"), ("CatBoost", 1, True, "matchups"),
             ("CatBoost", 1, False, "matchups")]

    def worker(job):
        family, seed, balanced, kind = job
        return train_one(family, seed, balanced, features[kind]["train"], y["train"])

    models = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for job, model in zip(jobs, pool.map(worker, jobs)):
            models[job] = model
            print("Fit", job, flush=True)
    ensemble = "Balanced three-model ensemble (5 seeds)"
    candidates = {ensemble: jobs[:len(FAMILIES)*len(SEEDS)]}
    for kind in ("stats", "matchups"):
        for balanced in (True, False):
            name = f"CatBoost {'balanced' if balanced else 'unweighted'}, {kind}"
            candidates[name] = [("CatBoost", 1, balanced, kind)]
    baseline = "Always " + CLASSES[majority]

    def probabilities(keys, split):
        return np.mean([models[key].predict_proba(features[key[3]][split]) for key in keys], axis=0)

    def predictions(split):
        pred = {name: probabilities(keys, split).argmax(1) for name, keys in candidates.items()}
        pred[baseline] = np.full(len(y[split]), majority)
        return pred

    validation = predictions("validation")
    winner = max(validation, key=lambda name: (accuracy_score(y["validation"], validation[name]),
                                                f1_score(y["validation"], validation[name], average="macro")))
    print("Selected on 2024:", winner, flush=True)
    if winner != baseline:
        keys = candidates[winner]
        kind = keys[0][3]
        bundle = {"members": {family: [models[key] for key in keys if key[0] == family]
                              for family in sorted({key[0] for key in keys})},
                  "feature_set": kind, "columns": list(features[kind]["train"].columns),
                  "classes": CLASSES, "trained_on": "2010-2023", "selected_model": winner}
        joblib.dump(bundle, OUTPUT / "selected_model.joblib", compress=3)
    test = predictions("test")
    records = []
    for split, preds in (("validation", validation), ("test", test)):
        for name, pred in preds.items():
            records.append(dict(split=split, model=name, selected=(name == winner), **metrics(y[split], pred)))
    table = pd.DataFrame(records)
    table.to_csv(OUTPUT / "scores.csv", index=False)
    out = data["test"][["date", "R_fighter", "B_fighter", "label"]].copy()
    for name, pred in test.items():
        out[name] = np.array(CLASSES)[pred]
    if winner != baseline:
        P = probabilities(candidates[winner], "test")
        for k, label in enumerate(CLASSES):
            out[f"p_{label}"] = P[:, k]
    out.to_csv(OUTPUT / "predictions_2025.csv", index=False)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-rows", type=Path, help="optional trusted local feature cache")
    args = parser.parse_args()
    main(pd.read_pickle(args.cached_rows) if args.cached_rows else load_rows())
