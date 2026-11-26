"""Train the direct and two-stage models; save the validation winner and test picks."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from accuracy_features import model_features
from build_personal_set import build
from fight_methods import MethodModel, fight_features
from train import CLASSES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/method_rates"


def load_rows():
    raw = pd.read_csv(ROOT / "raw_fights_greco.csv", sep=";")
    details = pd.read_csv(ROOT / "fighter_details_greco.csv")
    return build(raw, details)


def performance(y, pred):
    return {"n": len(y), "correct": int(np.sum(y == pred)), "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"), "sub_caught": int(np.sum((y == 2) & (pred == 2))),
            "sub_actual": int(np.sum(y == 2)), "sub_picks": int(np.sum(pred == 2))}


def main(rows, output_dir=OUT):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = rows[rows.label.isin(CLASSES) & rows.date.dt.year.between(2010, 2026)]
    train = data[data.date.dt.year <= 2023]
    val = data[data.date.dt.year == 2024]
    X = fight_features(train)
    Xv = fight_features(val).reindex(columns=X.columns, fill_value=0)
    yt, yv = (d.label.map(CLASSES.index) for d in (train, val))
    models, weights, scores, validation = {}, {}, [], {}
    for name, split in (("Direct", False), ("Two-stage", True)):
        m = MethodModel(two_stage=split).fit(X, yt)
        p = m.predict_proba(Xv)
        options = []
        for alpha in (0, .25, .5, .75, 1):
            w = m.frequency ** -alpha
            options.append((w, performance(yv, (p*w).argmax(1))))
        floor = max(s["accuracy"] for _, s in options) - .02
        w, result = max((o for o in options if o[1]["accuracy"] >= floor), key=lambda o: o[1]["macro_f1"])
        models[name], weights[name], validation[name] = m, w, result
        scores.append({"year": 2024, "model": name, **result})
    winner = max(models, key=lambda name: validation[name]["macro_f1"])
    joblib.dump({"members": {winner: [models[winner]]}, "feature_set": "method_rates",
                 "columns": list(X.columns), "decision_weights": weights[winner], "classes": CLASSES,
                 "trained_on": "2010–2023", "selected_model": winner}, output_dir / "selected_model.joblib", compress=3)
    pr = joblib.load(ROOT / "results/accuracy/selected_model.joblib")
    for year in (2025, 2026):
        test = data[data.date.dt.year == year]
        Xtest = fight_features(test).reindex(columns=X.columns, fill_value=0)
        old_X = model_features(test, pr["feature_set"], pr["columns"])
        picks = {"PR CatBoost": pr["members"]["CatBoost"][0].predict_proba(old_X).argmax(1),
                 "Always decision": np.zeros(len(test), dtype=int)}
        out = test[["date", "R_fighter", "B_fighter", "label"]].copy()
        for name, model in models.items():
            prob = model.predict_proba(Xtest)
            picks[name] = (prob * weights[name]).argmax(1)
            if name == winner:
                for i, cls in enumerate(CLASSES):
                    out[f"p_{cls}"] = prob[:, i]
        y = test.label.map(CLASSES.index)
        for name, pred in picks.items():
            scores.append({"year": year, "model": name, **performance(y, pred)})
            out[name] = np.asarray(CLASSES)[pred]
        out.to_csv(output_dir / f"picks_{year}.csv", index=False)
    results = pd.DataFrame(scores)
    results.to_csv(output_dir / "scores.csv", index=False)
    print("Selected on 2024:", winner)
    print(results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-rows", type=Path, help="optional trusted cache from build(raw, details)")
    args = parser.parse_args()
    main(pd.read_pickle(args.cached_rows) if args.cached_rows else load_rows())
