"""
train.py — baselines + modality ablation for UFC method-of-victory.

Split by date (no training on the future):
  train: 2010–2018 | val: 2019 | test: 2020–2021
  (2010 start chosen by the era experiment: early-UFC fights end very differently)
XGBoost scores are the mean of 5 random seeds.

Usage: python train.py            (after build_dataset.py)
       python train.py --era      (also rerun the training-start-year experiment)
Writes results/ablation.csv and results/confusion_*.png
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

CLASSES = ["DEC", "KO/TKO", "SUB"]
SEED = 42
SEEDS = (1, 2, 3, 4, 5)
TRAIN_FROM = 2010
os.makedirs("results", exist_ok=True)


# ----------------------------------------------------------------- features
def tabular_features(d):
    """Pre-fight stats from data.csv + matchup differences (red minus blue)."""
    skip = re.compile(r"_h_|^(R|B)_fighter$|Referee|location|Winner|win_by|last_round|date|label")
    cols = [c for c in d.columns if not skip.search(c)]
    X = d[cols].copy()
    X = pd.get_dummies(X, columns=["weight_class", "R_Stance", "B_Stance"], dtype=float)
    X["title_bout"] = X["title_bout"].astype(float)
    # debut flags: no UFC fights yet, so averages are blank because the fighter is NEW, not average
    for c in ("R", "B"):
        X[f"{c}_is_debut"] = d[f"{c}_avg_SIG_STR_landed"].isna().astype(float)
    X["n_debuts"] = X["R_is_debut"] + X["B_is_debut"]
    # matchup differences for every stat both corners have
    for c in [c for c in X.columns if c.startswith("R_")]:
        b = "B_" + c[2:]
        if b in X.columns:
            X["diff_" + c[2:]] = X[c] - X[b]
    return X.astype(float)


def history_features(d):
    cols = [c for c in d.columns if "_h_" in c]
    X = d[cols].copy()
    for c in [c for c in cols if c.startswith("R_")]:
        X["diff" + c[1:]] = X[c] - X["B" + c[1:]]
    return X.astype(float)


def context_features(d):
    """Fight context every model gets: weight class, title bout, scheduled rounds."""
    X = pd.get_dummies(d[["weight_class"]], dtype=float)
    X["title_bout"] = d["title_bout"].astype(float)
    X["no_of_rounds"] = d["no_of_rounds"].astype(float)
    return X


# ----------------------------------------------------------------- models
def xgb():
    return XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.6, min_child_weight=5, reg_lambda=2.0,
        objective="multi:softprob", eval_metric="mlogloss", random_state=SEED, n_jobs=2,
    )


def logreg():
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1),
    )


def evaluate(name, model, Xtr, ytr, Xs, weighted=True):
    if weighted and not isinstance(model, DummyClassifier):
        w = compute_sample_weight("balanced", ytr)
        step = model.steps[-1][0] + "__sample_weight" if hasattr(model, "steps") else "sample_weight"
        model.fit(Xtr, ytr, **{step: w})
    else:
        model.fit(Xtr, ytr)
    row = {"model": name}
    preds = {}
    for split, (X, y) in Xs.items():
        p = model.predict(X)
        preds[split] = p
        row[f"{split}_macro_f1"] = round(f1_score(y, p, average="macro"), 3)
        row[f"{split}_accuracy"] = round(accuracy_score(y, p), 3)
    return row, model, preds


def era_experiment(d, y, yr, va, te, seeds=(1, 2, 3, 4, 5)):
    """Does dropping early-UFC fights help on recent fights?
    Same model (stats + history), trained from different start years.
    Averaged over 5 random seeds so small differences aren't just luck."""
    X = pd.concat([tabular_features(d), history_features(d)], axis=1)
    rows = []
    for start in (1994, 2001, 2005, 2010):
        tr = ((yr >= start) & (yr <= 2018)).values
        scores = {"val": [], "test": []}
        for s in seeds:
            m = xgb()
            m.set_params(random_state=s)
            m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
            for split, mask in (("val", va), ("test", te)):
                scores[split].append(f1_score(y[mask], m.predict(X[mask]), average="macro"))
        rows.append({
            "train_from": start, "train_fights": int(tr.sum()),
            "val_macro_f1": round(np.mean(scores["val"]), 3),
            "test_macro_f1": round(np.mean(scores["test"]), 3),
            "test_sd": round(np.std(scores["test"]), 3),
        })
    res = pd.DataFrame(rows)
    res.to_csv("results/era_experiment.csv", index=False)
    print("\nEra experiment (stats + history, mean of 5 seeds):")
    print(res.to_string(index=False), "\n")


def main():
    d = pd.read_csv("data/fights.csv", parse_dates=["date"])
    y = d["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    yr = d["date"].dt.year
    va, te = (yr == 2019).values, (yr >= 2020).values
    if "--era" in sys.argv:
        era_experiment(d, y, yr, va, te)
    tr = ((yr >= TRAIN_FROM) & (yr <= 2018)).values
    print(f"train {tr.sum()} ({TRAIN_FROM}-2018) | val {va.sum()} | test {te.sum()}")

    T, H, C = tabular_features(d), history_features(d), context_features(d)
    if os.path.exists("data/fighter_profiles.csv"):
        from text_features import (coverage_report, fight_text_features,
                                   fighter_text_features, load_profiles)
        F = fighter_text_features(load_profiles())
        X_txt = fight_text_features(d, F)
        cov = coverage_report(d[tr | va | te], F)
        print("text coverage (2010+ fights):", cov)
        pd.Series(cov).to_csv("results/text_coverage.csv", header=["value"])
        blocks = {
            "all three modalities": pd.concat([T, H, X_txt], axis=1),
            "drop text (tabular + history)": pd.concat([T, H], axis=1),
            "drop history (tabular + text)": pd.concat([T, X_txt], axis=1),
            "drop tabular (history + text)": pd.concat([H, X_txt, C], axis=1),
            "text only": pd.concat([X_txt, C], axis=1),
            "context only": C,
        }
    else:
        blocks = {
            "tabular + history (all available)": pd.concat([T, H], axis=1),
            "tabular only (drop history)": T,
            "history only (drop tabular)": pd.concat([H, C], axis=1),
            "context only": C,
        }
    best = next(iter(blocks))

    def splits(X):
        return {"val": (X[va], y[va]), "test": (X[te], y[te])}

    rows = []
    r, _, _ = evaluate("Majority class", DummyClassifier(strategy="most_frequent"),
                       T[tr], y[tr], splits(T), weighted=False)
    rows.append(r)
    r, _, _ = evaluate("Logistic regression (tabular)", logreg(), T[tr], y[tr], splits(T))
    rows.append(r)
    for name, X in blocks.items():
        runs = []
        for s in SEEDS:
            m = xgb()
            m.set_params(random_state=s)
            r, m, preds = evaluate(f"XGBoost: {name}", m, X[tr], y[tr], splits(X))
            runs.append(r)
            if name == best and s == SEEDS[0]:
                cm = confusion_matrix(y[te], preds["test"])
                pd.DataFrame(cm, index=[f"true {c}" for c in CLASSES],
                             columns=[f"pred {c}" for c in CLASSES]).to_csv("results/confusion_test.csv")
                imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
                imp.head(20).to_csv("results/top_features.csv", header=["importance"])
        avg = pd.DataFrame(runs).drop(columns="model").mean().round(3).to_dict()
        avg["test_macro_f1_sd"] = round(float(np.std([x["test_macro_f1"] for x in runs])), 3)
        rows.append({"model": f"XGBoost: {name}", **avg})

    res = pd.DataFrame(rows)
    res.to_csv("results/ablation.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
