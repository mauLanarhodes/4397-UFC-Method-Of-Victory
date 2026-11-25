"""
evaluate_personal.py — run the final model on the 2025 personal test set.

Final model: average of CatBoost + LightGBM + XGBoost (each averaged over 5 seeds),
trained on every fight from 2010 to March 2021, then scored on the 20 fights from 2025.
Also runs the modality ablation on the personal set.

Usage: python evaluate_personal.py   (after build_personal_set.py)
Writes results/personal_ablation.csv and results/personal_predictions.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from compare_models import avg_proba
from text_features import fight_text_features, fighter_text_features, load_profiles
from train import CLASSES, TRAIN_FROM, context_features, history_features, tabular_features

warnings.filterwarnings("ignore")
MEMBERS = ["CatBoost", "LightGBM", "XGBoost (current)"]


def blocks(d, F):
    T, H, C = tabular_features(d), history_features(d), context_features(d)
    X_txt = fight_text_features(d, F)
    return {"all three modalities": pd.concat([T, H, X_txt], axis=1),
            "drop text": pd.concat([T, H], axis=1),
            "drop history": pd.concat([T, X_txt], axis=1),
            "drop tabular": pd.concat([H, X_txt, C], axis=1)}


def align(Xp, cols):
    """Give the 2025 rows exactly the training columns (missing one-hot columns = 0)."""
    out = Xp.reindex(columns=cols)
    dummy = [c for c in cols if c.startswith(("weight_class_", "R_Stance_", "B_Stance_"))]
    out[dummy] = out[dummy].fillna(0)
    return out.astype(float)


def ensemble(Xtr, ytr, Xte):
    return sum(avg_proba(m, Xtr, ytr, [Xte])[0] for m in MEMBERS) / len(MEMBERS)


def main():
    d = pd.read_csv("data/fights.csv", parse_dates=["date"])
    d = d[d["date"].dt.year >= TRAIN_FROM].reset_index(drop=True)
    p = pd.read_csv("personal_test_set/fights_2025.csv", parse_dates=["date"])
    y = d["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    yp = p["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    F = fighter_text_features(load_profiles())
    Btr, Bp = blocks(d, F), blocks(p.drop(columns=["sample_id"]), F)
    print(f"train: {len(d)} fights (2010 to March 2021) | personal set: {len(p)} fights (2025)")

    rows, P_all = [], None
    for name in Btr:
        Xtr = Btr[name]
        P = ensemble(Xtr, y, align(Bp[name], Xtr.columns))
        pred = P.argmax(1)
        rows.append({"model": f"Ensemble: {name}", "macro_f1": round(f1_score(yp, pred, average="macro"), 3),
                     "accuracy": round(accuracy_score(yp, pred), 3), "correct": f"{(pred == yp).sum()}/{len(yp)}"})
        print(rows[-1])
        if name == "all three modalities":
            P_all = P
    maj = np.zeros(len(yp), dtype=int)  # always "DEC"
    rows.append({"model": "Majority class (always Decision)", "macro_f1": round(f1_score(yp, maj, average="macro"), 3),
                 "accuracy": round(accuracy_score(yp, maj), 3), "correct": f"{(maj == yp).sum()}/{len(yp)}"})
    pd.DataFrame(rows).to_csv("results/personal_ablation.csv", index=False)

    pred = P_all.argmax(1)
    out = p[["sample_id", "R_fighter", "B_fighter", "weight_class", "label"]].copy()
    out["predicted"] = [CLASSES[i] for i in pred]
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = P_all[:, i].round(2)
    out["correct"] = out["predicted"] == out["label"]
    out.to_csv("results/personal_predictions.csv", index=False)
    print("\n", pd.DataFrame(rows).to_string(index=False))
    print("\nConfusion (rows = true DEC, KO/TKO, SUB):\n", confusion_matrix(yp, pred, labels=[0, 1, 2]))
    print("\n", out.to_string(index=False))


if __name__ == "__main__":
    main()
