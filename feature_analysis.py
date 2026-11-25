"""
feature_analysis.py — which features helped, which didn't, and does pruning help?

1. Groups all ~320 features into readable families (physical, striking, grappling, ...).
2. Permutation importance on the 2019 validation set: shuffle one family at a time
   and measure how much macro F1 drops. Big drop = the model relies on it.
   ~0 or negative = irrelevant (or noise).
3. Pruning test: drop the families that didn't help on 2019, retrain, score on 2020–21.
   Families are chosen on validation only, so the test set stays untouched.

Usage: python feature_analysis.py
Writes results/family_importance.csv, results/feature_importance_all.csv, results/pruning.csv
"""
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from text_features import fight_text_features, fighter_text_features, load_profiles
from train import CLASSES, SEEDS, TRAIN_FROM, history_features, tabular_features, xgb

warnings.filterwarnings("ignore")
N_REPEATS = 10


def family(col):
    """Map a column name to a readable feature family."""
    c = col
    if "_txt_" in c or c.startswith("txt_"):
        if "kw_" in c or "txt_any_" in c:
            return "Text: martial arts keywords"
        if "tfidf" in c or "emb_" in c:
            return "Text: TF-IDF word summary"
        return "Text: profile exists / length"
    if "_h_" in c or c.startswith("diff_h_"):
        if any(k in c for k in ("win_ko", "win_sub", "win_dec", "loss_ko", "loss_sub", "finish")):
            return "History: how recent fights ended"
        return "History: recent form & activity"
    if c.startswith("weight_class") or c in ("title_bout", "no_of_rounds"):
        return "Context: weight class, rounds, title"
    if "Stance" in c:
        return "Physical: stance"
    if re.search(r"Height|Reach|Weight_lbs|_age$|diff_age", c):
        return "Physical: height, reach, weight, age"
    if "is_debut" in c or "n_debuts" in c:
        return "Debut flags"
    if re.search(r"win_by|wins$|losses$|draw$|streak|longest|total_rounds|total_title|total_time|_title_bouts", c):
        return "Record: wins, losses, streaks, win methods"
    if "opp_" in c:
        return "Opponent stats (what opponents did to them)"
    if re.search(r"TD|SUB_ATT|REV|CTRL|GROUND|CLINCH", c):
        return "Grappling averages (takedowns, subs, control)"
    if re.search(r"STR|KD|HEAD|BODY|LEG|DISTANCE", c):
        return "Striking averages (strikes, knockdowns)"
    return "Other"


def fit_models(X, y, tr):
    models = []
    for s in SEEDS:
        m = xgb()
        m.set_params(random_state=s)
        m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        models.append(m)
    return models


def ensemble_predict(models, X):
    return sum(m.predict_proba(X) for m in models).argmax(1)


def main():
    d = pd.read_csv("data/fights.csv", parse_dates=["date"])
    y = d["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    yr = d["date"].dt.year
    tr = ((yr >= TRAIN_FROM) & (yr <= 2018)).values
    va, te = (yr == 2019).values, (yr >= 2020).values

    X = pd.concat([tabular_features(d), history_features(d),
                   fight_text_features(d, fighter_text_features(load_profiles()))], axis=1)
    fam = pd.Series({c: family(c) for c in X.columns})
    print(f"{X.shape[1]} features in {fam.nunique()} families")
    print(fam.value_counts().to_string(), "\n")

    models = fit_models(X, y, tr)
    Xva, yva = X[va].reset_index(drop=True), y[va]
    base = f1_score(yva, ensemble_predict(models, Xva), average="macro")
    rng = np.random.default_rng(0)

    # --- family-level permutation importance (validation set) ---
    rows = []
    for f, cols in fam.groupby(fam).groups.items():
        drops = []
        for _ in range(N_REPEATS):
            Xp = Xva.copy()
            perm = rng.permutation(len(Xp))
            Xp[list(cols)] = Xp[list(cols)].values[perm]  # shuffle the whole family together
            drops.append(base - f1_score(yva, ensemble_predict(models, Xp), average="macro"))
        rows.append({"family": f, "n_features": len(cols),
                     "f1_drop_mean": round(np.mean(drops), 4), "f1_drop_sd": round(np.std(drops), 4)})
    fi = pd.DataFrame(rows).sort_values("f1_drop_mean", ascending=False)
    fi.to_csv("results/family_importance.csv", index=False)
    print(f"Validation macro F1 (all features): {base:.3f}")
    print(fi.to_string(index=False), "\n")

    # --- per-feature gain importance (how often the trees use each feature) ---
    gain = sum(pd.Series(m.feature_importances_, index=X.columns) for m in models) / len(models)
    allf = pd.DataFrame({"feature": X.columns, "family": fam.values, "gain": gain.values})
    allf = allf.sort_values("gain", ascending=False)
    allf.to_csv("results/feature_importance_all.csv", index=False)
    unused = allf[allf.gain == 0]
    print(f"Features never used by any tree: {len(unused)}")
    print(unused.family.value_counts().to_string(), "\n")
    print("Bottom 10 used features:\n", allf[allf.gain > 0].tail(10).to_string(index=False), "\n")

    # --- pruning test: drop families that didn't help on validation ---
    weak = fi.loc[fi.f1_drop_mean <= 0, "family"].tolist()
    keep = [c for c in X.columns if fam[c] not in weak]
    res = []
    for name, cols in (("all features", list(X.columns)), ("pruned (drop unhelpful families)", keep)):
        ms = fit_models(X[cols], y, tr)
        res.append({
            "model": name, "n_features": len(cols),
            "val_macro_f1": round(f1_score(y[va], ensemble_predict(ms, X.loc[va, cols]), average="macro"), 3),
            "test_macro_f1": round(f1_score(y[te], ensemble_predict(ms, X.loc[te, cols]), average="macro"), 3),
        })
    res = pd.DataFrame(res)
    res.to_csv("results/pruning.csv", index=False)
    print("Dropped families:", weak)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
