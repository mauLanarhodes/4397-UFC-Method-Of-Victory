"""
compare_models.py — try other models and model combinations.

Same data and split as train.py: train 2010–2018, validate 2019, test 2020–2021.
All choices (tuning, ensemble members, class weights) are made on 2019 only;
2020–21 is scored once at the end.

Compared:
  A. Single models on all three modalities (early fusion):
     logistic regression, random forest, extra trees, HistGradientBoosting,
     LightGBM, CatBoost, XGBoost (current), XGBoost tuned, small neural net (MLP)
  B. Late fusion: one XGBoost per modality, then
     (a) average their probabilities, (b) a stacked logistic-regression combiner
  C. Ensembles of the best single models (chosen on 2019)
  D. Class-weight tuning: nudge predicted probabilities toward the rarer classes
     (submissions), weights picked on 2019

Usage: python compare_models.py
Writes results/model_comparison.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from text_features import fight_text_features, fighter_text_features, load_profiles
from train import CLASSES, SEEDS, TRAIN_FROM, context_features, history_features, tabular_features, xgb

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------- model zoo
def make(name, seed):
    if name == "Logistic regression":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=4000, C=0.05, class_weight="balanced"))
    if name == "Random forest":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, max_features="sqrt", class_weight="balanced",
            n_jobs=2, random_state=seed))
    if name == "Extra trees":
        return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=5, max_features="sqrt", class_weight="balanced",
            n_jobs=2, random_state=seed))
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_depth=4,
                                              l2_regularization=1.0, class_weight="balanced",
                                              random_state=seed)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.6, min_child_samples=20,
                              class_weight="balanced", random_state=seed, verbose=-1, n_jobs=2)
    if name == "CatBoost":
        return CatBoostClassifier(iterations=600, learning_rate=0.05, depth=5, l2_leaf_reg=5,
                                  auto_class_weights="Balanced", random_seed=seed, verbose=0,
                                  thread_count=2)
    if name == "XGBoost (current)":
        m = xgb()
        m.set_params(random_state=seed)
        return m
    if name == "Neural net (MLP)":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64,), alpha=0.01, early_stopping=True, max_iter=500,
            random_state=seed))
    raise ValueError(name)


NEEDS_WEIGHTS = {"XGBoost (current)", "XGBoost tuned", "Neural net (MLP)"}


def fit(model, X, y, name):
    if name in NEEDS_WEIGHTS and name != "Neural net (MLP)":
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    elif name == "Neural net (MLP)":
        # MLPClassifier has no sample weights: balance classes by oversampling
        rng = np.random.default_rng(0)
        idx = np.concatenate([rng.choice(np.where(y == c)[0], (y == 0).sum(), replace=True)
                              for c in np.unique(y)])
        model.fit(X.iloc[idx] if hasattr(X, "iloc") else X[idx], y[idx])
    else:
        model.fit(X, y)
    return model


def avg_proba(name, Xtr, ytr, Xs, maker=make):
    """Average predicted probabilities over seeds (deterministic models run once)."""
    seeds = [SEEDS[0]] if name == "Logistic regression" else SEEDS
    out = [0 for _ in Xs]
    for s in seeds:
        m = fit(maker(name, s), Xtr, ytr, name)
        for i, X in enumerate(Xs):
            out[i] = out[i] + m.predict_proba(X)
    return [p / len(seeds) for p in out]


def score(y, P):
    p = P.argmax(1)
    return round(f1_score(y, p, average="macro"), 3), round(accuracy_score(y, p), 3)


def main():
    d = pd.read_csv("data/fights.csv", parse_dates=["date"])
    y = d["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    yr = d["date"].dt.year
    tr = ((yr >= TRAIN_FROM) & (yr <= 2018)).values
    va, te = (yr == 2019).values, (yr >= 2020).values

    T, H, C = tabular_features(d), history_features(d), context_features(d)
    Xt = fight_text_features(d, fighter_text_features(load_profiles()))
    X = pd.concat([T, H, Xt], axis=1)
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    rows, probs = [], {}

    def record(name, group, Pva, Pte):
        probs[name] = (Pva, Pte)
        v, va_acc = score(y[va], Pva)
        t, te_acc = score(y[te], Pte)
        rows.append({"group": group, "model": name, "val_macro_f1": v,
                     "test_macro_f1": t, "test_accuracy": te_acc})
        print(f"{group:<22} {name:<42} val {v:.3f}  test {t:.3f}")

    # ---------- A. single models ----------
    singles = ["Logistic regression", "Random forest", "Extra trees", "HistGradientBoosting",
               "LightGBM", "CatBoost", "XGBoost (current)", "Neural net (MLP)"]
    for name in singles:
        Pva, Pte = avg_proba(name, Xtr, y[tr], [Xva, Xte])
        record(name, "A. single model", Pva, Pte)

    # XGBoost tuned: small grid, picked on 2019
    grid = list(itertools.product([3, 4, 6], [0.02, 0.05], [300, 600], [0.4, 0.8]))
    best, best_v = None, -1
    for depth, lr, n, col in grid:
        m = XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=lr, subsample=0.8,
                          colsample_bytree=col, min_child_weight=5, reg_lambda=2.0,
                          objective="multi:softprob", eval_metric="mlogloss",
                          random_state=SEEDS[0], n_jobs=2)
        m.fit(Xtr, y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        v = f1_score(y[va], m.predict(Xva), average="macro")
        if v > best_v:
            best, best_v = (depth, lr, n, col), v
    depth, lr, n, col = best
    print("XGBoost tuned params (depth, lr, trees, colsample):", best)

    def tuned_maker(name, s):
        return XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=lr, subsample=0.8,
                             colsample_bytree=col, min_child_weight=5, reg_lambda=2.0,
                             objective="multi:softprob", eval_metric="mlogloss",
                             random_state=s, n_jobs=2)
    Pva, Pte = avg_proba("XGBoost tuned", Xtr, y[tr], [Xva, Xte], maker=tuned_maker)
    record(f"XGBoost tuned {best}", "A. single model", Pva, Pte)

    # ---------- B. late fusion ----------
    blocks = {"tabular": T, "history": pd.concat([H, C], axis=1), "text": pd.concat([Xt, C], axis=1)}
    per_va, per_te, oof = {}, {}, {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    idx_tr = np.where(tr)[0]
    for b, Xb in blocks.items():
        per_va[b], per_te[b] = avg_proba("XGBoost (current)", Xb[tr], y[tr], [Xb[va], Xb[te]])
        o = np.zeros((tr.sum(), 3))
        for k, (a, bb) in enumerate(skf.split(idx_tr, y[tr])):
            m = make("XGBoost (current)", k)
            m.fit(Xb[tr].iloc[a], y[tr][a], sample_weight=compute_sample_weight("balanced", y[tr][a]))
            o[bb] = m.predict_proba(Xb[tr].iloc[bb])
        oof[b] = o
    record("Late fusion: average of 3 modality models", "B. late fusion",
           sum(per_va.values()) / 3, sum(per_te.values()) / 3)
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    meta.fit(np.hstack([oof[b] for b in blocks]), y[tr])
    record("Late fusion: stacked combiner", "B. late fusion",
           meta.predict_proba(np.hstack([per_va[b] for b in blocks])),
           meta.predict_proba(np.hstack([per_te[b] for b in blocks])))

    # ---------- C. ensembles of top single models (picked on 2019) ----------
    single_rows = [r for r in rows if r["group"] == "A. single model"]
    ranked = [r["model"] for r in sorted(single_rows, key=lambda r: -r["val_macro_f1"])]
    for k in (2, 3, 4):
        members = ranked[:k]
        Pva = sum(probs[m][0] for m in members) / k
        Pte = sum(probs[m][1] for m in members) / k
        record(f"Average of top {k} on 2019: " + " + ".join(m.split(" (")[0] for m in members),
               "C. ensemble", Pva, Pte)

    # ---------- D. class-weight tuning on the best 2019 model so far ----------
    best_name = max(rows, key=lambda r: r["val_macro_f1"])["model"]
    Pva, Pte = probs[best_name]
    bw, bv = (1, 1, 1), -1
    for wk in np.arange(0.7, 1.41, 0.05):
        for ws in np.arange(0.7, 1.81, 0.05):
            w = np.array([1.0, wk, ws])
            v = f1_score(y[va], (Pva * w).argmax(1), average="macro")
            if v > bv:
                bw, bv = w, v
    record(f"{best_name} + class weights {tuple(np.round(bw, 2))}", "D. class weights",
           Pva * bw, Pte * bw)

    res = pd.DataFrame(rows)
    res.to_csv("results/model_comparison.csv", index=False)
    print("\n", res.to_string(index=False))


if __name__ == "__main__":
    main()
