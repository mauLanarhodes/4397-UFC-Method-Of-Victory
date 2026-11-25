"""
predict.py — the project's entry point: predict_multimodal().

    from predict import predict_multimodal
    preds = predict_multimodal("personal_test_set/upload")

Give it a folder ("upload") with:
  fights.csv           the fights to predict: sample_id, date, R_fighter, B_fighter,
                       weight_class, title_bout, no_of_rounds
  raw_fights.csv       UFC fight history for these fighters, Kaggle raw format (';'-separated)
  fighter_details.csv  fighter_name, Height, Weight, Reach, Stance, DOB
  fighter_profiles.csv (optional) text profiles; fighters without one get "No profile available"

It returns one row per fight with the predicted method (KO/TKO, SUB, DEC) and the
probability of each.

How it works, for each fight:
  1. TABULAR  : pre-fight stats from the fighters' earlier fights (same math as Kaggle data.csv)
  2. HISTORY  : each fighter's last 5 fights (results, methods, rounds, layoff)
  3. TEXT     : martial-arts keywords + TF-IDF summary of each fighter's profile
  4. One row per fight -> CatBoost, LightGBM and XGBoost (5 seeds each) -> average probabilities

Only fights BEFORE each fight's date are used. If the fight itself is in raw_fights.csv,
it is replaced by a blank placeholder, so its result can never leak into the prediction.

The trained model lives in models/final_model.joblib (made by train_final.py).
"""
import os

import joblib
import numpy as np
import pandas as pd

from build_personal_set import build
from text_features import clean_profiles, fight_text_features
from train import CLASSES, history_features, tabular_features

MODEL_PATH = "models/final_model.joblib"
WC_NAMES = {"WomenStrawweight": "Women's Strawweight", "WomenFlyweight": "Women's Flyweight",
            "WomenBantamweight": "Women's Bantamweight", "WomenFeatherweight": "Women's Featherweight",
            "LightHeavyweight": "Light Heavyweight", "CatchWeight": "Catch Weight", "OpenWeight": "Open Weight"}
STAT_COLS = ["KD", "SIG_STR.", "SIG_STR_pct", "TOTAL_STR.", "TD", "TD_pct", "SUB_ATT", "REV", "CTRL",
             "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"]
BLANK = {"KD": 0, "SIG_STR_pct": "---", "TD_pct": "---", "SUB_ATT": 0, "REV": 0, "CTRL": "0:00"}


def _placeholder(f):
    """A blank row for the fight being predicted: right date, corners, weight class, rounds.
    Its stats are empty and it has no result, so nothing about the outcome is used."""
    wc = WC_NAMES.get(f.weight_class, f.weight_class)
    fight_type = f"UFC {wc} Title Bout" if bool(f.title_bout) else f"{wc} Bout"
    rounds = int(f.no_of_rounds)
    row = {"R_fighter": f.R_fighter, "B_fighter": f.B_fighter, "win_by": "", "last_round": 1,
           "last_round_time": "0:00", "Referee": "", "date": f.date, "location": "",
           "Format": f"{rounds} Rnd ({'-'.join(['5'] * rounds)})", "Fight_type": fight_type, "Winner": np.nan}
    for c in STAT_COLS:
        row["R_" + c] = row["B_" + c] = BLANK.get(c, "0 of 0")
    return row


def build_features(fights, raw, details, profiles, bundle):
    """Model-ready feature matrix for the fights, with exactly the training columns."""
    fights = fights.copy()
    fights["date"] = pd.to_datetime(fights["date"], format="mixed")
    raw = raw.copy()
    raw["_d"] = pd.to_datetime(raw["date"], format="mixed")
    # drop the target fights themselves (if present) and add blank placeholders instead
    for _, f in fights.iterrows():
        same = (raw._d == f.date) & (((raw.R_fighter == f.R_fighter) & (raw.B_fighter == f.B_fighter)) |
                                     ((raw.R_fighter == f.B_fighter) & (raw.B_fighter == f.R_fighter)))
        raw = raw[~same]
    raw = pd.concat([raw.drop(columns="_d"), pd.DataFrame([_placeholder(f) for _, f in fights.iterrows()])],
                    ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], format="mixed").dt.strftime("%B %d, %Y")

    targets = fights[["sample_id", "R_fighter", "B_fighter"]].assign(date=fights["date"].dt.strftime("%B %d, %Y"))
    d = build(raw, details, targets)
    d = d.set_index("sample_id").loc[fights["sample_id"]].reset_index()

    # text: training profiles, overridden by any uploaded profiles, encoded with the saved encoder
    p = bundle["profiles"]
    if profiles is not None:
        p = pd.concat([p[~p.index.isin(profiles["fighter_name"])].reset_index(),
                       profiles]).reset_index(drop=True)
        p = clean_profiles(p)
    else:
        p = clean_profiles(p.reset_index())
    names = set(d.R_fighter) | set(d.B_fighter)
    missing = sorted(names - set(p.index))
    if missing:
        extra = pd.DataFrame({"fighter_name": missing, "wiki_text": "", "ufc_text": ""})
        p = clean_profiles(pd.concat([p.reset_index()[["fighter_name", "wiki_text", "ufc_text"]], extra]))
    F = bundle["encoder"].transform(p)

    X = pd.concat([tabular_features(d.drop(columns=["sample_id"])), history_features(d),
                   fight_text_features(d, F)], axis=1)
    X = X.loc[:, ~X.columns.duplicated()].reindex(columns=bundle["columns"])
    dummies = [c for c in bundle["columns"] if c.startswith(("weight_class_", "R_Stance_", "B_Stance_"))]
    X[dummies] = X[dummies].fillna(0)
    return d, X.astype(float)


def predict_multimodal(upload_dir, model_path=MODEL_PATH):
    """Predict how each fight in upload_dir/fights.csv ends. Returns a DataFrame."""
    bundle = joblib.load(model_path)
    fights = pd.read_csv(os.path.join(upload_dir, "fights.csv"))
    raw = pd.read_csv(os.path.join(upload_dir, "raw_fights.csv"), sep=";")
    details = pd.read_csv(os.path.join(upload_dir, "fighter_details.csv"))
    prof_path = os.path.join(upload_dir, "fighter_profiles.csv")
    profiles = pd.read_csv(prof_path) if os.path.exists(prof_path) else None

    d, X = build_features(fights, raw, details, profiles, bundle)
    P = 0
    for name, models in bundle["members"].items():
        P = P + sum(m.predict_proba(X) for m in models) / len(models)
    P = P / len(bundle["members"])

    out = fights[["sample_id", "R_fighter", "B_fighter"]].copy()
    out["predicted"] = [CLASSES[i] for i in P.argmax(1)]
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = P[:, i].round(3)
    return out
