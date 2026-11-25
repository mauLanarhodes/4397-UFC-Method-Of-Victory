"""
build_personal_set.py — turn scraped UFCStats data into model-ready rows for the 2025 fights.

Inputs (made by scrape_ufcstats.py, in personal_test_set/):
  raw_fights_2025.csv, fighter_details_2025.csv, target_fights.csv

Steps:
  1. Pre-fight stats with ufc_features.py: the same math as the Kaggle data.csv
     (verified to match it on all 136 columns x 6,012 fights).
  2. Fight-history features with the same function build_dataset.py uses.
  3. Keep only the 20 target fights; add weight class, title bout, scheduled rounds.

Output: personal_test_set/fights_2025.csv (same columns as data/fights.csv)

Usage: python build_personal_set.py
"""
import re

import pandas as pd

from build_dataset import history_features, method_group
from ufc_features import fighter_features, prep_raw

P = "personal_test_set/"
WEIGHT_CLASSES = [  # order matters: "Women's ..." and "Light Heavyweight" before shorter names
    ("Women's Strawweight", "WomenStrawweight"), ("Women's Flyweight", "WomenFlyweight"),
    ("Women's Bantamweight", "WomenBantamweight"), ("Women's Featherweight", "WomenFeatherweight"),
    ("Light Heavyweight", "LightHeavyweight"), ("Heavyweight", "Heavyweight"),
    ("Middleweight", "Middleweight"), ("Welterweight", "Welterweight"),
    ("Lightweight", "Lightweight"), ("Featherweight", "Featherweight"),
    ("Bantamweight", "Bantamweight"), ("Flyweight", "Flyweight"), ("Open Weight", "OpenWeight"),
]


def weight_class(fight_type):
    for name, code in WEIGHT_CLASSES:
        if name in fight_type:
            return code
    return "CatchWeight"


def build(raw, details, targets=None):
    raw = raw.copy()
    raw["_date"] = pd.to_datetime(raw["date"], format="mixed")
    raw = raw.sort_values("_date", ascending=False, kind="stable").drop(columns="_date").reset_index(drop=True)
    f = prep_raw(raw)
    feats = fighter_features(f, details)

    d = pd.concat([f[["R_fighter", "B_fighter", "date", "title_bout", "Referee", "location"]], feats], axis=1)
    d["weight_class"] = raw["Fight_type"].map(weight_class)
    d["no_of_rounds"] = raw["Format"].str.extract(r"^(\d+)")[0].astype(float)
    d["win_by"] = raw["win_by"]
    d["last_round"] = raw["last_round"].astype(float)
    d["raw_winner"] = raw["Winner"].where(raw["Winner"].notna() & (raw["Winner"] != ""))
    d = d.join(history_features(d))

    if targets is None:
        d["label"] = d["win_by"].map(method_group).where(d["raw_winner"].notna())
        return d.drop(columns="raw_winner")

    out = []
    for _, t in targets.iterrows():
        hit = d[(d.R_fighter == t.R_fighter) & (d.B_fighter == t.B_fighter) &
                (d.date == pd.to_datetime(t.date, format="mixed"))]
        if len(hit) != 1:
            print(f"  {t.sample_id}: found {len(hit)} matching rows, skipping")
            continue
        row = hit.iloc[0].copy()
        row["sample_id"] = t.sample_id
        out.append(row)
    res = pd.DataFrame(out)
    res["label"] = res["win_by"].map(method_group)
    return res.drop(columns=["raw_winner"])


def main():
    raw = pd.read_csv(P + "raw_fights_2025.csv", sep=";")
    details = pd.read_csv(P + "fighter_details_2025.csv")
    targets = pd.read_csv(P + "target_fights.csv")
    res = build(raw, details, targets)
    labels = pd.read_csv(P + "labels.csv").set_index("sample_id")
    agree = (res.set_index("sample_id")["label"] == labels.loc[res.sample_id, "label"]).mean()
    print(f"Built {len(res)} of {len(targets)} fights; labels agree with labels.csv: {agree:.0%}")
    res.to_csv(P + "fights_2025.csv", index=False)
    print("Saved", P + "fights_2025.csv")


if __name__ == "__main__":
    main()
