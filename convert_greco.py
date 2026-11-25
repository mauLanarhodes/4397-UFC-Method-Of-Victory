"""
convert_greco.py — convert published UFCStats data into the Kaggle raw format.

UFCStats now blocks plain scripts with a browser check, so instead of scraping we use
the public, regularly updated UFCStats export at github.com/Greco1899/scrape_ufc_stats:
  ufc_event_details.csv, ufc_fight_results.csv, ufc_fight_stats.csv (per round),
  ufc_fighter_tott.csv (height, weight, reach, stance, DOB)

This script turns them into the same columns as raw_total_fight_data.csv and
raw_fighter_details.csv, so ufc_features.py can compute pre-fight stats exactly like
the training data. Per-round stats are summed into fight totals.

Usage:
  python convert_greco.py --src <folder with the 4 CSVs>
  (or with no --src: downloads them from GitHub)

Outputs (personal_test_set/):
  raw_fights_greco.csv     every UFC fight, newest first (semicolon-separated)
  fighter_details_greco.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

REPO = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
FILES = ["ufc_event_details.csv", "ufc_fight_results.csv", "ufc_fight_stats.csv", "ufc_fighter_tott.csv"]
OF_COLS = {"SIG.STR.": "SIG_STR.", "TOTAL STR.": "TOTAL_STR.", "TD": "TD", "HEAD": "HEAD", "BODY": "BODY",
           "LEG": "LEG", "DISTANCE": "DISTANCE", "CLINCH": "CLINCH", "GROUND": "GROUND"}
OUT = "personal_test_set/"


def load(src):
    if src:
        return [pd.read_csv(os.path.join(src, f)) for f in FILES]
    return [pd.read_csv(REPO + f) for f in FILES]


def _of(series):
    parts = series.fillna("0 of 0").astype(str).str.split(" of ")
    return parts.str[0].astype(float).fillna(0), parts.str[1].astype(float).fillna(0)


def _sec(series):
    def f(x):
        if pd.isna(x) or x in ("--", ""):
            return 0
        m, s = str(x).split(":")
        return int(m) * 60 + int(s)
    return series.map(f)


def totals(stats):
    """Sum per-round rows into one row per (bout URL-free key, fighter)."""
    s = stats.copy()
    s["EVENT"], s["BOUT"], s["FIGHTER"] = s["EVENT"].str.strip(), s["BOUT"].str.strip(), s["FIGHTER"].str.strip()
    agg = {}
    for c, k in OF_COLS.items():
        s[k + "_l"], s[k + "_a"] = _of(s[c])
        agg[k + "_l"] = "sum"; agg[k + "_a"] = "sum"
    for c, k in (("KD", "KD"), ("SUB.ATT", "SUB_ATT"), ("REV.", "REV")):
        s[k] = pd.to_numeric(s[c], errors="coerce").fillna(0)
        agg[k] = "sum"
    s["CTRL_s"] = _sec(s["CTRL"])
    agg["CTRL_s"] = "sum"
    t = s.groupby(["EVENT", "BOUT", "FIGHTER"], sort=False).agg(agg).reset_index()
    for k in OF_COLS.values():
        t[k] = t[k + "_l"].astype(int).astype(str) + " of " + t[k + "_a"].astype(int).astype(str)
    # UFCStats percentages are whole numbers, rounded down; "---" when nothing was attempted
    pct = lambda l, a: np.where(a > 0, (np.floor(100 * l / a.replace(0, 1))).astype(int).astype(str) + "%", "---")
    t["SIG_STR_pct"] = pct(t["SIG_STR._l"], t["SIG_STR._a"])
    t["TD_pct"] = pct(t["TD_l"], t["TD_a"])
    t["CTRL"] = (t["CTRL_s"] // 60).astype(int).astype(str) + ":" + (t["CTRL_s"] % 60).astype(int).astype(str).str.zfill(2)
    for k in ("KD", "SUB_ATT", "REV"):
        t[k] = t[k].astype(int)
    return t.set_index(["EVENT", "BOUT", "FIGHTER"])


def convert(events, results, stats, tott):
    ev = events.assign(EVENT=events["EVENT"].str.strip()).drop_duplicates("EVENT").set_index("EVENT")
    t = totals(stats)
    rows = []
    for _, r in results.iterrows():
        event, bout = str(r["EVENT"]).strip(), str(r["BOUT"]).strip()
        names = [n.strip() for n in bout.split(" vs. ")]
        if len(names) != 2 or event not in ev.index:
            continue
        a, b = names
        if (event, bout, a) not in t.index or (event, bout, b) not in t.index:
            continue  # no stats for this fight
        ta, tb = t.loc[(event, bout, a)], t.loc[(event, bout, b)]
        outcome = str(r["OUTCOME"]).strip()
        winner = a if outcome == "W/L" else b if outcome == "L/W" else np.nan
        row = {"R_fighter": a, "B_fighter": b}
        for k in ["KD", "SIG_STR.", "SIG_STR_pct", "TOTAL_STR.", "TD", "TD_pct", "SUB_ATT", "REV", "CTRL",
                  "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"]:
            row["R_" + k], row["B_" + k] = ta[k], tb[k]
        row.update({
            "win_by": str(r["METHOD"]).strip(), "last_round": int(r["ROUND"]),
            "last_round_time": str(r["TIME"]).strip(), "Format": str(r["TIME FORMAT"]).strip(),
            "Referee": r["REFEREE"], "date": ev.at[event, "DATE"], "location": ev.at[event, "LOCATION"],
            "Fight_type": str(r["WEIGHTCLASS"]).strip(), "Winner": winner,
        })
        rows.append(row)
    raw = pd.DataFrame(rows)
    raw["_d"] = pd.to_datetime(raw["date"], format="mixed")
    raw = raw.sort_values("_d", ascending=False, kind="stable").drop(columns="_d").reset_index(drop=True)

    det = tott.rename(columns={"FIGHTER": "fighter_name", "HEIGHT": "Height", "WEIGHT": "Weight",
                               "REACH": "Reach", "STANCE": "Stance", "DOB": "DOB"})
    det = det[["fighter_name", "Height", "Weight", "Reach", "Stance", "DOB"]].replace("--", np.nan)
    det["fighter_name"] = det["fighter_name"].str.strip()
    return raw, det.drop_duplicates("fighter_name")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="folder with the 4 Greco1899 CSVs (default: download)")
    args = ap.parse_args()
    raw, det = convert(*load(args.src))
    raw.to_csv(OUT + "raw_fights_greco.csv", sep=";", index=False)
    det.to_csv(OUT + "fighter_details_greco.csv", index=False)
    print(f"Saved {len(raw)} fights and {len(det)} fighters to {OUT}")


if __name__ == "__main__":
    main()
