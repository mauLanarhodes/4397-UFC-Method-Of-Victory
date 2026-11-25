"""
ufc_features.py — rebuild the Kaggle data.csv features from raw fight data.

This re-implements the feature math of the original dataset (rajeevw/ufcdata,
github.com/WarrierRajeev/UFC-Predictions, src/createdata/preprocess*.py) so that
NEW fights (our 2025 personal test set) get stats computed exactly the same way
as the training data. Key details copied from the original:

  * "avg" stats are exponentially weighted means over the fighter's PREVIOUS fights,
    oldest to newest: pandas .ewm(span=3, adjust=False).mean(), last value.
    So recent fights count more than old ones.
  * Records/streaks count draws as losses, and streaks are counted over a reversed
    fight list (both quirks of the original code, kept for consistency).
  * win_by counts sum the fighter's previous wins by each method.
  * Age = floor((fight date - DOB) / 365.25 days).

verify() checks the re-implementation against the Kaggle data.csv.

Usage: python ufc_features.py      (runs the check)
"""
import math
import re

import numpy as np
import pandas as pd

SPLIT_COLS = ["SIG_STR.", "TOTAL_STR.", "TD", "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"]
WIN_BY = ["Decision - Majority", "Decision - Split", "Decision - Unanimous", "KO/TKO",
          "Submission", "TKO - Doctor's Stoppage"]
NUM = ["KD", "SIG_STR_pct", "TD_pct", "SUB_ATT", "REV"]
FIRST_ROUND_SECONDS = {
    "3 Rnd (5-5-5)": 300, "5 Rnd (5-5-5-5-5)": 300, "1 Rnd + OT (12-3)": 720, "No Time Limit": 1,
    "3 Rnd + OT (5-5-5-5)": 300, "1 Rnd (20)": 20, "2 Rnd (5-5)": 300, "1 Rnd (15)": 900,
    "1 Rnd (10)": 600, "1 Rnd (12)": 720, "1 Rnd + OT (30-5)": 1800, "1 Rnd (18)": 1080,
    "1 Rnd + OT (15-3)": 900, "1 Rnd (30)": 1800, "1 Rnd + OT (31-5)": 155,
    "1 Rnd + OT (27-3)": 1620, "1 Rnd + OT (30-3)": 1800,
}
EXCEPTION_FORMATS = {"1 Rnd + 2OT (15-3-3)": (900, 180), "1 Rnd + 2OT (24-3-3)": (1440, 180)}


def _numerical_columns():
    base = [("KD",), ("SIG_STR_pct",), ("TD_pct",), ("SUB_ATT",), ("REV",)]
    cols = []
    for (b,) in base:
        cols += [f"hero_{b}", f"opp_{b}"]
    for b in SPLIT_COLS:
        cols += [f"hero_{b}_att", f"hero_{b}_landed", f"opp_{b}_att", f"opp_{b}_landed"]
    cols += ["hero_CTRL_time(seconds)", "opp_CTRL_time(seconds)", "total_time_fought(seconds)"]
    return cols


NUMERICAL = _numerical_columns()


def prep_raw(raw):
    """Clean the raw fight table the way the original Preprocessor did."""
    f = raw.copy().reset_index(drop=True)
    for c in SPLIT_COLS:
        for s in ("R_", "B_"):
            parts = f[s + c].astype(str).str.split("of")
            f[f"{s}{c}_att"] = parts.str[1].astype(int)
            f[f"{s}{c}_landed"] = parts.str[0].astype(int)
    f["Winner"] = f["Winner"].fillna("Draw")
    for c in ("R_SIG_STR_pct", "B_SIG_STR_pct", "R_TD_pct", "B_TD_pct"):
        f[c] = f[c].map(lambda x: 0 if x == "---" else float(str(x).replace("%", "")) / 100)
    f["title_bout"] = f["Fight_type"].str.contains("Title Bout")
    f["last_round_time"] = f["last_round_time"].map(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))
    for s in ("R_", "B_"):
        f[f"{s}CTRL_time(seconds)"] = f[s + "CTRL"].map(
            lambda x: 0 if x == "--" else int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    def total_time(r):
        if r["Format"] in FIRST_ROUND_SECONDS:
            return (r["last_round"] - 1) * FIRST_ROUND_SECONDS[r["Format"]] + r["last_round_time"]
        if r["Format"] in EXCEPTION_FORMATS:
            first, ot = EXCEPTION_FORMATS[r["Format"]]
            if r["last_round"] - 1 >= 2:
                return first + (r["last_round"] - 2) * ot + r["last_round_time"]
            return (r["last_round"] - 1) * first + r["last_round_time"]
        return np.nan
    f["total_time_fought(seconds)"] = f.apply(total_time, axis=1)
    for w in WIN_BY:
        f["win_by_" + w] = (f["win_by"] == w).astype(int)
    f["date"] = pd.to_datetime(f["date"], format="mixed")
    return f


def _as_hero(rows, corner):
    other = "B" if corner == "R" else "R"
    ren = {}
    for c in rows.columns:
        if c.startswith(corner + "_"):
            ren[c] = "hero_" + c[2:]
        elif c.startswith(other + "_"):
            ren[c] = "opp_" + c[2:]
    return rows.rename(columns=ren)


def _result_stats(results):
    """Copied quirk: the original reverses the oldest->newest list before counting, so
    'current' streaks are counted from the newest fight backward to the oldest.
    Each result is 'hero' or 'opp' (draws were mapped to 'opp' upstream)."""
    cws = cls = lws = wins = losses = 0
    for r in results:
        if r == "hero":
            wins += 1; cws += 1; cls = 0
            if lws < cws:
                lws += 1
        else:
            losses += 1; cws = 0; cls += 1
    return cws, cls, lws, wins, losses, 0


def fighter_features(f, details):
    """Per (fight row, corner) pre-fight features, like the R_/B_ columns of data.csv.

    `f` is prep_raw output where row order is NEWEST first (as in raw_total_fight_data.csv).
    Returns a DataFrame indexed like `f` with R_* and B_* columns.
    """
    out = {"R": {}, "B": {}}
    names = set(f["R_fighter"]) | set(f["B_fighter"])
    for name in names:
        parts = []
        for corner in ("R", "B"):
            rows = f[f[f"{corner}_fighter"] == name]
            if len(rows):
                h = _as_hero(rows, corner)
                h["_corner"] = corner
                parts.append(h)
        fighter = pd.concat(parts).sort_index()          # newest first (raw order)
        winner = np.where(fighter["opp_fighter"].notna() & (fighter["Winner"] == name), "hero", "opp")
        fighter = fighter.assign(_res=winner)
        for i, idx in enumerate(fighter.index):
            prev = fighter.iloc[i + 1:].sort_index(ascending=False)   # older fights, oldest first
            if len(prev):
                s = prev[NUMERICAL].ewm(span=3, adjust=False).mean().iloc[-1].to_dict()
            else:
                s = {c: np.nan for c in NUMERICAL}
            s["total_rounds_fought"] = prev["last_round"].sum()
            s["total_title_bouts"] = int(prev["title_bout"].sum())
            for k, v in zip(["current_win_streak", "current_lose_streak", "longest_win_streak",
                             "wins", "losses", "draw"], _result_stats(list(prev["_res"])[::-1])):
                s[k] = v
            won = prev[prev["_res"] == "hero"]
            for w in WIN_BY:
                s["win_by_" + w] = won["win_by_" + w].sum()
            out[fighter.at[idx, "_corner"]][idx] = s

    frames = []
    det = details.set_index("fighter_name")
    for corner in ("R", "B"):
        g = pd.DataFrame.from_dict(out[corner], orient="index")
        fd = det.reindex(f.loc[g.index, f"{corner}_fighter"].values)
        g["Stance"] = fd["Stance"].values
        g["Height_cms"] = fd["Height"].map(_to_cm).values
        g["Reach_cms"] = fd["Reach"].map(_to_cm).values
        g["Weight_lbs"] = fd["Weight"].map(lambda x: float(str(x).replace(" lbs.", "")) if pd.notna(x) else np.nan).values
        dob = pd.to_datetime(fd["DOB"], format="mixed", errors="coerce").values
        days = (f.loc[g.index, "date"].values - dob) / np.timedelta64(1, "D")
        g["age"] = [math.floor(x / 365.25) if not np.isnan(x) else np.nan for x in days]
        g.columns = [_rename(corner, c) for c in g.columns]
        frames.append(g)
    return pd.concat(frames, axis=1).reindex(f.index)


def _to_cm(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if len(x.split("'")) == 2:
        feet, inches = x.split("'")
        return float(feet) * 30.48 + int(inches.replace(" ", "").replace('"', "")) * 2.54
    return float(x.replace('"', "")) * 2.54


def _rename(corner, c):
    if c.startswith("hero_"):
        c = "avg_" + c[5:].replace(".", "")
    elif c.startswith("opp_"):
        c = "avg_opp_" + c[4:].replace(".", "")
    elif c.startswith("win_by_"):
        c = c.replace(" ", "").replace("-", "_").replace("'s", "_")
    return f"{corner}_{c}"


def verify(data_dir="data/"):
    raw = pd.read_csv(data_dir + "raw_total_fight_data.csv", sep=";")
    det = pd.read_csv(data_dir + "raw_fighter_details.csv")
    ref = pd.read_csv(data_dir + "data.csv")
    mine = fighter_features(prep_raw(raw), det)
    cols = [c for c in mine.columns if c in ref.columns]
    report = []
    for c in cols:
        a, b = mine[c].values, ref[c].values
        if not (pd.api.types.is_numeric_dtype(mine[c]) and pd.api.types.is_numeric_dtype(ref[c])):
            ok = np.mean([(x == y) or (pd.isna(x) and pd.isna(y)) for x, y in zip(a, b)])
        else:
            a, b = a.astype(float), b.astype(float)
            ok = np.mean(np.isclose(a, b, atol=1e-6, equal_nan=True))
        report.append((c, ok))
    rep = pd.DataFrame(report, columns=["column", "match_rate"]).sort_values("match_rate")
    print(f"Checked {len(cols)} columns x {len(ref)} fights")
    print(rep.head(12).to_string(index=False))
    print(f"Columns matching 100%: {(rep.match_rate == 1).sum()} / {len(rep)}")
    return rep


if __name__ == "__main__":
    verify()
