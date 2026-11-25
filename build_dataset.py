"""
build_dataset.py — join labels and build the tabular + fight-history modalities.

Inputs (Kaggle rajeevw/ufcdata, in data/):
  data.csv                  pre-fight stats per fight (R_ = red corner, B_ = blue corner)
  raw_total_fight_data.csv  fight-by-fight results incl. win_by (semicolon-separated)

Output: data/fights.csv — one row per fight with
  label (KO/TKO, SUB, DEC), date, tabular features, history features (h_ prefix)

Usage: python build_dataset.py
"""
import numpy as np
import pandas as pd

DATA = "data/"
N_HIST = 5  # how many previous fights the history modality looks at

LABEL_MAP = {
    "KO/TKO": "KO/TKO",
    "TKO - Doctor's Stoppage": "KO/TKO",
    "Submission": "SUB",
    "Decision - Unanimous": "DEC",
    "Decision - Split": "DEC",
    "Decision - Majority": "DEC",
}


def method_group(win_by):
    return LABEL_MAP.get(win_by)  # None for DQ, overturned, etc.


def load():
    d = pd.read_csv(DATA + "data.csv").copy()
    r =pd.read_csv(DATA + "raw_total_fight_data.csv", sep=";")
    r["date"] = pd.to_datetime(r["date"], format="mixed")
    d["date"] = pd.to_datetime(d["date"])
    # The two files list the same fights in the same order; check, then join by position.
    assert (d.R_fighter.values == r.R_fighter.values).all()
    assert (d.B_fighter.values == r.B_fighter.values).all()
    assert (d.date.values == r.date.values).all()
    d["win_by"] = r["win_by"].values
    d["last_round"] = r["last_round"].values  # outcome info: used only for history, never as a feature
    # scheduled rounds (3 or 5), known before the fight, e.g. "3 Rnd (5-5-5)" -> 3
    d["no_of_rounds"] = r["Format"].str.extract(r"^(\d+)")[0].astype(float).values
    d["raw_winner"] = r["Winner"].values  # winner's name; NaN for draws / no contests
    return d


# --------------------------------------------------------------- history modality
def history_features(d):
    """For each fight and corner, summarize that fighter's previous N_HIST fights.
    Built only from fights strictly before the current date (no leakage)."""
    # one row per (fighter, fight)
    rows = []
    for i, f in d.iterrows():
        for corner, opp in (("R", "B"), ("B", "R")):
            name = f[f"{corner}_fighter"]
            won = f["raw_winner"] == name
            lost = isinstance(f["raw_winner"], str) and not won
            m = method_group(f["win_by"])
            rows.append({
                "fight_idx": i, "corner": corner, "fighter": name, "date": f["date"],
                "won": won, "lost": lost, "method": m, "rounds": f["last_round"],
            })
    long = pd.DataFrame(rows).sort_values(["fighter", "date"])

    feats = {}
    for name, g in long.groupby("fighter", sort=False):
        g = g.reset_index(drop=True)
        for k in range(len(g)):
            prev = g.iloc[max(0, k - N_HIST):k]
            cur = g.iloc[k]
            n = len(prev)
            out = {"h_n_prev": n}
            if n:
                out.update({
                    "h_win_rate": prev.won.mean(),
                    "h_win_ko": (prev.won & (prev.method == "KO/TKO")).mean(),
                    "h_win_sub": (prev.won & (prev.method == "SUB")).mean(),
                    "h_win_dec": (prev.won & (prev.method == "DEC")).mean(),
                    "h_loss_ko": (prev.lost & (prev.method == "KO/TKO")).mean(),
                    "h_loss_sub": (prev.lost & (prev.method == "SUB")).mean(),
                    "h_finish_rate": prev.method.isin(["KO/TKO", "SUB"]).mean(),
                    "h_avg_rounds": prev.rounds.mean(),
                    "h_days_since": (cur.date - prev.date.iloc[-1]).days,
                    "h_last_won": float(prev.won.iloc[-1]),
                })
            feats[(cur.fight_idx, cur.corner)] = out

    cols = ["h_n_prev", "h_win_rate", "h_win_ko", "h_win_sub", "h_win_dec", "h_loss_ko",
            "h_loss_sub", "h_finish_rate", "h_avg_rounds", "h_days_since", "h_last_won"]
    H = pd.DataFrame(index=d.index)
    for corner in ("R", "B"):
        block = pd.DataFrame([feats[(i, corner)] for i in d.index], index=d.index).reindex(columns=cols)
        block.columns = [f"{corner}_{c}" for c in cols]
        H = H.join(block)
    return H


def main():
    d = load()
    H = history_features(d)
    d = d.join(H)
    d["label"] = d["win_by"].map(method_group)
    d = d[d["label"].notna() & d["raw_winner"].notna()]  # drop DQs, overturned, draws, NCs
    d = d.drop(columns=["raw_winner"])
    d.to_csv(DATA + "fights.csv", index=False)
    print(f"Saved {len(d)} fights -> {DATA}fights.csv")
    print(d["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
