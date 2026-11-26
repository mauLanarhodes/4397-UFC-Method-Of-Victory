"""Compact, corner-invariant fight features and direct/two-stage logistic models."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fight_features(d):
    blocks = []
    for corner in ("R", "B"):
        f = d.filter(regex="^" + corner + "_").copy()
        f.columns = f.columns.str[2:]
        n = f.wins + f.losses
        x = pd.DataFrame(index=d.index)
        x["bouts"] = np.log1p(n)
        x["debut"] = (n == 0).astype(float)
        for name in ("Height_cms", "Reach_cms", "age", "h_avg_rounds", "h_finish_rate",
                     "h_win_ko", "h_loss_ko", "h_win_sub", "h_loss_sub", "h_win_rate"):
            x[name] = f[name]
        for method in ("KO/TKO", "Submission"):
            x["wins_" + method] = f["win_by_" + method] / (n + 2)
        x["days_since"] = np.log1p(f.h_days_since)
        seconds = f["total_time_fought(seconds)"]
        x["mean_minutes"] = seconds / 60
        for name in ("KD","SUB_ATT","SIG_STR_landed","TD_landed","CTRL_time(seconds)"):
            for prefix in ("avg_", "avg_opp_"):
                # Small denominator floor damps rates after very short fights.
                x[prefix+name] = np.log1p(60 * f[prefix+name] / seconds.clip(lower=60))
        blocks.append(x)
    red, blue = blocks
    x = pd.concat([(red+blue).add_prefix("sum_"), (red-blue).abs().add_prefix("gap_")], axis=1)
    # If one athlete is making a debut, preserve the experienced athlete's information.
    for c in red:
        x["observed_"+c] = red[c].combine_first(blue[c]).where(red[c].isna() | blue[c].isna(), 0)
    for method in ("ko","sub"):
        a=(red["h_win_"+method]+blue["h_loss_"+method])/2
        b=(blue["h_win_"+method]+red["h_loss_"+method])/2
        x["stronger_match_"+method]=np.maximum(a,b)
    x["rounds"] = d.no_of_rounds
    x["title"] = d.title_bout.astype(int)
    return pd.concat([x,pd.get_dummies(d.weight_class, prefix="division", dtype=float)], axis=1)


class MethodModel:
    def __init__(self, two_stage):
        self.two_stage = two_stage

    def fit(self, X, y):
        self.frequency = y.value_counts(normalize=True).sort_index().to_numpy()
        def classifier():
            return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                                 LogisticRegression(C=0.05,max_iter=1000))
        if self.two_stage:
            self.finish = classifier().fit(X,y.ne(0))
            self.submission = classifier().fit(X[y.ne(0)],y[y.ne(0)].eq(2))
        else:
            self.direct = classifier().fit(X,y)
        return self

    def predict_proba(self, X):
        if not self.two_stage:
            return self.direct.predict_proba(X)
        finish = self.finish.predict_proba(X)[:,1]
        sub = self.submission.predict_proba(X)[:,1]
        return np.column_stack([1-finish, finish*(1-sub), finish*sub])
