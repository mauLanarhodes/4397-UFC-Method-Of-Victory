"""Pre-fight statistics used by the accuracy experiment and upload predictor."""
import pandas as pd

from train import history_features, tabular_features


def model_features(fights, feature_set, columns=None):
    # Listed weight and stance come from undated profiles. Use the bout's
    # scheduled division, and retain height, reach and age as physical data.
    tab = tabular_features(fights.drop(columns=["sample_id"], errors="ignore"))
    tab = tab.drop(columns=[c for c in tab if "Stance" in c or "Weight_lbs" in c])
    blocks = [tab, history_features(fights)]
    if feature_set == "matchups":
        matchup = {}
        for hero, opp in (("R", "B"), ("B", "R")):
            # Average of attacking accuracy and the opposing defense's concession rate.
            for stat in ("TD", "SIG_STR"):
                attack = fights[f"{hero}_avg_{stat}_landed"] / fights[f"{hero}_avg_{stat}_att"].replace(0, float("nan"))
                allowed = fights[f"{opp}_avg_opp_{stat}_landed"] / fights[f"{opp}_avg_opp_{stat}_att"].replace(0, float("nan"))
                matchup[f"{hero}_expected_{stat}_landed"] = fights[f"{hero}_avg_{stat}_att"] * (attack + allowed) / 2
            for method in ("ko", "sub"):
                matchup[f"{hero}_match_{method}"] = (fights[f"{hero}_h_win_{method}"] +
                                                     fights[f"{opp}_h_loss_{method}"]) / 2
        blocks.append(pd.DataFrame(matchup, index=fights.index))
    X = pd.concat(blocks, axis=1)
    if columns is not None:
        # Unknown event divisions have no active column in the training vocabulary.
        missing = set(columns) - set(X)
        if any(not c.startswith("weight_class_") for c in missing):
            raise ValueError(f"Missing expected numeric features: {sorted(missing)}")
        X = X.reindex(columns=columns, fill_value=0.0)
    return X.astype(float)
