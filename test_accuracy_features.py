import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from build_personal_set import build
from predict import build_features
from train import tabular_features


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parent
        source = pd.read_csv(root / "raw_fights.csv", sep=";").iloc[0]
        details = pd.read_csv(root / "fighter_details.csv").iloc[0]
        def make(date, red, blue):
            row = source.copy()
            row.update(pd.Series(dict(date=date, R_fighter=red, B_fighter=blue, Winner=red,
                                      win_by="Submission", Fight_type="Lightweight Bout")))
            return row
        cls.raw = pd.DataFrame([make("2022-01-01", "A", "B"), make("2022-01-01", "A", "C"),
                                make("2022-02-01", "A", "D")]).reset_index(drop=True)
        cls.details = pd.DataFrame([dict(details, fighter_name=name, Stance=stance)
                                    for name, stance in zip("ABCD", ["Orthodox", "Southpaw"] * 2)])
        cls.fights = cls.raw.iloc[[0, 2]][["date", "R_fighter", "B_fighter"]].assign(
            sample_id=["first", "later"], weight_class="Lightweight", title_bout=0, no_of_rounds=3)

    def test_strict_cutoff_in_both_feature_families(self):
        d = build(self.raw, self.details)
        first_day = d[d.date == pd.Timestamp("2022-01-01")]
        self.assertEqual(first_day.R_h_n_prev.tolist(), [0, 0])
        self.assertEqual(first_day.R_wins.tolist(), [0, 0])
        self.assertEqual(d.loc[d.date == pd.Timestamp("2022-02-01"), "R_wins"].item(), 2)

    def test_known_results_preserved_across_batch_dates(self):
        bundle = {"feature_set": "stats", "columns": ["R_wins", "R_h_win_sub"]}
        _, batch = build_features(self.fights, self.raw, self.details, None, bundle)
        _, one = build_features(self.fights.tail(1), self.raw, self.details, None, bundle)
        assert_frame_equal(batch.tail(1).reset_index(drop=True), one)
        self.assertEqual(one.R_wins.item(), 2)

    def test_stance_columns_independent_of_batch(self):
        d = build(self.raw, self.details)
        d.loc[d.index[-1], ["R_Stance", "B_Stance"]] = ["Southpaw", "Orthodox"]
        one = tabular_features(d.iloc[:1]).filter(like="diff_Stance")
        full = tabular_features(d).iloc[:1].filter(like="diff_Stance")
        assert_frame_equal(one, full)
        self.assertEqual(one["diff_Stance_Orthodox"].item(), 1)


if __name__ == "__main__":
    unittest.main()
