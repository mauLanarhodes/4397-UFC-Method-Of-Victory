"""
run.py — upload/run harness: predict a folder of fights and, if answers are given, score them.

Usage:
  python run.py --input personal_test_set/upload
  python run.py --input my_fights/ --out my_predictions.csv

The input folder needs fights.csv, raw_fights.csv, fighter_details.csv and optionally
fighter_profiles.csv (see predict.py). If it also has labels.csv (sample_id, label),
the script prints macro F1, accuracy and a fight-by-fight table.
"""
import argparse
import os
import warnings

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from predict import MODEL_PATH, predict_multimodal
from train import CLASSES

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="upload folder")
    ap.add_argument("--out", default=None, help="where to save predictions (default: <input>/predictions.csv)")
    ap.add_argument("--model", default=MODEL_PATH, help="trained model bundle")
    args = ap.parse_args()

    preds = predict_multimodal(args.input, model_path=args.model)
    out = args.out or os.path.join(args.input, "predictions.csv")

    lab_path = os.path.join(args.input, "labels.csv")
    if os.path.exists(lab_path):
        lab = pd.read_csv(lab_path)[["sample_id", "label"]]
        preds = preds.merge(lab, on="sample_id", how="left")
        preds["correct"] = preds["predicted"] == preds["label"]
        y, p = preds["label"], preds["predicted"]
        print(f"Macro F1: {f1_score(y, p, average='macro'):.3f}   "
              f"Accuracy: {accuracy_score(y, p):.3f} ({preds['correct'].sum()}/{len(preds)})")
        print("Confusion matrix (rows = true, columns = predicted):")
        print(pd.DataFrame(confusion_matrix(y, p, labels=CLASSES), index=CLASSES, columns=CLASSES), "\n")

    print(preds.to_string(index=False))
    preds.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
