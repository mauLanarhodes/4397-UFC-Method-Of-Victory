# UFC method-of-victory accuracy experiment

Use this update with [the original project at commit 4831eec](https://github.com/drew-000/4397-UFC-Method-Of-Victory/tree/4831eec58edc31266787968eea15e0820eeb3b41). Extract the files into a fresh checkout of that commit. All commands below run from the project root.

The selected model correctly classified 259 of 513 eligible fights in 2025. Always predicting decision got 253 right. The six extra correct picks have a paired 95% bootstrap interval of −2.9 to +5.0 percentage points for the accuracy difference. The model caught 212 of 253 decisions, 46 of 169 KOs and one of 91 submissions.

| Model | 2025 correct | Accuracy | Macro F1 | Submissions caught |
|---|---:|---:|---:|---:|
| Accuracy-selected CatBoost, unweighted | 259 | 50.5% | 0.335 | 1/91 |
| Published model (original saved weights) | 237 | 46.2% | 0.404 | 18/91 |
| Updated balanced three-model ensemble | 252 | 49.1% | 0.442 | 24/91 |
| Always decision | 253 | 49.3% | 0.220 | 0/91 |

Relative to the published weights, the accuracy-selected model gained 4.3 percentage points (paired 95% interval −0.6 to +9.0). Both intervals use 10,000 paired resamples of the 42 event dates, seed 4397. Fights on a sampled date stay together. All results are retrospective.

## Run the saved model

The root of the original checkout includes a demo upload with 20 fights. Output defaults to the input folder, so specify an output path to preserve the published CSV.

```bash
python -m pip install -r requirements.txt
python run.py --input . --model results/accuracy/selected_model.joblib --out demo_accuracy.csv
```

For a new card, replace the input directory with a folder containing fights.csv, raw_fights.csv and fighter_details.csv in the project's upload format. This model uses statistics, fight history and scheduled bout context. Text profiles are optional and unused by this model. For a batch spanning multiple dates, supply the actual completed fights in raw_fights.csv if those results should be available for later predictions. Calling the predictor without an explicit model still uses the original bundle.

## Reproduce the comparison

```bash
python -m unittest test_accuracy_features
python accuracy_experiment.py
```

The training run used Python 3.12, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, CatBoost 1.2.10, LightGBM 4.7.0 and XGBoost 3.4.1. To reproduce the original-weight comparator, run `python score_original.py` in a separate environment with the same booster versions, pandas 3.0.6 and scikit-learn 1.9.1. These versions can read the original model's serialized profiles and text encoder.

Training used the repository's raw_fights_greco.csv and fighter_details_greco.csv. Outcomes with no winner or outside KO/TKO, submission and decision were excluded as targets. Pre-2010 fights remained available as history. All training and evaluation features used fight results strictly before the target date. The new models exclude profile text and undated listed weight and stance. Height, reach and date of birth come from the supplied fighter details and are treated as fixed attributes.

Training used 6,051 fights from 2010–2023. Validation used 511 fights in 2024. The comparison included a majority baseline, four single-seed CatBoost configurations crossing class weighting with optional matchup features, and the project's balanced three-booster recipe with five seeds per booster on updated statistics and history. Each single CatBoost used the same settings and seed. Validation accuracy selected the winner, with macro F1 as the tie breaker. The unweighted stats model scored 55.8% on validation, compared with 55.0% for always picking decision. Adding matchup features did not improve validation accuracy for either weighting choice.

The frozen models were evaluated on all 513 eligible 2025 fights in the export. This includes the 20 bouts previously analyzed in the original project. The original model comparator used its published weights and text encoder with the corrected feature pipeline on the same bouts. Its predicted classes agreed with the published picks on all 20 of the previously reported fights.

results/accuracy/scores.csv holds the validation and test metrics for the new fits and majority baseline. predictions_2025.csv holds every test pick and the selected model's probabilities. original_2025.csv holds the original model's predictions for a one-to-one comparison.

## Feature checks

The patched builders exclude same-day outcomes from both rolling averages and recent history, use consistent stance indicator columns across uploads, and isolate each target date while preserving completed past fights in multi-date batches. The regression tests cover all three cases. The saved model's demo predictions and rounded probabilities matched the corresponding 20 rows in the evaluation file.
