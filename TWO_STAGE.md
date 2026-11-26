# Two-stage UFC finish predictor

Open [two_stage_predictor.ipynb](two_stage_predictor.ipynb) for the executed experiment. Run its cells from the repository root to rebuild features, train the models, compare their picks, and run the 20-fight demo. The trained model and per-fight predictions are also available in `results/method_rates/`.

Run the saved model on the included 20-bout demo with:

```bash
python -m pip install -r requirements.txt
python run.py --input . --model results/method_rates/selected_model.joblib --out demo_two_stage.csv
```

For a new fight card, replace the input folder with a folder in the same upload format. To retrain from the bundled raw fight data and reproduce the comparison, run `python train_method_rates.py`. The run used Python 3.12, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0 and CatBoost 1.2.10.

Two logistic regressions estimate the probability of a finish and the probability of submission given a finish. Their probabilities combine to give the three method probabilities. Features include bout division, scheduled rounds, previous finish history, and striking and grappling rates per minute. The model uses both fighters' totals and absolute differences so swapping their corners has no effect. It uses no profile text.

Both the direct and the two-stage alternatives use the same features and training fights from 2010–2023. For each architecture, five class-weight adjustments were tested on 2024. Among the adjustments within two percentage points of that architecture's best validation accuracy, the one with highest macro F1 was retained. The two-stage model then won on validation macro F1. Its winning adjustment divides each class probability by the square root of the class's training frequency when choosing the predicted label. The probability columns preserve the unadjusted model estimates, so the selected label can differ from the most probable label.

On 513 eligible 2025 fights, the two-stage model got 269 right (52.4%), including 16 of 91 submissions. The PR's selected CatBoost got 259 right (50.5%), including one submission. The direct logistic model got 266 right. On the 512 available fights through November 21, 2026, the two-stage model got 252 right (49.2%), including 15 of 87 submissions. CatBoost got 245 right (47.9%), including two submissions. The direct model got 253 right. The 2025 results had been examined in earlier project work. The available 2026 fights are a separate, partial-year check.

Across the 1,025 fights, the observed accuracy difference versus the PR's selected CatBoost was +1.7 percentage points. A paired bootstrap resampling event dates (10,000 draws, seed 4397) gives a 95% interval of −1.6 to +5.0 percentage points. The accuracy advantage remains uncertain. The two-stage model called submission 105 times and got 31 of those calls right.

The original three feature regression tests pass. The prediction command reproduced all 20 demo picks and all 60 probabilities at the command's three-decimal precision.
