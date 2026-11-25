# Personal Test Set — Group 4 (UFC Method of Victory)

20 real UFC fights from 2025, all after the training data ends (early 2021). The model never sees these during training or tuning.

## Files
| File | What it holds |
|---|---|
| `meta.csv` | Model inputs: `sample_id, event, date, weight_class, title_bout, fighter_1, fighter_2` |
| `labels.csv` | Answer key, kept separate: `sample_id, winner, method_raw, label, round, time` |
| `fighter_stats.csv` *(to collect)* | Each fighter's stats **as of the day before the fight**, same columns as the Kaggle `data.csv` |
| `fighter_profiles.csv` *(to collect)* | Text profile per fighter (Wikipedia + UFC.com), same format as training |

Fighters are listed alphabetically in `meta.csv`, so the order never gives away who won.

## How the fights were chosen
- Events: UFC 317 (June 28, 2025) and UFC 320 (Oct 4, 2025)
- Picked to balance the classes: 7 KO/TKO, 6 Submission, 7 Decision
- Skipped: debut fights (no UFC stats to use) and most catchweight bouts
- Results were cross-checked between event results pages and each event's KO/sub/decision totals

## Labels
`KO/TKO`, `SUB`, `DEC` (draws, no-contests and DQs are excluded, same as training).

## Collection and consent notes
All data is publicly published sports results, fighter statistics, and public profile text about professional athletes. No personal data was recorded from group members or anyone else, and no audio or face data is used, so no consent forms are needed.

## Why this tests generalization
Training covers 1993–2021. These fights are from 2025, after the sport has changed (more wrestling-heavy styles, new fighters, rule and judging changes). Some fighters here also had little or no UFC history in the training years.
