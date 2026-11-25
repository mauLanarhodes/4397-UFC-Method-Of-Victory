# Personal Test Set — Group 4 (UFC Method of Victory)

20 real UFC fights from 2025, all after the training data ends (early 2021). The model never sees these during training or tuning.

## Files
| File | What it holds |
|---|---|
| `meta.csv` | Model inputs: `sample_id, event, date, weight_class, title_bout, fighter_1, fighter_2` |
| `labels.csv` | Answer key, kept separate: `sample_id, winner, method_raw, label, round, time` |
| `raw_fights_greco.csv` | Every UFC fight 1994–2026 from UFCStats (via the Greco1899 export), converted to the Kaggle raw format |
| `fighter_details_greco.csv` | Height, weight, reach, stance, DOB for every fighter |
| `target_fights.csv` | The 20 fights with UFCStats red/blue corners |
| `fights_2025.csv` | Model-ready rows: pre-fight stats (same math as training) + fight history |

Text profiles for all 40 fighters are in `../data/fighter_profiles.csv`.

Fighters are listed alphabetically in `meta.csv`, so the order never gives away who won.

## How the fights were chosen
- Events: UFC 317 (June 28, 2025) and UFC 320 (Oct 4, 2025)
- Picked to balance the classes: 7 KO/TKO, 6 Submission, 7 Decision
- Skipped most debut fights (no UFC stats to use) and most catchweight bouts. One debut slipped through: Tre'ston Vines (P11) had no prior UFC fights.
- Results were cross-checked between event results pages and each event's KO/sub/decision totals

## Labels
`KO/TKO`, `SUB`, `DEC` (draws, no-contests and DQs are excluded, same as training).

## Collection and consent notes
All data is publicly published sports results, fighter statistics, and public profile text about professional athletes. No personal data was recorded from group members or anyone else, and no audio or face data is used, so no consent forms are needed.

## Why this tests generalization
Training covers 1993–2021. These fights are from 2025, after the sport has changed (more wrestling-heavy styles, new fighters, rule and judging changes). Some fighters here also had little or no UFC history in the training years.
