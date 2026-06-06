# OffSide 2026 Football Datathon

This project predicts the probability that a football player scores at least one goal in a given match. The final Kaggle submission is `outputs/solution.csv` with `appearance_id` and `scored_flag` probability columns.

## Folder Structure

```text
data/
  train.csv
  test.csv
notebooks/
  OffSide2026_Solution.ipynb
src/
  train_model.py
  validate_submission.py
outputs/
  solution.csv
  metrics.json
  eda_plots/
requirements.txt
```

## Setup

Copy the provided datathon CSV files into `data/` and name them:

```text
data/train.csv  # contains scored_flag
data/test.csv   # contains appearance_id and no scored_flag
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Run Training

Full run:

```bash
python3 src/train_model.py
```

Quick smoke test with fewer estimators:

```bash
python3 src/train_model.py --fast
```

The script writes:

- `outputs/solution.csv`
- `outputs/metrics.json`
- EDA plots in `outputs/eda_plots/`

## Validate Submission

```bash
python3 src/validate_submission.py outputs/solution.csv data/test.csv
```

The validator checks required columns, missing values, probability range, duplicate IDs, row count, and test-ID coverage.

## Method Summary

The pipeline performs EDA, median/mode-style missing value handling, categorical encoding, leakage-column removal, football-specific feature engineering, stratified cross-validation, and probability blending.

Main features include xG weighted by position, xG/shots scaled by playing time, attacker-starting interactions, xG composite, market-value interactions, international goals per cap, and home/away flags.

Main metric: Average Precision. Secondary metric: ROC-AUC. Random seed: `42`.

## Rulebook Notes

The code avoids AutoML and uses manually written preprocessing, feature engineering, model training, and blending. Final submitted code should run start-to-finish without errors and reproduce `solution.csv`.
