#!/usr/bin/env python3
"""Validate OffSide 2026 Kaggle submission format."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ID_COL = "appearance_id"
TARGET = "scored_flag"


def check(condition: bool, passing_msg: str, failing_msg: str) -> bool:
    if condition:
        print(f"  OK  {passing_msg}")
        return True
    print(f"  BAD {failing_msg}")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate solution.csv")
    parser.add_argument("submission", nargs="?", default="outputs/solution.csv")
    parser.add_argument("test", nargs="?", default="data/test.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission_path = Path(args.submission)
    test_path = Path(args.test)

    print(f"Checking {submission_path}\n")
    sub = pd.read_csv(submission_path)
    results = []

    results.append(check(
        list(sub.columns) == [ID_COL, TARGET],
        "Columns are exactly appearance_id, scored_flag",
        f"Wrong columns/order. Got: {list(sub.columns)}",
    ))
    results.append(check(sub.isna().sum().sum() == 0, "No missing values", f"{sub.isna().sum().sum()} missing values found"))
    results.append(check(sub[TARGET].between(0, 1).all(), "Probabilities are in [0, 1]", "Probabilities outside [0, 1]"))
    results.append(check(sub[ID_COL].duplicated().sum() == 0, "No duplicate appearance IDs", f"{sub[ID_COL].duplicated().sum()} duplicate IDs"))

    if test_path.exists():
        test = pd.read_csv(test_path)
        results.append(check(len(sub) == len(test), f"Row count matches test ({len(test):,})", f"Submission rows={len(sub):,}, test rows={len(test):,}"))
        missing = set(test[ID_COL]) - set(sub[ID_COL])
        extra = set(sub[ID_COL]) - set(test[ID_COL])
        results.append(check(not missing, "All test IDs are present", f"Missing {len(missing):,} test IDs"))
        results.append(check(not extra, "No extra IDs", f"Found {len(extra):,} extra IDs"))
    else:
        print(f"  WARN Test file not found at {test_path}; skipped ID coverage checks")

    print(f"\nMean probability: {sub[TARGET].mean():.5f}")
    print(f"Std probability : {sub[TARGET].std():.5f}")
    print("\nPreview:")
    print(sub.head(8).to_string(index=False))

    if all(results):
        print("\nAll checks passed. Ready to upload.")
    else:
        raise SystemExit(f"\n{results.count(False)} validation check(s) failed.")


if __name__ == "__main__":
    main()
