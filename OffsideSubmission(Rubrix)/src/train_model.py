#!/usr/bin/env python3
"""Train the OffSide 2026 scorer-probability model and create solution.csv.

The script is intentionally explicit rather than AutoML: it performs EDA, safe
feature engineering, cross-validation, model blending, and submission export.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
TARGET = "scored_flag"
ID_COL = "appearance_id"

IDENTIFIERS = [
    "appearance_id",
    "player_name",
    "date",
    "name_x",
    "name_y",
    "home_club_name",
    "away_club_name",
    "stadium",
    "referee",
    "home_club_id",
    "away_club_id",
]

LEAKAGE_COLUMNS = [
    "home_club_goals",
    "away_club_goals",
    "goal_diff_abs",
]

DEFAULT_CATEGORICALS = [
    "foot",
    "position",
    "sub_position",
    "country_of_citizenship",
    "country_name",
    "competition_type",
    "confederation",
    "market_value_tier",
    "age_bucket",
    "home_away",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OffSide 2026 model")
    parser.add_argument("--train", default="data/train.csv", help="Path to train CSV")
    parser.add_argument("--test", default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--output", default="outputs/solution.csv", help="Submission CSV path")
    parser.add_argument("--plots-dir", default="outputs/eda_plots", help="EDA plot output directory")
    parser.add_argument("--metrics", default="outputs/metrics.json", help="Metrics JSON path")
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified CV folds")
    parser.add_argument("--fast", action="store_true", help="Use fewer estimators for quick smoke testing")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def safe_divide(num: pd.Series, den: pd.Series | float, cap: float | None = None) -> pd.Series:
    result = num.astype(float) / (pd.Series(den, index=num.index).astype(float) + 1e-6)
    if cap is not None:
        result = result.clip(0, cap)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def has_cols(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(col in df.columns for col in cols)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if has_cols(df, ["avg_xG", "is_attacker", "is_midfielder", "is_defender"]):
        df["xG_weighted_by_position"] = df["avg_xG"] * (
            df["is_attacker"].astype(float) * 1.0
            + df["is_midfielder"].astype(float) * 0.4
            + df["is_defender"].astype(float) * 0.1
        )

    if has_cols(df, ["avg_xG", "minutes_ratio"]):
        df["xG_scaled_to_time_played"] = df["avg_xG"] * df["minutes_ratio"]
    if has_cols(df, ["avg_shots", "minutes_ratio"]):
        df["shots_scaled_to_time_played"] = df["avg_shots"] * df["minutes_ratio"]
    if has_cols(df, ["avg_npxG", "minutes_ratio"]):
        df["npxG_scaled_to_time_played"] = df["avg_npxG"] * df["minutes_ratio"]

    if has_cols(df, ["avg_xG", "is_attacker", "starter_flag"]):
        df["attacker_who_started"] = df["avg_xG"] * df["is_attacker"].astype(float) * df["starter_flag"].astype(float)
    if has_cols(df, ["avg_xG", "is_attacker", "full_match_flag"]):
        df["attacker_full_match"] = df["avg_xG"] * df["is_attacker"].astype(float) * df["full_match_flag"].astype(float)

    if has_cols(df, ["avg_xG", "avg_npxG", "avg_shots", "avg_xGChain"]):
        df["xG_composite"] = (
            df["avg_xG"] * 0.40
            + df["avg_npxG"] * 0.35
            + df["avg_shots"] * 0.15
            + df["avg_xGChain"] * 0.10
        )

    if has_cols(df, ["finisher_flag", "is_attacker", "avg_xG"]):
        df["proven_finisher_in_action"] = df["finisher_flag"].astype(float) * df["is_attacker"].astype(float) * df["avg_xG"]

    if has_cols(df, ["log_market_value", "avg_xG"]):
        df["market_value_times_xG"] = df["log_market_value"] * df["avg_xG"]
    if has_cols(df, ["log_market_value", "avg_shots"]):
        df["market_value_times_shots"] = df["log_market_value"] * df["avg_shots"]

    if has_cols(df, ["international_goals", "international_caps"]):
        df["international_goals_per_cap"] = safe_divide(df["international_goals"], df["international_caps"] + 1)
    if has_cols(df, ["international_goals", "is_attacker"]):
        df["international_attacker_goals"] = df["international_goals"] * df["is_attacker"].astype(float)

    if has_cols(df, ["avg_xGChain", "avg_xGBuildup"]):
        df["xGChain_to_xGBuildup"] = safe_divide(df["avg_xGChain"], df["avg_xGBuildup"], cap=20)

    if has_cols(df, ["avg_xG", "prime_age_flag"]):
        df["prime_age_xG"] = df["avg_xG"] * df["prime_age_flag"].astype(float)
    if has_cols(df, ["avg_xG", "veteran_flag"]):
        df["veteran_age_xG"] = df["avg_xG"] * df["veteran_flag"].astype(float)

    if "minutes_played" in df.columns:
        df["played_most_of_match"] = (df["minutes_played"] >= 70).astype(float)
        if "avg_xG" in df.columns:
            df["high_minutes_xG"] = df["avg_xG"] * df["played_most_of_match"]

    if "value_pct_of_peak" in df.columns:
        df["at_peak_value"] = (df["value_pct_of_peak"] >= 0.85).astype(float)
        if "avg_xG" in df.columns:
            df["peak_value_xG"] = df["avg_xG"] * df["at_peak_value"]

    if has_cols(df, ["avg_key_passes", "avg_shots"]):
        df["key_passes_per_shot"] = safe_divide(df["avg_key_passes"], df["avg_shots"], cap=10)

    if "home_away" in df.columns:
        df["playing_at_home"] = (df["home_away"].astype(str).str.lower() == "home").astype(float)

    return df


def make_eda(train: pd.DataFrame, plots_dir: Path) -> dict[str, object]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "train_shape": list(train.shape),
        "target_rate": float(train[TARGET].mean()),
        "missing_top_15": train.isna().sum().sort_values(ascending=False).head(15).astype(int).to_dict(),
        "insights": [],
    }

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        counts = train[TARGET].value_counts().sort_index()
        axes[0].bar(["Did not score", "Scored"], counts.values, color=["#4a90d9", "#e8563a"], edgecolor="white")
        axes[0].set_title("Target distribution")
        axes[0].set_ylabel("Player appearances")
        for i, value in enumerate(counts.values):
            axes[0].text(i, value, f"{value:,}\n{value / len(train):.1%}", ha="center", va="bottom")

        if "position" in train.columns:
            rates = train.groupby("position")[TARGET].mean().sort_values(ascending=False)
            rates.plot(kind="bar", ax=axes[1], color="#5cb85c", edgecolor="white")
            axes[1].set_title("Scoring rate by position")
            axes[1].set_ylabel("Probability")
            axes[1].tick_params(axis="x", rotation=30)
            summary["position_scoring_rate"] = rates.round(5).to_dict()
            summary["insights"].append("Position is important: attacking roles usually carry a higher scoring rate than deeper roles.")
        else:
            axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / "target_and_position.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        metrics = [c for c in ["avg_xG", "avg_shots", "avg_npxG", "avg_xA", "avg_xGChain", "avg_key_passes"] if c in train.columns]
        if metrics:
            rows = int(np.ceil(len(metrics) / 3))
            fig, axes = plt.subplots(rows, 3, figsize=(16, 4 * rows))
            axes = np.array(axes).reshape(-1)
            for i, metric in enumerate(metrics):
                upper = train[metric].quantile(0.99)
                for label, group in train.groupby(TARGET):
                    axes[i].hist(group[metric].clip(upper=upper), bins=40, alpha=0.55, density=True, label="Scored" if label else "Did not score")
                axes[i].set_title(metric)
                axes[i].legend(fontsize=8)
            for ax in axes[len(metrics):]:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(plots_dir / "xg_metric_distributions.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
            summary["insights"].append("xG, shot volume, and related attacking metrics should be strong predictors of a scoring appearance.")

        numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
        clean_numeric = [c for c in numeric_cols if c not in LEAKAGE_COLUMNS + [TARGET]]
        if clean_numeric:
            corr = train[clean_numeric + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).dropna()
            top = corr.abs().sort_values(ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(10, 7))
            colors = ["#5cb85c" if corr[col] > 0 else "#e8563a" for col in top.index]
            ax.barh(range(len(top)), corr[top.index].values, color=colors)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top.index, fontsize=9)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Top numeric correlations with scoring")
            fig.tight_layout()
            fig.savefig(plots_dir / "top_correlations.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
            summary["top_correlations"] = corr[top.index].round(5).to_dict()
            summary["insights"].append("Correlation analysis helps confirm which numeric football signals are most aligned with scoring.")
    except Exception as exc:
        summary["plot_warning"] = f"EDA plots skipped: {exc}"

    return summary


def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train.copy()
    test = test.copy()

    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col == TARGET or col not in test.columns:
            continue
        fill_value = train[col].median()
        train[col] = train[col].fillna(fill_value)
        test[col] = test[col].fillna(fill_value)

    categorical_cols = [col for col in DEFAULT_CATEGORICALS if col in train.columns and col in test.columns]
    object_cols = [col for col in train.select_dtypes(include=["object", "category", "bool"]).columns if col in test.columns]
    categorical_cols = sorted(set(categorical_cols + object_cols) - set(IDENTIFIERS))

    for col in categorical_cols:
        train[col] = train[col].astype(str).fillna("Unknown")
        test[col] = test[col].astype(str).fillna("Unknown")
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        codes, uniques = pd.factorize(combined, sort=True)
        train[col] = codes[: len(train)]
        test[col] = codes[len(train):]

    drop_cols = set(IDENTIFIERS + LEAKAGE_COLUMNS + [TARGET])
    features = [col for col in train.columns if col in test.columns and col not in drop_cols]

    for col in features:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
        fill_value = train[col].median() if not train[col].dropna().empty else 0.0
        train[col] = train[col].fillna(fill_value).replace([np.inf, -np.inf], fill_value)
        test[col] = test[col].fillna(fill_value).replace([np.inf, -np.inf], fill_value)

    return train, test, features


def train_models(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, folds: int, fast: bool) -> tuple[np.ndarray, dict[str, object]]:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        import lightgbm as lgb
    except Exception:
        lgb = None
    try:
        import xgboost as xgb
    except Exception:
        xgb = None

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    metrics: dict[str, object] = {"folds": folds, "models": {}, "used_seed": SEED}

    baseline_oof = np.zeros(len(X))
    baseline_test = np.zeros(len(X_test))
    baseline_scores = []
    for tr_idx, va_idx in skf.split(X, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        )
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        baseline_oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        baseline_test += model.predict_proba(X_test)[:, 1] / folds
        baseline_scores.append(average_precision_score(y.iloc[va_idx], baseline_oof[va_idx]))
    metrics["models"]["logistic_baseline"] = {
        "fold_average_precision": [float(v) for v in baseline_scores],
        "oof_average_precision": float(average_precision_score(y, baseline_oof)),
        "oof_roc_auc": float(roc_auc_score(y, baseline_oof)),
    }

    model_predictions: list[tuple[str, np.ndarray, np.ndarray]] = [("logistic_baseline", baseline_oof, baseline_test)]

    if lgb is not None:
        lgb_oof = np.zeros(len(X))
        lgb_test = np.zeros(len(X_test))
        fold_scores = []
        estimators = 250 if fast else 1200
        for tr_idx, va_idx in skf.split(X, y):
            model = lgb.LGBMClassifier(
                objective="binary",
                metric="average_precision",
                n_estimators=estimators,
                learning_rate=0.035,
                num_leaves=63,
                min_child_samples=30,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                is_unbalance=True,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[va_idx], y.iloc[va_idx])], callbacks=[lgb.early_stopping(100, verbose=False)])
            lgb_oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
            lgb_test += model.predict_proba(X_test)[:, 1] / folds
            fold_scores.append(average_precision_score(y.iloc[va_idx], lgb_oof[va_idx]))
        metrics["models"]["lightgbm"] = {
            "fold_average_precision": [float(v) for v in fold_scores],
            "oof_average_precision": float(average_precision_score(y, lgb_oof)),
            "oof_roc_auc": float(roc_auc_score(y, lgb_oof)),
        }
        model_predictions.append(("lightgbm", lgb_oof, lgb_test))
    else:
        hgb_oof = np.zeros(len(X))
        hgb_test = np.zeros(len(X_test))
        fold_scores = []
        for tr_idx, va_idx in skf.split(X, y):
            model = HistGradientBoostingClassifier(max_iter=120 if fast else 350, learning_rate=0.05, random_state=SEED)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            hgb_oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
            hgb_test += model.predict_proba(X_test)[:, 1] / folds
            fold_scores.append(average_precision_score(y.iloc[va_idx], hgb_oof[va_idx]))
        metrics["models"]["hist_gradient_boosting_fallback"] = {
            "fold_average_precision": [float(v) for v in fold_scores],
            "oof_average_precision": float(average_precision_score(y, hgb_oof)),
            "oof_roc_auc": float(roc_auc_score(y, hgb_oof)),
        }
        model_predictions.append(("hist_gradient_boosting_fallback", hgb_oof, hgb_test))

    if xgb is not None:
        xgb_oof = np.zeros(len(X))
        xgb_test = np.zeros(len(X_test))
        fold_scores = []
        estimators = 250 if fast else 1200
        scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
        for tr_idx, va_idx in skf.split(X, y):
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                n_estimators=estimators,
                learning_rate=0.035,
                max_depth=6,
                min_child_weight=25,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[va_idx], y.iloc[va_idx])], verbose=False)
            xgb_oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
            xgb_test += model.predict_proba(X_test)[:, 1] / folds
            fold_scores.append(average_precision_score(y.iloc[va_idx], xgb_oof[va_idx]))
        metrics["models"]["xgboost"] = {
            "fold_average_precision": [float(v) for v in fold_scores],
            "oof_average_precision": float(average_precision_score(y, xgb_oof)),
            "oof_roc_auc": float(roc_auc_score(y, xgb_oof)),
        }
        model_predictions.append(("xgboost", xgb_oof, xgb_test))
    else:
        rf_oof = np.zeros(len(X))
        rf_test = np.zeros(len(X_test))
        fold_scores = []
        for tr_idx, va_idx in skf.split(X, y):
            model = RandomForestClassifier(
                n_estimators=200 if fast else 600,
                max_depth=14,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            rf_oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
            rf_test += model.predict_proba(X_test)[:, 1] / folds
            fold_scores.append(average_precision_score(y.iloc[va_idx], rf_oof[va_idx]))
        metrics["models"]["random_forest_fallback"] = {
            "fold_average_precision": [float(v) for v in fold_scores],
            "oof_average_precision": float(average_precision_score(y, rf_oof)),
            "oof_roc_auc": float(roc_auc_score(y, rf_oof)),
        }
        model_predictions.append(("random_forest_fallback", rf_oof, rf_test))

    if len(model_predictions) == 1:
        final_oof = model_predictions[0][1]
        final_test = model_predictions[0][2]
        metrics["blend"] = {"method": "single_model", "model": model_predictions[0][0]}
    else:
        best_score = -1.0
        best_weights = None
        best_oof = None
        names = [item[0] for item in model_predictions]
        grid = np.arange(0.0, 1.01, 0.05)
        for w0 in grid:
            if len(model_predictions) == 2:
                weights = [w0, 1.0 - w0]
                oof = sum(w * pred[1] for w, pred in zip(weights, model_predictions))
                score = average_precision_score(y, oof)
                if score > best_score:
                    best_score, best_weights, best_oof = score, weights, oof
            else:
                for w1 in grid:
                    w2 = 1.0 - w0 - w1
                    if w2 < -1e-9:
                        continue
                    weights = [w0, w1, w2]
                    oof = sum(w * pred[1] for w, pred in zip(weights, model_predictions))
                    score = average_precision_score(y, oof)
                    if score > best_score:
                        best_score, best_weights, best_oof = score, weights, oof
        assert best_weights is not None and best_oof is not None
        final_oof = best_oof
        final_test = sum(w * pred[2] for w, pred in zip(best_weights, model_predictions))
        metrics["blend"] = {
            "method": "cv_grid_average_precision",
            "models": names,
            "weights": {name: float(weight) for name, weight in zip(names, best_weights)},
            "oof_average_precision": float(average_precision_score(y, final_oof)),
            "oof_roc_auc": float(roc_auc_score(y, final_oof)),
        }

    return np.clip(final_test, 0, 1), metrics


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)

    train_path = Path(args.train)
    test_path = Path(args.test)
    output_path = Path(args.output)
    plots_dir = Path(args.plots_dir)
    metrics_path = Path(args.metrics)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Expected data/train.csv and data/test.csv. Copy the provided datathon CSVs into data/ first."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    require_columns(train, [TARGET, ID_COL], "train.csv")
    require_columns(test, [ID_COL], "test.csv")

    eda_summary = make_eda(train, plots_dir)

    train_fe = build_features(train)
    test_fe = build_features(test)
    train_ready, test_ready, features = preprocess(train_fe, test_fe)

    X = train_ready[features].astype(float)
    y = train_ready[TARGET].astype(int)
    X_test = test_ready[features].astype(float)

    predictions, metrics = train_models(X, y, X_test, folds=args.folds, fast=args.fast)
    metrics["data"] = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "feature_count": len(features),
        "positive_rate": float(y.mean()),
        "features": features,
    }
    metrics["eda"] = eda_summary

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: predictions})
    submission.to_csv(output_path, index=False)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved {output_path} with {len(submission):,} rows")
    print(f"Probability range: {submission[TARGET].min():.5f} to {submission[TARGET].max():.5f}")
    if "blend" in metrics:
        print("Blend:", metrics["blend"])


if __name__ == "__main__":
    main()
