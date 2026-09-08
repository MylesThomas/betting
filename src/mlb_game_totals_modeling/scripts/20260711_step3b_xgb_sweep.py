"""
Step 3b — XGBoost Multi-Feature Model for MLB Game Totals.

Trains XGBoost with all features combined using OOF time-based cross-validation.
GroupKFold on game_pk to prevent leakage.

Models:
  1. XGB Classifier → hit_over  (AUC, precision, recall)
  2. XGB Classifier → hit_under (AUC, precision, recall)
  3. XGB Regressor  → total_runs (RMSE, MAE, r²)

Also runs a sanity comparison: line-only logistic regression as baseline.

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step3b_xgb_sweep.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_DIR  = Path.home() / "Downloads/tmp/mlb_game_totals"
LOCAL_PATH = LOCAL_DIR / "game_totals_spine.parquet"

N_FOLDS = 5
SEED    = 42


def load_spine() -> pd.DataFrame:
    df = pd.read_parquet(LOCAL_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["hit_over"]  = df["hit_over"].astype(int)
    df["hit_under"] = df["hit_under"].astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build derived features and encode categoricals."""
    df = df.copy()

    df["combined_rs_L5"]     = df["home_rs_L5"]    + df["away_rs_L5"]
    df["combined_ra_L5"]     = df["home_ra_L5"]    + df["away_ra_L5"]
    df["combined_rs_L10"]    = df["home_rs_L10"]   + df["away_rs_L10"]
    df["combined_ra_L10"]    = df["home_ra_L10"]   + df["away_ra_L10"]
    df["combined_rs_career"] = df["home_rs_career"] + df["away_rs_career"]
    df["combined_ra_career"] = df["home_ra_career"] + df["away_ra_career"]
    df["line_spread"]        = df["max_line"]      - df["min_line"]
    df["prob_spread_over"]   = df["max_raw_implied_prob_over"] - df["min_raw_implied_prob_over"]

    cat_cols = [
        "consensus_over_odds_bin", "consensus_over_odds_bin_granular",
        "consensus_under_odds_bin", "consensus_under_odds_bin_granular",
    ]
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    rolling = [
        "home_rs_L1",  "home_rs_L3",  "home_rs_L5",  "home_rs_L10", "home_rs_L20",
        "home_rs_season", "home_rs_career",
        "home_ra_L1",  "home_ra_L3",  "home_ra_L5",  "home_ra_L10", "home_ra_L20",
        "home_ra_season", "home_ra_career",
        "away_rs_L1",  "away_rs_L3",  "away_rs_L5",  "away_rs_L10", "away_rs_L20",
        "away_rs_season", "away_rs_career",
        "away_ra_L1",  "away_ra_L3",  "away_ra_L5",  "away_ra_L10", "away_ra_L20",
        "away_ra_season", "away_ra_career",
    ]
    combined = [
        "combined_rs_L5", "combined_ra_L5", "combined_rs_L10", "combined_ra_L10",
        "combined_rs_career", "combined_ra_career",
    ]
    market = [
        "line", "consensus_line", "min_line", "max_line",
        "novig_prob_over",
        "min_raw_implied_prob_over", "max_raw_implied_prob_over",
        "min_raw_implied_prob_under", "max_raw_implied_prob_under",
        "line_spread", "prob_spread_over", "n_books_total",
    ]
    categorical_enc = [
        "consensus_over_odds_bin_enc", "consensus_over_odds_bin_granular_enc",
        "consensus_under_odds_bin_enc", "consensus_under_odds_bin_granular_enc",
    ]
    temporal = [
        "park_factor", "month", "day_of_week", "is_weekend", "season", "line_is_integer",
    ]

    all_cols = rolling + combined + market + categorical_enc + temporal
    return [c for c in all_cols if c in df.columns]


def oof_xgb_classifier(df: pd.DataFrame, feature_cols: list[str], target: str,
                        label: str) -> None:
    valid = df[feature_cols + [target, "game_pk", "game_date"]].dropna(
        subset=feature_cols + [target]
    )
    valid = valid.sort_values("game_date").reset_index(drop=True)
    X = valid[feature_cols].values.astype(float)
    y = valid[target].values
    groups = valid["game_pk"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_proba = np.zeros(len(valid))

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        clf = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            eval_metric="logloss", random_state=SEED, verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        oof_proba[test_idx] = clf.predict_proba(X_te)[:, 1]

    auc   = roc_auc_score(y, oof_proba)
    preds = (oof_proba > 0.5).astype(int)
    prec  = precision_score(y, preds, zero_division=0)
    rec   = recall_score(y, preds, zero_division=0)

    print(f"  {label}: AUC={auc:.4f}  precision={prec:.4f}  recall={rec:.4f}  n={len(valid)}")

    # Feature importance
    # Retrain on all data for importance
    clf_full = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        eval_metric="logloss", random_state=SEED, verbosity=0,
    )
    clf_full.fit(X, y)
    importance = pd.Series(clf_full.feature_importances_, index=feature_cols)
    top10 = importance.sort_values(ascending=False).head(10)
    print(f"    Top-10 features (gain):")
    for feat, imp in top10.items():
        print(f"      {feat:<40s} {imp:.4f}")

    return auc, oof_proba, valid


def oof_xgb_regressor(df: pd.DataFrame, feature_cols: list[str]) -> None:
    valid = df[feature_cols + ["total_runs", "game_pk", "game_date"]].dropna(
        subset=feature_cols + ["total_runs"]
    )
    valid = valid.sort_values("game_date").reset_index(drop=True)
    X = valid[feature_cols].values.astype(float)
    y = valid["total_runs"].values.astype(float)
    groups = valid["game_pk"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred = np.zeros(len(valid))

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        reg = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            random_state=SEED, verbosity=0,
        )
        reg.fit(X_tr, y_tr)
        oof_pred[test_idx] = reg.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y, oof_pred))
    mae  = mean_absolute_error(y, oof_pred)
    ss_res = np.sum((y - oof_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print(f"  XGB Regressor (total_runs): RMSE={rmse:.4f}  MAE={mae:.4f}  r²={r2:.4f}  n={len(valid)}")
    print(f"  Baseline RMSE (predict mean): {y.std():.4f}")

    # Feature importance
    reg_full = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        random_state=SEED, verbosity=0,
    )
    reg_full.fit(X, y)
    importance = pd.Series(reg_full.feature_importances_, index=feature_cols)
    top10 = importance.sort_values(ascending=False).head(10)
    print(f"    Top-10 features (gain):")
    for feat, imp in top10.items():
        print(f"      {feat:<40s} {imp:.4f}")

    return rmse, r2, oof_pred, valid


def main() -> None:
    print("Loading spine...")
    df = load_spine()
    df = build_feature_matrix(df)
    feature_cols = get_feature_cols(df)
    print(f"  {len(df)} rows, {df['game_pk'].nunique()} games, {len(feature_cols)} features\n")

    # Baseline: line-only logistic regression
    print("=== BASELINE: Line-only logistic regression ===")
    for target in ["hit_over", "hit_under"]:
        valid = df[["line", target, "game_pk", "game_date"]].dropna()
        valid = valid.sort_values("game_date").reset_index(drop=True)
        X = valid[["line"]].values
        y = valid[target].values
        groups = valid["game_pk"].values

        gkf = GroupKFold(n_splits=N_FOLDS)
        oof_p = np.zeros(len(valid))
        for tr, te in gkf.split(X, y, groups):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr].astype(float))
            Xte = sc.transform(X[te].astype(float))
            clf = LogisticRegression(max_iter=500)
            clf.fit(Xtr, y[tr])
            oof_p[te] = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y, oof_p)
        print(f"  line → {target}: AUC={auc:.4f}")

    print("\n=== XGB ALL FEATURES ===")
    auc_over, oof_over, df_over   = oof_xgb_classifier(df, feature_cols, "hit_over",  "XGB Classifier → hit_over")
    auc_under, oof_under, df_under = oof_xgb_classifier(df, feature_cols, "hit_under", "XGB Classifier → hit_under")
    rmse, r2, oof_runs, df_runs   = oof_xgb_regressor(df, feature_cols)

    # XGB without market features (rolling + park + temporal only)
    no_market_cols = [c for c in feature_cols if c not in [
        "line", "consensus_line", "min_line", "max_line",
        "novig_prob_over", "min_raw_implied_prob_over", "max_raw_implied_prob_over",
        "min_raw_implied_prob_under", "max_raw_implied_prob_under",
        "line_spread", "prob_spread_over", "n_books_total",
        "consensus_over_odds_bin_enc", "consensus_over_odds_bin_granular_enc",
        "consensus_under_odds_bin_enc", "consensus_under_odds_bin_granular_enc",
    ]]
    print(f"\n=== XGB ROLLING + PARK/TEMPORAL ONLY (no market features, {len(no_market_cols)} cols) ===")
    oof_xgb_classifier(df, no_market_cols, "hit_over",  "XGB no-market → hit_over")
    oof_xgb_classifier(df, no_market_cols, "hit_under", "XGB no-market → hit_under")
    oof_xgb_regressor(df, no_market_cols)

    print("\n=== SUMMARY ===")
    print(f"  Individual best feature AUC (Step 3a): 0.5159 (line_is_integer)")
    print(f"  XGB all-features hit_over  AUC: {auc_over:.4f}")
    print(f"  XGB all-features hit_under AUC: {auc_under:.4f}")
    print(f"  XGB all-features total_runs r²: {r2:.4f}  RMSE: {rmse:.4f}")
    print(f"  Baseline RMSE (mean): ~4.406")


if __name__ == "__main__":
    main()
