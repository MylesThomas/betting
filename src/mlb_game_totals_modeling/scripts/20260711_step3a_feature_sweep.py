"""
Step 3a — Individual Feature Sweep for MLB Game Totals.

Tests each candidate feature independently using OOF time-based cross-validation.
Groups by game_pk so the same game never appears in both train and test.

Targets:
  - hit_over  (binary logistic regression → AUC, precision, recall)
  - total_runs (linear regression → RMSE, MAE, r²)

Output:
  Printed sweep table + saved to ~/Downloads/tmp/mlb_game_totals/step3a_sweep.csv

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step3a_feature_sweep.py
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_DIR  = Path.home() / "Downloads/tmp/mlb_game_totals"
LOCAL_PATH = LOCAL_DIR / "game_totals_spine.parquet"
OUT_CSV    = LOCAL_DIR / "step3a_sweep.csv"

N_FOLDS = 5


def load_spine() -> pd.DataFrame:
    df = pd.read_parquet(LOCAL_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["hit_over"]  = df["hit_over"].astype(int)
    df["hit_under"] = df["hit_under"].astype(int)
    return df


def oof_logistic(df: pd.DataFrame, feature: str, target: str) -> dict:
    """OOF logistic regression for a single feature vs binary target."""
    valid = df[[feature, target, "game_pk", "game_date"]].dropna()
    if len(valid) < 500:
        return {"n": len(valid), "auc": np.nan, "precision": np.nan, "recall": np.nan}

    valid = valid.sort_values("game_date").reset_index(drop=True)
    groups = valid["game_pk"].values

    if valid[feature].dtype == object:
        le = LabelEncoder()
        X = le.fit_transform(valid[feature]).reshape(-1, 1)
    else:
        X = valid[[feature]].values.astype(float)

    y = valid[target].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_proba = np.zeros(len(valid))

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr.astype(float))
        X_te_sc = scaler.transform(X_te.astype(float))

        clf = LogisticRegression(max_iter=500)
        if len(np.unique(y_tr)) < 2:
            oof_proba[test_idx] = y_tr.mean()
            continue
        clf.fit(X_tr_sc, y_tr)
        oof_proba[test_idx] = clf.predict_proba(X_te_sc)[:, 1]

    auc = roc_auc_score(y, oof_proba)
    preds = (oof_proba > 0.5).astype(int)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)

    return {"n": len(valid), "auc": round(auc, 4), "precision": round(prec, 4), "recall": round(rec, 4)}


def oof_linear(df: pd.DataFrame, feature: str) -> dict:
    """OOF linear regression for a single feature vs total_runs."""
    valid = df[[feature, "total_runs", "game_pk", "game_date"]].dropna()
    if valid[feature].dtype == object or len(valid) < 500:
        return {"n_reg": len(valid), "rmse": np.nan, "mae": np.nan, "r2": np.nan}

    valid = valid.sort_values("game_date").reset_index(drop=True)
    X = valid[[feature]].values.astype(float)
    y = valid["total_runs"].values.astype(float)
    groups = valid["game_pk"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred = np.zeros(len(valid))

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        reg = LinearRegression()
        reg.fit(X_tr_sc, y_tr)
        oof_pred[test_idx] = reg.predict(X_te_sc)

    rmse = np.sqrt(mean_squared_error(y, oof_pred))
    mae  = mean_absolute_error(y, oof_pred)
    ss_res = np.sum((y - oof_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"n_reg": len(valid), "rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def main() -> None:
    print("Loading spine...")
    df = load_spine()
    print(f"  {len(df)} rows, {df['game_pk'].nunique()} unique games\n")

    # Derive combined features
    df["combined_rs_L5"]     = df["home_rs_L5"]    + df["away_rs_L5"]
    df["combined_ra_L5"]     = df["home_ra_L5"]    + df["away_ra_L5"]
    df["combined_rs_L10"]    = df["home_rs_L10"]   + df["away_rs_L10"]
    df["combined_ra_L10"]    = df["home_ra_L10"]   + df["away_ra_L10"]
    df["combined_rs_career"] = df["home_rs_career"] + df["away_rs_career"]
    df["combined_ra_career"] = df["home_ra_career"] + df["away_ra_career"]
    df["line_spread"]        = df["max_line"]      - df["min_line"]
    df["prob_spread_over"]   = df["max_raw_implied_prob_over"] - df["min_raw_implied_prob_over"]

    rolling_features = [
        "home_rs_L1",  "home_rs_L3",  "home_rs_L5",  "home_rs_L10", "home_rs_L20",
        "home_rs_season", "home_rs_career",
        "home_ra_L1",  "home_ra_L3",  "home_ra_L5",  "home_ra_L10", "home_ra_L20",
        "home_ra_season", "home_ra_career",
        "away_rs_L1",  "away_rs_L3",  "away_rs_L5",  "away_rs_L10", "away_rs_L20",
        "away_rs_season", "away_rs_career",
        "away_ra_L1",  "away_ra_L3",  "away_ra_L5",  "away_ra_L10", "away_ra_L20",
        "away_ra_season", "away_ra_career",
    ]
    combined_features = [
        "combined_rs_L5", "combined_ra_L5", "combined_rs_L10", "combined_ra_L10",
        "combined_rs_career", "combined_ra_career",
    ]
    market_features = [
        "line", "consensus_line", "min_line", "max_line",
        "novig_prob_over", "novig_prob_under",
        "min_raw_implied_prob_over", "max_raw_implied_prob_over",
        "min_raw_implied_prob_under", "max_raw_implied_prob_under",
        "line_spread", "prob_spread_over",
        "n_books_total",
    ]
    categorical_features = [
        "consensus_over_odds_bin", "consensus_over_odds_bin_granular",
        "consensus_under_odds_bin", "consensus_under_odds_bin_granular",
    ]
    temporal_features = [
        "park_factor", "month", "day_of_week", "is_weekend", "season", "line_is_integer",
    ]

    all_features = (rolling_features + combined_features + market_features +
                    categorical_features + temporal_features)

    rows = []
    print(f"Running sweep over {len(all_features)} features...\n")

    for i, feat in enumerate(all_features):
        if feat not in df.columns:
            print(f"  [{i+1}/{len(all_features)}] {feat} — MISSING, skip")
            continue

        print(f"  [{i+1}/{len(all_features)}] {feat}", end="", flush=True)

        over_res  = oof_logistic(df, feat, "hit_over")
        under_res = oof_logistic(df, feat, "hit_under")
        lin_res   = oof_linear(df, feat)

        row = {
            "feature":    feat,
            "n":          over_res["n"],
            "auc_over":   over_res["auc"],
            "auc_under":  under_res["auc"],
            "prec_over":  over_res["precision"],
            "prec_under": under_res["precision"],
            "rec_over":   over_res["recall"],
            "rec_under":  under_res["recall"],
            "rmse":       lin_res["rmse"],
            "mae":        lin_res["mae"],
            "r2":         lin_res["r2"],
        }
        rows.append(row)
        print(f"  → auc_over={over_res['auc']} | auc_under={under_res['auc']} | r²={lin_res['r2']}")

    results = pd.DataFrame(rows)
    results = results.sort_values("auc_over", ascending=False)

    print("\n\n=== STEP 3a SWEEP — Sorted by auc_over ===")
    print(results.to_string(index=False))

    print("\n=== TOP 10 by auc_over ===")
    print(results.head(10)[["feature", "auc_over", "auc_under", "r2"]].to_string(index=False))

    print("\n=== TOP 10 by auc_under ===")
    print(results.sort_values("auc_under", ascending=False).head(10)[["feature", "auc_over", "auc_under", "r2"]].to_string(index=False))

    print("\n=== TOP 10 by r² ===")
    print(results.sort_values("r2", ascending=False).head(10)[["feature", "r2", "rmse", "mae"]].to_string(index=False))

    results.to_csv(OUT_CSV, index=False)
    print(f"\nSaved sweep → {OUT_CSV}")

    over_rate  = df["hit_over"].mean()
    under_rate = df["hit_under"].mean()
    total_mean = df["total_runs"].mean()
    total_std  = df["total_runs"].std()
    print(f"\nBaseline — over rate: {over_rate:.3f}, under rate: {under_rate:.3f}")
    print(f"Baseline RMSE (predict mean): {total_std:.3f}  (mean total: {total_mean:.2f})")


if __name__ == "__main__":
    main()
