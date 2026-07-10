"""
Step 3a — Individual feature sweep for MLB pitcher_walks model.

For each candidate feature independently:
  - OOF design: train on 2025, test on 2026 (primary OOS AUC)
  - Also 5-fold stratified CV on full set for in-sample AUC
  - Logistic regression for binary target (target_over)
  - Linear regression for continuous target (walks)
  - Reports: OOS AUC, IS AUC, Precision, Recall, RMSE, R²

Post-spine filter applied here (NOT during spine build):
  abs(home_run_line_point) <= 2.0  — removes ~160 live-game contaminated rows

Usage:
  python src/mlb_pitcher_walks_modeling/scripts/20260706_step3a_feature_sweep.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    auc,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE = Path.home() / "Downloads/tmp/mlb_pitcher_walks_spine.parquet"

# Features to sweep — every candidate
NUMERIC_FEATURES = [
    # Rolling walks — all windows
    "walks_roll_L1",
    "walks_roll_L3",
    "walks_roll_L5",
    "walks_roll_L10",
    "walks_roll_career",
    "walks_roll_season",
    "walks_roll_c5",
    # Related pitcher stats
    "strikeouts_roll_L5",
    "strikeouts_roll_career",
    "innings_pitched_roll_L5",
    "innings_pitched_roll_career",
    "pitches_roll_L5",
    "pitches_roll_career",
    # Opponent quality
    "opp_walks_against_season",
    # Game context
    "is_home",
    "consensus_line",
    "days_rest",
    "games_into_season",
    # Min/max market features
    "min_line",
    "max_line",
    "min_raw_implied_prob_over",
    "max_raw_implied_prob_over",
    "min_raw_implied_prob_under",
    "max_raw_implied_prob_under",
    # Team context
    "team_moneyline_odds",
    "team_run_line_point",
    # Per-book odds features
    "novig_prob_over",
]

CATEGORICAL_FEATURES = [
    "over_price_bucket_fine",
    "under_price_bucket_fine",
    "consensus_over_odds_bin",
    "consensus_over_odds_bin_granular",
    "consensus_under_odds_bin",
    "consensus_under_odds_bin_granular",
]


def encode_categorical(series: pd.Series) -> np.ndarray:
    le = LabelEncoder()
    valid = series.dropna()
    le.fit(valid)
    encoded = series.map(lambda x: le.transform([x])[0] if pd.notna(x) else np.nan)
    return encoded.values.reshape(-1, 1)


def sweep_feature(
    spine: pd.DataFrame,
    feature: str,
    is_categorical: bool,
) -> dict:
    if is_categorical:
        X_raw = encode_categorical(spine[feature])
        X_series = pd.Series(X_raw.ravel(), index=spine.index)
    else:
        X_series = spine[feature]

    valid_mask = X_series.notna() & spine["target_over"].notna() & spine["walks"].notna()
    df = spine[valid_mask].copy()
    X = X_series[valid_mask].values.reshape(-1, 1)
    y_bin = df["target_over"].values.astype(int)
    y_cont = df["walks"].values.astype(float)

    if len(df) < 50:
        return {"feature": feature, "n": len(df), "note": "insufficient data"}

    results: dict = {"feature": feature, "n": len(df)}

    # --- OOS: train 2025, test 2026 ---
    train_mask = df["season"] == 2025
    test_mask  = df["season"] == 2026

    if train_mask.sum() > 20 and test_mask.sum() > 20:
        X_tr, X_te = X[train_mask], X[test_mask]
        y_bin_tr, y_bin_te = y_bin[train_mask], y_bin[test_mask]
        y_cont_tr, y_cont_te = y_cont[train_mask], y_cont[test_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Logistic — OOS
        try:
            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(X_tr_s, y_bin_tr)
            y_prob = lr.predict_proba(X_te_s)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            results["oos_auc"]       = roc_auc_score(y_bin_te, y_prob)
            results["oos_precision"] = precision_score(y_bin_te, y_pred, zero_division=0)
            results["oos_recall"]    = recall_score(y_bin_te, y_pred, zero_division=0)
        except Exception:
            results["oos_auc"] = np.nan

        # Linear — OOS
        try:
            lin = LinearRegression()
            lin.fit(X_tr_s, y_cont_tr)
            y_hat = lin.predict(X_te_s)
            results["oos_rmse"] = np.sqrt(mean_squared_error(y_cont_te, y_hat))
            results["oos_r2"]   = r2_score(y_cont_te, y_hat)
        except Exception:
            results["oos_rmse"] = np.nan
    else:
        results["oos_auc"] = np.nan

    # --- IS: 5-fold stratified CV on 2025+2026 ---
    is_mask = df["season"].isin([2025, 2026])
    X_is    = X[is_mask]
    y_is    = y_bin[is_mask]

    if is_mask.sum() > 50:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        is_aucs = []
        for tr_idx, te_idx in skf.split(X_is, y_is):
            scaler_is = StandardScaler()
            X_tr_is = scaler_is.fit_transform(X_is[tr_idx])
            X_te_is = scaler_is.transform(X_is[te_idx])
            try:
                lr_is = LogisticRegression(max_iter=1000, C=1.0)
                lr_is.fit(X_tr_is, y_is[tr_idx])
                prob_is = lr_is.predict_proba(X_te_is)[:, 1]
                is_aucs.append(roc_auc_score(y_is[te_idx], prob_is))
            except Exception:
                pass
        results["is_auc"] = np.mean(is_aucs) if is_aucs else np.nan
    else:
        results["is_auc"] = np.nan

    return results


def main() -> None:
    print("Loading spine...")
    spine = pd.read_parquet(LOCAL_SPINE)

    # Post-spine filter — live-game contamination
    before = len(spine)
    spine = spine[spine["home_run_line_point"].abs() <= 2.0].copy()
    print(f"Post-filter: {before:,} → {len(spine):,} rows ({before - len(spine)} dropped)")
    print(f"Seasons: {sorted(spine['season'].unique())}")
    print(f"2025 rows: {(spine['season']==2025).sum():,}  |  2026 rows: {(spine['season']==2026).sum():,}\n")

    all_results = []

    print("Running numeric feature sweep...")
    for feat in NUMERIC_FEATURES:
        if feat not in spine.columns:
            print(f"  SKIP {feat} (not in spine)")
            continue
        r = sweep_feature(spine, feat, is_categorical=False)
        all_results.append(r)
        oos = r.get("oos_auc", np.nan)
        is_ = r.get("is_auc", np.nan)
        print(f"  {feat:<40}  OOS AUC={oos:.4f}  IS AUC={is_:.4f}")

    print("\nRunning categorical feature sweep...")
    for feat in CATEGORICAL_FEATURES:
        if feat not in spine.columns:
            print(f"  SKIP {feat} (not in spine)")
            continue
        r = sweep_feature(spine, feat, is_categorical=True)
        all_results.append(r)
        oos = r.get("oos_auc", np.nan)
        is_ = r.get("is_auc", np.nan)
        print(f"  {feat:<40}  OOS AUC={oos:.4f}  IS AUC={is_:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("oos_auc", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("STEP 3a FEATURE RANKING — sorted by OOS AUC (train 2025 → test 2026)")
    print("=" * 90)

    display_cols = ["feature", "n", "oos_auc", "is_auc", "oos_precision", "oos_recall", "oos_rmse", "oos_r2"]
    display_cols = [c for c in display_cols if c in results_df.columns]
    with pd.option_context("display.max_rows", 100, "display.width", 120, "display.float_format", "{:.4f}".format):
        print(results_df[display_cols].to_string(index=False))

    # Save for HTML
    out = Path.home() / "Downloads/tmp/mlb_pitcher_walks_step3a_results.csv"
    results_df.to_csv(out, index=False)
    print(f"\nResults saved → {out}")

    # Spot-check: Freddy Peralta rolling walk features
    print("\n=== Spot-check: Freddy Peralta (rolling walks) ===")
    fp = spine[spine["player_name"].str.lower().str.contains("peralta", na=False)].drop_duplicates("game_date").sort_values("game_date")
    cols = ["game_date", "season", "walks", "walks_roll_L1", "walks_roll_L5", "walks_roll_career", "walks_roll_season", "line", "novig_prob_over"]
    print(fp[cols].tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
