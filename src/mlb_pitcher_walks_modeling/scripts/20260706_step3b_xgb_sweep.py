"""
Step 3b — XGBoost individual feature sweep + multi-feature model.

Same OOF design as Step 3a (train 2025, test 2026).
Reports per-feature AUC alongside logistic regression results for comparison.
Also runs a multi-feature XGBoost with the top features from Step 3a
and reports feature importances.

Usage:
  python src/mlb_pitcher_walks_modeling/scripts/20260706_step3b_xgb_sweep.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE   = Path.home() / "Downloads/tmp/mlb_pitcher_walks_spine.parquet"
STEP3A_RESULTS = Path.home() / "Downloads/tmp/mlb_pitcher_walks_step3a_results.csv"

NUMERIC_FEATURES = [
    "walks_roll_L1", "walks_roll_L3", "walks_roll_L5", "walks_roll_L10",
    "walks_roll_career", "walks_roll_season", "walks_roll_c5",
    "strikeouts_roll_L5", "strikeouts_roll_career",
    "innings_pitched_roll_L5", "innings_pitched_roll_career",
    "pitches_roll_L5", "pitches_roll_career",
    "opp_walks_against_season",
    "is_home", "consensus_line", "days_rest", "games_into_season",
    "min_line", "max_line",
    "min_raw_implied_prob_over", "max_raw_implied_prob_over",
    "min_raw_implied_prob_under", "max_raw_implied_prob_under",
    "team_moneyline_odds", "team_run_line_point",
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

# Top features from Step 3a for multi-feature model
TOP_FEATURES_NUMERIC = [
    "novig_prob_over",
    "min_raw_implied_prob_over",
    "max_raw_implied_prob_over",
    "min_raw_implied_prob_under",
    "max_raw_implied_prob_under",
    "consensus_line",
    "walks_roll_career",
    "walks_roll_season",
    "games_into_season",
    "min_line",
    "max_line",
]

TOP_FEATURES_CATEGORICAL = [
    "consensus_under_odds_bin",
    "consensus_over_odds_bin",
    "consensus_under_odds_bin_granular",
    "consensus_over_odds_bin_granular",
]


def encode_cat(series: pd.Series) -> np.ndarray:
    le = LabelEncoder()
    valid = series.dropna()
    le.fit(valid)
    return series.map(lambda x: le.transform([x])[0] if pd.notna(x) else -1).values


def sweep_xgb_single(spine: pd.DataFrame, feature: str, is_cat: bool) -> dict:
    if is_cat:
        X_raw = encode_cat(spine[feature])
        X_series = pd.Series(X_raw, index=spine.index)
    else:
        X_series = spine[feature]

    valid_mask = X_series.notna() & spine["target_over"].notna()
    df = spine[valid_mask].copy()
    X = X_series[valid_mask].values.reshape(-1, 1)
    y = df["target_over"].values.astype(int)

    if len(df) < 50:
        return {"feature": feature, "xgb_oos_auc": np.nan}

    tr = df["season"] == 2025
    te = df["season"] == 2026
    if tr.sum() < 20 or te.sum() < 20:
        return {"feature": feature, "xgb_oos_auc": np.nan}

    clf = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    try:
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        return {"feature": feature, "xgb_oos_auc": roc_auc_score(y[te], prob)}
    except Exception:
        return {"feature": feature, "xgb_oos_auc": np.nan}


def run_multi_feature_model(spine: pd.DataFrame) -> tuple[float, pd.Series]:
    """Multi-feature XGBoost with top features from Step 3a."""
    df = spine.copy()

    # Encode categoricals
    for c in TOP_FEATURES_CATEGORICAL:
        if c in df.columns:
            df[c + "_enc"] = encode_cat(df[c])
        else:
            df[c + "_enc"] = 0

    num_cols = [c for c in TOP_FEATURES_NUMERIC if c in df.columns]
    cat_cols = [c + "_enc" for c in TOP_FEATURES_CATEGORICAL]
    all_feats = num_cols + cat_cols

    valid_mask = df[all_feats + ["target_over"]].notna().all(axis=1)
    df_v = df[valid_mask].copy()
    X = df_v[all_feats].values
    y = df_v["target_over"].values.astype(int)

    tr = df_v["season"] == 2025
    te = df_v["season"] == 2026

    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    clf.fit(X[tr.values], y[tr.values])
    prob = clf.predict_proba(X[te.values])[:, 1]
    oos_auc = roc_auc_score(y[te.values], prob)

    importances = pd.Series(clf.feature_importances_, index=all_feats).sort_values(ascending=False)
    return oos_auc, importances


def main() -> None:
    print("Loading spine...")
    spine = pd.read_parquet(LOCAL_SPINE)
    spine = spine[spine["home_run_line_point"].abs() <= 2.0].copy()
    print(f"Filtered spine: {len(spine):,} rows")

    # --- Individual XGBoost sweep ---
    print("\nRunning XGBoost single-feature sweep...")
    xgb_results = []
    for feat in NUMERIC_FEATURES:
        if feat not in spine.columns:
            continue
        r = sweep_xgb_single(spine, feat, is_cat=False)
        xgb_results.append(r)
        print(f"  {feat:<40}  XGB OOS AUC={r['xgb_oos_auc']:.4f}")

    for feat in CATEGORICAL_FEATURES:
        if feat not in spine.columns:
            continue
        r = sweep_xgb_single(spine, feat, is_cat=True)
        xgb_results.append(r)
        print(f"  {feat:<40}  XGB OOS AUC={r['xgb_oos_auc']:.4f}")

    xgb_df = pd.DataFrame(xgb_results).sort_values("xgb_oos_auc", ascending=False).reset_index(drop=True)

    # Merge with Step 3a logistic regression results
    if STEP3A_RESULTS.exists():
        lr_df = pd.read_csv(STEP3A_RESULTS)[["feature", "oos_auc"]].rename(columns={"oos_auc": "lr_oos_auc"})
        comparison = xgb_df.merge(lr_df, on="feature", how="left")
    else:
        comparison = xgb_df.rename(columns={"xgb_oos_auc": "xgb_oos_auc"})

    print("\n" + "=" * 80)
    print("STEP 3b — XGBoost vs Logistic Regression (OOS AUC, train 2025 → test 2026)")
    print("=" * 80)
    with pd.option_context("display.max_rows", 50, "display.width", 100, "display.float_format", "{:.4f}".format):
        print(comparison.to_string(index=False))

    # --- Multi-feature XGBoost ---
    print("\nRunning multi-feature XGBoost (top features from Step 3a)...")
    oos_auc, importances = run_multi_feature_model(spine)
    print(f"\nMulti-feature XGBoost OOS AUC (train 2025 → test 2026): {oos_auc:.4f}")
    print("\nFeature importances:")
    print(importances.round(4).to_string())

    # Save
    out = Path.home() / "Downloads/tmp/mlb_pitcher_walks_step3b_results.csv"
    comparison.to_csv(out, index=False)
    imp_out = Path.home() / "Downloads/tmp/mlb_pitcher_walks_step3b_importances.csv"
    importances.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(imp_out, index=False)
    print(f"\nResults saved → {out}")
    print(f"Importances saved → {imp_out}")


if __name__ == "__main__":
    main()
