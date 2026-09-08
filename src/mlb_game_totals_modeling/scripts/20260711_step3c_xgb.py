"""
Step 3c — XGBoost model for MLB Game Totals.

Trains XGBoost on hit_over at (game_pk, line) grain with GroupKFold.
Compares to Step 3b logistic baseline. Reports AUC, Brier, calibration by line.

Output:
  ~/Downloads/tmp/mlb_game_totals/step3c_xgb_oof.parquet

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step3c_xgb.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE = Path.home() / "Downloads/tmp/mlb_game_totals/game_totals_spine.parquet"
LOCAL_OUT   = Path.home() / "Downloads/tmp/mlb_game_totals/step3c_xgb_oof.parquet"

N_SPLITS = 5

NUMERIC_FEATURES = [
    "home_rs_L1", "home_rs_L3", "home_rs_L5", "home_rs_L10", "home_rs_L20",
    "home_rs_season", "home_rs_career",
    "home_ra_L1", "home_ra_L3", "home_ra_L5", "home_ra_L10", "home_ra_L20",
    "home_ra_season", "home_ra_career",
    "away_rs_L1", "away_rs_L3", "away_rs_L5", "away_rs_L10", "away_rs_L20",
    "away_rs_season", "away_rs_career",
    "away_ra_L1", "away_ra_L3", "away_ra_L5", "away_ra_L10", "away_ra_L20",
    "away_ra_season", "away_ra_career",
    "novig_prob_over",
    "consensus_line", "line",
    "min_line", "max_line",
    "min_raw_implied_prob_over", "max_raw_implied_prob_over",
    "min_raw_implied_prob_under", "max_raw_implied_prob_under",
    "park_factor", "month", "day_of_week", "season", "is_weekend", "line_is_integer",
]

CATEGORICAL_FEATURES = [
    "consensus_over_odds_bin",
    "consensus_over_odds_bin_granular",
    "consensus_under_odds_bin",
    "consensus_under_odds_bin_granular",
]


def dedup_to_game_line(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"novig_prob_over": "mean", "novig_prob_under": "mean"}
    first_cols = {c: "first" for c in df.columns
                  if c not in agg and c not in ("game_pk", "line", "novig_prob_over", "novig_prob_under")}
    return df.groupby(["game_pk", "line"]).agg({**first_cols, **agg}).reset_index()


def main() -> None:
    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not installed — trying sklearn GradientBoostingClassifier")
        from sklearn.ensemble import GradientBoostingClassifier
        xgb = None

    print("Loading spine...")
    df_all = pd.read_parquet(LOCAL_SPINE)
    print(f"  {len(df_all)} rows, {df_all.game_pk.nunique()} games")

    print("Deduplicating to (game_pk, line) grain...")
    df = dedup_to_game_line(df_all)
    df["hit_over"]  = df["hit_over"].astype(float)
    df["hit_under"] = df["hit_under"].astype(float)
    print(f"  → {len(df)} rows")

    # Encode categoricals as integers
    le = LabelEncoder()
    for cat in CATEGORICAL_FEATURES:
        df[cat + "_enc"] = le.fit_transform(df[cat].fillna("unknown").astype(str))

    enc_feats = [c + "_enc" for c in CATEGORICAL_FEATURES]
    all_feats = NUMERIC_FEATURES + enc_feats

    valid = df.dropna(subset=NUMERIC_FEATURES + ["hit_over"])
    X = valid[all_feats].values.astype(float)
    y = valid["hit_over"].values.astype(int)
    groups = valid["game_pk"].values

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_probs = np.full(len(y), np.nan)

    for fold, (tr, val) in enumerate(gkf.split(X, y, groups)):
        print(f"  Fold {fold+1}/{N_SPLITS}...")
        if xgb is not None:
            clf = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                verbosity=0,
                use_label_encoder=False,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8)
        clf.fit(X[tr], y[tr])
        oof_probs[val] = clf.predict_proba(X[val])[:, 1]

    mask = ~np.isnan(oof_probs)
    auc   = roc_auc_score(y[mask], oof_probs[mask])
    brier = brier_score_loss(y[mask], oof_probs[mask])
    print(f"\nXGBoost hit_over → AUC={auc:.4f}, Brier={brier:.4f}")

    valid = valid.copy()
    valid["xgb_p_over"]  = oof_probs
    valid["xgb_p_under"] = 1.0 - oof_probs

    # Calibration by line
    print("\nXGBoost calibration by line (vs actual and market):")
    by_line = valid.groupby("line").agg(
        n=("hit_over","count"),
        actual_over=("hit_over","mean"),
        xgb_over=("xgb_p_over","mean"),
        market_over=("novig_prob_over","mean"),
    ).round(3)
    print(by_line[by_line["n"] >= 30].to_string())

    # Edge distribution
    valid["over_edge"]  = valid["xgb_p_over"]  - valid["novig_prob_over"]
    valid["under_edge"] = valid["xgb_p_under"] - valid["novig_prob_under"]

    print("\nEdge distribution (over):")
    print(valid["over_edge"].describe().round(4))
    print("\nEdge distribution (under):")
    print(valid["under_edge"].describe().round(4))

    # Feature importance
    print("\nFeature importance (last fold):")
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=all_feats).sort_values(ascending=False)
    print(feat_imp.head(20).round(4).to_string())

    # Spot check NYY @ BOS
    print("\nSpot check NYY @ BOS:")
    spot_pks = df_all[
        (df_all["home_team"] == "Boston Red Sox") &
        (df_all["away_team"] == "New York Yankees")
    ]["game_pk"].unique()
    spot = valid[valid["game_pk"].isin(spot_pks)].sort_values("game_date")
    print(spot[["game_date","line","total_runs","xgb_p_over","over_edge","novig_prob_over","hit_over"]].head(10).to_string(index=False))

    valid.to_parquet(LOCAL_OUT, index=False)
    print(f"\nSaved → {LOCAL_OUT}")


if __name__ == "__main__":
    main()
