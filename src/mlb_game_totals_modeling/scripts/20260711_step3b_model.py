"""
Step 3b — Multivariate OOF model for MLB Game Totals.

Two approaches:
  1. OLS on total_runs (continuous) → convert to P(over line) via Poisson CDF
  2. Logistic regression on hit_over (binary) with top features from Step 3a

GroupKFold by game_pk. Reports AUC, Brier score, RMSE/MAE for OLS.
Computes OOF p_model per (game_pk, line) and saves to spine grain for edge calc.

Output:
  ~/Downloads/tmp/mlb_game_totals/step3b_oof_predictions.parquet

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step3b_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (brier_score_loss, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE = Path.home() / "Downloads/tmp/mlb_game_totals/game_totals_spine.parquet"
LOCAL_OUT   = Path.home() / "Downloads/tmp/mlb_game_totals/step3b_oof_predictions.parquet"

N_SPLITS = 5

# Game-level features (book-invariant — same value for all books at the same game)
GAME_FEATURES = [
    "home_rs_L5", "home_rs_L10", "home_rs_season",
    "home_ra_L5", "home_ra_L10", "home_ra_L20", "home_ra_season",
    "away_rs_L3", "away_rs_L5", "away_rs_season",
    "away_ra_L3", "away_ra_L5", "away_ra_L10", "away_ra_season",
    "park_factor",
    "month",
    "day_of_week",
    "line_is_integer",
    "line",  # include line in logistic model but NOT in OLS total_runs model
]

# Features for OLS total_runs prediction (exclude line — we want a line-agnostic run total pred)
OLS_FEATURES = [
    "home_rs_L5", "home_rs_L10", "home_rs_season",
    "home_ra_L5", "home_ra_L10", "home_ra_L20", "home_ra_season",
    "away_rs_L3", "away_rs_L5", "away_rs_season",
    "away_ra_L3", "away_ra_L5", "away_ra_L10", "away_ra_season",
    "park_factor",
    "month",
    "day_of_week",
]

CATEGORICAL_FEATURES = ["consensus_under_odds_bin_granular"]


def dedup_to_game_line(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"novig_prob_over": "mean", "novig_prob_under": "mean"}
    first_cols = {c: "first" for c in df.columns
                  if c not in agg and c not in ("game_pk", "line", "novig_prob_over", "novig_prob_under")}
    return df.groupby(["game_pk", "line"]).agg({**first_cols, **agg}).reset_index()


def run_ols_model(df_game: pd.DataFrame) -> pd.DataFrame:
    """
    OLS on total_runs (line-agnostic). Returns game-level OOF predictions.
    Then expand back to (game_pk, line) grain and convert to P(over line) via Poisson.
    """
    valid = df_game[OLS_FEATURES + ["total_runs", "game_pk"]].dropna()
    X = valid[OLS_FEATURES].values
    y = valid["total_runs"].values.astype(float)
    groups = valid["game_pk"].values

    scaler = StandardScaler()
    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_pred = np.full(len(y), np.nan)
    for tr, val in gkf.split(X, y, groups):
        X_tr_s = scaler.fit_transform(X[tr])
        X_val_s = scaler.transform(X[val])
        clf = Ridge(alpha=1.0)
        clf.fit(X_tr_s, y[tr])
        oof_pred[val] = clf.predict(X_val_s)

    rmse = np.sqrt(mean_squared_error(y, oof_pred))
    mae  = mean_absolute_error(y, oof_pred)
    r2   = 1 - np.sum((y - oof_pred)**2) / np.sum((y - y.mean())**2)

    print(f"  OLS total_runs → RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.4f}")

    result = valid[["game_pk"]].copy()
    result["ols_pred_runs"] = oof_pred
    return result


def poisson_p_over(mu: float, line: float) -> float:
    """P(total_runs > line) under Poisson(mu)."""
    k = int(np.floor(line))
    return 1.0 - poisson.cdf(k, mu)


def run_logistic_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Logistic regression on hit_over at (game_pk, line) grain.
    Includes line as a feature so the model can learn per-line calibration.
    """
    feat_cols = GAME_FEATURES + CATEGORICAL_FEATURES
    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=False, dtype=float)
    all_feats = GAME_FEATURES + list(dummies.columns)

    base_cols = ["hit_over", "game_pk", "novig_prob_over"] + [c for c in GAME_FEATURES if c not in ("game_pk",)]
    # include line only once (it's in GAME_FEATURES already)
    if "line" not in base_cols:
        base_cols.append("line")
    df2 = pd.concat([df[base_cols], dummies], axis=1)
    valid = df2.dropna(subset=GAME_FEATURES + ["hit_over"])

    X = valid[all_feats].values
    y = valid["hit_over"].values.astype(int)
    groups = valid["game_pk"].values

    scaler = StandardScaler()
    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_probs = np.full(len(y), np.nan)
    for tr, val in gkf.split(X, y, groups):
        X_tr_s = scaler.fit_transform(X[tr])
        X_val_s = scaler.transform(X[val])
        clf = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
        clf.fit(X_tr_s, y[tr])
        oof_probs[val] = clf.predict_proba(X_val_s)[:, 1]

    mask = ~np.isnan(oof_probs)
    auc   = roc_auc_score(y[mask], oof_probs[mask])
    brier = brier_score_loss(y[mask], oof_probs[mask])
    print(f"  Logistic hit_over → AUC={auc:.4f}, Brier={brier:.4f}")

    result = valid[["game_pk", "line", "novig_prob_over"]].copy()
    result["logit_p_over"] = oof_probs
    return result


def main() -> None:
    print("Loading spine...")
    df_all = pd.read_parquet(LOCAL_SPINE)
    print(f"  {len(df_all)} rows, {df_all.game_pk.nunique()} games")

    print("Deduplicating to (game_pk, line) grain...")
    df = dedup_to_game_line(df_all)
    df["hit_over"]  = df["hit_over"].astype(float)
    df["hit_under"] = df["hit_under"].astype(float)
    df["hit_push"]  = df["hit_push"].astype(float)
    print(f"  → {len(df)} rows (one per game+line)")

    # Deduplicate to game level for OLS
    df_game = df.drop_duplicates("game_pk").copy()
    print(f"  → {len(df_game)} game-level rows for OLS")

    print("\n--- OLS model (predict total_runs) ---")
    ols_result = run_ols_model(df_game)

    # Join OLS predictions back to (game_pk, line) grain
    df = df.merge(ols_result, on="game_pk", how="left")

    # Convert OLS run prediction to P(over line) via Poisson
    df["ols_p_over"] = df.apply(
        lambda r: poisson_p_over(max(r["ols_pred_runs"], 1.0), r["line"])
        if pd.notna(r["ols_pred_runs"]) else np.nan,
        axis=1,
    )
    df["ols_p_under"] = 1.0 - df["ols_p_over"]

    # OLS model eval as a classifier of hit_over
    valid_ols = df.dropna(subset=["ols_p_over", "hit_over"])
    auc_ols = roc_auc_score(valid_ols["hit_over"].astype(int), valid_ols["ols_p_over"])
    brier_ols = brier_score_loss(valid_ols["hit_over"].astype(int), valid_ols["ols_p_over"])
    print(f"  OLS→Poisson → AUC={auc_ols:.4f}, Brier={brier_ols:.4f}")

    # By line: over rate vs OLS model P(over)
    print("\n  OLS calibration by line:")
    by_line = df.dropna(subset=["ols_p_over","hit_over"]).groupby("line").agg(
        n=("hit_over","count"),
        actual_over=("hit_over","mean"),
        model_over=("ols_p_over","mean"),
        market_over=("novig_prob_over","mean"),
    ).round(3)
    print(by_line[by_line["n"] >= 30].to_string())

    print("\n--- Logistic model (predict hit_over, includes line as feature) ---")
    logit_result = run_logistic_model(df)
    df = df.merge(logit_result[["game_pk", "line", "logit_p_over"]],
                  on=["game_pk", "line"], how="left")
    df["logit_p_under"] = 1.0 - df["logit_p_over"]

    # Logistic calibration by line
    print("\n  Logistic calibration by line:")
    by_line2 = df.dropna(subset=["logit_p_over","hit_over"]).groupby("line").agg(
        n=("hit_over","count"),
        actual_over=("hit_over","mean"),
        model_over=("logit_p_over","mean"),
        market_over=("novig_prob_over","mean"),
    ).round(3)
    print(by_line2[by_line2["n"] >= 30].to_string())

    # Spot check: NYY @ BOS
    print("\n--- Spot check: NYY @ BOS ---")
    spot = df_all[
        (df_all["home_team"] == "Boston Red Sox") &
        (df_all["away_team"] == "New York Yankees")
    ].drop_duplicates("game_pk")
    spot_pks = spot["game_pk"].tolist()
    spot_df = df[df["game_pk"].isin(spot_pks)].sort_values("game_date")
    print(spot_df[["game_date","line","total_runs","ols_pred_runs","ols_p_over","logit_p_over","novig_prob_over","hit_over"]].head(10).to_string(index=False))

    # Save
    df.to_parquet(LOCAL_OUT, index=False)
    print(f"\nSaved OOF predictions → {LOCAL_OUT}")


if __name__ == "__main__":
    main()
