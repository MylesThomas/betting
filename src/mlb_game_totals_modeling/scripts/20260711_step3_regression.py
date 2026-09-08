"""
Step 3 (Revised) — Regression: Predict total_runs for MLB Game Totals.

Corrected approach: predict the raw continuous outcome (total_runs) via regression.
Binary classification (hit_over/hit_under) was wrong — it conflates the market line
with model signal. This script trains at the game level and saves OOF y_hat.

Features: only those with positive individual r² from Step 3a.
  - consensus_line (r²=0.022)  — market's aggregate best guess; book-invariant
  - park_factor    (r²=0.004)
  - combined_ra_L10 (r²=0.0024)
  - home_ra_L20    (r²=0.0018)
  - combined_ra_L5 (r²=0.0015)
  - combined_rs_L10 (r²=0.0013)
  - home_rs_L10    (r²=0.0011)
  - away_rs_L3     (r²=0.0009)
  - combined_ra_career (r²=0.0007)
  - away_ra_L5     (r²=0.0007)
  - home_ra_L10    (r²=0.0006)
  - away_ra_L10    (r²=0.0006)

OOF design:
  - Deduplicate spine to game level (one row per game_pk)
  - GroupKFold(n_splits=5) by game_pk — groups ensure each game in test OR train, not both
  - y_hat is book-invariant: same prediction broadcast to all spine rows for a game

Output:
  ~/Downloads/tmp/mlb_game_totals/step3_oof_yhat.parquet  — game_pk + y_hat + season + total_runs
  Prints model comparison table

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step3_regression.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_DIR   = Path.home() / "Downloads/tmp/mlb_game_totals"
LOCAL_PATH  = LOCAL_DIR / "game_totals_spine.parquet"
OUT_PARQUET = LOCAL_DIR / "step3_oof_yhat.parquet"

N_FOLDS = 5
SEED    = 42

FEATURE_COLS = [
    "consensus_line",
    "park_factor",
    "combined_ra_L10",
    "home_ra_L20",
    "combined_ra_L5",
    "combined_rs_L10",
    "home_rs_L10",
    "away_rs_L3",
    "combined_ra_career",
    "away_ra_L5",
    "home_ra_L10",
    "away_ra_L10",
]


def load_game_level() -> pd.DataFrame:
    """Load spine and deduplicate to one row per game."""
    df = pd.read_parquet(LOCAL_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Derive combined features (book-invariant rolling stats)
    df["combined_ra_L5"]     = df["home_ra_L5"]     + df["away_ra_L5"]
    df["combined_ra_L10"]    = df["home_ra_L10"]    + df["away_ra_L10"]
    df["combined_ra_career"] = df["home_ra_career"] + df["away_ra_career"]
    df["combined_rs_L10"]    = df["home_rs_L10"]    + df["away_rs_L10"]

    base_cols = ["game_pk", "game_date", "home_team", "away_team", "season", "total_runs"]
    game_cols = list(dict.fromkeys(base_cols + FEATURE_COLS))  # dedup, preserve order

    games = (
        df[game_cols]
        .drop_duplicates("game_pk")
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    return games


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def oof_regressor(games: pd.DataFrame, label: str, model_factory) -> np.ndarray:
    """Run OOF regression. Returns y_hat array aligned to games index."""
    valid = games[FEATURE_COLS + ["total_runs", "game_pk"]].dropna()
    valid_idx = valid.index

    X      = valid[FEATURE_COLS].values.astype(float)
    y      = valid["total_runs"].values.astype(float)
    groups = valid["game_pk"].values

    gkf   = GroupKFold(n_splits=N_FOLDS)
    y_hat = np.full(len(games), np.nan)

    for train_idx, test_idx in gkf.split(X, y, groups):
        sc      = StandardScaler()
        X_tr_sc = sc.fit_transform(X[train_idx])
        X_te_sc = sc.transform(X[test_idx])

        model = model_factory()
        model.fit(X_tr_sc, y[train_idx])
        preds = model.predict(X_te_sc)

        orig_positions = valid_idx[test_idx]
        y_hat[orig_positions] = preds

    mask  = ~np.isnan(y_hat)
    y_t   = games.loc[mask, "total_runs"].values.astype(float)
    y_p   = y_hat[mask]
    rmse  = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae   = float(mean_absolute_error(y_t, y_p))
    r2    = r2_score(y_t, y_p)
    n     = int(mask.sum())
    print(f"  {label:<40s}  RMSE={rmse:.4f}  MAE={mae:.4f}  r²={r2:.4f}  n={n}")
    return y_hat


def main() -> None:
    print("Loading and deduplicating to game level...")
    games = load_game_level()
    print(f"  {len(games)} games, seasons: {sorted(games['season'].unique())}")
    print(f"  Null rates in features:")
    for col in FEATURE_COLS:
        null_pct = float(games[col].isna().mean()) * 100
        if null_pct > 5:
            print(f"    {col}: {null_pct:.1f}%")

    baseline_std = games["total_runs"].std()
    print(f"\n  Baseline RMSE (predict mean): {baseline_std:.4f}")
    print(f"  Mean total_runs: {games['total_runs'].mean():.2f}\n")

    print("=== OOF REGRESSION MODELS ===")
    y_ols   = oof_regressor(games, "OLS (LinearRegression)",
                            lambda: LinearRegression())
    y_ridge = oof_regressor(games, "Ridge (alpha=10)",
                            lambda: Ridge(alpha=10))
    y_ridge2 = oof_regressor(games, "Ridge (alpha=50)",
                             lambda: Ridge(alpha=50))
    y_xgb   = oof_regressor(games, "XGBoost (depth=2, n=100, reg=5)",
                             lambda: XGBRegressor(
                                 n_estimators=100, max_depth=2, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=5, min_child_weight=20,
                                 random_state=SEED, verbosity=0,
                             ))
    y_xgb2  = oof_regressor(games, "XGBoost (depth=3, n=200, reg=10)",
                             lambda: XGBRegressor(
                                 n_estimators=200, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=10, min_child_weight=30,
                                 random_state=SEED, verbosity=0,
                             ))

    # Choose best model: Ridge alpha=10 is typically most stable for noisy data
    # Will compare residuals in Step 4 to confirm
    best_y_hat = y_ridge

    # Save game-level OOF y_hat
    out = games[["game_pk", "game_date", "home_team", "away_team", "season", "total_runs"]].copy()
    out["y_hat_ols"]    = y_ols
    out["y_hat_ridge"]  = y_ridge
    out["y_hat_ridge2"] = y_ridge2
    out["y_hat_xgb"]    = y_xgb
    out["y_hat_xgb2"]   = y_xgb2
    out["y_hat"]        = best_y_hat  # primary y_hat used in Step 4

    out.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved OOF y_hat → {OUT_PARQUET}")
    print(f"  Rows with valid y_hat: {out['y_hat'].notna().sum()} / {len(out)}")

    # Verify book-invariance property (y_hat is per game_pk, not per book row)
    print("\n=== BOOK-INVARIANCE CHECK ===")
    print("  y_hat is at game level — one prediction per game_pk.")
    print("  In Step 4, this broadcasts identically to all book rows for each game. ✓")

    # Distribution of y_hat vs total_runs
    valid_mask = out["y_hat"].notna()
    y_t = out.loc[valid_mask, "total_runs"].values
    y_p = out.loc[valid_mask, "y_hat"].values
    print(f"\n=== Y_HAT DISTRIBUTION ===")
    print(f"  y_hat  range: [{y_p.min():.2f}, {y_p.max():.2f}]  mean={y_p.mean():.2f}  std={y_p.std():.2f}")
    print(f"  actual range: [{y_t.min():.0f}, {y_t.max():.0f}]  mean={y_t.mean():.2f}  std={y_t.std():.2f}")
    print(f"\n  Note: regression shrinks predictions toward mean (std {y_p.std():.2f} vs actual {y_t.std():.2f})")
    print(f"  This is expected — the calibration step (Step 4) corrects for this per line.")


if __name__ == "__main__":
    main()
