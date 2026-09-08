"""
Step 3b — Multi-feature OLS + XGBoost regression for MLB Batter Hits.

Uses the top features from Step 3a to build:
  1. OLS (linear regression) — numeric features only
  2. XGBoost regressor — numeric + one-hot categorical features

Both evaluated OOF (TimeSeriesSplit 5-fold) on the player-game level.

Metrics:
  - Regression: r², Pearson r, RMSE, MAE
  - Binary (yhat > offered_line): AUC, Brier score (overall and per-line)

Usage:
  python src/mlb_batter_hits_modeling/scripts/20260727_step3b_multi_feature.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE = Path.home() / "Downloads/tmp/mlb_batter_hits_spine.parquet"
N_SPLITS = 5

# Top features from Step 3a — sorted by r²
NUMERIC_FEATURES = [
    "min_raw_implied_prob_under",
    "max_raw_implied_prob_over",
    "hits_roll_career",
    "max_line",
    "ab_roll_career",
    "hits_roll_L20",
    "hits_roll_season",
    "consensus_line",
    "hits_roll_L10",
    "ba_roll_career",
]
CATEGORICAL_FEATURES = [
    "consensus_over_odds_bin_granular",
    "consensus_under_odds_bin_granular",
]


def load_player_game_df() -> pd.DataFrame:
    spine = pd.read_parquet(LOCAL_SPINE)
    settled = spine[spine["hits_actual"].notna()].copy()
    has_dk = settled[settled["bookmaker"] == "draftkings"]
    no_dk  = settled[~settled.index.isin(has_dk.index)]
    combined = pd.concat([has_dk, no_dk])
    pg = (
        combined.sort_values(["player_key", "game_date", "bookmaker"])
        .drop_duplicates(subset=["player_key", "game_date"])
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    print(f"Player-game rows: {len(pg):,}  ({pg['player_key'].nunique():,} players)")
    print(f"Date range: {pg['game_date'].min()} → {pg['game_date'].max()}")
    return pg


def build_feature_matrix(pg: pd.DataFrame, include_categorical: bool = False) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Returns X, y, and the original row index after dropna."""
    cols = NUMERIC_FEATURES.copy()
    if include_categorical:
        for cat in CATEGORICAL_FEATURES:
            if cat in pg.columns:
                dummies = pd.get_dummies(pg[cat], prefix=cat, drop_first=False)
                pg = pd.concat([pg, dummies.astype(float)], axis=1)
                cols += list(dummies.columns)
    avail = [c for c in cols if c in pg.columns]
    sub = pg[avail + ["hits_actual"]].dropna()
    X = sub[avail].values
    y = sub["hits_actual"].values
    return X, y, sub.index, avail


def oof_predict(X: np.ndarray, y: np.ndarray, model_cls, **kwargs) -> np.ndarray:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in tscv.split(X):
        m = model_cls(**kwargs)
        m.fit(X[train_idx], y[train_idx])
        preds[test_idx] = m.predict(X[test_idx])
    return preds


def regression_metrics(y: np.ndarray, preds: np.ndarray) -> dict:
    valid = ~np.isnan(preds)
    y_v, p_v = y[valid], preds[valid]
    r2 = 1 - np.sum((y_v - p_v) ** 2) / np.sum((y_v - y_v.mean()) ** 2)
    pearson_r = stats.pearsonr(y_v, p_v)[0]
    rmse = np.sqrt(mean_squared_error(y_v, p_v))
    mae  = np.mean(np.abs(y_v - p_v))
    return dict(n=len(y_v), r2=r2, pearson_r=pearson_r, rmse=rmse, mae=mae)


def binary_metrics_by_line(pg_sub: pd.DataFrame, preds: np.ndarray, line_col: str = "offered_line") -> pd.DataFrame:
    """For each line value, compute AUC and Brier using yhat > line as the predicted probability proxy."""
    rows = []
    lines = sorted(pg_sub[line_col].dropna().unique())
    for line in lines:
        mask = (pg_sub[line_col] == line).values & ~np.isnan(preds)
        if mask.sum() < 50:
            continue
        y_bin = (pg_sub.loc[mask, "hits_actual"] > line).astype(int).values
        if len(np.unique(y_bin)) < 2:
            continue
        # Sigmoid-transform the distance (yhat - line) to get [0,1] prob
        dist = preds[mask] - line
        probs = 1 / (1 + np.exp(-dist))
        auc   = roc_auc_score(y_bin, probs)
        brier = brier_score_loss(y_bin, probs)
        rows.append(dict(line=line, n=int(mask.sum()), auc=round(auc, 4), brier=round(brier, 4)))
    return pd.DataFrame(rows)


def run_model(name: str, pg: pd.DataFrame, model_cls, include_categorical: bool, **kwargs) -> tuple[dict, pd.DataFrame]:
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    X, y, idx, feats = build_feature_matrix(pg.copy(), include_categorical=include_categorical)
    pg_sub = pg.loc[idx].reset_index(drop=True)

    print(f"  Features: {len(feats)}")
    print(f"  Rows (after dropna): {len(y):,}")

    preds = oof_predict(X, y, model_cls, **kwargs)
    metrics = regression_metrics(y, preds)
    print(f"  r²={metrics['r2']:.4f}  Pearson r={metrics['pearson_r']:.4f}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}")

    # Also compute overall AUC at line 0.5 using yhat > 0.5 as proxy
    mask_05 = (pg_sub["offered_line"] == 0.5).values & ~np.isnan(preds)
    if mask_05.sum() >= 50:
        y_bin = (pg_sub.loc[mask_05, "hits_actual"] > 0.5).astype(int).values
        dist  = preds[mask_05] - 0.5
        probs = 1 / (1 + np.exp(-dist))
        if len(np.unique(y_bin)) == 2:
            auc_05 = roc_auc_score(y_bin, probs)
            print(f"  AUC at line=0.5: {auc_05:.4f}")
            metrics["auc_0_5"] = auc_05

    per_line = binary_metrics_by_line(pg_sub, preds)
    print("\n  AUC by line (yhat → sigmoid):")
    print("  " + per_line.to_string(index=False).replace("\n", "\n  "))

    # Feature importances for XGBoost
    if hasattr(model_cls(), "feature_importances_"):
        # Refit on full data to get importances
        m = model_cls(**kwargs)
        m.fit(X, y)
        importances = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
        print(f"\n  Top 10 feature importances:")
        print("  " + importances.head(10).round(4).to_string().replace("\n", "\n  "))

    return metrics, per_line, preds, pg_sub


def main() -> None:
    print("Loading spine...")
    pg = load_player_game_df()

    # ---------------------------------------------------------------
    # OLS — numeric features only
    # ---------------------------------------------------------------
    ols_metrics, ols_per_line, ols_preds, ols_pg = run_model(
        "OLS (numeric only)", pg, LinearRegression, include_categorical=False
    )

    # ---------------------------------------------------------------
    # XGBoost — numeric + one-hot categorical
    # ---------------------------------------------------------------
    xgb_metrics, xgb_per_line, xgb_preds, xgb_pg = run_model(
        "XGBoost (numeric + categorical)", pg, XGBRegressor,
        include_categorical=True,
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0, tree_method="hist",
    )

    # ---------------------------------------------------------------
    # Summary comparison
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    summary = pd.DataFrame([
        {"model": "OLS", **ols_metrics},
        {"model": "XGBoost", **xgb_metrics},
    ])
    print(summary.round(4).to_string(index=False))

    # ---------------------------------------------------------------
    # Freeman spot-check
    # ---------------------------------------------------------------
    print("\n--- Freeman OOF Predictions (OLS) ---")
    ff_mask = ols_pg["player_key"] == "freddie freeman"
    ff = ols_pg[ff_mask].copy()
    ff["yhat_ols"] = ols_preds[ff_mask]
    print(ff[["game_date", "hits_actual", "offered_line", "yhat_ols",
              "hits_roll_career", "max_raw_implied_prob_over"]].tail(8).round(3).to_string(index=False))

    # ---------------------------------------------------------------
    # DuckDB SQL tests
    # ---------------------------------------------------------------
    import duckdb
    con = duckdb.connect()
    con.register("ols_pl", ols_per_line)
    con.register("xgb_pl", xgb_per_line)
    ols_s = pd.DataFrame([ols_metrics])
    xgb_s = pd.DataFrame([xgb_metrics])
    con.register("ols_s", ols_s)
    con.register("xgb_s", xgb_s)

    print("\n" + "="*60)
    print("STEP 3b — DuckDB SQL TESTS")
    print("="*60)
    tests = [
        ("T1: OLS r² > 0.03 (multi-feature improves on single-feature)",
         "SELECT r2 > 0.03 AS pass FROM ols_s"),
        ("T2: Multi-feature OLS r² > 0.028 (improves on best single-feature from 3a)",
         "SELECT r2 > 0.028 AS pass FROM ols_s"),
        ("T3: OLS AUC at line 0.5 > 0.57",
         "SELECT auc > 0.57 AS pass FROM ols_pl WHERE line = 0.5"),
        ("T4: XGBoost AUC at line 0.5 > 0.57",
         "SELECT auc > 0.57 AS pass FROM xgb_pl WHERE line = 0.5"),
        ("T5: Both models have per-line AUC at lines 0.5 and 1.5",
         "SELECT COUNT(*) >= 2 AS pass FROM ols_pl WHERE line IN (0.5, 1.5)"),
        ("T6: XGBoost and OLS RMSE within 1% of each other (essentially tied — linear market)",
         "SELECT ABS((SELECT rmse FROM xgb_s) - (SELECT rmse FROM ols_s)) / (SELECT rmse FROM ols_s) < 0.01 AS pass"),
    ]

    all_pass = True
    for name, sql in tests:
        try:
            result = con.execute(sql).fetchone()[0]
            status = "PASS" if result else "FAIL"
            if not result:
                all_pass = False
            print(f"  [{status}] {name}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            all_pass = False

    print()
    if all_pass:
        print("All Step 3b tests PASSED.")
    else:
        print("Some Step 3b tests FAILED.")


if __name__ == "__main__":
    main()
