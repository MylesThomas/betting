"""
Step 3a — Individual feature sweep for MLB Batter Hits.

For each candidate feature independently:
  - Linear regression → hits_actual (OOF, TimeSeriesSplit 5-fold): r², Pearson r, RMSE, MAE
  - Logistic regression → over_flag at line 0.5 (OOF): AUC, Brier score

OOF = out-of-fold: predictions made on held-out folds, sorted by game_date.
No feature is tested against data it was trained on.

Usage:
  python src/mlb_batter_hits_modeling/scripts/20260727_step3a_feature_sweep.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    roc_auc_score, mean_squared_error, brier_score_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE = Path.home() / "Downloads/tmp/mlb_batter_hits_spine.parquet"
N_SPLITS    = 5


def load_player_game_df() -> pd.DataFrame:
    """
    Deduplicate spine to one row per (player_key, game_date).
    Use one bookmaker row per player-game (prefer draftkings; else first).
    Keep only settled rows (hits_actual not null).
    """
    spine = pd.read_parquet(LOCAL_SPINE)
    settled = spine[spine["hits_actual"].notna()].copy()

    # Prefer a consistent book (DraftKings) for the player-game representation
    has_dk = settled[settled["bookmaker"] == "draftkings"]
    no_dk  = settled[~settled.index.isin(has_dk.index)]
    combined = pd.concat([has_dk, no_dk])
    pg = (
        combined.sort_values(["player_key", "game_date", "bookmaker"])
        .drop_duplicates(subset=["player_key", "game_date"])
        .sort_values("game_date")
        .reset_index(drop=True)
    )

    # Add over_flag_0_5 (binary hit/no-hit for primary line 0.5 analysis)
    pg["over_flag_0_5"] = (pg["hits_actual"] >= 1).astype(int)

    print(f"Player-game rows: {len(pg):,} ({pg['player_key'].nunique():,} players)")
    print(f"Date range: {pg['game_date'].min()} → {pg['game_date'].max()}")
    print(f"Seasons: {sorted(pg['season'].dropna().unique().astype(int).tolist())}")
    return pg


def oof_linear(df: pd.DataFrame, feat_col: str, target: str = "hits_actual") -> dict:
    """OOF linear regression. Returns r², Pearson r, RMSE, MAE."""
    sub = df[[feat_col, target]].dropna().copy()
    if len(sub) < 100:
        return dict(n=len(sub), r2=np.nan, pearson_r=np.nan, rmse=np.nan, mae=np.nan)

    X = sub[[feat_col]].values
    y = sub[target].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    preds = np.full(len(sub), np.nan)
    for train_idx, test_idx in tscv.split(X):
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])

    valid = ~np.isnan(preds)
    y_v, p_v = y[valid], preds[valid]
    r2 = 1 - np.sum((y_v - p_v) ** 2) / np.sum((y_v - y_v.mean()) ** 2)
    pearson_r = stats.pearsonr(y_v, p_v)[0] if len(y_v) > 10 else np.nan
    rmse = np.sqrt(mean_squared_error(y_v, p_v))
    mae  = np.mean(np.abs(y_v - p_v))
    return dict(n=len(sub), r2=r2, pearson_r=pearson_r, rmse=rmse, mae=mae)


def oof_logistic(df: pd.DataFrame, feat_col: str, target: str = "over_flag_0_5") -> dict:
    """OOF logistic regression. Returns AUC, Brier score."""
    sub = df[[feat_col, target]].dropna().copy()
    if len(sub) < 100 or sub[target].nunique() < 2:
        return dict(n=len(sub), auc=np.nan, brier=np.nan)

    X = sub[[feat_col]].values
    y = sub[target].values.astype(int)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    probs = np.full(len(sub), np.nan)
    for train_idx, test_idx in tscv.split(X):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = LogisticRegression(max_iter=500, solver="lbfgs")
        model.fit(X[train_idx], y[train_idx])
        probs[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    valid = ~np.isnan(probs)
    y_v, p_v = y[valid], probs[valid]
    if len(y_v) < 50 or len(np.unique(y_v)) < 2:
        return dict(n=len(sub), auc=np.nan, brier=np.nan)
    auc   = roc_auc_score(y_v, p_v)
    brier = brier_score_loss(y_v, p_v)
    return dict(n=len(sub), auc=auc, brier=brier)


def oof_logistic_categorical(df: pd.DataFrame, feat_col: str, target: str = "over_flag_0_5") -> dict:
    """OOF logistic for categorical features (one-hot encoded)."""
    sub = df[[feat_col, target]].dropna().copy()
    if len(sub) < 100 or sub[target].nunique() < 2:
        return dict(n=len(sub), auc=np.nan, brier=np.nan)

    dummies = pd.get_dummies(sub[feat_col], drop_first=True).astype(float)
    if dummies.shape[1] == 0:
        return dict(n=len(sub), auc=np.nan, brier=np.nan)

    X = dummies.values
    y = sub[target].values.astype(int)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    probs = np.full(len(sub), np.nan)
    for train_idx, test_idx in tscv.split(X):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = LogisticRegression(max_iter=500, solver="lbfgs")
        model.fit(X[train_idx], y[train_idx])
        probs[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    valid = ~np.isnan(probs)
    y_v, p_v = y[valid], probs[valid]
    if len(y_v) < 50 or len(np.unique(y_v)) < 2:
        return dict(n=len(sub), auc=np.nan, brier=np.nan)
    auc   = roc_auc_score(y_v, p_v)
    brier = brier_score_loss(y_v, p_v)
    return dict(n=len(sub), auc=auc, brier=brier)


def run_sweep(pg: pd.DataFrame) -> pd.DataFrame:
    """Run individual feature sweep. Returns results DataFrame sorted by r²."""
    numeric_features = [
        "hits_roll_L1", "hits_roll_L5", "hits_roll_L10", "hits_roll_L20",
        "hits_roll_season", "hits_roll_career",
        "ba_roll_L5", "ba_roll_career", "ab_roll_career",
        "opp_h_rate_career", "opp_h_rate_L5",
        "is_home",
        "offered_line",
        "min_line", "max_line",
        "min_raw_implied_prob_over", "max_raw_implied_prob_over",
        "min_raw_implied_prob_under", "max_raw_implied_prob_under",
        "consensus_line",
        "novig_prob_over",
        "consensus_decimal_over",
    ]
    categorical_features = [
        "stand",
        "consensus_over_odds_bin",
        "consensus_over_odds_bin_granular",
        "consensus_under_odds_bin",
        "consensus_under_odds_bin_granular",
    ]

    rows = []
    for feat in numeric_features:
        if feat not in pg.columns:
            print(f"  SKIP {feat} (not in spine)")
            continue
        lin  = oof_linear(pg, feat)
        logit = oof_logistic(pg, feat)
        rows.append({
            "feature": feat,
            "type":    "numeric",
            "n_lin":   lin["n"],
            "r2":      lin["r2"],
            "pearson_r": lin["pearson_r"],
            "rmse":    lin["rmse"],
            "mae":     lin["mae"],
            "n_logit": logit["n"],
            "auc":     logit["auc"],
            "brier":   logit["brier"],
        })
        print(f"  {feat:40s} r²={lin['r2']:.4f}  AUC={logit['auc']:.4f}")

    for feat in categorical_features:
        if feat not in pg.columns:
            print(f"  SKIP {feat} (not in spine)")
            continue
        lin_enc = pg.copy()
        lin_enc[feat + "_enc"] = LabelEncoder().fit_transform(pg[feat].fillna("__missing__"))
        lin   = oof_linear(lin_enc, feat + "_enc")
        logit = oof_logistic_categorical(pg, feat)
        rows.append({
            "feature": feat,
            "type":    "categorical",
            "n_lin":   lin["n"],
            "r2":      lin["r2"],
            "pearson_r": lin["pearson_r"],
            "rmse":    lin["rmse"],
            "mae":     lin["mae"],
            "n_logit": logit["n"],
            "auc":     logit["auc"],
            "brier":   logit["brier"],
        })
        print(f"  {feat:40s} r²={lin['r2']:.4f}  AUC={logit['auc']:.4f}")

    results = pd.DataFrame(rows).sort_values("r2", ascending=False)
    return results


def spot_check_freeman(pg: pd.DataFrame) -> None:
    """Print Freeman's feature values for a manual sanity check."""
    ff = pg[pg["player_key"] == "freddie freeman"].sort_values("game_date")
    print(f"\nFreeman: {len(ff)} player-game rows")
    cols = ["game_date", "hits_actual", "hits_roll_L5", "hits_roll_career",
            "ba_roll_career", "opp_h_rate_career", "stand", "is_home",
            "novig_prob_over", "consensus_over_odds_bin"]
    print(ff[[c for c in cols if c in ff.columns]].tail(8).to_string(index=False))


def main() -> None:
    print("Loading spine...")
    pg = load_player_game_df()

    print("\n" + "="*70)
    print("STEP 3a — INDIVIDUAL FEATURE SWEEP")
    print("(OOF linear → hits_actual  |  OOF logistic → over/under at line 0.5)")
    print("="*70)

    results = run_sweep(pg)

    print("\n--- Results sorted by r² (hits_actual linear regression) ---")
    print(results.round(4).to_string(index=False))

    print("\n--- Top 10 by AUC (logistic → over at 0.5) ---")
    print(results.sort_values("auc", ascending=False).head(10).round(4).to_string(index=False))

    spot_check_freeman(pg)

    # DuckDB SQL tests
    import duckdb
    con = duckdb.connect()
    con.register("results", results)
    con.register("pg", pg)

    print("\n" + "="*70)
    print("STEP 3a — DuckDB SQL TESTS")
    print("="*70)
    tests = [
        ("T1: At least 20 features tested",
         "SELECT COUNT(*) >= 20 AS pass FROM results"),
        ("T2: Best r² > 0.02 (some feature has predictive signal)",
         "SELECT MAX(r2) > 0.02 AS pass FROM results"),
        ("T3: Best AUC > 0.55 (above chance for binary classification)",
         "SELECT MAX(auc) > 0.55 AS pass FROM results"),
        ("T4: Player-game row count >= 50,000",
         "SELECT COUNT(*) >= 50000 AS pass FROM pg"),
        ("T5: hits_roll_career has lowest RMSE of all rolling features",
         """SELECT rmse = (SELECT MIN(rmse) FROM results WHERE feature LIKE 'hits_roll%' AND rmse IS NOT NULL) AS pass
            FROM results WHERE feature = 'hits_roll_career'"""),
        ("T6: min_raw_implied_prob_under AUC > 0.55 (best market feature has signal)",
         "SELECT auc > 0.55 AS pass FROM results WHERE feature = 'min_raw_implied_prob_under'"),
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
        print("All Step 3a tests PASSED.")
    else:
        print("Some Step 3a tests FAILED.")

    return results


if __name__ == "__main__":
    main()
