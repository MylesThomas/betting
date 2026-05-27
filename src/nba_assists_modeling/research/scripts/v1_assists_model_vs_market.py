"""
v1 Assists: can a regression model beat the market on MAE/RMSE?

Approach mirrors rebounds v5:
  - Features: min_line, max_line, spread (line_range), rolling AST means/std, rolling MIN mean
  - Baseline: market consensus_line as prediction
  - Models: OLS, Ridge, XGB (walk-forward OOS)
  - Gate: model_RMSE < market_RMSE in 2/2 OOS folds → pursue further

Usage:
    uv run python src/nba_assists_modeling/research/scripts/v1_assists_model_vs_market.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
from src.nba_rebounds_modeling.duckdb_s3_creds import connect_duckdb_s3

# ── Config ─────────────────────────────────────────────────────────────────────
SEASONS = ["2023-24", "2024-25", "2025-26"]
SEASON_DATE_RANGES = {
    "2023-24": ("2023-10-01", "2024-06-30"),
    "2024-25": ("2024-10-01", "2025-06-30"),
    "2025-26": ("2025-10-01", "2026-06-30"),
}
MARKET = "player_assists"
TARGET = "AST"
MIN_MINUTES = 10

ROLL_WINDOWS = [5, 10, 20, 60]
RIDGE_ALPHAS = np.array([0.1, 1.0, 3.0, 10.0, 30.0, 100.0])

FEATURES = [
    "min_line",
    "max_line",
    "line_range",
    "ast_roll_mean_5",
    "ast_roll_mean_10",
    "ast_roll_mean_20",
    "ast_roll_mean_60",
    "ast_roll_std_5",
    "min_roll_mean_10",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def metrics(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": label,
        "n": len(y_true),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "r2": round(r2(y_true, y_pred), 4),
    }


# ── Data loading ───────────────────────────────────────────────────────────────

def load_logs(con) -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        print(f"  logs {season}...", flush=True)
        q = f"""
            SELECT PLAYER_NAME,
                   CAST(AST AS DOUBLE) AS AST,
                   CAST(MIN AS DOUBLE) AS MIN,
                   GAME_DATE
            FROM read_csv_auto('s3://nba-api-mt/player_game_logs/{season}/*.csv',
                               header=true, ignore_errors=true)
        """
        df = con.execute(q).df()
        df["season"] = season
        frames.append(df)
    logs = pd.concat(frames, ignore_index=True)
    logs = logs[logs["MIN"] >= MIN_MINUTES].copy()
    logs["game_date"] = pd.to_datetime(logs["GAME_DATE"], format="mixed").dt.date
    logs["player_key"] = logs["PLAYER_NAME"].str.lower().str.strip()
    return logs


def load_props(con) -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        start_date, end_date = SEASON_DATE_RANGES[season]
        print(f"  props {season}...", flush=True)
        q = f"""
            SELECT player,
                   CAST(prop_line AS DOUBLE) AS prop_line,
                   game_time
            FROM read_csv_auto('s3://the-odds-api-mt/nba/historical_player_props/{season}/*.csv',
                               header=true, ignore_errors=true)
            WHERE market = 'player_assists'
              AND game_time >= '{start_date}'
              AND game_time <= '{end_date}'
        """
        df = con.execute(q).df()
        df["season"] = season
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["game_time"] = pd.to_datetime(raw["game_time"], format="mixed")
    raw["game_date"] = raw["game_time"].dt.date
    raw["player_key"] = raw["player"].str.lower().str.strip()

    # Aggregate per player × game: consensus = median, min/max across bookmakers
    props = (
        raw.groupby(["player_key", "game_date", "season"], as_index=False)
        .agg(
            player=("player", "first"),
            consensus_line=("prop_line", "median"),
            min_line=("prop_line", "min"),
            max_line=("prop_line", "max"),
            n_books=("prop_line", "count"),
        )
    )
    props["line_range"] = props["max_line"] - props["min_line"]
    return props


# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_key", "season", "game_date"]).copy()

    grp = df.groupby(["player_key", "season"])

    for w in ROLL_WINDOWS:
        df[f"ast_roll_mean_{w}"] = grp["AST"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=max(3, w // 4)).mean()
        )
    df["ast_roll_std_5"] = grp["AST"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=3).std()
    )
    df["min_roll_mean_10"] = grp["MIN"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean()
    )

    return df


# ── OOS evaluation ─────────────────────────────────────────────────────────────

def run_oos(df: pd.DataFrame) -> list[dict]:
    """Walk-forward OOS: for each test season, train on all prior seasons."""
    folds = [
        ("2024-25", ["2023-24"]),
        ("2025-26", ["2023-24", "2024-25"]),
    ]
    results = []
    for test_season, train_seasons in folds:
        train = df[df["season"].isin(train_seasons)].dropna(subset=FEATURES + [TARGET])
        test  = df[df["season"] == test_season].dropna(subset=FEATURES + [TARGET])

        if len(train) < 50 or len(test) < 50:
            print(f"  skip {test_season}: train={len(train)}, test={len(test)}")
            continue

        y_tr  = train[TARGET].to_numpy(dtype=float)
        y_te  = test[TARGET].to_numpy(dtype=float)
        X_tr  = train[FEATURES].to_numpy(dtype=float)
        X_te  = test[FEATURES].to_numpy(dtype=float)
        mkt   = test["consensus_line"].to_numpy(dtype=float)

        # Market baseline
        r = metrics("market", y_te, mkt)
        r["fold"] = test_season
        results.append(r)

        # OLS
        X_tr_c = sm.add_constant(X_tr, has_constant="add")
        X_te_c = sm.add_constant(X_te, has_constant="add")
        ols_m = sm.OLS(y_tr, X_tr_c).fit()
        yhat_ols = ols_m.predict(X_te_c)
        r = metrics("OLS", y_te, yhat_ols)
        r["fold"] = test_season
        results.append(r)

        # Ridge
        pipe = Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=RIDGE_ALPHAS))])
        pipe.fit(X_tr, y_tr)
        yhat_ridge = pipe.predict(X_te)
        r = metrics("Ridge", y_te, yhat_ridge)
        r["fold"] = test_season
        results.append(r)

        # XGBoost
        xgb_m = xgb.XGBRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, reg_alpha=0.5,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb_m.fit(X_tr, y_tr)
        yhat_xgb = xgb_m.predict(X_te)
        r = metrics("XGB", y_te, yhat_xgb)
        r["fold"] = test_season
        results.append(r)

        print(f"\n  fold={test_season}  train_n={len(train):,}  test_n={len(test):,}")
        print(f"  Market: RMSE={rmse(y_te,mkt):.4f}  MAE={mae(y_te,mkt):.4f}")
        print(f"  OLS:    RMSE={rmse(y_te,yhat_ols):.4f}  MAE={mae(y_te,yhat_ols):.4f}")
        print(f"  Ridge:  RMSE={rmse(y_te,yhat_ridge):.4f}  MAE={mae(y_te,yhat_ridge):.4f}")
        print(f"  XGB:    RMSE={rmse(y_te,yhat_xgb):.4f}  MAE={mae(y_te,yhat_xgb):.4f}")

    return results


# ── Segment analysis ───────────────────────────────────────────────────────────

def segment_by_line_tier(df: pd.DataFrame, results: list[dict]):
    """For each line tier, report market/OLS/Ridge/XGB RMSE (combined OOS seasons)."""
    df["line_tier"] = pd.cut(df["consensus_line"], bins=[0, 2, 4, 6, 100],
        labels=["low (0.5-1.5)", "mid (2.5-3.5)", "high (4.5-5.5)", "star (6.5+)"])

    oos_seasons = ["2024-25", "2025-26"]
    oos = df[df["season"].isin(oos_seasons)].dropna(subset=FEATURES + [TARGET]).copy()

    if oos.empty:
        print("No OOS data for segmentation.")
        return

    train_for_oos = df[~df["season"].isin(oos_seasons)].dropna(subset=FEATURES + [TARGET])
    if len(train_for_oos) < 50:
        return

    X_tr = train_for_oos[FEATURES].to_numpy(dtype=float)
    y_tr = train_for_oos[TARGET].to_numpy(dtype=float)
    X_oos = oos[FEATURES].to_numpy(dtype=float)

    # OLS
    X_tr_c = sm.add_constant(X_tr, has_constant="add")
    X_oos_c = sm.add_constant(X_oos, has_constant="add")
    ols_m = sm.OLS(y_tr, X_tr_c).fit()
    oos["yhat_ols"] = ols_m.predict(X_oos_c)

    # Ridge
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=RIDGE_ALPHAS))])
    pipe.fit(X_tr, y_tr)
    oos["yhat_ridge"] = pipe.predict(X_oos)

    # XGB
    xgb_m = xgb.XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, reg_alpha=0.5,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_m.fit(X_tr, y_tr)
    oos["yhat_xgb"] = xgb_m.predict(X_oos)

    print("\n=== RMSE by line tier (OOS 2024-25 + 2025-26) ===")
    print(f"  {'tier':<20}  {'n':>5}  {'mkt':>7}  {'OLS':>7}  {'Ridge':>7}  {'XGB':>7}")
    for tier, g in oos.groupby("line_tier", observed=True):
        y = g[TARGET].to_numpy(dtype=float)
        mkt_r = rmse(y, g["consensus_line"].to_numpy(dtype=float))
        ols_r = rmse(y, g["yhat_ols"].to_numpy(dtype=float))
        rid_r = rmse(y, g["yhat_ridge"].to_numpy(dtype=float))
        xgb_r = rmse(y, g["yhat_xgb"].to_numpy(dtype=float))
        print(f"  {str(tier):<20}  {len(g):>5}  {mkt_r:>7.4f}  {ols_r:>7.4f}  {rid_r:>7.4f}  {xgb_r:>7.4f}")

    print(f"\n=== MAE by line tier (OOS 2024-25 + 2025-26) ===")
    print(f"  {'tier':<20}  {'n':>5}  {'mkt':>7}  {'OLS':>7}  {'Ridge':>7}  {'XGB':>7}")
    for tier, g in oos.groupby("line_tier", observed=True):
        y = g[TARGET].to_numpy(dtype=float)
        mkt_m = mae(y, g["consensus_line"].to_numpy(dtype=float))
        ols_m = mae(y, g["yhat_ols"].to_numpy(dtype=float))
        rid_m = mae(y, g["yhat_ridge"].to_numpy(dtype=float))
        xgb_m = mae(y, g["yhat_xgb"].to_numpy(dtype=float))
        print(f"  {str(tier):<20}  {len(g):>5}  {mkt_m:>7.4f}  {ols_m:>7.4f}  {rid_m:>7.4f}  {xgb_m:>7.4f}")


# ── Individual feature regressions ────────────────────────────────────────────

def univariate_oos(df: pd.DataFrame):
    """OOS RMSE for each feature alone (OLS with constant) vs market baseline."""
    folds = [
        ("2024-25", ["2023-24"]),
        ("2025-26", ["2023-24", "2024-25"]),
    ]
    rows = []
    for test_season, train_seasons in folds:
        base = df[df["season"].isin(train_seasons + [test_season])].copy()
        train = base[base["season"].isin(train_seasons)]
        test  = base[base["season"] == test_season]
        y_te  = test[TARGET].to_numpy(dtype=float)
        mkt   = test["consensus_line"].to_numpy(dtype=float)

        for feat in FEATURES:
            sub_tr = train.dropna(subset=[feat, TARGET])
            sub_te = test.dropna(subset=[feat, TARGET])
            if len(sub_tr) < 30 or len(sub_te) < 30:
                continue
            y_te_f = sub_te[TARGET].to_numpy(dtype=float)
            X_tr_c = sm.add_constant(sub_tr[[feat]].to_numpy(dtype=float), has_constant="add")
            X_te_c = sm.add_constant(sub_te[[feat]].to_numpy(dtype=float), has_constant="add")
            m = sm.OLS(sub_tr[TARGET].to_numpy(dtype=float), X_tr_c).fit()
            yhat = m.predict(X_te_c)
            mkt_f = sub_te["consensus_line"].to_numpy(dtype=float)
            rows.append({
                "fold": test_season,
                "feature": feat,
                "n": len(sub_te),
                "rmse": round(rmse(y_te_f, yhat), 4),
                "mae": round(mae(y_te_f, yhat), 4),
                "r2": round(r2(y_te_f, yhat), 4),
                "mkt_rmse": round(rmse(y_te_f, mkt_f), 4),
                "delta_rmse": round(rmse(y_te_f, yhat) - rmse(y_te_f, mkt_f), 4),
            })

    tbl = pd.DataFrame(rows)
    # Average delta_rmse across folds per feature
    summary = (
        tbl.groupby("feature")[["rmse", "mae", "r2", "mkt_rmse", "delta_rmse"]]
        .mean().round(4)
        .sort_values("delta_rmse")
    )
    print("\n=== Univariate OLS OOS — avg across both folds ===")
    print("  (delta_rmse < 0 means feature alone beats market alone)")
    print(summary.to_string())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Connecting...", flush=True)
    con = connect_duckdb_s3()

    print("\nLoading game logs...")
    logs = load_logs(con)
    print(f"Logs: {len(logs):,} rows  (MIN >= {MIN_MINUTES})")
    print(logs.groupby("season")[TARGET].count().to_string())

    print("\nLoading props...")
    props = load_props(con)
    print(f"Props (consensus): {len(props):,} rows")
    print(props.groupby("season")["player_key"].count().to_string())

    df = pd.merge(props, logs, on=["player_key", "game_date", "season"], how="inner")
    print(f"\nJoined: {len(df):,} rows")
    print(df.groupby("season")[TARGET].count().to_string())

    print("\nBuilding features...")
    df = build_features(df)
    valid = df.dropna(subset=FEATURES)
    print(f"Rows with all features: {len(valid):,} / {len(df):,}")

    print("\nLine distribution (consensus):")
    print(df["consensus_line"].value_counts().sort_index().to_string())
    print(f"\nMulti-book spread: mean line_range={df['line_range'].mean():.4f}  "
          f"pct_split={( df['line_range'] > 0 ).mean()*100:.1f}%")

    print("\n=== Univariate feature regressions ===")
    univariate_oos(df)

    print("\n=== OOS model vs market ===")
    results = run_oos(df)

    segment_by_line_tier(df, results)

    print("\n\n=== Summary table ===")
    if results:
        tbl = pd.DataFrame(results)[["fold", "model", "n", "rmse", "mae", "r2"]]
        print(tbl.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
