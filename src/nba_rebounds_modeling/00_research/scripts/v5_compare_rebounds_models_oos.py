"""
Compare OLS vs Ridge vs XGB on same 6 B_min_max features: OOS regression + under-only P&L.

Context:
- For each test season, trains on all other seasons, predicts REB on test-season feat rows.
- Regression QC: RMSE, MAE, R² on test player-games (same rows for all models).
- Betting QC: same Option A logic as v4 (under_only, roll_reb_std_5, shrink=0,
  min_edge 0.05 / 0.10), P&L at posted under_odds.

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/v5_compare_rebounds_models_oos.py \\
        --feat ~/Downloads/tmp/rebounds_model_features_v2.parquet \\
        --v3 ~/Downloads/tmp/v3_rebounds_props_raw.parquet \\
        --models ols,ridge,xgb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ensure_repo_root_on_syspath() -> Path:
    current = Path.cwd().resolve()
    while True:
        if (current / ".gitignore").exists() and (current / "src").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


ensure_repo_root_on_syspath()

TARGET = "REB"
B_MIN_MAX_FEATS = [
    "min_line",
    "max_line",
    "spread_signed",
    "roll_reb_mean_60",
    "roll_fg3a_mean_20",
    "roll_reb_std_5",
]
GROUP_KEYS = ["season", "date", "player_normalized", "game_id"]
SIGMA_COL = "roll_reb_std_5"
SHRINKAGE = 0.0
MIN_EDGES = [0.05, 0.10]
DEFAULT_TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]
SIGMA_FLOOR = 0.25

RIDGE_ALPHAS = np.array([0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])


def american_profit_on_win(american: float) -> float:
    if np.isnan(american):
        return float("nan")
    if american >= 100:
        return float(american) / 100.0
    return 100.0 / float(abs(american))


def american_to_implied_prob_vigged(american: float) -> float:
    if np.isnan(american):
        return float("nan")
    if american < 0:
        return float((-american) / ((-american) + 100.0))
    return float(100.0 / (american + 100.0))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    err = y_true.astype(float) - y_pred.astype(float)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    yt = y_true.astype(float)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, mae, r2


def fit_predict_ols(train_m: pd.DataFrame, test_m: pd.DataFrame) -> np.ndarray:
    X_tr = sm.add_constant(train_m[B_MIN_MAX_FEATS].astype(float), has_constant="add")
    y_tr = train_m[TARGET].astype(float)
    m = sm.OLS(y_tr, X_tr).fit()
    X_te = sm.add_constant(test_m[B_MIN_MAX_FEATS].astype(float), has_constant="add")
    return m.predict(X_te).to_numpy()


def fit_predict_ridge(train_m: pd.DataFrame, test_m: pd.DataFrame) -> np.ndarray:
    X_tr = train_m[B_MIN_MAX_FEATS].astype(float)
    y_tr = train_m[TARGET].astype(float)
    X_te = test_m[B_MIN_MAX_FEATS].astype(float)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


def fit_predict_xgb(train_m: pd.DataFrame, test_m: pd.DataFrame) -> np.ndarray:
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("Install xgboost to use model=xgb") from e

    X_tr = train_m[B_MIN_MAX_FEATS].astype(float)
    y_tr = train_m[TARGET].astype(float)
    X_te = test_m[B_MIN_MAX_FEATS].astype(float)

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=69,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


FITTERS = {
    "ols": fit_predict_ols,
    "ridge": fit_predict_ridge,
    "xgb": fit_predict_xgb,
}


def under_only_pnl_rows(
    base: pd.DataFrame,
    test_season: str,
    model_name: str,
    train_seasons_str: str,
    n_train_rows: int,
    n_test_feat_rows: int,
    n_v3_test_rows: int,
) -> list[dict]:
    sigma = base[SIGMA_COL].astype(float).clip(lower=SIGMA_FLOOR).to_numpy()
    consensus = base["consensus_reb_line"].astype(float).to_numpy()
    line = base["line"].astype(float).to_numpy()
    reb = base["REB"].astype(float).to_numpy()
    yhat_arr = base["yhat"].to_numpy()
    p_nov_u = base["p_under_novig"].astype(float).to_numpy()
    under_odds = base["under_odds"].astype(float).to_numpy()

    mean_adj = consensus + (1.0 - SHRINKAGE) * (yhat_arr - consensus)
    z = (line - mean_adj) / sigma
    p_under = norm.cdf(z)
    edge_u = p_under - p_nov_u

    rows = []
    for min_edge in MIN_EDGES:
        pnl_total = 0.0
        n_bets = 0
        n_win = 0
        n_push = 0
        sum_imp = 0.0
        sum_am = 0.0
        for i in range(len(base)):
            if edge_u[i] <= min_edge:
                continue
            if reb[i] == line[i]:
                n_push += 1
                continue
            odds_am = under_odds[i]
            won = reb[i] < line[i]
            n_bets += 1
            sum_imp += american_to_implied_prob_vigged(odds_am)
            sum_am += float(odds_am)
            if won:
                pnl_total += american_profit_on_win(odds_am)
                n_win += 1
            else:
                pnl_total -= 1.0

        roi = pnl_total / n_bets if n_bets else float("nan")
        hit = n_win / n_bets if n_bets else float("nan")
        mean_imp = sum_imp / n_bets if n_bets else float("nan")
        mean_am = sum_am / n_bets if n_bets else float("nan")
        rows.append({
            "model":             model_name,
            "test_season":       test_season,
            "train_seasons":     train_seasons_str,
            "min_edge":          min_edge,
            "n_train_rows":      n_train_rows,
            "n_test_feat_rows":  n_test_feat_rows,
            "n_v3_test_rows":    n_v3_test_rows,
            "n_merged_rows":     int(len(base)),
            "n_bets":            n_bets,
            "n_push":            n_push,
            "n_win":             n_win,
            "hit_rate":          hit,
            "mean_implied_prob_vigged": mean_imp,
            "mean_american_odds":      mean_am,
            "total_pnl_u":       pnl_total,
            "roi":               roi,
        })
    return rows


def run_one_model_one_season(
    feat: pd.DataFrame,
    v3: pd.DataFrame,
    test_season: str,
    model_name: str,
) -> tuple[list[dict], dict]:
    train = feat[feat["season"] != test_season].copy()
    feat_test = feat[feat["season"] == test_season].copy()

    cols_needed = B_MIN_MAX_FEATS + [TARGET]
    train_m = train.dropna(subset=cols_needed)
    test_m = feat_test.dropna(subset=cols_needed + GROUP_KEYS)

    if len(train_m) < 100:
        raise ValueError(f"train_m too small: test_season={test_season}")

    fitter = FITTERS[model_name]
    yhat = fitter(train_m, test_m)
    y_true = test_m[TARGET].to_numpy()
    rmse, mae, r2 = regression_metrics(y_true, yhat)

    pred = test_m[GROUP_KEYS].copy()
    pred["yhat"] = yhat
    pred[SIGMA_COL] = test_m[SIGMA_COL].to_numpy()

    v3_test = v3[v3["season"] == test_season].copy()
    base = v3_test.merge(pred, on=GROUP_KEYS, how="inner")

    train_seasons_str = ",".join(sorted(s for s in feat["season"].unique() if s != test_season))

    reg_row = {
        "model": model_name,
        "test_season": test_season,
        "n_test_regression": int(len(test_m)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    pnl_rows = under_only_pnl_rows(
        base,
        test_season,
        model_name,
        train_seasons_str,
        int(len(train_m)),
        int(len(test_m)),
        int(len(v3_test)),
    )
    return pnl_rows, reg_row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare OLS / Ridge / XGB OOS rebounds models.")
    p.add_argument("--feat", type=str, default="~/Downloads/tmp/rebounds_model_features_v2.parquet")
    p.add_argument("--v3", type=str, default="~/Downloads/tmp/v3_rebounds_props_raw.parquet")
    p.add_argument(
        "--test-seasons",
        type=str,
        default=",".join(DEFAULT_TEST_SEASONS),
    )
    p.add_argument(
        "--models",
        type=str,
        default="ols,ridge,xgb",
        help="Comma-separated: ols, ridge, xgb",
    )
    p.add_argument("--out-csv-prefix", type=str, default="", help="Write reg_ and pnl_ CSVs if set")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    feat_path = Path(args.feat).expanduser()
    v3_path = Path(args.v3).expanduser()
    test_seasons = [s.strip() for s in args.test_seasons.split(",") if s.strip()]
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    for m in models:
        if m not in FITTERS:
            raise ValueError(f"Unknown model: {m}. Choose from {list(FITTERS)}")

    feat = pd.read_parquet(feat_path)
    v3 = pd.read_parquet(v3_path)

    for c in B_MIN_MAX_FEATS + [TARGET, "consensus_reb_line"]:
        if c not in feat.columns:
            raise ValueError(f"feat missing: {c}")

    all_pnl: list[dict] = []
    all_reg: list[dict] = []

    for ts in test_seasons:
        if ts not in feat["season"].unique():
            raise ValueError(f"test_season {ts!r} not in feat")
        for model_name in models:
            print(f"\n--- model={model_name}  test_season={ts} ---")
            pnl_part, reg_part = run_one_model_one_season(feat, v3, ts, model_name)
            all_reg.append(reg_part)
            all_pnl.extend(pnl_part)
            print(
                f"  regression  n={reg_part['n_test_regression']}  "
                f"RMSE={reg_part['rmse']:.4f}  MAE={reg_part['mae']:.4f}  R2={reg_part['r2']:.4f}"
            )
            for r in pnl_part:
                print(
                    f"  pnl  min_edge={r['min_edge']:.2f}  n_bets={r['n_bets']:,}  "
                    f"roi={r['roi']:.6f}  pnl_u={r['total_pnl_u']:.2f}"
                )

    reg_df = pd.DataFrame(all_reg).sort_values(["test_season", "model"]).reset_index(drop=True)
    pnl_df = pd.DataFrame(all_pnl).sort_values(["test_season", "model", "min_edge"]).reset_index(drop=True)

    print("\n=== TABLE A — regression (test season, same rows per model) ===")
    print(reg_df.to_string(index=False))

    print("\n=== TABLE B — under_only P&L ===")
    print(pnl_df.to_string(index=False))

    print("\n=== TABLE C — robustness (roi > 0) ===")
    for me in MIN_EDGES:
        sub = pnl_df[pnl_df["min_edge"] == me]
        flags = sub.groupby("model")["roi"].apply(lambda s: int((s > 0).sum()))
        print(f"min_edge={me}  seasons with roi>0 by model:\n{flags.to_string()}")

    both_pos = (
        pnl_df.groupby(["model", "test_season"])["roi"]
        .apply(lambda s: bool((s > 0).all()))
        .groupby("model")
        .sum()
    )
    print(f"\nseasons where BOTH min_edges positive (count / {len(test_seasons)}):\n{both_pos.to_string()}")

    if args.out_csv_prefix:
        p = Path(args.out_csv_prefix).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        stem = p.stem if p.suffix else p.name
        reg_path = p.parent / f"{stem}_regression.csv"
        pnl_path = p.parent / f"{stem}_pnl.csv"
        reg_df.to_csv(reg_path, index=False)
        pnl_df.to_csv(pnl_path, index=False)
        print(f"wrote {reg_path}")
        print(f"wrote {pnl_path}")


if __name__ == "__main__":
    main()
