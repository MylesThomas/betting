"""
Evaluate simple rolling-mean 3PM predictors on top NBA shooters.

Context:
- This research script compares multiple lagged 3PM mean variants on 2025-26
  player game logs.
- Cohort is top-N players by average FG3M/game (with minimum games filter).
- Goal is to identify which rolling-window setup has the strongest predictive
  signal for next-game FG3M.

Output table columns:
- model
- features
- rmse
- mae
- r2
- rmse_gain_vs_baseline
- mae_gain_vs_baseline
- n_train_rows
- n_test_rows
- n_total_rows

Optional detailed output columns (per-row predictions):
- model
- features
- game_id
- date
- matchup
- player_normalized
- projection_3pm
- actual_3pm
- residual
- abs_residual
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent.parent
UTILS_DIR = ROOT_DIR / "99_utils"
SRC_DIR = ROOT_DIR.parent
REPO_ROOT = SRC_DIR.parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from duckdb_s3 import connect_duckdb_s3
from odds import american_to_implied_prob
from odds import remove_vig_two_way
from src.player_team_history.name_normalization import normalize_from_nba_api
from src.player_team_history.name_normalization import normalize_from_odds_api


def parse_args() -> argparse.Namespace:
    """Parse CLI args for v3 simple-model sweep."""
    parser = argparse.ArgumentParser(
        description="Evaluate rolling 3PM mean model variants on top shooters."
    )
    parser.add_argument("--season", type=str, default="2025-26")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--min-games", type=int, default=25)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--min-train-games", type=int, default=20)
    parser.add_argument(
        "--windows",
        type=str,
        default="5,10,15,20,40,80,160,320,999",
        help="Comma-separated rolling windows.",
    )
    parser.add_argument("--include-ewm", action="store_true")
    parser.add_argument("--include-shrinkage", action="store_true")
    parser.add_argument("--include-market-consensus", action="store_true")
    parser.add_argument("--shrinkage-k", type=float, default=10.0)
    parser.add_argument(
        "--xgboost-regression",
        type=str,
        default="false",
        help="Whether to include an XGBoost regressor row (true/false).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument(
        "--output-predictions-csv",
        type=str,
        default="",
        help="Optional path for per-row model predictions and residuals.",
    )
    return parser.parse_args()


def parse_bool_flag(value: str) -> bool:
    """Parse common string forms of booleans from CLI."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean flag value: {value}")


def load_logs_from_s3(season: str) -> pd.DataFrame:
    """Load required player game log columns from S3 for one season."""
    con = connect_duckdb_s3()
    query = f"""
    SELECT
      PLAYER_ID,
      PLAYER_NAME,
      GAME_ID,
      GAME_DATE,
      MATCHUP,
      FG3M
    FROM read_csv_auto('s3://nba-api-mt/player_game_logs/{season}/*.csv', union_by_name=true)
    """
    logs = con.execute(query).fetchdf()
    con.close()

    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs["FG3M"] = logs["FG3M"].astype(float)
    logs["player_normalized"] = logs["PLAYER_NAME"].apply(normalize_from_nba_api)
    logs["date"] = logs["GAME_DATE"].dt.date.astype(str)
    return logs


def load_market_consensus_lines(season: str) -> pd.DataFrame:
    """Load per-player/date market consensus line from player_threes odds."""
    con = connect_duckdb_s3()
    query = f"""
    SELECT
      player,
      game_time,
      market,
      prop_line,
      over_odds,
      under_odds
    FROM read_csv_auto(
      's3://the-odds-api-mt/nba/historical_player_props/{season}/*.csv',
      union_by_name=true
    )
    """
    props = con.execute(query).fetchdf()
    con.close()

    props = props[(props["market"] == "player_threes") & props["prop_line"].notna()].copy()
    props["player_normalized"] = props["player"].apply(normalize_from_odds_api)
    props = props[props["player_normalized"].notna()].copy()
    game_time_utc = pd.to_datetime(props["game_time"], utc=True)
    props["date"] = game_time_utc.dt.tz_convert("America/New_York").dt.date.astype(str)
    props["p_over_raw"] = props["over_odds"].astype(float).apply(american_to_implied_prob)
    props["p_under_raw"] = props["under_odds"].astype(float).apply(american_to_implied_prob)
    no_vig = props.apply(
        lambda row: remove_vig_two_way(row["p_over_raw"], row["p_under_raw"]),
        axis=1,
    )
    props["p_over_novig"] = [x[0] for x in no_vig]

    line_balance = (
        props.groupby(["date", "player_normalized", "prop_line"], as_index=False)["p_over_novig"]
        .median()
        .rename(columns={"p_over_novig": "median_p_over_novig"})
    )
    line_balance["distance_to_5050"] = (line_balance["median_p_over_novig"] - 0.5).abs()
    consensus = (
        line_balance.sort_values(
            ["date", "player_normalized", "distance_to_5050", "prop_line"]
        )
        .groupby(["date", "player_normalized"], as_index=False)
        .first()
        .rename(columns={"prop_line": "market_consensus_line"})
    )
    return consensus[["date", "player_normalized", "market_consensus_line"]].copy()


def build_top_shooter_cohort(
    logs_df: pd.DataFrame,
    top_n: int,
    min_games: int,
) -> pd.DataFrame:
    """Filter to top shooters by average 3PM with minimum-games guardrail."""
    player_summary = (
        logs_df.groupby(["PLAYER_ID", "PLAYER_NAME"], as_index=False)
        .agg(
            games=("GAME_ID", "count"),
            avg_3pm=("FG3M", "mean"),
        )
        .query("games >= @min_games")
        .sort_values("avg_3pm", ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )
    top_ids = set(player_summary["PLAYER_ID"].tolist())
    return logs_df[logs_df["PLAYER_ID"].isin(top_ids)].copy()


def build_lagged_3pm_features(
    cohort_df: pd.DataFrame,
    windows: list[int],
    include_ewm: bool,
    include_shrinkage: bool,
    shrinkage_k: float,
) -> pd.DataFrame:
    """Create lagged per-player 3PM feature variants with strict no-leakage."""
    df = cohort_df.copy().sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    grouped = df.groupby("PLAYER_ID", group_keys=False)

    df["expanding_mean_fg3m"] = grouped["FG3M"].transform(
        lambda s: s.expanding(min_periods=1).mean().shift(1)
    )

    for window in windows:
        col = f"rolling_mean_fg3m_w{window}"
        df[col] = grouped["FG3M"].transform(
            lambda s, w=window: s.rolling(window=w, min_periods=1).mean().shift(1)
        )

    if include_ewm:
        df["ewm_mean_fg3m_alpha_0_20"] = grouped["FG3M"].transform(
            lambda s: s.ewm(alpha=0.20, adjust=False).mean().shift(1)
        )

    if include_shrinkage:
        short_col = "rolling_mean_fg3m_w10"
        if short_col not in df.columns:
            raise ValueError("Shrinkage model requires rolling window 10 in --windows")
        prior_n = grouped.cumcount().astype(float)
        weight_short = prior_n / (prior_n + float(shrinkage_k))
        weight_long = float(shrinkage_k) / (prior_n + float(shrinkage_k))
        df["shrinkage_mean_fg3m_w10_to_expanding"] = (
            weight_short * df[short_col] + weight_long * df["expanding_mean_fg3m"]
        )

    return df


def split_train_test_by_player(
    features_df: pd.DataFrame,
    train_frac: float,
    min_train_games: int,
) -> pd.DataFrame:
    """Apply per-player time split and mark rows as train/test."""

    def _mark_player_split(player_df: pd.DataFrame) -> pd.DataFrame:
        player_df = player_df.sort_values("GAME_DATE").copy()
        n_rows = len(player_df)
        split_idx = max(int(n_rows * train_frac), int(min_train_games))
        split_idx = min(split_idx, n_rows - 1)
        player_df["is_train"] = False
        player_df.iloc[:split_idx, player_df.columns.get_loc("is_train")] = True
        return player_df

    split_frames = [
        _mark_player_split(player_df)
        for _, player_df in features_df.groupby("PLAYER_ID", sort=False)
    ]
    return pd.concat(split_frames, ignore_index=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and R2 for model predictions."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_feature_as_predictor(
    split_df: pd.DataFrame,
    feature_col: str,
) -> dict[str, float]:
    """Evaluate one feature as direct prediction on test rows."""
    eligible_rows = split_df[split_df[feature_col].notna()].copy()
    train_rows = eligible_rows[eligible_rows["is_train"]].copy()
    test_rows = eligible_rows[~eligible_rows["is_train"]].copy()
    if test_rows.empty:
        raise ValueError(f"No test rows available for feature '{feature_col}'")
    y_true = test_rows["FG3M"].to_numpy(dtype=float)
    y_pred = test_rows[feature_col].to_numpy(dtype=float)
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    metrics["n_train_rows"] = int(len(train_rows))
    metrics["n_test_rows"] = int(len(test_rows))
    metrics["n_total_rows"] = int(len(eligible_rows))
    return metrics


def build_feature_prediction_rows(
    split_df: pd.DataFrame,
    model_name: str,
    feature_col: str,
) -> pd.DataFrame:
    """Build per-row test predictions/residuals for one feature model."""
    scored = split_df[(~split_df["is_train"]) & split_df[feature_col].notna()].copy()
    if scored.empty:
        raise ValueError(f"No scored rows available for feature '{feature_col}'")
    scored["projection_3pm"] = scored[feature_col].astype(float)
    scored["actual_3pm"] = scored["FG3M"].astype(float)
    scored["residual"] = scored["actual_3pm"] - scored["projection_3pm"]
    scored["abs_residual"] = scored["residual"].abs()
    return scored.assign(
        model=model_name,
        features=feature_col,
    )[
        [
            "model",
            "features",
            "GAME_ID",
            "date",
            "MATCHUP",
            "player_normalized",
            "projection_3pm",
            "actual_3pm",
            "residual",
            "abs_residual",
        ]
    ].rename(
        columns={
            "GAME_ID": "game_id",
            "MATCHUP": "matchup",
        }
    )


def run_model_sweep(
    split_df: pd.DataFrame,
    windows: list[int],
    include_ewm: bool,
    include_shrinkage: bool,
    include_market_consensus: bool,
    include_xgboost: bool,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all model variants and return comparison table."""
    models = [("expanding_mean_fg3m", "expanding_mean_fg3m")]
    for window in windows:
        models.append((f"rolling_mean_fg3m_w{window}", f"rolling_mean_fg3m_w{window}"))
    if include_ewm:
        models.append(("ewm_mean_fg3m_alpha_0_20", "ewm_mean_fg3m_alpha_0_20"))
    if include_shrinkage:
        models.append(
            (
                "shrinkage_mean_fg3m_w10_to_expanding",
                "shrinkage_mean_fg3m_w10_to_expanding",
            )
        )
    if include_market_consensus:
        models.append(("market_consensus_line", "market_consensus_line"))

    baseline_metrics = evaluate_feature_as_predictor(
        split_df=split_df,
        feature_col="expanding_mean_fg3m",
    )

    rows: list[dict[str, float | str | int]] = []
    prediction_frames: list[pd.DataFrame] = []
    for model_name, feature_col in models:
        metrics = evaluate_feature_as_predictor(split_df=split_df, feature_col=feature_col)
        rows.append(
            {
                "model": model_name,
                "features": feature_col,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "rmse_gain_vs_baseline": baseline_metrics["rmse"] - metrics["rmse"],
                "mae_gain_vs_baseline": baseline_metrics["mae"] - metrics["mae"],
                "n_train_rows": int(metrics["n_train_rows"]),
                "n_test_rows": int(metrics["n_test_rows"]),
                "n_total_rows": int(metrics["n_total_rows"]),
            }
        )
        prediction_frames.append(
            build_feature_prediction_rows(
                split_df=split_df,
                model_name=model_name,
                feature_col=feature_col,
            )
        )

    if include_xgboost:
        for model_name, feature_col in models:
            xgb_metrics = evaluate_xgboost_regression(
                split_df=split_df,
                feature_cols=[feature_col],
                seed=seed,
            )
            rows.append(
                {
                    "model": f"xgb_{model_name}",
                    "features": xgb_metrics["features"],
                    "rmse": xgb_metrics["rmse"],
                    "mae": xgb_metrics["mae"],
                    "r2": xgb_metrics["r2"],
                    "rmse_gain_vs_baseline": baseline_metrics["rmse"] - xgb_metrics["rmse"],
                    "mae_gain_vs_baseline": baseline_metrics["mae"] - xgb_metrics["mae"],
                    "n_train_rows": int(xgb_metrics["n_train_rows"]),
                    "n_test_rows": int(xgb_metrics["n_test_rows"]),
                    "n_total_rows": int(xgb_metrics["n_total_rows"]),
                }
            )
            prediction_frames.append(
                build_xgboost_prediction_rows(
                    split_df=split_df,
                    model_name=f"xgb_{model_name}",
                    feature_cols=[feature_col],
                    seed=seed,
                )
            )
    results_df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    predictions_df = predictions_df.sort_values(
        ["abs_residual", "model"], ascending=[False, True]
    ).reset_index(drop=True)
    return results_df, predictions_df


def evaluate_xgboost_regression(
    split_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> dict[str, float | int | str]:
    """Fit one XGBoost regressor on selected feature columns."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for --xgboost-regression true. Install via pip/uv."
        ) from exc

    eligible_rows = split_df.dropna(subset=feature_cols).copy()
    train_rows = eligible_rows[eligible_rows["is_train"]].copy()
    test_rows = eligible_rows[~eligible_rows["is_train"]].copy()
    if train_rows.empty or test_rows.empty:
        raise ValueError("XGBoost requires non-empty train and test rows")

    X_train = train_rows[feature_cols].astype(float)
    y_train = train_rows["FG3M"].to_numpy(dtype=float)
    X_test = test_rows[feature_cols].astype(float)
    y_test = test_rows["FG3M"].to_numpy(dtype=float)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=int(seed),
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_test), 0.0, None)
    metrics = compute_metrics(y_true=y_test, y_pred=y_pred)
    metrics["features"] = ",".join(feature_cols)
    metrics["n_train_rows"] = int(len(train_rows))
    metrics["n_test_rows"] = int(len(test_rows))
    metrics["n_total_rows"] = int(len(eligible_rows))
    return metrics


def build_xgboost_prediction_rows(
    split_df: pd.DataFrame,
    model_name: str,
    feature_cols: list[str],
    seed: int,
) -> pd.DataFrame:
    """Build per-row test predictions/residuals for one XGBoost model."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for --xgboost-regression true. Install via pip/uv."
        ) from exc

    eligible_rows = split_df.dropna(subset=feature_cols).copy()
    train_rows = eligible_rows[eligible_rows["is_train"]].copy()
    test_rows = eligible_rows[~eligible_rows["is_train"]].copy()
    if train_rows.empty or test_rows.empty:
        raise ValueError(f"XGBoost has empty train/test rows for model '{model_name}'")

    X_train = train_rows[feature_cols].astype(float)
    y_train = train_rows["FG3M"].to_numpy(dtype=float)
    X_test = test_rows[feature_cols].astype(float)
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=int(seed),
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_test), 0.0, None)

    scored = test_rows.copy()
    scored["projection_3pm"] = y_pred.astype(float)
    scored["actual_3pm"] = scored["FG3M"].astype(float)
    scored["residual"] = scored["actual_3pm"] - scored["projection_3pm"]
    scored["abs_residual"] = scored["residual"].abs()
    return scored.assign(
        model=model_name,
        features=",".join(feature_cols),
    )[
        [
            "model",
            "features",
            "GAME_ID",
            "date",
            "MATCHUP",
            "player_normalized",
            "projection_3pm",
            "actual_3pm",
            "residual",
            "abs_residual",
        ]
    ].rename(
        columns={
            "GAME_ID": "game_id",
            "MATCHUP": "matchup",
        }
    )


def main() -> None:
    """Run top-shooter simple 3PM model sweep and report metrics."""
    args = parse_args()
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip() != ""]
    include_xgboost = parse_bool_flag(args.xgboost_regression)
    if len(windows) == 0:
        raise ValueError("--windows cannot be empty")
    np.random.seed(int(args.seed))

    logs = load_logs_from_s3(season=args.season)
    cohort = build_top_shooter_cohort(
        logs_df=logs,
        top_n=args.top_n,
        min_games=args.min_games,
    )
    features_df = build_lagged_3pm_features(
        cohort_df=cohort,
        windows=windows,
        include_ewm=args.include_ewm,
        include_shrinkage=args.include_shrinkage,
        shrinkage_k=args.shrinkage_k,
    )
    if args.include_market_consensus:
        consensus_df = load_market_consensus_lines(season=args.season)
        features_df = features_df.merge(
            consensus_df,
            on=["date", "player_normalized"],
            how="left",
        )
    split_df = split_train_test_by_player(
        features_df=features_df,
        train_frac=args.train_frac,
        min_train_games=args.min_train_games,
    )
    results, predictions = run_model_sweep(
        split_df=split_df,
        windows=windows,
        include_ewm=args.include_ewm,
        include_shrinkage=args.include_shrinkage,
        include_market_consensus=args.include_market_consensus,
        include_xgboost=include_xgboost,
        seed=int(args.seed),
    )

    print(
        "baseline_model=expanding_mean_fg3m",
        f"season={args.season}",
        f"top_n={args.top_n}",
        f"min_games={args.min_games}",
        f"seed={args.seed}",
        sep=" | ",
    )
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.output_csv != "":
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        print(f"\nSaved CSV: {out_path}")
    if args.output_predictions_csv != "":
        preds_out_path = Path(args.output_predictions_csv)
        preds_out_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(preds_out_path, index=False)
        print(f"Saved predictions CSV: {preds_out_path}")


if __name__ == "__main__":
    main()
