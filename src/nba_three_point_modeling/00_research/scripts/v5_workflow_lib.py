"""
Shared helpers for v5 3PM decomposition research workflow.

Context:
- The v5 workflow decomposes FG3M prediction into MIN, FG3A_per_min, and FG3_PCT.
- All phases must run on the exact same market-eligible player/date universe.
- This module centralizes deterministic data loading, feature engineering,
  scoring, selection, and reporting utilities used by phase scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import subprocess
import sys
from typing import Any

import duckdb
import numpy as np
import pandas as pd


def ensure_repo_root_on_syspath() -> Path:
    """Find repo root from cwd and add it to sys.path."""
    current = Path.cwd().resolve()
    while True:
        gitignore = current / ".gitignore"
        src_dir = current / "src"
        if gitignore.exists() and src_dir.exists():
            repo_root = current
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            return repo_root
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root with .gitignore and src/")
        current = current.parent


REPO_ROOT = ensure_repo_root_on_syspath()

from src.player_team_history.name_normalization import normalize_from_nba_api
from src.player_team_history.name_normalization import normalize_from_odds_api


def set_seed(seed: int) -> None:
    """Set deterministic numpy seed for all phase scripts."""
    np.random.seed(int(seed))


def resolve_output_path(path_arg: str, default_filename: str) -> Path:
    """Resolve output path and ensure parent directory exists."""
    raw = path_arg.strip()
    path = Path(raw).expanduser() if raw else Path("~/Downloads/tmp").expanduser() / default_filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect_duckdb_s3() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection configured for S3 access in us-east-2."""
    access_key: str
    secret_key: str
    if "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
        access_key = os.environ["AWS_ACCESS_KEY_ID"]
        secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    else:
        access_key = subprocess.check_output(
            ["aws", "configure", "get", "aws_access_key_id"], text=True
        ).strip()
        secret_key = subprocess.check_output(
            ["aws", "configure", "get", "aws_secret_access_key"], text=True
        ).strip()
        if access_key == "" or secret_key == "":
            raise ValueError(
                "Missing AWS credentials. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                "or configure via `aws configure`."
            )

    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("SET s3_region='us-east-2'")
    con.execute(f"SET s3_access_key_id='{access_key}'")
    con.execute(f"SET s3_secret_access_key='{secret_key}'")
    if "AWS_SESSION_TOKEN" in os.environ:
        con.execute(f"SET s3_session_token='{os.environ['AWS_SESSION_TOKEN']}'")
    return con


def season_predicate(alias: str, season: str) -> str:
    """Build SQL predicate fragment for season filtering."""
    if season.strip() == "*" or season.strip() == "":
        return "TRUE"
    values = [x.strip() for x in season.split(",") if x.strip() != ""]
    if len(values) == 1:
        return f"{alias}.season = '{values[0]}'"
    quoted = ", ".join([f"'{x}'" for x in values])
    return f"{alias}.season IN ({quoted})"


def maybe_read_cache(cache_path: Path, enabled: bool, force_refresh: bool) -> pd.DataFrame | None:
    """Read parquet cache when enabled and present."""
    if enabled and (not force_refresh) and cache_path.exists():
        return pd.read_parquet(cache_path)
    return None


def maybe_write_cache(df: pd.DataFrame, cache_path: Path, enabled: bool) -> None:
    """Write parquet cache when enabled."""
    if enabled:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)


def load_player_logs(
    season: str,
    cache_dir: str,
    use_cache: bool,
    force_refresh_cache: bool,
) -> pd.DataFrame:
    """Load player logs from S3 with optional local parquet cache."""
    cache_path = Path(cache_dir).expanduser() / f"v5_logs_{season.replace(',', '_')}.parquet"
    cached = maybe_read_cache(cache_path, enabled=use_cache, force_refresh=force_refresh_cache)
    if cached is not None:
        return cached

    con = connect_duckdb_s3()
    query = f"""
    WITH raw AS (
      SELECT
        PLAYER_ID,
        PLAYER_NAME,
        GAME_ID,
        GAME_DATE,
        MATCHUP,
        FG3M,
        FG3A,
        MIN,
        FGA,
        FGM,
        PTS,
        AST,
        REB,
        STL,
        BLK,
        TOV,
        FTA,
        FTM,
        FG_PCT,
        FG3_PCT,
        regexp_extract(filename, '/player_game_logs/([^/]+)/', 1) AS season
      FROM read_csv_auto(
        's3://nba-api-mt/player_game_logs/*/*.csv',
        union_by_name=true,
        filename=true
      )
    )
    SELECT *
    FROM raw r
    WHERE {season_predicate('r', season)}
    """
    logs = con.execute(query).fetchdf()
    con.close()

    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs["player_normalized"] = logs["PLAYER_NAME"].apply(normalize_from_nba_api)
    logs["date"] = logs["GAME_DATE"].dt.date.astype(str)
    numeric_cols = [
        "FG3M",
        "FG3A",
        "MIN",
        "FGA",
        "FGM",
        "PTS",
        "AST",
        "REB",
        "STL",
        "BLK",
        "TOV",
        "FTA",
        "FTM",
        "FG_PCT",
        "FG3_PCT",
    ]
    for col in numeric_cols:
        logs[col] = logs[col].astype(float)
    maybe_write_cache(logs, cache_path, enabled=use_cache)
    return logs


def load_player_props(
    season: str,
    cache_dir: str,
    use_cache: bool,
    force_refresh_cache: bool,
) -> pd.DataFrame:
    """Load player props from S3 with optional local parquet cache."""
    cache_path = Path(cache_dir).expanduser() / f"v5_props_{season.replace(',', '_')}.parquet"
    cached = maybe_read_cache(cache_path, enabled=use_cache, force_refresh=force_refresh_cache)
    if cached is not None:
        return cached

    con = connect_duckdb_s3()
    query = f"""
    WITH raw AS (
      SELECT
        player,
        game_time,
        market,
        prop_line,
        over_odds,
        under_odds,
        regexp_extract(filename, '/historical_player_props/([^/]+)/', 1) AS season
      FROM read_csv_auto(
        's3://the-odds-api-mt/nba/historical_player_props/*/*.csv',
        union_by_name=true,
        filename=true
      )
    )
    SELECT *
    FROM raw r
    WHERE {season_predicate('r', season)}
    """
    props = con.execute(query).fetchdf()
    con.close()

    props["player_normalized"] = props["player"].apply(normalize_from_odds_api)
    game_time_utc = pd.to_datetime(props["game_time"], utc=True)
    props["date"] = game_time_utc.dt.tz_convert("America/New_York").dt.date.astype(str)
    props["prop_line"] = pd.to_numeric(props["prop_line"], errors="coerce")
    props["over_odds"] = pd.to_numeric(props["over_odds"], errors="coerce")
    props["under_odds"] = pd.to_numeric(props["under_odds"], errors="coerce")
    maybe_write_cache(props, cache_path, enabled=use_cache)
    return props


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if np.isnan(odds):
        return float("nan")
    if odds < 0:
        return float((-odds) / ((-odds) + 100.0))
    return float(100.0 / (odds + 100.0))


def remove_vig_two_way(p_over: float, p_under: float) -> tuple[float, float]:
    """Remove vig from a two-way market."""
    total = p_over + p_under
    if total <= 0.0:
        return 0.5, 0.5
    return float(p_over / total), float(p_under / total)


def build_market_eligibility(props_df: pd.DataFrame) -> pd.DataFrame:
    """Build unique market-eligible player/date rows with consensus line."""
    props = props_df.copy()
    props = props[(props["market"] == "player_threes") & props["prop_line"].notna()].copy()
    props = props[props["player_normalized"].notna()].copy()
    props["p_over_raw"] = props["over_odds"].apply(american_to_implied_prob)
    props["p_under_raw"] = props["under_odds"].apply(american_to_implied_prob)
    no_vig = props.apply(
        lambda row: remove_vig_two_way(row["p_over_raw"], row["p_under_raw"]),
        axis=1,
    )
    props["p_over_novig"] = [x[0] for x in no_vig]
    line_balance = (
        props.groupby(
            ["season", "player_normalized", "date", "prop_line"], as_index=False
        )["p_over_novig"]
        .median()
        .rename(columns={"p_over_novig": "median_p_over_novig"})
    )
    line_balance["distance_to_5050"] = (line_balance["median_p_over_novig"] - 0.5).abs()
    consensus = (
        line_balance.sort_values(
            ["season", "player_normalized", "date", "distance_to_5050", "prop_line"]
        )
        .groupby(["season", "player_normalized", "date"], as_index=False)
        .first()
        .rename(columns={"prop_line": "market_consensus_line"})
    )
    return consensus[
        ["season", "player_normalized", "date", "market_consensus_line", "median_p_over_novig"]
    ].drop_duplicates()


def build_eval_universe(logs_df: pd.DataFrame, eligible_df: pd.DataFrame) -> pd.DataFrame:
    """Join logs to market-eligible rows and add derived base features."""
    universe = logs_df.merge(
        eligible_df,
        on=["season", "player_normalized", "date"],
        how="inner",
    ).copy()
    universe = universe[universe["FG3M"].notna()].copy()
    universe = universe.rename(columns={"GAME_ID": "game_id", "MATCHUP": "matchup"})
    universe["home_game"] = universe["matchup"].str.contains("vs.", regex=False).astype(int)
    universe = universe.sort_values(["player_normalized", "GAME_DATE", "game_id"]).reset_index(
        drop=True
    )
    return universe


def add_research_features(universe_df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic lagged/context features for modeling."""
    df = universe_df.copy()
    grouped = df.groupby("player_normalized", group_keys=False)
    df["days_since_last_game"] = grouped["GAME_DATE"].diff().dt.days.astype(float)
    df["is_back_to_back"] = (df["days_since_last_game"] <= 1.0).astype(int)

    for base_col in ["MIN", "FG3A", "FG3M", "FG3_PCT", "PTS", "AST", "REB", "FGA"]:
        for window in [5, 10, 20]:
            out_col = f"roll_{base_col.lower()}_{window}"
            df[out_col] = grouped[base_col].transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).mean().shift(1)
            )

    for base_col in ["MIN", "FG3A", "FG3_PCT"]:
        out_col = f"player_season_mean_{base_col.lower()}"
        df[out_col] = (
            df.groupby(["player_normalized", "season"])[base_col]
            .transform(lambda s: s.expanding(min_periods=1).mean().shift(1))
            .astype(float)
        )

    safe_min = np.where(df["MIN"] > 0.0, df["MIN"], np.nan)
    df["FG3A_per_min"] = df["FG3A"] / safe_min
    df["FG3A_per_min"] = df["FG3A_per_min"].replace([np.inf, -np.inf], np.nan)
    df["FG3A_per_min"] = df["FG3A_per_min"].clip(lower=0.0)
    return df


def build_universe_qc(universe_df: pd.DataFrame) -> pd.DataFrame:
    """Build row-count/null/duplicate QC summary rows."""
    rows: list[dict[str, Any]] = []

    by_season = (
        universe_df.groupby("season", as_index=False)
        .agg(row_count=("game_id", "count"))
        .sort_values("season")
    )
    for _, row in by_season.iterrows():
        rows.append(
            {
                "check_type": "row_count_by_season",
                "season": row["season"],
                "metric_name": "row_count",
                "metric_value": float(row["row_count"]),
            }
        )

    keys = ["FG3M", "FG3A", "MIN", "FG3_PCT", "market_consensus_line"]
    for col in keys:
        rows.append(
            {
                "check_type": "null_rate",
                "season": "*",
                "metric_name": col,
                "metric_value": float(universe_df[col].isna().mean()),
            }
        )

    dup_count = universe_df.duplicated(
        subset=["player_normalized", "date", "game_id"], keep=False
    ).sum()
    rows.append(
        {
            "check_type": "duplicate_keys",
            "season": "*",
            "metric_name": "player_normalized_date_game_id",
            "metric_value": float(dup_count),
        }
    )
    return pd.DataFrame(rows)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and R2."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


@dataclass
class ModelSpec:
    """Model definition for registry sweeps and selection."""

    name: str
    feature_cols: list[str]
    fit_type: str  # baseline_mean | baseline_player_season_mean | ols
    target_col: str
    clip_low: float | None
    clip_high: float | None


def format_features(spec: ModelSpec) -> str:
    """Format features for output tables."""
    if spec.fit_type in {"baseline_mean", "baseline_player_season_mean"}:
        return spec.fit_type
    return ",".join(spec.feature_cols)


def apply_clip(values: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    """Apply optional prediction clipping."""
    out = values.copy()
    if low is not None:
        out = np.maximum(out, float(low))
    if high is not None:
        out = np.minimum(out, float(high))
    return out


def fit_predict(
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    spec: ModelSpec,
) -> np.ndarray:
    """Fit model on train_df and predict score_df."""
    if spec.fit_type == "baseline_mean":
        pred = np.full(len(score_df), train_df[spec.target_col].mean(), dtype=float)
        return apply_clip(pred, spec.clip_low, spec.clip_high)

    if spec.fit_type == "baseline_player_season_mean":
        grp = (
            train_df.groupby(["player_normalized", "season"], as_index=False)[spec.target_col]
            .mean()
            .rename(columns={spec.target_col: "ps_mean"})
        )
        merged = score_df.merge(grp, on=["player_normalized", "season"], how="left")
        fallback = float(train_df[spec.target_col].mean())
        pred = merged["ps_mean"].fillna(fallback).to_numpy(dtype=float)
        return apply_clip(pred, spec.clip_low, spec.clip_high)

    if spec.fit_type == "ols":
        required = [spec.target_col] + spec.feature_cols
        train = train_df.dropna(subset=required).copy()
        score = score_df.dropna(subset=spec.feature_cols).copy()
        x_train = train[spec.feature_cols].to_numpy(dtype=float)
        y_train = train[spec.target_col].to_numpy(dtype=float)
        X_train = np.column_stack([np.ones(len(train)), x_train])
        coefs = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

        x_score = score[spec.feature_cols].to_numpy(dtype=float)
        X_score = np.column_stack([np.ones(len(score)), x_score])
        pred_partial = X_score @ coefs
        pred_partial = apply_clip(pred_partial, spec.clip_low, spec.clip_high)

        full = score_df[[spec.target_col]].copy()
        full["pred"] = np.nan
        full.loc[score.index, "pred"] = pred_partial
        return full["pred"].to_numpy(dtype=float)

    raise ValueError(f"Unsupported fit_type: {spec.fit_type}")


def score_model(
    df: pd.DataFrame,
    spec: ModelSpec,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit+score one model on a fixed universe."""
    required = [spec.target_col]
    if spec.fit_type == "ols":
        required = [spec.target_col] + spec.feature_cols
    scored = df.dropna(subset=required).copy()
    pred = fit_predict(train_df=scored, score_df=scored, spec=spec)
    y_true = scored[spec.target_col].to_numpy(dtype=float)
    valid_mask = np.isfinite(pred)
    y_true = y_true[valid_mask]
    y_pred = pred[valid_mask]
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)

    row: dict[str, Any] = {
        "model": spec.name,
        "fit_type": spec.fit_type,
        "features": format_features(spec),
        "target": spec.target_col,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "n_total_rows": int(len(y_true)),
    }

    pred_df = scored.loc[valid_mask].copy()
    pred_df["model"] = spec.name
    pred_df["fit_type"] = spec.fit_type
    pred_df["features"] = format_features(spec)
    pred_df["target"] = spec.target_col
    pred_df["prediction"] = y_pred
    pred_df["actual"] = y_true
    pred_df["residual"] = pred_df["actual"] - pred_df["prediction"]
    pred_df["abs_residual"] = pred_df["residual"].abs()
    return row, pred_df


def metric_higher_better(metric: str) -> bool:
    """Return metric direction."""
    return metric == "r2"


def metric_improvement(metric: str, old_value: float, new_value: float) -> float:
    """Normalize improvement direction by metric type."""
    if metric_higher_better(metric):
        return float(new_value - old_value)
    return float(old_value - new_value)


def choose_better(metric: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Choose better row by optimization metric."""
    if metric_higher_better(metric):
        return left if float(left[metric]) >= float(right[metric]) else right
    return left if float(left[metric]) <= float(right[metric]) else right


def run_forward_selection(
    df: pd.DataFrame,
    target_col: str,
    candidate_features: list[str],
    metric: str,
    max_features: int,
    improvement_threshold: float,
    clip_low: float | None,
    clip_high: float | None,
    baseline_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Greedy forward selection from baseline."""
    selected: list[str] = []
    accepted_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    current = baseline_row

    while len(selected) < int(max_features):
        remaining = [f for f in candidate_features if f not in selected]
        if len(remaining) == 0:
            break
        best_row: dict[str, Any] | None = None
        best_feature = ""
        for feature in remaining:
            spec = ModelSpec(
                name=f"step_fwd_k{len(selected) + 1}",
                feature_cols=selected + [feature],
                fit_type="ols",
                target_col=target_col,
                clip_low=clip_low,
                clip_high=clip_high,
            )
            row, _ = score_model(df=df, spec=spec)
            if best_row is None:
                best_row = row
                best_feature = feature
            else:
                chosen = choose_better(metric, row, best_row)
                if chosen is row:
                    best_row = row
                    best_feature = feature
        improvement = metric_improvement(metric, float(current[metric]), float(best_row[metric]))
        traces.append(
            {
                "mode": "forward",
                "step": int(len(selected) + 1),
                "action_feature": best_feature,
                "selected_features": ",".join(selected + [best_feature]),
                "metric": metric,
                "metric_value": float(best_row[metric]),
                "improvement": float(improvement),
            }
        )
        if improvement < float(improvement_threshold):
            break
        selected.append(best_feature)
        accepted_rows.append(best_row)
        current = best_row
    if len(selected) > 0:
        final_row, _ = score_model(
            df=df,
            spec=ModelSpec(
                name="step_fwd_best",
                feature_cols=selected,
                fit_type="ols",
                target_col=target_col,
                clip_low=clip_low,
                clip_high=clip_high,
            ),
        )
    else:
        final_row = baseline_row.copy()
        final_row["model"] = "step_fwd_best"
    return accepted_rows + [final_row], traces, selected


def run_backward_selection(
    df: pd.DataFrame,
    target_col: str,
    candidate_features: list[str],
    metric: str,
    min_features: int,
    improvement_threshold: float,
    clip_low: float | None,
    clip_high: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Greedy backward elimination from full feature set."""
    current_features = list(candidate_features)
    current_row, _ = score_model(
        df=df,
        spec=ModelSpec(
            name=f"step_bwd_k{len(current_features)}",
            feature_cols=current_features,
            fit_type="ols",
            target_col=target_col,
            clip_low=clip_low,
            clip_high=clip_high,
        ),
    )
    accepted_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    step = 0
    while len(current_features) > int(min_features):
        best_row: dict[str, Any] | None = None
        removed_feature = ""
        for feature in current_features:
            trial_features = [f for f in current_features if f != feature]
            trial_row, _ = score_model(
                df=df,
                spec=ModelSpec(
                    name=f"step_bwd_k{len(trial_features)}",
                    feature_cols=trial_features,
                    fit_type="ols",
                    target_col=target_col,
                    clip_low=clip_low,
                    clip_high=clip_high,
                ),
            )
            if best_row is None:
                best_row = trial_row
                removed_feature = feature
            else:
                chosen = choose_better(metric, trial_row, best_row)
                if chosen is trial_row:
                    best_row = trial_row
                    removed_feature = feature
        step += 1
        improvement = metric_improvement(
            metric, float(current_row[metric]), float(best_row[metric])
        )
        traces.append(
            {
                "mode": "backward",
                "step": int(step),
                "action_feature": removed_feature,
                "selected_features": best_row["features"],
                "metric": metric,
                "metric_value": float(best_row[metric]),
                "improvement": float(improvement),
            }
        )
        if improvement < float(improvement_threshold):
            break
        current_features = [f for f in current_features if f != removed_feature]
        current_row = best_row
        accepted_rows.append(best_row)
    final_row, _ = score_model(
        df=df,
        spec=ModelSpec(
            name="step_bwd_best",
            feature_cols=current_features,
            fit_type="ols",
            target_col=target_col,
            clip_low=clip_low,
            clip_high=clip_high,
        ),
    )
    return accepted_rows + [final_row], traces, current_features


def build_feature_importance(
    results_df: pd.DataFrame,
    forward_selected: list[str],
    backward_selected: list[str],
    top_k: int,
) -> pd.DataFrame:
    """Build a compact feature-importance summary for research outputs."""
    candidate = results_df[
        (~results_df["model"].str.startswith("step_")) & (~results_df["features"].str.startswith("baseline"))
    ].copy()
    top = candidate.sort_values(["rmse", "model"]).head(int(top_k))
    freq: dict[str, int] = {}
    for feature_str in top["features"].tolist():
        for feature in [x for x in feature_str.split(",") if x != ""]:
            freq[feature] = freq.get(feature, 0) + 1

    fwd_lookup = {feature: i + 1 for i, feature in enumerate(forward_selected)}
    bwd_set = set(backward_selected)
    all_features = set(freq.keys()) | set(forward_selected) | bwd_set
    rows = []
    for feature in sorted(all_features):
        rows.append(
            {
                "feature": feature,
                "selected_in_forward_step": int(fwd_lookup[feature]) if feature in fwd_lookup else 0,
                "selected_in_backward_final": int(feature in bwd_set),
                "frequency_across_top_models": int(freq.get(feature, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selected_in_forward_step", "frequency_across_top_models", "feature"],
        ascending=[True, False, True],
    )


def calibration_bins(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Build equal-width calibration bins for bounded targets."""
    frame = pd.DataFrame({"actual": y_true, "pred": y_pred})
    frame["bin"] = pd.cut(frame["pred"], bins=np.linspace(0.0, 1.0, n_bins + 1), include_lowest=True)
    out = (
        frame.groupby("bin", as_index=False)
        .agg(
            n=("actual", "count"),
            pred_mean=("pred", "mean"),
            actual_mean=("actual", "mean"),
        )
        .sort_values("bin")
        .reset_index(drop=True)
    )
    out["calibration_gap"] = (out["pred_mean"] - out["actual_mean"]).abs()
    return out


def poisson_tail_prob(k_threshold: int, lam: float) -> float:
    """Compute P[X >= k_threshold] for Poisson(lam) without scipy."""
    if k_threshold <= 0:
        return 1.0
    lam = max(float(lam), 1e-6)
    cdf = 0.0
    term = math.exp(-lam)
    cdf += term
    for k in range(1, k_threshold):
        term *= lam / float(k)
        cdf += term
    tail = 1.0 - cdf
    return float(min(max(tail, 0.0), 1.0))

