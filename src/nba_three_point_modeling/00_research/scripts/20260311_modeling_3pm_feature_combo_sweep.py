"""
Feature-combo sweep for same-game NBA 3PM ideation.

Context:
- This script is intentionally research-only and does not enforce production
  constraints like lagged features or walk-forward simulation.
- It ranks model specs by RMSE/MAE/R2 on player-game rows where the market had
  at least one player_threes line for that player/date.
- It supports both manual model registries and automated selection
  (forward/backward/both) to surface predictive feature combinations.

Primary output columns:
- model
- features
- rmse
- mae
- r2
- rmse_gain_vs_baseline
- mae_gain_vs_baseline
- n_total_rows
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent.parent
UTILS_DIR = ROOT_DIR / "99_utils"
SRC_DIR = ROOT_DIR.parent
REPO_ROOT = SRC_DIR.parent
for extra_path in [str(UTILS_DIR), str(REPO_ROOT)]:
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from duckdb_s3 import connect_duckdb_s3
from src.player_team_history.name_normalization import normalize_from_nba_api
from src.player_team_history.name_normalization import normalize_from_odds_api


@dataclass
class ModelSpec:
    """Model configuration for registry-driven scoring."""

    name: str
    feature_cols: list[str]
    fit_type: str  # baseline_mean | ols
    tags: list[str]


def parse_args() -> argparse.Namespace:
    """Parse CLI args for v4 feature-combo sweep."""
    parser = argparse.ArgumentParser(
        description="Run feature-combo sweep and optional stepwise selection."
    )
    parser.add_argument("--season", type=str, default="*")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--output-predictions-csv", type=str, default="")
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="none",
        choices=["none", "forward", "backward", "both"],
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="rmse",
        choices=["rmse", "mae", "r2"],
    )
    parser.add_argument("--selection-max-features", type=int, default=8)
    parser.add_argument("--selection-min-features", type=int, default=1)
    parser.add_argument("--selection-improvement-threshold", type=float, default=0.0)
    parser.add_argument("--candidate-features", type=str, default="")
    parser.add_argument("--include-manual-models", type=str, default="true")
    parser.add_argument("--save-selection-trace-csv", type=str, default="")
    parser.add_argument("--feature-importance-csv", type=str, default="")
    parser.add_argument("--model-config", type=str, default="")
    parser.add_argument("--top-feature-frequency-k", type=int, default=10)
    return parser.parse_args()


def parse_bool_flag(value: str) -> bool:
    """Parse common string forms of booleans from CLI."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean flag value: {value}")


def load_player_logs_from_s3(season: str) -> pd.DataFrame:
    """Load player game logs from S3 for one season glob."""
    con = connect_duckdb_s3()
    query = f"""
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
      FG3_PCT
    FROM read_csv_auto('s3://nba-api-mt/player_game_logs/{season}/*.csv', union_by_name=true)
    """
    logs = con.execute(query).fetchdf()
    con.close()

    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    for col in [
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
    ]:
        logs[col] = logs[col].astype(float)
    logs["player_normalized"] = logs["PLAYER_NAME"].apply(normalize_from_nba_api)
    logs["date"] = logs["GAME_DATE"].dt.date.astype(str)
    return logs


def load_market_eligible_player_dates(season: str) -> pd.DataFrame:
    """Load player/date combos where market has >=1 player_threes line."""
    con = connect_duckdb_s3()
    query = f"""
    SELECT
      player,
      game_time,
      market,
      prop_line
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
    return props[["player_normalized", "date"]].drop_duplicates().reset_index(drop=True)


def build_eval_universe(logs_df: pd.DataFrame, eligible_df: pd.DataFrame) -> pd.DataFrame:
    """Join logs to market-eligible player/date universe."""
    eval_df = logs_df.merge(
        eligible_df,
        on=["player_normalized", "date"],
        how="inner",
    ).copy()
    eval_df = eval_df[eval_df["FG3M"].notna()].copy()
    if eval_df.empty:
        raise ValueError("No market-eligible player-game rows available for evaluation")
    return eval_df


def default_candidate_features() -> list[str]:
    """Return default raw box-score feature pool."""
    return [
        "MIN",
        "FG3A",
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


def resolve_candidate_features(eval_df: pd.DataFrame, arg_value: str) -> list[str]:
    """Resolve candidate features from CLI override or defaults."""
    if arg_value.strip() == "":
        features = default_candidate_features()
    else:
        features = [x.strip() for x in arg_value.split(",") if x.strip() != ""]
    missing = [f for f in features if f not in eval_df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing candidate feature columns: {missing}")
    return features


def build_manual_registry(candidate_features: list[str]) -> list[ModelSpec]:
    """Build manual model registry; easy to append new combos."""
    specs = [
        ModelSpec(name="baseline", feature_cols=[], fit_type="baseline_mean", tags=["base"]),
        ModelSpec(name="m_min", feature_cols=["MIN"], fit_type="ols", tags=["manual"]),
        ModelSpec(name="m_fga3", feature_cols=["FG3A"], fit_type="ols", tags=["manual"]),
        ModelSpec(name="m_pts", feature_cols=["PTS"], fit_type="ols", tags=["manual"]),
        ModelSpec(name="m_min_fga3", feature_cols=["MIN", "FG3A"], fit_type="ols", tags=["manual"]),
        ModelSpec(name="m_fga3_pts", feature_cols=["FG3A", "PTS"], fit_type="ols", tags=["manual"]),
        ModelSpec(name="m_min_pts", feature_cols=["MIN", "PTS"], fit_type="ols", tags=["manual"]),
        ModelSpec(
            name="m_min_fga3_pts",
            feature_cols=["MIN", "FG3A", "PTS"],
            fit_type="ols",
            tags=["manual"],
        ),
        ModelSpec(
            name="m_box3",
            feature_cols=["MIN", "FG3A", "PTS", "AST", "REB"],
            fit_type="ols",
            tags=["manual"],
        ),
        ModelSpec(
            name="m_box5",
            feature_cols=["MIN", "FG3A", "PTS", "AST", "REB", "TOV", "FGA"],
            fit_type="ols",
            tags=["manual"],
        ),
    ]
    candidate_set = set(candidate_features)
    for spec in specs:
        unknown = [f for f in spec.feature_cols if f not in candidate_set]
        if len(unknown) > 0:
            raise ValueError(
                f"Manual model '{spec.name}' uses features outside candidate pool: {unknown}"
            )
    return specs


def load_model_config_specs(model_config_path: str) -> list[ModelSpec]:
    """Load optional custom specs from JSON or YAML config file."""
    if model_config_path.strip() == "":
        return []
    path = Path(model_config_path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    elif path.suffix.lower() in {".yml", ".yaml"}:
        payload = yaml.safe_load(text)
    else:
        raise ValueError("model_config must be .json, .yml, or .yaml")
    if not isinstance(payload, list):
        raise ValueError("model_config payload must be a list of model specs")

    specs: list[ModelSpec] = []
    for item in payload:
        specs.append(
            ModelSpec(
                name=item["name"],
                feature_cols=list(item["feature_cols"]),
                fit_type=item["fit_type"],
                tags=list(item["tags"]) if "tags" in item else [],
            )
        )
    return specs


def format_features(feature_cols: list[str], fit_type: str) -> str:
    """Format feature string for output table."""
    if fit_type == "baseline_mean":
        return "baseline_mean"
    return ",".join(feature_cols)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and R2."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def score_model_spec(eval_df: pd.DataFrame, spec: ModelSpec) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit and score one model spec over the market-eligible universe."""
    if spec.fit_type not in {"baseline_mean", "ols"}:
        raise ValueError(f"Unsupported fit_type: {spec.fit_type}")

    required = ["FG3M"] + spec.feature_cols
    missing_cols = [col for col in required if col not in eval_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing columns for model '{spec.name}': {missing_cols}")

    scored = eval_df.dropna(subset=required).copy()
    if scored.empty:
        raise ValueError(f"No rows available after null drop for model '{spec.name}'")

    y_true = scored["FG3M"].to_numpy(dtype=float)
    if spec.fit_type == "baseline_mean":
        mean_value = float(np.mean(y_true))
        y_pred = np.full(len(scored), fill_value=mean_value, dtype=float)
    else:
        x = scored[spec.feature_cols].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(scored)), x])
        coefs = np.linalg.lstsq(X, y_true, rcond=None)[0]
        y_pred = X @ coefs
    y_pred = np.clip(y_pred, 0.0, None)

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    row = {
        "model": spec.name,
        "features": format_features(spec.feature_cols, spec.fit_type),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "n_total_rows": int(len(scored)),
    }
    pred_df = scored.assign(
        model=spec.name,
        features=format_features(spec.feature_cols, spec.fit_type),
        projection_3pm=y_pred.astype(float),
        actual_3pm=y_true.astype(float),
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
        ]
    ].rename(columns={"GAME_ID": "game_id", "MATCHUP": "matchup"})
    pred_df["residual"] = pred_df["actual_3pm"] - pred_df["projection_3pm"]
    pred_df["abs_residual"] = pred_df["residual"].abs()
    return row, pred_df


def metric_is_higher_better(metric: str) -> bool:
    """Return whether higher metric values are better."""
    if metric == "r2":
        return True
    return False


def metric_improvement(metric: str, old_value: float, new_value: float) -> float:
    """Compute positive-improvement direction by metric type."""
    if metric_is_higher_better(metric):
        return float(new_value - old_value)
    return float(old_value - new_value)


def choose_better(metric: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Pick better score row by selected optimization metric."""
    if metric_is_higher_better(metric):
        return left if float(left[metric]) >= float(right[metric]) else right
    return left if float(left[metric]) <= float(right[metric]) else right


def score_feature_set(
    eval_df: pd.DataFrame,
    model_name: str,
    feature_cols: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Convenience scorer for OLS feature-set models."""
    return score_model_spec(
        eval_df=eval_df,
        spec=ModelSpec(
            name=model_name,
            feature_cols=feature_cols,
            fit_type="ols",
            tags=["selection"],
        ),
    )


def run_forward_selection(
    eval_df: pd.DataFrame,
    candidate_features: list[str],
    metric: str,
    max_features: int,
    improvement_threshold: float,
    baseline_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]], list[str]]:
    """Run greedy forward selection with threshold stopping."""
    selected: list[str] = []
    accepted_rows: list[dict[str, Any]] = []
    accepted_preds: list[pd.DataFrame] = []
    trace_rows: list[dict[str, Any]] = []
    current_row = baseline_row

    while len(selected) < int(max_features):
        remaining = [f for f in candidate_features if f not in selected]
        if len(remaining) == 0:
            break

        best_candidate_row: dict[str, Any] | None = None
        best_candidate_preds: pd.DataFrame | None = None
        best_feature = ""
        for feature in remaining:
            trial_features = selected + [feature]
            trial_name = f"step_fwd_k{len(trial_features)}"
            trial_row, trial_preds = score_feature_set(
                eval_df=eval_df,
                model_name=trial_name,
                feature_cols=trial_features,
            )
            if best_candidate_row is None:
                best_candidate_row = trial_row
                best_candidate_preds = trial_preds
                best_feature = feature
            else:
                chosen = choose_better(metric, trial_row, best_candidate_row)
                if chosen is trial_row:
                    best_candidate_row = trial_row
                    best_candidate_preds = trial_preds
                    best_feature = feature

        assert best_candidate_row is not None
        assert best_candidate_preds is not None
        improvement = metric_improvement(
            metric=metric,
            old_value=float(current_row[metric]),
            new_value=float(best_candidate_row[metric]),
        )
        trace_rows.append(
            {
                "mode": "forward",
                "step": int(len(selected) + 1),
                "action_feature": best_feature,
                "selected_features": ",".join(selected + [best_feature]),
                "metric": metric,
                "metric_value": float(best_candidate_row[metric]),
                "improvement": float(improvement),
                "rmse": float(best_candidate_row["rmse"]),
                "mae": float(best_candidate_row["mae"]),
                "r2": float(best_candidate_row["r2"]),
                "n_total_rows": int(best_candidate_row["n_total_rows"]),
            }
        )
        if improvement < float(improvement_threshold):
            break

        selected.append(best_feature)
        accepted_rows.append(best_candidate_row)
        accepted_preds.append(best_candidate_preds)
        current_row = best_candidate_row

    if len(selected) == 0:
        final_row = {
            "model": "step_fwd_best",
            "features": "baseline_mean",
            "rmse": float(baseline_row["rmse"]),
            "mae": float(baseline_row["mae"]),
            "r2": float(baseline_row["r2"]),
            "n_total_rows": int(baseline_row["n_total_rows"]),
        }
        final_preds = pd.DataFrame()
    else:
        final_row, final_preds = score_feature_set(
            eval_df=eval_df,
            model_name="step_fwd_best",
            feature_cols=selected,
        )
    return accepted_rows + [final_row], accepted_preds + [final_preds], trace_rows, selected


def run_backward_selection(
    eval_df: pd.DataFrame,
    candidate_features: list[str],
    metric: str,
    min_features: int,
    improvement_threshold: float,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]], list[str]]:
    """Run greedy backward elimination with threshold stopping."""
    current_features = list(candidate_features)
    current_row, _ = score_feature_set(
        eval_df=eval_df,
        model_name=f"step_bwd_k{len(current_features)}",
        feature_cols=current_features,
    )
    accepted_rows: list[dict[str, Any]] = []
    accepted_preds: list[pd.DataFrame] = []
    trace_rows: list[dict[str, Any]] = []
    step_counter = 0

    while len(current_features) > int(min_features):
        best_candidate_row: dict[str, Any] | None = None
        best_candidate_preds: pd.DataFrame | None = None
        removed_feature = ""

        for feature in current_features:
            trial_features = [f for f in current_features if f != feature]
            trial_row, trial_preds = score_feature_set(
                eval_df=eval_df,
                model_name=f"step_bwd_k{len(trial_features)}",
                feature_cols=trial_features,
            )
            if best_candidate_row is None:
                best_candidate_row = trial_row
                best_candidate_preds = trial_preds
                removed_feature = feature
            else:
                chosen = choose_better(metric, trial_row, best_candidate_row)
                if chosen is trial_row:
                    best_candidate_row = trial_row
                    best_candidate_preds = trial_preds
                    removed_feature = feature

        assert best_candidate_row is not None
        assert best_candidate_preds is not None
        step_counter += 1
        improvement = metric_improvement(
            metric=metric,
            old_value=float(current_row[metric]),
            new_value=float(best_candidate_row[metric]),
        )
        trace_rows.append(
            {
                "mode": "backward",
                "step": int(step_counter),
                "action_feature": removed_feature,
                "selected_features": best_candidate_row["features"],
                "metric": metric,
                "metric_value": float(best_candidate_row[metric]),
                "improvement": float(improvement),
                "rmse": float(best_candidate_row["rmse"]),
                "mae": float(best_candidate_row["mae"]),
                "r2": float(best_candidate_row["r2"]),
                "n_total_rows": int(best_candidate_row["n_total_rows"]),
            }
        )
        if improvement < float(improvement_threshold):
            break

        current_features = [f for f in current_features if f != removed_feature]
        current_row = best_candidate_row
        accepted_rows.append(best_candidate_row)
        accepted_preds.append(best_candidate_preds)

    final_row, final_preds = score_feature_set(
        eval_df=eval_df,
        model_name="step_bwd_best",
        feature_cols=current_features,
    )
    return accepted_rows + [final_row], accepted_preds + [final_preds], trace_rows, current_features


def compute_feature_importance_summary(
    results_df: pd.DataFrame,
    forward_selected: list[str],
    backward_selected: list[str],
    top_k: int,
) -> pd.DataFrame:
    """Build interpretable feature summary across selected/top models."""
    forward_step_lookup = {feature: i + 1 for i, feature in enumerate(forward_selected)}
    backward_set = set(backward_selected)

    candidate_rows = results_df[
        (~results_df["model"].str.startswith("step_")) & (results_df["model"] != "baseline")
    ].copy()
    top_rows = candidate_rows.sort_values("rmse").head(int(top_k))
    frequency: dict[str, int] = {}
    for feature_str in top_rows["features"].tolist():
        if feature_str in {"", "baseline_mean"}:
            continue
        for feature in [x for x in feature_str.split(",") if x != ""]:
            frequency[feature] = frequency.get(feature, 0) + 1

    all_features = set(list(frequency.keys()) + forward_selected + list(backward_set))
    rows = []
    for feature in sorted(all_features):
        rows.append(
            {
                "feature": feature,
                "selected_in_forward_step": forward_step_lookup[feature]
                if feature in forward_step_lookup
                else 0,
                "selected_in_backward_final": int(feature in backward_set),
                "frequency_across_top_models": int(frequency.get(feature, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selected_in_forward_step", "frequency_across_top_models", "feature"],
        ascending=[True, False, True],
    )


def main() -> None:
    """Run v4 feature-combo sweep and optional selection routines."""
    args = parse_args()
    np.random.seed(int(args.seed))
    include_manual_models = parse_bool_flag(args.include_manual_models)

    logs_df = load_player_logs_from_s3(season=args.season)
    eligible_df = load_market_eligible_player_dates(season=args.season)
    eval_df = build_eval_universe(logs_df=logs_df, eligible_df=eligible_df)
    candidate_features = resolve_candidate_features(eval_df=eval_df, arg_value=args.candidate_features)

    registry: list[ModelSpec] = []
    if include_manual_models:
        registry.extend(build_manual_registry(candidate_features=candidate_features))
    else:
        registry.append(
            ModelSpec(name="baseline", feature_cols=[], fit_type="baseline_mean", tags=["base"])
        )
    registry.extend(load_model_config_specs(model_config_path=args.model_config))

    names = [spec.name for spec in registry]
    if len(names) != len(set(names)):
        raise ValueError("Model names must be unique across registry + model-config")

    rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    baseline_row: dict[str, Any] | None = None
    for spec in registry:
        row, pred_df = score_model_spec(eval_df=eval_df, spec=spec)
        rows.append(row)
        pred_frames.append(pred_df)
        if row["model"] == "baseline":
            baseline_row = row
    if baseline_row is None:
        raise ValueError("Baseline model must be present and named 'baseline'")

    selection_traces: list[dict[str, Any]] = []
    forward_selected: list[str] = []
    backward_selected: list[str] = []

    if args.selection_mode in {"forward", "both"}:
        f_rows, f_preds, f_trace, f_selected = run_forward_selection(
            eval_df=eval_df,
            candidate_features=candidate_features,
            metric=args.selection_metric,
            max_features=args.selection_max_features,
            improvement_threshold=args.selection_improvement_threshold,
            baseline_row=baseline_row,
        )
        rows.extend(f_rows)
        pred_frames.extend([x for x in f_preds if not x.empty])
        selection_traces.extend(f_trace)
        forward_selected = f_selected

    if args.selection_mode in {"backward", "both"}:
        b_rows, b_preds, b_trace, b_selected = run_backward_selection(
            eval_df=eval_df,
            candidate_features=candidate_features,
            metric=args.selection_metric,
            min_features=args.selection_min_features,
            improvement_threshold=args.selection_improvement_threshold,
        )
        rows.extend(b_rows)
        pred_frames.extend([x for x in b_preds if not x.empty])
        selection_traces.extend(b_trace)
        backward_selected = b_selected

    results_df = pd.DataFrame(rows)
    baseline_rmse = float(results_df.loc[results_df["model"] == "baseline", "rmse"].iloc[0])
    baseline_mae = float(results_df.loc[results_df["model"] == "baseline", "mae"].iloc[0])
    results_df["rmse_gain_vs_baseline"] = baseline_rmse - results_df["rmse"]
    results_df["mae_gain_vs_baseline"] = baseline_mae - results_df["mae"]
    results_df = results_df[
        [
            "model",
            "features",
            "rmse",
            "mae",
            "r2",
            "rmse_gain_vs_baseline",
            "mae_gain_vs_baseline",
            "n_total_rows",
        ]
    ].sort_values("rmse").reset_index(drop=True)

    print(
        "baseline_model=baseline",
        f"season={args.season}",
        f"selection_mode={args.selection_mode}",
        f"selection_metric={args.selection_metric}",
        f"seed={args.seed}",
        sep=" | ",
    )
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.output_csv.strip() != "":
        out_path = Path(args.output_csv).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved CSV: {out_path}")

    if args.output_predictions_csv.strip() != "":
        pred_out_path = Path(args.output_predictions_csv).expanduser()
        pred_out_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df = pd.concat(pred_frames, ignore_index=True)
        predictions_df = predictions_df.sort_values(
            ["abs_residual", "model"], ascending=[False, True]
        ).reset_index(drop=True)
        predictions_df.to_csv(pred_out_path, index=False)
        print(f"Saved predictions CSV: {pred_out_path}")

    if args.save_selection_trace_csv.strip() != "":
        trace_out_path = Path(args.save_selection_trace_csv).expanduser()
        trace_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(selection_traces).to_csv(trace_out_path, index=False)
        print(f"Saved selection trace CSV: {trace_out_path}")

    if args.feature_importance_csv.strip() != "":
        importance_df = compute_feature_importance_summary(
            results_df=results_df,
            forward_selected=forward_selected,
            backward_selected=backward_selected,
            top_k=args.top_feature_frequency_k,
        )
        imp_out_path = Path(args.feature_importance_csv).expanduser()
        imp_out_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(imp_out_path, index=False)
        print(f"Saved feature importance CSV: {imp_out_path}")


if __name__ == "__main__":
    main()
