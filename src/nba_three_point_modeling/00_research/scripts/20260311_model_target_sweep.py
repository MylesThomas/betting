"""
Phase 1 target sweep for v5 3PM decomposition workflow.

Context:
- Compare models for one upstream target at a time:
  - min
  - fga_per_min
  - fg3_pct
- Preserve a baseline row for every target sweep.
- Support manual registry and stepwise selection on the canonical phase-0 universe.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5_workflow_lib import build_feature_importance
from v5_workflow_lib import calibration_bins
from v5_workflow_lib import ModelSpec
from v5_workflow_lib import resolve_output_path
from v5_workflow_lib import run_backward_selection
from v5_workflow_lib import run_forward_selection
from v5_workflow_lib import score_model
from v5_workflow_lib import set_seed


TARGET_CONFIG = {
    "min": {"target_col": "MIN", "clip_low": 0.0, "clip_high": None},
    "fga_per_min": {"target_col": "FG3A_per_min", "clip_low": 0.0, "clip_high": None},
    "fg3_pct": {"target_col": "FG3_PCT", "clip_low": 0.0, "clip_high": 1.0},
}


def parse_args() -> argparse.Namespace:
    """Parse CLI args for phase 1 target sweeps."""
    parser = argparse.ArgumentParser(description="Run v5 target model sweep.")
    parser.add_argument("--phase", type=str, default="phase1")
    parser.add_argument("--target", type=str, required=True, choices=["min", "fga_per_min", "fg3_pct"])
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument(
        "--input-universe",
        "--input-universe-path",
        dest="input_universe",
        type=str,
        default="~/Downloads/tmp/v5_eval_universe.parquet",
    )
    parser.add_argument("--selection-mode", type=str, default="both", choices=["none", "forward", "backward", "both"])
    parser.add_argument("--selection-metric", type=str, default="rmse", choices=["rmse", "mae", "r2"])
    parser.add_argument("--selection-max-features", type=int, default=8)
    parser.add_argument("--selection-min-features", type=int, default=1)
    parser.add_argument("--selection-improvement-threshold", type=float, default=0.0)
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--output-trace-csv", type=str, default="")
    parser.add_argument("--output-importance-csv", type=str, default="")
    parser.add_argument("--output-calibration-csv", type=str, default="")
    return parser.parse_args()


def candidate_features(df: pd.DataFrame, target: str) -> list[str]:
    """Return deterministic feature pool by target."""
    common = [
        "home_game",
        "days_since_last_game",
        "is_back_to_back",
        "PTS",
        "AST",
        "REB",
        "FGA",
        "FGM",
        "TOV",
        "FTA",
        "FTM",
        "FG_PCT",
        "roll_min_5",
        "roll_min_10",
        "roll_fg3a_5",
        "roll_fg3a_10",
        "roll_fg3m_5",
        "roll_fg3m_10",
        "roll_fg3_pct_5",
        "roll_fg3_pct_10",
        "player_season_mean_min",
        "player_season_mean_fg3a",
        "player_season_mean_fg3_pct",
        "market_consensus_line",
        "median_p_over_novig",
    ]
    if target == "min":
        blocked = {"MIN", "FG3A", "FG3M", "FG3_PCT", "FG3A_per_min"}
        return [x for x in common if x in df.columns and x not in blocked]
    if target == "fga_per_min":
        blocked = {"FG3A_per_min", "FG3A", "MIN", "FG3M", "FG3_PCT"}
        return [x for x in common if x in df.columns and x not in blocked]
    blocked = {"FG3_PCT", "FG3M", "FG3A", "FG3A_per_min", "MIN"}
    return [x for x in common if x in df.columns and x not in blocked]


def build_manual_registry(target: str, features: list[str]) -> list[ModelSpec]:
    """Build compact, explicit manual registry including baseline."""
    cfg = TARGET_CONFIG[target]
    target_col = cfg["target_col"]
    clip_low = cfg["clip_low"]
    clip_high = cfg["clip_high"]
    registry = [
        ModelSpec(
            name="baseline",
            feature_cols=[],
            fit_type="baseline_player_season_mean",
            target_col=target_col,
            clip_low=clip_low,
            clip_high=clip_high,
        ),
        ModelSpec(
            name="m_core",
            feature_cols=[f for f in ["roll_fg3m_5", "roll_fg3a_5", "roll_min_5"] if f in features],
            fit_type="ols",
            target_col=target_col,
            clip_low=clip_low,
            clip_high=clip_high,
        ),
        ModelSpec(
            name="m_box",
            feature_cols=[f for f in ["PTS", "AST", "REB", "FGA", "FG_PCT"] if f in features],
            fit_type="ols",
            target_col=target_col,
            clip_low=clip_low,
            clip_high=clip_high,
        ),
        ModelSpec(
            name="m_context",
            feature_cols=[
                f
                for f in ["home_game", "days_since_last_game", "is_back_to_back", "market_consensus_line"]
                if f in features
            ],
            fit_type="ols",
            target_col=target_col,
            clip_low=clip_low,
            clip_high=clip_high,
        ),
        ModelSpec(
            name="m_box_context",
            feature_cols=[
                f
                for f in [
                    "PTS",
                    "AST",
                    "REB",
                    "FGA",
                    "FG_PCT",
                    "home_game",
                    "days_since_last_game",
                    "is_back_to_back",
                    "market_consensus_line",
                ]
                if f in features
            ],
            fit_type="ols",
            target_col=target_col,
            clip_low=clip_low,
            clip_high=clip_high,
        ),
    ]
    return registry


def calibration_summary(pred_df: pd.DataFrame, model_name: str) -> tuple[float, float]:
    """Compute calibration summaries for FG3_PCT rows."""
    subset = pred_df[pred_df["model"] == model_name].copy()
    bins = calibration_bins(
        y_true=subset["actual"].to_numpy(dtype=float),
        y_pred=subset["prediction"].to_numpy(dtype=float),
        n_bins=10,
    )
    mean_abs_gap = float(bins["calibration_gap"].mean())
    weighted_abs_gap = float((bins["calibration_gap"] * bins["n"]).sum() / bins["n"].sum())
    return mean_abs_gap, weighted_abs_gap


def default_out_name(target: str, suffix: str) -> str:
    """Build default output filename for target."""
    return f"v5_{target}_{suffix}"


def main() -> None:
    """Run phase 1 target sweep and save model/trace/importance outputs."""
    args = parse_args()
    set_seed(int(args.seed))
    cfg = TARGET_CONFIG[args.target]
    df = pd.read_parquet(Path(args.input_universe).expanduser())
    df = df.sort_values(["season", "date", "player_normalized", "game_id"]).reset_index(drop=True)

    feats = candidate_features(df=df, target=args.target)
    registry = build_manual_registry(target=args.target, features=feats)
    rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    baseline_row: dict[str, Any] | None = None

    for spec in registry:
        if spec.fit_type == "ols" and len(spec.feature_cols) == 0:
            continue
        row, pred_df = score_model(df=df, spec=spec)
        rows.append(row)
        pred_frames.append(pred_df)
        if spec.name == "baseline":
            baseline_row = row

    if baseline_row is None:
        raise ValueError("Baseline row missing from manual registry")

    traces: list[dict[str, Any]] = []
    forward_selected: list[str] = []
    backward_selected: list[str] = []

    if args.selection_mode in {"forward", "both"}:
        f_rows, f_traces, f_selected = run_forward_selection(
            df=df,
            target_col=cfg["target_col"],
            candidate_features=feats,
            metric=args.selection_metric,
            max_features=args.selection_max_features,
            improvement_threshold=args.selection_improvement_threshold,
            clip_low=cfg["clip_low"],
            clip_high=cfg["clip_high"],
            baseline_row=baseline_row,
        )
        rows.extend(f_rows)
        traces.extend(f_traces)
        forward_selected = f_selected

    if args.selection_mode in {"backward", "both"}:
        b_rows, b_traces, b_selected = run_backward_selection(
            df=df,
            target_col=cfg["target_col"],
            candidate_features=feats,
            metric=args.selection_metric,
            min_features=args.selection_min_features,
            improvement_threshold=args.selection_improvement_threshold,
            clip_low=cfg["clip_low"],
            clip_high=cfg["clip_high"],
        )
        rows.extend(b_rows)
        traces.extend(b_traces)
        backward_selected = b_selected

    results = pd.DataFrame(rows).drop_duplicates(subset=["model", "features"], keep="first")
    baseline_rmse = float(results.loc[results["model"] == "baseline", "rmse"].iloc[0])
    baseline_mae = float(results.loc[results["model"] == "baseline", "mae"].iloc[0])
    results["rmse_gain_vs_baseline"] = baseline_rmse - results["rmse"]
    results["mae_gain_vs_baseline"] = baseline_mae - results["mae"]

    if args.target == "fg3_pct":
        merged_preds = pd.concat(pred_frames, ignore_index=True)
        cal_map: dict[str, tuple[float, float]] = {}
        for model_name in sorted(merged_preds["model"].unique().tolist()):
            cal_map[model_name] = calibration_summary(merged_preds, model_name)
        results["calibration_gap_mean"] = results["model"].map(
            lambda x: cal_map[x][0] if x in cal_map else np.nan
        )
        results["calibration_gap_weighted"] = results["model"].map(
            lambda x: cal_map[x][1] if x in cal_map else np.nan
        )

    results = results.sort_values(["rmse", "model"]).reset_index(drop=True)
    importance_df = build_feature_importance(
        results_df=results,
        forward_selected=forward_selected,
        backward_selected=backward_selected,
        top_k=10,
    )
    traces_df = pd.DataFrame(traces)

    out_models = resolve_output_path(args.output_csv, default_out_name(args.target, "models.csv"))
    out_trace = resolve_output_path(args.output_trace_csv, default_out_name(args.target, "trace.csv"))
    out_importance = resolve_output_path(
        args.output_importance_csv, default_out_name(args.target, "importance.csv")
    )
    results.to_csv(out_models, index=False)
    traces_df.to_csv(out_trace, index=False)
    importance_df.to_csv(out_importance, index=False)

    if args.target == "fg3_pct" and args.output_calibration_csv.strip() != "":
        cal_out = resolve_output_path(args.output_calibration_csv, "v5_fg3_pct_calibration.csv")
        top_model = results.iloc[0]["model"]
        merged_preds = pd.concat(pred_frames, ignore_index=True)
        bins_df = calibration_bins(
            y_true=merged_preds.loc[merged_preds["model"] == top_model, "actual"].to_numpy(dtype=float),
            y_pred=merged_preds.loc[merged_preds["model"] == top_model, "prediction"].to_numpy(dtype=float),
            n_bins=10,
        )
        bins_df["model"] = top_model
        bins_df.to_csv(cal_out, index=False)

    print(
        "phase=phase1",
        f"target={args.target}",
        f"seed={args.seed}",
        f"selection_mode={args.selection_mode}",
        f"rows={len(df)}",
        f"models={out_models}",
        f"trace={out_trace}",
        f"importance={out_importance}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()

