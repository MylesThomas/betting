"""
Phase 3 robustness and segment analysis for v5 3PM workflow.

Context:
- Re-score top FG3M candidate models across key segments:
  season, line bucket, player volume tier, home/away, rest bucket.
- Surface stability risks where segment ranking diverges from global ranking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v5_workflow_lib import compute_metrics
from v5_workflow_lib import resolve_output_path
from v5_workflow_lib import set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI args for phase 3."""
    parser = argparse.ArgumentParser(description="Run v5 segment robustness analysis.")
    parser.add_argument("--phase", type=str, default="phase3")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument(
        "--input-universe",
        "--input-universe-path",
        dest="input_universe",
        type=str,
        default="~/Downloads/tmp/v5_eval_universe.parquet",
    )
    parser.add_argument(
        "--input-predictions", type=str, default="~/Downloads/tmp/v5_fg3m_recompose_predictions.csv"
    )
    parser.add_argument(
        "--input-comparison", type=str, default="~/Downloads/tmp/v5_fg3m_recompose_comparison.csv"
    )
    parser.add_argument("--top-k-models", type=int, default=4)
    parser.add_argument("--output-segment-csv", type=str, default="")
    parser.add_argument("--output-stability-csv", type=str, default="")
    return parser.parse_args()


def line_bucket(line: float) -> str:
    """Bucket market lines for segment analysis."""
    if line <= 1.5:
        return "line_1_5"
    if line <= 2.5:
        return "line_2_5"
    return "line_3_5_plus"


def rest_bucket(days: float) -> str:
    """Bucket rest days for segment analysis."""
    if np.isnan(days):
        return "unknown"
    if days <= 1.0:
        return "rest_0_1"
    if days <= 2.0:
        return "rest_2"
    return "rest_3_plus"


def metric_row(df: pd.DataFrame, segment_type: str, segment_value: str, model: str) -> dict[str, float | str | int]:
    """Compute metrics row for one model/segment slice."""
    y_true = df["actual_fg3m"].to_numpy(dtype=float)
    y_pred = df["prediction_fg3m"].to_numpy(dtype=float)
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    return {
        "segment_type": segment_type,
        "segment_value": segment_value,
        "model": model,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "n_rows": int(len(df)),
    }


def main() -> None:
    """Run phase 3 robustness slices and write outputs."""
    args = parse_args()
    set_seed(int(args.seed))

    universe = pd.read_parquet(Path(args.input_universe).expanduser())[
        [
            "season",
            "date",
            "game_id",
            "player_normalized",
            "market_consensus_line",
            "home_game",
            "days_since_last_game",
            "player_season_mean_fg3a",
        ]
    ].copy()
    preds = pd.read_csv(Path(args.input_predictions).expanduser())
    comparison = pd.read_csv(Path(args.input_comparison).expanduser())
    universe["game_id"] = universe["game_id"].astype(str)
    preds["game_id"] = preds["game_id"].astype(str)
    top_models = comparison.sort_values(["rmse", "model"]).head(int(args.top_k_models))["model"].tolist()
    preds = preds[preds["model"].isin(top_models)].copy()
    merged = preds.merge(
        universe,
        on=["season", "date", "game_id", "player_normalized", "market_consensus_line"],
        how="left",
    )

    merged["line_bucket"] = merged["market_consensus_line"].apply(line_bucket)
    merged["home_away"] = merged["home_game"].map({1: "home", 0: "away"})
    merged["rest_bucket"] = merged["days_since_last_game"].apply(rest_bucket)
    volume = merged["player_season_mean_fg3a"].fillna(0.0).astype(float)
    q33 = float(volume.quantile(0.33))
    q66 = float(volume.quantile(0.66))
    merged["volume_tier"] = np.where(
        volume <= q33,
        "vol_low",
        np.where(volume <= q66, "vol_mid", "vol_high"),
    )

    rows = []
    segment_defs = {
        "season": "season",
        "line_bucket": "line_bucket",
        "volume_tier": "volume_tier",
        "home_away": "home_away",
        "rest_bucket": "rest_bucket",
    }
    for segment_type, col in segment_defs.items():
        for segment_value, seg_df in merged.groupby(col, dropna=False):
            for model, model_df in seg_df.groupby("model"):
                if len(model_df) < 25:
                    continue
                rows.append(metric_row(model_df, segment_type, str(segment_value), model))
    segment_metrics = pd.DataFrame(rows).sort_values(
        ["segment_type", "segment_value", "rmse", "model"]
    )

    global_rank = comparison.sort_values(["rmse", "model"]).reset_index(drop=True)
    global_rank["global_rank"] = np.arange(1, len(global_rank) + 1)
    rank_lookup = {row["model"]: int(row["global_rank"]) for _, row in global_rank.iterrows()}

    stability_rows = []
    for (segment_type, segment_value), seg_df in segment_metrics.groupby(["segment_type", "segment_value"]):
        ranked = seg_df.sort_values(["rmse", "model"]).reset_index(drop=True)
        ranked["segment_rank"] = np.arange(1, len(ranked) + 1)
        for _, row in ranked.iterrows():
            model = str(row["model"])
            segment_rank = int(row["segment_rank"])
            global_model_rank = int(rank_lookup[model])
            stability_rows.append(
                {
                    "segment_type": segment_type,
                    "segment_value": segment_value,
                    "model": model,
                    "global_rank": global_model_rank,
                    "segment_rank": segment_rank,
                    "rank_shift": int(segment_rank - global_model_rank),
                    "unstable_flag": int(abs(segment_rank - global_model_rank) >= 2),
                    "rmse": float(row["rmse"]),
                    "n_rows": int(row["n_rows"]),
                }
            )
    stability = pd.DataFrame(stability_rows).sort_values(
        ["unstable_flag", "segment_type", "segment_value", "segment_rank"],
        ascending=[False, True, True, True],
    )

    out_segment = resolve_output_path(args.output_segment_csv, "v5_robustness_segment_metrics.csv")
    out_stability = resolve_output_path(args.output_stability_csv, "v5_model_stability_summary.csv")
    segment_metrics.to_csv(out_segment, index=False)
    stability.to_csv(out_stability, index=False)

    print(
        "phase=phase3",
        f"seed={args.seed}",
        f"models={','.join(top_models)}",
        f"segment_metrics={out_segment}",
        f"stability={out_stability}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()

