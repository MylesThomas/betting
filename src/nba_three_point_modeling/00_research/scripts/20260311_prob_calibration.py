"""
Phase 4 calibration and betting-utility analysis for v5 workflow.

Context:
- Convert FG3M point predictions to over/under probabilities using a simple
  Poisson assumption in research mode.
- Compare model probabilities to market-implied probabilities and evaluate:
  Brier score, calibration bins, and edge bucket realized outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v5_workflow_lib import poisson_tail_prob
from v5_workflow_lib import resolve_output_path
from v5_workflow_lib import set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI args for phase 4."""
    parser = argparse.ArgumentParser(description="Run v5 probability calibration analysis.")
    parser.add_argument("--phase", type=str, default="phase4")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument(
        "--input-predictions", type=str, default="~/Downloads/tmp/v5_fg3m_recompose_predictions.csv"
    )
    parser.add_argument("--input-comparison", type=str, default="~/Downloads/tmp/v5_fg3m_recompose_comparison.csv")
    parser.add_argument(
        "--input-universe",
        "--input-universe-path",
        dest="input_universe",
        type=str,
        default="~/Downloads/tmp/v5_eval_universe.parquet",
    )
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--output-calibration-csv", type=str, default="")
    parser.add_argument("--output-edge-csv", type=str, default="")
    return parser.parse_args()


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Compute Brier score for binary outcomes."""
    return float(np.mean((p_pred - y_true) ** 2))


def main() -> None:
    """Run phase 4 and write calibration + edge-bucket outputs."""
    args = parse_args()
    set_seed(int(args.seed))
    preds = pd.read_csv(Path(args.input_predictions).expanduser())
    comparison = pd.read_csv(Path(args.input_comparison).expanduser())
    universe = pd.read_parquet(Path(args.input_universe).expanduser())[
        ["season", "date", "game_id", "player_normalized", "market_consensus_line", "median_p_over_novig"]
    ].copy()
    universe["game_id"] = universe["game_id"].astype(str)
    preds["game_id"] = preds["game_id"].astype(str)

    best_model = (
        args.model_name.strip()
        if args.model_name.strip() != ""
        else str(comparison.sort_values(["rmse", "model"]).iloc[0]["model"])
    )
    model_preds = preds[preds["model"] == best_model].copy()
    merged = model_preds.merge(
        universe,
        on=["season", "date", "game_id", "player_normalized", "market_consensus_line"],
        how="inner",
        suffixes=("_pred", "_u"),
    )
    merged["actual_over"] = (
        merged["actual_fg3m"].astype(float) > merged["market_consensus_line"].astype(float)
    ).astype(int)
    merged["k_threshold"] = np.floor(merged["market_consensus_line"].astype(float)).astype(int) + 1
    merged["p_over_model"] = merged.apply(
        lambda row: poisson_tail_prob(int(row["k_threshold"]), float(row["prediction_fg3m"])),
        axis=1,
    )
    merged["p_over_market"] = merged["median_p_over_novig"].fillna(0.5).astype(float)
    merged["edge_model_vs_market"] = merged["p_over_model"] - merged["p_over_market"]

    merged["prob_bin"] = pd.cut(
        merged["p_over_model"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
    )
    calibration = (
        merged.groupby("prob_bin", as_index=False)
        .agg(
            n=("actual_over", "count"),
            p_model_mean=("p_over_model", "mean"),
            p_market_mean=("p_over_market", "mean"),
            realized_over_rate=("actual_over", "mean"),
        )
        .sort_values("prob_bin")
        .reset_index(drop=True)
    )
    calibration["calibration_gap_abs"] = (
        calibration["p_model_mean"] - calibration["realized_over_rate"]
    ).abs()
    calibration["model_brier"] = brier_score(
        merged["actual_over"].to_numpy(dtype=float),
        merged["p_over_model"].to_numpy(dtype=float),
    )
    calibration["market_brier"] = brier_score(
        merged["actual_over"].to_numpy(dtype=float),
        merged["p_over_market"].to_numpy(dtype=float),
    )
    calibration["model"] = best_model

    merged["edge_bucket"] = pd.cut(
        merged["edge_model_vs_market"],
        bins=[-1.0, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 1.0],
        include_lowest=True,
        labels=[
            "edge_lt_-0_10",
            "edge_-0_10_-0_05",
            "edge_-0_05_-0_02",
            "edge_neutral",
            "edge_0_02_0_05",
            "edge_0_05_0_10",
            "edge_gt_0_10",
        ],
    )
    edge_eval = (
        merged.groupby("edge_bucket", as_index=False)
        .agg(
            n=("actual_over", "count"),
            mean_edge=("edge_model_vs_market", "mean"),
            p_model_mean=("p_over_model", "mean"),
            p_market_mean=("p_over_market", "mean"),
            realized_over_rate=("actual_over", "mean"),
        )
        .sort_values("edge_bucket")
        .reset_index(drop=True)
    )
    edge_eval["model"] = best_model

    out_cal = resolve_output_path(args.output_calibration_csv, "v5_prob_calibration.csv")
    out_edge = resolve_output_path(args.output_edge_csv, "v5_edge_bucket_eval.csv")
    calibration.to_csv(out_cal, index=False)
    edge_eval.to_csv(out_edge, index=False)

    print(
        "phase=phase4",
        f"seed={args.seed}",
        f"model={best_model}",
        f"rows={len(merged)}",
        f"calibration={out_cal}",
        f"edge={out_edge}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()

