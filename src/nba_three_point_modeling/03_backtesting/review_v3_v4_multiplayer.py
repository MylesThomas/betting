"""
Review v3 vs v4 multi-player comparison artifacts.

Context:
- Consumes outputs from compare_v3_v4_multiplayer.py and prints a compact
  robustness summary to evaluate whether v4 lift is broad or concentrated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI args for v3/v4 multiplayer review."""
    parser = argparse.ArgumentParser(
        description="Review v3 vs v4 multiplayer artifacts and print robustness diagnostics."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="~/Downloads/tmp/v3_v4_multiplayer_summary.csv",
        help="Summary CSV from compare_v3_v4_multiplayer.py",
    )
    parser.add_argument(
        "--predictions-parquet",
        type=str,
        default="~/Downloads/tmp/v3_v4_multiplayer_predictions.parquet",
        help="Predictions parquet from compare_v3_v4_multiplayer.py",
    )
    parser.add_argument("--worst-n", type=int, default=20, help="Rows to show in worst regression table.")
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, required_cols: list[str], source_name: str) -> None:
    """Fail fast if required columns are missing in a loaded artifact."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source_name}: {missing}")


def _metrics_block(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    """Compute rmse/mae/r2 for one prediction column."""
    y = df["FG3M"].to_numpy(dtype=float)
    p = df[pred_col].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    if len(y) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _print_block(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("(no rows)")
        return
    print(df.to_string(index=False))


def main() -> None:
    """Run review blocks for v3/v4 multiplayer comparison."""
    args = parse_args()
    summary_path = Path(args.summary_csv).expanduser()
    preds_path = Path(args.predictions_parquet).expanduser()
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions parquet: {preds_path}")
    summary = pd.read_csv(summary_path)
    preds = pd.read_parquet(preds_path)
    _require_columns(summary, ["model", "rmse", "mae", "r2"], "summary_csv")
    _require_columns(
        preds,
        [
            "season",
            "date",
            "game_id",
            "player_normalized",
            "FG3M",
            "pred_v3",
            "pred_v4",
            "team_point_spread",
            "player_consensus_prop_line",
        ],
        "predictions_parquet",
    )
    if "team_point_spread_bucket" not in preds.columns:
        preds["team_point_spread_bucket"] = pd.cut(
            preds["team_point_spread"].astype(float),
            bins=[float("-inf"), -12.0, -8.0, -4.0, -1.0, 1.0, 4.0, 8.0, 12.0, float("inf")],
            labels=["(-inf,-12]", "(-12,-8]", "(-8,-4]", "(-4,-1]", "(-1,1]", "(1,4]", "(4,8]", "(8,12]", "(12,inf)"],
            right=True,
        ).astype(str)

    print(f"summary_csv={summary_path}")
    print(f"predictions_parquet={preds_path}")

    _print_block("Overall Metrics", summary.sort_values("rmse"))

    by_season_rows: list[dict] = []
    for season, g in preds.groupby("season", as_index=False):
        m_v3 = _metrics_block(g, "pred_v3")
        m_v4 = _metrics_block(g, "pred_v4")
        by_season_rows.append(
            {
                "season": season,
                "n_rows": int(len(g)),
                "rmse_v3": m_v3["rmse"],
                "rmse_v4": m_v4["rmse"],
                "rmse_delta_v4_minus_v3": m_v4["rmse"] - m_v3["rmse"],
                "mae_v3": m_v3["mae"],
                "mae_v4": m_v4["mae"],
                "mae_delta_v4_minus_v3": m_v4["mae"] - m_v3["mae"],
                "r2_v3": m_v3["r2"],
                "r2_v4": m_v4["r2"],
                "r2_delta_v4_minus_v3": m_v4["r2"] - m_v3["r2"],
            }
        )
    by_season = pd.DataFrame(by_season_rows).sort_values("season")
    _print_block("Per-Season Metrics", by_season)
    if len(by_season) <= 1:
        print(
            "\nWARNING: only one season is present in predictions; "
            "cross-season robustness inference is weak."
        )

    by_bucket_rows: list[dict] = []
    for bucket, g in preds.groupby("team_point_spread_bucket", as_index=False):
        if str(bucket) == "nan":
            continue
        m_v3 = _metrics_block(g, "pred_v3")
        m_v4 = _metrics_block(g, "pred_v4")
        by_bucket_rows.append(
            {
                "team_point_spread_bucket": bucket,
                "n_rows": int(len(g)),
                "rmse_v3": m_v3["rmse"],
                "rmse_v4": m_v4["rmse"],
                "rmse_delta_v4_minus_v3": m_v4["rmse"] - m_v3["rmse"],
                "mae_v3": m_v3["mae"],
                "mae_v4": m_v4["mae"],
                "mae_delta_v4_minus_v3": m_v4["mae"] - m_v3["mae"],
            }
        )
    by_bucket = pd.DataFrame(by_bucket_rows).sort_values("team_point_spread_bucket")
    _print_block("Per-Spread-Bucket Metrics", by_bucket)

    preds = preds.copy()
    preds["abs_err_v3"] = (preds["FG3M"] - preds["pred_v3"]).abs()
    preds["abs_err_v4"] = (preds["FG3M"] - preds["pred_v4"]).abs()
    preds["abs_err_delta_v4_minus_v3"] = preds["abs_err_v4"] - preds["abs_err_v3"]
    worst = preds.sort_values("abs_err_delta_v4_minus_v3", ascending=False).head(int(args.worst_n))
    _print_block(
        "Worst Abs-Error Regressions (v4 worse than v3)",
        worst[
            [
                "season",
                "date",
                "player_normalized",
                "game_id",
                "FG3M",
                "pred_v3",
                "pred_v4",
                "team_point_spread",
                "player_consensus_prop_line",
                "abs_err_v3",
                "abs_err_v4",
                "abs_err_delta_v4_minus_v3",
            ]
        ],
    )

    season_rmse_wins = int((by_season["rmse_delta_v4_minus_v3"] < 0).sum())
    season_mae_wins = int((by_season["mae_delta_v4_minus_v3"] < 0).sum())
    bucket_rmse_wins = int((by_bucket["rmse_delta_v4_minus_v3"] < 0).sum())
    bucket_mae_wins = int((by_bucket["mae_delta_v4_minus_v3"] < 0).sum())
    season_concentrated = season_rmse_wins <= 1
    bucket_concentrated = bucket_rmse_wins <= max(1, int(0.3 * len(by_bucket)))
    stability = pd.DataFrame(
        [
            {
                "slice_type": "season",
                "n_slices": int(len(by_season)),
                "v4_rmse_wins": season_rmse_wins,
                "v4_mae_wins": season_mae_wins,
                "concentrated_flag": int(season_concentrated),
            },
            {
                "slice_type": "spread_bucket",
                "n_slices": int(len(by_bucket)),
                "v4_rmse_wins": bucket_rmse_wins,
                "v4_mae_wins": bucket_mae_wins,
                "concentrated_flag": int(bucket_concentrated),
            },
        ]
    )
    _print_block("Stability Summary", stability)

    overall = summary.set_index("model")
    v3_rmse = float(overall.loc["v3_three_input_regression", "rmse"])
    v4_rmse = float(overall.loc["v4_market_spread_regression", "rmse"])
    v3_mae = float(overall.loc["v3_three_input_regression", "mae"])
    v4_mae = float(overall.loc["v4_market_spread_regression", "mae"])
    recommendation = "promote_v4_candidate" if (v4_rmse < v3_rmse and v4_mae < v3_mae) else "keep_v3_default"
    rationale = (
        "v4 beats v3 on both RMSE and MAE overall"
        if recommendation == "promote_v4_candidate"
        else "v4 does not beat v3 on both RMSE and MAE overall"
    )
    print("\n=== Recommendation ===")
    print(f"recommendation={recommendation}")
    print(f"rationale={rationale}")


if __name__ == "__main__":
    main()
