"""
Phase 2 recomposition study for v5 3PM workflow.

Context:
- Compare direct FG3M baselines versus decomposition families:
  A) predict FG3A and FG3_PCT, multiply
  B) predict MIN, FG3A_per_min, FG3_PCT, multiply
- Operates on the same canonical phase-0 universe for fair model comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5_workflow_lib import compute_metrics
from v5_workflow_lib import fit_predict
from v5_workflow_lib import ModelSpec
from v5_workflow_lib import resolve_output_path
from v5_workflow_lib import set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI args for phase 2."""
    parser = argparse.ArgumentParser(description="Run v5 FG3M recomposition comparison.")
    parser.add_argument("--phase", type=str, default="phase2")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument(
        "--input-universe",
        "--input-universe-path",
        dest="input_universe",
        type=str,
        default="~/Downloads/tmp/v5_eval_universe.parquet",
    )
    parser.add_argument("--min-model-csv", type=str, default="~/Downloads/tmp/v5_min_models.csv")
    parser.add_argument(
        "--fga-per-min-model-csv", type=str, default="~/Downloads/tmp/v5_fga_per_min_models.csv"
    )
    parser.add_argument("--fg3-pct-model-csv", type=str, default="~/Downloads/tmp/v5_fg3_pct_models.csv")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--output-predictions-csv", type=str, default="")
    parser.add_argument("--output-outliers-csv", type=str, default="")
    parser.add_argument("--output-memo-md", type=str, default="")
    return parser.parse_args()


def parse_spec_row(row: pd.Series, target_col: str, clip_low: float | None, clip_high: float | None) -> ModelSpec:
    """Recreate ModelSpec from phase-1 row."""
    fit_type = str(row["fit_type"])
    if fit_type == "ols":
        features = [x for x in str(row["features"]).split(",") if x != ""]
    else:
        features = []
    return ModelSpec(
        name=str(row["model"]),
        feature_cols=features,
        fit_type=fit_type,
        target_col=target_col,
        clip_low=clip_low,
        clip_high=clip_high,
    )


def top_row(path: str) -> pd.Series:
    """Read model CSV and return top-ranked row by rmse then model."""
    df = pd.read_csv(Path(path).expanduser())
    return df.sort_values(["rmse", "model"]).iloc[0]


def score_fg3m_predictions(
    df: pd.DataFrame,
    model_name: str,
    prediction: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score one FG3M prediction vector and return comparison row + per-row frame."""
    scored = df.copy()
    scored["prediction"] = np.clip(prediction.astype(float), 0.0, None)
    scored = scored[np.isfinite(scored["prediction"])].copy()
    y_true = scored["FG3M"].to_numpy(dtype=float)
    y_pred = scored["prediction"].to_numpy(dtype=float)
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    row = {
        "model": model_name,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "n_total_rows": int(len(scored)),
        "residual_abs_p90": float(np.quantile(np.abs(y_true - y_pred), 0.90)),
        "residual_abs_p95": float(np.quantile(np.abs(y_true - y_pred), 0.95)),
    }
    pred_df = scored[
        [
            "season",
            "date",
            "game_id",
            "matchup",
            "player_normalized",
            "FG3M",
            "market_consensus_line",
        ]
    ].copy()
    pred_df["model"] = model_name
    pred_df["prediction_fg3m"] = y_pred
    pred_df["actual_fg3m"] = y_true
    pred_df["residual"] = pred_df["actual_fg3m"] - pred_df["prediction_fg3m"]
    pred_df["abs_residual"] = pred_df["residual"].abs()
    return row, pred_df


def build_memo(
    best_min_model: str,
    best_fga_per_min_model: str,
    best_fg3_pct_model: str,
    best_fg3m_model: str,
    comparison_df: pd.DataFrame,
) -> str:
    """Create concise recommendation memo text."""
    baseline_rmse = float(comparison_df.loc[comparison_df["model"] == "direct_baseline", "rmse"].iloc[0])
    best_rmse = float(comparison_df.loc[comparison_df["model"] == best_fg3m_model, "rmse"].iloc[0])
    gain = baseline_rmse - best_rmse
    return "\n".join(
        [
            "# v5 3PM Decomposition Recommendation Memo",
            "",
            "## Best component models",
            f"- MIN: `{best_min_model}`",
            f"- FG3A_per_min: `{best_fga_per_min_model}`",
            f"- FG3_PCT: `{best_fg3_pct_model}`",
            "",
            "## Best FG3M model",
            f"- Winner: `{best_fg3m_model}`",
            f"- RMSE gain vs direct baseline: `{gain:.4f}`",
            "",
            "## Caveats",
            "- Current study is in-sample scoring on the market-eligible universe (research mode).",
            "- Re-run in walk-forward mode before production deployment decisions.",
            "- Monitor outlier rows where absolute residual remains in the top tail.",
            "",
            "## Next production-safe steps",
            "- Freeze the universe cut date and run strict temporal train/validation splits.",
            "- Add model monitoring for line bucket and player volume segments.",
            "- Gate deployment on calibration and edge-bucket stability checks.",
            "",
        ]
    )


def main() -> None:
    """Run phase 2 recomposition experiments and save outputs."""
    args = parse_args()
    set_seed(int(args.seed))
    df = pd.read_parquet(Path(args.input_universe).expanduser())
    df = df.sort_values(["season", "date", "player_normalized", "game_id"]).reset_index(drop=True)

    min_top = top_row(args.min_model_csv)
    fga_per_min_top = top_row(args.fga_per_min_model_csv)
    fg3_pct_top = top_row(args.fg3_pct_model_csv)

    min_spec = parse_spec_row(min_top, target_col="MIN", clip_low=0.0, clip_high=None)
    fga_per_min_spec = parse_spec_row(
        fga_per_min_top, target_col="FG3A_per_min", clip_low=0.0, clip_high=None
    )
    fg3_pct_spec = parse_spec_row(fg3_pct_top, target_col="FG3_PCT", clip_low=0.0, clip_high=1.0)

    direct_baseline_spec = ModelSpec(
        name="direct_baseline",
        feature_cols=[],
        fit_type="baseline_player_season_mean",
        target_col="FG3M",
        clip_low=0.0,
        clip_high=None,
    )
    direct_v4_core = ModelSpec(
        name="direct_v4_core",
        feature_cols=[
            "MIN",
            "FG3A",
            "FG3_PCT",
            "roll_fg3m_5",
            "roll_fg3a_5",
            "roll_fg3_pct_5",
            "market_consensus_line",
        ],
        fit_type="ols",
        target_col="FG3M",
        clip_low=0.0,
        clip_high=None,
    )
    direct_v4_box = ModelSpec(
        name="direct_v4_box",
        feature_cols=[
            "MIN",
            "FG3A",
            "FG3_PCT",
            "PTS",
            "AST",
            "REB",
            "FGA",
            "home_game",
            "is_back_to_back",
        ],
        fit_type="ols",
        target_col="FG3M",
        clip_low=0.0,
        clip_high=None,
    )

    pred_min = fit_predict(train_df=df, score_df=df, spec=min_spec)
    pred_fga_per_min = fit_predict(train_df=df, score_df=df, spec=fga_per_min_spec)
    pred_fg3_pct = fit_predict(train_df=df, score_df=df, spec=fg3_pct_spec)
    pred_fga_direct = fit_predict(
        train_df=df,
        score_df=df,
        spec=ModelSpec(
            name="fga_direct",
            feature_cols=["MIN", "FGA", "roll_fg3a_5", "player_season_mean_fg3a"],
            fit_type="ols",
            target_col="FG3A",
            clip_low=0.0,
            clip_high=None,
        ),
    )

    # Clamp decomposed components to observed high-end empirical ranges.
    min_cap = float(df["MIN"].quantile(0.995))
    fga_per_min_cap = float(df["FG3A_per_min"].dropna().quantile(0.995))
    fg3_pct_cap = float(df["FG3_PCT"].quantile(0.995))
    fga_cap = float(df["FG3A"].quantile(0.995))
    pred_min = np.clip(pred_min, 0.0, min_cap)
    pred_fga_per_min = np.clip(pred_fga_per_min, 0.0, fga_per_min_cap)
    pred_fg3_pct = np.clip(pred_fg3_pct, 0.0, fg3_pct_cap)
    pred_fga_direct = np.clip(pred_fga_direct, 0.0, fga_cap)

    model_predictions: dict[str, np.ndarray] = {
        "direct_baseline": fit_predict(train_df=df, score_df=df, spec=direct_baseline_spec),
        "direct_v4_core": fit_predict(train_df=df, score_df=df, spec=direct_v4_core),
        "direct_v4_box": fit_predict(train_df=df, score_df=df, spec=direct_v4_box),
        "recomp_a_fga_x_pct": np.clip(pred_fga_direct, 0.0, None) * np.clip(pred_fg3_pct, 0.0, 1.0),
        "recomp_b_min_x_fga_per_min_x_pct": np.clip(pred_min, 0.0, None)
        * np.clip(pred_fga_per_min, 0.0, None)
        * np.clip(pred_fg3_pct, 0.0, 1.0),
        "recomp_b_market_blend_30": 0.70
        * (
            np.clip(pred_min, 0.0, None)
            * np.clip(pred_fga_per_min, 0.0, None)
            * np.clip(pred_fg3_pct, 0.0, 1.0)
        )
        + 0.30 * df["market_consensus_line"].to_numpy(dtype=float),
    }

    common_valid = np.logical_and.reduce(
        [np.isfinite(arr) for arr in model_predictions.values()]
    )
    common_df = df.loc[common_valid].copy().reset_index(drop=True)
    for model_name in list(model_predictions.keys()):
        model_predictions[model_name] = model_predictions[model_name][common_valid]

    comparison_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    for model_name in sorted(model_predictions.keys()):
        row, pred_df = score_fg3m_predictions(
            df=common_df,
            model_name=model_name,
            prediction=model_predictions[model_name],
        )
        comparison_rows.append(row)
        pred_frames.append(pred_df)

    comparison = pd.DataFrame(comparison_rows).sort_values(["rmse", "model"]).reset_index(drop=True)
    baseline_rmse = float(comparison.loc[comparison["model"] == "direct_baseline", "rmse"].iloc[0])
    baseline_mae = float(comparison.loc[comparison["model"] == "direct_baseline", "mae"].iloc[0])
    comparison["rmse_gain_vs_baseline"] = baseline_rmse - comparison["rmse"]
    comparison["mae_gain_vs_baseline"] = baseline_mae - comparison["mae"]

    predictions = pd.concat(pred_frames, ignore_index=True).sort_values(
        ["abs_residual", "model"], ascending=[False, True]
    )
    outliers = predictions.head(250).copy()

    out_comparison = resolve_output_path(args.output_csv, "v5_fg3m_recompose_comparison.csv")
    out_predictions = resolve_output_path(
        args.output_predictions_csv, "v5_fg3m_recompose_predictions.csv"
    )
    out_outliers = resolve_output_path(args.output_outliers_csv, "v5_fg3m_recompose_outliers.csv")
    out_memo = resolve_output_path(args.output_memo_md, "v5_recommendation_memo.md")

    comparison.to_csv(out_comparison, index=False)
    predictions.to_csv(out_predictions, index=False)
    outliers.to_csv(out_outliers, index=False)

    best_model = comparison.iloc[0]["model"]
    memo_text = build_memo(
        best_min_model=str(min_top["model"]),
        best_fga_per_min_model=str(fga_per_min_top["model"]),
        best_fg3_pct_model=str(fg3_pct_top["model"]),
        best_fg3m_model=str(best_model),
        comparison_df=comparison,
    )
    Path(out_memo).write_text(memo_text)

    print(
        "phase=phase2",
        f"seed={args.seed}",
        f"rows={len(df)}",
        f"best_fg3m_model={best_model}",
        f"comparison={out_comparison}",
        f"predictions={out_predictions}",
        f"outliers={out_outliers}",
        f"memo={out_memo}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()

