"""
Compare v3 vs v4 FG3M mean models on multi-player universe.

Context:
- v3 (original): three-input regression using lagged player form features
  [mean_3pm, predicted_3pa, predicted_minutes].
- v4 (new): v3 features + spread context and player market context
  [team_point_spread, player_consensus_prop_line].
- This script runs on the full v6 universe (not a single-player Curry slice)
  and writes deterministic OOS comparison artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI args for v3 vs v4 multi-player comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare v3 and v4 FG3M models on a multi-player universe parquet "
            "using deterministic date-split OOS evaluation."
        )
    )
    parser.add_argument(
        "--input-universe",
        type=str,
        default="~/Downloads/tmp/v6_spread_universe.parquet",
        help="Parquet with FG3M plus canonical context columns.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of trailing unique dates used for test set.",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=str,
        default="~/Downloads/tmp/v3_v4_multiplayer_summary.csv",
        help="Output CSV path for overall metrics by model.",
    )
    parser.add_argument(
        "--output-predictions-parquet",
        type=str,
        default="~/Downloads/tmp/v3_v4_multiplayer_predictions.parquet",
        help="Output parquet path for row-level predictions and features.",
    )
    parser.add_argument(
        "--output-manifest-json",
        type=str,
        default="~/Downloads/tmp/v3_v4_multiplayer_manifest.json",
        help="Output JSON sidecar with input metadata and feature lists.",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Fail fast if required columns are missing."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _add_v3_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged per-player base features for v3/v4 models."""
    out = df.copy().sort_values(["player_normalized", "date", "game_id"]).reset_index(drop=True)
    grouped = out.groupby("player_normalized", group_keys=False)
    out["mean_3pm"] = grouped["FG3M"].transform(lambda s: s.astype(float).expanding(min_periods=1).mean().shift(1))
    out["predicted_3pa"] = grouped["FG3A"].transform(lambda s: s.astype(float).expanding(min_periods=1).mean().shift(1))
    out["predicted_minutes"] = grouped["MIN"].transform(lambda s: s.astype(float).expanding(min_periods=1).mean().shift(1))
    for col, source in [("mean_3pm", "FG3M"), ("predicted_3pa", "FG3A"), ("predicted_minutes", "MIN")]:
        out[col] = out[col].fillna(out[source].astype(float).mean())
    return out


def _split_by_date(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Date split to avoid leakage."""
    unique_dates = sorted(df["date"].unique().tolist())
    n_test = max(1, int(len(unique_dates) * float(test_fraction)))
    train_dates = set(unique_dates[:-n_test])
    test_dates = set(unique_dates[-n_test:])
    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced empty train or test set")
    return train_df, test_df


def _fit_predict_ols(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Fit OLS on train and predict on test."""
    train = train_df.dropna(subset=["FG3M"] + feature_cols).copy()
    test = test_df.dropna(subset=feature_cols).copy()
    x_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train["FG3M"].to_numpy(dtype=float)
    X_train = np.column_stack([np.ones(len(train)), x_train])
    coefs = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

    x_test = test[feature_cols].to_numpy(dtype=float)
    X_test = np.column_stack([np.ones(len(test)), x_test])
    y_pred_partial = np.clip(X_test @ coefs, 0.0, None)

    full = test_df[["FG3M"]].copy()
    full["pred"] = np.nan
    full.loc[test.index, "pred"] = y_pred_partial
    return full["pred"].to_numpy(dtype=float)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute rmse/mae/r2."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main() -> None:
    """Run v3 vs v4 comparison and save summary + predictions artifacts."""
    args = parse_args()
    input_path = Path(args.input_universe).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input universe parquet: {input_path}")
    if not 0.0 < float(args.test_fraction) < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1 (exclusive)")
    df = pd.read_parquet(input_path)
    _require_columns(
        df=df,
        required_cols=[
            "season",
            "date",
            "game_id",
            "player_normalized",
            "FG3M",
            "FG3A",
            "MIN",
            "team_point_spread",
            "player_consensus_prop_line",
        ],
    )
    df = _add_v3_base_features(df=df)
    df = df.dropna(subset=["FG3M"]).copy()
    train_df, test_df = _split_by_date(df=df, test_fraction=args.test_fraction)

    baseline_pred = np.full(len(test_df), train_df["FG3M"].astype(float).mean(), dtype=float)
    v3_features = ["mean_3pm", "predicted_3pa", "predicted_minutes"]
    v4_features = v3_features + ["player_consensus_prop_line", "team_point_spread"]
    v3_pred = _fit_predict_ols(train_df=train_df, test_df=test_df, feature_cols=v3_features)
    v4_pred = _fit_predict_ols(train_df=train_df, test_df=test_df, feature_cols=v4_features)

    y_true = test_df["FG3M"].to_numpy(dtype=float)
    rows = []
    for model_name, pred in [("baseline", baseline_pred), ("v3_three_input_regression", v3_pred), ("v4_market_spread_regression", v4_pred)]:
        valid = np.isfinite(pred)
        m = _metrics(y_true=y_true[valid], y_pred=pred[valid])
        rows.append(
            {
                "model": model_name,
                "n_rows": int(valid.sum()),
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
            }
        )
    summary = pd.DataFrame(rows)
    baseline = summary[summary["model"] == "baseline"].iloc[0]
    summary["rmse_gain_vs_baseline"] = float(baseline["rmse"]) - summary["rmse"]
    summary["mae_gain_vs_baseline"] = float(baseline["mae"]) - summary["mae"]
    summary["r2_gain_vs_baseline"] = summary["r2"] - float(baseline["r2"])
    summary = summary.sort_values("rmse").reset_index(drop=True)

    pred_out = test_df[
        [
            "season",
            "date",
            "game_id",
            "player_normalized",
            "FG3M",
            "mean_3pm",
            "predicted_3pa",
            "predicted_minutes",
            "player_consensus_prop_line",
            "team_point_spread",
        ]
    ].copy()
    pred_out["pred_baseline"] = baseline_pred
    pred_out["pred_v3"] = v3_pred
    pred_out["pred_v4"] = v4_pred

    summary_path = Path(args.output_summary_csv).expanduser()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    pred_path = Path(args.output_predictions_parquet).expanduser()
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_out.to_parquet(pred_path, index=False)
    manifest_path = Path(args.output_manifest_json).expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_universe_path": str(input_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "model_features": {
            "v3_three_input_regression": v3_features,
            "v4_market_spread_regression": v4_features,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        "phase=compare_v3_v4_multiplayer",
        f"rows_total={len(df)}",
        f"rows_test={len(test_df)}",
        f"summary={summary_path}",
        f"predictions={pred_path}",
        f"manifest={manifest_path}",
        sep=" | ",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
