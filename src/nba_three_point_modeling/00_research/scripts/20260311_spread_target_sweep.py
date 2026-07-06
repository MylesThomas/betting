"""
Run v6 spread-context regression sweeps across player-level targets.

Context:
- Evaluates whether pregame spread adds predictive value on top of baseline
  target-level behavior for player outcomes (MIN, FG3M, FG3A, FG3A_per_min,
  FG3_PCT, plus optional box-score targets).
- Uses deterministic out-of-sample scoring via date-based split and reports:
  RMSE/MAE/R2 gains, linear spread diagnostics, and binned spread effects.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5_workflow_lib import resolve_output_path
from v5_workflow_lib import set_seed


REQUIRED_TARGETS = ["MIN", "FG3M", "FG3A", "FG3A_per_min", "FG3_PCT"]
OPTIONAL_TARGETS = ["PTS", "REB", "AST", "FGA", "FGM", "TOV", "FTA", "FTM"]
NEUTRAL_BIN = "(-1,1]"


def parse_args() -> argparse.Namespace:
    """Parse CLI args for v6 spread target sweep."""
    parser = argparse.ArgumentParser(description="Run v6 spread target/model sweep.")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument(
        "--input-universe",
        type=str,
        default="~/Downloads/tmp/v6_spread_universe.parquet",
    )
    parser.add_argument("--include-optional-targets", type=str, default="true")
    parser.add_argument("--include-incremental-model", type=str, default="true")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--output-summary-csv", type=str, default="")
    parser.add_argument("--output-bin-effects-csv", type=str, default="")
    parser.add_argument("--output-ranked-targets-csv", type=str, default="")
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    """Parse common string boolean variants."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and R2 on aligned vectors."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def split_train_test_by_date(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by sorted unique date to avoid leakage across timeline."""
    unique_dates = sorted(df["date"].unique().tolist())
    test_count = max(1, int(len(unique_dates) * float(test_fraction)))
    train_dates = set(unique_dates[:-test_count])
    test_dates = set(unique_dates[-test_count:])
    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced empty train or test set")
    return train_df, test_df


def fit_ols_with_intercept(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit OLS with intercept and return coefficients."""
    X_design = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X_design, y, rcond=None)[0]


def predict_ols_with_intercept(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Predict from OLS coefficients including intercept."""
    X_design = np.column_stack([np.ones(len(X)), X])
    return X_design @ coef


def linear_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    spread_feature_idx: int,
) -> tuple[float, float, float, float]:
    """Compute spread coefficient, p-value (normal approx), and 95% CI."""
    X_design = np.column_stack([np.ones(len(X)), X])
    y_hat = X_design @ coef
    resid = y - y_hat
    n_obs = X_design.shape[0]
    n_params = X_design.shape[1]
    dof = n_obs - n_params
    if dof <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    sigma2 = float((resid @ resid) / dof)
    xtx_inv = np.linalg.inv(X_design.T @ X_design)
    coef_idx = spread_feature_idx + 1
    spread_coef = float(coef[coef_idx])
    se = float(math.sqrt(sigma2 * xtx_inv[coef_idx, coef_idx]))
    if se == 0.0:
        return spread_coef, float("nan"), float("nan"), float("nan")
    z = spread_coef / se
    p_value = float(math.erfc(abs(z) / math.sqrt(2.0)))
    ci_low = float(spread_coef - 1.96 * se)
    ci_high = float(spread_coef + 1.96 * se)
    return spread_coef, p_value, ci_low, ci_high


def build_binned_design(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build one-hot matrix for spread bins with neutral bin as dropped reference."""
    bins = sorted(train_df["spread_bin"].dropna().unique().tolist())
    bins = [x for x in bins if x != NEUTRAL_BIN]
    if len(bins) == 0:
        X_train = np.zeros((len(train_df), 0), dtype=float)
        X_test = np.zeros((len(test_df), 0), dtype=float)
        return X_train, X_test, bins
    X_train = np.column_stack([(train_df["spread_bin"] == b).astype(float).to_numpy() for b in bins])
    X_test = np.column_stack([(test_df["spread_bin"] == b).astype(float).to_numpy() for b in bins])
    return X_train, X_test, bins


def score_target_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    model_name: str,
) -> dict[str, Any]:
    """Fit one model spec and return test metrics + linear diagnostics."""
    out: dict[str, Any] = {
        "target": target,
        "model": model_name,
        "n_rows": int(len(test_df)),
        "rmse": float("nan"),
        "mae": float("nan"),
        "r2": float("nan"),
        "intercept": float("nan"),
        "coef_market_consensus_line": float("nan"),
        "coef_spread_signed": float("nan"),
        "p_value_spread_signed": float("nan"),
        "ci_low_spread_signed": float("nan"),
        "ci_high_spread_signed": float("nan"),
        "equation": "",
    }
    y_train = train_df[target].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)

    if model_name == "baseline":
        y_pred = np.full(len(test_df), y_train.mean(), dtype=float)
        metrics = compute_metrics(y_true=y_test, y_pred=y_pred)
        out.update(metrics)
        out["intercept"] = float(y_train.mean())
        out["equation"] = f"{target} = {out['intercept']:.6f}"
        return out

    if model_name == "spread_linear":
        X_train = train_df[["spread_signed"]].to_numpy(dtype=float)
        X_test = test_df[["spread_signed"]].to_numpy(dtype=float)
        coef = fit_ols_with_intercept(X=X_train, y=y_train)
        y_pred = predict_ols_with_intercept(X=X_test, coef=coef)
        metrics = compute_metrics(y_true=y_test, y_pred=y_pred)
        spread_coef, p_value, ci_low, ci_high = linear_diagnostics(
            X=X_train, y=y_train, coef=coef, spread_feature_idx=0
        )
        out.update(metrics)
        out["intercept"] = float(coef[0])
        out["coef_spread_signed"] = spread_coef
        out["p_value_spread_signed"] = p_value
        out["ci_low_spread_signed"] = ci_low
        out["ci_high_spread_signed"] = ci_high
        out["equation"] = (
            f"{target} = {out['intercept']:.6f} "
            f"+ ({out['coef_spread_signed']:.6f})*spread_signed"
        )
        return out

    if model_name == "spread_binned":
        X_train, X_test, _ = build_binned_design(train_df=train_df, test_df=test_df)
        coef = fit_ols_with_intercept(X=X_train, y=y_train)
        y_pred = predict_ols_with_intercept(X=X_test, coef=coef)
        metrics = compute_metrics(y_true=y_test, y_pred=y_pred)
        out.update(metrics)
        out["intercept"] = float(coef[0])
        out["equation"] = (
            f"{target} = {out['intercept']:.6f} "
            "+ spread_bin_dummies_vs_neutral"
        )
        return out

    if model_name == "consensus_plus_spread":
        X_train = train_df[["market_consensus_line", "spread_signed"]].to_numpy(dtype=float)
        X_test = test_df[["market_consensus_line", "spread_signed"]].to_numpy(dtype=float)
        coef = fit_ols_with_intercept(X=X_train, y=y_train)
        y_pred = predict_ols_with_intercept(X=X_test, coef=coef)
        metrics = compute_metrics(y_true=y_test, y_pred=y_pred)
        spread_coef, p_value, ci_low, ci_high = linear_diagnostics(
            X=X_train, y=y_train, coef=coef, spread_feature_idx=1
        )
        out.update(metrics)
        out["intercept"] = float(coef[0])
        out["coef_market_consensus_line"] = float(coef[1])
        out["coef_spread_signed"] = spread_coef
        out["p_value_spread_signed"] = p_value
        out["ci_low_spread_signed"] = ci_low
        out["ci_high_spread_signed"] = ci_high
        out["equation"] = (
            f"{target} = {out['intercept']:.6f} "
            f"+ ({out['coef_market_consensus_line']:.6f})*market_consensus_line "
            f"+ ({out['coef_spread_signed']:.6f})*spread_signed"
        )
        return out

    raise ValueError(f"Unsupported model_name: {model_name}")


def build_bin_effects(df: pd.DataFrame, target: str, model_name: str) -> pd.DataFrame:
    """Compute per-bin outcome means and deltas vs neutral bin."""
    grouped = (
        df.groupby("spread_bin", as_index=False)
        .agg(n_rows=(target, "count"), mean_outcome=(target, "mean"))
        .sort_values("spread_bin")
    )
    neutral_rows = grouped[grouped["spread_bin"] == NEUTRAL_BIN]
    neutral_mean = float(neutral_rows["mean_outcome"].iloc[0]) if not neutral_rows.empty else float("nan")
    grouped["neutral_bin_mean"] = neutral_mean
    grouped["delta_vs_neutral"] = grouped["mean_outcome"] - grouped["neutral_bin_mean"]
    grouped["target"] = target
    grouped["model"] = model_name
    return grouped[["target", "model", "spread_bin", "n_rows", "mean_outcome", "neutral_bin_mean", "delta_vs_neutral"]]


def rank_targets(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Rank targets by best spread-model lift vs baseline."""
    candidates = summary_df[summary_df["model"] != "baseline"].copy()
    best = (
        candidates.sort_values(
            ["target", "rmse_gain_vs_baseline", "r2_gain_vs_baseline", "model"],
            ascending=[True, False, False, True],
        )
        .groupby("target", as_index=False)
        .first()
    )
    ranked = best.sort_values(
        ["rmse_gain_vs_baseline", "r2_gain_vs_baseline", "target"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranked["target_rank"] = ranked.index + 1
    cols = [
        "target_rank",
        "target",
        "model",
        "n_rows",
        "rmse",
        "mae",
        "r2",
        "rmse_gain_vs_baseline",
        "mae_gain_vs_baseline",
        "r2_gain_vs_baseline",
    ]
    return ranked[cols]


def main() -> None:
    """Run v6 spread model sweep and save summary/bin/ranking outputs."""
    args = parse_args()
    set_seed(int(args.seed))
    include_optional = parse_bool(args.include_optional_targets)
    include_incremental = parse_bool(args.include_incremental_model)

    universe = pd.read_parquet(Path(args.input_universe).expanduser())
    required_cols = ["date", "spread_signed", "spread_bin", "market_consensus_line"] + REQUIRED_TARGETS
    for col in required_cols:
        if col not in universe.columns:
            raise ValueError(f"Required column missing from universe: {col}")

    targets = REQUIRED_TARGETS + (OPTIONAL_TARGETS if include_optional else [])
    models = ["baseline", "spread_linear", "spread_binned"]
    if include_incremental:
        models.append("consensus_plus_spread")

    summary_rows: list[dict[str, Any]] = []
    bin_frames: list[pd.DataFrame] = []
    for target in targets:
        if target not in universe.columns:
            continue
        target_df = universe.dropna(subset=[target, "spread_signed", "spread_bin", "market_consensus_line"]).copy()
        if len(target_df) < 100:
            continue
        train_df, test_df = split_train_test_by_date(df=target_df, test_fraction=args.test_fraction)

        baseline_row = score_target_model(
            train_df=train_df,
            test_df=test_df,
            target=target,
            model_name="baseline",
        )
        summary_rows.append(baseline_row)
        for model_name in models:
            if model_name == "baseline":
                continue
            row = score_target_model(
                train_df=train_df,
                test_df=test_df,
                target=target,
                model_name=model_name,
            )
            row["rmse_gain_vs_baseline"] = float(baseline_row["rmse"] - row["rmse"])
            row["mae_gain_vs_baseline"] = float(baseline_row["mae"] - row["mae"])
            row["r2_gain_vs_baseline"] = float(row["r2"] - baseline_row["r2"])
            summary_rows.append(row)
            if model_name == "spread_binned":
                bin_frames.append(build_bin_effects(df=test_df, target=target, model_name=model_name))

        summary_rows[-len(models)]["rmse_gain_vs_baseline"] = 0.0
        summary_rows[-len(models)]["mae_gain_vs_baseline"] = 0.0
        summary_rows[-len(models)]["r2_gain_vs_baseline"] = 0.0

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["target", "model"]).reset_index(drop=True)
    bin_effects_df = pd.concat(bin_frames, ignore_index=True) if len(bin_frames) > 0 else pd.DataFrame()
    ranked_df = rank_targets(summary_df=summary_df)

    summary_path = resolve_output_path(args.output_summary_csv, "v6_spread_model_summary.csv")
    bin_path = resolve_output_path(args.output_bin_effects_csv, "v6_spread_bin_effects.csv")
    ranked_path = resolve_output_path(args.output_ranked_targets_csv, "v6_spread_ranked_targets.csv")
    summary_df.to_csv(summary_path, index=False)
    bin_effects_df.to_csv(bin_path, index=False)
    ranked_df.to_csv(ranked_path, index=False)
    print(
        "phase=v6_spread_target_sweep",
        f"seed={args.seed}",
        f"rows={len(universe)}",
        f"targets_scored={summary_df['target'].nunique()}",
        f"summary={summary_path}",
        f"bin_effects={bin_path}",
        f"ranked={ranked_path}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()
