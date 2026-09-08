"""
Evaluate the logistic calibration layer's predicted probabilities — MLB Total Bases v2.

Added: 2026-07-19

Bins model-predicted P(under | line) into 0.05-wide buckets and measures the actual
under rate in each bucket across all historical seasons.

Calibration thresholds are read from config.yaml (model.calibration.evaluation):
  max_calibration_error: 0.03  — ✓ if |error| ≤ 3pp, ✗ if > 3pp
  min_bin_n:             100   — ~ (thin) if n < 100, no judgment applied

Modes (--mode):
  calibration  Reliability of p_model_under across bins — aggregate, by year, by month,
               and a tension slice (p_under > 0.60 where y_hat > line).

Usage:
  python src/mlb_total_bases_modeling/scripts/evaluate_logistic_predicted_probs.py --mode calibration
  python src/mlb_total_bases_modeling/scripts/evaluate_logistic_predicted_probs.py --mode calibration --line 0.5
"""
from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
import yaml

REPO_ROOT   = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET = "the-odds-api-mt"
SPINE_KEY = "mlb/total_bases_model/regression/mlb_tb_reg_spine.parquet"
MODEL_KEY = "mlb/total_bases_model/model/mlb_tb_regression_v2.joblib"

SUPPORTED_MODES = ["calibration"]

BIN_WIDTH     = 0.05
DISPLAY_MIN_N = 20   # hide bins with fewer than this many observations entirely


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_data(line: float) -> tuple[pd.DataFrame, dict]:
    s3 = boto3.client("s3")

    print("Loading regression spine from S3 ...")
    body  = s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()
    spine = pd.read_parquet(BytesIO(body))
    print(f"  {len(spine):,} total rows | seasons: {sorted(spine['season'].unique())}")

    spine = spine[spine["line"] == line].copy()
    print(f"  {len(spine):,} rows for line={line}")

    print("Loading model bundle from S3 ...")
    body   = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)["Body"].read()
    bundle = joblib.load(BytesIO(body))
    print(f"  Model: {bundle.get('combo_name','?')} | calib lines: {list(bundle.get('calib_models', {}).keys())}")
    return spine, bundle


def score(df: pd.DataFrame, bundle: dict, line: float) -> pd.DataFrame:
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    features     = bundle["features_numeric"]
    calib_models = bundle.get("calib_models", {})

    if line not in calib_models:
        raise RuntimeError(f"No calibration model for line={line}. Available: {list(calib_models.keys())}")

    line_range = (
        df.groupby(["name_norm", "game_date"])["line"]
        .agg(min_line="min", max_line="max")
        .reset_index()
    )
    df = df.drop(columns=["min_line", "max_line"], errors="ignore").merge(
        line_range, on=["name_norm", "game_date"], how="left"
    )

    unique_pg = df[["name_norm", "game_date"] + features].drop_duplicates(
        subset=["name_norm", "game_date"]
    ).dropna(subset=features)

    X     = unique_pg[features].values.astype(float)
    X_sc  = scaler.transform(X)
    y_hat = model.predict(X_sc).astype(float)
    unique_pg = unique_pg[["name_norm", "game_date"]].copy()
    unique_pg["y_hat"] = y_hat

    calib = calib_models[line]
    p_over = calib.predict_proba(y_hat.reshape(-1, 1))[:, 1]
    unique_pg["p_model_over"]  = np.clip(p_over, 0.01, 0.99)
    unique_pg["p_model_under"] = 1.0 - unique_pg["p_model_over"]

    scored = df.merge(unique_pg, on=["name_norm", "game_date"], how="inner")
    scored["actual_under"]   = (scored["total_bases"] < line).astype(int)
    scored["raw_prob_under"] = 1.0 / scored["under_price"]
    scored["edge_under"]     = scored["p_model_under"] - scored["raw_prob_under"]

    print(f"  Scored {len(scored):,} rows | overall actual under rate: {scored['actual_under'].mean():.1%}")
    return scored


def assign_bin(p: pd.Series) -> pd.Series:
    edges  = np.arange(0.0, 1.0 + BIN_WIDTH, BIN_WIDTH)
    labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f})" for i in range(len(edges) - 1)]
    return pd.cut(p, bins=edges, labels=labels, right=False, include_lowest=True)


def add_flag(df: pd.DataFrame, min_bin_n: int, max_calib_error: float) -> pd.DataFrame:
    def _flag(row) -> str:
        if row["n"] < min_bin_n:
            return "~"
        return "✓" if abs(row["calib_error_pp"]) <= max_calib_error * 100 else "✗"
    df["flag"] = df.apply(_flag, axis=1)
    return df


def calibration_table(
    df: pd.DataFrame,
    min_bin_n: int,
    max_calib_error: float,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df["bin"] = assign_bin(df["p_model_under"])
    by = (group_cols or []) + ["bin"]

    agg = (
        df.groupby(by, observed=True)
        .agg(
            n               = ("actual_under",   "count"),
            actual_pct      = ("actual_under",   "mean"),
            model_pct       = ("p_model_under",  "mean"),
            avg_y_hat       = ("y_hat",           "mean"),
            yhat_over_pct   = ("y_hat",           lambda s: (s > df.loc[s.index, "line"]).mean()),
            raw_prob_pct    = ("raw_prob_under",  "mean"),
            edge_pp         = ("edge_under",      "mean"),
        )
        .reset_index()
    )

    # Convert rates → percentage points for 2dp readability
    for col in ["actual_pct", "model_pct", "yhat_over_pct", "raw_prob_pct"]:
        agg[col] = agg[col] * 100
    agg["edge_pp"]        = agg["edge_pp"] * 100
    agg["calib_error_pp"] = agg["actual_pct"] - agg["model_pct"]

    agg = agg[agg["n"] >= DISPLAY_MIN_N].copy()
    agg = add_flag(agg, min_bin_n, max_calib_error)
    return agg


def print_table(df: pd.DataFrame, title: str, line: float) -> None:
    print(f"\n{'─'*100}")
    print(f"  {title}  (line={line})")
    print(f"  Columns in %: actual_pct, model_pct, yhat_over_pct, raw_prob_pct, edge_pp, calib_error_pp")
    print(f"  flag: ✓ = |error| ≤ 3pp · ✗ = |error| > 3pp · ~ = thin (n < judgment threshold)")
    print(f"{'─'*100}")
    pd.options.display.float_format = "{:.2f}".format
    print(df.to_string(index=False))


def tension_slice(
    df: pd.DataFrame,
    threshold: float,
    line: float,
    min_bin_n: int,
    max_calib_error: float,
) -> pd.DataFrame:
    mask = (df["p_model_under"] > threshold) & (df["y_hat"] > line)
    sub  = df[mask].copy()
    if sub.empty:
        return sub
    sub["bin"] = assign_bin(sub["p_model_under"])
    agg = (
        sub.groupby("bin", observed=True)
        .agg(
            n             = ("actual_under",   "count"),
            actual_pct    = ("actual_under",   "mean"),
            model_pct     = ("p_model_under",  "mean"),
            avg_y_hat     = ("y_hat",           "mean"),
            raw_prob_pct  = ("raw_prob_under",  "mean"),
            edge_pp       = ("edge_under",      "mean"),
        )
        .reset_index()
    )
    for col in ["actual_pct", "model_pct", "raw_prob_pct"]:
        agg[col] = agg[col] * 100
    agg["edge_pp"]        = agg["edge_pp"] * 100
    agg["calib_error_pp"] = agg["actual_pct"] - agg["model_pct"]
    agg = agg[agg["n"] >= DISPLAY_MIN_N].copy()
    agg = add_flag(agg, min_bin_n, max_calib_error)
    return agg


def run_calibration(
    scored: pd.DataFrame,
    line: float,
    min_bin_n: int,
    max_calib_error: float,
) -> None:
    # ── 1. Aggregate ──────────────────────────────────────────────────────────
    print_table(
        calibration_table(scored, min_bin_n, max_calib_error),
        "Aggregate calibration — all seasons",
        line,
    )

    # ── 2. By year ────────────────────────────────────────────────────────────
    scored["year"] = pd.to_datetime(scored["game_date"]).dt.year
    print_table(
        calibration_table(scored, min_bin_n, max_calib_error, group_cols=["year"]),
        "Calibration by year",
        line,
    )

    # ── 3. By month ───────────────────────────────────────────────────────────
    scored["month"] = pd.to_datetime(scored["game_date"]).dt.month
    print_table(
        calibration_table(scored, min_bin_n, max_calib_error, group_cols=["month"]),
        "Calibration by month",
        line,
    )

    # ── 4. Tension slice ──────────────────────────────────────────────────────
    print(f"\n{'─'*100}")
    print(f"  TENSION SLICE: p_model_under > 0.60 AND y_hat > {line}  (line={line})")
    print(f"  (Calibration says high UNDER confidence, but regression still projects above the line)")
    print(f"{'─'*100}")
    ts = tension_slice(scored, threshold=0.60, line=line, min_bin_n=min_bin_n, max_calib_error=max_calib_error)
    n_tension   = int(((scored["p_model_under"] > 0.60) & (scored["y_hat"] > line)).sum())
    pct_tension = n_tension / max(1, (scored["p_model_under"] > 0.60).sum())
    print(f"  {n_tension:,} rows in tension slice | {pct_tension:.1%} of all p_under>0.60 rows")
    if not ts.empty:
        print(ts.to_string(index=False))
    else:
        print("  (No bins with sufficient observations)")

    # ── 5. Tension slice by year ──────────────────────────────────────────────
    print(f"\n{'─'*100}")
    print(f"  TENSION SLICE by year: p_model_under > 0.60 AND y_hat > {line}")
    print(f"{'─'*100}")
    sub_tension = scored[(scored["p_model_under"] > 0.60) & (scored["y_hat"] > line)].copy()
    if not sub_tension.empty:
        ts_by_year = (
            sub_tension.groupby(["year", assign_bin(sub_tension["p_model_under"]).rename("bin")], observed=True)
            .agg(
                n             = ("actual_under", "count"),
                actual_pct    = ("actual_under", "mean"),
                model_pct     = ("p_model_under", "mean"),
                avg_y_hat     = ("y_hat", "mean"),
                edge_pp       = ("edge_under", "mean"),
            )
            .reset_index()
        )
        for col in ["actual_pct", "model_pct"]:
            ts_by_year[col] = ts_by_year[col] * 100
        ts_by_year["edge_pp"]        = ts_by_year["edge_pp"] * 100
        ts_by_year["calib_error_pp"] = ts_by_year["actual_pct"] - ts_by_year["model_pct"]
        ts_by_year = ts_by_year[ts_by_year["n"] >= DISPLAY_MIN_N].copy()
        ts_by_year = add_flag(ts_by_year, min_bin_n, max_calib_error)
        if not ts_by_year.empty:
            print(ts_by_year.to_string(index=False))
        else:
            print("  (No bins with sufficient observations after year split)")
    else:
        print("  No tension-slice rows found")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate logistic calibration layer predicted probabilities — MLB Total Bases v2"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=SUPPORTED_MODES,
        help=f"Evaluation mode. Supported: {SUPPORTED_MODES}",
    )
    parser.add_argument("--line", type=float, default=1.5)
    args = parser.parse_args()

    cfg            = load_config()
    eval_cfg       = cfg["model"]["calibration"]["evaluation"]
    min_bin_n      = eval_cfg["min_bin_n"]
    max_calib_err  = eval_cfg["max_calibration_error"]
    print(f"Calibration thresholds — max_error: {max_calib_err*100:.0f}pp · min_bin_n: {min_bin_n}")

    df, bundle = load_data(args.line)
    scored     = score(df, bundle, args.line)

    if args.mode == "calibration":
        run_calibration(scored, args.line, min_bin_n, max_calib_err)


if __name__ == "__main__":
    main()
