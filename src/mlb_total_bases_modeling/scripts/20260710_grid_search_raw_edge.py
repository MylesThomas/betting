"""
MLB Total Bases — Grid Search with Raw-Edge Fix
================================================
Re-runs the IS/OOS strategy grid replacing novig-based edge qualification
with raw-based edge: edge_under = p_model_under - raw_prob_under.

Why:
  The original grid used edge_under = p_model_under - novig_prob_under.
  Novig strips the vig, making each side look ~2-4pp cheaper than reality.
  A bet only makes sense if p_model > raw_prob (true positive EV).
  Payout was already correct (under_price - 1 = actual decimal odds).
  So this is purely an edge-qualification fix — not a payout fix.

IS  = 2024 + 2025   (model training seasons)
OOS = 2026          (true hold-out)

Grid dimensions:
  edge_threshold : [0.00, 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
  odds_bucket    : ["all", "dogs (+odds)", "favs (-odds)"]
                    dogs = raw_prob_under < 0.50 = under_price > 2.0 = plus odds
  line_bucket    : ["all_lines", "0.5_only", "1.5_only"]

Usage:
  python src/mlb_total_bases_modeling/scripts/20260710_grid_search_raw_edge.py
  python src/mlb_total_bases_modeling/scripts/20260710_grid_search_raw_edge.py --no-upload
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET   = "the-odds-api-mt"
SPINE_KEY   = "mlb/total_bases_model/regression/mlb_tb_reg_spine.parquet"
MODEL_KEY   = "mlb/total_bases_model/model/mlb_tb_regression_v2.joblib"
IS_OUT_KEY  = "mlb/total_bases_model/backtest/mlb_tb_v2_is_grid_rawedge.csv"
OOS_OUT_KEY = "mlb/total_bases_model/backtest/mlb_tb_v2_oos_grid_rawedge.csv"
OUT_DIR     = Path.home() / "Downloads/tmp/mlb_total_bases"

TARGET      = "total_bases"
IS_SEASONS  = [2024, 2025]
OOS_SEASON  = 2026

EDGE_THRESHOLDS = [0.00, 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
ODDS_BUCKETS    = ["all", "dogs (+odds)", "favs (-odds)"]
LINE_BUCKETS    = ["all_lines", "0.5_only", "1.5_only"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, dict]:
    s3 = boto3.client("s3")

    print("Loading regression spine from S3 ...")
    body  = s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()
    spine = pd.read_parquet(BytesIO(body))
    print(f"  {len(spine):,} rows")

    print("Loading model bundle from S3 ...")
    body   = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)["Body"].read()
    bundle = joblib.load(BytesIO(body))
    print(f"  Model: {bundle.get('combo_name','?')} | features: {bundle['features_numeric']}")
    return spine, bundle


# ── Feature engineering ───────────────────────────────────────────────────────

def prepare_spine(spine: pd.DataFrame) -> pd.DataFrame:
    df = spine.copy()
    df["season"]          = pd.to_datetime(df["game_date"]).dt.year
    df["raw_prob_under"]  = 1.0 / df["under_price"]
    df["raw_prob_over"]   = 1.0 / df["over_price"]

    # min_line / max_line per player-game (book-invariant feature)
    line_range = (
        df.groupby(["name_norm", "game_date"])["line"]
        .agg(min_line="min", max_line="max")
        .reset_index()
    )
    df = df.merge(line_range, on=["name_norm", "game_date"], how="left")
    return df


def score_spine(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Add y_hat and p_model_under to every row in spine."""
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    features     = bundle["features_numeric"]
    calib_models = bundle.get("calib_models", {})

    # Compute y_hat at (player, game_date, line) grain — same for all books
    unique_pg = df[["name_norm", "game_date", "line"] + features].drop_duplicates(
        subset=["name_norm", "game_date", "line"]
    )
    unique_pg = unique_pg.dropna(subset=features)

    X     = unique_pg[features].values.astype(float)
    X_sc  = scaler.transform(X)
    y_hat = model.predict(X_sc).astype(float)
    unique_pg = unique_pg[["name_norm", "game_date", "line"]].copy()
    unique_pg["y_hat"] = y_hat

    # Calibrate per line
    p_model_rows = []
    for line, calib in calib_models.items():
        sub = unique_pg[unique_pg["line"] == line].copy()
        if sub.empty:
            continue
        proba = calib.predict_proba(sub["y_hat"].values.reshape(-1, 1))[:, 1]
        sub["p_model_over"]  = np.clip(proba, 0.01, 0.99)
        sub["p_model_under"] = 1.0 - sub["p_model_over"]
        p_model_rows.append(sub)

    if not p_model_rows:
        raise RuntimeError("No calibration models matched any line in spine")
    p_model_df = pd.concat(p_model_rows, ignore_index=True)

    # Merge back to full per-book spine
    df = df.merge(p_model_df[["name_norm", "game_date", "line", "y_hat", "p_model_under"]],
                  on=["name_norm", "game_date", "line"], how="inner")
    df["edge_under_raw"]   = df["p_model_under"] - df["raw_prob_under"]
    df["edge_under_novig"] = df["p_model_under"] - df["novig_prob_under"]
    return df


# ── Grid search ───────────────────────────────────────────────────────────────

def max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum))


def run_grid(df: pd.DataFrame, label: str) -> pd.DataFrame:
    records = []
    for edge_thresh in EDGE_THRESHOLDS:
        for odds_bucket in ODDS_BUCKETS:
            for line_bucket in LINE_BUCKETS:
                # Line filter
                if line_bucket == "0.5_only":
                    sub = df[df["line"] == 0.5]
                elif line_bucket == "1.5_only":
                    sub = df[df["line"] == 1.5]
                else:
                    sub = df

                # Edge filter (raw-based)
                sub = sub[sub["edge_under_raw"] >= edge_thresh]

                # Odds filter
                if odds_bucket == "dogs (+odds)":
                    sub = sub[sub["raw_prob_under"] < 0.50]   # under_price > 2.0 = plus odds
                elif odds_bucket == "favs (-odds)":
                    sub = sub[sub["raw_prob_under"] >= 0.50]

                if sub.empty:
                    continue

                # Payout = actual decimal odds - 1 (correct from day one)
                won  = (sub[TARGET] < sub["line"]).values.astype(float)
                payout = (sub["under_price"].values - 1.0)
                pnl  = np.where(won, payout, -1.0)

                n        = len(pnl)
                units    = float(pnl.sum())
                win_rate = float(won.mean())
                roi      = float(pnl.mean())
                mdd      = max_drawdown(pnl)

                records.append({
                    "edge_threshold":     edge_thresh,
                    "direction":          "UNDER",
                    "odds_bucket":        odds_bucket,
                    "line_bucket":        line_bucket,
                    "n_bets":             n,
                    "win_rate":           round(win_rate, 4),
                    "units_won":          round(units, 2),
                    "roi":                round(roi, 4),
                    "avg_raw_prob_under": round(float(sub["raw_prob_under"].mean()), 4),
                    "avg_edge_raw":       round(float(sub["edge_under_raw"].mean()), 4),
                    "max_drawdown":       round(mdd, 2),
                })

    result = pd.DataFrame(records).sort_values("units_won", ascending=False).reset_index(drop=True)
    print(f"\n{label} grid — top 10 by units:")
    print(result.head(10)[
        ["edge_threshold", "odds_bucket", "line_bucket", "n_bets", "win_rate", "units_won", "roi", "avg_edge_raw"]
    ].to_string(index=False))
    return result


# ── Comparison vs original novig grid ────────────────────────────────────────

def compare_vs_original(raw_oos: pd.DataFrame) -> None:
    # Original novig results for key strategies
    original = [
        {"edge": 0.05, "odds": "dogs (+odds)", "lines": "1.5_only", "novig_units": 211.90, "novig_roi": 0.0172, "novig_n": 12323},
        {"edge": 0.07, "odds": "dogs (+odds)", "lines": "1.5_only", "novig_units": 124.06, "novig_roi": 0.0121, "novig_n": 10283},
        {"edge": 0.15, "odds": "dogs (+odds)", "lines": "1.5_only", "novig_units":  71.99, "novig_roi": 0.0251, "novig_n":  2872},
        {"edge": 0.20, "odds": "dogs (+odds)", "lines": "1.5_only", "novig_units":  40.03, "novig_roi": 0.0338, "novig_n":  1186},
    ]
    print("\n── Comparison: novig edge vs raw edge (OOS, dogs, 1.5 only) ──")
    print(f"{'edge':>6}  {'novig_n':>8}  {'novig_u':>8}  {'novig_roi':>10}  │  {'raw_n':>8}  {'raw_u':>8}  {'raw_roi':>10}")
    print("─" * 78)
    for o in original:
        mask = (
            (raw_oos["edge_threshold"] == o["edge"]) &
            (raw_oos["odds_bucket"]    == o["odds"]) &
            (raw_oos["line_bucket"]    == o["lines"])
        )
        row = raw_oos[mask]
        if row.empty:
            raw_n, raw_u, raw_roi = "—", "—", "—"
        else:
            r = row.iloc[0]
            raw_n   = f"{int(r['n_bets']):,}"
            raw_u   = f"{r['units_won']:+.2f}"
            raw_roi = f"{r['roi']:+.4f}"
        print(
            f"{o['edge']:>6.2f}  {o['novig_n']:>8,}  {o['novig_units']:>+8.2f}  {o['novig_roi']:>+10.4f}"
            f"  │  {raw_n:>8}  {raw_u:>8}  {raw_roi:>10}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-upload", action="store_true", help="Skip S3 upload")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spine, bundle = load_data()
    df = prepare_spine(spine)
    print(f"\nSeason distribution: {df['season'].value_counts().sort_index().to_dict()}")

    print("\nScoring spine with model + calibration ...")
    df = score_spine(df, bundle)
    print(f"  Scored {len(df):,} rows")
    print(f"  avg raw_prob_under: {df['raw_prob_under'].mean():.4f}")
    print(f"  avg novig_prob_under: {df['novig_prob_under'].mean():.4f}")
    print(f"  avg edge_under_raw: {df['edge_under_raw'].mean():.4f}")
    print(f"  avg edge_under_novig: {df['edge_under_novig'].mean():.4f}")
    print(f"  rows with positive raw edge: {(df['edge_under_raw'] > 0).sum():,} / {len(df):,}")

    is_df  = df[df["season"].isin(IS_SEASONS)].copy()
    oos_df = df[df["season"] == OOS_SEASON].copy()
    print(f"\n  IS rows: {len(is_df):,} | OOS rows: {len(oos_df):,}")

    is_grid  = run_grid(is_df,  f"IS  ({', '.join(str(s) for s in IS_SEASONS)})")
    oos_grid = run_grid(oos_df, f"OOS ({OOS_SEASON})")

    compare_vs_original(oos_grid)

    # Save locally
    is_path  = OUT_DIR / "mlb_tb_v2_is_grid_rawedge.csv"
    oos_path = OUT_DIR / "mlb_tb_v2_oos_grid_rawedge.csv"
    is_grid.to_csv(is_path, index=False)
    oos_grid.to_csv(oos_path, index=False)
    print(f"\nSaved locally → {OUT_DIR}/mlb_tb_v2_*_grid_rawedge.csv")

    if not args.no_upload:
        s3 = boto3.client("s3")
        s3.put_object(Bucket=S3_BUCKET, Key=IS_OUT_KEY,  Body=is_grid.to_csv(index=False).encode())
        s3.put_object(Bucket=S3_BUCKET, Key=OOS_OUT_KEY, Body=oos_grid.to_csv(index=False).encode())
        print(f"Uploaded to s3://{S3_BUCKET}/mlb/total_bases_model/backtest/")


if __name__ == "__main__":
    main()
