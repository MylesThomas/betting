"""
Compile rebounds strategy records from scored run artifacts in S3.

Context:
- Reads scored parquet artifacts written by run_nba_rebounds_daily_pipeline.py.
- Aggregates records for four strategy buckets:
  - ols (play_ols_only)
  - xgb (play_xgb_only)
  - both (play_both)
  - neither (play_neither)
- If REB is present in scored rows, computes outcomes and PnL in units using
  under odds (under-only policy).
"""

from __future__ import annotations

import argparse
from io import BytesIO

import boto3
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile rebounds strategy records from S3 scored artifacts.")
    p.add_argument("--bucket", type=str, required=True, help="S3 bucket name.")
    p.add_argument("--prefix", type=str, required=True, help="S3 prefix containing scored parquet files.")
    p.add_argument("--output-csv", type=str, required=True, help="Local output CSV path.")
    p.add_argument("--output-s3-uri", type=str, default="", help="Optional s3://bucket/key for summary CSV.")
    return p.parse_args()


def american_profit_on_win(american: float) -> float:
    if american >= 100:
        return float(american) / 100.0
    return 100.0 / float(abs(american))


def list_scored_keys(bucket: str, prefix: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(".parquet") and "rebounds_scored_" in key:
                keys.append(key)
    return sorted(keys)


def read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = boto3.client("s3")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def add_strategy_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["strategy_bucket"] = np.where(
        out["play_both"],
        "both",
        np.where(out["play_ols_only"], "ols", np.where(out["play_xgb_only"], "xgb", "neither")),
    )
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    work = add_strategy_bucket(df)
    work["is_bet"] = work["strategy_bucket"] != "neither"
    has_reb = "REB" in work.columns
    if has_reb:
        work["is_push"] = work["is_bet"] & (work["REB"] == work["line"])
        work["is_win"] = work["is_bet"] & (work["REB"] < work["line"])
        work["is_loss"] = work["is_bet"] & (work["REB"] > work["line"])
        work["pnl_units"] = 0.0
        win_mask = work["is_win"]
        loss_mask = work["is_loss"]
        work.loc[win_mask, "pnl_units"] = work.loc[win_mask, "under_odds"].apply(american_profit_on_win)
        work.loc[loss_mask, "pnl_units"] = -1.0
    else:
        work["is_push"] = False
        work["is_win"] = False
        work["is_loss"] = False
        work["pnl_units"] = 0.0

    summary = (
        work.groupby("strategy_bucket", as_index=False)
        .agg(
            n_rows=("strategy_bucket", "size"),
            n_bets=("is_bet", "sum"),
            n_win=("is_win", "sum"),
            n_loss=("is_loss", "sum"),
            n_push=("is_push", "sum"),
            pnl_units=("pnl_units", "sum"),
        )
        .sort_values("strategy_bucket")
        .reset_index(drop=True)
    )
    summary["hit_rate"] = np.where(summary["n_bets"] > 0, summary["n_win"] / summary["n_bets"], np.nan)
    summary["roi_units_per_bet"] = np.where(summary["n_bets"] > 0, summary["pnl_units"] / summary["n_bets"], np.nan)
    return summary


def upload_csv(csv_path: str, s3_uri: str) -> None:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    rest = s3_uri[5:]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=open(csv_path, "rb").read())
    print(f"uploaded {s3_uri}")


def main() -> None:
    args = parse_args()
    keys = list_scored_keys(args.bucket, args.prefix)
    if len(keys) == 0:
        raise ValueError(f"No scored parquet keys found under s3://{args.bucket}/{args.prefix}")

    all_frames = []
    for key in keys:
        df = read_parquet_s3(args.bucket, key)
        df["scored_s3_key"] = key
        all_frames.append(df)
    all_scored = pd.concat(all_frames, ignore_index=True)
    summary = summarize(all_scored)
    summary.to_csv(args.output_csv, index=False)
    print(f"compiled_records | files={len(keys)} | rows={len(all_scored):,} | output={args.output_csv}")
    print(summary.to_string(index=False))
    if args.output_s3_uri.strip():
        upload_csv(args.output_csv, args.output_s3_uri.strip())


if __name__ == "__main__":
    main()
