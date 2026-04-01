"""
Settle rebounds scored runs in S3 with realized REB outcomes.

Context:
- Daily scoring writes run artifacts under `nba/rebounds/daily_runs/<date>/<run_id>/`.
- This script reads each `rebounds_scored_<date>.parquet`, joins player actual REB
  from NBA player game logs, and writes:
  - `rebounds_scored_settled_<date>.parquet` (row-level settlement)
  - `strategy_summary_<date>.csv` (ols/xgb/both/neither summary)
  - `settlement_manifest.json` (counts + diagnostics)
- Settlement is idempotent by default (skip existing settled parquet unless --overwrite).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def ensure_repo_root_on_syspath() -> Path:
    current = Path(__file__).resolve().parent
    while True:
        if (current / ".gitignore").exists() and (current / "src").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


ensure_repo_root_on_syspath()

from src.player_team_history.name_normalization import normalize_from_nba_api  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Settle rebounds scored run artifacts in S3.")
    p.add_argument("--bucket", type=str, required=True, help="S3 bucket with scored run artifacts.")
    p.add_argument("--runs-prefix", type=str, required=True, help="S3 prefix root, e.g. nba/rebounds/daily_runs.")
    p.add_argument("--date", type=str, default="", help="Single slate date YYYY-MM-DD.")
    p.add_argument("--start-date", type=str, default="", help="Start date YYYY-MM-DD (inclusive).")
    p.add_argument("--end-date", type=str, default="", help="End date YYYY-MM-DD (inclusive).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing settled artifacts.")
    p.add_argument(
        "--latest-only",
        action="store_true",
        help="Settle only the latest run_id per date.",
    )
    p.add_argument(
        "--actuals-loader",
        type=str,
        choices=["duckdb", "boto3"],
        default="duckdb",
        help="How to load game-log actuals (default: duckdb).",
    )
    p.add_argument("--rollup-s3-uri", type=str, default="", help="Optional s3://bucket/key for combined strategy rollup CSV.")
    return p.parse_args()


def connect_duckdb_s3() -> duckdb.DuckDBPyConnection:
    if "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
        access_key = os.environ["AWS_ACCESS_KEY_ID"]
        secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    else:
        access_key = subprocess.check_output(["aws", "configure", "get", "aws_access_key_id"], text=True).strip()
        secret_key = subprocess.check_output(["aws", "configure", "get", "aws_secret_access_key"], text=True).strip()
        if access_key == "" or secret_key == "":
            raise ValueError("Missing AWS credentials.")
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("SET s3_region='us-east-2'")
    con.execute(f"SET s3_access_key_id='{access_key}'")
    con.execute(f"SET s3_secret_access_key='{secret_key}'")
    if "AWS_SESSION_TOKEN" in os.environ:
        con.execute(f"SET s3_session_token='{os.environ['AWS_SESSION_TOKEN']}'")
    return con


def list_scored_keys(bucket: str, runs_prefix: str, dates: list[str]) -> list[str]:
    import boto3

    s3 = boto3.client("s3")
    keys: list[str] = []
    for date_str in dates:
        prefix = f"{runs_prefix.rstrip('/')}/{date_str}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith(".parquet") and "/rebounds_scored_" in key and "_settled_" not in key:
                    keys.append(key)
    return sorted(keys)


def keep_latest_run_per_date(keys: list[str]) -> list[str]:
    latest_by_date: dict[str, tuple[str, str]] = {}
    for key in keys:
        # expected: .../<date>/<run_id>/rebounds_scored_<date>.parquet
        parts = key.split("/")
        if len(parts) < 4:
            continue
        date_part = parts[-3]
        run_id = parts[-2]
        if date_part not in latest_by_date or run_id > latest_by_date[date_part][0]:
            latest_by_date[date_part] = (run_id, key)
    out = [v[1] for _, v in sorted(latest_by_date.items())]
    return out


def parse_date_inputs(args: argparse.Namespace) -> list[str]:
    if args.date:
        return [str(pd.Timestamp(args.date).date())]
    if args.start_date and args.end_date:
        start = pd.Timestamp(args.start_date).normalize()
        end = pd.Timestamp(args.end_date).normalize()
        if end < start:
            raise ValueError("end-date must be >= start-date")
        return [str(d.date()) for d in pd.date_range(start, end, freq="D")]
    raise ValueError("Provide --date OR --start-date and --end-date.")


def read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    import boto3

    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def write_bytes_s3(bucket: str, key: str, body: bytes) -> None:
    import boto3

    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=body)


def american_profit_on_win(american: float) -> float:
    if american >= 100:
        return float(american) / 100.0
    return 100.0 / float(abs(american))


def load_actuals_for_dates_duckdb(seasons: list[str], dates: list[str]) -> pd.DataFrame:
    season_list = ", ".join([f"'{s}'" for s in sorted(set(seasons))])
    date_list = ", ".join([f"'{d}'" for d in sorted(set(dates))])
    con = connect_duckdb_s3()
    q = f"""
    WITH raw AS (
      SELECT
        regexp_extract(filename, '/player_game_logs/([^/]+)/', 1) AS season,
        regexp_extract(filename, '/player_game_logs/[^/]+/([^/]+)\\.csv$', 1) AS file_date,
        NULLIF(PLAYER_NAME, '') AS player_name,
        NULLIF(GAME_ID, '') AS game_id,
        NULLIF(REB, '') AS reb
      FROM read_csv_auto(
        's3://nba-api-mt/player_game_logs/*/*.csv',
        union_by_name=true,
        filename=true,
        all_varchar=true,
        ignore_errors=true
      )
    )
    SELECT season, file_date AS date, player_name AS PLAYER_NAME, game_id AS GAME_ID, reb AS REB
    FROM raw
    WHERE season IN ({season_list})
      AND file_date IN ({date_list})
      AND player_name IS NOT NULL
      AND game_id IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    con.close()
    df["player_normalized"] = df["PLAYER_NAME"].apply(normalize_from_nba_api)
    df["game_id"] = df["GAME_ID"].astype(str)
    df["reb_actual"] = pd.to_numeric(df["REB"], errors="coerce")
    out = df[["season", "date", "player_normalized", "game_id", "reb_actual"]].drop_duplicates()
    return out


def load_actuals_for_dates_boto3(seasons: list[str], dates: list[str]) -> pd.DataFrame:
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3")
    bucket = "nba-api-mt"
    frames: list[pd.DataFrame] = []
    for season in sorted(set(seasons)):
        for date_str in sorted(set(dates)):
            key = f"player_game_logs/{season}/{date_str}.csv"
            try:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            except ClientError as exc:
                err_code = exc.response["Error"]["Code"]
                if err_code in {"NoSuchKey", "404"}:
                    continue
                raise
            one = pd.read_csv(BytesIO(body))
            one["season"] = season
            one["date"] = date_str
            frames.append(one)

    if not frames:
        return pd.DataFrame(columns=["season", "date", "player_normalized", "game_id", "reb_actual"])

    df = pd.concat(frames, ignore_index=True)
    if "PLAYER_NAME" in df.columns:
        df["player_normalized"] = df["PLAYER_NAME"].apply(normalize_from_nba_api)
    else:
        df["player_normalized"] = df["PLAYER"].apply(normalize_from_nba_api)
    if "GAME_ID" in df.columns:
        df["game_id"] = df["GAME_ID"].astype(str)
    else:
        df["game_id"] = df["game_id"].astype(str)
    if "REB" in df.columns:
        df["reb_actual"] = pd.to_numeric(df["REB"], errors="coerce")
    else:
        df["reb_actual"] = pd.to_numeric(df["reb"], errors="coerce")
    out = df[["season", "date", "player_normalized", "game_id", "reb_actual"]].drop_duplicates()
    return out


def load_actuals_for_dates(seasons: list[str], dates: list[str], actuals_loader: str) -> pd.DataFrame:
    if actuals_loader == "duckdb":
        return load_actuals_for_dates_duckdb(seasons, dates)
    return load_actuals_for_dates_boto3(seasons, dates)


def add_strategy_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["strategy_bucket"] = np.where(
        out["play_both"],
        "both",
        np.where(out["play_ols_only"], "ols", np.where(out["play_xgb_only"], "xgb", "neither")),
    )
    return out


def settle_rows(scored: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "date", "player_normalized", "game_id"]
    settled = scored.copy()
    settled["game_id"] = settled["game_id"].astype(str)
    settled = settled.merge(actuals, on=keys, how="left")
    settled = add_strategy_bucket(settled)
    settled["is_bet"] = settled["strategy_bucket"] != "neither"
    settled["result"] = "unsettled"
    has_actual = settled["reb_actual"].notna()
    settled.loc[has_actual & (settled["reb_actual"] < settled["line"]), "result"] = "win"
    settled.loc[has_actual & (settled["reb_actual"] > settled["line"]), "result"] = "loss"
    settled.loc[has_actual & (settled["reb_actual"] == settled["line"]), "result"] = "push"
    settled["pnl_units"] = 0.0
    win_mask = settled["is_bet"] & (settled["result"] == "win")
    loss_mask = settled["is_bet"] & (settled["result"] == "loss")
    settled.loc[win_mask, "pnl_units"] = settled.loc[win_mask, "under_odds"].apply(american_profit_on_win)
    settled.loc[loss_mask, "pnl_units"] = -1.0
    settled["settled_at_utc"] = datetime.now(timezone.utc).isoformat()
    settled["settlement_version"] = "v1_under_only"
    return settled


def summarize_strategy(settled: pd.DataFrame) -> pd.DataFrame:
    summary = (
        settled.groupby("strategy_bucket", as_index=False)
        .agg(
            n_rows=("strategy_bucket", "size"),
            n_bets=("is_bet", "sum"),
            n_win=("result", lambda x: int((x == "win").sum())),
            n_loss=("result", lambda x: int((x == "loss").sum())),
            n_push=("result", lambda x: int((x == "push").sum())),
            n_unsettled=("result", lambda x: int((x == "unsettled").sum())),
            pnl_units=("pnl_units", "sum"),
        )
        .sort_values("strategy_bucket")
        .reset_index(drop=True)
    )
    settled_bets = summary["n_win"] + summary["n_loss"]
    summary["hit_rate"] = np.where(settled_bets > 0, summary["n_win"] / settled_bets, np.nan)
    summary["roi_units_per_bet"] = np.where(summary["n_bets"] > 0, summary["pnl_units"] / summary["n_bets"], np.nan)
    return summary


def upload_rollup_if_requested(rollup: pd.DataFrame, rollup_s3_uri: str) -> None:
    if rollup_s3_uri.strip() == "":
        return
    if not rollup_s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid rollup s3 uri: {rollup_s3_uri}")
    rest = rollup_s3_uri[5:]
    bucket, _, key = rest.partition("/")
    if bucket == "" or key == "":
        raise ValueError(f"Invalid rollup s3 uri: {rollup_s3_uri}")
    body = rollup.to_csv(index=False).encode("utf-8")
    write_bytes_s3(bucket, key, body)
    print(f"uploaded {rollup_s3_uri}")


def main() -> None:
    args = parse_args()
    date_list = parse_date_inputs(args)
    scored_keys = list_scored_keys(args.bucket, args.runs_prefix, date_list)
    if args.latest_only:
        scored_keys = keep_latest_run_per_date(scored_keys)
    if len(scored_keys) == 0:
        raise ValueError("No scored parquet files found for requested date range.")

    scored_frames = []
    for key in scored_keys:
        df = read_parquet_s3(args.bucket, key)
        df["__scored_s3_key"] = key
        scored_frames.append(df)
    all_scored = pd.concat(scored_frames, ignore_index=True)

    seasons = sorted(all_scored["season"].dropna().astype(str).unique().tolist())
    dates = sorted(pd.to_datetime(all_scored["date"]).dt.date.astype(str).unique().tolist())
    actuals = load_actuals_for_dates(seasons, dates, args.actuals_loader)

    rollup_rows = []
    for key in scored_keys:
        run_df = all_scored.loc[all_scored["__scored_s3_key"] == key].drop(columns=["__scored_s3_key"]).copy()
        settled = settle_rows(run_df, actuals)
        summary = summarize_strategy(settled)

        run_prefix = key.rsplit("/", 1)[0]
        slate = str(pd.to_datetime(run_df["date"]).dt.date.iloc[0])
        settled_key = f"{run_prefix}/rebounds_scored_settled_{slate}.parquet"
        summary_key = f"{run_prefix}/strategy_summary_{slate}.csv"
        manifest_key = f"{run_prefix}/settlement_manifest.json"

        settled_buf = BytesIO()
        settled.to_parquet(settled_buf, index=False)
        settled_buf.seek(0)
        write_bytes_s3(args.bucket, settled_key, settled_buf.getvalue())
        write_bytes_s3(args.bucket, summary_key, summary.to_csv(index=False).encode("utf-8"))

        manifest = {
            "settled_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_scored_key": key,
            "settled_key": settled_key,
            "summary_key": summary_key,
            "n_rows": int(len(settled)),
            "n_unsettled_rows": int((settled["result"] == "unsettled").sum()),
            "n_distinct_players": int(settled["player_normalized"].nunique()),
        }
        write_bytes_s3(args.bucket, manifest_key, json.dumps(manifest, indent=2).encode("utf-8"))

        summary["source_scored_key"] = key
        rollup_rows.append(summary)
        summary_print = summary.copy()
        for col in ["hit_rate", "roi_units_per_bet", "pnl_units"]:
            summary_print[col] = summary_print[col].round(3)
        print(
            "settled_run",
            f"source={key}",
            f"settled=s3://{args.bucket}/{settled_key}",
            f"summary=s3://{args.bucket}/{summary_key}",
            sep=" | ",
        )
        print("strategy_summary")
        print(summary_print.to_string(index=False))

    rollup = pd.concat(rollup_rows, ignore_index=True)
    upload_rollup_if_requested(rollup, args.rollup_s3_uri)
    print("settlement_complete", f"runs={len(scored_keys)}", sep=" | ")


if __name__ == "__main__":
    main()
