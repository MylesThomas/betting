"""
Settle rebounds scored runs in S3 with realized REB outcomes.

Context:
- Daily scoring writes run artifacts under `rebounds/daily_runs/<date>/<run_id>/`.
- This script reads each `rebounds_scored_<date>.parquet`, joins player actual REB
  from NBA player game logs, and writes:
  - `rebounds_scored_settled_<date>.parquet` (row-level settlement)
  - `strategy_summary_<date>.csv` (ols/xgb/both/neither summary)
  - `settlement_manifest.json` (counts + diagnostics)
- Settlement is idempotent by default (skip existing settled parquet unless --overwrite).
- When `--rollup-s3-uri` ends with `yesterday.csv`, also writes `email_plays_yesterday.csv`
  (comma-delimited both/ols/xgb plays) and `email_plays_yesterday.html` (browser-openable table).
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

import yaml  # noqa: E402

from src.player_team_history.name_normalization import normalize_from_nba_api  # noqa: E402
from src.nba_rebounds_settlement_email import (  # noqa: E402
    format_settlement_email_plays_table,
    format_settlement_email_plays_table_html,
    prepare_email_plays_dataframe,
    wrap_email_plays_html_document,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Settle rebounds scored run artifacts in S3.")
    p.add_argument("--bucket", type=str, required=True, help="S3 bucket with scored run artifacts.")
    p.add_argument("--runs-prefix", type=str, required=True, help="S3 prefix root, e.g. rebounds/daily_runs.")
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
        "--allow-empty",
        action="store_true",
        help="Exit 0 when no scored parquet files exist for requested date range.",
    )
    p.add_argument(
        "--actuals-loader",
        type=str,
        choices=["duckdb", "boto3"],
        default="duckdb",
        help="How to load game-log actuals (default: duckdb).",
    )
    p.add_argument("--rollup-s3-uri", type=str, default="", help="Optional s3://bucket/key for combined strategy rollup CSV.")
    p.add_argument(
        "--sns-topic-arn",
        type=str,
        default="",
        help="Optional SNS topic ARN to publish settlement summary text.",
    )
    p.add_argument(
        "--max-unmatched-bet-rows",
        type=int,
        default=0,
        help="Guardrail: max allowed unsettled bet rows before marking run partial.",
    )
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
    dates_set = set(dates)

    if len(dates) > 30:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{runs_prefix.rstrip('/')}/"):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith(".parquet") and "/rebounds_scored_" in key and "_settled_" not in key:
                    parts = key.split("/")
                    if len(parts) >= 4:
                        date_part = parts[-3]
                        if date_part in dates_set:
                            keys.append(key)
    else:
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


def american_to_implied_prob(american: float) -> float:
    if american < 0:
        return float(-american) / (float(-american) + 100.0)
    return 100.0 / (float(american) + 100.0)


def load_actuals_for_dates_duckdb(seasons: list[str], dates: list[str]) -> pd.DataFrame:
    season_list = ", ".join([f"'{s}'" for s in sorted(set(seasons))])
    date_list = ", ".join([f"'{d}'" for d in sorted(set(dates))])
    
    # Construct exact S3 URIs to avoid DuckDB globbing issues with large buckets
    s3_uris = []
    for s in sorted(set(seasons)):
        for d in sorted(set(dates)):
            s3_uris.append(f"'s3://nba-api-mt/player_game_logs/{s}/{d}.csv'")
    
    if not s3_uris:
        return pd.DataFrame(columns=["season", "date", "player_normalized", "game_id", "reb_actual"])
        
    s3_uri_list = ", ".join(s3_uris)
    
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
        [{s3_uri_list}],
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
    settled["actuals_match_source"] = np.where(settled["reb_actual"].notna(), "exact_game_id", "unmatched")

    # Fallback for ID-system mismatches (e.g., NBA stats IDs vs ESPN IDs):
    # fill unresolved rows by (season, date, player_normalized) when there is
    # exactly one matching actual row for that player-date.
    unresolved = settled["reb_actual"].isna()
    if unresolved.any():
        by_player_date = (
            actuals.groupby(["season", "date", "player_normalized"], as_index=False)
            .agg(
                reb_actual=("reb_actual", "first"),
                n_actual_rows=("reb_actual", "size"),
            )
        )
        unique_player_date = by_player_date.loc[by_player_date["n_actual_rows"] == 1].drop(
            columns=["n_actual_rows"]
        )
        fallback = settled.loc[unresolved, ["season", "date", "player_normalized"]].merge(
            unique_player_date,
            on=["season", "date", "player_normalized"],
            how="left",
        )
        fallback_values = fallback["reb_actual"].to_numpy()
        fallback_hit = pd.notna(fallback_values)
        settled.loc[unresolved, "reb_actual"] = fallback_values
        unresolved_indices = settled.index[unresolved]
        settled.loc[unresolved_indices[fallback_hit], "actuals_match_source"] = "player_date_fallback"

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
    summary_input = settled.copy()
    summary_input["placed_under_implied_prob"] = summary_input["under_odds"].astype(float).apply(american_to_implied_prob)
    summary_input["reference_pnl_units"] = 0.0
    win_result_mask = summary_input["result"] == "win"
    loss_result_mask = summary_input["result"] == "loss"
    summary_input.loc[win_result_mask, "reference_pnl_units"] = summary_input.loc[
        win_result_mask, "under_odds"
    ].apply(american_profit_on_win)
    summary_input.loc[loss_result_mask, "reference_pnl_units"] = -1.0

    summary = (
        summary_input.groupby("strategy_bucket", as_index=False)
        .agg(
            n_rows=("strategy_bucket", "size"),
            n_bets=("is_bet", "sum"),
            n_win=("result", lambda x: int((x == "win").sum())),
            n_loss=("result", lambda x: int((x == "loss").sum())),
            n_push=("result", lambda x: int((x == "push").sum())),
            n_unsettled=("result", lambda x: int((x == "unsettled").sum())),
            pnl_units=("pnl_units", "sum"),
            reference_pnl_units=("reference_pnl_units", "sum"),
            avg_implied_prob_taken=("placed_under_implied_prob", "mean"),
        )
        .sort_values("strategy_bucket")
        .reset_index(drop=True)
    )
    settled_decisions = summary["n_win"] + summary["n_loss"] + summary["n_push"]
    summary["n_settled_bets"] = settled_decisions
    summary["hit_rate"] = np.where(
        (summary["n_win"] + summary["n_loss"]) > 0,
        summary["n_win"] / (summary["n_win"] + summary["n_loss"]),
        np.nan,
    )
    summary["roi_units_per_settled_bet"] = np.where(
        summary["n_settled_bets"] > 0,
        summary["reference_pnl_units"] / summary["n_settled_bets"],
        np.nan,
    )
    summary["roi_units_per_bet_all_placed"] = np.where(
        summary["n_bets"] > 0,
        summary["reference_pnl_units"] / summary["n_bets"],
        summary["reference_pnl_units"] / summary["n_rows"],
    )
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


def _rollup_sibling_email_plays_key(rollup_s3_uri: str) -> tuple[str, str] | None:
    """Same folder as yesterday.csv → email_plays_yesterday.csv (+ .html sibling)."""
    u = rollup_s3_uri.strip()
    if not u.startswith("s3://") or not u.endswith("yesterday.csv"):
        return None
    rest = u[5:]
    bucket, _, key = rest.partition("/")
    if bucket == "" or key == "":
        return None
    parent, _, _fname = key.rpartition("/")
    if parent == "":
        return None
    out_key = f"{parent}/email_plays_yesterday.csv"
    return bucket, out_key


def upload_email_plays_table_if_yesterday(plays: pd.DataFrame, rollup_s3_uri: str) -> None:
    if plays is None or len(plays) == 0:
        return
    loc = _rollup_sibling_email_plays_key(rollup_s3_uri)
    if loc is None:
        return
    bucket, out_key = loc
    prepared = prepare_email_plays_dataframe(plays)
    if len(prepared) == 0:
        return
    body = prepared.to_csv(index=False).encode("utf-8")
    write_bytes_s3(bucket, out_key, body)
    print(f"uploaded s3://{bucket}/{out_key}")

    fragment = format_settlement_email_plays_table_html(prepared)
    if fragment.strip():
        parent = out_key.rpartition("/")[0]
        html_key = f"{parent}/email_plays_yesterday.html"
        inner = (
            '<p style="margin:0 0 12px;font-weight:600;font-size:13px;">'
            "SETTLE-WINDOW PLAYS (both / ols / xgb)</p>"
            f"{fragment}"
        )
        html_doc = wrap_email_plays_html_document(inner)
        write_bytes_s3(bucket, html_key, html_doc.encode("utf-8"))
        print(f"uploaded s3://{bucket}/{html_key}")


def publish_settlement_summary_to_sns(topic_arn: str, body: str) -> str:
    import boto3

    resp = boto3.client("sns").publish(
        TopicArn=topic_arn,
        Subject="NBA rebounds settled results",
        Message=body[:256_000],
    )
    return resp["MessageId"]


def _fmt_metric(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def format_sns_summary_lines(rollup: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    ordered = rollup.sort_values("strategy_bucket").reset_index(drop=True)
    for _, row in ordered.iterrows():
        strategy = str(row["strategy_bucket"])
        lines.extend(
            [
                f"- {strategy}",
                (
                    "  rows={rows} bets={bets} wins={wins} losses={losses} "
                    "pushes={pushes} unsettled={unsettled}"
                ).format(
                    rows=int(row["n_rows"]),
                    bets=int(row["n_bets"]),
                    wins=int(row["n_win"]),
                    losses=int(row["n_loss"]),
                    pushes=int(row["n_push"]),
                    unsettled=int(row["n_unsettled"]),
                ),
                (
                    "  pnl_units={pnl} hit_rate={hit_rate} avg_implied_prob_taken={avg_prob} "
                    "roi_per_settled_bet={roi_settled} roi_per_bet_all_placed={roi_all}"
                ).format(
                    pnl=_fmt_metric(row["pnl_units"]),
                    hit_rate=_fmt_metric(row["hit_rate"]),
                    avg_prob=_fmt_metric(row["avg_implied_prob_taken"]),
                    roi_settled=_fmt_metric(row["roi_units_per_settled_bet"]),
                    roi_all=_fmt_metric(row["roi_units_per_bet_all_placed"]),
                ),
            ]
        )
    return lines


def format_sns_source_footer(rollup: pd.DataFrame) -> list[str]:
    unique_sources = sorted(rollup["source_scored_key"].dropna().astype(str).unique().tolist())
    lines = ["source files"]
    for idx, key in enumerate(unique_sources, start=1):
        lines.append(f"{idx}. {key}")
    return lines


def format_partial_settlement_lines(partials: list[dict]) -> list[str]:
    lines = ["partial settlement detected"]
    for item in partials:
        lines.append(
            (
                "- run_id={run_id} settle_date={slate} "
                "unmatched_bet_rows={unmatched_bets} unmatched_total_rows={unmatched_total} "
                "threshold={threshold}"
            ).format(
                run_id=item["run_id"],
                slate=item["slate"],
                unmatched_bets=item["unmatched_bets"],
                unmatched_total=item["unmatched_total"],
                threshold=item["threshold"],
            )
        )
    lines.append("Strategy summary suppressed due to settlement incompleteness.")
    return lines


def parse_run_id_from_key(scored_key: str) -> str:
    parts = scored_key.split("/")
    if len(parts) < 2:
        return "unknown"
    return parts[-2]


def format_strategy_summary_for_console(summary: pd.DataFrame) -> str:
    con = duckdb.connect()
    con.register("strategy_summary", summary)
    formatted_summary = con.execute(
        """
        SELECT
            strategy_bucket,
            n_rows,
            n_bets,
            printf('%d-%d-%d', n_win, n_loss, n_push) AS "W-L-P",
            n_unsettled,
            round(
                CASE
                    WHEN strategy_bucket = 'neither' THEN reference_pnl_units
                    ELSE pnl_units
                END,
                3
            ) AS pnl_units,
            round(hit_rate, 3) AS hit_rate,
            round(avg_implied_prob_taken, 3) AS avg_implied_prob_taken,
            round(roi_units_per_settled_bet, 3) AS roi_units_per_settled_bet,
            round(roi_units_per_bet_all_placed, 3) AS roi_units_per_bet_all_placed
        FROM strategy_summary
        ORDER BY
            CASE WHEN strategy_bucket = 'neither' THEN 1 ELSE 0 END,
            strategy_bucket
        """
    ).fetchdf()
    con.close()
    # Avoid hard dependency on optional `tabulate` in Lambda runtime.
    try:
        return formatted_summary.to_markdown(index=False)
    except ImportError:
        return formatted_summary.to_string(index=False)


def main() -> None:
    args = parse_args()
    date_list = parse_date_inputs(args)
    scored_keys = list_scored_keys(args.bucket, args.runs_prefix, date_list)
    if args.latest_only:
        scored_keys = keep_latest_run_per_date(scored_keys)
    if len(scored_keys) == 0:
        if args.allow_empty:
            print("settlement_noop", f"reason=no_scored_keys", f"dates={','.join(date_list)}", sep=" | ")
            sns_topic_arn = args.sns_topic_arn.strip()
            if sns_topic_arn:
                msg_lines = [
                    "NBA rebounds settled results",
                    f"settle_dates={', '.join(date_list)}",
                    "status=no_scored_runs",
                    "runs=0",
                    "",
                    "No scored run artifacts found for settlement window.",
                ]
                msg_id = publish_settlement_summary_to_sns(sns_topic_arn, "\n".join(msg_lines))
                print(
                    "published_settlement_to_sns",
                    f"topic_arn={sns_topic_arn}",
                    f"message_id={msg_id}",
                    "mode=noop",
                    sep=" | ",
                )
            return
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
    partial_runs: list[dict] = []
    email_plays_frames: list[pd.DataFrame] = []
    for key in scored_keys:
        run_df = all_scored.loc[all_scored["__scored_s3_key"] == key].drop(columns=["__scored_s3_key"]).copy()
        settled = settle_rows(run_df, actuals)
        summary = summarize_strategy(settled)

        run_prefix = key.rsplit("/", 1)[0]
        slate = str(pd.to_datetime(run_df["date"]).dt.date.iloc[0])
        settled_key = f"{run_prefix}/rebounds_scored_settled_{slate}.parquet"
        summary_key = f"{run_prefix}/strategy_summary_{slate}.csv"
        manifest_key = f"{run_prefix}/settlement_manifest.json"
        unmatched_key = f"{run_prefix}/unmatched_rows_{slate}.csv"

        unmatched_rows = settled.loc[settled["result"] == "unsettled"].copy()
        unmatched_bet_rows = int(unmatched_rows["is_bet"].sum())
        unmatched_total_rows = int(len(unmatched_rows))
        is_partial = unmatched_bet_rows > int(args.max_unmatched_bet_rows)

        unmatched_export_cols = [
            c
            for c in [
                "season",
                "date",
                "player_normalized",
                "game_id",
                "bookmaker",
                "line",
                "under_odds",
                "strategy_bucket",
                "is_bet",
                "result",
                "actuals_match_source",
            ]
            if c in unmatched_rows.columns
        ]
        write_bytes_s3(
            args.bucket,
            unmatched_key,
            unmatched_rows.loc[:, unmatched_export_cols].to_csv(index=False).encode("utf-8"),
        )

        settled_buf = BytesIO()
        settled.to_parquet(settled_buf, index=False)
        settled_buf.seek(0)
        write_bytes_s3(args.bucket, settled_key, settled_buf.getvalue())
        write_bytes_s3(args.bucket, summary_key, summary.to_csv(index=False).encode("utf-8"))

        manifest = {
            "settled_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_scored_key": key,
            "source_run_id": parse_run_id_from_key(key),
            "settle_date": slate,
            "settled_key": settled_key,
            "summary_key": summary_key,
            "n_rows": int(len(settled)),
            "n_unsettled_rows": int((settled["result"] == "unsettled").sum()),
            "n_unsettled_bet_rows": unmatched_bet_rows,
            "max_unmatched_bet_rows": int(args.max_unmatched_bet_rows),
            "settlement_status": "partial" if is_partial else "complete",
            "n_distinct_players": int(settled["player_normalized"].nunique()),
            "unmatched_rows_key": unmatched_key,
            "actuals_match_source_counts": {
                k: int(v)
                for k, v in settled["actuals_match_source"].value_counts(dropna=False).sort_index().items()
            },
        }
        if "game_id_source" in settled.columns:
            manifest["game_id_source_counts"] = {
                k: int(v) for k, v in settled["game_id_source"].value_counts(dropna=False).sort_index().items()
            }
        write_bytes_s3(args.bucket, manifest_key, json.dumps(manifest, indent=2).encode("utf-8"))

        summary["source_scored_key"] = key
        rollup_rows.append(summary)
        bet_mask = settled["strategy_bucket"].isin(["both", "ols", "xgb"])
        if bet_mask.any():
            email_plays_frames.append(
                settled.loc[
                    bet_mask,
                    [
                        "player_normalized",
                        "strategy_bucket",
                        "bookmaker",
                        "line",
                        "reb_actual",
                        "under_odds",
                        "date",
                        "result",
                    ],
                ].copy()
            )
        summary_print = summary.copy()
        for col in [
            "hit_rate",
            "pnl_units",
            "avg_implied_prob_taken",
            "roi_units_per_settled_bet",
            "roi_units_per_bet_all_placed",
        ]:
            summary_print[col] = summary_print[col].round(3)
        print(
            "settled_run",
            f"source={key}",
            f"settled=s3://{args.bucket}/{settled_key}",
            f"summary=s3://{args.bucket}/{summary_key}",
            f"unmatched_rows=s3://{args.bucket}/{unmatched_key}",
            f"settlement_status={'partial' if is_partial else 'complete'}",
            sep=" | ",
        )
        print("strategy_summary")
        print(format_strategy_summary_for_console(summary_print))
        if is_partial:
            partial_runs.append(
                {
                    "run_id": parse_run_id_from_key(key),
                    "slate": slate,
                    "unmatched_bets": unmatched_bet_rows,
                    "unmatched_total": unmatched_total_rows,
                    "threshold": int(args.max_unmatched_bet_rows),
                }
            )
            print(
                "settlement_guardrail",
                f"source={key}",
                f"status=partial",
                f"unmatched_bet_rows={unmatched_bet_rows}",
                f"max_unmatched_bet_rows={int(args.max_unmatched_bet_rows)}",
                sep=" | ",
            )

    rollup = pd.concat(rollup_rows, ignore_index=True)
    upload_rollup_if_requested(rollup, args.rollup_s3_uri)
    if email_plays_frames:
        email_plays = pd.concat(email_plays_frames, ignore_index=True)
        upload_email_plays_table_if_yesterday(email_plays, args.rollup_s3_uri)

    print("\n" + "=" * 80)
    print("TOTAL AGGREGATED SUMMARY (Across all processed runs)")
    print("=" * 80)
    
    rollup["_prob_sum"] = rollup["avg_implied_prob_taken"] * rollup["n_rows"]
    
    total_summary = (
        rollup.groupby("strategy_bucket", as_index=False)
        .agg(
            n_rows=("n_rows", "sum"),
            n_bets=("n_bets", "sum"),
            n_win=("n_win", "sum"),
            n_loss=("n_loss", "sum"),
            n_push=("n_push", "sum"),
            n_unsettled=("n_unsettled", "sum"),
            pnl_units=("pnl_units", "sum"),
            reference_pnl_units=("reference_pnl_units", "sum"),
            _prob_sum=("_prob_sum", "sum"),
        )
        .sort_values("strategy_bucket")
        .reset_index(drop=True)
    )
    
    total_summary["avg_implied_prob_taken"] = total_summary["_prob_sum"] / total_summary["n_rows"]
    
    total_summary["hit_rate"] = np.where(
        (total_summary["n_win"] + total_summary["n_loss"]) > 0,
        total_summary["n_win"] / (total_summary["n_win"] + total_summary["n_loss"]),
        np.nan,
    )
    
    settled_decisions = total_summary["n_win"] + total_summary["n_loss"] + total_summary["n_push"]
    total_summary["n_settled_bets"] = settled_decisions
    
    total_summary["roi_units_per_settled_bet"] = np.where(
        total_summary["n_settled_bets"] > 0,
        total_summary["reference_pnl_units"] / total_summary["n_settled_bets"],
        np.nan,
    )
    
    total_summary["roi_units_per_bet_all_placed"] = np.where(
        total_summary["n_bets"] > 0,
        total_summary["reference_pnl_units"] / total_summary["n_bets"],
        total_summary["reference_pnl_units"] / total_summary["n_rows"],
    )
    
    for col in ["hit_rate", "pnl_units", "roi_units_per_settled_bet", "roi_units_per_bet_all_placed", "avg_implied_prob_taken"]:
        if col in total_summary.columns:
            total_summary[col] = total_summary[col].round(3)
            
    print(format_strategy_summary_for_console(total_summary))
    print("=" * 80 + "\n")

    # ── superstar split ───────────────────────────────────────────────────────
    try:
        _cfg_path = Path(__file__).resolve().parent.parent / "src" / "nba_rebounds_modeling" / "config" / "model_config.yaml"
        with open(_cfg_path) as _f:
            _cfg = yaml.safe_load(_f)
        _superstar_set = set(_cfg.get("llm_features", {}).get("superstar_players", []))
        if _superstar_set:
            rollup["_is_superstar"] = rollup["player_normalized"].isin(_superstar_set) if "player_normalized" in rollup.columns else False
            _star_split = (
                rollup.groupby(["strategy_bucket", "_is_superstar"], as_index=False)
                .agg(n_bets=("n_bets","sum"), pnl=("reference_pnl_units","sum"), n_win=("n_win","sum"), n_loss=("n_loss","sum"))
            )
            _star_split["roi"] = (_star_split["pnl"] / _star_split["n_bets"]).round(3)
            _star_split["hit_rate"] = (_star_split["n_win"] / (_star_split["n_win"] + _star_split["n_loss"])).round(3)
            _star_split["tier"] = _star_split["_is_superstar"].map({True: "superstar", False: "non_superstar"})
            print("SUPERSTAR SPLIT (strategy_bucket × tier)")
            print("-" * 60)
            print(_star_split[["strategy_bucket","tier","n_bets","pnl","roi","hit_rate"]].to_string(index=False))
            print("-" * 60 + "\n")

            # Append summary rows into the rollup so the email panels can read them.
            # strategy_bucket values "star_split_superstar" / "star_split_non_superstar"
            # are picked up by _load_records in prod_notify_rebounds_sns.py.
            _both_star = _star_split[
                (_star_split["strategy_bucket"] == "both") & (_star_split["_is_superstar"] == True)  # noqa: E712
            ]
            _both_non = _star_split[
                (_star_split["strategy_bucket"] == "both") & (_star_split["_is_superstar"] == False)  # noqa: E712
            ]
            _extra_rows = []
            if len(_both_star):
                _r = _both_star.iloc[0]
                _extra_rows.append({**{c: np.nan for c in rollup.columns}, "strategy_bucket": "star_split_superstar", "pnl_units": float(_r["pnl"]), "reference_pnl_units": float(_r["pnl"]), "n_bets": int(_r["n_bets"]), "n_win": int(_r["n_win"]), "n_loss": int(_r["n_loss"])})
            if len(_both_non):
                _r = _both_non.iloc[0]
                _extra_rows.append({**{c: np.nan for c in rollup.columns}, "strategy_bucket": "star_split_non_superstar", "pnl_units": float(_r["pnl"]), "reference_pnl_units": float(_r["pnl"]), "n_bets": int(_r["n_bets"]), "n_win": int(_r["n_win"]), "n_loss": int(_r["n_loss"])})
            if _extra_rows:
                rollup = pd.concat([rollup, pd.DataFrame(_extra_rows)], ignore_index=True)
    except Exception as _e:
        print(f"[superstar split skipped: {_e}]")

    sns_topic_arn = args.sns_topic_arn.strip()
    if sns_topic_arn:
        source_lines = format_sns_source_footer(rollup)
        if len(partial_runs) > 0:
            msg_lines = [
                "NBA rebounds settled results",
                f"settle_dates={', '.join(date_list)}",
                f"runs={len(scored_keys)}",
                "",
                *format_partial_settlement_lines(partial_runs),
                "",
                *source_lines,
            ]
        else:
            summary_lines = format_sns_summary_lines(rollup)
            msg_lines = [
                "NBA rebounds settled results",
                f"settle_dates={', '.join(date_list)}",
                f"runs={len(scored_keys)}",
                "",
                "strategy summary",
                *summary_lines,
                "",
                *source_lines,
            ]
        msg_id = publish_settlement_summary_to_sns(sns_topic_arn, "\n".join(msg_lines))
        print("published_settlement_to_sns", f"topic_arn={sns_topic_arn}", f"message_id={msg_id}", sep=" | ")

    print("settlement_complete", f"runs={len(scored_keys)}", sep=" | ")


if __name__ == "__main__":
    main()
