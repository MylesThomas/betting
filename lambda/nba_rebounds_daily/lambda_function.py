"""
NBA rebounds daily Lambda orchestrator.

Runs two steps in order (fail-hard):
1) Daily pipeline scoring + notify
2) Settlement for latest run on ET date

Env:
- ODDS_API_KEY (required by live props fetch)
- SNS_TOPIC_ARN (required if notify_enabled=true in config)
- CONFIG_PATH (optional; default: config/nba_rebounds_prod.yaml)
- SETTLE_BUCKET (optional; default: nba-betting-mt)
- SETTLE_PREFIX (optional; default: rebounds/daily_runs)
- SETTLE_DAYS_LAG (optional; default: 1, so settlement end date is yesterday ET)
- SETTLE_WINDOW_DAYS (optional; default: 3, re-settle rolling window for late actuals)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import pandas as pd


ET = ZoneInfo("America/New_York")


def _repo_root() -> Path:
    current = Path(__file__).resolve().parent
    while True:
        if (current / "src").exists() and (current / ".gitignore").exists():
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


def _resolve_mode(event: dict | None) -> str:
    if event is None:
        return "both"
    if "mode" not in event:
        return "both"
    mode = str(event["mode"]).strip().lower()
    if mode not in {"pipeline", "settlement", "both"}:
        raise ValueError(f"Unsupported mode: {mode}")
    return mode


def _run(cmd: list[str], cwd: Path) -> None:
    print("run", " ".join(cmd), sep=" | ")
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _run_capture(cmd: list[str], cwd: Path) -> str:
    print("run", " ".join(cmd), sep=" | ")
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout}")
    return result.stdout


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    rest = s3_uri[5:]
    bucket, _, key = rest.partition("/")
    if bucket == "" or key == "":
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    return bucket, key


def _read_csv_s3(s3_uri: str) -> pd.DataFrame | None:
    import botocore.exceptions
    bucket, key = _parse_s3_uri(s3_uri)
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return pd.read_csv(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] in ['NoSuchKey', '404']:
            return None
        raise

def _format_window_section(label: str, rollup: pd.DataFrame | None) -> list[str]:
    if rollup is None or len(rollup) == 0:
        return [f"{label} strategy summary", "- no scored runs found for this window"]
        
    grouped = (
        rollup.groupby("strategy_bucket", as_index=False)
        .agg(
            n_rows=("n_rows", "sum"),
            n_bets=("n_bets", "sum"),
            n_win=("n_win", "sum"),
            n_loss=("n_loss", "sum"),
            n_push=("n_push", "sum"),
            n_unsettled=("n_unsettled", "sum"),
            pnl_units=("pnl_units", "sum"),
        )
        .sort_values("strategy_bucket")
        .reset_index(drop=True)
    )
    
    import duckdb
    con = duckdb.connect()
    con.register("rollup", grouped)
    formatted_summary = con.execute(
        """
        SELECT
            strategy_bucket AS strategy,
            n_rows AS rows,
            n_bets AS bets,
            printf('%d-%d-%d', n_win, n_loss, n_push) AS "w-l-p",
            n_unsettled AS un,
            round(pnl_units, 3) AS pnl,
            round(CASE WHEN (n_win + n_loss) > 0 THEN n_win * 1.0 / (n_win + n_loss) ELSE 0.0 END, 3) AS hit_rate,
            round(CASE WHEN (n_win + n_loss + n_push) > 0 THEN pnl_units / (n_win + n_loss + n_push) ELSE 0.0 END, 3) AS roi
        FROM rollup
        ORDER BY
            CASE WHEN strategy_bucket = 'neither' THEN 1 ELSE 0 END,
            strategy_bucket
        """
    ).fetchdf()
    con.close()
    
    lines = [f"{label} strategy summary"]
    
    # Add the bullet point list
    for _, row in formatted_summary.iterrows():
        lines.append(
            (
                "- {strategy}: rows={rows} bets={bets} "
                "w-l-p={wlp} unsettled={un} pnl_units={pnl:.3f} "
                "hit_rate={hit_rate:.3f} roi={roi:.3f}"
            ).format(
                strategy=str(row["strategy"]),
                rows=int(row["rows"]),
                bets=int(row["bets"]),
                wlp=str(row["w-l-p"]),
                un=int(row["un"]),
                pnl=float(row["pnl"]),
                hit_rate=float(row["hit_rate"]),
                roi=float(row["roi"]),
            )
        )
        
    lines.append("")
    
    # Add the markdown table
    try:
        table_str = formatted_summary.to_markdown(index=False)
        lines.extend(table_str.splitlines())
    except ImportError:
        table_str = formatted_summary.to_string(index=False)
        lines.extend(table_str.splitlines())
        
    return lines


def _publish_combined_settlement_sns(
    topic_arn: str,
    settle_end_date_et,
    yesterday_rollup_uri: str,
    all_time_rollup_uri: str,
    warnings: list[str] = None,
) -> str:
    yesterday_rollup = _read_csv_s3(yesterday_rollup_uri)
    all_time_rollup = _read_csv_s3(all_time_rollup_uri)
    lines = [
        "NBA rebounds settled results",
        f"settle_end_date_et={settle_end_date_et.isoformat()}",
        "",
    ]
    
    if warnings:
        lines.append("WARNING: partial settlement detected")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
        
    lines.extend([
        *_format_window_section("yesterday", yesterday_rollup),
        "",
        *_format_window_section("all-time", all_time_rollup),
        "",
        "rollup files",
        f"1. {yesterday_rollup_uri}",
        f"2. {all_time_rollup_uri}",
    ])
    resp = boto3.client("sns").publish(
        TopicArn=topic_arn,
        Subject="NBA rebounds settled results",
        Message="\n".join(lines)[:256_000],
    )
    return resp["MessageId"]


def lambda_handler(event, context):
    # DuckDB httpfs expects a writable home directory in Lambda.
    os.environ.setdefault("HOME", "/tmp")
    root = _repo_root()
    today_et = datetime.now(ET).strftime("%Y-%m-%d")

    config_path = os.environ.get("CONFIG_PATH", "config/nba_rebounds_prod.yaml")
    settle_bucket = os.environ.get("SETTLE_BUCKET", "nba-betting-mt")
    settle_prefix = os.environ.get("SETTLE_PREFIX", "rebounds/daily_runs")
    settle_days_lag = int(os.environ.get("SETTLE_DAYS_LAG", "1"))
    settle_window_days = int(os.environ.get("SETTLE_WINDOW_DAYS", "1"))
    if settle_window_days < 1:
        raise ValueError("SETTLE_WINDOW_DAYS must be >= 1")
    settle_all_time_days = int(os.environ.get("SETTLE_ALL_TIME_DAYS", "999999"))
    if settle_all_time_days < 1:
        raise ValueError("SETTLE_ALL_TIME_DAYS must be >= 1")
    settle_max_unmatched_bet_rows = int(os.environ.get("SETTLE_MAX_UNMATCHED_BET_ROWS", "0"))
    settle_end_date_et = (datetime.now(ET) - timedelta(days=settle_days_lag)).date()
    settle_start_date_et = settle_end_date_et - timedelta(days=settle_window_days - 1)
    if settle_all_time_days >= 999999:
        settle_all_time_start_date_et = datetime(1900, 1, 1).date()
    else:
        settle_all_time_start_date_et = settle_end_date_et - timedelta(days=settle_all_time_days - 1)
    sns_topic_arn = os.environ.get("SNS_TOPIC_ARN", "").strip()
    mode = _resolve_mode(event if isinstance(event, dict) else None)

    step_results = []
    try:
        if mode in {"pipeline", "both"}:
            pipeline_cmd = [
                sys.executable,
                "src/nba_rebounds_modeling/00_research/scripts/run_rebounds_daily_pipeline.py",
                "--config",
                config_path,
                "--slate-date",
                today_et,
            ]
            _run(pipeline_cmd, root)
            step_results.append({"step": "pipeline", "status": "ok"})

        if mode in {"settlement", "both"}:
            stamp = datetime.now(ET).strftime("%Y%m%dT%H%M%S")
            base_rollup_prefix = f"{settle_prefix.rstrip('/')}/_rollups/{today_et}/{stamp}"
            yesterday_rollup_uri = f"s3://{settle_bucket}/{base_rollup_prefix}/yesterday.csv"
            all_time_rollup_uri = f"s3://{settle_bucket}/{base_rollup_prefix}/all_time.csv"

            settle_yesterday_cmd = [
                sys.executable,
                "src/nba_rebounds_modeling/00_research/scripts/settle_rebounds_runs.py",
                "--bucket",
                settle_bucket,
                "--runs-prefix",
                settle_prefix,
                "--start-date",
                settle_start_date_et.isoformat(),
                "--end-date",
                settle_end_date_et.isoformat(),
                "--latest-only",
                "--allow-empty",
                "--overwrite",
                "--max-unmatched-bet-rows",
                str(settle_max_unmatched_bet_rows),
                "--rollup-s3-uri",
                yesterday_rollup_uri,
            ]
            yesterday_out = _run_capture(settle_yesterday_cmd, root)

            settle_all_time_cmd = [
                sys.executable,
                "src/nba_rebounds_modeling/00_research/scripts/settle_rebounds_runs.py",
                "--bucket",
                settle_bucket,
                "--runs-prefix",
                settle_prefix,
                "--start-date",
                settle_all_time_start_date_et.isoformat(),
                "--end-date",
                settle_end_date_et.isoformat(),
                "--latest-only",
                "--allow-empty",
                "--overwrite",
                "--max-unmatched-bet-rows",
                str(settle_max_unmatched_bet_rows),
                "--rollup-s3-uri",
                all_time_rollup_uri,
            ]
            all_time_out = _run_capture(settle_all_time_cmd, root)

            warnings = []
            for line in (yesterday_out + "\n" + all_time_out).splitlines():
                if "status=partial" in line and "settlement_guardrail" in line:
                    warnings.append(line)
            warnings = sorted(list(set(warnings)))

            if sns_topic_arn:
                msg_id = _publish_combined_settlement_sns(
                    sns_topic_arn,
                    settle_end_date_et,
                    yesterday_rollup_uri,
                    all_time_rollup_uri,
                    warnings,
                )
                print("published_settlement_to_sns", f"topic_arn={sns_topic_arn}", f"message_id={msg_id}", sep=" | ")
            step_results.append({"step": "settlement", "status": "ok"})

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "status": "ok",
                    "mode": mode,
                    "date_et": today_et,
                    "settle_start_date_et": settle_start_date_et.isoformat(),
                    "settle_all_time_start_date_et": settle_all_time_start_date_et.isoformat(),
                    "settle_end_date_et": settle_end_date_et.isoformat(),
                    "settle_window_days": settle_window_days,
                    "settle_all_time_days": settle_all_time_days,
                    "steps": step_results,
                }
            ),
        }
    except Exception as exc:
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "status": "error",
                    "mode": mode,
                    "date_et": today_et,
                    "settle_start_date_et": settle_start_date_et.isoformat(),
                    "settle_all_time_start_date_et": settle_all_time_start_date_et.isoformat(),
                    "settle_end_date_et": settle_end_date_et.isoformat(),
                    "settle_window_days": settle_window_days,
                    "settle_all_time_days": settle_all_time_days,
                    "steps": step_results,
                    "error": str(exc),
                }
            ),
        }
