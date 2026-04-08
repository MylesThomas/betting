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
from pathlib import Path
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _run(cmd: list[str], cwd: Path) -> None:
    print("run", " ".join(cmd), sep=" | ")
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def lambda_handler(event, context):
    # DuckDB httpfs expects a writable home directory in Lambda.
    os.environ.setdefault("HOME", "/tmp")
    root = _repo_root()
    today_et = datetime.now(ET).strftime("%Y-%m-%d")

    config_path = os.environ.get("CONFIG_PATH", "config/nba_rebounds_prod.yaml")
    settle_bucket = os.environ.get("SETTLE_BUCKET", "nba-betting-mt")
    settle_prefix = os.environ.get("SETTLE_PREFIX", "rebounds/daily_runs")
    settle_days_lag = int(os.environ.get("SETTLE_DAYS_LAG", "1"))
    settle_window_days = int(os.environ.get("SETTLE_WINDOW_DAYS", "3"))
    if settle_window_days < 1:
        raise ValueError("SETTLE_WINDOW_DAYS must be >= 1")
    settle_end_date_et = (datetime.now(ET) - timedelta(days=settle_days_lag)).date()
    settle_start_date_et = settle_end_date_et - timedelta(days=settle_window_days - 1)
    sns_topic_arn = os.environ.get("SNS_TOPIC_ARN", "").strip()

    step_results = []
    try:
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

        settle_cmd = [
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
        ]
        if sns_topic_arn:
            settle_cmd.extend(["--sns-topic-arn", sns_topic_arn])
        _run(settle_cmd, root)
        step_results.append({"step": "settlement", "status": "ok"})

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "status": "ok",
                    "date_et": today_et,
                    "settle_start_date_et": settle_start_date_et.isoformat(),
                    "settle_end_date_et": settle_end_date_et.isoformat(),
                    "settle_window_days": settle_window_days,
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
                    "date_et": today_et,
                    "settle_start_date_et": settle_start_date_et.isoformat(),
                    "settle_end_date_et": settle_end_date_et.isoformat(),
                    "settle_window_days": settle_window_days,
                    "steps": step_results,
                    "error": str(exc),
                }
            ),
        }
