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
- SETTLE_PREFIX (optional; default: nba/rebounds/daily_runs)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
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
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def lambda_handler(event, context):
    # DuckDB httpfs expects a writable home directory in Lambda.
    os.environ.setdefault("HOME", "/tmp")
    root = _repo_root()
    today_et = datetime.now(ET).strftime("%Y-%m-%d")

    config_path = os.environ.get("CONFIG_PATH", "config/nba_rebounds_prod.yaml")
    settle_bucket = os.environ.get("SETTLE_BUCKET", "nba-betting-mt")
    settle_prefix = os.environ.get("SETTLE_PREFIX", "nba/rebounds/daily_runs")

    step_results = []
    try:
        pipeline_cmd = [
            sys.executable,
            "scripts/run_nba_rebounds_daily_pipeline.py",
            "--config",
            config_path,
            "--slate-date",
            today_et,
        ]
        _run(pipeline_cmd, root)
        step_results.append({"step": "pipeline", "status": "ok"})

        settle_cmd = [
            sys.executable,
            "scripts/rebounds_settle_runs.py",
            "--bucket",
            settle_bucket,
            "--runs-prefix",
            settle_prefix,
            "--date",
            today_et,
            "--latest-only",
        ]
        _run(settle_cmd, root)
        step_results.append({"step": "settlement", "status": "ok"})

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "status": "ok",
                    "date_et": today_et,
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
                    "steps": step_results,
                    "error": str(exc),
                }
            ),
        }
