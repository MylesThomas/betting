"""
NFL Sacks Lambda orchestrator.

Modes (set via EventBridge payload {"mode": "..."}):
  spine_update  — Re-fetch current season from nfl_data_py, upload spine to S3  (Tue 9am ET)
  spine_verify  — Rebuild spine locally, compare vs S3 spine, do NOT upload     (Wed 9am ET)
  pipeline      — Fetch live props, score, upload bet sheet, notify              (Thu/Sun/Mon 9am ET)
  settle        — Settle yesterday's bets, send HTML summary email               (daily 10am ET)

Env vars:
  ODDS_API_KEY     (required for pipeline mode)
  SNS_TOPIC_ARN    (required for notifications)
  SES_SOURCE       (optional; verified SES sender for HTML emails)
  SES_TO           (optional; comma-separated recipients)
  NFL_SEASON       (optional; defaults to computed season from today's date)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3

ET = ZoneInfo("America/New_York")

VALID_MODES = {"spine_update", "spine_verify", "pipeline", "settle"}


def _repo_root() -> Path:
    current = Path(__file__).resolve().parent
    while True:
        if (current / "src").exists() and (current / ".gitignore").exists():
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


def _current_nfl_season() -> int:
    now = datetime.now(ET)
    return now.year if now.month >= 8 else now.year - 1


def _resolve_mode(event: dict | None) -> str:
    if not event or "mode" not in event:
        raise ValueError("Event must include 'mode' key")
    mode = str(event["mode"]).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Valid: {VALID_MODES}")
    return mode


def _run_capture(cmd: list[str], cwd: Path) -> str:
    print("run |", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout}")
    return result.stdout


def _publish_sns(topic_arn: str, subject: str, message: str) -> None:
    if not topic_arn:
        return
    boto3.client("sns").publish(TopicArn=topic_arn, Subject=subject, Message=message)


def lambda_handler(event, context):
    os.environ.setdefault("HOME", "/tmp")
    root = _repo_root()
    today_et = datetime.now(ET).strftime("%Y-%m-%d")
    topic_arn = os.environ.get("SNS_TOPIC_ARN", "").strip()

    nfl_season = int(os.environ.get("NFL_SEASON", _current_nfl_season()))
    mode = _resolve_mode(event if isinstance(event, dict) else None)

    print(f"NFL Sacks Lambda | mode={mode} | date={today_et} | season={nfl_season}")

    scripts_dir = root / "src" / "nfl_sacks_modeling" / "scripts"

    step_results = []
    try:
        if mode == "spine_update":
            out = _run_capture(
                [sys.executable, str(scripts_dir / "update_spine.py"), "--season", str(nfl_season)],
                cwd=root,
            )
            step_results.append({"step": "spine_update", "status": "ok"})
            _publish_sns(
                topic_arn,
                subject=f"NFL sacks spine updated — {today_et}",
                message=f"Spine update complete for season {nfl_season}.\n\n{out[-3000:]}",
            )

        elif mode == "spine_verify":
            out = _run_capture(
                [sys.executable, str(scripts_dir / "update_spine.py"),
                 "--season", str(nfl_season), "--verify"],
                cwd=root,
            )
            step_results.append({"step": "spine_verify", "status": "ok"})
            _publish_sns(
                topic_arn,
                subject=f"NFL sacks spine verified — {today_et}",
                message=f"Spine verify complete for season {nfl_season}.\n\n{out[-3000:]}",
            )

        elif mode == "pipeline":
            out = _run_capture(
                [sys.executable, str(scripts_dir / "run_pipeline.py"), "--gameday", today_et],
                cwd=root,
            )
            step_results.append({"step": "pipeline", "status": "ok"})
            # run_pipeline.py sends its own SES+SNS notification with bet details

        elif mode == "settle":
            out = _run_capture(
                [sys.executable, str(scripts_dir / "settle_sacks.py")],
                cwd=root,
            )
            step_results.append({"step": "settle", "status": "ok"})
            # settle_sacks.py sends its own SES+SNS settlement email

        return {
            "statusCode": 200,
            "body": json.dumps({
                "status":     "ok",
                "mode":       mode,
                "date_et":    today_et,
                "nfl_season": nfl_season,
                "steps":      step_results,
            }),
        }

    except Exception as exc:
        err_msg = str(exc)
        print(f"ERROR: {err_msg}")
        _publish_sns(
            topic_arn,
            subject=f"NFL sacks Lambda FAILED — {mode} — {today_et}",
            message=f"Mode: {mode}\nDate: {today_et}\nSeason: {nfl_season}\n\nError:\n{err_msg}",
        )
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status":     "error",
                "mode":       mode,
                "date_et":    today_et,
                "nfl_season": nfl_season,
                "steps":      step_results,
                "error":      err_msg,
            }),
        }
