"""
NFL Rec Yards Lambda orchestrator.

Modes (set via EventBridge payload {"mode": "..."}):
  settle_and_rebuild — Settle yesterday + rebuild spine + send ops email  (8:30am ET daily)
  pipeline           — Fetch live props, score, send plays email           (9:00am ET daily)
  spine_update       — Full spine rebuild from scratch (pre-season / weekly)

Env vars:
  ODDS_API_KEY           (required for pipeline mode)
  SNS_TOPIC_ARN          (optional — SNS notifications on failure)
  SETTLEMENT_SES_SOURCE  (verified SES sender for HTML emails)
  SETTLEMENT_SES_TO      (comma-separated recipients)
  NFL_SEASON             (optional; defaults to computed season from today's date)
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

VALID_MODES = {"settle_and_rebuild", "pipeline", "spine_update", "settle"}


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
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout}"
        )
    return result.stdout


def _publish_sns(topic_arn: str, subject: str, message: str) -> None:
    if not topic_arn:
        return
    boto3.client("sns").publish(TopicArn=topic_arn, Subject=subject, Message=message)


def lambda_handler(event, context):
    os.environ.setdefault("HOME", "/tmp")
    root      = _repo_root()
    today_et  = datetime.now(ET).strftime("%Y-%m-%d")
    topic_arn = os.environ.get("SNS_TOPIC_ARN", "").strip()

    nfl_season = int(os.environ.get("NFL_SEASON", _current_nfl_season()))
    mode = _resolve_mode(event if isinstance(event, dict) else None)

    print(f"NFL Rec Yards Lambda | mode={mode} | date={today_et} | season={nfl_season}")

    scripts_dir  = root / "src" / "nfl_rec_yards_modeling" / "scripts"
    step_results: list[dict] = []

    try:
        if mode == "settle_and_rebuild":
            # Step 1: Settle yesterday's bets + send ops email (settle_rec_yards handles email itself)
            _run_capture(
                [sys.executable, str(scripts_dir / "settle_rec_yards.py")],
                cwd=root,
            )
            step_results.append({"step": "settle", "status": "ok"})

            # Step 2: Rebuild spine with latest box scores
            out = _run_capture(
                [sys.executable, str(scripts_dir / "update_spine.py"),
                 "--season", str(nfl_season)],
                cwd=root,
            )
            step_results.append({"step": "spine_rebuild", "status": "ok"})

        elif mode == "pipeline":
            gameday = (event or {}).get("gameday", today_et)
            _run_capture(
                [sys.executable, str(scripts_dir / "run_pipeline.py"),
                 "--gameday", gameday],
                cwd=root,
            )
            step_results.append({"step": "pipeline", "status": "ok"})

        elif mode == "spine_update":
            out = _run_capture(
                [sys.executable, str(scripts_dir / "update_spine.py"),
                 "--season", str(nfl_season)],
                cwd=root,
            )
            step_results.append({"step": "spine_update", "status": "ok"})
            _publish_sns(
                topic_arn,
                subject=f"NFL rec yards spine updated — {today_et}",
                message=f"Spine update complete for season {nfl_season}.\n\n{out[-3000:]}",
            )

        elif mode == "settle":
            # Legacy: settle only (no spine rebuild)
            _run_capture(
                [sys.executable, str(scripts_dir / "settle_rec_yards.py")],
                cwd=root,
            )
            step_results.append({"step": "settle", "status": "ok"})

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
            subject=f"NFL rec yards Lambda FAILED — {mode} — {today_et}",
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
