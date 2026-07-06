"""
MLB Total Bases Lambda orchestrator.

Modes (set via EventBridge payload {"mode": "..."}):
  spine_update  — Fetch current-season Statcast data, rebuild rolling spine (weekly)
  pipeline      — Fetch live props, score, email bets                     (daily during season)
  settle        — Settle yesterday's bets, update history, email          (daily during season)

Strategy: UNDER 1.5 · dogs only (novig_under<50%) · edge ≥ 5pp
OOS ROI: +1.72% (n=12,323) · v2 XGBoost regression + Method C calibration

Env vars:
  ODDS_API_KEY           (required for pipeline mode)
  SNS_TOPIC_ARN          (optional — SNS notifications)
  SES_SOURCE             (verified SES sender)
  SES_TO                 (comma-separated recipients)
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

ET          = ZoneInfo("America/New_York")
VALID_MODES = {"spine_update", "pipeline", "settle"}


def _repo_root() -> Path:
    current = Path(__file__).resolve().parent
    while True:
        if (current / "src").exists() and (current / ".gitignore").exists():
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


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
    boto3.client("sns").publish(TopicArn=topic_arn, Subject=subject[:100], Message=message)


def lambda_handler(event, context):
    os.environ.setdefault("HOME", "/tmp")
    root      = _repo_root()
    today_et  = datetime.now(ET).strftime("%Y-%m-%d")
    topic_arn = os.environ.get("SNS_TOPIC_ARN", "").strip()
    mode      = _resolve_mode(event if isinstance(event, dict) else None)

    print(f"MLB Total Bases Lambda | mode={mode} | date={today_et}")

    scripts_dir   = root / "src" / "mlb_total_bases_modeling" / "scripts"
    step_results: list[dict] = []

    try:
        if mode == "spine_update":
            out = _run_capture(
                [sys.executable, str(scripts_dir / "update_spine.py")],
                cwd=root,
            )
            step_results.append({"step": "spine_update", "status": "ok"})
            _publish_sns(topic_arn,
                subject=f"MLB TB spine updated — {today_et}",
                message=f"Spine rebuild complete.\n\n{out[-3000:]}")

        elif mode == "pipeline":
            gameday = (event or {}).get("gameday", today_et)
            out = _run_capture(
                [sys.executable, str(scripts_dir / "run_pipeline.py"), "--gameday", gameday],
                cwd=root,
            )
            step_results.append({"step": "pipeline", "status": "ok"})

        elif mode == "settle":
            gameday = (event or {}).get("gameday", None)
            cmd = [sys.executable, str(scripts_dir / "settle_total_bases.py")]
            if gameday:
                cmd += ["--gameday", gameday]
            out = _run_capture(cmd, cwd=root)
            step_results.append({"step": "settle", "status": "ok"})

        return {
            "statusCode": 200,
            "body": json.dumps({
                "status":  "ok",
                "mode":    mode,
                "date_et": today_et,
                "steps":   step_results,
            }),
        }

    except Exception as exc:
        err_msg = str(exc)
        print(f"ERROR: {err_msg}")
        _publish_sns(topic_arn,
            subject=f"MLB TB Lambda FAILED — {mode} — {today_et}",
            message=f"Mode: {mode}\nDate: {today_et}\n\nError:\n{err_msg}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status":  "error",
                "mode":    mode,
                "date_et": today_et,
                "steps":   step_results,
                "error":   err_msg,
            }),
        }
