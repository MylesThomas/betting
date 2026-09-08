"""
MLB Pitcher Outs Lambda orchestrator.

Modes (set via EventBridge payload {"mode": "..."}):
  spine_update  — Rebuild rolling spine from existing gamelogs on S3 (weekly)
  daily         — Settle yesterday's bets + score today + send combined email (daily 9am ET)

Strategy: UNDER minus_odds · edge ≥ 5pp (raw implied) · shrinkage=0.25 · line ≤ 17.5
OOS ROI: +13.83% (n=689, 2025+2026) · model: yhat = consensus_line (no ML)

Env vars:
  ODDS_API_KEY           (required for daily mode)
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
VALID_MODES = {"spine_update", "daily"}


def _repo_root() -> Path:
    current = Path(__file__).resolve().parent
    while True:
        if (current / "src").exists() and (current / ".gitignore").exists():
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


_ROOT = _repo_root()
sys.path.insert(0, str(_ROOT / "src"))
from spine_email_utils import build_spine_update_html, build_spine_update_subject  # noqa: E402


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


def _send_ses(subject: str, html_body: str) -> None:
    ses_source = os.environ.get("SES_SOURCE", "").strip()
    ses_to_raw = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()
    if not ses_source or not ses_to_raw:
        print("  SES not configured — skipping email")
        return
    to_list = [e.strip() for e in ses_to_raw.split(",") if e.strip()]
    boto3.client("ses", region_name="us-east-2").send_email(
        Source=ses_source,
        Destination={"ToAddresses": to_list},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {"Html": {"Data": html_body, "Charset": "UTF-8"}},
        },
    )
    print(f"  Email sent: {subject[:80]}")


def lambda_handler(event, context):
    os.environ.setdefault("HOME", "/tmp")
    root      = _repo_root()
    today_et  = datetime.now(ET).strftime("%Y-%m-%d")
    topic_arn = os.environ.get("SNS_TOPIC_ARN", "").strip()
    mode      = _resolve_mode(event if isinstance(event, dict) else None)

    print(f"MLB Pitcher Outs Lambda | mode={mode} | date={today_et}")

    scripts_dir  = root / "src" / "mlb_pitcher_outs_modeling" / "scripts"
    step_results: list[dict] = []

    try:
        if mode == "spine_update":
            out = _run_capture(
                [sys.executable, str(scripts_dir / "build_spine.py")],
                cwd=root,
            )
            step_results.append({"step": "spine_update", "status": "ok"})
            _publish_sns(topic_arn,
                subject=f"MLB pitcher outs spine updated — {today_et}",
                message=f"Spine rebuild complete.\n\n{out[-3000:]}")

        elif mode == "daily":
            # run_pipeline.py handles settle(yesterday) + score(today) + send combined email
            gameday = (event or {}).get("gameday", today_et)
            out = _run_capture(
                [sys.executable, str(scripts_dir / "run_pipeline.py"),
                 "--gameday", gameday],
                cwd=root,
            )
            step_results.append({"step": "daily", "status": "ok"})

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
            subject=f"MLB Pitcher Outs Lambda ERROR — {mode} — {today_et}",
            message=err_msg[:2000])
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "mode": mode, "error": err_msg}),
        }
