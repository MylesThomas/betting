"""
MLB Batter Hits Lambda orchestrator.

Modes (set via EventBridge payload {"mode": "..."}):
  spine_update  — Rebuild spine + retrain models (8:30 AM ET)
  combined      — Settle yesterday + run today's pipeline → ONE email (9:00 AM ET daily)
  pipeline      — Fetch props, score, email bets (standalone)
  settle        — Settle yesterday's bets (standalone)

Strategy: 0.5 UNDER · edge >= 2pp
OOS: 4,600 bets · 53.5% hit · +426u · +9.27% ROI · net/MDD=4.15

Env vars:
  ODDS_API_KEY           (required for pipeline / combined modes)
  SES_SOURCE             (verified SES sender)
  SES_TO                 (comma-separated recipients)
  SNS_TOPIC_ARN          (optional SNS push notification)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3

ET          = ZoneInfo("America/New_York")
VALID_MODES = {"spine_update", "pipeline", "settle", "combined"}


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


def _send_ses(subject: str, html_body: str) -> None:
    ses_source  = os.environ.get("SES_SOURCE", "").strip()
    ses_to_raw  = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()
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


def _merge_html(html_pipeline: str, html_settle: str) -> str:
    def extract_body(html: str) -> str:
        m = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else html

    def extract_style(html: str) -> str:
        m = re.search(r"<style[^>]*>(.*?)</style>", html, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{extract_style(html_pipeline)}\n{extract_style(html_settle)}</style>"
        "</head><body>"
        f"{extract_body(html_pipeline)}"
        "<hr style='margin:36px 0;border:none;border-top:2px solid #bdc3c7'>"
        f"{extract_body(html_settle)}"
        "</body></html>"
    )


def lambda_handler(event, context):
    os.environ.setdefault("HOME", "/tmp")
    root     = _repo_root()
    today_et = datetime.now(ET).strftime("%Y-%m-%d")
    mode     = _resolve_mode(event if isinstance(event, dict) else None)

    print(f"MLB Batter Hits Lambda | mode={mode} | date={today_et}")

    scripts_dir   = root / "src" / "mlb_batter_hits_modeling" / "scripts"
    step_results: list[dict] = []

    try:
        if mode == "spine_update":
            spine_out = _run_capture(
                [sys.executable, str(scripts_dir / "update_spine.py")],
                cwd=root,
            )
            step_results.append({"step": "spine_update", "status": "ok"})

        elif mode == "pipeline":
            gameday = (event or {}).get("gameday", today_et)
            _run_capture(
                [sys.executable, str(scripts_dir / "run_pipeline.py"), "--gameday", gameday],
                cwd=root,
            )
            step_results.append({"step": "pipeline", "status": "ok"})

        elif mode == "settle":
            gameday = (event or {}).get("gameday", None)
            cmd = [sys.executable, str(scripts_dir / "settle_hits.py")]
            if gameday:
                cmd += ["--gameday", gameday]
            _run_capture(cmd, cwd=root)
            step_results.append({"step": "settle", "status": "ok"})

        elif mode == "combined":
            settle_out   = "/tmp/hits_settle_out.json"
            pipeline_out = "/tmp/hits_pipeline_out.json"

            # Step 1: Settle yesterday (suppress individual email)
            settle_gameday = (event or {}).get("settle_gameday", None)
            settle_cmd = [sys.executable, str(scripts_dir / "settle_hits.py"),
                          "--output", settle_out, "--no-email"]
            if settle_gameday:
                settle_cmd += ["--gameday", settle_gameday]
            _run_capture(settle_cmd, cwd=root)
            step_results.append({"step": "settle", "status": "ok"})

            # Step 2: Run today's pipeline (suppress individual email)
            pipeline_gameday = (event or {}).get("gameday", today_et)
            _run_capture(
                [sys.executable, str(scripts_dir / "run_pipeline.py"),
                 "--gameday", pipeline_gameday, "--output", pipeline_out, "--no-email"],
                cwd=root,
            )
            step_results.append({"step": "pipeline", "status": "ok"})

            # Step 3: Merge and send ONE email
            settle_data   = json.loads(Path(settle_out).read_text())
            pipeline_data = json.loads(Path(pipeline_out).read_text())

            n_plays    = pipeline_data.get("n_play_bets", 0)
            yest_wins  = settle_data.get("yesterday_wins",  0)
            yest_loss  = settle_data.get("yesterday_losses", 0)
            yest_units = settle_data.get("yesterday_units",  0.0)
            s_wins     = settle_data.get("season_wins",  0)
            s_loss     = settle_data.get("season_losses", 0)
            s_units    = settle_data.get("season_units",  0.0)

            yesterday_str = f"{yest_wins}W/{yest_loss}L {yest_units:+.2f}u"
            season_str    = f"{s_wins}W/{s_loss}L {s_units:+.2f}u"
            play_word     = "play" if n_plays == 1 else "plays"
            subject = (
                f"MLB Batter Hits — {n_plays} {play_word} today · "
                f"Yesterday: {yesterday_str} · "
                f"Season: {season_str} — {today_et}"
            )

            html_settle   = settle_data.get("html", "")
            html_pipeline = pipeline_data.get("html", "")
            combined_html = _merge_html(html_pipeline, html_settle) if html_settle and html_pipeline else (html_pipeline or html_settle)
            _send_ses(subject, combined_html)
            step_results.append({"step": "email", "status": "ok"})

    except Exception as exc:
        print(f"ERROR: {exc}")
        step_results.append({"step": "error", "status": str(exc)})
        raise

    return {"statusCode": 200, "steps": step_results}
