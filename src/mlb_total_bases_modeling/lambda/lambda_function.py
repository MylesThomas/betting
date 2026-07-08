"""
MLB Total Bases Lambda orchestrator.

Modes (set via EventBridge payload {"mode": "..."}):
  spine_update  — Fetch current-season Statcast data, rebuild rolling spine (8:30am ET)
  combined      — Settle yesterday + run today's pipeline → ONE email      (9:00am ET daily)
  pipeline      — Fetch live props, score, email bets                     (standalone)
  settle        — Settle yesterday's bets, update history, email          (standalone)

Strategy: UNDER 1.5 · dogs only (novig_under<50%) · edge ≥ 5pp
OOS ROI: +1.72% (n=12,323) · v2 XGBoost regression + Method C calibration

Env vars:
  ODDS_API_KEY           (required for pipeline / combined modes)
  SES_SOURCE             (verified SES sender)
  SES_TO                 (comma-separated recipients)
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


def _merge_html(html_pipeline: str, html_settle: str) -> str:
    """Combine pipeline and settle HTML bodies into one email document."""
    def extract_body(html: str) -> str:
        m = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else html

    def extract_style(html: str) -> str:
        m = re.search(r"<style[^>]*>(.*?)</style>", html, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    _SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
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

    print(f"MLB Total Bases Lambda | mode={mode} | date={today_et}")

    scripts_dir   = root / "src" / "mlb_total_bases_modeling" / "scripts"
    step_results: list[dict] = []

    try:
        if mode == "spine_update":
            _run_capture(
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
            cmd = [sys.executable, str(scripts_dir / "settle_total_bases.py")]
            if gameday:
                cmd += ["--gameday", gameday]
            _run_capture(cmd, cwd=root)
            step_results.append({"step": "settle", "status": "ok"})

        elif mode == "combined":
            # Step 1: settle yesterday → write result JSON to /tmp
            settle_out  = "/tmp/tb_settle_out.json"
            pipeline_out = "/tmp/tb_pipeline_out.json"

            settle_gameday = (event or {}).get("settle_gameday", None)
            settle_cmd = [sys.executable, str(scripts_dir / "settle_total_bases.py"),
                          "--output", settle_out]
            if settle_gameday:
                settle_cmd += ["--gameday", settle_gameday]
            _run_capture(settle_cmd, cwd=root)
            step_results.append({"step": "settle", "status": "ok"})

            # Step 2: pipeline today → write result JSON to /tmp
            pipeline_gameday = (event or {}).get("gameday", today_et)
            _run_capture(
                [sys.executable, str(scripts_dir / "run_pipeline.py"),
                 "--gameday", pipeline_gameday, "--output", pipeline_out],
                cwd=root,
            )
            step_results.append({"step": "pipeline", "status": "ok"})

            # Step 3: merge and send ONE email
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
                f"MLB Total Bases — {n_plays} {play_word} today · "
                f"Yesterday: {yesterday_str} · "
                f"Season: {season_str} — {today_et}"
            )

            combined_html = _merge_html(pipeline_data["html_body"], settle_data["html_body"])
            _send_ses(subject, combined_html)
            step_results.append({"step": "email", "status": "ok"})

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
