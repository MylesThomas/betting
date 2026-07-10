"""
MLB Strikeouts Lambda orchestrator.

Modes (set via EventBridge payload {"mode": "..."}):
  settle_and_rebuild — Settle yesterday + rebuild spine + send Email 1 (8:30 AM ET)
  pipeline           — Score today + send Email 2 (plays + yesterday + all-time) (9:00 AM ET)
  spine_update       — Full spine rebuild from scratch (pre-season / weekly)
  settle             — (legacy) Settle only, no email

Env vars:
  ODDS_API_KEY           (required for pipeline mode)
  SNS_TOPIC_ARN          (optional — SNS notifications)
  SETTLEMENT_SES_SOURCE  (verified SES sender)
  SETTLEMENT_SES_TO      (comma-separated recipients, default: mylescgthomas@gmail.com)
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

ET          = ZoneInfo("America/New_York")
VALID_MODES = {"settle_and_rebuild", "pipeline", "spine_update", "settle"}

S3_BUCKET          = "the-odds-api-mt"
SETTLE_SUMMARY_KEY = "mlb/strikeouts_model/settled/last_settle_summary.json"

_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"


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


def _send_email(subject: str, html_body: str) -> None:
    ses_source = os.environ.get("SETTLEMENT_SES_SOURCE", "").strip()
    ses_to_raw = os.environ.get("SETTLEMENT_SES_TO", "mylescgthomas@gmail.com").strip()
    to_list    = [a.strip() for a in ses_to_raw.split(",") if a.strip()]
    if not ses_source or not to_list:
        print(f"  SES not configured (SES_SOURCE={ses_source!r}), skipping email")
        return
    try:
        boto3.client("ses", region_name="us-east-2").send_email(
            Source=ses_source,
            Destination={"ToAddresses": to_list},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Html": {"Data": html_body, "Charset": "UTF-8"},
                    "Text": {"Data": subject, "Charset": "UTF-8"},
                },
            },
        )
        print(f"  Email sent: {subject}")
    except Exception as e:
        print(f"  Email failed: {e}")


def _read_settle_summary() -> dict | None:
    try:
        body = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=SETTLE_SUMMARY_KEY)["Body"].read()
        return json.loads(body)
    except Exception as e:
        print(f"  Could not read settle summary: {e}")
        return None


def _build_email1_html(summary: dict | None, spine_out: str, today_et: str) -> str:
    now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")

    # ── Settle section ────────────────────────────────────────────────────────
    if summary:
        gd      = summary.get("gameday", "—")
        wins    = summary.get("wins", 0)
        losses  = summary.get("losses", 0)
        pushes  = summary.get("pushes", 0)
        dnps    = summary.get("dnps", 0)
        pnl     = summary.get("pnl", 0.0)
        at_bets = summary.get("all_time_bets", 0)
        at_wins = summary.get("all_time_wins", 0)
        at_losses = summary.get("all_time_losses", 0)
        at_pnl  = summary.get("all_time_pnl", 0.0)
        record  = f"{wins}W–{losses}L{f'–{pushes}P' if pushes else ''}{f'–{dnps}DNP' if dnps else ''}"
        pnl_color   = "#276221" if pnl >= 0 else "#c0392b"
        at_pnl_color = "#276221" if at_pnl >= 0 else "#c0392b"
        settle_html = f"""
<h3 style='margin-bottom:4px'>Settlement — {gd}</h3>
<p style='margin:0 0 12px'>
  Record: <strong>{record}</strong> &nbsp;·&nbsp;
  P&amp;L: <strong style='color:{pnl_color}'>{pnl:+.2f}u</strong> &nbsp;·&nbsp;
  All-time: <strong>{at_wins}W–{at_losses}L</strong> &nbsp;·&nbsp;
  <strong style='color:{at_pnl_color}'>{at_pnl:+.2f}u</strong> ({at_bets} bets)
</p>"""
    else:
        settle_html = "<p style='color:#c0392b'><strong>⚠ Settle summary not found</strong> — settle_strikeouts.py may have failed. Check CloudWatch logs.</p>"

    # ── Spine section (parse key lines from stdout) ───────────────────────────
    spine_lines = [l.strip() for l in spine_out.splitlines() if l.strip()]
    spine_preview = "<br>".join(spine_lines[-20:]) if spine_lines else "(no output)"
    spine_html = f"""
<h3 style='margin-bottom:4px'>Spine rebuild</h3>
<pre style='background:#f4f4f4;padding:10px;border-radius:4px;font-size:11px;overflow-x:auto'>{spine_preview}</pre>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{font-family:{_SANS};color:#222;max-width:900px;margin:auto;padding:20px}}
  h2 {{color:#2c3e50;margin-bottom:4px}}
  h3 {{color:#2c3e50}}
  pre {{white-space:pre-wrap;word-break:break-word}}
</style>
</head><body>
<h2>MLB Strikeouts — Daily Update — {today_et}</h2>
<p style='color:#666;font-size:13px;margin-top:0'>{now_str} &nbsp;·&nbsp; settle + spine rebuild complete</p>

{settle_html}
{spine_html}

<p style='font-size:12px;color:#888;margin-top:16px'>
  9:00 AM ET scoring run will use today's rebuilt spine. If spine timestamp above is not from {today_et}, the scoring run will be blocked.
</p>
</body></html>"""


def lambda_handler(event, context):
    os.environ.setdefault("HOME", "/tmp")
    root      = _repo_root()
    today_et  = datetime.now(ET).strftime("%Y-%m-%d")
    topic_arn = os.environ.get("SNS_TOPIC_ARN", "").strip()
    mode      = _resolve_mode(event if isinstance(event, dict) else None)

    print(f"MLB Strikeouts Lambda | mode={mode} | date={today_et}")

    scripts_dir   = root / "src" / "mlb_strikeouts_modeling" / "scripts"
    step_results: list[dict] = []

    try:
        if mode == "settle_and_rebuild":
            # Step 1: Settle yesterday's bets (writes summary JSON to S3, no email)
            _run_capture(
                [sys.executable, str(scripts_dir / "settle_strikeouts.py")],
                cwd=root,
            )
            step_results.append({"step": "settle", "status": "ok"})

            # Step 2: Rebuild spine
            spine_out = _run_capture(
                [sys.executable, str(scripts_dir / "update_spine.py")],
                cwd=root,
            )
            step_results.append({"step": "spine_rebuild", "status": "ok"})

            # Step 3: Send Email 1 — ops confirmation
            summary   = _read_settle_summary()
            email1    = _build_email1_html(summary, spine_out, today_et)
            yesterday = (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")
            pnl_str   = f"{summary['pnl']:+.2f}u" if summary else "?"
            record_str = f"{summary.get('wins',0)}W/{summary.get('losses',0)}L" if summary else "?"
            _send_email(
                f"MLB Strikeouts — {today_et} — Spine updated · {record_str} yesterday · {pnl_str}",
                email1,
            )

        elif mode == "pipeline":
            gameday = (event or {}).get("gameday", today_et)
            _run_capture(
                [sys.executable, str(scripts_dir / "run_pipeline.py"), "--gameday", gameday],
                cwd=root,
            )
            step_results.append({"step": "pipeline", "status": "ok"})

        elif mode == "spine_update":
            out = _run_capture(
                [sys.executable, str(scripts_dir / "update_spine.py")],
                cwd=root,
            )
            step_results.append({"step": "spine_update", "status": "ok"})
            _publish_sns(topic_arn,
                subject=f"MLB strikeouts spine updated — {today_et}",
                message=f"Spine rebuild complete.\n\n{out[-3000:]}")

        elif mode == "settle":
            # Legacy: settle only, no email, no spine rebuild
            gameday = (event or {}).get("gameday", None)
            cmd = [sys.executable, str(scripts_dir / "settle_strikeouts.py")]
            if gameday:
                cmd += ["--gameday", gameday]
            _run_capture(cmd, cwd=root)
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
            subject=f"MLB strikeouts Lambda FAILED — {mode} — {today_et}",
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
