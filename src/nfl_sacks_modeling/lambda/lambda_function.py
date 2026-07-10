"""
NFL Sacks Lambda orchestrator.

Modes (set via EventBridge payload {"mode": "..."}):
  settle_and_rebuild — Settle yesterday + rebuild spine + send Email 1  (daily 8:30am ET)
  pipeline           — Fetch live props, score + send Email 2            (daily 9:00am ET)
  spine_update       — Full spine rebuild from scratch (pre-season use)
  settle             — (legacy) Settle only, no email

Env vars:
  ODDS_API_KEY     (required for pipeline mode)
  SNS_TOPIC_ARN    (optional; SNS notifications on failure)
  SES_SOURCE       (verified SES sender for HTML emails)
  SES_TO           (comma-separated recipients)
  NFL_SEASON       (optional; defaults to computed season from today's date)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3

ET = ZoneInfo("America/New_York")

VALID_MODES        = {"settle_and_rebuild", "pipeline", "spine_update", "settle"}
SETTLE_SUMMARY_KEY = "nfl/sacks_model/settled/last_settle_summary.json"
S3_BUCKET          = "the-odds-api-mt"


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


def _send_email(subject: str, html_body: str) -> None:
    ses_source = os.environ.get("SES_SOURCE", "").strip()
    ses_to_raw = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()
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
                "Body":    {"Html": {"Data": html_body, "Charset": "UTF-8"}},
            },
        )
        print(f"  SES email sent: {subject}")
    except Exception as e:
        print(f"  SES send failed: {e}")


def _read_settle_summary() -> dict | None:
    try:
        body = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=SETTLE_SUMMARY_KEY)["Body"].read()
        return json.loads(body)
    except Exception as e:
        print(f"  Could not read settle summary: {e}")
        return None


def _build_email1_html(summary: dict | None, spine_out: str, today_et: str) -> str:
    yesterday = (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")

    if summary:
        yw = summary.get("yesterday_wins", 0)
        yl = summary.get("yesterday_losses", 0)
        yp = summary.get("yesterday_pushes", 0)
        y_pnl = summary.get("yesterday_pnl", 0.0)
        at_bets   = summary.get("all_time_bets", 0)
        at_wins   = summary.get("all_time_wins", 0)
        at_losses = summary.get("all_time_losses", 0)
        at_pushes = summary.get("all_time_pushes", 0)
        at_pnl    = summary.get("all_time_pnl", 0.0)

        y_pnl_s  = f"+{y_pnl:.2f}u"  if y_pnl  >= 0 else f"{y_pnl:.2f}u"
        at_pnl_s = f"+{at_pnl:.2f}u" if at_pnl >= 0 else f"{at_pnl:.2f}u"
        y_color  = "#16a34a" if y_pnl  >= 0 else "#dc2626"
        at_color = "#16a34a" if at_pnl >= 0 else "#dc2626"

        no_bets_yesterday = (yw + yl + yp) == 0
        if no_bets_yesterday:
            settle_html = f"<p style='color:#6b7280'>No NFL games or no qualifying bets yesterday ({yesterday}).</p>"
        else:
            settle_html = f"""
<table style="border-collapse:collapse;font-size:13px;margin-bottom:16px">
<tr>
  <td style="padding:6px 16px 6px 0;color:#6b7280">Yesterday ({yesterday})</td>
  <td style="padding:6px 16px;font-weight:600">{yw}W–{yl}L–{yp}P</td>
  <td style="padding:6px 16px;font-weight:600;color:{y_color}">{y_pnl_s}</td>
</tr>
<tr>
  <td style="padding:6px 16px 6px 0;color:#6b7280">All-time ({at_bets} bets)</td>
  <td style="padding:6px 16px;font-weight:600">{at_wins}W–{at_losses}L–{at_pushes}P</td>
  <td style="padding:6px 16px;font-weight:600;color:{at_color}">{at_pnl_s}</td>
</tr>
</table>"""
    else:
        settle_html = "<p style='color:#c0392b'><strong>⚠ Settle summary not found</strong> — settle_sacks.py may have failed. Check CloudWatch logs.</p>"

    spine_lines   = [l.strip() for l in spine_out.splitlines() if l.strip()]
    spine_preview = "<br>".join(spine_lines[-20:]) if spine_lines else "(no output)"

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>NFL Sacks — {today_et} — Spine + Settle</title></head>
<body style="margin:0;padding:16px;background:#f4f4f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;font-size:13px;color:#1a1a1a">
<div style="max-width:700px;margin:0 auto;background:#fff;padding:24px;border-radius:8px;border:1px solid #e2e2e4">
  <h2 style="font-size:18px;margin:0 0 4px;color:#1d2d44">NFL Sacks — {today_et}</h2>
  <p style="color:#6b7280;font-size:12px;margin:0 0 20px">{datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')} &nbsp;·&nbsp; settle + spine rebuild complete</p>
  <h3 style="font-size:14px;margin:0 0 8px;color:#374151">Results</h3>
  {settle_html}
  <h3 style="font-size:14px;margin:16px 0 8px;color:#374151">Spine Rebuild</h3>
  <pre style="background:#f4f4f4;padding:10px;border-radius:4px;font-size:11px;overflow-x:auto;white-space:pre-wrap">{spine_preview}</pre>
  <p style="color:#6b7280;font-size:12px;margin-top:12px">
    9:00 AM ET scoring run will use today's rebuilt spine.
  </p>
</div>
</body>
</html>"""


def lambda_handler(event, context):
    os.environ.setdefault("HOME", "/tmp")
    root = _repo_root()
    today_et  = datetime.now(ET).strftime("%Y-%m-%d")
    yesterday = (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")
    topic_arn = os.environ.get("SNS_TOPIC_ARN", "").strip()

    nfl_season = int(os.environ.get("NFL_SEASON", _current_nfl_season()))
    mode       = _resolve_mode(event if isinstance(event, dict) else None)

    print(f"NFL Sacks Lambda | mode={mode} | date={today_et} | season={nfl_season}")

    scripts_dir = root / "src" / "nfl_sacks_modeling" / "scripts"

    step_results = []
    try:
        if mode == "settle_and_rebuild":
            # Step 1: Settle yesterday's bets (writes summary JSON to S3, no email)
            settle_cmd = [sys.executable, str(scripts_dir / "settle_sacks.py"),
                          "--gameday", yesterday]
            _run_capture(settle_cmd, cwd=root)
            step_results.append({"step": "settle", "status": "ok"})

            # Step 2: Rebuild spine
            spine_out = _run_capture(
                [sys.executable, str(scripts_dir / "update_spine.py"), "--season", str(nfl_season)],
                cwd=root,
            )
            step_results.append({"step": "spine_rebuild", "status": "ok"})

            # Step 3: Send Email 1 (settle summary + spine status)
            summary = _read_settle_summary()
            email1  = _build_email1_html(summary, spine_out, today_et)

            yw = summary.get("yesterday_wins", 0)  if summary else 0
            yl = summary.get("yesterday_losses", 0) if summary else 0
            yp = summary.get("yesterday_pushes", 0) if summary else 0
            y_pnl = summary.get("yesterday_pnl", 0.0) if summary else 0.0
            record_str = f"{yw}W–{yl}L–{yp}P" if (yw + yl + yp) > 0 else "no games"
            pnl_str    = (f"+{y_pnl:.2f}u" if y_pnl >= 0 else f"{y_pnl:.2f}u") if (yw + yl + yp) > 0 else ""

            _send_email(
                f"NFL Sacks — {today_et} — Spine rebuilt · {record_str} yesterday{' · ' + pnl_str if pnl_str else ''}",
                email1,
            )

        elif mode == "pipeline":
            gameday = str(event.get("gameday", today_et))
            _run_capture(
                [sys.executable, str(scripts_dir / "run_pipeline.py"), "--gameday", gameday],
                cwd=root,
            )
            step_results.append({"step": "pipeline", "status": "ok"})
            # run_pipeline.py sends Email 2 (plays + yesterday results + all-time) via SES

        elif mode == "spine_update":
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

        elif mode == "settle":
            settle_cmd = [sys.executable, str(scripts_dir / "settle_sacks.py")]
            if isinstance(event, dict) and "gameday" in event:
                settle_cmd += ["--gameday", str(event["gameday"])]
            _run_capture(settle_cmd, cwd=root)
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
