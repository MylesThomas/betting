"""
Lambda: nba-dispersion-daily
=============================
Fires daily (EventBridge 10:05 AM ET — props collected at 10:00 AM). Detects qualifying UNDER bets on non-star
teammates after a star night, writes today_plays.csv to S3, sends SES email.

Env vars:
  SES_SOURCE   — verified SES sender identity (e.g. noreply@yourdomain.com)
  SES_TO       — comma-separated recipient list (e.g. mylescgthomas@gmail.com)
  SETTLE_BUCKET — S3 bucket for outputs (default: nba-betting-mt)
  DISPERSION_PREFIX — S3 prefix for outputs (default: dispersion)

IAM requires: s3:GetObject on nba-api-mt/* and the-odds-api-mt/*,
              s3:PutObject on nba-betting-mt/dispersion/*,
              ses:SendEmail.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import pandas as pd

ET = ZoneInfo("America/New_York")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dispersion_signal import (  # noqa: E402  (same-directory import)
    compute_todays_plays,
    current_season,
    load_lines_season,
    load_logs,
    load_props_today,
)
from email_renderer import render_html, render_subject, render_text  # noqa: E402

OUTPUT_BUCKET = os.environ.get("SETTLE_BUCKET", "nba-betting-mt")
OUTPUT_PREFIX = os.environ.get("DISPERSION_PREFIX", "dispersion")


# =============================================================================
# S3 WRITE
# =============================================================================

def write_plays_s3(plays: pd.DataFrame, today: "date") -> str:
    key = f"{OUTPUT_PREFIX}/today_plays.csv"
    csv_buf = StringIO()
    plays.to_csv(csv_buf, index=False)
    boto3.client("s3").put_object(
        Bucket=OUTPUT_BUCKET,
        Key=key,
        Body=csv_buf.getvalue(),
        ContentType="text/csv",
    )
    uri = f"s3://{OUTPUT_BUCKET}/{key}"
    print(f"  Wrote {len(plays)} rows → {uri}")
    return uri


# =============================================================================
# EMAIL
# =============================================================================

def send_email(plays: pd.DataFrame, skipped: pd.DataFrame, today: "date") -> str | None:
    ses_source = os.environ.get("SES_SOURCE", "").strip()
    ses_to_raw = os.environ.get("SES_TO", "").strip()
    if not ses_source or not ses_to_raw:
        print("  SES_SOURCE / SES_TO not set — skipping email.")
        return None

    to_list = [a.strip() for a in ses_to_raw.split(",") if a.strip()]
    subject = render_subject(plays, today)
    text_body = render_text(plays, skipped, today)
    html_body = render_html(plays, skipped, today)

    resp = boto3.client("ses").send_email(
        Source=ses_source,
        Destination={"ToAddresses": to_list},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {
                "Text": {"Data": text_body, "Charset": "UTF-8"},
                "Html": {"Data": html_body, "Charset": "UTF-8"},
            },
        },
    )
    msg_id = resp["MessageId"]
    print(f"  Email sent → {to_list}  (MessageId: {msg_id})")
    return msg_id


# =============================================================================
# HANDLER
# =============================================================================

def handler(event: dict, context) -> dict:
    today = datetime.now(ET).date()
    season = current_season(today)

    print("=" * 52)
    print(f"NBA DISPERSION DAILY — {today}  (season {season})")
    print("=" * 52)

    # Load data
    logs = load_logs(season)
    props_today = load_props_today(season, today)
    lines = load_lines_season(season)

    # Compute plays
    print("\nComputing plays ...")
    if props_today is None or props_today.empty:
        print("  No props available for today — sending no-plays email.")
        plays = pd.DataFrame()
        skipped = pd.DataFrame(columns=["team", "player"])
    else:
        plays, skipped = compute_todays_plays(logs, props_today, lines, today)

    # Write CSV to S3 (always — even empty, so Streamlit always has a fresh file)
    s3_uri = write_plays_s3(plays, today)

    # Send email
    send_email(plays, skipped, today)

    n = len(plays)
    print(f"\nDone — {n} play{'s' if n != 1 else ''} today.")
    return {"statusCode": 200, "plays": n, "s3_uri": s3_uri}
