"""
Settle MLB pitcher outs bets for a given gameday and send summary email.

Settlement logic (UNDER outs_recorded):
  Win : outs_recorded < line  → +(under_price − 1) units
  Loss: outs_recorded > line  → −1 unit
  Push: outs_recorded == line → 0 units

Actuals come from the existing pitcher gamelogs on S3:
  s3://the-odds-api-mt/mlb/strikeouts_model/pitcher_gamelogs/{season}/

Reads from S3:
  mlb/pitcher_outs_model/daily_runs/{gameday}/recommendations.csv

Writes/updates S3:
  mlb/pitcher_outs_model/daily_runs/{gameday}/settled.csv
  mlb/pitcher_outs_model/settled/settled_bets.parquet  (appended)

Sends SES HTML email with P&L summary.

Usage:
  python src/mlb_pitcher_outs_modeling/scripts/settle_pitcher_outs.py
  python src/mlb_pitcher_outs_modeling/scripts/settle_pitcher_outs.py --gameday 2026-07-06
"""
from __future__ import annotations

import argparse
import html as html_module
import os
import re
import sys
import unicodedata
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import botocore.exceptions
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET    = "the-odds-api-mt"
DAILY_PREFIX = "mlb/pitcher_outs_model/daily_runs"
SETTLED_KEY  = "mlb/pitcher_outs_model/settled/settled_bets.parquet"
GAMELOG_BUCKET  = "the-odds-api-mt"
GAMELOG_PREFIX  = "mlb/strikeouts_model/pitcher_gamelogs"

SES_SOURCE    = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()

ET = ZoneInfo("America/New_York")

_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"

NAME_MAP = {
    "louie varland": "louis varland",
    "luis l ortiz":  "luis ortiz",
}


def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def yesterday_et() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.lower()
    n = unicodedata.normalize("NFD", n)
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")
    n = re.sub(r"[.,'\-]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    n = " ".join(n.split())
    return NAME_MAP.get(n, n)


def ip_to_outs(ip_val) -> float | None:
    """Convert MLB innings_pitched float (e.g. 5.2 = 5 full + 2 extra outs = 17 outs)."""
    if ip_val is None or (isinstance(ip_val, float) and np.isnan(ip_val)):
        return None
    try:
        ip = float(ip_val)
        full = int(ip)
        frac_digit = round((ip - full) * 10)
        return float(full * 3 + frac_digit)
    except (TypeError, ValueError):
        return None


# ─── S3 ──────────────────────────────────────────────────────────────────────

def _s3():
    return boto3.client("s3")


def s3_get_parquet(key: str) -> pd.DataFrame:
    body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def s3_put_parquet(key: str, df: pd.DataFrame) -> None:
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())


def s3_put_csv(key: str, df: pd.DataFrame) -> None:
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=df.to_csv(index=False).encode())


def load_gamelogs_for_date(game_date: str) -> pd.DataFrame:
    """Load pitcher gamelogs for the given date from S3 (reusing strikeouts gamelogs)."""
    season = int(game_date[:4])
    prefix = f"{GAMELOG_PREFIX}/{season}/"
    s3 = _s3()
    resp = s3.list_objects_v2(Bucket=GAMELOG_BUCKET, Prefix=prefix)
    if not resp.get("Contents"):
        return pd.DataFrame()

    frames = []
    for obj in resp["Contents"]:
        key = obj["Key"]
        if not key.endswith(".parquet"):
            continue
        try:
            body = s3.get_object(Bucket=GAMELOG_BUCKET, Key=key)["Body"].read()
            df = pd.read_parquet(BytesIO(body))
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    gamelogs = pd.concat(frames, ignore_index=True)
    gamelogs["game_date_str"] = gamelogs["game_date"].astype(str)

    # Filter to target date
    daily = gamelogs[gamelogs["game_date_str"] == game_date].copy()
    if daily.empty:
        return pd.DataFrame()

    daily["player_key"]  = daily["player_name"].map(normalize_name)
    daily["outs_actual"] = daily["innings_pitched"].map(ip_to_outs)
    return daily[["player_key", "game_date_str", "outs_actual"]].dropna(subset=["outs_actual"])


# ─── Email ────────────────────────────────────────────────────────────────────

def build_settle_email(settled: pd.DataFrame, gameday: str, all_time_stats: dict) -> str:
    he = html_module.escape

    def fmt(v, fmt_str):
        try:
            return format(float(v), fmt_str)
        except (TypeError, ValueError):
            return "—"

    day_plays = settled[settled.get("tier","play") == "play"] if "tier" in settled.columns else settled
    n_win  = int((day_plays["pnl"] > 0).sum())
    n_loss = int((day_plays["pnl"] < 0).sum())
    n_push = int((day_plays["pnl"] == 0).sum())
    day_units = day_plays["pnl"].sum() if not day_plays.empty else 0.0

    rows_html = ""
    for _, r in settled.sort_values("player_key").iterrows():
        pnl = r.get("pnl", np.nan)
        if isinstance(pnl, float) and pnl > 0:
            bg = "background:#eaf6ea"
            outcome = "<span style='color:#276221;font-weight:bold'>WIN</span>"
        elif isinstance(pnl, float) and pnl < 0:
            bg = "background:#fdecea"
            outcome = "<span style='color:#c0392b;font-weight:bold'>LOSS</span>"
        else:
            bg = ""
            outcome = "<span style='color:#888'>PUSH</span>"

        rows_html += (
            f"<tr style='{bg}'>"
            f"<td>{he(str(r.get('player_name', r.get('player_key',''))))}</td>"
            f"<td style='text-align:center'>{he(str(r.get('bookmaker','')))}</td>"
            f"<td style='text-align:center'>{fmt(r.get('line'), '.1f')}</td>"
            f"<td style='text-align:center;font-weight:bold;color:#1d4ed8'>{fmt(r.get('under_price'), '.2f')}</td>"
            f"<td style='text-align:center'>{fmt(r.get('outs_actual'), '.0f')}</td>"
            f"<td style='text-align:center;font-weight:bold'>{fmt(pnl, '+.2f')}u</td>"
            f"<td style='text-align:center'>{outcome}</td>"
            f"</tr>\n"
        )

    atw  = all_time_stats.get("wins", 0)
    atl  = all_time_stats.get("losses", 0)
    atu  = all_time_stats.get("units", 0.0)
    atr  = all_time_stats.get("roi", 0.0)

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<style>
  body {{font-family:{_SANS};color:#222;max-width:900px;margin:auto;padding:20px}}
  h2 {{color:#2c3e50;margin-bottom:4px}}
  table {{border-collapse:collapse;width:100%;margin-top:8px}}
  th {{background:#2c3e50;color:#fff;padding:7px 8px;text-align:left;font-size:12px;white-space:nowrap}}
  td {{padding:5px 8px;border-bottom:1px solid #e0e0e0;font-size:12px}}
</style>
</head><body>
<h2>MLB Pitcher Outs Settlement — {gameday}</h2>
<p>Plays: {n_win}W – {n_loss}L – {n_push}P &nbsp;·&nbsp; <strong>{day_units:+.2f}u</strong></p>
<p>All-time: {atw}W – {atl}L &nbsp;·&nbsp; {atu:+.2f}u &nbsp;·&nbsp; ROI: {atr*100:+.1f}%</p>
<table>
  <tr>
    <th>Pitcher</th><th style='text-align:center'>Book</th>
    <th style='text-align:center'>Line</th><th style='text-align:center'>Under Odds</th>
    <th style='text-align:center'>Outs</th><th style='text-align:center'>P&amp;L</th>
    <th style='text-align:center'>Result</th>
  </tr>
  {rows_html}
</table>
</body></html>"""


def send_ses(subject: str, html_body: str) -> None:
    if not SES_SOURCE or not SES_TO_RAW:
        print("  SES not configured — skipping email")
        return
    to_list = [e.strip() for e in SES_TO_RAW.split(",") if e.strip()]
    boto3.client("ses", region_name="us-east-2").send_email(
        Source=SES_SOURCE,
        Destination={"ToAddresses": to_list},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {"Html": {"Data": html_body, "Charset": "UTF-8"}},
        },
    )
    print(f"  Email sent to {to_list}")


def publish_sns(subject: str, message: str) -> None:
    if not SNS_TOPIC_ARN:
        return
    boto3.client("sns").publish(TopicArn=SNS_TOPIC_ARN, Subject=subject[:100], Message=message)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=yesterday_et(),
                        help="Gameday to settle (default: yesterday ET)")
    args = parser.parse_args()
    gameday = args.gameday

    print(f"MLB Pitcher Outs settle | gameday={gameday}")

    # Load recommendations
    recs_key = f"{DAILY_PREFIX}/{gameday}/recommendations.csv"
    try:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=recs_key)["Body"].read()
        recs = pd.read_csv(BytesIO(body))
        print(f"  Loaded {len(recs)} recommendations from S3")
    except _s3().exceptions.NoSuchKey:
        print(f"  No recommendations found for {gameday} — nothing to settle")
        return
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            print(f"  No recommendations found for {gameday}")
            return
        raise

    if recs.empty:
        print("  Recommendations file is empty — nothing to settle")
        return

    # Load actuals from gamelogs
    print(f"  Loading gamelogs for {gameday} ...")
    actuals = load_gamelogs_for_date(gameday)
    print(f"  Actuals found: {len(actuals)} pitcher-games")

    if actuals.empty:
        print(f"  ⚠️  No gamelogs found for {gameday} — try again later")
        return

    # Join
    recs["player_key"] = recs["player_key"].map(normalize_name) if "player_key" in recs.columns else recs["player_name"].map(normalize_name)
    actuals_map = actuals.set_index("player_key")["outs_actual"].to_dict()
    recs["outs_actual"] = recs["player_key"].map(actuals_map)

    # Settlement
    n_matched = recs["outs_actual"].notna().sum()
    print(f"  Matched actuals: {n_matched}/{len(recs)}")

    def settle_row(row):
        if pd.isna(row["outs_actual"]):
            return np.nan
        outs = float(row["outs_actual"])
        line = float(row["line"])
        if outs < line:
            return float(row["under_price"]) - 1.0   # win
        elif outs > line:
            return -1.0                                # loss
        else:
            return 0.0                                 # push

    recs["pnl"] = recs.apply(settle_row, axis=1)
    recs["game_date"] = gameday

    # Stats
    settled = recs[recs["pnl"].notna()].copy()
    if settled.empty:
        print("  No settled bets (actuals not available yet)")
        return

    n_win  = int((settled["pnl"] > 0).sum())
    n_loss = int((settled["pnl"] < 0).sum())
    n_push = int((settled["pnl"] == 0).sum())
    day_u  = float(settled["pnl"].sum())
    print(f"\n  Result: {n_win}W – {n_loss}L – {n_push}P = {day_u:+.2f}u")

    # Save daily settled
    settled_day_key = f"{DAILY_PREFIX}/{gameday}/settled.csv"
    s3_put_csv(settled_day_key, settled)
    print(f"  Daily settled → s3://{S3_BUCKET}/{settled_day_key}")

    # Append to all-time parquet
    try:
        all_time = s3_get_parquet(SETTLED_KEY)
    except Exception:
        all_time = pd.DataFrame()

    # Remove any prior settlement for this gameday (idempotent)
    if not all_time.empty and "game_date" in all_time.columns:
        all_time = all_time[all_time["game_date"].astype(str) != gameday]

    all_time = pd.concat([all_time, settled], ignore_index=True)
    s3_put_parquet(SETTLED_KEY, all_time)
    print(f"  All-time settled → s3://{S3_BUCKET}/{SETTLED_KEY} ({len(all_time)} rows)")

    # All-time stats
    plays = all_time if "tier" not in all_time.columns else all_time[all_time["tier"] == "play"]
    at_stats = {
        "wins":   int((plays["pnl"] > 0).sum()),
        "losses": int((plays["pnl"] < 0).sum()),
        "units":  float(plays["pnl"].sum()),
        "roi":    float(plays["pnl"].mean()) if len(plays) else 0.0,
    }

    # Email
    subject = (
        f"MLB Pitcher Outs settled — {n_win}W {n_loss}L {n_push}P "
        f"{day_u:+.2f}u — {gameday}"
    )
    html_body = build_settle_email(settled, gameday, at_stats)
    send_ses(subject, html_body)


if __name__ == "__main__":
    main()
