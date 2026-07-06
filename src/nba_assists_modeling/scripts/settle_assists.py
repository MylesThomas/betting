"""
Settle NBA assists bets for a given gameday and update S3 history.

Settlement logic (OVER · line is x.5 or integer):
  Win  : AST > offered_line
  Loss : AST < offered_line
  Push : AST == offered_line

Reads:
  s3://the-odds-api-mt/nba/assists_model/daily_runs/{gameday}/recommendations.csv
  s3://nba-api-mt/player_game_logs/{season}/{gameday}.csv

Writes / appends:
  s3://the-odds-api-mt/nba/assists_model/settled/settled_bets.parquet

Run:
    python src/nba_assists_modeling/scripts/settle_assists.py
    python src/nba_assists_modeling/scripts/settle_assists.py --gameday 2026-11-01
"""
from __future__ import annotations

import argparse
import html as html_module
import os
import sys
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

S3_BUCKET     = "the-odds-api-mt"
S3_PREFIX     = "nba/assists_model"
GL_BUCKET     = "nba-api-mt"
GL_PREFIX     = "player_game_logs"
SES_SOURCE    = os.environ.get("SETTLEMENT_SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SETTLEMENT_SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()

ET    = ZoneInfo("America/New_York")
_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"


def yesterday_et() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")


def current_nba_season(gameday: str) -> str:
    yr = int(gameday[:4])
    mo = int(gameday[5:7])
    yr = yr if mo >= 10 else yr - 1
    return f"{yr}-{str(yr+1)[-2:]}"


def _normalize_name(name: str) -> str:
    import unicodedata, re
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name.strip().lower())
    return re.sub(r"\s+", " ", name).strip()


def _s3():
    return boto3.client("s3")


def load_recommendations(gameday: str) -> pd.DataFrame | None:
    key = f"{S3_PREFIX}/daily_runs/{gameday}/recommendations.csv"
    try:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        return pd.read_csv(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def load_game_logs(gameday: str) -> pd.DataFrame:
    season = current_nba_season(gameday)
    key = f"{GL_PREFIX}/{season}/{gameday}.csv"
    try:
        body = _s3().get_object(Bucket=GL_BUCKET, Key=key)["Body"].read()
        df = pd.read_csv(BytesIO(body))
        df["player_key"] = df["PLAYER_NAME"].apply(_normalize_name)
        df["GAME_DATE"]  = gameday
        return df[["player_key", "PLAYER_NAME", "AST", "MIN", "GAME_DATE"]]
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            print(f"  No game log for {gameday} (season={season}) — may not have played yet")
            return pd.DataFrame()
        raise


def load_settled() -> pd.DataFrame:
    key = f"{S3_PREFIX}/settled/settled_bets.parquet"
    try:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return pd.DataFrame()
        raise


def save_settled(df: pd.DataFrame) -> None:
    key = f"{S3_PREFIX}/settled/settled_bets.parquet"
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    print(f"  Saved settled → s3://{S3_BUCKET}/{key}")


def settle(recs: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Join recommendations with actuals, compute outcome and PnL."""
    df = recs.merge(
        logs[["player_key", "AST", "MIN"]].rename(columns={"AST": "actual_ast", "MIN": "actual_min"}),
        on="player_key",
        how="left",
    )
    df["actual_ast"] = pd.to_numeric(df["actual_ast"], errors="coerce")
    df["line"]       = pd.to_numeric(df["consensus_line"], errors="coerce")

    def outcome(row):
        if pd.isna(row["actual_ast"]):
            return "DNP"
        if row["actual_ast"] > row["line"]:
            return "WIN"
        if row["actual_ast"] == row["line"]:
            return "PUSH"
        return "LOSS"

    df["outcome"] = df.apply(outcome, axis=1)

    def pnl(row):
        if row["outcome"] == "WIN":
            profit = row.get("best_over_odds", -110)
            return (profit / 100.0) if profit >= 0 else (100.0 / abs(profit))
        if row["outcome"] in ("PUSH", "DNP"):
            return 0.0
        return -1.0

    df["pnl"]      = df.apply(pnl, axis=1)
    df["is_hit"]   = (df["outcome"] == "WIN").astype(int)
    df["settled_at"] = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    return df


def fmt_odds(price) -> str:
    try:
        return f"{int(float(price)):+d}"
    except Exception:
        return "—"


def _card(label: str, value: str, color: str = "#e2e8f0", size: str = "22px") -> str:
    return f"""<div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;padding:14px 20px;min-width:120px;">
  <div style="font-size:11px;color:#6b7280;text-transform:uppercase;">{label}</div>
  <div style="font-size:{size};font-weight:700;color:{color};">{value}</div>
</div>"""


def build_settlement_html(df_today: pd.DataFrame, df_season: pd.DataFrame, gameday: str) -> str:
    now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")

    def _stats(df: pd.DataFrame) -> dict:
        wins    = (df["outcome"] == "WIN").sum()
        losses  = (df["outcome"] == "LOSS").sum()
        pushes  = (df["outcome"] == "PUSH").sum()
        dnps    = (df["outcome"] == "DNP").sum()
        pnl     = df["pnl"].sum()
        primary = df[df.get("is_primary", pd.Series(False, index=df.index)).astype(bool)]["pnl"].sum()
        decided = wins + losses
        return {"wins": int(wins), "losses": int(losses), "pushes": int(pushes),
                "dnps": int(dnps), "pnl": float(pnl), "primary_pnl": float(primary),
                "roi": pnl / decided if decided > 0 else 0.0}

    td = _stats(df_today)
    ss = _stats(df_season) if not df_season.empty else td

    def _pnl_color(v: float) -> str:
        return "#4ade80" if v >= 0 else "#f87171"

    def color(outcome: str) -> str:
        return {"WIN": "#4ade80", "LOSS": "#f87171", "PUSH": "#fbbf24", "DNP": "#9ca3af"}.get(outcome, "#e2e8f0")

    rows_html = ""
    for _, row in df_today.sort_values("outcome").iterrows():
        oc = row["outcome"]
        rows_html += f"""
        <tr>
          <td style="padding:6px 10px;font-weight:600;">{html_module.escape(str(row.get('player','—')))}</td>
          <td style="padding:6px 10px;text-align:center;">{row.get('consensus_line','—')}</td>
          <td style="padding:6px 10px;text-align:center;font-family:{_MONO};">{fmt_odds(row.get('best_over_odds'))}</td>
          <td style="padding:6px 10px;text-align:center;font-family:{_MONO};">{"—" if pd.isna(row.get('actual_ast')) else int(row['actual_ast'])}</td>
          <td style="padding:6px 10px;text-align:center;font-family:{_MONO};">+{row.get('edge',0)*100:.1f}pp</td>
          <td style="padding:6px 10px;text-align:center;font-weight:bold;color:{color(oc)};">{oc}</td>
          <td style="padding:6px 10px;text-align:center;font-family:{_MONO};color:{_pnl_color(row['pnl'])};">{row['pnl']:+.2f}u</td>
        </tr>"""

    td_push   = f"–{td['pushes']}P" if td['pushes'] else ""
    ss_push   = f"–{ss['pushes']}P" if ss['pushes'] else ""
    td_record = f"{td['wins']}W–{td['losses']}L{td_push}"
    ss_record = f"{ss['wins']}W–{ss['losses']}L{ss_push}"

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/></head>
<body style="font-family:{_SANS};background:#0f1117;color:#e2e8f0;margin:0;padding:24px;">
<h2 style="color:#93c5fd;margin-bottom:4px;">NBA Assists Settlement — {gameday}</h2>
<p style="color:#6b7280;font-size:13px;margin-top:0;">{now_str}</p>

<p style="color:#9ca3af;font-size:12px;margin:8px 0 4px;">Today</p>
<div style="display:flex;gap:16px;margin-bottom:20px;flex-wrap:wrap;">
  {_card("PnL", f"{td['pnl']:+.2f}u", _pnl_color(td['pnl']))}
  {_card("Primary PnL", f"{td['primary_pnl']:+.2f}u", _pnl_color(td['primary_pnl']))}
  {_card("Record", td_record, size="18px")}
  {_card("ROI", f"{td['roi']*100:+.1f}%" if td['wins']+td['losses'] > 0 else "—", _pnl_color(td['roi']))}
</div>

<p style="color:#9ca3af;font-size:12px;margin:8px 0 4px;">Season to Date</p>
<div style="display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap;">
  {_card("PnL", f"{ss['pnl']:+.2f}u", _pnl_color(ss['pnl']))}
  {_card("Primary PnL", f"{ss['primary_pnl']:+.2f}u", _pnl_color(ss['primary_pnl']))}
  {_card("Record", ss_record, size="18px")}
  {_card("ROI", f"{ss['roi']*100:+.1f}%" if ss['wins']+ss['losses'] > 0 else "—", _pnl_color(ss['roi']))}
</div>

<table style="border-collapse:collapse;width:100%;font-size:13px;">
  <tr style="background:#1e3a5f;">
    <th style="padding:8px 10px;text-align:left;color:#93c5fd;">Player</th>
    <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Line</th>
    <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Odds</th>
    <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Actual AST</th>
    <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Edge</th>
    <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Outcome</th>
    <th style="padding:8px 10px;text-align:center;color:#93c5fd;">PnL</th>
  </tr>
  {rows_html}
</table>
</body></html>"""


def send_email(subject: str, html_body: str) -> None:
    to_list = [a.strip() for a in SES_TO_RAW.split(",") if a.strip()]
    if not SES_SOURCE or not to_list:
        print(f"  SES not configured, skipping email")
        return
    try:
        boto3.client("ses", region_name="us-east-2").send_email(
            Source=SES_SOURCE,
            Destination={"ToAddresses": to_list},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Html": {"Data": html_body, "Charset": "UTF-8"},
                         "Text": {"Data": subject, "Charset": "UTF-8"}},
            },
        )
        print(f"  Email sent: {subject}")
    except Exception as e:
        print(f"  Email failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=yesterday_et())
    args = parser.parse_args()
    gameday = args.gameday

    print(f"\nNBA Assists Settlement | gameday={gameday}", flush=True)

    recs = load_recommendations(gameday)
    if recs is None:
        print(f"  No recommendations found for {gameday} — nothing to settle")
        return
    print(f"  Recommendations: {len(recs)}")

    logs = load_game_logs(gameday)
    if logs.empty:
        print(f"  No game logs yet for {gameday} — retrying later")
        return

    settled_new = settle(recs, logs)
    print(f"  Settled: {len(settled_new)}")
    print(settled_new[["player", "consensus_line", "actual_ast", "outcome", "pnl"]].to_string(index=False))

    # Append to history
    settled_all = load_settled()
    if not settled_all.empty:
        # Remove any existing rows for this gameday before appending
        settled_all = settled_all[settled_all["game_date"] != gameday]
    settled_all = pd.concat([settled_all, settled_new], ignore_index=True)
    save_settled(settled_all)

    # Summary stats (all time)
    total_pnl = settled_all["pnl"].sum()
    total_bets = len(settled_all[settled_all["outcome"] != "DNP"])
    print(f"\n  All-time: {total_bets} bets · {total_pnl:+.2f}u cumulative PnL")

    wins = (settled_new["outcome"] == "WIN").sum()
    losses = (settled_new["outcome"] == "LOSS").sum()
    day_pnl = settled_new["pnl"].sum()

    season = current_nba_season()
    season_col = settled_all.get("season", pd.Series("", index=settled_all.index))
    settled_season = settled_all[season_col == season] if "season" in settled_all.columns else settled_all

    subject = f"NBA Assists Settlement {gameday} — {wins}W/{losses}L · {day_pnl:+.2f}u"
    html_body = build_settlement_html(settled_new, settled_season, gameday)
    send_email(subject, html_body)
    print("Done.")


if __name__ == "__main__":
    main()
