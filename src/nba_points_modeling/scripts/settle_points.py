"""
Settle NBA player points bets for a given gameday and update S3 history.

Settlement logic (UNDER · line is x.5 or integer):
  Win  : PTS < offered_line
  Loss : PTS > offered_line
  Push : PTS == offered_line

Reads:
  s3://the-odds-api-mt/nba/points_model/daily_runs/{gameday}/recommendations.csv
  s3://nba-api-mt/player_game_logs/{season}/{gameday}.csv

Writes / appends:
  s3://the-odds-api-mt/nba/points_model/settled/settled_bets.parquet

Run:
    python src/nba_points_modeling/scripts/settle_points.py
    python src/nba_points_modeling/scripts/settle_points.py --gameday 2026-11-01
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
S3_PREFIX     = "nba/points_model"
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
        df["game_date"]  = gameday
        return df[["player_key", "PLAYER_NAME", "PTS", "MIN", "game_date"]]
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            print(f"  No game log for {gameday} (season={season}) — may not be available yet")
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
    """Join recommendations with actuals and classify each bet."""
    bets = recs[recs.get("bet", pd.Series(True, index=recs.index)).astype(bool)].copy()

    merged = bets.merge(
        logs[["player_key", "PTS"]].rename(columns={"PTS": "pts_actual"}),
        on="player_key",
        how="left",
    )
    merged["pts_actual"] = merged["pts_actual"].where(merged["MIN"].notna() if "MIN" in merged else pd.Series(True, index=merged.index), other=np.nan) if "MIN" in merged else merged["pts_actual"]

    def _settle_row(row):
        pts = row.get("pts_actual")
        line = row.get("offered_line")
        if pd.isna(pts) or pd.isna(line):
            return "DNP"
        pts  = float(pts)
        line = float(line)
        # Direction is always UNDER
        if pts < line:
            return "WIN"
        elif pts > line:
            return "LOSS"
        else:
            return "PUSH"

    merged["result"] = merged.apply(_settle_row, axis=1)

    def _pnl(row):
        if row["result"] == "WIN":
            p = float(row.get("p_market_under", 0.5))
            am = -(p / (1 - p) * 100) if p >= 0.5 else (1 - p) / p * 100
            return am / 100.0 if am >= 0 else 100.0 / abs(am)
        elif row["result"] == "LOSS":
            return -1.0
        return 0.0

    merged["pnl_units"] = merged.apply(_pnl, axis=1)
    return merged


def _stat_card(label: str, value: str, color: str = "#111") -> str:
    return f"""<div style="background:#f0f7ff;border:1px solid #bfdbfe;border-radius:6px;padding:10px 16px;min-width:100px;">
  <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">{label}</div>
  <div style="font-size:20px;font-weight:700;color:{color};">{value}</div>
</div>"""


def build_settlement_html(settled_today: pd.DataFrame, settled_season: pd.DataFrame, gameday: str) -> str:
    now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")

    def _stats(df: pd.DataFrame) -> dict:
        wins   = (df["result"] == "WIN").sum()
        losses = (df["result"] == "LOSS").sum()
        pushes = df.get("result", pd.Series()).eq("PUSH").sum()
        dnps   = df.get("result", pd.Series()).eq("DNP").sum()
        units  = df["pnl_units"].sum() if "pnl_units" in df.columns else 0.0
        decided = wins + losses
        return {"wins": int(wins), "losses": int(losses), "pushes": int(pushes),
                "dnps": int(dnps), "units": float(units),
                "win_pct": wins / decided if decided > 0 else 0.0,
                "roi": units / decided if decided > 0 else 0.0}

    td = _stats(settled_today)
    ss = _stats(settled_season) if not settled_season.empty else td

    td_color = "#16a34a" if td["units"] >= 0 else "#dc2626"
    ss_color = "#16a34a" if ss["units"] >= 0 else "#dc2626"

    decided_td = td["wins"] + td["losses"]

    def _td_cell(val, mono=True, bold=False, color=None):
        style = "padding:6px 10px;text-align:center;"
        if mono:
            style += f"font-family:{_MONO};font-size:12px;"
        if bold:
            style += "font-weight:bold;"
        if color:
            style += f"color:{color};"
        return f'<td style="{style}">{html_module.escape(str(val))}</td>'

    rows_html = ""
    for _, row in settled_today.sort_values("result").iterrows():
        result_color = {"WIN": "#16a34a", "LOSS": "#dc2626", "PUSH": "#6b7280", "DNP": "#9ca3af"}.get(row["result"], "#111")
        rows_html += f"""<tr>
          {_td_cell(str(row.get("player_name", row.get("player_key",""))), mono=False, bold=True)}
          {_td_cell(f"{row.get('offered_line','—'):.1f}" if pd.notna(row.get('offered_line')) else "—")}
          {_td_cell(f"{row.get('pts_actual','—'):.0f}" if pd.notna(row.get('pts_actual')) else "DNP")}
          {_td_cell(row["result"], bold=True, color=result_color)}
          {_td_cell(f"{row['pnl_units']:+.2f}u")}
          {_td_cell(f"{row.get('edge_under',0)*100:.1f}pp" if pd.notna(row.get('edge_under')) else "—")}
          {_td_cell(f"{row.get('p_model_under',0)*100:.1f}%" if pd.notna(row.get('p_model_under')) else "—")}
          {_td_cell(f"{row.get('p_market_under',0)*100:.1f}%" if pd.notna(row.get('p_market_under')) else "—")}
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family:{_SANS}; font-size:14px; color:#111; background:#fff; margin:20px; }}
    h2 {{ color:#1d4ed8; margin-bottom:4px; }}
    h3 {{ color:#374151; margin:20px 0 6px; font-size:14px; }}
    table {{ border-collapse:collapse; width:100%; }}
    th {{ background:#1d4ed8; color:#fff; padding:8px 10px; text-align:center; font-size:12px; }}
    td {{ border-bottom:1px solid #e5e7eb; }}
  </style>
</head>
<body>
  <h2>NBA Player Points — Settlement Report</h2>
  <p style="color:#6b7280;margin-top:0;">{gameday} &nbsp;·&nbsp; {now_str}</p>

  <h3>Today</h3>
  <div style="display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;">
    {_stat_card("PnL", f"{td['units']:+.2f}u", td_color)}
    {_stat_card("Record", f"{td['wins']}W–{td['losses']}L")}
    {_stat_card("Win %", f"{td['win_pct']*100:.1f}%" if decided_td > 0 else "—")}
    {_stat_card("ROI", f"{td['roi']*100:+.1f}%" if decided_td > 0 else "—", td_color)}
  </div>

  <h3>Season to Date</h3>
  <div style="display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;">
    {_stat_card("PnL", f"{ss['units']:+.2f}u", ss_color)}
    {_stat_card("Record", f"{ss['wins']}W–{ss['losses']}L")}
    {_stat_card("Win %", f"{ss['win_pct']*100:.1f}%" if ss['wins']+ss['losses'] > 0 else "—")}
    {_stat_card("ROI", f"{ss['roi']*100:+.1f}%" if ss['wins']+ss['losses'] > 0 else "—", ss_color)}
  </div>

  <table>
    <thead>
      <tr>
        <th>Player</th><th>Line</th><th>Actual PTS</th><th>Result</th>
        <th>PnL</th><th>Edge</th><th>P(und) model</th><th>P(und) mkt</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  <p style="font-size:11px;color:#9ca3af;">Strategy S3 — UNDER only · shrinkage=0.25 · edge≥5pp · fav_only</p>
</body>
</html>"""


def _send_email(subject: str, html_body: str) -> None:
    if not SES_SOURCE:
        print("  SES_SOURCE not set — skipping email")
        return
    recipients = [r.strip() for r in SES_TO_RAW.split(",") if r.strip()]
    boto3.client("ses", region_name="us-east-1").send_email(
        Source=SES_SOURCE,
        Destination={"ToAddresses": recipients},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body":    {"Html": {"Data": html_body, "Charset": "UTF-8"}},
        },
    )
    print(f"  Email sent to {recipients}")


def _publish_sns(subject: str, message: str) -> None:
    if not SNS_TOPIC_ARN:
        return
    boto3.client("sns").publish(TopicArn=SNS_TOPIC_ARN, Subject=subject[:100], Message=message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=None,
                        help="Date to settle (YYYY-MM-DD). Default: yesterday ET.")
    args = parser.parse_args()
    gameday = args.gameday or yesterday_et()

    print(f"NBA Points Settlement | gameday={gameday}", flush=True)

    recs = load_recommendations(gameday)
    if recs is None:
        print(f"  No recommendations found for {gameday} — nothing to settle.")
        return
    bets = recs[recs.get("bet", pd.Series(True, index=recs.index)).astype(bool)] if "bet" in recs.columns else recs
    print(f"  Recommendations: {len(recs):,} rows  Bets: {len(bets):,}")

    logs = load_game_logs(gameday)
    if logs.empty:
        print(f"  No game logs — settlement deferred.")
        return
    print(f"  Game logs: {len(logs):,} rows")

    settled = settle(recs, logs)
    wins   = (settled["result"] == "WIN").sum()
    losses = (settled["result"] == "LOSS").sum()
    units  = settled["pnl_units"].sum()
    print(f"  {wins}W {losses}L  {units:+.2f}u")

    existing = load_settled()
    if not existing.empty:
        already = set(zip(existing.get("player_key", []), existing.get("game_date", [])))
        new_rows = settled[~settled.apply(
            lambda r: (r.get("player_key"), gameday) in already, axis=1
        )]
    else:
        new_rows = settled

    if new_rows.empty:
        print("  All rows already settled — no update needed.")
        updated = existing
    else:
        updated = pd.concat([existing, new_rows], ignore_index=True)
        save_settled(updated)

    season = current_nba_season(gameday)
    settled_season = updated[updated.get("season", pd.Series("", index=updated.index)) == season] if "season" in updated.columns else updated

    html_body = build_settlement_html(settled, settled_season, gameday)
    subject   = f"NBA Points settlement {gameday} — {wins}W {losses}L {units:+.2f}u"

    _send_email(subject, html_body)
    _publish_sns(subject, f"{wins}W {losses}L {units:+.2f}u on {gameday}.")
    print("Done.")


if __name__ == "__main__":
    main()
