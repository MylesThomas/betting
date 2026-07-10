"""
Settle NFL WR/TE receiving yards bets for a given gameday and update S3 history.

Settlement logic (OVER offered_line, all lines are x.5):
  Win  : actual_receiving_yards > offered_line
  Loss : actual_receiving_yards <= offered_line

Reads:
  s3://the-odds-api-mt/nfl/rec_yards_model/daily_runs/{gameday}/recommendations.csv

Writes / appends:
  s3://the-odds-api-mt/nfl/rec_yards_model/settled/settled_bets.parquet

Run:
  python src/nfl_rec_yards_modeling/scripts/settle_rec_yards.py --gameday 2026-09-14
  python src/nfl_rec_yards_modeling/scripts/settle_rec_yards.py  # defaults to yesterday ET
"""

import argparse
import html as html_module
import os
import sys
import warnings
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import botocore.exceptions
import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from build_historical_spine import _normalize_name

S3_BUCKET     = "the-odds-api-mt"
S3_PREFIX     = "nfl/rec_yards_model"
SES_SOURCE    = os.environ.get("SETTLEMENT_SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SETTLEMENT_SES_TO", "").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()
ENABLE_SNS    = os.environ.get("ENABLE_SNS", "").strip().lower() in ("1", "true", "yes")

ET    = ZoneInfo("America/New_York")
_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"


def yesterday_et() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")


def current_nfl_season() -> int:
    now = datetime.now(ET)
    return now.year if now.month >= 8 else now.year - 1


def _s3():
    return boto3.client("s3")


def load_s3_csv(key: str) -> pd.DataFrame | None:
    try:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        return pd.read_csv(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def load_settled_parquet() -> pd.DataFrame:
    key = f"{S3_PREFIX}/settled/settled_bets.parquet"
    try:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return pd.DataFrame()
        raise


def save_settled_parquet(df: pd.DataFrame) -> None:
    key = f"{S3_PREFIX}/settled/settled_bets.parquet"
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    print(f"  Settled parquet saved → s3://{S3_BUCKET}/{key}")


def get_actual_rec_yards(gameday: str, season: int) -> pd.DataFrame:
    """Pull weekly receiving stats for the given gameday from nfl_data_py."""
    warnings.filterwarnings("ignore")
    import nfl_data_py as nfl
    import numpy as np

    schedule = nfl.import_schedules([season])
    schedule = schedule[schedule["game_type"] == "REG"].copy()
    schedule["gameday"] = pd.to_datetime(schedule["gameday"]).dt.strftime("%Y-%m-%d")
    weeks_on_day = schedule[schedule["gameday"] == gameday]["week"].unique().tolist()

    if not weeks_on_day:
        print(f"  No schedule entries found for {gameday} (season {season})")
        return pd.DataFrame()

    print(f"  Week(s) on {gameday}: {weeks_on_day}")

    # Try weekly data first; fall back to PBP for current/unreleased seasons
    try:
        weekly = nfl.import_weekly_data(years=[season])
        weekly = weekly[
            (weekly["season_type"] == "REG") &
            (weekly["week"].isin(weeks_on_day)) &
            (weekly["position"].isin(["WR", "TE"]))
        ].copy()
        if weekly.empty:
            print(f"  No weekly data for weeks {weeks_on_day} (may not be posted yet)")
            return pd.DataFrame()
        weekly["actual_yards"] = weekly["receiving_yards"].fillna(0)
        weekly["player_norm"]  = weekly["player_display_name"].apply(_normalize_name)
        return weekly[["player_norm", "player_display_name", "position", "recent_team",
                       "week", "actual_yards"]].rename(
            columns={"player_display_name": "player", "recent_team": "team"}
        ).copy()
    except Exception as e:
        print(f"  import_weekly_data failed ({e}) — using PBP fallback")

    # PBP fallback
    pbp = nfl.import_pbp_data(years=[season], columns=[
        "season", "week", "season_type", "posteam",
        "play_type", "receiver_player_id",
        "yards_gained", "complete_pass",
    ])
    pbp = pbp[
        (pbp["season_type"] == "REG") &
        (pbp["week"].isin(weeks_on_day)) &
        (pbp["play_type"] == "pass") &
        pbp["receiver_player_id"].notna()
    ].copy()
    if pbp.empty:
        print(f"  No PBP data for weeks {weeks_on_day}")
        return pd.DataFrame()

    pbp["rec_yds"] = np.where(pbp["complete_pass"] == 1, pbp["yards_gained"].fillna(0), 0)
    agg = (
        pbp.groupby(["receiver_player_id", "posteam", "season", "week"])
        .agg(receiving_yards=("rec_yds", "sum"))
        .reset_index()
    )
    players_df = nfl.import_players()[["gsis_id", "display_name", "position"]].dropna(subset=["gsis_id"])
    agg = agg.merge(players_df, left_on="receiver_player_id", right_on="gsis_id", how="left")
    agg = agg[agg["position"].isin(["WR", "TE"])].copy()
    agg["actual_yards"] = agg["receiving_yards"].fillna(0)
    agg["player_norm"]  = agg["display_name"].apply(_normalize_name)
    return agg[["player_norm", "display_name", "position", "posteam",
                "week", "actual_yards"]].rename(
        columns={"display_name": "player", "posteam": "team"}
    ).copy()


def settle(recs: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    merged = recs.merge(
        actuals[["player_norm", "actual_yards", "week"]],
        on="player_norm", how="left",
    )

    def _settle_row(r):
        if pd.isna(r["actual_yards"]):
            return "unmatched"
        return "win" if r["actual_yards"] > r["offered_line"] else "loss"

    merged["outcome"] = merged.apply(_settle_row, axis=1)
    merged["hit"] = (merged["outcome"] == "win").where(
        merged["outcome"] != "unmatched", other=np.nan
    )
    return merged


def american_to_payout(price: float) -> float:
    return 100.0 / abs(price) if price < 0 else price / 100.0


def compute_summary(df: pd.DataFrame) -> dict:
    settled  = df[df["outcome"].isin(["win", "loss"])]
    n_win    = (settled["outcome"] == "win").sum()
    n_loss   = (settled["outcome"] == "loss").sum()
    n_bets   = n_win + n_loss
    hit_rate = n_win / n_bets if n_bets > 0 else float("nan")
    wins     = settled[settled["outcome"] == "win"]
    pnl      = wins["consensus_over_price"].apply(american_to_payout).sum() - n_loss
    roi      = pnl / n_bets if n_bets > 0 else float("nan")
    return {"n_bets": n_bets, "n_win": n_win, "n_loss": n_loss,
            "hit_rate": hit_rate, "pnl": pnl, "roi": roi}


def _outcome_badge(outcome: str) -> str:
    styles = {
        "win":       "background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:4px;font-weight:600",
        "loss":      "background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:4px;font-weight:600",
        "unmatched": "background:#f3f4f6;color:#6b7280;padding:2px 8px;border-radius:4px",
    }
    labels = {"win": "WIN", "loss": "LOSS", "unmatched": "N/A"}
    return f'<span style="{styles.get(outcome, "")}">{labels.get(outcome, outcome.upper())}</span>'


def _pnl_cell(outcome: str, price: float) -> str:
    if outcome == "win":
        val = american_to_payout(price)
        return (f'<td style="text-align:right;padding:8px 12px;font-family:{_MONO};'
                f'font-weight:600;color:#065f46">+{val:.3f}u</td>')
    if outcome == "loss":
        return (f'<td style="text-align:right;padding:8px 12px;font-family:{_MONO};'
                f'font-weight:600;color:#991b1b">−1.000u</td>')
    return f'<td style="text-align:right;padding:8px 12px;color:#9ca3af">—</td>'


def _summary_box(label: str, s: dict) -> str:
    pnl   = s["pnl"]
    pnl_c = "#065f46" if pnl >= 0 else "#991b1b"
    pnl_s = f"+{pnl:.2f}u" if pnl >= 0 else f"{pnl:.2f}u"
    hit_s = f"{s['hit_rate']:.1%}" if not pd.isna(s.get("hit_rate", float("nan"))) else "—"
    roi   = s.get("roi", float("nan"))
    roi_s = f"{roi:+.1%}" if not pd.isna(roi) else "—"
    roi_c = "#065f46" if not pd.isna(roi) and roi >= 0 else "#991b1b"
    return f"""
<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:14px 18px;margin-bottom:16px">
  <div style="font-weight:600;font-size:13px;color:#374151;margin-bottom:8px">{html_module.escape(label)}</div>
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    <div><span style="color:#6b7280;font-size:11px">RECORD</span><br>
      <span style="font-size:15px;font-weight:600">{s['n_win']}W–{s['n_loss']}L</span></div>
    <div><span style="color:#6b7280;font-size:11px">HIT RATE</span><br>
      <span style="font-size:15px;font-weight:600">{hit_s}</span></div>
    <div><span style="color:#6b7280;font-size:11px">P&amp;L</span><br>
      <span style="font-size:15px;font-weight:600;color:{pnl_c}">{pnl_s}</span></div>
    <div><span style="color:#6b7280;font-size:11px">ROI</span><br>
      <span style="font-size:15px;font-weight:600;color:{roi_c}">{roi_s}</span></div>
    <div><span style="color:#6b7280;font-size:11px">BETS</span><br>
      <span style="font-size:15px;font-weight:600">{s['n_bets']}</span></div>
  </div>
</div>"""


def build_ops_health_email(gameday: str, today_settled: pd.DataFrame | None,
                           all_time_summary: dict, had_games: bool) -> str:
    today_summary = compute_summary(today_settled) if today_settled is not None and len(today_settled) else {
        "n_bets": 0, "n_win": 0, "n_loss": 0, "hit_rate": float("nan"), "pnl": 0.0, "roi": float("nan"),
    }

    # Unmatched player warnings
    unmatched_html = ""
    if today_settled is not None and len(today_settled):
        unmatched = today_settled[today_settled["outcome"] == "unmatched"]
        if not unmatched.empty:
            urows = "".join(
                f'<tr><td style="padding:4px 8px;font-family:{_MONO}">'
                f'{html_module.escape(str(r.get("player_name", "")))}</td>'
                f'<td style="padding:4px 8px;color:#6b7280">'
                f'{html_module.escape(str(r.get("player_norm", "")))}</td></tr>'
                for _, r in unmatched.iterrows()
            )
            unmatched_html = f"""
<div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:6px;padding:12px 14px;margin-top:12px">
  <div style="font-weight:600;font-size:12px;color:#c2410c;margin-bottom:6px">&#9888; Unmatched players ({len(unmatched)})</div>
  <p style="font-size:12px;color:#7c2d12;margin:0 0 8px">Name in recommendations.csv didn't match nfl_data_py. Add to NAME_MAP if the player played.</p>
  <table style="font-size:12px;border-collapse:collapse">
  <thead><tr style="color:#9ca3af">
    <th style="padding:4px 8px;text-align:left">Player (from rec)</th>
    <th style="padding:4px 8px;text-align:left">Normalized</th>
  </tr></thead>
  <tbody>{urows}</tbody>
  </table>
</div>"""

    if had_games and today_settled is not None and len(today_settled):
        settle_section = _summary_box(f"Yesterday ({gameday})", today_summary)
    elif not had_games:
        settle_section = f'<p style="color:#6b7280;font-size:13px">No NFL games on {gameday}.</p>'
    else:
        settle_section = f'<p style="color:#6b7280;font-size:13px">No qualifying bets placed on {gameday}.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>NFL Rec Yards Ops — {gameday}</title></head>
<body style="margin:0;padding:16px;background:#f4f4f5;font-family:{_SANS};font-size:13px;color:#1a1a1a">
<div style="max-width:700px;margin:0 auto;background:#fff;padding:24px;border-radius:8px;border:1px solid #e2e2e4">
  <h2 style="font-size:18px;margin:0 0 4px">NFL Rec Yards — Ops Health Check</h2>
  <p style="color:#6b7280;font-size:12px;margin:0 0 20px">Generated {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')} &nbsp;|&nbsp; Settle + spine rebuild</p>
  <h3 style="font-size:14px;font-weight:600;margin:0 0 8px">Settlement — {gameday}</h3>
  {settle_section}
  {unmatched_html}
  <h3 style="font-size:14px;font-weight:600;margin:20px 0 8px">All-Time</h3>
  {_summary_box("All-Time Results (OVER receiving yards)", all_time_summary)}
  <p style="font-size:11px;color:#9ca3af;margin-top:16px">Full per-bet results in the 9:00 AM email.</p>
</div>
</body>
</html>"""


def send_email(subject: str, html_body: str, text_body: str) -> None:
    to_list = [a.strip() for a in SES_TO_RAW.split(",") if a.strip()]
    if SES_SOURCE and to_list:
        try:
            boto3.client("ses", region_name="us-east-2").send_email(
                Source=SES_SOURCE,
                Destination={"ToAddresses": to_list},
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": {
                        "Html": {"Data": html_body,  "Charset": "UTF-8"},
                        "Text": {"Data": text_body,  "Charset": "UTF-8"},
                    },
                },
            )
            print(f"  SES email sent: {subject}")
        except Exception as e:
            print(f"  SES send failed: {e}")
    if ENABLE_SNS and SNS_TOPIC_ARN:
        boto3.client("sns").publish(
            TopicArn=SNS_TOPIC_ARN, Subject=subject[:100], Message=text_body[:256_000],
        )
        print(f"  SNS published")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", type=str, default=None)
    args    = parser.parse_args()
    gameday = args.gameday or yesterday_et()
    season  = current_nfl_season()

    print(f"\nNFL Rec Yards Settlement — gameday={gameday}  season={season}")
    print("=" * 55)

    rec_key = f"{S3_PREFIX}/daily_runs/{gameday}/recommendations.csv"
    recs    = load_s3_csv(rec_key)
    had_bets = recs is not None and len(recs) > 0

    today_settled = None
    had_games     = False
    combined      = pd.DataFrame()

    if had_bets:
        recs["player_norm"] = recs["player_norm"].fillna(
            recs.get("player_name", pd.Series()).apply(_normalize_name)
        )
        print(f"  Recommendations loaded: {len(recs)} bets")

        print(f"  Fetching actual rec yards for {gameday}...")
        actuals  = get_actual_rec_yards(gameday, season)
        had_games = not actuals.empty

        if actuals.empty:
            print("  No data yet — may not be posted. Nothing settled.")
        else:
            print(f"  Weekly data rows: {len(actuals)}")
            today_settled = settle(recs, actuals)
            s_sum = compute_summary(today_settled)
            print(f"  Settled: {s_sum['n_win']}W  {s_sum['n_loss']}L  "
                  f"P&L={s_sum['pnl']:+.3f}u")

            history  = load_settled_parquet()
            if not history.empty and "gameday" in history.columns:
                history = history[history["gameday"] != gameday].copy()

            new_rows = today_settled[today_settled["outcome"].isin(["win", "loss"])].copy()
            new_rows["gameday"] = gameday
            new_rows["season"]  = season
            combined = pd.concat([history, new_rows], ignore_index=True)
            save_settled_parquet(combined)
    else:
        print(f"  No recommendations CSV found for {gameday}")
        combined = load_settled_parquet()

    at = compute_summary(combined) if not combined.empty else {
        "n_bets": 0, "n_win": 0, "n_loss": 0, "hit_rate": float("nan"), "pnl": 0.0, "roi": float("nan"),
    }
    print(f"  All-time: {at['n_win']}W {at['n_loss']}L  "
          f"P&L={at['pnl']:+.3f}u  ROI={at['roi']:+.1%}  ({at['n_bets']} bets)")

    ts = compute_summary(today_settled) if today_settled is not None and len(today_settled) else {
        "n_bets": 0, "n_win": 0, "n_loss": 0, "pnl": 0.0,
    }
    if had_games and ts["n_bets"] > 0:
        subject = (f"NFL Rec Yards Ops — {gameday} — "
                   f"{ts['n_win']}W {ts['n_loss']}L ({ts['pnl']:+.2f}u)")
    else:
        subject = (f"NFL Rec Yards Ops — {gameday} — No games settled "
                   f"(all-time: {at['n_bets']} bets, {at['pnl']:+.2f}u)")

    html_body = build_ops_health_email(gameday, today_settled, at, had_games)
    text_body = (
        f"NFL Rec Yards Settlement — {gameday}\n\n"
        f"Yesterday : {ts['n_win']}W {ts['n_loss']}L  P&L={ts['pnl']:+.3f}u\n"
        f"All-time  : {at['n_win']}W {at['n_loss']}L  P&L={at['pnl']:+.3f}u  "
        f"ROI={at['roi']:+.1%}  ({at['n_bets']} bets)\n"
    )
    send_email(subject, html_body, text_body)

    print(f"\n{'='*55}")
    print(f"  Settlement complete — {gameday}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
