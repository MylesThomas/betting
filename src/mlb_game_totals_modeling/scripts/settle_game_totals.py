"""
Settle MLB game totals bets for a given gameday and send summary email.

Settlement logic (uses side + line columns from qualifying_bets.csv):
  Win:  total_runs < line (under) or total_runs > line (over) → +payout at price - 1
  Loss: opposite result → -1 unit
  Push: total_runs == line (impossible at .5 lines since scores are integers) → 0

Reads from S3:
  mlb/game_totals_model/daily_runs/{gameday}/qualifying_bets.csv

Scores actuals from MLB Stats API (statsapi).

Writes to S3:
  mlb/game_totals_model/daily_runs/{gameday}/settled.csv
  mlb/game_totals_model/settled/mlb_gt_settled_bets.parquet  (cumulative)

Sends SES HTML settle email with yesterday P&L + season summary.

Usage:
  python src/mlb_game_totals_modeling/scripts/settle_game_totals.py
  python src/mlb_game_totals_modeling/scripts/settle_game_totals.py --gameday 2026-06-24
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
import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

S3_BUCKET     = "the-odds-api-mt"
DAILY_PREFIX  = "mlb/game_totals_model/daily_runs"
SETTLED_KEY   = "mlb/game_totals_model/settled/mlb_gt_settled_bets.parquet"

SES_SOURCE    = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()

ET = ZoneInfo("America/New_York")
_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"

STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
TEAM_NORMALIZE = {"Athletics": "Oakland Athletics"}


def normalize_team(name: str) -> str:
    return TEAM_NORMALIZE.get(name, name)


def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def yesterday_et() -> str:
    return (datetime.now(ET).date() - timedelta(days=1)).strftime("%Y-%m-%d")


# ── S3 ────────────────────────────────────────────────────────────────────────

def _s3():
    return boto3.client("s3")


def s3_get_parquet(key: str) -> pd.DataFrame:
    body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def s3_put_parquet(key: str, df: pd.DataFrame) -> None:
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())


def s3_put_csv(key: str, df: pd.DataFrame) -> None:
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=df.to_csv(index=False).encode())


def s3_key_exists(key: str) -> bool:
    try:
        _s3().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False


# ── MLB Stats API scores ──────────────────────────────────────────────────────

def fetch_game_scores(gameday: str) -> list[dict]:
    """Fetch all MLB games for a date from MLB Stats API."""
    r = requests.get(
        f"{STATSAPI_BASE}/schedule",
        params={
            "sportId": 1,
            "date": gameday,
            "hydrate": "linescore",
        },
        timeout=30,
    )
    if r.status_code != 200:
        print(f"  Stats API error: {r.status_code}")
        return []

    games = []
    for date_block in r.json().get("dates", []):
        for game in date_block.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status != "Final":
                continue
            linescore = game.get("linescore", {})
            away_runs = linescore.get("teams", {}).get("away", {}).get("runs")
            home_runs = linescore.get("teams", {}).get("home", {}).get("runs")
            if away_runs is None or home_runs is None:
                continue

            home_name = normalize_team(game.get("teams", {}).get("home", {}).get("team", {}).get("name", ""))
            away_name = normalize_team(game.get("teams", {}).get("away", {}).get("team", {}).get("name", ""))

            games.append({
                "home_team":   home_name,
                "away_team":   away_name,
                "home_runs":   int(home_runs),
                "away_runs":   int(away_runs),
                "total_runs":  int(home_runs) + int(away_runs),
                "game_date":   gameday,
            })
    return games


# ── Settlement ────────────────────────────────────────────────────────────────

def compute_pnl(price: float, hit: int, is_push: bool) -> float:
    """Compute P&L for 1-unit bet at given price."""
    if is_push:
        return 0.0
    if hit == 1:
        p = float(price)
        return float(p / 100) if p > 0 else float(100 / abs(p))
    return -1.0


def settle_bets(bets: pd.DataFrame, scores: list[dict]) -> pd.DataFrame:
    """Join bets with actual scores and compute P&L using side + line columns."""
    if bets.empty or not scores:
        return pd.DataFrame()

    scores_df = pd.DataFrame(scores)
    merged = bets.merge(
        scores_df[["home_team", "away_team", "total_runs"]],
        on=["home_team", "away_team"],
        how="left",
    )

    def _hit(r) -> int:
        if pd.isna(r["total_runs"]):
            return 0
        if r["side"] == "under":
            return int(r["total_runs"] < r["line"])
        return int(r["total_runs"] > r["line"])

    merged["hit"]         = merged.apply(_hit, axis=1)
    merged["is_push"]     = False
    merged["pnl"]         = merged.apply(
        lambda r: compute_pnl(r["price"], r["hit"], r["is_push"]),
        axis=1,
    )
    merged["actual_runs"] = merged["total_runs"].copy()
    merged = merged.drop(columns=["total_runs"], errors="ignore")
    return merged


# ── Email ─────────────────────────────────────────────────────────────────────

def build_settle_email(settled: pd.DataFrame, gameday: str,
                       season_stats: dict | None, yesterday_stats: dict | None) -> str:
    he = html_module.escape

    def fmt(v, fmt_str: str) -> str:
        try:
            return format(float(v), fmt_str)
        except (TypeError, ValueError):
            return "—"

    def stat_card(label: str, value: str, green: bool = False) -> str:
        color = "#276221" if green else "#222"
        return (
            f"<div style='border:1px solid #ddd;border-radius:6px;padding:12px 20px;"
            f"min-width:120px;background:#fff'>"
            f"<div style='font-size:10px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px'>{label}</div>"
            f"<div style='font-size:22px;font-weight:700;color:{color}'>{value}</div>"
            f"</div>"
        )

    def stats_row(stats: dict | None, label: str) -> str:
        if not stats:
            return ""
        u   = stats.get("units", 0.0)
        w   = stats.get("wins", 0)
        l   = stats.get("losses", 0)
        roi = stats.get("roi", 0.0)
        return (
            f"<div style='display:flex;gap:12px;margin:8px 0;flex-wrap:wrap'>"
            f"{stat_card(f'{label} PNL', f'{u:+.2f}u', green=u>=0)}"
            f"{stat_card(f'{label} Record', f'{w}W–{l}L')}"
            f"{stat_card(f'{label} Win%', f'{w/(w+l)*100:.1f}%' if (w+l)>0 else '—', green=(w/(w+l)>0.5) if (w+l)>0 else False)}"
            f"{stat_card(f'{label} ROI', f'{roi*100:+.1f}%', green=roi>=0)}"
            f"</div>"
        )

    rows = ""
    if not settled.empty:
        for _, r in settled.sort_values(["strategy_tag", "home_team", "bookmaker"]).iterrows():
            hit    = int(r.get("hit", 0))
            result = "WIN ✓" if hit == 1 else "LOSS ✗"
            bg     = "#eaf6ea" if hit == 1 else "#fdecea"
            tag    = str(r.get("strategy_tag", "benchmark"))
            side   = str(r.get("side", "under")).upper()
            tag_label = {"both": "★★", "model": "★", "benchmark": "—"}.get(tag, tag)
            rows += (
                f"<tr style='background:{bg}'>"
                f"<td style='text-align:center'>{he(tag_label)}</td>"
                f"<td>{he(r['away_team'])} @ {he(r['home_team'])}</td>"
                f"<td style='text-align:center'>{he(str(r.get('bookmaker','')))}</td>"
                f"<td style='text-align:center'>{fmt(r.get('line'),'.1f')}</td>"
                f"<td style='text-align:center'>{he(side)}</td>"
                f"<td style='text-align:center'>{fmt(r.get('price'),'.0f')}</td>"
                f"<td style='text-align:center'>{fmt(r.get('display_edge', r.get('edge_under')),'+.1%')}</td>"
                f"<td style='text-align:center'>{r.get('actual_runs','—')}</td>"
                f"<td style='text-align:center;font-weight:bold'>{result}</td>"
                f"<td style='text-align:center;font-weight:bold'>{fmt(r.get('pnl'),'+.2f')}u</td>"
                f"</tr>\n"
            )

    n     = len(settled) if not settled.empty else 0
    n_win = int((settled["hit"] == 1).sum()) if not settled.empty else 0
    n_los = int((settled["hit"] == 0).sum()) if not settled.empty else 0
    pnl   = float(settled["pnl"].sum()) if not settled.empty else 0.0

    # Per-strategy breakdown
    strat_rows = ""
    if not settled.empty and "strategy_tag" in settled.columns:
        for tag, label in [("both", "★★ Both"), ("model", "★ Model"), ("benchmark", "Benchmark")]:
            sub = settled[settled["strategy_tag"] == tag]
            if sub.empty:
                continue
            sw = int((sub["hit"] == 1).sum())
            sl = int((sub["hit"] == 0).sum())
            sp = float(sub["pnl"].sum())
            strat_rows += (
                f"<tr>"
                f"<td>{he(label)}</td>"
                f"<td style='text-align:center'>{sw+sl}</td>"
                f"<td style='text-align:center'>{sw}W–{sl}L</td>"
                f"<td style='text-align:center;font-weight:bold;color:{'#276221' if sp>=0 else '#c0392b'}'>{sp:+.2f}u</td>"
                f"</tr>\n"
            )

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<style>
  body {{font-family:{_SANS};color:#222;max-width:1000px;margin:auto;padding:20px}}
  h2 {{color:#2c3e50;margin-bottom:4px}}
  table {{border-collapse:collapse;width:100%;margin-top:8px}}
  th {{background:#2c3e50;color:#fff;padding:7px 8px;text-align:left;font-size:12px}}
  td {{padding:5px 8px;border-bottom:1px solid #e0e0e0;font-size:12px}}
  .footer {{background:#ecf0f1;border-radius:6px;padding:10px 16px;margin-top:16px;font-size:12px;color:#555}}
</style>
</head><body>
<h2>MLB Game Totals — Settle {he(gameday)}</h2>
<p style='margin-top:4px'>
  {n_win}W–{n_los}L · <strong>{pnl:+.2f}u</strong>
  {'&nbsp;·&nbsp;<span style="color:#276221">✓ Positive day</span>' if pnl >= 0 else '&nbsp;·&nbsp;<span style="color:#c0392b">Negative day</span>'}
</p>
{stats_row(yesterday_stats, 'Yesterday')}
{stats_row(season_stats, f'{datetime.now(ET).year} Season')}

<table style='margin-top:12px;margin-bottom:16px;max-width:400px'>
  <tr><th>Strategy</th><th>n</th><th>Record</th><th>P&amp;L</th></tr>
  {strat_rows if strat_rows else '<tr><td colspan="4" style="color:#888">—</td></tr>'}
</table>

<table>
  <tr>
    <th style='text-align:center'>Tag</th>
    <th>Matchup</th><th>Book</th><th>Line</th><th>Side</th><th>Price</th>
    <th>Edge</th><th>Actual</th><th>Result</th><th>P&amp;L</th>
  </tr>
  {rows if rows else '<tr><td colspan="10" style="color:#888;text-align:center">No bets settled</td></tr>'}
</table>

<div class='footer'>
  Benchmark OOS: +8.45% ROI (n=2,324) · Strategy B OOS: +285.2u (n=1,623, on probation 2026)
</div>
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=yesterday_et())
    args = parser.parse_args()
    gameday = args.gameday

    print(f"MLB Game Totals settle | gameday={gameday}")

    # Load qualifying bets from S3
    bets_key = f"{DAILY_PREFIX}/{gameday}/qualifying_bets.csv"
    if not s3_key_exists(bets_key):
        print(f"  No qualifying bets file found: {bets_key}")
        bets = pd.DataFrame()
    else:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=bets_key)["Body"].read()
        bets = pd.read_csv(BytesIO(body))
        print(f"  Loaded {len(bets)} qualifying bets from S3")

    # Fetch game scores
    print("Fetching game scores from MLB Stats API...")
    scores = fetch_game_scores(gameday)
    print(f"  {len(scores)} final games found")
    for g in scores:
        print(f"    {g['away_team']} @ {g['home_team']}: {g['away_runs']}-{g['home_runs']} = {g['total_runs']} runs")

    # Settle
    if not bets.empty and scores:
        settled = settle_bets(bets, scores)
        print(f"\nSettled {len(settled)} bets")
        if not settled.empty:
            n_win = int((settled["hit"] == 1).sum())
            n_los = int((settled["hit"] == 0).sum())
            pnl   = float(settled["pnl"].sum())
            print(f"  Result: {n_win}W–{n_los}L  PnL: {pnl:+.2f}u")
            if "strategy_tag" in settled.columns:
                for tag in ["both", "model", "benchmark"]:
                    sub = settled[settled["strategy_tag"] == tag]
                    if not sub.empty:
                        sw = int((sub["hit"] == 1).sum())
                        sl = int((sub["hit"] == 0).sum())
                        print(f"    {tag}: {sw}W–{sl}L  {float(sub['pnl'].sum()):+.2f}u")
    else:
        settled = pd.DataFrame()
        print("  No bets to settle")

    # Save settled
    if not settled.empty:
        settled_day_key = f"{DAILY_PREFIX}/{gameday}/settled.csv"
        s3_put_csv(settled_day_key, settled)
        print(f"  Settled → s3://{S3_BUCKET}/{settled_day_key}")

        # Append to cumulative settled parquet
        try:
            existing = s3_get_parquet(SETTLED_KEY)
            cumulative = pd.concat([existing, settled], ignore_index=True)
        except Exception:
            cumulative = settled.copy()

        s3_put_parquet(SETTLED_KEY, cumulative)
        print(f"  Cumulative → s3://{S3_BUCKET}/{SETTLED_KEY}  ({len(cumulative)} rows)")

    # Load season + yesterday stats for email cards
    season_stats    = None
    yesterday_stats = None
    try:
        hist = s3_get_parquet(SETTLED_KEY)
        if not hist.empty and "pnl" in hist.columns:
            year      = datetime.now(ET).year
            all_plays = hist[hist["pnl"].notna()]
            season_pl = all_plays[all_plays["game_date"].astype(str).str[:4] == str(year)]
            if not season_pl.empty:
                season_stats = {
                    "units":  float(season_pl["pnl"].sum()),
                    "wins":   int((season_pl["pnl"] > 0).sum()),
                    "losses": int((season_pl["pnl"] < 0).sum()),
                    "roi":    float(season_pl["pnl"].mean()),
                }
            yest_pl = all_plays[all_plays["game_date"].astype(str).str[:10] == gameday]
            if not yest_pl.empty:
                yesterday_stats = {
                    "units":  float(yest_pl["pnl"].sum()),
                    "wins":   int((yest_pl["pnl"] > 0).sum()),
                    "losses": int((yest_pl["pnl"] < 0).sum()),
                    "roi":    float(yest_pl["pnl"].mean()),
                }
    except Exception as e:
        print(f"  Season stats unavailable: {e}")

    # Email
    n_bets = len(settled)
    n_win  = int((settled["hit"] == 1).sum()) if not settled.empty else 0
    n_los  = n_bets - n_win
    pnl    = float(settled["pnl"].sum()) if not settled.empty else 0.0

    subject  = f"MLB Game Totals Settle — {n_win}W–{n_los}L · {pnl:+.2f}u — {gameday}"
    html_body = build_settle_email(settled, gameday, season_stats, yesterday_stats)
    send_ses(subject, html_body)
    publish_sns(subject, f"{n_win}W–{n_los}L · {pnl:+.2f}u")

    print(f"\nDone: {n_win}W–{n_los}L · {pnl:+.2f}u")


if __name__ == "__main__":
    main()
