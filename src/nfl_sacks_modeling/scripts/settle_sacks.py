"""
Settle NFL sacks bets for a given gameday and send summary email.

Settlement logic (Under 0.5 sacks):
  Win  : player sacks == 0    → +units_to_win at recorded under odds
  Push : player sacks == 0.5  → 0
  Loss : player sacks >= 1    → -1 unit

Reads:
  s3://the-odds-api-mt/nfl/sacks_model/daily_runs/{gameday}/bet_sheet.csv

Writes:
  s3://the-odds-api-mt/nfl/sacks_model/daily_runs/{gameday}/bet_sheet_settled.csv

Sends SES HTML email (if SES_SOURCE + SES_TO set):
  Yesterday W-L-P + P&L + all-time summary

Run:
  python src/nfl_sacks_modeling/scripts/settle_sacks.py --gameday 2026-09-11
  python src/nfl_sacks_modeling/scripts/settle_sacks.py   # defaults to yesterday ET
"""

import argparse
import html as html_module
import os
import sys
import warnings
from datetime import date, datetime, timedelta
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
S3_PREFIX     = "nfl/sacks_model"
SES_SOURCE    = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SES_TO", "").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()

ET = ZoneInfo("America/New_York")


# ── Date helpers ────────────────────────────────────────────────────────────────

def yesterday_et() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")


def current_nfl_season() -> int:
    now = datetime.now(ET)
    return now.year if now.month >= 8 else now.year - 1


def full_to_pbp(name: str) -> str:
    """Convert 'Myles Garrett' → 'M.Garrett' (matches nfl_data_py PBP format)."""
    parts = name.strip().split()
    return f"{parts[0][0]}.{parts[-1]}" if len(parts) >= 2 else name


def units_to_win(price: float) -> float:
    """Units won per 1 unit risked at American odds."""
    if pd.isna(price):
        return float("nan")
    return 100.0 / abs(price) if price < 0 else price / 100.0


# ── S3 helpers ──────────────────────────────────────────────────────────────────

def s3_key_bet_sheet(gameday: str) -> str:
    return f"{S3_PREFIX}/daily_runs/{gameday}/bet_sheet.csv"


def s3_key_settled(gameday: str) -> str:
    return f"{S3_PREFIX}/daily_runs/{gameday}/bet_sheet_settled.csv"


def load_s3_csv(key: str) -> pd.DataFrame | None:
    try:
        body = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        return pd.read_csv(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def save_s3_csv(key: str, df: pd.DataFrame) -> None:
    boto3.client("s3").put_object(
        Bucket=S3_BUCKET, Key=key, Body=df.to_csv(index=False).encode()
    )


def list_settled_keys() -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/daily_runs/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("bet_sheet_settled.csv"):
                keys.append(obj["Key"])
    return sorted(keys)


def load_all_time_settled() -> pd.DataFrame:
    keys = list_settled_keys()
    if not keys:
        return pd.DataFrame()
    frames = []
    for key in keys:
        df = load_s3_csv(key)
        if df is not None and len(df):
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── nfl_data_py: actual sack results ───────────────────────────────────────────

def get_sack_results_for_date(gameday: str, season: int) -> pd.DataFrame:
    """
    Pull actual sack results for all games on the given date.
    Returns DataFrame with columns: pbp_name, defteam, sacks (per player per game).
    """
    warnings.filterwarnings("ignore")
    import nfl_data_py as nfl

    # Get schedule to find game_ids for this date
    schedule = nfl.import_schedules([season])
    schedule = schedule[schedule["game_type"] == "REG"].copy()
    schedule["gameday"] = pd.to_datetime(schedule["gameday"]).dt.strftime("%Y-%m-%d")
    todays_games = schedule[schedule["gameday"] == gameday]["game_id"].tolist()

    if not todays_games:
        return pd.DataFrame(columns=["pbp_name", "defteam", "sacks"])

    print(f"  Games on {gameday}: {todays_games}")

    pbp = nfl.import_pbp_data([season], columns=[
        "game_id", "season_type", "defteam",
        "sack", "sack_player_name",
        "half_sack_1_player_name", "half_sack_2_player_name",
        "lateral_sack_player_name",
    ])
    pbp = pbp[
        (pbp["season_type"] == "REG") &
        (pbp["game_id"].isin(todays_games))
    ].copy()

    sack_rows = []
    full_sacks = pbp[pbp["sack"] == 1]
    for _, r in full_sacks.iterrows():
        if pd.notna(r["half_sack_1_player_name"]):
            sack_rows.append({"pbp_name": r["half_sack_1_player_name"], "defteam": r["defteam"], "sacks": 0.5})
            if pd.notna(r["half_sack_2_player_name"]):
                sack_rows.append({"pbp_name": r["half_sack_2_player_name"], "defteam": r["defteam"], "sacks": 0.5})
        elif pd.notna(r["sack_player_name"]):
            sack_rows.append({"pbp_name": r["sack_player_name"], "defteam": r["defteam"], "sacks": 1.0})
            if pd.notna(r["lateral_sack_player_name"]):
                sack_rows.append({"pbp_name": r["lateral_sack_player_name"], "defteam": r["defteam"], "sacks": 1.0})

    if not sack_rows:
        return pd.DataFrame(columns=["pbp_name", "defteam", "sacks"])

    return (
        pd.DataFrame(sack_rows)
        .groupby(["pbp_name", "defteam"], as_index=False)["sacks"]
        .sum()
    )


# ── Settlement logic ────────────────────────────────────────────────────────────

def settle_bets(bets: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Match bet sheet rows against actual sack results and compute P&L.
    Only rows where bet==True are settled; the rest are carried through as-is.
    """
    bets = bets.copy()
    bets["pbp_name"] = bets["player"].apply(full_to_pbp)

    # Merge on (pbp_name, team = defteam)
    merged = bets.merge(
        actuals.rename(columns={"sacks": "actual_sacks"}),
        left_on=["pbp_name", "team"],
        right_on=["pbp_name", "defteam"],
        how="left",
    )
    merged["actual_sacks"] = merged["actual_sacks"].fillna(0.0)

    def settle_row(r):
        if not r.get("bet", False):
            return pd.Series({"outcome": "no_bet", "pnl": 0.0})
        actual = r["actual_sacks"]
        price  = r.get("prop_median_price_under", float("nan"))
        if actual == 0.0:
            return pd.Series({"outcome": "win",  "pnl": units_to_win(price)})
        elif actual == 0.5:
            return pd.Series({"outcome": "push", "pnl": 0.0})
        else:
            return pd.Series({"outcome": "loss", "pnl": -1.0})

    outcomes = merged.apply(settle_row, axis=1)
    merged["outcome"] = outcomes["outcome"]
    merged["pnl"]     = outcomes["pnl"]
    merged = merged.drop(columns=["pbp_name", "defteam"], errors="ignore")
    return merged


# ── Summary stats ───────────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> dict:
    bets = df[df["outcome"].isin(["win", "loss", "push"])] if "outcome" in df.columns else pd.DataFrame()
    if bets.empty:
        return {"n_bets": 0, "n_win": 0, "n_loss": 0, "n_push": 0, "pnl": 0.0, "roi": float("nan")}
    n_win  = int((bets["outcome"] == "win").sum())
    n_loss = int((bets["outcome"] == "loss").sum())
    n_push = int((bets["outcome"] == "push").sum())
    pnl    = float(bets["pnl"].sum())
    n_decided = n_win + n_loss
    roi = pnl / max(n_win + n_loss + n_push, 1)
    return {"n_bets": len(bets), "n_win": n_win, "n_loss": n_loss, "n_push": n_push,
            "pnl": pnl, "roi": roi}


# ── HTML email ──────────────────────────────────────────────────────────────────

_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"


def _outcome_badge(outcome: str) -> str:
    styles = {
        "win":    "background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:4px;font-weight:600",
        "loss":   "background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:4px;font-weight:600",
        "push":   "background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:4px;font-weight:600",
        "no_bet": "color:#9ca3af",
    }
    labels = {"win": "WIN", "loss": "LOSS", "push": "PUSH", "no_bet": "—"}
    style = styles.get(outcome, "")
    label = labels.get(outcome, outcome.upper())
    return f'<span style="{style}">{label}</span>'


def _pnl_cell(val: float) -> str:
    if pd.isna(val) or val == 0.0:
        return f'<td style="text-align:right;padding:8px 12px;font-family:{_MONO};color:#6b7280">—</td>'
    color = "#065f46" if val > 0 else "#991b1b"
    sign  = "+" if val > 0 else ""
    return f'<td style="text-align:right;padding:8px 12px;font-family:{_MONO};font-weight:600;color:{color}">{sign}{val:.3f}u</td>'


def _summary_box(label: str, s: dict) -> str:
    pnl   = s["pnl"]
    pnl_c = "#065f46" if pnl >= 0 else "#991b1b"
    pnl_s = f"+{pnl:.2f}u" if pnl >= 0 else f"{pnl:.2f}u"
    roi_s = f"+{s['roi']:.1%}" if not pd.isna(s["roi"]) and s["roi"] >= 0 else (
            f"{s['roi']:.1%}" if not pd.isna(s["roi"]) else "—")
    roi_c = "#065f46" if not pd.isna(s["roi"]) and s["roi"] >= 0 else "#991b1b"
    return f"""
<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:14px 18px;margin-bottom:16px">
  <div style="font-weight:600;font-size:13px;color:#374151;margin-bottom:8px">{html_module.escape(label)}</div>
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    <div><span style="color:#6b7280;font-size:11px">RECORD</span><br>
      <span style="font-size:15px;font-weight:600">{s['n_win']}W–{s['n_loss']}L–{s['n_push']}P</span></div>
    <div><span style="color:#6b7280;font-size:11px">P&L</span><br>
      <span style="font-size:15px;font-weight:600;color:{pnl_c}">{pnl_s}</span></div>
    <div><span style="color:#6b7280;font-size:11px">ROI</span><br>
      <span style="font-size:15px;font-weight:600;color:{roi_c}">{roi_s}</span></div>
    <div><span style="color:#6b7280;font-size:11px">BETS</span><br>
      <span style="font-size:15px;font-weight:600">{s['n_bets']}</span></div>
  </div>
</div>"""


def build_settlement_html(
    gameday: str,
    settled_yesterday: pd.DataFrame | None,
    all_time: pd.DataFrame,
    had_games: bool,
) -> str:
    yesterday_summary = compute_summary(settled_yesterday) if settled_yesterday is not None and len(settled_yesterday) else {"n_bets": 0, "n_win": 0, "n_loss": 0, "n_push": 0, "pnl": 0.0, "roi": float("nan")}
    all_time_summary  = compute_summary(all_time) if not all_time.empty else {"n_bets": 0, "n_win": 0, "n_loss": 0, "n_push": 0, "pnl": 0.0, "roi": float("nan")}

    yesterday_section = ""
    if had_games and settled_yesterday is not None and len(settled_yesterday):
        bets = settled_yesterday[settled_yesterday["outcome"].isin(["win", "loss", "push"])].copy()
        bets = bets.sort_values("outcome")
        rows_html = ""
        for _, r in bets.iterrows():
            actual = r.get("actual_sacks", float("nan"))
            actual_s = f"{actual:.1f}" if not pd.isna(actual) else "—"
            mkt = r.get("prop_median_impl_over", float("nan"))
            mkt_s = f"{mkt:.1%}" if not pd.isna(mkt) else "—"
            odds = r.get("prop_median_price_under", float("nan"))
            odds_s = f"{int(odds):+d}" if not pd.isna(odds) else "—"
            rows_html += f"""
<tr style="border-bottom:1px solid #f3f4f6">
  <td style="padding:8px 12px">{html_module.escape(str(r['player']))}</td>
  <td style="padding:8px 12px">{html_module.escape(str(r.get('team','')))} vs {html_module.escape(str(r.get('opponent','')))}</td>
  <td style="padding:8px 12px;text-align:center">{actual_s}</td>
  <td style="padding:8px 12px;text-align:center">{html_module.escape(mkt_s)}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{odds_s}</td>
  <td style="padding:8px 12px;text-align:center">{_outcome_badge(r['outcome'])}</td>
  {_pnl_cell(r['pnl'])}
</tr>"""

        yesterday_section = f"""
<h3 style="font-size:14px;font-weight:600;margin:20px 0 8px;color:#111827">Yesterday ({gameday})</h3>
{_summary_box(f"Results — {gameday}", yesterday_summary)}
<table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px">
<thead><tr style="background:#1d2d44;color:#fff">
  <th style="padding:9px 12px;text-align:left">Player</th>
  <th style="padding:9px 12px;text-align:left">Matchup</th>
  <th style="padding:9px 12px;text-align:center">Actual Sacks</th>
  <th style="padding:9px 12px;text-align:center">Mkt P(Over)</th>
  <th style="padding:9px 12px;text-align:center">Under Odds</th>
  <th style="padding:9px 12px;text-align:center">Outcome</th>
  <th style="padding:9px 12px;text-align:right">P&L</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""
    elif not had_games:
        yesterday_section = f'<p style="color:#6b7280;font-size:13px;margin-bottom:20px">No NFL games yesterday ({gameday}).</p>'
    else:
        yesterday_section = f'<p style="color:#6b7280;font-size:13px;margin-bottom:20px">No qualifying bets were placed on {gameday}.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>NFL Sacks Settled — {gameday}</title></head>
<body style="margin:0;padding:16px;background:#f4f4f5;font-family:{_SANS};font-size:13px;color:#1a1a1a">
<div style="max-width:700px;margin:0 auto;background:#fff;padding:24px;border-radius:8px;border:1px solid #e2e2e4">
  <h2 style="font-size:18px;margin:0 0 4px">NFL Sacks Settlement</h2>
  <p style="color:#6b7280;font-size:12px;margin:0 0 20px">Generated {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}</p>

  {yesterday_section}

  <h3 style="font-size:14px;font-weight:600;margin:20px 0 8px;color:#111827">All-Time</h3>
  {_summary_box("All-Time Results (Under 0.5 sacks)", all_time_summary)}
</div>
</body>
</html>"""


# ── Plays HTML email ────────────────────────────────────────────────────────────

def build_plays_html(df: pd.DataFrame, gameday: str) -> str:
    bets = df[df["bet"]].copy().sort_values("lr_prob") if "bet" in df.columns else pd.DataFrame()

    def fmt_odds(p):
        return f"{int(p):+d}" if not pd.isna(p) else "—"

    def edge_color(edge: float) -> str:
        if pd.isna(edge): return "#6b7280"
        if edge >= 0.10:  return "#065f46"
        if edge >= 0.05:  return "#1d6fa4"
        return "#6b7280"

    rows_html = ""
    for _, r in bets.iterrows():
        mkt  = r.get("prop_median_impl_over", float("nan"))
        edge = mkt - r["lr_prob"] if not pd.isna(mkt) else float("nan")
        ec   = edge_color(edge)
        rows_html += f"""
<tr style="border-bottom:1px solid #f3f4f6">
  <td style="padding:8px 12px;font-weight:600">{html_module.escape(str(r['player']))}</td>
  <td style="padding:8px 12px">{html_module.escape(str(r.get('team','')))} vs {html_module.escape(str(r.get('opponent','')))}</td>
  <td style="padding:8px 12px;color:#6b7280">{html_module.escape(str(r.get('pos_group','')))} / {html_module.escape(str(r.get('pos_side','')))}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{r['lr_prob']:.1%}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{f'{mkt:.1%}' if not pd.isna(mkt) else '—'}</td>
  <td style="padding:8px 12px;text-align:center;font-weight:600;color:{ec}">{f'+{edge:.1%}' if not pd.isna(edge) else '—'}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{fmt_odds(r.get('prop_median_price_under', float('nan')))}</td>
  <td style="padding:8px 12px;text-align:center;font-weight:600;color:#1d2d44">Under 0.5</td>
</tr>"""

    no_bets_msg = "" if len(bets) else '<p style="color:#6b7280">No qualifying bets today (prob &lt; 0.30 threshold not met for any player).</p>'

    table = "" if not len(bets) else f"""
<table style="width:100%;border-collapse:collapse;font-size:13px">
<thead><tr style="background:#1d2d44;color:#fff">
  <th style="padding:9px 12px;text-align:left">Player</th>
  <th style="padding:9px 12px;text-align:left">Matchup</th>
  <th style="padding:9px 12px;text-align:left">Pos</th>
  <th style="padding:9px 12px;text-align:center">Model P(Over)</th>
  <th style="padding:9px 12px;text-align:center">Mkt P(Over)</th>
  <th style="padding:9px 12px;text-align:center">Edge</th>
  <th style="padding:9px 12px;text-align:center">Under Odds</th>
  <th style="padding:9px 12px;text-align:center">Bet</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>NFL Sacks — {gameday}</title></head>
<body style="margin:0;padding:16px;background:#f4f4f5;font-family:{_SANS};font-size:13px;color:#1a1a1a">
<div style="max-width:700px;margin:0 auto;background:#fff;padding:24px;border-radius:8px;border:1px solid #e2e2e4">
  <h2 style="font-size:18px;margin:0 0 4px">NFL Sacks — {gameday}</h2>
  <p style="color:#6b7280;font-size:12px;margin:0 0 20px">
    Generated {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')} &nbsp;|&nbsp;
    {len(df)} players with props &nbsp;|&nbsp; {len(bets)} qualifying bet{'s' if len(bets) != 1 else ''}
  </p>
  {no_bets_msg}{table}
  <p style="color:#9ca3af;font-size:11px;margin-top:20px">
    Strategy: Bet Under 0.5 sacks when model P(Over) &lt; 0.30 &nbsp;|&nbsp;
    OOS: +71.8u, +4.0% ROI (2024–2025, 1,779 bets)
  </p>
</div>
</body>
</html>"""


# ── SES / SNS send ──────────────────────────────────────────────────────────────

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

    if SNS_TOPIC_ARN:
        boto3.client("sns").publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject[:100],
            Message=text_body[:256_000],
        )
        print(f"  SNS published: {subject}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", type=str, default=None,
                        help="Gameday to settle (YYYY-MM-DD, default: yesterday ET)")
    args = parser.parse_args()
    gameday = args.gameday or yesterday_et()
    season  = current_nfl_season()

    print(f"\nNFL Sacks Settlement — gameday={gameday}  season={season}")
    print(f"{'='*55}")

    # ── 1. Load bet sheet for this gameday ─────────────────────────────────────
    bet_key  = s3_key_bet_sheet(gameday)
    bets_df  = load_s3_csv(bet_key)
    had_bets = bets_df is not None and len(bets_df) > 0
    had_games = False

    settled_yesterday = None
    if had_bets:
        placed_bets = bets_df[bets_df.get("bet", pd.Series(dtype=bool)).astype(bool)] if "bet" in bets_df.columns else pd.DataFrame()
        print(f"  Bet sheet loaded: {len(bets_df)} rows  ({len(placed_bets)} bets placed)")

        # ── 2. Get actual sack results ─────────────────────────────────────────
        print(f"  Fetching sack results for {gameday}...")
        actuals = get_sack_results_for_date(gameday, season)
        had_games = len(actuals) > 0 or len(placed_bets) > 0

        if actuals.empty:
            print(f"  No PBP sack data found for {gameday} — may not be settled yet.")

        # ── 3. Settle ──────────────────────────────────────────────────────────
        settled_yesterday = settle_bets(bets_df, actuals)
        save_s3_csv(s3_key_settled(gameday), settled_yesterday)
        print(f"  Settled CSV saved → s3://{S3_BUCKET}/{s3_key_settled(gameday)}")

        ys = compute_summary(settled_yesterday)
        print(f"  Yesterday: {ys['n_win']}W {ys['n_loss']}L {ys['n_push']}P  "
              f"P&L={ys['pnl']:+.3f}u  ROI={ys['roi']:+.1%}")
    else:
        print(f"  No bet sheet found for {gameday} (no games or no props that day)")

    # ── 4. All-time ────────────────────────────────────────────────────────────
    print("  Loading all-time settled results...")
    all_time = load_all_time_settled()
    if not all_time.empty:
        at = compute_summary(all_time)
        print(f"  All-time: {at['n_win']}W {at['n_loss']}L {at['n_push']}P  "
              f"P&L={at['pnl']:+.3f}u  ROI={at['roi']:+.1%}  ({at['n_bets']} bets)")

    # ── 5. Email ───────────────────────────────────────────────────────────────
    ys   = compute_summary(settled_yesterday) if settled_yesterday is not None else {"n_win": 0, "n_loss": 0, "n_push": 0, "pnl": 0.0, "roi": float("nan"), "n_bets": 0}
    at   = compute_summary(all_time) if not all_time.empty else {"n_win": 0, "n_loss": 0, "n_push": 0, "pnl": 0.0, "roi": float("nan"), "n_bets": 0}

    if had_games and ys["n_bets"] > 0:
        pnl_s = f"{ys['pnl']:+.2f}u"
        subject = f"NFL Sacks Settled — {gameday} — {ys['n_win']}W {ys['n_loss']}L {ys['n_push']}P ({pnl_s})"
    else:
        subject = f"NFL Sacks — {gameday} — No games yesterday (all-time: {at['n_bets']} bets, {at['pnl']:+.2f}u)"

    html_body = build_settlement_html(gameday, settled_yesterday, all_time, had_games)
    text_body = (
        f"NFL Sacks Settlement — {gameday}\n\n"
        f"Yesterday: {ys['n_win']}W {ys['n_loss']}L {ys['n_push']}P  P&L={ys['pnl']:+.3f}u\n"
        f"All-time : {at['n_win']}W {at['n_loss']}L {at['n_push']}P  P&L={at['pnl']:+.3f}u  ROI={at['roi']:+.1%}  ({at['n_bets']} bets)\n"
    )

    send_email(subject, html_body, text_body)

    print(f"\n{'='*55}")
    print(f"  Settlement complete — {gameday}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
