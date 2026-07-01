"""
Settle NFL rush attempts bets for a given gameday and update running S3 parquet.

Settlement logic (UNDER offered_line):
  Win  : actual_carries < offered_line
  Loss : actual_carries >= offered_line
  Push : actual_carries == offered_line (only if line is integer — very rare)

Reads:
  s3://the-odds-api-mt/nfl/rush_attempts_model/daily_runs/{gameday}/recommendations.csv

Writes / appends:
  s3://the-odds-api-mt/nfl/rush_attempts_model/settled/settled_bets.parquet

Sends SES HTML + SNS settlement email.

Run:
  python src/nfl_rush_attempts_modeling/scripts/settle_rush_attempts.py --gameday 2026-09-14
  python src/nfl_rush_attempts_modeling/scripts/settle_rush_attempts.py  # defaults to yesterday ET
"""

from __future__ import annotations

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
REPO_ROOT   = SCRIPTS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from step2_build_spine import _norm

S3_BUCKET     = "the-odds-api-mt"
S3_PREFIX     = "nfl/rush_attempts_model"
SES_SOURCE    = os.environ.get("SETTLEMENT_SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SETTLEMENT_SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()
ENABLE_SNS    = os.environ.get("ENABLE_SNS", "").strip().lower() in ("1", "true", "yes")

ET = ZoneInfo("America/New_York")
warnings.filterwarnings("ignore")

_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"


# ── Date helpers ─────────────────────────────────────────────────────────────

def yesterday_et() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")


def current_nfl_season() -> int:
    now = datetime.now(ET)
    return now.year if now.month >= 8 else now.year - 1


# ── S3 helpers ────────────────────────────────────────────────────────────────

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


# ── Actual rush attempt results ───────────────────────────────────────────────

def get_actual_carries(gameday: str, season: int) -> pd.DataFrame:
    """
    Pull actual carry counts for the given gameday from nfl_data_py.
    Returns: player_norm, player_name, team, actual_carries.
    """
    import nfl_data_py as nfl

    schedule = nfl.import_schedules([season])
    schedule = schedule[schedule["game_type"] == "REG"].copy()
    schedule["gameday"] = pd.to_datetime(schedule["gameday"]).dt.strftime("%Y-%m-%d")
    weeks_on_day = schedule[schedule["gameday"] == gameday]["week"].unique().tolist()

    if not weeks_on_day:
        print(f"  No schedule entries found for {gameday} (season {season})")
        return pd.DataFrame()
    print(f"  Week(s) on {gameday}: {weeks_on_day}")

    # Try nfl.import_weekly_data first (includes 0-carry players)
    try:
        weekly = nfl.import_weekly_data(years=[season])
        weekly = weekly[weekly["week"].isin(weeks_on_day)].copy()
        if not weekly.empty:
            weekly["actual_carries"] = weekly["carries"].fillna(0).astype(int)
            weekly["player_norm"]    = weekly["player_display_name"].apply(_norm)
            return weekly[["player_norm", "player_display_name", "recent_team",
                           "week", "actual_carries"]].rename(
                columns={"player_display_name": "player_name",
                         "recent_team": "team"}
            ).copy()
    except Exception:
        pass

    # Fallback: PFR rush data
    pfr = nfl.import_weekly_pfr(s_type="rush", years=[season])
    pfr = pfr[pfr["week"].isin(weeks_on_day)].copy()
    if pfr.empty:
        print(f"  No PFR data for weeks {weeks_on_day} (may not be posted yet)")
        return pd.DataFrame()

    pfr["actual_carries"] = pfr["carries"].fillna(0).astype(int)
    pfr["player_norm"]    = pfr["pfr_player_name"].apply(_norm)
    return pfr[["player_norm", "pfr_player_name", "team",
               "week", "actual_carries"]].rename(
        columns={"pfr_player_name": "player_name"}
    ).copy()


# ── Settlement logic ──────────────────────────────────────────────────────────

def settle(recs: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Match recommendations to actuals on player_norm.
    Win = actual_carries < offered_line (UNDER).
    Push = actual_carries == offered_line (integer line only).
    """
    merged = recs.merge(
        actuals[["player_norm", "actual_carries", "week"]],
        on="player_norm",
        how="left",
    )

    def _settle_row(r):
        actual = r.get("actual_carries", float("nan"))
        line   = r.get("line", r.get("offered_line", float("nan")))
        if pd.isna(actual):
            return "unmatched"
        if actual == line:
            return "push"
        return "win" if actual < line else "loss"

    merged["actual_carries"] = merged["actual_carries"].where(
        merged["actual_carries"].notna(), other=np.nan
    )
    merged["outcome"] = merged.apply(_settle_row, axis=1)
    merged["hit"] = (merged["outcome"] == "win").where(
        ~merged["outcome"].isin(["unmatched", "push"]), other=np.nan
    )
    return merged


def american_to_payout(price: float) -> float:
    """Units won per 1 unit risked."""
    return 100.0 / abs(price) if price < 0 else price / 100.0


def compute_summary(df: pd.DataFrame) -> dict:
    settled = df[df["outcome"].isin(["win", "loss"])]
    n_win   = (settled["outcome"] == "win").sum()
    n_loss  = (settled["outcome"] == "loss").sum()
    n_push  = (df["outcome"] == "push").sum() if "outcome" in df.columns else 0
    n_bets  = n_win + n_loss
    hit_rate = n_win / n_bets if n_bets > 0 else float("nan")
    price_col = "offered_price" if "offered_price" in settled.columns else "under_price"
    wins   = settled[settled["outcome"] == "win"]
    pnl    = wins[price_col].apply(american_to_payout).sum() - n_loss
    roi    = pnl / n_bets if n_bets > 0 else float("nan")
    return {
        "n_bets":   n_bets,
        "n_win":    n_win,
        "n_loss":   n_loss,
        "n_push":   n_push,
        "hit_rate": hit_rate,
        "pnl":      pnl,
        "roi":      roi,
    }


# ── HTML email ────────────────────────────────────────────────────────────────

def _outcome_badge(outcome: str) -> str:
    styles = {
        "win":       "background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:4px;font-weight:600",
        "loss":      "background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:4px;font-weight:600",
        "push":      "background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:4px;font-weight:600",
        "unmatched": "background:#f3f4f6;color:#6b7280;padding:2px 8px;border-radius:4px",
    }
    labels = {"win": "WIN", "loss": "LOSS", "push": "PUSH", "unmatched": "N/A"}
    return f'<span style="{styles.get(outcome, "")}">{labels.get(outcome, outcome.upper())}</span>'


def _pnl_cell(outcome: str, price: float) -> str:
    if outcome == "win":
        val = american_to_payout(price)
        return (f'<td style="text-align:right;padding:8px 12px;font-family:{_MONO};'
                f'font-weight:600;color:#065f46">+{val:.3f}u</td>')
    if outcome == "loss":
        return (f'<td style="text-align:right;padding:8px 12px;font-family:{_MONO};'
                f'font-weight:600;color:#991b1b">−1.000u</td>')
    if outcome == "push":
        return (f'<td style="text-align:right;padding:8px 12px;font-family:{_MONO};'
                f'color:#92400e">0.000u</td>')
    return f'<td style="text-align:right;padding:8px 12px;color:#9ca3af">—</td>'


def _summary_box(label: str, s: dict) -> str:
    pnl   = s["pnl"]
    pnl_c = "#065f46" if pnl >= 0 else "#991b1b"
    pnl_s = f"+{pnl:.2f}u" if pnl >= 0 else f"{pnl:.2f}u"
    hit_s = f"{s['hit_rate']:.1%}" if not pd.isna(s.get("hit_rate", float("nan"))) else "—"
    roi   = s.get("roi", float("nan"))
    roi_s = f"+{roi:.1%}" if not pd.isna(roi) and roi >= 0 else (f"{roi:.1%}" if not pd.isna(roi) else "—")
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


def build_settlement_html(
    gameday: str,
    today_settled: pd.DataFrame | None,
    all_time_summary: dict,
    had_games: bool,
) -> str:
    today_summary = compute_summary(today_settled) if today_settled is not None and len(today_settled) else {
        "n_bets": 0, "n_win": 0, "n_loss": 0, "n_push": 0, "hit_rate": float("nan"),
        "pnl": 0.0, "roi": float("nan"),
    }

    if had_games and today_settled is not None and len(today_settled):
        settled_rows = today_settled[today_settled["outcome"].isin(["win", "loss", "push", "unmatched"])].copy()
        settled_rows["_order"] = settled_rows["outcome"].map({"win": 0, "loss": 1, "push": 2, "unmatched": 3})
        settled_rows = settled_rows.sort_values(["_order", "edge"]).drop(columns=["_order"])

        price_col = "offered_price" if "offered_price" in settled_rows.columns else "under_price"
        rows_html = ""
        for _, r in settled_rows.iterrows():
            actual_s  = f"{r['actual_carries']:.0f}" if not pd.isna(r.get("actual_carries", float("nan"))) else "—"
            price     = float(r.get(price_col, float("nan")))
            odds_s    = f"{int(price):+d}" if not pd.isna(price) else "—"
            mkt_p_s   = f"{float(r['p_market'])*100:.1f}%" if not pd.isna(r.get("p_market", float("nan"))) else "—"
            mdl_p_s   = f"{float(r['p_model'])*100:.1f}%" if not pd.isna(r.get("p_model", float("nan"))) else "—"
            edge_s    = f"+{abs(float(r['edge']))*100:.1f}pp" if not pd.isna(r.get("edge", float("nan"))) else "—"
            rows_html += f"""
<tr style="border-bottom:1px solid #f3f4f6">
  <td style="padding:8px 12px;font-weight:600">{html_module.escape(str(r.get('player_display_name', '')))}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{r.get('line', r.get('offered_line', '—'))}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{actual_s}</td>
  <td style="padding:8px 12px;text-align:center">
    <span style="background:#dbeafe;color:#1e40af;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700">UNDER</span>
  </td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{odds_s}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{mkt_p_s}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{mdl_p_s}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{edge_s}</td>
  <td style="padding:8px 12px;text-align:center">{_outcome_badge(r['outcome'])}</td>
  {_pnl_cell(r['outcome'], price)}
</tr>"""

        today_section = f"""
<h3 style="font-size:14px;font-weight:600;margin:20px 0 8px;color:#111827">Yesterday ({gameday})</h3>
{_summary_box(f"Results — {gameday}", today_summary)}
<table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px">
<thead><tr style="background:#1d2d44;color:#fff">
  <th style="padding:9px 12px;text-align:left">Player</th>
  <th style="padding:9px 12px;text-align:center">Line</th>
  <th style="padding:9px 12px;text-align:center">Actual</th>
  <th style="padding:9px 12px;text-align:center">Side</th>
  <th style="padding:9px 12px;text-align:center">Odds</th>
  <th style="padding:9px 12px;text-align:center">Mkt P(over)</th>
  <th style="padding:9px 12px;text-align:center">Model P(over)</th>
  <th style="padding:9px 12px;text-align:center">Edge</th>
  <th style="padding:9px 12px;text-align:center">Outcome</th>
  <th style="padding:9px 12px;text-align:right">P&amp;L</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""
    elif not had_games:
        today_section = f'<p style="color:#6b7280;font-size:13px;margin-bottom:20px">No NFL games on {gameday}.</p>'
    else:
        today_section = f'<p style="color:#6b7280;font-size:13px;margin-bottom:20px">No qualifying bets were placed on {gameday}.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>NFL Rush Attempts Settlement — {gameday}</title></head>
<body style="margin:0;padding:16px;background:#f4f4f5;font-family:{_SANS};font-size:13px;color:#1a1a1a">
<div style="max-width:700px;margin:0 auto;background:#fff;padding:24px;border-radius:8px;border:1px solid #e2e2e4">
  <h2 style="font-size:18px;margin:0 0 4px">NFL Rush Attempts Settlement — QB Unders</h2>
  <p style="color:#6b7280;font-size:12px;margin:0 0 20px">Generated {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}</p>
  {today_section}
  <h3 style="font-size:14px;font-weight:600;margin:20px 0 8px;color:#111827">All-Time</h3>
  {_summary_box("All-Time Results (QB UNDER, line ≥ 6.5, edge ≥ 3pp)", all_time_summary)}
</div>
</body>
</html>"""


# ── SES / SNS ─────────────────────────────────────────────────────────────────

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
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject[:100],
            Message=text_body[:256_000],
        )
        print(f"  SNS published: {subject}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", type=str, default=None,
                        help="Gameday to settle (YYYY-MM-DD, default: yesterday ET)")
    args    = parser.parse_args()
    gameday = args.gameday or yesterday_et()
    season  = current_nfl_season()

    print(f"\nNFL Rush Attempts Settlement — gameday={gameday}  season={season}")
    print("=" * 60)

    # ── 1. Load recommendations ────────────────────────────────────────────
    rec_key = f"{S3_PREFIX}/daily_runs/{gameday}/recommendations.csv"
    recs    = load_s3_csv(rec_key)
    had_bets = recs is not None and len(recs) > 0

    today_settled = None
    had_games     = False

    if had_bets:
        recs["player_norm"] = recs["player_norm"].fillna(
            recs.get("player_display_name", pd.Series()).apply(_norm)
        )
        print(f"  Recommendations loaded: {len(recs)} bets")

        # ── 2. Pull actual carries ─────────────────────────────────────────
        print(f"  Fetching actual carries for {gameday}...")
        actuals   = get_actual_carries(gameday, season)
        had_games = not actuals.empty

        if actuals.empty:
            print("  No carry data yet — may not be posted. Nothing settled.")
        else:
            print(f"  Actuals: {len(actuals)} player rows")

            # ── 3. Settle ──────────────────────────────────────────────────
            today_settled = settle(recs, actuals)
            s_sum         = compute_summary(today_settled)
            n_unmatched   = (today_settled["outcome"] == "unmatched").sum()
            n_push        = (today_settled["outcome"] == "push").sum()
            print(f"  Settled: {s_sum['n_win']}W  {s_sum['n_loss']}L  "
                  f"{n_push} push  {n_unmatched} unmatched  P&L={s_sum['pnl']:+.3f}u")

            # ── 4. Append to running parquet ───────────────────────────────
            print("  Loading existing settled history...")
            history = load_settled_parquet()
            if not history.empty and "gameday" in history.columns:
                history = history[history["gameday"] != gameday].copy()

            new_rows = today_settled[today_settled["outcome"].isin(["win", "loss", "push"])].copy()
            new_rows["gameday"] = gameday
            new_rows["season"]  = season

            combined = pd.concat([history, new_rows], ignore_index=True)
            combined = combined.sort_values(["player_norm", "season"]).reset_index(drop=True)
            save_settled_parquet(combined)
            print(f"  Settled parquet: {len(combined):,} total rows")
    else:
        print(f"  No recommendations CSV found for {gameday}")
        combined = load_settled_parquet()

    # ── 5. All-time summary ────────────────────────────────────────────────
    all_time_df = combined if (had_bets and had_games) else load_settled_parquet()
    at = compute_summary(all_time_df) if not all_time_df.empty else {
        "n_bets": 0, "n_win": 0, "n_loss": 0, "n_push": 0,
        "hit_rate": float("nan"), "pnl": 0.0, "roi": float("nan"),
    }
    print(f"  All-time: {at['n_win']}W {at['n_loss']}L  "
          f"P&L={at['pnl']:+.3f}u  ROI={at['roi']:+.1%}  ({at['n_bets']} bets)")

    # ── 6. Email ───────────────────────────────────────────────────────────
    ts = compute_summary(today_settled) if today_settled is not None and len(today_settled) else {
        "n_bets": 0, "n_win": 0, "n_loss": 0, "pnl": 0.0,
    }
    if had_games and ts["n_bets"] > 0:
        pnl_s   = f"{ts['pnl']:+.2f}u"
        subject = (f"NFL Rush Attempts Settled — {gameday} — "
                   f"{ts['n_win']}W {ts['n_loss']}L ({pnl_s})")
    else:
        subject = (f"NFL Rush Attempts — {gameday} — No games settled "
                   f"(all-time: {at['n_bets']} bets, {at['pnl']:+.2f}u)")

    html_body  = build_settlement_html(gameday, today_settled, at, had_games)
    text_body  = (
        f"NFL Rush Attempts Settlement — {gameday}\n\n"
        f"Yesterday : {ts['n_win']}W {ts['n_loss']}L  P&L={ts['pnl']:+.3f}u\n"
        f"All-time  : {at['n_win']}W {at['n_loss']}L  P&L={at['pnl']:+.3f}u  "
        f"ROI={at['roi']:+.1%}  ({at['n_bets']} bets)\n"
    )
    send_email(subject, html_body, text_body)

    print(f"\n{'='*60}")
    print(f"  Settlement complete — {gameday}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
