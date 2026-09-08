"""
Settle MLB batter strikeouts bets for a given gameday.

Settlement (UNDER 1.5):
  Win : strikeouts <= 1  → +(under_price - 1) units
  Loss: strikeouts >= 2  → -1 unit
  Push: strikeouts == 1.5 — impossible for integer K count

Reads:   s3://the-odds-api-mt/mlb/batter_strikeouts_model/daily_runs/{gameday}/recommendations.csv
Writes:  s3://the-odds-api-mt/mlb/batter_strikeouts_model/daily_runs/{gameday}/settled.csv
         s3://the-odds-api-mt/mlb/batter_strikeouts_model/settled/mlb_bs_settled_bets.parquet

Usage:
  python src/mlb_batter_strikeouts_modeling/scripts/settle_batter_strikeouts.py
  python src/mlb_batter_strikeouts_modeling/scripts/settle_batter_strikeouts.py --gameday 2026-07-10
"""
from __future__ import annotations

import argparse
import html as html_module
import os
import re
import sys
import time
import unicodedata
import warnings
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import botocore.exceptions
import numpy as np
import pandas as pd
import pybaseball as pb

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

pb.cache.enable()

S3_BUCKET    = "the-odds-api-mt"
DAILY_PREFIX = "mlb/batter_strikeouts_model/daily_runs"
SETTLED_KEY  = "mlb/batter_strikeouts_model/settled/mlb_bs_settled_bets.parquet"

SES_SOURCE    = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()

ET = ZoneInfo("America/New_York")

_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"

K_EVENTS = {"strikeout", "strikeout_double_play"}
PA_EVENTS = {
    "single", "double", "triple", "home_run",
    "strikeout", "strikeout_double_play",
    "field_out", "force_out", "grounded_into_double_play",
    "double_play", "triple_play", "field_error",
    "fielders_choice", "fielders_choice_out",
    "walk", "hit_by_pitch", "sac_fly", "sac_bunt", "intent_walk",
}

NAME_MAP = {
    "daniel vogelbach": "Dan Vogelbach",
    "michael a taylor": "Michael Taylor",
}


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[-]", " ", name)
    name = re.sub(r"[.,']", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    name = re.sub(r"\b([a-z])\s([a-z])\b", r"\1\2", name)
    name = re.sub(r"\b([a-z])\b(?=\s+[a-z]{2})", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def fetch_actuals(gameday: str) -> pd.DataFrame:
    """Fetch Statcast for gameday and aggregate K counts."""
    print(f"  Fetching Statcast for {gameday}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pb.statcast(start_dt=gameday, end_dt=gameday)
    time.sleep(0.5)

    if raw.empty:
        return pd.DataFrame()

    raw = raw[raw["game_type"] == "R"].copy()
    raw["game_date"] = pd.to_datetime(raw["game_date"]).dt.date.astype(str)
    events = raw[raw["events"].notna()].copy()
    events["k_count"] = events["events"].isin(K_EVENTS).astype(int)
    events["pa"]      = events["events"].isin(PA_EVENTS).astype(int)

    agg = events.groupby(["game_date", "batter", "game_pk"]).agg(
        strikeouts  = ("k_count", "sum"),
        pa          = ("pa", "sum"),
    ).reset_index()

    # Name lookup
    batter_ids = agg["batter"].dropna().unique().tolist()
    nl = pb.playerid_reverse_lookup(batter_ids, key_type="mlbam")
    nl["player_name"] = nl["name_first"].str.title() + " " + nl["name_last"].str.title()
    agg = agg.merge(nl[["key_mlbam", "player_name"]].rename(columns={"key_mlbam": "batter"}),
                    on="batter", how="left")

    agg["name_norm"] = agg["player_name"].map(normalize_name)
    name_map_norm = {normalize_name(k): normalize_name(v) for k, v in NAME_MAP.items()}
    agg["name_norm"] = agg["name_norm"].map(lambda n: name_map_norm.get(n, n))

    return agg.groupby("name_norm").agg(
        strikeouts_actual = ("strikeouts", "sum"),
        pa_actual         = ("pa", "sum"),
    ).reset_index()


def load_recommendations(gameday: str) -> pd.DataFrame | None:
    key = f"{DAILY_PREFIX}/{gameday}/recommendations.csv"
    try:
        body = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        return pd.read_csv(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def load_settled_history() -> pd.DataFrame:
    try:
        body = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=SETTLED_KEY)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return pd.DataFrame()
        raise


def settle_row(row: pd.Series, actuals_map: dict) -> tuple[str, float]:
    name_norm = normalize_name(row.get("player_name", ""))
    actual = actuals_map.get(name_norm)
    line   = float(row.get("offered_line", 1.5))
    under_price = float(row.get("under_price", 1.0))

    if actual is None:
        return "DNP", 0.0
    if actual < line:
        return "WIN", round(under_price - 1, 4)
    return "LOSS", -1.0


def _to_am(d: object) -> str:
    try:
        d = float(d)
        return f"+{int((d-1)*100)}" if d >= 2 else str(int(-100/(d-1)))
    except Exception:
        return "—"


def _tile(value: str, label: str, color: str = "#1a3a5c") -> str:
    return (
        f"<div style='background:#fff;border:1px solid #e5e7eb;border-radius:8px;"
        f"padding:14px 20px;text-align:center;min-width:90px'>"
        f"<div style='font-size:22px;font-weight:700;color:{color}'>{value}</div>"
        f"<div style='font-size:11px;color:#888;margin-top:3px'>{label}</div>"
        "</div>"
    )


def build_settle_html(settled: pd.DataFrame, gameday: str, history: pd.DataFrame) -> str:
    plays = settled[settled["tier"] == "play"].copy()
    yest_wins  = int((plays["result"] == "WIN").sum())
    yest_loss  = int((plays["result"] == "LOSS").sum())
    yest_units = float(plays["units"].sum())

    all_hist = pd.concat([history, plays], ignore_index=True) if not history.empty else plays.copy()
    s_wins   = int((all_hist["result"] == "WIN").sum())
    s_loss   = int((all_hist["result"] == "LOSS").sum())
    s_units  = float(all_hist["units"].sum())
    s_bets   = len(all_hist)
    s_roi    = (s_units / s_bets * 100) if s_bets > 0 else 0.0
    s_wr     = (s_wins / s_bets) if s_bets > 0 else 0.0
    s_mdd    = 0.0
    if s_bets > 0:
        profit_arr = all_hist["units"].fillna(0).values
        cumul      = np.cumsum(profit_arr)
        peak       = np.maximum.accumulate(cumul)
        s_mdd      = float((peak - cumul).max())

    color_u = "#2e7d32" if yest_units >= 0 else "#c62828"

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>body{{font-family:{_SANS};font-size:14px;background:#f5f5f5;margin:0;padding:20px;color:#1a1a1a}}</style>"
        "</head><body>"
    )

    # ── Section 2: Yesterday's Results ──────────────────────────────────────
    html += (
        f"<hr style='margin:32px 0;border:none;border-top:2px solid #e5e7eb'>"
        f"<h2 style='font-size:16px;font-weight:700;margin:0 0 4px;color:#111827'>"
        f"Yesterday's Results — {gameday}</h2>"
        "<div style='display:flex;gap:24px;padding:14px 0 20px;flex-wrap:wrap'>"
        + _tile(str(len(plays)), "Bets")
        + _tile(f"{yest_wins}W", "Won", "#2e7d32")
        + _tile(f"{yest_loss}L", "Lost", "#c62828")
        + _tile(f"{yest_units:+.2f}u", "Units", color_u)
        + (f"<div style='background:#fff;border:1px solid #e5e7eb;border-radius:8px;"
           f"padding:14px 20px;text-align:center;min-width:90px'>"
           f"<div style='font-size:22px;font-weight:700;color:{color_u}'>"
           f"{(yest_units/len(plays)*100):+.1f}%</div>"
           f"<div style='font-size:11px;color:#888;margin-top:3px'>ROI</div></div>"
           if len(plays) > 0 else "")
        + "</div>"
    )

    # Bet-by-bet table (grouped by game)
    html += "<h3 style='font-size:13px;font-weight:600;margin:0 0 6px;color:#374151'>Bet-by-Bet</h3>"
    html += (
        "<table style='border-collapse:collapse;font-size:12px;width:100%;background:#fff;margin-bottom:20px'>"
        "<thead><tr style='background:#1d2d44;color:#fff'>"
        "<th style='padding:8px 12px;text-align:left'>Player</th>"
        "<th style='padding:8px 12px;text-align:left'>Team</th>"
        "<th style='padding:8px 12px;text-align:left'>Opponent</th>"
        "<th style='padding:8px 12px;text-align:center'>Line</th>"
        "<th style='padding:8px 12px;text-align:left'>Book</th>"
        "<th style='padding:8px 12px;text-align:center'>Under Odds</th>"
        "<th style='padding:8px 12px;text-align:center'>Edge</th>"
        "<th style='padding:8px 12px;text-align:center'>Actual K</th>"
        "<th style='padding:8px 12px;text-align:center'>Outcome</th>"
        "<th style='padding:8px 12px;text-align:right'>P&amp;L</th>"
        "</tr></thead><tbody>"
    )

    # Group by game for context header rows
    game_col = "home_team" if "home_team" in plays.columns else None
    if game_col and "away_team" in plays.columns:
        plays_sorted = plays.sort_values(["away_team", "home_team", "edge_under"], ascending=[True, True, False])
        last_game = None
        for _, row in plays_sorted.iterrows():
            game_key = (row.get("away_team", ""), row.get("home_team", ""))
            if game_key != last_game:
                last_game = game_key
                away, home = game_key
                game_w = int((plays[(plays["away_team"]==away) & (plays["home_team"]==home)]["result"] == "WIN").sum())
                game_l = int((plays[(plays["away_team"]==away) & (plays["home_team"]==home)]["result"] == "LOSS").sum())
                game_u = float(plays[(plays["away_team"]==away) & (plays["home_team"]==home)]["units"].sum())
                gu_c   = "#166534" if game_u >= 0 else "#991b1b"
                bg_g   = "#f0fdf4" if game_u > 0 else ("#fee2e2" if game_u < 0 else "#fff3e0")
                html += (
                    f"<tr style='background:{bg_g}'>"
                    f"<td colspan='10' style='padding:6px 12px;font-weight:600;font-size:11px;color:{gu_c}'>"
                    f"{html_module.escape(away)} @ {html_module.escape(home)} — "
                    f"{game_w}W / {game_l}L · {game_u:+.2f}u</td></tr>"
                )
            res     = row.get("result", "?")
            res_cls = {"WIN": "#dcfce7;color:#166534", "LOSS": "#fee2e2;color:#991b1b", "DNP": "#f3f4f6;color:#888"}.get(res, "")
            actual  = row.get("strikeouts_actual")
            actual_s = str(int(actual)) if pd.notna(actual) else "—"
            edge_s  = f"{row['edge_under']*100:.1f}pp" if pd.notna(row.get("edge_under")) else "—"
            units_s = f"{row['units']:+.3f}u" if pd.notna(row.get("units")) else "—"
            u_color = "#2e7d32" if row.get("units", 0) > 0 else "#c62828"
            html += (
                f"<tr style='border-bottom:1px solid #f3f4f6'>"
                f"<td style='padding:7px 12px'>{html_module.escape(str(row.get('player_name','?')))}</td>"
                f"<td style='padding:7px 12px'>{html_module.escape(str(row.get('away_team','?')))}</td>"
                f"<td style='padding:7px 12px'>{html_module.escape(str(row.get('home_team','?')))}</td>"
                f"<td style='padding:7px 12px;text-align:center'>{row.get('offered_line','?')}</td>"
                f"<td style='padding:7px 12px'>{html_module.escape(str(row.get('bookmaker','?')))}</td>"
                f"<td style='padding:7px 12px;text-align:center'>{_to_am(row.get('under_price',1.0))}</td>"
                f"<td style='padding:7px 12px;text-align:center'>{edge_s}</td>"
                f"<td style='padding:7px 12px;text-align:center'>{actual_s}</td>"
                f"<td style='padding:7px 12px;text-align:center'>"
                f"<span style='background:{res_cls.split(';')[0] if ';' in res_cls else '#f3f4f6'};"
                f"{res_cls.split(';')[1] if ';' in res_cls else 'color:#888'};"
                f"border-radius:4px;padding:2px 8px;font-size:11px'>{res}</span></td>"
                f"<td style='padding:7px 12px;text-align:right;color:{u_color};font-weight:600'>{units_s}</td>"
                "</tr>"
            )
    else:
        for _, row in plays.sort_values("edge_under", ascending=False).iterrows():
            res     = row.get("result", "?")
            actual  = row.get("strikeouts_actual")
            actual_s = str(int(actual)) if pd.notna(actual) else "—"
            edge_s  = f"{row['edge_under']*100:.1f}pp" if pd.notna(row.get("edge_under")) else "—"
            units_s = f"{row['units']:+.3f}u" if pd.notna(row.get("units")) else "—"
            u_color = "#2e7d32" if row.get("units", 0) > 0 else "#c62828"
            html += (
                f"<tr style='border-bottom:1px solid #f3f4f6'>"
                f"<td style='padding:7px 12px'>{html_module.escape(str(row.get('player_name','?')))}</td>"
                f"<td style='padding:7px 12px' colspan='2'>—</td>"
                f"<td style='padding:7px 12px;text-align:center'>{row.get('offered_line','?')}</td>"
                f"<td style='padding:7px 12px'>{html_module.escape(str(row.get('bookmaker','?')))}</td>"
                f"<td style='padding:7px 12px;text-align:center'>{_to_am(row.get('under_price',1.0))}</td>"
                f"<td style='padding:7px 12px;text-align:center'>{edge_s}</td>"
                f"<td style='padding:7px 12px;text-align:center'>{actual_s}</td>"
                f"<td style='padding:7px 12px;text-align:center'>{res}</td>"
                f"<td style='padding:7px 12px;text-align:right;color:{u_color};font-weight:600'>{units_s}</td>"
                "</tr>"
            )

    html += "</tbody></table>"

    # ── Section 3: All-Time Live Record ──────────────────────────────────────
    html += (
        f"<hr style='margin:32px 0;border:none;border-top:2px solid #e5e7eb'>"
        f"<h2 style='font-size:16px;font-weight:700;margin:0 0 4px;color:#111827'>"
        "All-Time Record (Live Plays, edge ≥5pp)</h2>"
        "<p style='font-size:12px;color:#6b7280;margin:0 0 16px'>Settled production bets. Updated daily.</p>"
        "<div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px'>"
        + _tile(str(s_bets), "Total Bets")
        + _tile(f"{s_units:+.2f}u", "Units Won", "#2e7d32" if s_units >= 0 else "#c62828")
        + _tile(f"{s_roi:+.1f}%", "ROI", "#2e7d32" if s_roi >= 0 else "#c62828")
        + _tile(f"{s_wr:.1%}", "Win Rate")
        + _tile(f"{s_mdd:.1f}u", "Max Drawdown", "#c62828")
        + "</div>"
    )

    # Season-by-season from all_hist
    if "game_date" in all_hist.columns or "commence_time" in all_hist.columns:
        date_col = "game_date" if "game_date" in all_hist.columns else "commence_time"
        all_hist["_year"] = pd.to_datetime(all_hist[date_col], errors="coerce").dt.year
        html += (
            "<h3 style='font-size:13px;font-weight:600;margin:0 0 6px;color:#374151'>Season-by-Season (Live)</h3>"
            "<table style='border-collapse:collapse;font-size:12px;background:#fff;margin-bottom:8px'>"
            "<thead><tr style='background:#1d2d44;color:#fff'>"
            "<th style='padding:8px 16px;text-align:left'>Season</th>"
            "<th style='padding:8px 16px;text-align:center'>Bets</th>"
            "<th style='padding:8px 16px;text-align:center'>Win Rate</th>"
            "<th style='padding:8px 16px;text-align:right'>Units</th>"
            "<th style='padding:8px 16px;text-align:right'>ROI</th>"
            "</tr></thead><tbody>"
        )
        for yr, grp in all_hist.groupby("_year"):
            yr_n  = len(grp)
            yr_w  = int((grp["result"] == "WIN").sum())
            yr_u  = float(grp["units"].sum())
            yr_r  = yr_u / yr_n * 100 if yr_n > 0 else 0.0
            yr_wr = yr_w / yr_n if yr_n > 0 else 0.0
            uc    = "#2e7d32" if yr_u >= 0 else "#c62828"
            html += (
                f"<tr style='border-bottom:1px solid #f3f4f6'>"
                f"<td style='padding:7px 16px;font-weight:600'>{yr}</td>"
                f"<td style='padding:7px 16px;text-align:center'>{yr_n}</td>"
                f"<td style='padding:7px 16px;text-align:center'>{yr_wr:.1%}</td>"
                f"<td style='padding:7px 16px;text-align:right;color:{uc};font-weight:600'>{yr_u:+.2f}u</td>"
                f"<td style='padding:7px 16px;text-align:right;color:{uc}'>{yr_r:+.1f}%</td>"
                "</tr>"
            )
        uc_tot = "#1d4ed8"
        html += (
            f"<tr style='background:#e0e7ff;font-weight:700'>"
            f"<td style='padding:7px 16px'>Total</td>"
            f"<td style='padding:7px 16px;text-align:center'>{s_bets}</td>"
            f"<td style='padding:7px 16px;text-align:center'>{s_wr:.1%}</td>"
            f"<td style='padding:7px 16px;text-align:right;color:{uc_tot}'>{s_units:+.2f}u</td>"
            f"<td style='padding:7px 16px;text-align:right;color:{uc_tot}'>{s_roi:+.1f}%</td>"
            "</tr>"
            "</tbody></table>"
        )

    # ── Section 4: Backtest OOF ───────────────────────────────────────────────
    html += (
        f"<hr style='margin:32px 0;border:none;border-top:2px solid #e5e7eb'>"
        f"<h2 style='font-size:16px;font-weight:700;margin:0 0 4px;color:#111827'>Backtest Results (OOF)</h2>"
        "<p style='font-size:12px;color:#6b7280;margin:0 0 16px'>"
        "Out-of-fold grid search across 2024–2026 training data. Not production P&amp;L.</p>"
        "<div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px'>"
        + _tile("2,580", "OOF Bets")
        + _tile("+140.4u", "Units Won", "#2e7d32")
        + _tile("+5.44%", "ROI", "#2e7d32")
        + _tile("58.3%", "Win Rate")
        + _tile("3.02", "Calmar")
        + _tile("28 days", "Max DD Duration")
        + _tile("46.5u", "Max Drawdown", "#c62828")
        + "</div>"
        "<p style='font-size:11px;color:#9ca3af;margin-top:16px;padding-top:12px;border-top:1px solid #f3f4f6'>"
        "consensus_line · no_fliff · both · 1.5_only · edge≥5pp · "
        "Max DD: 2024-07-12→2024-08-09 (28 days) · "
        "Structural: books misprice OVER on batter 1.5 K line for low-K hitters"
        "</p>"
    )

    html += "</body></html>"
    return html


def main(gameday: str | None = None, output_path: str | None = None):
    if gameday is None:
        gameday = (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Settling batter_strikeouts | gameday={gameday}")
    s3 = boto3.client("s3")

    # 1. Load recs
    recs = load_recommendations(gameday)
    if recs is None or recs.empty:
        print(f"  No recommendations found for {gameday} — skipping settlement.")
        if output_path:
            import json
            Path(output_path).write_text(json.dumps({
                "yesterday_wins": 0, "yesterday_losses": 0, "yesterday_units": 0.0,
                "season_wins": 0, "season_losses": 0, "season_units": 0.0,
                "html_body": "<p>No bets to settle.</p>",
            }))
        return

    plays = recs[recs["tier"] == "play"].copy()
    print(f"  {len(plays)} play bets to settle")

    # 2. Fetch actuals
    actuals = fetch_actuals(gameday)
    actuals_map = dict(zip(actuals["name_norm"], actuals["strikeouts_actual"])) if not actuals.empty else {}

    # 3. Settle
    results = plays.apply(lambda r: settle_row(r, actuals_map), axis=1, result_type="expand")
    plays["result"] = results[0]
    plays["units"]  = results[1]
    plays["strikeouts_actual"] = plays["player_name"].map(
        lambda n: actuals_map.get(normalize_name(n))
    )

    w = (plays["result"] == "WIN").sum()
    l = (plays["result"] == "LOSS").sum()
    u = plays["units"].sum()
    print(f"  Settled: {w}W / {l}L  {u:+.2f}u")

    # 4. Save settled CSV
    settled_key = f"{DAILY_PREFIX}/{gameday}/settled.csv"
    buf = BytesIO()
    plays.to_csv(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=settled_key, Body=buf.getvalue())
    print(f"  Saved → s3://{S3_BUCKET}/{settled_key}")

    # 5. Update history
    history = load_settled_history()
    new_history = pd.concat([history, plays], ignore_index=True)
    buf2 = BytesIO()
    new_history.to_parquet(buf2, index=False)
    buf2.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=SETTLED_KEY, Body=buf2.getvalue())

    s_wins  = (new_history["result"] == "WIN").sum()
    s_loss  = (new_history["result"] == "LOSS").sum()
    s_units = new_history["units"].sum()

    # 6. Build + send email
    html_body = build_settle_html(plays, gameday, history)
    subject   = (
        f"MLB Strikeouts Settlement — {w}W/{l}L {u:+.2f}u · "
        f"Season: {s_wins}W/{s_loss}L {s_units:+.2f}u — {gameday}"
    )

    if SES_SOURCE and SES_TO_RAW:
        to_list = [e.strip() for e in SES_TO_RAW.split(",") if e.strip()]
        boto3.client("ses", region_name="us-east-2").send_email(
            Source=SES_SOURCE,
            Destination={"ToAddresses": to_list},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Html": {"Data": html_body, "Charset": "UTF-8"}},
            },
        )
        print(f"  Email sent: {subject[:80]}")

    if output_path:
        import json
        Path(output_path).write_text(json.dumps({
            "yesterday_wins":   int(w),
            "yesterday_losses": int(l),
            "yesterday_units":  round(float(u), 4),
            "season_wins":      int(s_wins),
            "season_losses":    int(s_loss),
            "season_units":     round(float(s_units), 4),
            "html_body":        html_body,
        }))

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", type=str, default=None)
    parser.add_argument("--output",  type=str, default=None)
    args = parser.parse_args()
    main(gameday=args.gameday, output_path=args.output)
