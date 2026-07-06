"""
Settle MLB total-bases bets for a given gameday and send summary email.

Settlement logic (UNDER 1.5 total bases):
  Win : total_bases <= 1  → +units_to_win at recorded dec_odds_under - 1
  Loss: total_bases >= 2  → -1 unit

Reads from S3:
  mlb/total_bases_model/daily_runs/{gameday}/recommendations.csv

Writes to S3:
  mlb/total_bases_model/daily_runs/{gameday}/settled.csv

Sends SES HTML email with yesterday P&L + all-time summary.

Actuals come from Statcast via pybaseball (covers games through yesterday).

Usage:
  python src/mlb_total_bases_modeling/scripts/settle_total_bases.py
  python src/mlb_total_bases_modeling/scripts/settle_total_bases.py --gameday 2026-07-04
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

S3_BUCKET     = "the-odds-api-mt"
DAILY_PREFIX  = "mlb/total_bases_model/daily_runs"
SETTLED_KEY   = "mlb/total_bases_model/settled/mlb_tb_settled_bets.parquet"

SES_SOURCE    = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()

ET = ZoneInfo("America/New_York")

_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"

def _time_sort_key(t: str) -> int:
    try:
        h, rest = t.split(":")
        m, ampm = rest.strip().split(" ")
        h, m = int(h), int(m)
        if ampm == "PM" and h != 12:
            h += 12
        elif ampm == "AM" and h == 12:
            h = 0
        return h * 60 + m
    except Exception:
        return 9999


TB_EVENTS = {"single", "double", "triple", "home_run"}
AB_EVENTS = {
    "single", "double", "triple", "home_run",
    "strikeout", "strikeout_double_play",
    "field_out", "force_out", "grounded_into_double_play",
    "double_play", "triple_play", "field_error",
    "fielders_choice", "fielders_choice_out",
}

MANUAL_MAP = {
    "daniel vogelbach":   "Dan Vogelbach",
    "michael a taylor":   "Michael Taylor",
    "max muncy (2002)":   "Max Muncy",
    "diego a castillo":   "Diego Castillo",
    "james jarvis":       "Jim Jarvis",
    "donnie walton":      "Donovan Walton",
    "josh kuroda-grauer": "Joshua Kuroda-Grauer",
}


def yesterday_et() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    manual_norm = {_normalize_raw(k): _normalize_raw(v) for k, v in MANUAL_MAP.items()}
    n = _normalize_raw(name)
    return manual_norm.get(n, n)


def _normalize_raw(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[.,'\-]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    name = re.sub(r"\s+", "", name)
    return name.strip()


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


def save_s3_csv(key: str, df: pd.DataFrame) -> None:
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=df.to_csv(index=False).encode())


def load_settled_history() -> pd.DataFrame:
    try:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=SETTLED_KEY)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return pd.DataFrame()
        raise


def save_settled_history(df: pd.DataFrame) -> None:
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=SETTLED_KEY, Body=buf.getvalue())


# ── Actuals from Statcast ─────────────────────────────────────────────────────

def get_tb_for_date(gameday: str) -> pd.DataFrame:
    """Fetch Statcast and return total_bases per batter for the given date."""
    warnings.filterwarnings("ignore")
    try:
        raw = pb.statcast(start_dt=gameday, end_dt=gameday)
    except Exception as ex:
        print(f"  pybaseball error: {ex}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    raw = raw[raw["game_type"] == "R"].copy()
    events = raw[raw["events"].notna()].copy()
    if events.empty:
        return pd.DataFrame()

    events["tb"] = (
        (events["events"] == "single").astype(int) * 1 +
        (events["events"] == "double").astype(int) * 2 +
        (events["events"] == "triple").astype(int) * 3 +
        (events["events"] == "home_run").astype(int) * 4
    )
    events["is_ab"] = events["events"].isin(AB_EVENTS).astype(int)

    agg = events.groupby(["batter", "game_pk"]).agg(
        total_bases = ("tb", "sum"),
        ab          = ("is_ab", "sum"),
    ).reset_index()
    agg = agg[agg["ab"] >= 1]  # exclude DNP rows

    # Sum DH games (player-date level)
    agg = agg.groupby("batter").agg(total_bases=("total_bases", "sum")).reset_index()

    # Add player name
    batter_ids = agg["batter"].dropna().unique().tolist()
    if not batter_ids:
        return pd.DataFrame()
    nl = pb.playerid_reverse_lookup(batter_ids, key_type="mlbam")
    nl["player_name"] = nl["name_first"].str.title() + " " + nl["name_last"].str.title()
    agg = agg.merge(nl[["key_mlbam", "player_name"]].rename(columns={"key_mlbam": "batter"}), on="batter", how="left")
    agg["name_norm"] = agg["player_name"].map(normalize_name)
    return agg[["batter", "player_name", "name_norm", "total_bases"]].copy()


# ── Settlement ────────────────────────────────────────────────────────────────

def settle_bets(bets: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Join bets to actuals and compute P&L."""
    merged = bets.merge(
        actuals[["name_norm", "total_bases"]].rename(columns={"total_bases": "actual_tb"}),
        on="name_norm",
        how="left",
    )
    def result(row):
        if pd.isna(row["actual_tb"]):
            return "no_data", np.nan  # push/exclude
        if row["actual_tb"] <= 1:
            return "win", float(row["dec_odds_under"]) - 1.0
        return "loss", -1.0

    merged[["outcome", "pnl"]] = merged.apply(lambda r: pd.Series(result(r)), axis=1)
    return merged


# ── Email ─────────────────────────────────────────────────────────────────────

def build_html_email(settled_today: pd.DataFrame, history: pd.DataFrame, gameday: str) -> str:
    he = html_module.escape
    settled = settled_today[settled_today["outcome"] != "no_data"].copy()
    wins    = (settled["outcome"] == "win").sum()
    losses  = (settled["outcome"] == "loss").sum()
    no_data = (settled_today["outcome"] == "no_data").sum()
    units   = settled["pnl"].sum()

    HAS_BOOK = "bookmaker" in settled.columns
    HAS_GAME = all(c in settled.columns for c in ["game_time_et", "home_team", "away_team"])

    def fmt_pnl(v):
        try:
            return f"{float(v):+.3f}u"
        except (TypeError, ValueError):
            return "—"

    def dec_to_american(dec):
        try:
            dec = float(dec)
            if dec >= 2.0:
                return f"+{int(round((dec - 1) * 100))}"
            return str(int(round(-100 / (dec - 1))))
        except Exception:
            return "—"

    def outcome_td(outcome):
        if outcome == "win":
            return "<td style='text-align:center;color:#276221;font-weight:bold'>win</td>"
        if outcome == "loss":
            return "<td style='text-align:center;color:#c0392b;font-weight:bold'>loss</td>"
        return "<td style='text-align:center;color:#888'>—</td>"

    def row_style(outcome):
        return "background:#eaf6ea" if outcome == "win" else ""

    def tier_bg(r):
        tier = str(r.get("tier", "")).lower()
        if r["outcome"] == "win":
            return "background:#eaf6ea"
        if tier == "play":
            return "background:#f9fcf9"
        if tier == "track":
            return "background:#fffde7"
        return ""

    def tier_badge(r):
        tier = str(r.get("tier", "")).lower()
        if tier == "play":
            return "<span style='color:#276221;font-weight:bold'>play</span>"
        if tier == "track":
            return "<span style='color:#b8860b;font-weight:bold'>track</span>"
        return "<span style='color:#888'>—</span>"

    def bet_row(r):
        book = he(str(r.get("bookmaker", "—"))) if HAS_BOOK else "—"
        bg = tier_bg(r)
        return (
            f"<tr style='{bg}'>"
            f"<td>{he(str(r['player_name']))}</td>"
            f"<td style='text-align:center'>{tier_badge(r)}</td>"
            f"<td style='text-align:center;color:#555'>{book}</td>"
            f"<td style='text-align:center'>{r.get('actual_tb', '—')}</td>"
            f"<td style='text-align:center'>{float(r.get('line', 1.5)):.1f}</td>"
            f"<td style='text-align:center'>{float(r.get('novig_prob_under', 0)):.1%}</td>"
            f"<td style='text-align:center'>{float(r.get('edge_under', 0)):.1%}</td>"
            f"<td style='text-align:center'>{dec_to_american(r.get('dec_odds_under'))}</td>"
            f"{outcome_td(r['outcome'])}"
            f"<td style='text-align:center;font-weight:bold'>{fmt_pnl(r['pnl'])}</td>"
            f"</tr>\n"
        )

    COL_HEADERS = "<tr><th>Player</th><th>Tier</th><th>Book</th><th>Actual TB</th><th>Line</th><th>Mkt Under%</th><th>Edge</th><th>Under Odds</th><th>Outcome</th><th>P&amp;L</th></tr>"

    # ── Part 1: All bets sorted by P&L ────────────────────────────────────────
    part1_rows = "".join(bet_row(r) for _, r in settled.sort_values("pnl", ascending=False).iterrows())

    # ── Part 2: By game summary ───────────────────────────────────────────────
    part2_html = ""
    part3_html = ""
    if HAS_GAME and not settled.empty:
        settled["_tsort"] = settled["game_time_et"].map(_time_sort_key)
        game_order = (
            settled[["game_time_et", "home_team", "away_team", "_tsort"]]
            .drop_duplicates()
            .sort_values("_tsort")
            [["game_time_et", "home_team", "away_team"]]
            .values.tolist()
        )
        summary_rows = ""
        detail_blocks = ""
        total_gplayers = total_gbets = 0
        for gtime, home, away in game_order:
            grp = settled[
                (settled["game_time_et"] == gtime) &
                (settled["home_team"] == home) &
                (settled["away_team"] == away)
            ]
            gw = (grp["outcome"] == "win").sum()
            gl = (grp["outcome"] == "loss").sum()
            gnet = grp["pnl"].sum()
            gplayers = grp["name_norm"].nunique() if "name_norm" in grp.columns else len(grp)
            gbets = len(grp)
            total_gplayers += gplayers
            total_gbets += gbets
            net_color = "#276221" if gnet >= 0 else "#c0392b"
            summary_rows += (
                f"<tr>"
                f"<td>{he(away)} @ {he(home)}</td>"
                f"<td style='text-align:center;color:#555'>{he(gtime)}</td>"
                f"<td style='text-align:center'>{gplayers}</td>"
                f"<td style='text-align:center'>{gbets}</td>"
                f"<td style='text-align:center;color:#276221;font-weight:bold'>{gw}</td>"
                f"<td style='text-align:center;color:#c0392b;font-weight:bold'>{gl}</td>"
                f"<td style='text-align:center;font-weight:bold;color:{net_color}'>{fmt_pnl(gnet)}</td>"
                f"</tr>\n"
            )
            total_color = "#276221" if units >= 0 else "#c0392b"
            wl_color = "#276221" if gnet >= 0 else "#888"
            detail_rows = "".join(bet_row(r) for _, r in grp.sort_values("pnl", ascending=False).iterrows())
            detail_blocks += f"""
<table style='margin-top:12px'>
  <tr style='background:#edf1f5'>
    <td colspan='9' style='padding:7px 10px;font-weight:600;font-size:12px;color:#2c3e50;border-top:2px solid #bdc3c7;border-bottom:1px solid #bdc3c7'>
      {he(gtime)} ET &nbsp;·&nbsp; {he(away)} @ {he(home)} &nbsp;·&nbsp; {gplayers} player{'s' if gplayers != 1 else ''} · {gbets} bet{'s' if gbets != 1 else ''} &nbsp;·&nbsp; <span style='color:{wl_color}'>{gw}W / {gl}L &nbsp;·&nbsp; <span style='color:{net_color}'>{fmt_pnl(gnet)}</span></span>
    </td>
  </tr>
  {COL_HEADERS}
  {detail_rows}
</table>"""

        total_net_color = "#276221" if units >= 0 else "#c0392b"
        summary_rows += (
            f"<tr style='background:#f2f2f2;font-weight:600'>"
            f"<td>Total</td><td style='text-align:center'>—</td>"
            f"<td style='text-align:center'>{total_gplayers}</td>"
            f"<td style='text-align:center'>{total_gbets}</td>"
            f"<td style='text-align:center;color:#276221'>{wins}</td>"
            f"<td style='text-align:center;color:#c0392b'>{losses}</td>"
            f"<td style='text-align:center;color:{total_net_color};font-weight:bold'>{fmt_pnl(units)}</td>"
            f"</tr>\n"
        )
        part2_html = f"""
<p style='font-weight:600;font-size:13px;color:#555;margin:20px 0 6px'>By game — summary</p>
<table style='width:auto'>
  <tr><th>Game</th><th>Time ET</th><th>Players</th><th>Bets</th><th>W</th><th>L</th><th>Net</th></tr>
  {summary_rows}
</table>"""
        part3_html = f"""
<p style='font-weight:600;font-size:13px;color:#555;margin:20px 0 6px'>By game — detail</p>
{detail_blocks}"""

    # ── All-time summary ──────────────────────────────────────────────────────
    if not history.empty and "pnl" in history.columns:
        settled_hist = history[history["pnl"].notna()].copy()
        all_n     = len(settled_hist)
        all_win   = (settled_hist["outcome"] == "win").sum() if "outcome" in settled_hist.columns else "—"
        all_units = settled_hist["pnl"].sum()
        all_roi   = settled_hist["pnl"].mean()
        summary_html = f"<strong>All-time:</strong> n={all_n}, {all_win}W, units={all_units:+.2f}u, ROI={all_roi*100:+.2f}%"
        if "tier" in settled_hist.columns:
            for tier_name, tier_color in [("play", "#276221"), ("track", "#b8860b")]:
                t = settled_hist[settled_hist["tier"] == tier_name]
                if not t.empty:
                    tw = (t["outcome"] == "win").sum()
                    summary_html += (
                        f" &nbsp;·&nbsp; <span style='color:{tier_color}'>"
                        f"<strong>{tier_name}:</strong> n={len(t)}, {tw}W, "
                        f"{t['pnl'].sum():+.2f}u, ROI={t['pnl'].mean()*100:+.2f}%</span>"
                    )
    else:
        summary_html = "No history yet."

    # ── Today's P&L by tier ───────────────────────────────────────────────────
    today_tier_html = ""
    if "tier" in settled.columns and not settled.empty:
        tier_parts = []
        for tier_name, tier_color in [("play", "#276221"), ("track", "#b8860b")]:
            t = settled[settled["tier"] == tier_name]
            if not t.empty:
                tw = (t["outcome"] == "win").sum()
                tl = (t["outcome"] == "loss").sum()
                tier_parts.append(
                    f"<span style='color:{tier_color}'><strong>{tier_name}:</strong> "
                    f"{tw}W/{tl}L {t['pnl'].sum():+.3f}u</span>"
                )
        if tier_parts:
            today_tier_html = " &nbsp;·&nbsp; ".join(tier_parts)

    no_data_str = f" / {no_data} no data" if no_data else ""
    tier_line = f"<p style='font-size:12px;color:#555;margin-top:4px'>{today_tier_html}</p>" if today_tier_html else ""

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<style>
  body {{font-family:{_SANS};color:#222;max-width:900px;margin:auto;padding:20px}}
  h2 {{color:#2c3e50}}
  table {{border-collapse:collapse;width:100%;margin-top:8px}}
  th {{background:#2c3e50;color:#fff;padding:7px 10px;text-align:left;font-size:12px;white-space:nowrap}}
  td {{padding:6px 10px;border-bottom:1px solid #e0e0e0;font-size:12px}}
  tr:nth-child(even) td {{background:#f9f9f9}}
  .footer {{background:#ecf0f1;border-radius:6px;padding:12px 16px;margin-top:20px;font-size:12px;color:#555}}
</style>
</head><body>
<h2>MLB Total Bases — Settlement — {gameday}</h2>
<p><strong>{wins}W / {losses}L{no_data_str}</strong> &nbsp;·&nbsp; <strong>{units:+.3f}u</strong></p>
{tier_line}

<p style='font-weight:600;font-size:13px;color:#555;margin:16px 0 6px'>All bets — sorted by P&L</p>
<table>
  {COL_HEADERS}
  {part1_rows}
</table>

{part2_html}

{part3_html}

<div class='footer'>{summary_html}</div>
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=yesterday_et())
    args = parser.parse_args()
    gameday = args.gameday

    print(f"MLB Total Bases settlement | gameday={gameday}")

    # Load bet sheet
    bets_key = f"{DAILY_PREFIX}/{gameday}/recommendations.csv"
    bets = load_s3_csv(bets_key)
    if bets is None or bets.empty:
        msg = f"No bet sheet found for {gameday}"
        print(msg)
        publish_sns(f"MLB TB settlement — no bets {gameday}", msg)
        return

    bets["name_norm"] = bets["player_name"].map(normalize_name)
    print(f"  {len(bets)} bets to settle")

    # Fetch actuals
    print(f"Fetching Statcast actuals for {gameday} ...")
    actuals = get_tb_for_date(gameday)
    print(f"  {len(actuals)} batter rows from Statcast")

    if actuals.empty:
        print("  No Statcast data — skipping settlement (will retry tomorrow)")
        publish_sns(f"MLB TB settlement — no Statcast data {gameday}", "Statcast data not yet available.")
        return

    # Settle
    settled = settle_bets(bets, actuals)
    wins    = (settled["outcome"] == "win").sum()
    losses  = (settled["outcome"] == "loss").sum()
    no_data = (settled["outcome"] == "no_data").sum()
    units   = settled["pnl"].dropna().sum()
    print(f"  Settled: {wins}W / {losses}L / {no_data} no_data | units={units:+.3f}")

    # Save settled sheet
    settled["game_date"] = gameday
    settled_key = f"{DAILY_PREFIX}/{gameday}/settled.csv"
    save_s3_csv(settled_key, settled)
    print(f"  Saved → s3://{S3_BUCKET}/{settled_key}")

    # Update all-time history
    history = load_settled_history()
    settled_rows = settled[settled["outcome"] != "no_data"].copy()
    settled_rows["won"] = (settled_rows["outcome"] == "win").astype(int)

    keep_cols = [c for c in ["game_date", "player_name", "name_norm", "line", "bet_direction",
                              "bookmaker", "home_team", "away_team", "game_time_et",
                              "n_books", "novig_prob_over", "novig_prob_under",
                              "p_model", "p_market", "edge_under", "dec_odds_under",
                              "tier", "actual_tb", "outcome", "won", "pnl"] if c in settled_rows.columns]
    if "bet_direction" not in settled_rows.columns:
        settled_rows["bet_direction"] = "under"
    if not history.empty:
        updated = pd.concat([history, settled_rows[keep_cols]], ignore_index=True)
        dedup_cols = ["game_date", "player_name", "line"]
        if "bookmaker" in updated.columns:
            dedup_cols.append("bookmaker")
        updated = updated.drop_duplicates(subset=dedup_cols, keep="last")
    else:
        updated = settled_rows[keep_cols].copy()

    save_settled_history(updated)
    print(f"  Updated history: {len(updated)} total rows")

    # Email
    n_unique = settled[settled["outcome"] != "no_data"]["player_name"].nunique() if not settled.empty else 0
    subject = f"MLB Total Bases — {n_unique} players ({len(settled[settled['outcome'] != 'no_data'])} bets) {wins}W/{losses}L {units:+.2f}u — {gameday}"
    html_body = build_html_email(settled, updated, gameday)
    send_ses(subject, html_body)
    publish_sns(subject, f"{wins}W/{losses}L, {units:+.3f}u on {gameday}")


if __name__ == "__main__":
    main()
