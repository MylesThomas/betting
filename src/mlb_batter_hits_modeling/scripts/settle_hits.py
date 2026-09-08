"""
Settle yesterday's MLB Batter Hits bets.

  1. Load yesterday's recommendations CSV from S3
  2. Load Statcast actuals from S3 (or fetch fresh from pybaseball)
  3. For each bet: over → hits_actual > line? · under → hits_actual < line?
  4. Compute P&L: win → +(decimal_price - 1), lose → -1
  5. Update settled_bets parquet in S3
  6. Email HTML settlement summary + JSON output

S3 paths read:
  s3://the-odds-api-mt/mlb/batter_hits_model/daily_runs/{yesterday}/recommendations.csv
  s3://the-odds-api-mt/mlb/total_bases_model/actuals/mlb_batting_statcast.parquet

S3 paths written:
  s3://the-odds-api-mt/mlb/batter_hits_model/settled/mlb_batter_hits_settled_bets.parquet

Usage:
  python src/mlb_batter_hits_modeling/scripts/settle_hits.py
  python src/mlb_batter_hits_modeling/scripts/settle_hits.py --gameday 2026-07-26
  python src/mlb_batter_hits_modeling/scripts/settle_hits.py --output /tmp/settle_out.json
"""
from __future__ import annotations

import argparse
import html as html_module
import json
import os
import sys
import unicodedata
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
ET          = ZoneInfo("America/New_York")

SES_SOURCE  = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW  = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def normalize_name(name: str) -> str:
    import re
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"\s*\(\d{4}\)", "", name)
    name = re.sub(r"[''`]", "", name)
    name = re.sub(r"[-]", " ", name)
    name = re.sub(r"\.", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name)
    name = re.sub(r"\b([a-z]) ([a-z])\b", r"\1\2", name)
    name = re.sub(r"\b([a-z]) ([a-z])\b", r"\1\2", name)
    name = re.sub(r"(?<=\s)[a-z](?=\s)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    NAME_MAP = {
        "daniel vogelbach":   "dan vogelbach",
        "donnie walton":      "donovan walton",
        "eddy alvarez":       "francisco alvarez",
        "josh kuroda grauer": "joshua kuroda grauer",
    }
    return NAME_MAP.get(name, name)


def _dec_to_american(d) -> str:
    if not d or pd.isna(d):
        return "N/A"
    if d >= 2.0:
        return f"+{int(round((d - 1) * 100))}"
    return f"-{int(round(100 / (d - 1)))}"


def load_recs_from_s3(cfg: dict, gameday: str) -> pd.DataFrame:
    key = f"{cfg['data']['daily_prefix']}/{gameday}/recommendations.csv"
    s3  = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=cfg["data"]["s3_bucket"], Key=key)
        df  = pd.read_csv(BytesIO(obj["Body"].read()))
        print(f"  Loaded {len(df)} recs from {key}")
        return df
    except s3.exceptions.NoSuchKey:
        print(f"  No recommendations found for {gameday}")
        return pd.DataFrame()


def load_actuals_from_s3(cfg: dict) -> pd.DataFrame:
    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket=cfg["data"]["s3_bucket"], Key=cfg["data"]["actuals_key"])
    df  = pd.read_parquet(BytesIO(obj["Body"].read()))
    df["name_norm"] = df["player_name"].apply(normalize_name)
    return df


def settle(recs: pd.DataFrame, actuals: pd.DataFrame, gameday: str) -> pd.DataFrame:
    """Join recs to actuals, compute outcome and P&L."""
    actuals_day = actuals[actuals["game_date"] == gameday].copy()
    # Dedup: one row per player per date (highest AB)
    actuals_day = (
        actuals_day.sort_values("ab", ascending=False)
        .drop_duplicates(subset=["name_norm"])
    )

    recs["name_norm"] = recs["player_key"].apply(lambda x: x if isinstance(x, str) else "")
    settled = recs.merge(
        actuals_day[["name_norm", "hits", "ab"]],
        on="name_norm", how="left",
    )
    settled["hits_actual"] = settled["hits"]

    # Play bets only (not tracks) for P&L
    plays = settled[settled["tier"] == "play"].copy()
    if len(plays) == 0:
        print("  No plays to settle")
        return settled

    # Outcome
    plays["over_flag"]  = (plays["hits_actual"] > plays["line"]).astype(float)
    plays["under_flag"] = (plays["hits_actual"] < plays["line"]).astype(float)
    plays["push_flag"]  = (plays["hits_actual"] == plays["line"]).astype(float)

    strat_dir = "under"
    plays["result"] = plays.apply(lambda r: (
        "win"  if r[f"{strat_dir}_flag"] == 1 else
        "push" if r["push_flag"] == 1 else
        "loss" if pd.notna(r["hits_actual"]) else
        "pending"
    ), axis=1)
    plays["pnl"] = plays.apply(lambda r: (
        r["under_price"] - 1 if r["result"] == "win"  else
        0.0                   if r["result"] == "push" else
        -1.0                  if r["result"] == "loss" else
        0.0
    ), axis=1)

    return plays


def build_settle_html(plays: pd.DataFrame, cfg: dict, gameday: str) -> str:
    settled_plays = plays[plays["result"] != "pending"]
    n_win  = (settled_plays["result"] == "win").sum()
    n_loss = (settled_plays["result"] == "loss").sum()
    n_push = (settled_plays["result"] == "push").sum()
    net    = settled_plays["pnl"].sum()
    roi    = net / max(len(settled_plays), 1) * 100

    _ss = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
    css = f"""
    body{{font-family:{_ss};font-size:13px;color:#222;background:#f5f5f5;margin:0;padding:16px}}
    .card{{background:#fff;border-radius:8px;padding:16px 20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
    h1{{font-size:18px;margin:0 0 4px}} h2{{font-size:14px;color:#555;margin:0 0 12px;font-weight:normal}}
    table{{border-collapse:collapse;width:100%}}
    th{{background:#f0f0f0;padding:6px 8px;text-align:left;font-size:12px;color:#666;border-bottom:2px solid #ddd}}
    td{{padding:5px 8px;border-bottom:1px solid #eee}}
    .win{{background:#e8f5e9}} .loss{{background:#ffebee}} .push{{background:#fff3e0}}
    .pos{{color:#1a7f37;font-weight:bold}} .neg{{color:#d32f2f;font-weight:bold}}
    """

    def _row(r: pd.Series) -> str:
        cls    = r["result"]
        result = r["result"].upper()
        pnl    = r["pnl"]
        pnlcls = "pos" if pnl > 0 else "neg"
        return (
            f'<tr class="{cls}">'
            f'<td>{html_module.escape(str(r["player_name"]))}</td>'
            f'<td>{html_module.escape(str(r["bookmaker"]))}</td>'
            f'<td>{r["line"]}</td>'
            f'<td>{_dec_to_american(r.get("under_price", None))}</td>'
            f'<td>{r.get("hits_actual", "DNP")}</td>'
            f'<td>{result}</td>'
            f'<td class="{pnlcls}">{pnl:+.2f}u</td>'
            f'</tr>'
        )

    rows_html = "".join(_row(r) for _, r in settled_plays.iterrows())
    if not rows_html:
        rows_html = '<tr><td colspan="7" style="color:#888;text-align:center">No settled bets</td></tr>'

    netcls = "pos" if net > 0 else "neg"
    header = (
        f'<h1>MLB Batter Hits — Settlement {gameday}</h1>'
        f'<h2>{n_win}W / {n_loss}L / {n_push}P &nbsp;|&nbsp; '
        f'<span class="{netcls}">{net:+.2f}u</span> &nbsp;|&nbsp; '
        f'<span class="{netcls}">{roi:+.1f}% ROI</span></h2>'
    )
    table_header = (
        '<tr><th>Player</th><th>Book</th><th>Line</th>'
        '<th>Under $</th><th>Hits</th><th>Result</th><th>P&L</th></tr>'
    )
    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>{css}</style></head><body>'
        f'<div class="card">{header}'
        f'<table>{table_header}{rows_html}</table></div></body></html>'
    )


def update_settled_history(plays: pd.DataFrame, cfg: dict) -> dict:
    """Append newly settled plays to the season history parquet in S3."""
    s3  = boto3.client("s3")
    key = cfg["data"]["settled_key"]
    bucket = cfg["data"]["s3_bucket"]

    settled_plays = plays[plays["result"] != "pending"].copy()
    if len(settled_plays) == 0:
        return {"yesterday_wins": 0, "yesterday_losses": 0, "yesterday_units": 0.0}

    # Load existing history
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        hist = pd.read_parquet(BytesIO(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        hist = pd.DataFrame()

    combined = pd.concat([hist, settled_plays], ignore_index=True)
    # Remove duplicate bets by (game_date, player_key, bookmaker, line)
    dedup_cols = ["game_date", "player_key", "bookmaker", "line"]
    avail_dedup = [c for c in dedup_cols if c in combined.columns]
    if avail_dedup:
        combined = combined.drop_duplicates(subset=avail_dedup, keep="last")

    buf = BytesIO()
    combined.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    print(f"  Settled history updated: {len(combined)} total bets")

    # Season stats
    season_plays = combined[combined["result"].isin(["win", "loss", "push"])]
    return {
        "yesterday_wins":    int((settled_plays["result"] == "win").sum()),
        "yesterday_losses":  int((settled_plays["result"] == "loss").sum()),
        "yesterday_units":   float(settled_plays["pnl"].sum()),
        "season_wins":       int((season_plays["result"] == "win").sum()),
        "season_losses":     int((season_plays["result"] == "loss").sum()),
        "season_units":      float(season_plays["pnl"].sum()),
    }


def _send_ses(subject: str, html_body: str) -> None:
    if not SES_SOURCE:
        print("  SES not configured — skipping email send")
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
    print(f"  Settlement email sent: {subject[:80]}")


def main(gameday: str | None = None, output: str | None = None, no_email: bool = False) -> dict:
    cfg = _load_config()
    if not gameday:
        yesterday = (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")
        gameday = yesterday
    print(f"\nMLB Batter Hits settlement | {gameday}")

    recs = load_recs_from_s3(cfg, gameday)
    if recs.empty:
        result = {"yesterday_wins": 0, "yesterday_losses": 0, "yesterday_units": 0.0}
        if output:
            Path(output).write_text(json.dumps(result))
        return result

    print("Loading actuals from S3...")
    actuals = load_actuals_from_s3(cfg)

    print("Settling bets...")
    plays = settle(recs, actuals, gameday)

    html_body = build_settle_html(plays, cfg, gameday)

    stats = update_settled_history(plays, cfg)

    yest_str   = f"{stats['yesterday_wins']}W/{stats['yesterday_losses']}L {stats['yesterday_units']:+.2f}u"
    season_str = f"{stats.get('season_wins', 0)}W/{stats.get('season_losses', 0)}L {stats.get('season_units', 0.0):+.2f}u"
    subject    = f"MLB Batter Hits SETTLED {gameday} — {yest_str} · Season: {season_str}"

    if not no_email:
        _send_ses(subject, html_body)

    stats["html"]    = html_body
    stats["subject"] = subject
    stats["gameday"] = gameday

    if output:
        Path(output).write_text(json.dumps(stats))

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-email", action="store_true")
    args = parser.parse_args()
    main(gameday=args.gameday, output=args.output, no_email=args.no_email)
