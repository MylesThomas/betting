"""
Settle MLB pitcher strikeouts bets for a given gameday.

Settlement logic:
  OVER: WIN if actual_k > line, LOSS if actual_k < line, PUSH if actual_k == line
  UNDER: WIN if actual_k < line, LOSS if actual_k > line, PUSH if actual_k == line
  DNP: pitcher did not start (no result)

Reads:
  s3://the-odds-api-mt/mlb/strikeouts_model/daily_runs/{gameday}/recommendations.csv
  MLB Stats API: per-pitcher game log for {gameday}

Writes / appends:
  s3://the-odds-api-mt/mlb/strikeouts_model/settled/settled_bets.parquet

Run:
    python src/mlb_strikeouts_modeling/scripts/settle_strikeouts.py
    python src/mlb_strikeouts_modeling/scripts/settle_strikeouts.py --gameday 2026-07-03
"""
from __future__ import annotations

import argparse
import html as html_module
import os
import sys
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import botocore.exceptions
import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

MLB_API_BASE  = "https://statsapi.mlb.com/api/v1"
SLEEP_S       = 0.05

S3_BUCKET     = "the-odds-api-mt"
S3_PREFIX     = "mlb/strikeouts_model"
SES_SOURCE    = os.environ.get("SETTLEMENT_SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SETTLEMENT_SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()

ET    = ZoneInfo("America/New_York")
_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"


def yesterday_et() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")


def _normalize_name(name: str) -> str:
    import unicodedata, re
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name.strip().lower())
    return re.sub(r"\s+", " ", name).strip()


def _s3():
    return boto3.client("s3")


# ── Load / save ───────────────────────────────────────────────────────────────

def load_recommendations(gameday: str) -> pd.DataFrame | None:
    key = f"{S3_PREFIX}/daily_runs/{gameday}/recommendations.csv"
    try:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        return pd.read_csv(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
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


# ── MLB Stats API ─────────────────────────────────────────────────────────────

def fetch_pitcher_results(player_ids: list[int], gameday: str) -> dict[int, dict]:
    """
    Returns {player_id: {strikeouts, innings_pitched}} for starters on gameday.
    Falls back to name-keyed lookup if player_id lookup fails.
    """
    year   = int(gameday[:4])
    result = {}
    for pid in player_ids:
        if not pid or pd.isna(pid):
            continue
        pid = int(pid)
        try:
            r = requests.get(
                f"{MLB_API_BASE}/people/{pid}/stats",
                params={"stats": "gameLog", "season": year, "group": "pitching"},
                timeout=20,
            )
            if r.status_code != 200:
                continue
            for split in r.json().get("stats", [{}])[0].get("splits", []):
                if split.get("date", "") != gameday:
                    continue
                stat = split.get("stat", {})
                if stat.get("gamesStarted", 0) >= 1:
                    result[pid] = {
                        "strikeouts":      stat.get("strikeOuts", 0),
                        "innings_pitched": stat.get("inningsPitched", "0.0"),
                    }
        except Exception as e:
            print(f"  MLB API error for player {pid}: {e}")
        time.sleep(SLEEP_S)
    return result


def fetch_all_starters(gameday: str) -> dict[str, dict]:
    """
    Fetch all starting pitchers for gameday via schedule endpoint.
    Returns {player_key: {strikeouts, innings_pitched}}.
    """
    try:
        r = requests.get(
            f"{MLB_API_BASE}/schedule",
            params={"sportId": 1, "date": gameday, "hydrate": "linescore(matchup,runners)"},
            timeout=20,
        )
        r.raise_for_status()
    except Exception as e:
        print(f"  MLB API schedule error: {e}")
        return {}

    # Collect game PKs
    game_pks = [
        g.get("gamePk")
        for date_entry in r.json().get("dates", [])
        for g in date_entry.get("games", [])
        if g.get("status", {}).get("statusCode", "") == "F"
    ]
    if not game_pks:
        print("  No final games found via schedule endpoint")
        return {}

    result = {}
    for game_pk in game_pks:
        try:
            br = requests.get(
                f"{MLB_API_BASE}/game/{game_pk}/boxscore",
                timeout=20,
            )
            br.raise_for_status()
        except Exception:
            continue

        for side in ("home", "away"):
            pitchers = br.json().get("teams", {}).get(side, {}).get("pitchers", [])
            players  = br.json().get("teams", {}).get(side, {}).get("players", {})
            if not pitchers:
                continue
            starter_id = pitchers[0]  # first pitcher = starter
            pkey = f"ID{starter_id}"
            pdata = players.get(pkey, {})
            name  = pdata.get("person", {}).get("fullName", "")
            stats = pdata.get("stats", {}).get("pitching", {})
            if name:
                result[_normalize_name(name)] = {
                    "strikeouts":      stats.get("strikeOuts", 0),
                    "innings_pitched": stats.get("inningsPitched", "0"),
                }
        time.sleep(SLEEP_S)

    return result


# ── Settlement ────────────────────────────────────────────────────────────────

def settle(recs: pd.DataFrame, actuals: dict[str, dict]) -> pd.DataFrame:
    df = recs.copy()
    df["side"] = df["side"].str.lower()
    # Only settle rows with an actual recommendation — drop no-edge rows (side=NaN)
    df = df[df["side"].isin(["over", "under"])].copy()
    df["actual_k"] = df["player_key"].map(
        {k: v.get("strikeouts") for k, v in actuals.items()}
    )
    df["actual_k"] = pd.to_numeric(df["actual_k"], errors="coerce")
    df["line"]     = pd.to_numeric(df["line"], errors="coerce")

    def outcome(row):
        if pd.isna(row["actual_k"]):
            return "DNP"
        k, line, side = row["actual_k"], row["line"], row["side"]
        if side == "over":
            if k > line:  return "WIN"
            if k == line: return "PUSH"
            return "LOSS"
        else:  # under
            if k < line:  return "WIN"
            if k == line: return "PUSH"
            return "LOSS"

    def pnl(row):
        if row["outcome"] in ("PUSH", "DNP"):
            return 0.0
        odds = row.get("odds") if row["side"] == "over" else row.get("odds_u")
        if pd.isna(odds):
            return np.nan  # missing odds = data problem, not a default
        odds = float(odds)
        if row["outcome"] == "WIN":
            return odds / 100.0 if odds >= 0 else 100.0 / abs(odds)
        return -1.0

    df["outcome"]    = df.apply(outcome, axis=1)
    df["pnl"]        = df.apply(pnl, axis=1)
    df["is_hit"]     = (df["outcome"] == "WIN").astype(int)
    df["settled_at"] = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    return df


# ── HTML email ────────────────────────────────────────────────────────────────

def build_settlement_html(df: pd.DataFrame, gameday: str) -> str:
    now_str    = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    total_pnl  = df["pnl"].sum()
    total_bets = len(df)
    wins   = (df["outcome"] == "WIN").sum()
    losses = (df["outcome"] == "LOSS").sum()
    pushes = (df["outcome"] == "PUSH").sum()
    dnps   = (df["outcome"] == "DNP").sum()
    roi    = total_pnl / total_bets * 100 if total_bets > 0 else 0.0

    def color(oc: str) -> str:
        return {"WIN": "#4ade80", "LOSS": "#f87171", "PUSH": "#fbbf24", "DNP": "#9ca3af"}.get(oc, "#e2e8f0")

    rows_html = ""
    for _, row in df.sort_values("outcome").iterrows():
        oc      = row["outcome"]
        side    = str(row.get("side", "")).upper()
        side_c  = "#4ade80" if side == "OVER" else "#f87171"
        raw_odds = row.get("odds") if side == "OVER" else row.get("odds_u")
        odds_disp = f"{int(float(raw_odds)):+d}" if pd.notna(raw_odds) else "—"
        rows_html += f"""
        <tr>
          <td style="padding:6px 10px;font-weight:600;">{html_module.escape(str(row.get('player','—')))}</td>
          <td style="padding:6px 10px;text-align:center;font-weight:bold;color:{side_c};">{side}</td>
          <td style="padding:6px 10px;text-align:center;">{row.get('line','—')}</td>
          <td style="padding:6px 10px;text-align:center;font-family:{_MONO};">{odds_disp}</td>
          <td style="padding:6px 10px;text-align:center;font-family:{_MONO};">{"—" if pd.isna(row.get('actual_k')) else int(row['actual_k'])}</td>
          <td style="padding:6px 10px;text-align:center;font-family:{_MONO};">+{row.get('edge',0)*100:.1f}pp</td>
          <td style="padding:6px 10px;text-align:center;font-weight:bold;color:{color(oc)};">{oc}</td>
          <td style="padding:6px 10px;text-align:center;font-family:{_MONO};color:{'#4ade80' if row['pnl']>0 else '#f87171' if row['pnl']<0 else '#9ca3af'};">{row['pnl']:+.2f}u</td>
        </tr>"""

    pnl_color = "#4ade80" if total_pnl >= 0 else "#f87171"
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/></head>
<body style="font-family:{_SANS};background:#0f1117;color:#e2e8f0;margin:0;padding:24px;">
<h2 style="color:#93c5fd;margin-bottom:4px;">MLB Strikeouts Settlement — {gameday}</h2>
<p style="color:#6b7280;font-size:13px;margin-top:0;">{now_str}</p>

<div style="display:flex;gap:24px;margin-bottom:20px;flex-wrap:wrap;">
  <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;padding:14px 20px;min-width:120px;">
    <div style="font-size:11px;color:#6b7280;text-transform:uppercase;">Total PnL</div>
    <div style="font-size:22px;font-weight:700;color:{pnl_color};">{total_pnl:+.2f}u</div>
  </div>
  <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;padding:14px 20px;min-width:120px;">
    <div style="font-size:11px;color:#6b7280;text-transform:uppercase;">Record</div>
    <div style="font-size:18px;font-weight:700;">{wins}W–{losses}L{f'–{pushes}P' if pushes else ''}{f'–{dnps}DNP' if dnps else ''}</div>
  </div>
  <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;padding:14px 20px;min-width:120px;">
    <div style="font-size:11px;color:#6b7280;text-transform:uppercase;">ROI</div>
    <div style="font-size:18px;font-weight:700;">{roi:+.1f}%</div>
  </div>
  <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;padding:14px 20px;min-width:120px;">
    <div style="font-size:11px;color:#6b7280;text-transform:uppercase;">Bets</div>
    <div style="font-size:18px;font-weight:700;">{total_bets}</div>
  </div>
</div>

<table style="border-collapse:collapse;width:100%;font-size:13px;">
  <tr style="background:#1e3a5f;">
    <th style="padding:8px 10px;text-align:left;color:#93c5fd;">Player</th>
    <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Side</th>
    <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Line</th>
    <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Odds</th>
    <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Actual K</th>
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
                "Body": {
                    "Html": {"Data": html_body, "Charset": "UTF-8"},
                    "Text": {"Data": subject, "Charset": "UTF-8"},
                },
            },
        )
        print(f"  Email sent: {subject}")
    except Exception as e:
        print(f"  Email failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=yesterday_et())
    args    = parser.parse_args()
    gameday = args.gameday

    print(f"\nMLB Strikeouts Settlement | gameday={gameday}", flush=True)

    recs = load_recommendations(gameday)
    if recs is None:
        print(f"  No recommendations found for {gameday} — nothing to settle")
        return
    print(f"  Recommendations: {len(recs)} bets  {recs['player_key'].nunique()} pitchers")

    # Fetch actual K results from MLB Stats API via boxscores
    print("Fetching pitcher results from MLB API...", flush=True)
    actuals = fetch_all_starters(gameday)

    if not actuals:
        # Fallback: fetch per player_id if boxscore approach fails
        player_ids = recs["player_id"].dropna().astype(int).unique().tolist()
        actuals_by_id = fetch_pitcher_results(player_ids, gameday)
        # Map back to player_key via recommendations
        id_to_key = recs.dropna(subset=["player_id"]).set_index("player_id")["player_key"].to_dict()
        actuals = {id_to_key.get(int(pid), ""): v for pid, v in actuals_by_id.items() if id_to_key.get(int(pid))}

    if not actuals:
        print(f"  No game results found for {gameday} — games may not be complete yet")
        return

    print(f"  Actuals found: {len(actuals)} pitchers")

    settled_new = settle(recs, actuals)
    print(f"  Settled: {len(settled_new)} bets")
    cols_show = [c for c in ["player", "side", "line", "actual_k", "outcome", "pnl"] if c in settled_new.columns]
    print(settled_new[cols_show].to_string(index=False))

    # Append to history
    settled_all = load_settled()
    if not settled_all.empty:
        settled_all = settled_all[settled_all["game_date"] != gameday]
    settled_all = pd.concat([settled_all, settled_new], ignore_index=True)
    save_settled(settled_all)

    total_pnl  = settled_all["pnl"].sum()
    total_bets = len(settled_all[settled_all["outcome"] != "DNP"])
    print(f"\n  All-time: {total_bets} bets · {total_pnl:+.2f}u cumulative PnL")

    wins   = (settled_new["outcome"] == "WIN").sum()
    losses = (settled_new["outcome"] == "LOSS").sum()
    day_pnl = settled_new["pnl"].sum()
    subject = f"MLB Strikeouts Settlement {gameday} — {wins}W/{losses}L · {day_pnl:+.2f}u"
    send_email(subject, build_settlement_html(settled_new, gameday))
    print("Done.")


if __name__ == "__main__":
    main()
