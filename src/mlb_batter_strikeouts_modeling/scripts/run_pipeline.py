"""
Live gameday pipeline for MLB batter strikeouts props.

Strategy: consensus_line price filter — NOT an ML model.
  - consensus_yhat = avg offered_line per player-game across all books (≈1.5 for line=1.5)
  - logistic(consensus_yhat, offered_line) → p_under ≈ 65%
  - edge_under = p_under − raw_implied_prob_under (vig-inclusive)
  - Bet: line=1.5, bookmaker≠fliff, edge≥5pp (play) / edge≥3pp (track)

OOF backtest: 2,580 bets, +140.41u, +5.44% ROI, calmar=3.02, max_dd=28 days (2026-07-11)

S3 paths read:
  s3://the-odds-api-mt/mlb/batter_strikeouts_model/artifacts/calib_logistic_consensus.pkl

S3 paths written:
  s3://the-odds-api-mt/mlb/batter_strikeouts_model/daily_runs/{gameday}/recommendations.csv

Usage:
  python src/mlb_batter_strikeouts_modeling/scripts/run_pipeline.py
  python src/mlb_batter_strikeouts_modeling/scripts/run_pipeline.py --gameday 2026-07-11
"""
from __future__ import annotations

import argparse
import html as html_module
import os
import pickle
import re
import sys
import time
import unicodedata
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

CONFIG_PATH  = Path(__file__).resolve().parents[1] / "config.yaml"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "baseball_mlb"
MARKET        = "batter_strikeouts"
REGIONS       = "us,us2"
SLEEP_S       = 0.25

S3_BUCKET    = "the-odds-api-mt"
CALIB_KEY    = "mlb/batter_strikeouts_model/artifacts/calib_logistic_consensus.pkl"
DAILY_PREFIX = "mlb/batter_strikeouts_model/daily_runs"

SES_SOURCE = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()

ET    = ZoneInfo("America/New_York")
_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"

TEAM_ABBREV: dict[str, str] = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}

BOOK_FULLNAME = {
    "draftkings":     "DraftKings",
    "fanduel":        "FanDuel",
    "betmgm":         "BetMGM",
    "betrivers":      "BetRivers",
    "caesars":        "Caesars",
    "pointsbetus":    "PointsBet",
    "betparx":        "BetParx",
    "espnbet":        "ESPN Bet",
    "fliff":          "Fliff",
    "hardrockbet":    "Hard Rock Bet",
    "mybookieag":     "MyBookie",
    "bovada":         "Bovada",
    "lowvig":         "LowVig",
    "superbook":      "SuperBook",
    "betus":          "BetUS",
    "williamhill_us": "Caesars",
}


# ── Utilities ──────────────────────────────────────────────────────────────────

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


def to_american(dec: float) -> str:
    try:
        dec = float(dec)
        if dec >= 2.0:
            return f"+{int(round((dec - 1) * 100))}"
        return str(int(round(-100 / (dec - 1))))
    except (TypeError, ValueError, ZeroDivisionError):
        return "—"


def commence_to_et(commence_time: str) -> str:
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        return dt.astimezone(ET).strftime("%I:%M %p").lstrip("0")
    except Exception:
        return ""


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def book_name(key: str) -> str:
    return BOOK_FULLNAME.get(key, key.title())


def team_abbrev(name: str) -> str:
    return TEAM_ABBREV.get(name, name[:3].upper() if name else "?")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_calib() -> dict:
    body = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=CALIB_KEY)["Body"].read()
    return pickle.loads(body)


# ── Odds API fetch ─────────────────────────────────────────────────────────────

def fetch_events(gameday: str) -> list[dict]:
    url = f"{ODDS_API_BASE}/sports/{SPORT}/events"
    resp = requests.get(url, params={"apiKey": ODDS_API_KEY, "dateFormat": "iso"}, timeout=30)
    resp.raise_for_status()
    return [e for e in resp.json() if e.get("commence_time", "")[:10] == gameday]


def fetch_odds_for_event(event_id: str, commence_time: str) -> list[dict]:
    url = f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY, "regions": REGIONS,
        "markets": MARKET, "oddsFormat": "decimal", "dateFormat": "iso",
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 422:
        return []
    resp.raise_for_status()
    time.sleep(SLEEP_S)
    return resp.json().get("bookmakers", [])


def parse_bookmakers(bookmakers: list[dict], event_id: str, home: str, away: str,
                     commence_time: str) -> list[dict]:
    rows = []
    for bk in bookmakers:
        bk_key = bk.get("key", "")
        for market in bk.get("markets", []):
            if market.get("key") != MARKET:
                continue
            outcomes = {o["name"]: o for o in market.get("outcomes", [])}
            over_o  = outcomes.get("Over")
            under_o = outcomes.get("Under")
            if not over_o or not under_o:
                continue
            player = over_o.get("description", "") or over_o.get("name", "")
            if not player:
                continue
            rows.append({
                "event_id":      event_id,
                "home_team":     home,
                "away_team":     away,
                "commence_time": commence_time,
                "game_time_et":  commence_to_et(commence_time),
                "bookmaker":     bk_key,
                "player_name":   player,
                "offered_line":  float(over_o.get("point", 0)),
                "over_price":    float(over_o.get("price", 1.0)),
                "under_price":   float(under_o.get("price", 1.0)),
            })
    return rows


# ── Scoring ────────────────────────────────────────────────────────────────────

def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["raw_implied_prob_over"]  = 1.0 / df["over_price"]
    df["raw_implied_prob_under"] = 1.0 / df["under_price"]
    total = df["raw_implied_prob_over"] + df["raw_implied_prob_under"]
    df["novig_prob_over"]  = df["raw_implied_prob_over"]  / total
    df["novig_prob_under"] = df["raw_implied_prob_under"] / total
    return df


def score_consensus(df: pd.DataFrame, calib: dict) -> pd.DataFrame:
    """
    consensus_yhat = avg offered_line per player-game (book-invariant).
    logistic(consensus_yhat, offered_line) → p_over → p_under.
    Edge = p_under - raw_implied_prob_under (vig-inclusive).
    """
    df = df.copy()
    df["consensus_yhat"] = (
        df.groupby(["player_name", "commence_time"])["offered_line"].transform("mean")
    )
    X = np.column_stack([df["consensus_yhat"].values, df["offered_line"].values])
    X_s = calib["scaler"].transform(X)
    p_over = calib["logistic"].predict_proba(X_s)[:, 1]
    df["p_model_over"]  = p_over
    df["p_model_under"] = 1.0 - p_over
    df["edge_under"] = df["p_model_under"] - df["raw_implied_prob_under"]
    df["edge_over"]  = df["p_model_over"]  - df["raw_implied_prob_over"]
    return df


# ── Strategy filtering ─────────────────────────────────────────────────────────

def apply_strategy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    strategy = cfg["strategy"]
    min_bet   = strategy["min_bet_edge"]
    min_track = strategy["min_track_edge"]
    lines     = strategy.get("lines", [1.5])
    book_filter = strategy.get("book_filter", "no_fliff")

    df = df[df["offered_line"].isin(lines)].copy()

    if book_filter == "no_fliff":
        df = df[df["bookmaker"] != "fliff"].copy()

    df["tier"] = None
    df.loc[df["edge_under"] >= min_bet,   "tier"] = "play"
    df.loc[(df["edge_under"] >= min_track) & (df["edge_under"] < min_bet), "tier"] = "track"
    return df


# ── HTML email ─────────────────────────────────────────────────────────────────

def build_email_html(all_rows: pd.DataFrame, gameday: str, n_plays: int, n_tracks: int) -> str:
    """Full column-group table email for Today's Plays section."""
    def e(s: object) -> str:
        return html_module.escape(str(s))

    def fp(v: object) -> str:
        return f"{float(v)*100:.1f}%" if pd.notna(v) else "—"

    def fe(v: object) -> str:
        return f"{float(v)*100:+.1f}pp" if pd.notna(v) else "—"

    today_et = datetime.now(ET).strftime("%Y-%m-%d %I:%M %p ET")
    n_games  = all_rows.groupby(["home_team", "away_team", "commence_time"]).ngroups

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>body{{font-family:{_SANS};font-size:14px;background:#f5f5f5;margin:0;padding:20px;color:#1a1a1a}}</style>"
        "</head><body>"
        f"<h2 style='margin-bottom:4px'>⚾ MLB Batter Strikeouts — {e(gameday)}</h2>"
        f"<p style='font-size:15px;font-weight:600;margin:0 0 4px'>"
        f"{n_plays} {'play' if n_plays == 1 else 'plays'} · "
        f"{n_tracks} tracked · across {n_games} {'game' if n_games == 1 else 'games'}</p>"
        f"<p style='color:#888;font-size:12px;margin:0 0 4px'>Generated {today_et}</p>"
        "<p style='color:#888;font-size:12px;margin:0 0 16px'>"
        "Strategy: UNDER 1.5 strikeouts · edge ≥5pp (play) · ≥3pp (track) · "
        "consensus_line price filter · no Fliff</p>"
        "<h3 style='font-size:14px;font-weight:600;margin:0 0 8px;color:#111827'>Today's Plays</h3>"
    )

    _GRP = "padding:5px 8px;text-align:center;font-size:10px;font-weight:600;letter-spacing:.5px;text-transform:uppercase"
    _COL = "padding:6px 10px"
    _S1  = "border-right:2px solid #374f5e"
    _S2  = "border-right:2px solid #1e2a35"

    html += (
        "<div style='overflow-x:auto'>"
        "<table style='border-collapse:collapse;font-size:12px;width:100%;background:#fff;white-space:nowrap'>"
        "<thead>"
        f"<tr style='background:#1e2a35;color:#aab8c2'>"
        f"<th colspan='5' style='{_GRP};{_S1}'>Player / Game</th>"
        f"<th colspan='1' style='{_GRP};{_S1}'>Book</th>"
        f"<th colspan='2' style='{_GRP};{_S1}'>American Odds</th>"
        f"<th colspan='3' style='{_GRP};{_S1}'>Implied</th>"
        f"<th colspan='4' style='{_GRP};{_S1}'>No-Vig</th>"
        f"<th colspan='3' style='{_GRP};{_S1}'>Model Prediction</th>"
        f"<th colspan='2' style='{_GRP};{_S1}'>Edge</th>"
        f"<th colspan='2' style='{_GRP};{_S1}'>Model Inputs</th>"
        f"<th colspan='1' style='{_GRP}'>Status</th>"
        "</tr>"
        f"<tr style='background:#2c3e50;color:white'>"
        f"<th style='{_COL};text-align:left;min-width:130px'>Player</th>"
        f"<th style='{_COL}'>Away</th>"
        f"<th style='{_COL}'>Home</th>"
        f"<th style='{_COL}'>Time (ET)</th>"
        f"<th style='{_COL};{_S2}'>Line</th>"
        f"<th style='{_COL};{_S2};min-width:90px'>Book</th>"
        f"<th style='{_COL}'>Over</th>"
        f"<th style='{_COL};{_S2}'>Under</th>"
        f"<th style='{_COL}'>Raw Over</th>"
        f"<th style='{_COL}'>Raw Under</th>"
        f"<th style='{_COL};{_S2}'>Raw Total</th>"
        f"<th style='{_COL}'>Fair Over</th>"
        f"<th style='{_COL}'>Fair Under</th>"
        f"<th style='{_COL}'>Fair Total</th>"
        f"<th style='{_COL};{_S2}'>Vig</th>"
        f"<th style='{_COL}'>Consensus Line</th>"
        f"<th style='{_COL}'>Pred Over</th>"
        f"<th style='{_COL};{_S2}'>Pred Under</th>"
        f"<th style='{_COL}'>Over Edge</th>"
        f"<th style='{_COL};{_S2}'>Under Edge</th>"
        f"<th style='{_COL}'>Consensus Line</th>"
        f"<th style='{_COL};{_S2}'>Rolling Features</th>"
        f"<th style='{_COL}'>Status</th>"
        "</tr>"
        "</thead><tbody>"
    )

    for (home, away, commence_time), game_rows in (
        all_rows.sort_values("commence_time")
                .groupby(["home_team", "away_team", "commence_time"], sort=False)
    ):
        time_str     = commence_to_et(commence_time)
        n_gp         = int((game_rows["tier"] == "play").sum())
        n_gt         = int((game_rows["tier"] == "track").sum())

        html += (
            f"<tr style='background:#1a1a2e;color:white;font-weight:bold'>"
            f"<td colspan='23' style='padding:8px 12px'>"
            f"⚾ {e(away)} @ {e(home)} · {time_str} · "
            f"{n_gp} {'play' if n_gp == 1 else 'plays'}, {n_gt} tracked"
            f"</td></tr>"
        )

        play_rows  = game_rows[game_rows["tier"] == "play"].sort_values("edge_under", ascending=False)
        track_rows = game_rows[game_rows["tier"] == "track"].sort_values("edge_under", ascending=False)
        nobet_rows = (
            game_rows[game_rows["tier"].isna()]
            .sort_values("edge_under", ascending=False)
            .drop_duplicates(subset=["player_name"], keep="first")
        )

        for _, row in pd.concat([play_rows, track_rows, nobet_rows]).iterrows():
            tier = row.get("tier")
            if pd.isna(tier) or tier is None:
                tier = "no-bet"

            if tier == "play":
                bg, tc, fw = "#e8f5e9", "#1b5e20", "font-weight:600"
                ec = "#1b5e20;font-weight:700"
                status_label = "BET UNDER"
            elif tier == "track":
                bg, tc, fw = "#fff8e1", "#6d4c00", "font-weight:600"
                ec = "#6d4c00;font-weight:600"
                status_label = "TRACK UNDER"
            else:
                bg, tc, fw = "#fafafa", "#888", ""
                ec = "#aaa"
                status_label = "—"

            def td(val: object, c: str = "") -> str:
                cc = c or tc
                return f"<td style='padding:7px 10px;color:{cc};{fw}'>{val}</td>"

            raw_o  = row.get("raw_implied_prob_over",  np.nan)
            raw_u  = row.get("raw_implied_prob_under", np.nan)
            raw_t  = (float(raw_o) + float(raw_u)) if (pd.notna(raw_o) and pd.notna(raw_u)) else np.nan
            vig_v  = (float(raw_t) - 1.0) if pd.notna(raw_t) else np.nan
            cyhat  = row.get("consensus_yhat", row.get("offered_line", 1.5))
            cyhat_s = f"{float(cyhat):.1f}" if pd.notna(cyhat) else "—"

            html += (
                f"<tr style='background:{bg}'>"
                + td(e(row["player_name"]))
                + td(e(team_abbrev(row.get("away_team", ""))))
                + td(e(team_abbrev(row.get("home_team", ""))))
                + td(e(row.get("game_time_et", "")))
                + td(e(row.get("offered_line", "")))
                + td(e(book_name(row.get("bookmaker", ""))))
                + td(e(to_american(row.get("over_price"))))
                + td(e(to_american(row.get("under_price"))))
                + td(fp(raw_o))
                + td(fp(raw_u))
                + td(fp(raw_t))
                + td(fp(row.get("novig_prob_over")))
                + td(fp(row.get("novig_prob_under")))
                + td("100.0%")
                + td(fp(vig_v))
                + td(cyhat_s)
                + td(fp(row.get("p_model_over")))
                + td(fp(row.get("p_model_under")))
                + td(fe(row.get("edge_over")), "#aaa")
                + f"<td style='padding:7px 10px;color:{ec}'>{fe(row.get('edge_under'))}</td>"
                + td(cyhat_s)
                + td("N/A", "#aaa")
                + f"<td style='padding:7px 10px;color:{tc};{fw}'>{status_label}</td>"
                + "</tr>"
            )

    html += (
        "</tbody></table></div>"
        "<p style='font-size:11px;color:#9ca3af;margin-top:24px;padding-top:12px;border-top:1px solid #f3f4f6'>"
        "MLB Batter Strikeouts · consensus_line price filter · "
        "OOF: 2,580 bets, +140.41u, +5.44% ROI, calmar=3.02, max DD=28 days · "
        "Strategy: books misprice OVER on batter 1.5 K line for low-K hitters · "
        "sklearn 1.6.1"
        "</p>"
    )
    return html


# ── S3 save ────────────────────────────────────────────────────────────────────

def save_recommendations(df: pd.DataFrame, gameday: str) -> None:
    key = f"{DAILY_PREFIX}/{gameday}/recommendations.csv"
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    print(f"  Saved recs → s3://{S3_BUCKET}/{key}")


def send_email(subject: str, html_body: str) -> None:
    if not SES_SOURCE or not SES_TO_RAW:
        print("  SES not configured — skipping email")
        return
    to_list = [e.strip() for e in SES_TO_RAW.split(",") if e.strip()]
    boto3.client("ses", region_name="us-east-2").send_email(
        Source=SES_SOURCE,
        Destination={"ToAddresses": to_list},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body":    {"Html":  {"Data": html_body, "Charset": "UTF-8"}},
        },
    )
    print(f"  Email sent: {subject[:80]}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(gameday: str | None = None, output_path: str | None = None):
    if gameday is None:
        gameday = datetime.now(ET).strftime("%Y-%m-%d")

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    print(f"MLB Batter Strikeouts pipeline | gameday={gameday}")

    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY not set")

    # 1. Fetch events
    print("Fetching events...")
    events = fetch_events(gameday)
    print(f"  {len(events)} games on {gameday}")
    if not events:
        print("No games — exiting.")
        return

    # 2. Fetch odds
    print("Fetching odds...")
    all_rows = []
    for ev in events:
        bks = fetch_odds_for_event(ev["id"], ev.get("commence_time", ""))
        rows = parse_bookmakers(bks, ev["id"], ev.get("home_team", ""),
                                ev.get("away_team", ""), ev.get("commence_time", ""))
        all_rows.extend(rows)

    if not all_rows:
        print(f"No {MARKET} lines found for {gameday}.")
        if output_path:
            import json
            Path(output_path).write_text(json.dumps(
                {"n_play_bets": 0, "n_track_bets": 0, "html_body": "<p>No lines found.</p>"}
            ))
        return

    df = pd.DataFrame(all_rows)
    print(f"  {len(df):,} odds rows · {df['player_name'].nunique()} players · "
          f"{df['bookmaker'].nunique()} books")

    # 3. Market features + consensus scoring
    df = compute_market_features(df)
    print("Loading calibration + scoring...")
    calib = load_calib()
    df = score_consensus(df, calib)

    # 4. Apply strategy
    scored = apply_strategy(df, cfg)
    plays  = scored[scored["tier"] == "play"]
    tracks = scored[scored["tier"] == "track"]
    print(f"  Plays: {len(plays)} · Tracks: {len(tracks)}")

    # 5. Build + send email
    # pass all rows (including no-bet) for display; save only play/track
    html_body = build_email_html(scored, gameday, len(plays), len(tracks))
    n_plays   = len(plays)
    subject   = f"MLB Batter Strikeouts — {n_plays} {'play' if n_plays == 1 else 'plays'} · {gameday}"

    save_df = scored[scored["tier"].notna()].copy()
    if not save_df.empty:
        save_recommendations(save_df, gameday)
    send_email(subject, html_body)

    if output_path:
        import json
        Path(output_path).write_text(json.dumps({
            "n_play_bets":  n_plays,
            "n_track_bets": int(len(tracks)),
            "html_body":    html_body,
        }))

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", type=str, default=None)
    parser.add_argument("--output",  type=str, default=None)
    args = parser.parse_args()
    main(gameday=args.gameday, output_path=args.output)
