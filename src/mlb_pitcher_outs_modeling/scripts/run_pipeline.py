"""
MLB Pitcher Outs daily pipeline — 9am ET combined email.

Sections (in order):
  1. Today's plays  — live props scored, per-book rows, grouped by game
  2. Yesterday's results — settle yesterday's bets from recommendations.csv
  3. All-time results    — season stat cards + by-season table

Strategy (from config.yaml):
  direction:    under
  odds_bucket:  minus_odds (under_price ≤ 2.00)
  edge_play:    ≥ 5pp  (raw implied: p_model - 1/under_price)
  edge_show:    ≥ 2pp  (display only)
  shrinkage:    0.25
  line_max:     17.5
  model:        consensus_line (yhat = consensus_line; no OLS/ML model)

S3 reads:
  mlb/pitcher_outs_model/spine/mlb_pitcher_outs_spine.parquet
  mlb/pitcher_outs_model/model/mlb_pitcher_outs_residuals.npy
  mlb/pitcher_outs_model/daily_runs/{yesterday}/recommendations.csv
  mlb/pitcher_outs_model/settled/settled_bets.parquet

S3 writes:
  mlb/pitcher_outs_model/daily_runs/{gameday}/scored.csv
  mlb/pitcher_outs_model/daily_runs/{gameday}/recommendations.csv
  mlb/pitcher_outs_model/daily_runs/{yesterday}/settled.csv
  mlb/pitcher_outs_model/settled/settled_bets.parquet
"""
from __future__ import annotations

import argparse
import html as html_module
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import botocore.exceptions
import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "baseball_mlb"
SLEEP_S       = 0.25

S3_BUCKET    = "the-odds-api-mt"
SPINE_KEY    = "mlb/pitcher_outs_model/spine/mlb_pitcher_outs_spine.parquet"
RESID_KEY    = "mlb/pitcher_outs_model/model/mlb_pitcher_outs_residuals.npy"
DAILY_PREFIX = "mlb/pitcher_outs_model/daily_runs"
SETTLED_KEY  = "mlb/pitcher_outs_model/settled/settled_bets.parquet"
GAMELOG_PREFIX = "mlb/strikeouts_model/pitcher_gamelogs"

SES_SOURCE    = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()

ET  = ZoneInfo("America/New_York")

N_BOOT   = 10000
RNG_SEED = 42

_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"

BOOK_DISPLAY = {
    "draftkings": "DK", "fanduel": "FD", "betmgm": "MGM", "betrivers": "BR",
    "caesars": "CZR", "betparx": "PX", "espnbet": "ESPN", "fliff": "Fliff",
    "hardrockbet": "HardRock", "bovada": "Bovada", "betonlineag": "BOL",
    "superbook": "SBK", "fanatics": "Fanatics", "ballybet": "Bally",
    "williamhill_us": "WH", "mybookieag": "MB", "pinnacle": "PIN",
}

TEAM_ABBREV = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    "Sacramento River Cats": "SAC",
}

NAME_MAP = {
    "louie varland": "louis varland",
    "luis l ortiz":  "luis ortiz",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")

def yesterday_et() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")

def commence_to_et(commence_time: str) -> str:
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        s = dt.astimezone(ET).strftime("%I:%M %p")
        return s.lstrip("0")
    except Exception:
        return ""

def to_american(dec) -> str:
    try:
        d = float(dec)
        return f"+{int(round((d-1)*100))}" if d >= 2.0 else str(int(round(-100/(d-1))))
    except (TypeError, ValueError, ZeroDivisionError):
        return "—"

def display_book(key: str) -> str:
    return BOOK_DISPLAY.get(str(key).lower(), str(key)[:6].upper())

def team_abbr(name: str) -> str:
    return TEAM_ABBREV.get(name, name[:3].upper())

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.lower()
    n = unicodedata.normalize("NFD", n)
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")
    n = re.sub(r"[.,'\-]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    n = " ".join(n.split())
    return NAME_MAP.get(n, n)

def ip_to_outs(ip_val) -> float | None:
    if ip_val is None or (isinstance(ip_val, float) and np.isnan(ip_val)):
        return None
    try:
        ip = float(ip_val)
        full = int(ip)
        frac = round((ip - full) * 10)
        return float(full * 3 + frac)
    except (TypeError, ValueError):
        return None

def _time_sort_key(t: str) -> int:
    try:
        h, rest = t.split(":")
        m, ampm = rest.strip().split(" ")
        h, m = int(h), int(m)
        if ampm == "PM" and h != 12: h += 12
        elif ampm == "AM" and h == 12: h = 0
        return h * 60 + m
    except Exception:
        return 9999

def fmt(v, fmt_str):
    try: return format(float(v), fmt_str)
    except: return "—"


# ─── S3 ───────────────────────────────────────────────────────────────────────

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

def s3_get_csv(key: str) -> pd.DataFrame:
    body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_csv(BytesIO(body))


# ─── Odds API ─────────────────────────────────────────────────────────────────

def fetch_live_events(gameday: str) -> list[dict]:
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY not set")
    r = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT}/events",
        params={"apiKey": ODDS_API_KEY},
        timeout=30,
    )
    if r.status_code != 200:
        print(f"  Events API error: {r.status_code}")
        return []
    events = []
    for e in r.json():
        ct = e.get("commence_time", "")
        if ct:
            try:
                dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                if dt.astimezone(ET).strftime("%Y-%m-%d") == gameday:
                    events.append(e)
            except Exception:
                pass
    print(f"  Live events for {gameday}: {len(events)}")
    return events


def fetch_event_pitcher_outs(event_id: str, regions: str) -> list[dict]:
    r = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds",
        params={"apiKey": ODDS_API_KEY, "markets": "pitcher_outs",
                "regions": regions, "oddsFormat": "decimal"},
        timeout=30,
    )
    time.sleep(SLEEP_S)
    if r.status_code != 200 or not r.json():
        return []
    rows = []
    for bm in r.json().get("bookmakers", []):
        book = bm["key"]
        for mkt in bm.get("markets", []):
            if mkt["key"] != "pitcher_outs":
                continue
            under_map = {
                (o.get("description", ""), o.get("point")): o.get("price")
                for o in mkt.get("outcomes", [])
                if o.get("name") == "Under"
            }
            for o in mkt.get("outcomes", []):
                if o.get("name") != "Over":
                    continue
                player = o.get("description", "")
                pt = o.get("point")
                rows.append({
                    "bookmaker":   book,
                    "player_name": player,
                    "line":        pt,
                    "over_price":  o.get("price"),
                    "under_price": under_map.get((player, pt)),
                })
    return rows


# ─── Market build ─────────────────────────────────────────────────────────────

def build_market_df(
    rows: list[dict],
    event_id: str,
    home_team: str,
    away_team: str,
    gameday: str,
    game_time_et: str,
    line_min: float,
    line_max: float,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[
        df["over_price"].notna() & df["under_price"].notna()
        & (df["over_price"] > 1.0) & (df["under_price"] > 1.0)
    ].copy()
    if df.empty:
        return pd.DataFrame()

    df = df[(df["line"] >= line_min) & (df["line"] <= line_max)].copy()
    if df.empty:
        return pd.DataFrame()

    df["player_key"]   = df["player_name"].map(normalize_name)
    df["event_id"]     = event_id
    df["home_team"]    = home_team
    df["away_team"]    = away_team
    df["game_date"]    = gameday
    df["game_time_et"] = game_time_et

    df = df.drop_duplicates(subset=["player_key", "bookmaker", "line"])

    # Per-book novig
    raw_o = 1.0 / df["over_price"]
    raw_u = 1.0 / df["under_price"]
    total = raw_o + raw_u
    df["novig_prob_over"]  = raw_o / total
    df["novig_prob_under"] = raw_u / total

    # Consensus line per player (avg across all books/lines for that player)
    df["consensus_line"] = df.groupby("player_key")["line"].transform("mean")

    # n_books per player-line
    df["n_books"] = df.groupby(["player_key", "line"])["bookmaker"].transform("nunique")

    # Consensus avg over price per player-line (for Mkt Over column)
    df["mkt_over_am"] = (
        df.groupby(["player_key", "line"])["over_price"]
        .transform("mean")
        .map(to_american)
    )

    return df


# ─── Spine ────────────────────────────────────────────────────────────────────

def get_latest_features(spine: pd.DataFrame) -> pd.DataFrame:
    """Most recent row per player_key — used for rolling feature display in email."""
    return spine.sort_values("game_date").groupby("player_key").tail(1).set_index("player_key")


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_slate(
    market_df: pd.DataFrame,
    latest_feats: pd.DataFrame,
    residuals: np.ndarray,
    shrinkage: float,
    min_books: int,
) -> pd.DataFrame:
    """
    Score one row per (player_key, line, bookmaker).
    Model: yhat = consensus_line (no OLS/ML).
    Bootstrap P(under) from residuals distribution.
    Shrinkage applied after bootstrap.
    """
    np.random.seed(RNG_SEED)

    # Draw one bootstrap sample set per player (ensures monotonicity across lines)
    player_samples: dict[str, np.ndarray] = {}

    rows_out = []
    for _, row in market_df.iterrows():
        pk = row["player_key"]

        if row.get("n_books", 0) < min_books:
            continue

        line   = float(row["line"])
        y_hat  = float(row["consensus_line"])

        if pk not in player_samples:
            player_samples[pk] = y_hat + np.random.choice(residuals, N_BOOT, replace=True)

        boot    = player_samples[pk]
        p_under = float(np.mean(boot < line))
        p_over  = float(np.mean(boot > line))

        p_under_s = p_under * (1 - shrinkage) + 0.5 * shrinkage
        p_over_s  = p_over  * (1 - shrinkage) + 0.5 * shrinkage

        edge_under = p_under_s - (1.0 / float(row["under_price"]))
        edge_over  = p_over_s  - (1.0 / float(row["over_price"]))

        # Rolling features from spine (display only)
        feat = latest_feats.loc[pk] if pk in latest_feats.index else {}

        rows_out.append({
            "player_key":      pk,
            "player_name":     row.get("player_name", pk),
            "home_team":       row["home_team"],
            "away_team":       row["away_team"],
            "game_time_et":    row["game_time_et"],
            "event_id":        row["event_id"],
            "bookmaker":       row["bookmaker"],
            "line":            line,
            "over_price":      row["over_price"],
            "under_price":     row["under_price"],
            "novig_prob_over":  row["novig_prob_over"],
            "novig_prob_under": row["novig_prob_under"],
            "mkt_over_am":     row.get("mkt_over_am", "—"),
            "n_books":         row["n_books"],
            "consensus_line":  y_hat,
            "y_hat":           y_hat,
            "p_under":         p_under_s,
            "p_over":          p_over_s,
            "edge_under":      edge_under,
            "edge_over":       edge_over,
            # Rolling features for email
            "outs_roll_career": _get(feat, "outs_roll_career"),
            "outs_roll_c5":     _get(feat, "outs_roll_c5"),
            "k_roll_career":    _get(feat, "k_roll_career"),
            "opp_k_against_season": _get(feat, "opp_k_against_season"),
        })

    return pd.DataFrame(rows_out) if rows_out else pd.DataFrame()


def _get(feat, key):
    if isinstance(feat, pd.Series):
        v = feat.get(key, np.nan)
    elif isinstance(feat, dict):
        v = feat.get(key, np.nan)
    else:
        return np.nan
    return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan


# ─── Settlement ───────────────────────────────────────────────────────────────

def settle_yesterday(yesterday: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Settle yesterday's recommendations. Returns (settled_today, all_time).
    If no recs or no gamelogs, returns (empty, all_time_from_s3).
    """
    # Load all-time history first
    try:
        all_time = s3_get_parquet(SETTLED_KEY)
    except Exception:
        all_time = pd.DataFrame()

    # Load yesterday's recommendations
    recs_key = f"{DAILY_PREFIX}/{yesterday}/recommendations.csv"
    try:
        recs = s3_get_csv(recs_key)
        print(f"  Loaded {len(recs)} recs for {yesterday}")
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            print(f"  No recommendations for {yesterday} (no bets placed)")
            return pd.DataFrame(), all_time
        raise

    if recs.empty:
        return pd.DataFrame(), all_time

    # Load actuals from pitcher gamelogs
    season = int(yesterday[:4])
    prefix = f"{GAMELOG_PREFIX}/{season}/"
    s3 = _s3()
    try:
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        frames = []
        for obj in resp.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                try:
                    body = s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read()
                    frames.append(pd.read_parquet(BytesIO(body)))
                except Exception:
                    pass
        gamelogs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception:
        gamelogs = pd.DataFrame()

    if gamelogs.empty:
        print(f"  ⚠ No gamelogs for {yesterday} — try again later")
        return pd.DataFrame(), all_time

    gamelogs["game_date_str"] = gamelogs["game_date"].astype(str)
    daily_logs = gamelogs[gamelogs["game_date_str"] == yesterday].copy()
    daily_logs["player_key"]  = daily_logs["player_name"].map(normalize_name)
    daily_logs["outs_actual"] = daily_logs["innings_pitched"].map(ip_to_outs)
    actuals_map = daily_logs.dropna(subset=["outs_actual"]).set_index("player_key")["outs_actual"].to_dict()

    if "player_key" not in recs.columns:
        recs["player_key"] = recs["player_name"].map(normalize_name)
    recs["outs_actual"] = recs["player_key"].map(actuals_map)

    n_matched = recs["outs_actual"].notna().sum()
    print(f"  Actuals matched: {n_matched}/{len(recs)}")

    def _settle(row):
        if pd.isna(row["outs_actual"]):
            return np.nan
        o, l = float(row["outs_actual"]), float(row["line"])
        return float(row["under_price"]) - 1.0 if o < l else (-1.0 if o > l else 0.0)

    recs["pnl"]       = recs.apply(_settle, axis=1)
    recs["game_date"] = yesterday
    settled = recs[recs["pnl"].notna()].copy()

    if settled.empty:
        print("  No settled bets (actuals not available yet)")
        return pd.DataFrame(), all_time

    # Save daily settled
    s3_put_csv(f"{DAILY_PREFIX}/{yesterday}/settled.csv", settled)
    print(f"  Saved settled.csv for {yesterday} ({len(settled)} rows)")

    # Append to all-time (idempotent — remove prior settlement for this date first)
    if not all_time.empty and "game_date" in all_time.columns:
        all_time = all_time[all_time["game_date"].astype(str) != yesterday]
    all_time = pd.concat([all_time, settled], ignore_index=True)
    s3_put_parquet(SETTLED_KEY, all_time)
    print(f"  All-time settled: {len(all_time)} rows")

    w = int((settled["pnl"] > 0).sum())
    l = int((settled["pnl"] < 0).sum())
    print(f"  Yesterday: {w}W / {l}L = {settled['pnl'].sum():+.2f}u")
    return settled, all_time


# ─── Email ────────────────────────────────────────────────────────────────────

def _card(label: str, value: str, green: bool = False) -> str:
    color = "#276221" if green else "#222"
    return (
        f"<div style='border:1px solid #ddd;border-radius:6px;padding:12px 20px;"
        f"min-width:120px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.06)'>"
        f"<div style='font-size:10px;color:#888;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:.5px;margin-bottom:4px'>{label}</div>"
        f"<div style='font-size:22px;font-weight:700;color:{color}'>{value}</div>"
        f"</div>"
    )

def _cards_row(*cards) -> str:
    return "<div style='display:flex;gap:12px;margin:16px 0 20px;flex-wrap:wrap'>" + "".join(cards) + "</div>"


def _build_section1(scored: pd.DataFrame, gameday: str, edge_play: float, edge_show: float, season_stats: dict | None) -> str:
    """Today's plays — full scored table grouped by game."""
    he = html_module.escape

    df = scored.copy()
    df["tier"] = "none"
    df.loc[df["edge_under"] >= edge_show, "tier"] = "show"
    df.loc[df["edge_under"] >= edge_play, "tier"] = "play"

    n_play     = int((df["tier"] == "play").sum())
    n_show     = int((df["tier"] == "show").sum())
    n_play_pit = df[df["tier"] == "play"]["player_key"].nunique()
    n_show_pit = df[df["tier"] == "show"]["player_key"].nunique()

    df["_tsort"] = df["game_time_et"].map(_time_sort_key)
    df = df.sort_values(["_tsort", "home_team", "edge_under"], ascending=[True, True, False])

    if season_stats:
        u, w, l = season_stats.get("units", 0.0), season_stats.get("wins", 0), season_stats.get("losses", 0)
        cards_html = _cards_row(
            _card(f"{datetime.now(ET).year} PNL (plays)", f"{u:+.2f}u", green=u >= 0),
            _card("Record (plays)", f"{w}W – {l}L"),
            _card("Win %", f"{w/(w+l)*100:.1f}%" if w+l else "—", green=w > l),
            _card("ROI (plays)", f"{u/max(1,w+l)*100:+.1f}%", green=u >= 0),
        )
    else:
        cards_html = _cards_row(
            _card(f"{datetime.now(ET).year} PNL (plays)", "—"),
            _card("Record (plays)", "—"),
            _card("Win %", "—"),
            _card("ROI (plays)", "—"),
        )

    _GH = "background:#1e2a35;color:#aab8c2;padding:5px 8px;text-align:center;font-size:10px;font-weight:600;letter-spacing:.5px;text-transform:uppercase"
    _BR = "border-right:2px solid #374f5e"
    _br = "border-right:2px solid #1e2a35"

    def _ec(edge_val: float) -> str:
        if edge_val >= edge_play:  return "#276221"
        if edge_val >= edge_show:  return "#b8860b"
        if edge_val < 0:           return "#c0392b"
        return "#555"

    games = df[["_tsort","game_time_et","home_team","away_team"]].drop_duplicates(["home_team","away_team"]).sort_values("_tsort")
    rows_html = ""
    for _, g in games.iterrows():
        gdf = df[(df["home_team"]==g["home_team"]) & (df["away_team"]==g["away_team"])]
        n_gplay = int((gdf["tier"]=="play").sum())
        n_gshow = int((gdf["tier"]=="show").sum())
        badges = []
        if n_gplay: badges.append(f"<span style='color:#276221'>{n_gplay} bet{'s' if n_gplay!=1 else ''}</span>")
        if n_gshow: badges.append(f"<span style='color:#b8860b'>{n_gshow} watch</span>")
        if not badges: badges.append("<span style='color:#888'>no plays</span>")

        rows_html += (
            f"<tr style='background:#edf1f5'><td colspan='26' style='padding:7px 10px;"
            f"font-weight:600;font-size:12px;color:#2c3e50;border-top:2px solid #bdc3c7;"
            f"border-bottom:1px solid #bdc3c7'>"
            f"{he(g['game_time_et'])} ET &nbsp;·&nbsp; {he(g['away_team'])} @ {he(g['home_team'])}"
            f" &nbsp;·&nbsp; {gdf['player_key'].nunique()} pitcher{'s' if gdf['player_key'].nunique()!=1 else ''}"
            f" &nbsp;·&nbsp; {' &nbsp;·&nbsp; '.join(badges)}</td></tr>\n"
        )

        for _, r in gdf.iterrows():
            tier  = r["tier"]
            bg    = "background:#eaf6ea" if tier=="play" else ("background:#fffde7" if tier=="show" else "")
            status_html = (
                "<span style='color:#276221;font-weight:bold'>PLAY ✓</span>" if tier=="play"
                else "<span style='color:#b8860b;font-weight:bold'>WATCH</span>" if tier=="show"
                else "<span style='color:#aaa'>—</span>"
            )

            raw_o   = 1.0 / float(r["over_price"])  if float(r["over_price"])  > 0 else 0.0
            raw_u   = 1.0 / float(r["under_price"]) if float(r["under_price"]) > 0 else 0.0
            raw_tot = raw_o + raw_u
            vig_val = raw_tot - 1.0
            delta   = float(r["y_hat"]) - float(r["line"])
            eu      = float(r["edge_under"])
            eo      = float(r["edge_over"])
            d_color = "#276221" if delta >= 0 else "#c0392b"

            rows_html += (
                f"<tr style='{bg}'>"
                # Player / Game
                f"<td>{he(r['player_name'])}</td>"
                f"<td style='text-align:center;color:#555'>{team_abbr(r['home_team'])}</td>"
                f"<td style='text-align:center;color:#555'>{team_abbr(r['away_team'])}</td>"
                f"<td style='text-align:center;color:#555'>{he(r['game_time_et'])}</td>"
                f"<td style='text-align:center;{_br}'>{fmt(r['line'], '.1f')}</td>"
                # Book
                f"<td style='text-align:center'>{he(display_book(r['bookmaker']))}</td>"
                # American Odds
                f"<td style='text-align:center;color:#555'>{to_american(r['over_price'])}</td>"
                f"<td style='text-align:center;font-weight:bold;color:#1d4ed8;{_br}'>{to_american(r['under_price'])}</td>"
                # Implied
                f"<td style='text-align:center'>{fmt(raw_o, '.1%')}</td>"
                f"<td style='text-align:center'>{fmt(raw_u, '.1%')}</td>"
                f"<td style='text-align:center;{_br}'>{fmt(raw_tot, '.1%')}</td>"
                # No-Vig
                f"<td style='text-align:center'>{fmt(r['novig_prob_over'], '.1%')}</td>"
                f"<td style='text-align:center'>{fmt(r['novig_prob_under'], '.1%')}</td>"
                f"<td style='text-align:center'>100.0%</td>"
                f"<td style='text-align:center;{_br}'>{fmt(vig_val, '.1%')}</td>"
                # Model Prediction
                f"<td style='text-align:center;font-size:11px;color:#1565c0;font-weight:bold'>{fmt(r['y_hat'], '.2f')}</td>"
                f"<td style='text-align:center;font-weight:bold;color:{d_color}'>{fmt(delta, '+.2f')}</td>"
                f"<td style='text-align:center'>{fmt(r['p_over'], '.1%')}</td>"
                f"<td style='text-align:center;{_br}'>{fmt(r['p_under'], '.1%')}</td>"
                # Edge
                f"<td style='text-align:center;color:{_ec(eo)}'>{fmt(eo, '+.1%')}</td>"
                f"<td style='text-align:center;font-weight:bold;color:{_ec(eu)};{_br}'>{fmt(eu, '+.1%')}</td>"
                # Model Inputs
                f"<td style='text-align:center;font-size:11px;color:#555'>{fmt(r['outs_roll_career'], '.1f')}</td>"
                f"<td style='text-align:center;font-size:11px;color:#555'>{fmt(r['outs_roll_c5'], '.1f')}</td>"
                f"<td style='text-align:center;font-size:11px;color:#555'>{fmt(r['k_roll_career'], '.1f')}</td>"
                f"<td style='text-align:center;font-size:11px;color:#555;{_br}'>{fmt(r['opp_k_against_season'], '.2f')}</td>"
                # Status
                f"<td style='text-align:center'>{status_html}</td>"
                f"</tr>\n"
            )

    return f"""
<h2 style='color:#2c3e50;margin-bottom:4px'>MLB Pitcher Outs — {gameday}</h2>
<p style='margin-top:4px'>
  <span style='color:#276221;font-weight:bold'>{n_play} bets across {n_play_pit} pitchers (edge ≥{edge_play*100:.0f}%)</span>
  &nbsp;·&nbsp;
  <span style='color:#b8860b;font-weight:bold'>{n_show} on watch list ({edge_show*100:.0f}–{edge_play*100:.0f}%)</span>
  &nbsp;·&nbsp; UNDER · favorites only · outs ≤17.5
</p>
<p style='font-size:11px;color:#888;margin-top:2px'>
  <span class='lp'></span>Green = bet &nbsp;&nbsp;
  <span class='ls'></span>Yellow = watch list &nbsp;&nbsp;
  Grey = no edge (context only)
</p>
{cards_html}
<details open>
  <summary>▸ Strategy: Pitcher Outs UNDER &nbsp;<span style='font-weight:normal;color:#666'>({len(df)} rows evaluated · {n_play} plays)</span></summary>
  <table>
    <tr>
      <th colspan='5' style='{_GH};{_BR}'>Player / Game</th>
      <th colspan='1' style='{_GH}'>Book</th>
      <th colspan='2' style='{_GH};{_BR}'>American Odds</th>
      <th colspan='3' style='{_GH};{_BR}'>Implied</th>
      <th colspan='4' style='{_GH};{_BR}'>No-Vig</th>
      <th colspan='4' style='{_GH};{_BR}'>Model Prediction</th>
      <th colspan='2' style='{_GH};{_BR}'>Edge</th>
      <th colspan='4' style='{_GH};{_BR}'>Model Inputs</th>
      <th colspan='1' style='{_GH}'>Status</th>
    </tr>
    <tr>
      <th>Pitcher</th>
      <th style='text-align:center'>Home</th>
      <th style='text-align:center'>Away</th>
      <th style='text-align:center'>Time (ET)</th>
      <th style='text-align:center;{_br}'>Line</th>
      <th style='text-align:center'>Book</th>
      <th style='text-align:center'>Over</th>
      <th style='text-align:center;{_br}'>Under</th>
      <th style='text-align:center'>Raw<br>Over%</th>
      <th style='text-align:center'>Raw<br>Under%</th>
      <th style='text-align:center;{_br}'>Raw<br>Total</th>
      <th style='text-align:center'>Fair<br>Over%</th>
      <th style='text-align:center'>Fair<br>Under%</th>
      <th style='text-align:center'>Fair<br>Total</th>
      <th style='text-align:center;{_br}'>Vig</th>
      <th style='text-align:center'>Proj<br>Outs</th>
      <th style='text-align:center'>Delta</th>
      <th style='text-align:center'>Pred<br>Over%</th>
      <th style='text-align:center;{_br}'>Pred<br>Under%</th>
      <th style='text-align:center'>Over<br>Edge</th>
      <th style='text-align:center;{_br}'>Under<br>Edge</th>
      <th style='text-align:center'>Career<br>Outs</th>
      <th style='text-align:center'>c5<br>Outs</th>
      <th style='text-align:center'>Career<br>Ks</th>
      <th style='text-align:center;{_br}'>Opp<br>K/G</th>
      <th style='text-align:center'>Status</th>
    </tr>
    {rows_html}
  </table>
</details>"""


def _build_section2(settled: pd.DataFrame, yesterday: str) -> str:
    """Yesterday's results — per-bet table (wins first) + game breakdown."""
    he = html_module.escape

    if settled.empty:
        return f"""
<hr style='border:none;border-top:2px solid #e0e0e0;margin:28px 0'>
<h3 style='color:#2c3e50;margin-bottom:6px'>Yesterday's Results — {yesterday}</h3>
<p style='color:#888;font-size:13px'>No bets placed yesterday.</p>"""

    w = int((settled["pnl"] > 0).sum())
    l = int((settled["pnl"] < 0).sum())
    p = int((settled["pnl"] == 0).sum())
    u = float(settled["pnl"].sum())
    roi = u / max(1, w + l)
    result_color = "#276221" if u >= 0 else "#c0392b"

    sorted_bets = pd.concat([settled[settled["pnl"] > 0], settled[settled["pnl"] <= 0]])
    bet_rows = ""
    for _, r in sorted_bets.iterrows():
        pnl = r["pnl"]
        win = pnl > 0
        bg  = "background:#eaf6ea" if win else ("background:#fdecea" if pnl < 0 else "")
        oc  = "#276221" if win else ("#c0392b" if pnl < 0 else "#888")
        out_str = "WIN" if win else ("LOSS" if pnl < 0 else "PUSH")
        name = r.get("player_name", r.get("player_key", ""))
        bet_rows += (
            f"<tr style='{bg}'>"
            f"<td>{he(str(name))}</td>"
            f"<td style='text-align:center'>{he(display_book(str(r.get('bookmaker',''))))}</td>"
            f"<td style='text-align:center'>{fmt(r.get('line'), '.1f')}</td>"
            f"<td style='text-align:center;font-weight:bold;color:#1d4ed8'>{to_american(r.get('under_price'))}</td>"
            f"<td style='text-align:center'>{fmt(r.get('edge_under'), '+.1%')}</td>"
            f"<td style='text-align:center;font-weight:bold'>{fmt(r.get('outs_actual'), '.0f')}</td>"
            f"<td style='text-align:center;color:{oc};font-weight:bold'>{out_str}</td>"
            f"<td style='text-align:center;font-weight:bold;color:{oc}'>{pnl:+.2f}u</td>"
            f"</tr>\n"
        )

    # Game breakdown
    game_rows = ""
    for (away, home), g in settled.groupby(["away_team", "home_team"]):
        gw = int((g["pnl"] > 0).sum()); gl = int((g["pnl"] < 0).sum()); gu = float(g["pnl"].sum())
        gc = "#276221" if gu >= 0 else "#c0392b"
        game_rows += (
            f"<tr><td>{he(away)} @ {he(home)}</td>"
            f"<td style='text-align:center'>{len(g)}</td>"
            f"<td style='text-align:center;color:#276221;font-weight:bold'>{gw}</td>"
            f"<td style='text-align:center;color:#c0392b;font-weight:bold'>{gl}</td>"
            f"<td style='text-align:center;font-weight:bold;color:{gc}'>{gu:+.2f}u</td></tr>\n"
        )
    tc = "#276221" if u >= 0 else "#c0392b"
    game_rows += (
        f"<tr style='background:#f5f5f5;font-weight:bold'><td>Total</td>"
        f"<td style='text-align:center'>{w+l+p}</td>"
        f"<td style='text-align:center;color:#276221'>{w}</td>"
        f"<td style='text-align:center;color:#c0392b'>{l}</td>"
        f"<td style='text-align:center;color:{tc}'>{u:+.2f}u</td></tr>\n"
    )

    push_str = f" – {p}P" if p else ""
    return f"""
<hr style='border:none;border-top:2px solid #e0e0e0;margin:28px 0'>
<h3 style='color:#2c3e50;margin-bottom:6px'>Yesterday's Results — {yesterday}</h3>
<p style='font-size:13px;margin-top:0'>
  <strong style='color:{result_color}'>{w}W / {l}L{push_str}
  &nbsp;·&nbsp; {u:+.2f}u &nbsp;·&nbsp; {roi*100:+.1f}% ROI</strong>
</p>
<table style='width:auto;min-width:600px'>
  <tr>
    <th>Pitcher</th><th>Book</th><th>Line</th><th>Under Odds</th>
    <th>Edge</th><th>Actual Outs</th><th>Outcome</th><th>P&amp;L</th>
  </tr>
  {bet_rows}
</table>
<p style='font-size:12px;font-weight:600;color:#555;margin:20px 0 4px'>By game</p>
<table style='width:auto'>
  <tr><th>Game</th><th>Bets</th><th>W</th><th>L</th><th>Net</th></tr>
  {game_rows}
</table>"""


def _build_section3(all_time: pd.DataFrame) -> str:
    """All-time results — stat cards + by-season table."""
    if all_time.empty or "pnl" not in all_time.columns:
        return """
<hr style='border:none;border-top:2px solid #e0e0e0;margin:28px 0'>
<h3 style='color:#2c3e50'>All-Time Results</h3>
<p style='color:#888;font-size:13px'>No settled history yet.</p>"""

    plays = all_time[all_time["pnl"].notna()].copy()
    at_w = int((plays["pnl"] > 0).sum())
    at_l = int((plays["pnl"] < 0).sum())
    at_u = float(plays["pnl"].sum())
    at_n = len(plays)

    cards_html = _cards_row(
        _card("All-Time PNL", f"{at_u:+.2f}u", green=at_u >= 0),
        _card("All-Time Record", f"{at_w}W – {at_l}L"),
        _card("Win %", f"{at_w/max(1,at_w+at_l)*100:.1f}%", green=at_w > at_l),
        _card("ROI", f"{at_u/max(1,at_n)*100:+.1f}%", green=at_u >= 0),
    )

    season_rows = ""
    plays["_yr"] = plays["game_date"].astype(str).str[:4]
    for yr, g in plays.groupby("_yr"):
        sw = int((g["pnl"] > 0).sum()); sl = int((g["pnl"] < 0).sum()); su = float(g["pnl"].sum())
        sc = "#276221" if su >= 0 else "#c0392b"
        season_rows += (
            f"<tr><td>{yr}</td><td style='text-align:center'>{len(g)}</td>"
            f"<td style='text-align:center'>{sw}W – {sl}L</td>"
            f"<td style='text-align:center'>{sw/max(1,sw+sl)*100:.1f}%</td>"
            f"<td style='text-align:center;font-weight:bold;color:{sc}'>{su:+.2f}u</td>"
            f"<td style='text-align:center;color:{sc}'>{su/max(1,len(g))*100:+.1f}%</td></tr>\n"
        )

    return f"""
<hr style='border:none;border-top:2px solid #e0e0e0;margin:28px 0'>
<h3 style='color:#2c3e50;margin-bottom:6px'>All-Time Results</h3>
{cards_html}
<table style='width:auto;min-width:500px'>
  <tr><th>Season</th><th>Bets</th><th>Record</th><th>Win %</th><th>Units</th><th>ROI</th></tr>
  {season_rows}
</table>"""


def build_html_email(
    scored: pd.DataFrame,
    gameday: str,
    yesterday: str,
    settled: pd.DataFrame,
    all_time: pd.DataFrame,
    edge_play: float,
    edge_show: float,
    season_stats: dict | None,
) -> str:
    n_play = int((scored["edge_under"] >= edge_play).sum()) if not scored.empty else 0
    n_show = int(((scored["edge_under"] >= edge_show) & (scored["edge_under"] < edge_play)).sum()) if not scored.empty else 0

    if scored.empty:
        s1 = f"<h2 style='color:#2c3e50'>MLB Pitcher Outs UNDER — {gameday}</h2><p style='color:#888'>No pitcher outs props found today.</p>"
    else:
        s1 = _build_section1(scored, gameday, edge_play, edge_show, season_stats)

    s2 = _build_section2(settled, yesterday)
    s3 = _build_section3(all_time)

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<style>
  body {{font-family:{_SANS};color:#222;max-width:1700px;margin:auto;padding:20px}}
  table {{border-collapse:collapse;width:100%;margin-top:8px}}
  th {{background:#2c3e50;color:#fff;padding:7px 8px;text-align:left;font-size:12px;white-space:nowrap}}
  td {{padding:5px 8px;border-bottom:1px solid #e0e0e0;font-size:12px}}
  details {{margin-top:16px;border:1px solid #ddd;border-radius:6px;padding:0 12px 8px}}
  summary {{font-weight:600;font-size:14px;cursor:pointer;padding:10px 0;color:#2c3e50;user-select:none}}
  .footer {{background:#ecf0f1;border-radius:6px;padding:10px 16px;margin-top:28px;font-size:12px;color:#555}}
  .lp {{display:inline-block;width:12px;height:12px;background:#eaf6ea;border:1px solid #276221;margin-right:4px;vertical-align:middle}}
  .ls {{display:inline-block;width:12px;height:12px;background:#fffde7;border:1px solid #b8860b;margin-right:4px;vertical-align:middle}}
</style>
</head><body>
{s1}
{s2}
{s3}
<div class='footer'>
  Flat 1u per book bet &nbsp;·&nbsp; consensus_line bootstrap (10k samples) · shrinkage=0.25 &nbsp;·&nbsp;
  Strategy (UNDER · minus odds · edge≥5pp raw · line≤17.5): OOS +95.3u · +13.83% ROI · n=689 (2025+2026)
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
            "Body":    {"Html": {"Data": html_body, "Charset": "UTF-8"}},
        },
    )
    print(f"  Email sent to {to_list}")


def publish_sns(subject: str, message: str) -> None:
    if not SNS_TOPIC_ARN:
        return
    boto3.client("sns").publish(TopicArn=SNS_TOPIC_ARN, Subject=subject[:100], Message=message)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=today_et())
    args = parser.parse_args()
    gameday   = args.gameday
    yesterday = (datetime.strptime(gameday, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"MLB Pitcher Outs pipeline | gameday={gameday} | yesterday={yesterday}")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    inf = cfg["mlb_pitcher_outs_model"]["inference"]

    EDGE_PLAY  = inf["edge_threshold_under"]   # 0.10
    EDGE_SHOW  = inf["edge_threshold_show"]    # 0.05
    SHRINKAGE  = inf["shrinkage"]              # 0.25
    MIN_BOOKS  = inf["min_books"]              # 2
    LINE_MIN   = inf["line_min"]               # 12.5
    LINE_MAX   = inf["line_max"]               # 17.5
    REGIONS    = cfg["mlb_pitcher_outs_model"]["market"]["regions"]

    print(f"  Strategy: UNDER minus_odds edge≥{EDGE_PLAY*100:.0f}pp shrink={SHRINKAGE} line≤{LINE_MAX}")

    # ── Step 1: settle yesterday ──────────────────────────────────────────────
    print(f"\n[1/3] Settling {yesterday} ...")
    settled, all_time = settle_yesterday(yesterday)

    # ── Step 2: score today ───────────────────────────────────────────────────
    print(f"\n[2/3] Scoring {gameday} ...")
    print("  Loading spine ...")
    spine = s3_get_parquet(SPINE_KEY)
    latest_feats = get_latest_features(spine)
    print(f"  Spine: {len(spine):,} rows | {len(latest_feats):,} pitchers")

    print("  Loading residuals ...")
    residuals = np.load(BytesIO(_s3().get_object(Bucket=S3_BUCKET, Key=RESID_KEY)["Body"].read()))
    print(f"  Residuals: n={len(residuals):,}")

    print("  Fetching live events ...")
    events = fetch_live_events(gameday)
    if not events:
        print("  No events — sending email with no plays today")
        scored = pd.DataFrame()
    else:
        all_rows: list[pd.DataFrame] = []
        for ev in events:
            props = fetch_event_pitcher_outs(ev["id"], REGIONS)
            mdf   = build_market_df(
                props, ev["id"], ev.get("home_team", ""), ev.get("away_team", ""),
                gameday, commence_to_et(ev.get("commence_time", "")),
                LINE_MIN, LINE_MAX,
            )
            if not mdf.empty:
                all_rows.append(mdf)
                print(f"  {ev.get('away_team','?')[:15]:15} @ {ev.get('home_team','?')[:15]:15}  "
                      f"{mdf['player_key'].nunique()} pitchers  {len(mdf)} rows")
            else:
                print(f"  {ev.get('away_team','?')[:15]:15} @ {ev.get('home_team','?')[:15]:15}  no props")

        if all_rows:
            market_df = pd.concat(all_rows, ignore_index=True)
            # Apply minus-odds filter
            market_df = market_df[market_df["under_price"] <= 2.0].copy()
            scored = score_slate(market_df, latest_feats, residuals, SHRINKAGE, MIN_BOOKS)
            print(f"  Scored: {len(scored):,} rows")
        else:
            scored = pd.DataFrame()

    if not scored.empty:
        s3_put_csv(f"{DAILY_PREFIX}/{gameday}/scored.csv", scored)
        plays = scored[scored["edge_under"] >= EDGE_PLAY]
        if not plays.empty:
            s3_put_csv(f"{DAILY_PREFIX}/{gameday}/recommendations.csv", plays)
            print(f"  Saved {len(plays)} recommendations to S3")

    # ── Step 3: season stats & email ──────────────────────────────────────────
    print(f"\n[3/3] Building email ...")
    season_stats = None
    if not all_time.empty and "pnl" in all_time.columns:
        year = gameday[:4]
        szn = all_time[
            (all_time["game_date"].astype(str).str[:4] == year)
            & all_time["pnl"].notna()
        ]
        if not szn.empty:
            season_stats = {
                "units":  float(szn["pnl"].sum()),
                "wins":   int((szn["pnl"] > 0).sum()),
                "losses": int((szn["pnl"] < 0).sum()),
            }

    n_play = int((scored["edge_under"] >= EDGE_PLAY).sum()) if not scored.empty else 0
    n_show = int(((scored["edge_under"] >= EDGE_SHOW) & (scored["edge_under"] < EDGE_PLAY)).sum()) if not scored.empty else 0
    yest_w = int((settled["pnl"] > 0).sum()) if not settled.empty else 0
    yest_l = int((settled["pnl"] < 0).sum()) if not settled.empty else 0
    yest_u = float(settled["pnl"].sum()) if not settled.empty else 0.0

    subject = (
        f"MLB Pitcher Outs — {n_play}p {n_play}b plays · "
        f"yest {yest_w}W/{yest_l}L {yest_u:+.2f}u — {gameday}"
    )
    html_body = build_html_email(
        scored, gameday, yesterday, settled, all_time,
        EDGE_PLAY, EDGE_SHOW, season_stats,
    )
    send_ses(subject, html_body)
    if n_play > 0:
        publish_sns(subject, f"{n_play} UNDER pitcher outs plays for {gameday}. Check email.")

    print(f"\nDone. Plays={n_play} Shows={n_show} | Yest={yest_w}W/{yest_l}L {yest_u:+.2f}u")


if __name__ == "__main__":
    main()
