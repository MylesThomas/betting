"""
Live gameday pipeline for MLB game totals (over/under).

Strategy: line=9.5, UNDER, edge > 0
  edge = p_model_under - raw_prob_under  (raw vig-inclusive price, per skill spec)
Model: Ridge(alpha=50) regression on 12 game-level features → Method C per-line calibration
  Re-trained daily from the historical spine on S3.

For each game with a 9.5-line on the given gameday:
  1. Fetch live game totals from Odds API (totals market)
  2. Compute consensus_line and other market features from live odds
  3. Look up team rolling stats (home/away) from historical spine on S3
  4. Build feature vector for today's game
  5. Train Ridge regression on all historical games (re-trained daily)
  6. Apply Method C calibration to get P(under | line=9.5)
  7. Compute edge = p_model_under - raw_prob_under (per book)
  8. Qualify: line=9.5 and edge > 0
  9. Send SES HTML email and write picks CSV to S3

S3 paths read:
  mlb/game_totals_model/spine/mlb_game_totals_spine.parquet

S3 paths written:
  mlb/game_totals_model/daily_runs/{gameday}/picks.csv

Usage:
  python src/mlb_game_totals_modeling/scripts/run_pipeline.py
  python src/mlb_game_totals_modeling/scripts/run_pipeline.py --gameday 2026-06-24
"""
from __future__ import annotations

import argparse
import html as html_module
import os
import sys
import time
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "baseball_mlb"
REGIONS       = "us"
SLEEP_S       = 0.25

S3_BUCKET    = "the-odds-api-mt"
SPINE_KEY    = "mlb/game_totals_model/spine/mlb_game_totals_spine.parquet"
DAILY_PREFIX = "mlb/game_totals_model/daily_runs"

SES_SOURCE    = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

FEATURE_COLS = [
    "consensus_line",
    "park_factor",
    "combined_ra_L10",
    "home_ra_L20",
    "combined_ra_L5",
    "combined_rs_L10",
    "home_rs_L10",
    "away_rs_L3",
    "combined_ra_career",
    "away_ra_L5",
    "home_ra_L10",
    "away_ra_L10",
]

MIN_GAMES_CAL = 30
TARGET_LINE   = 9.5
MAX_LINE      = 13.0

# Strategy A (benchmark): bet ALL 9.5 unders, no edge filter
HIGH_CONVICTION   = 0.01   # ◄◄ marker in benchmark/both sections
POSITIVE_EDGE_MIN = 0.00   # ◄  marker

# Strategy B (model_both): Ridge model picks side at [8.5,9.5], edge≥2pp, shrink=0.50
MODEL_LINES    = [8.5, 9.5]
MODEL_EDGE_MIN = 0.02
SHRINKAGE      = 0.50

PARK_FACTORS: dict[str, float] = {
    "Arizona Diamondbacks": 1.05,
    "Atlanta Braves":       0.99,
    "Baltimore Orioles":    1.02,
    "Boston Red Sox":       1.06,
    "Chicago Cubs":         0.97,
    "Chicago White Sox":    0.97,
    "Cincinnati Reds":      1.05,
    "Cleveland Guardians":  0.96,
    "Colorado Rockies":     1.39,
    "Detroit Tigers":       0.97,
    "Houston Astros":       0.96,
    "Kansas City Royals":   0.98,
    "Los Angeles Angels":   0.98,
    "Los Angeles Dodgers":  0.97,
    "Miami Marlins":        0.93,
    "Milwaukee Brewers":    0.96,
    "Minnesota Twins":      1.01,
    "New York Mets":        0.98,
    "New York Yankees":     1.00,
    "Oakland Athletics":    0.96,
    "Philadelphia Phillies": 1.03,
    "Pittsburgh Pirates":   0.99,
    "San Diego Padres":     0.94,
    "San Francisco Giants": 0.92,
    "Seattle Mariners":     0.93,
    "St. Louis Cardinals":  0.97,
    "Tampa Bay Rays":       0.97,
    "Texas Rangers":        1.01,
    "Toronto Blue Jays":    1.00,
    "Washington Nationals": 1.01,
}

TEAM_NORMALIZE = {"Athletics": "Oakland Athletics"}

BOOK_DISPLAY = {
    "draftkings":      "DraftKings",
    "fanduel":         "FanDuel",
    "betmgm":          "BetMGM",
    "betrivers":       "BetRivers",
    "caesars":         "Caesars",
    "betonlineag":     "BetOnline",
    "bovada":          "Bovada",
    "mybookieag":      "MyBookie",
    "betus":           "BetUS",
    "lowvig":          "LowVig",
    "windcreek":       "Wind Creek",
    "williamhill_us":  "William Hill",
    "superbook":       "SuperBook",
    "pointsbetus":     "PointsBet",
    "unibet_us":       "Unibet",
    "fanatics":        "Fanatics",
    "pinnacle":        "Pinnacle",
    "betparx":         "BetParx",
    "espnbet":         "ESPN Bet",
    "hardrockbet":     "Hard Rock Bet",
    "ballybet":        "Bally Bet",
    "betanysports":    "BetAnySports",
}

_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"


def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def full_book_name(key: str) -> str:
    return BOOK_DISPLAY.get(str(key).lower(), str(key))


def normalize_team(name: str) -> str:
    return TEAM_NORMALIZE.get(name, name)


def commence_to_et(commence_time: str) -> str:
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        s = dt.astimezone(ET).strftime("%I:%M %p")
        return s.lstrip("0")
    except Exception:
        return ""


def format_american(american: float) -> str:
    try:
        v = int(american)
        return f"+{v}" if v > 0 else str(v)
    except Exception:
        return "—"


# ── S3 ────────────────────────────────────────────────────────────────────────

def _s3():
    return boto3.client("s3")


def s3_get_parquet(key: str) -> pd.DataFrame:
    body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def s3_put_csv(key: str, df: pd.DataFrame) -> None:
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=df.to_csv(index=False).encode())


# ── Historical spine ──────────────────────────────────────────────────────────

def load_spine(s3: bool = True) -> pd.DataFrame:
    if s3:
        spine = s3_get_parquet(SPINE_KEY)
    else:
        local = Path.home() / "Downloads/tmp/mlb_game_totals/game_totals_spine.parquet"
        spine = pd.read_parquet(local)

    spine["game_date"] = pd.to_datetime(spine["game_date"])
    spine["combined_ra_L5"]     = spine["home_ra_L5"]     + spine["away_ra_L5"]
    spine["combined_ra_L10"]    = spine["home_ra_L10"]    + spine["away_ra_L10"]
    spine["combined_ra_career"] = spine["home_ra_career"] + spine["away_ra_career"]
    spine["combined_rs_L10"]    = spine["home_rs_L10"]    + spine["away_rs_L10"]

    # Normalize team names
    spine["home_team"] = spine["home_team"].map(normalize_team)
    spine["away_team"] = spine["away_team"].map(normalize_team)

    return spine


def get_team_latest_features(spine: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, dict]:
    """
    For each team, extract their most recent rolling stats (as of target_date).
    Returns dict: team_name → {rs_career, ra_career, rs_L1, ra_L1, ..., rs_L20, ra_L20, rs_season, ra_season}
    The rolling stat columns use a generic prefix (no home_/away_) for lookup.
    """
    historical = spine[spine["game_date"] < target_date].copy()
    games = historical.drop_duplicates("game_pk").sort_values("game_date")

    # Build team-level view: one row per (team, game) with generic rolling stat names
    home_view = games[["game_pk", "game_date", "home_team"]].copy()
    home_view = home_view.rename(columns={"home_team": "team"})
    for col in games.columns:
        if col.startswith("home_rs_") or col.startswith("home_ra_"):
            generic = col.replace("home_rs_", "rs_").replace("home_ra_", "ra_")
            home_view[generic] = games[col].values

    away_view = games[["game_pk", "game_date", "away_team"]].copy()
    away_view = away_view.rename(columns={"away_team": "team"})
    for col in games.columns:
        if col.startswith("away_rs_") or col.startswith("away_ra_"):
            generic = col.replace("away_rs_", "rs_").replace("away_ra_", "ra_")
            away_view[generic] = games[col].values

    team_log = pd.concat([home_view, away_view], ignore_index=True).sort_values(
        ["team", "game_date"], ascending=[True, False]
    )

    stat_cols = [c for c in team_log.columns if c.startswith("rs_") or c.startswith("ra_")]
    latest = team_log.drop_duplicates("team", keep="first").set_index("team")[stat_cols]
    return latest.to_dict("index")


def get_game_level_historical(spine: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """Deduplicate to game level for model training. Excludes target_date."""
    historical = spine[spine["game_date"] < target_date].copy()
    cols = ["game_pk", "game_date", "season", "total_runs", "hit_over"] + FEATURE_COLS
    cols = [c for c in dict.fromkeys(cols) if c in historical.columns]
    return (
        historical[cols]
        .drop_duplicates("game_pk")
        .dropna(subset=["total_runs"] + [c for c in FEATURE_COLS if c in historical.columns])
    )


# ── Odds API ──────────────────────────────────────────────────────────────────

def fetch_live_events(gameday: str) -> list[dict]:
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY not set")
    r = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT}/events",
        params={"apiKey": ODDS_API_KEY},
        timeout=30,
    )
    if r.status_code != 200:
        print(f"  Events API error: {r.status_code} {r.text[:200]}")
        return []
    events = r.json()
    today_events = [e for e in events if e.get("commence_time", "")[:10] == gameday]
    print(f"  Live events for {gameday}: {len(today_events)}")
    return today_events


def fetch_event_totals(event_id: str) -> list[dict]:
    r = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds",
        params={"apiKey": ODDS_API_KEY, "markets": "totals", "regions": REGIONS, "oddsFormat": "american"},
        timeout=30,
    )
    time.sleep(SLEEP_S)
    if r.status_code != 200:
        return []
    rows = []
    for bm in r.json().get("bookmakers", []):
        book = bm["key"]
        for mkt in bm.get("markets", []):
            if mkt["key"] != "totals":
                continue
            over_outcomes  = [o for o in mkt.get("outcomes", []) if o["name"] == "Over"]
            under_outcomes = [o for o in mkt.get("outcomes", []) if o["name"] == "Under"]
            for o in over_outcomes:
                pt    = o.get("point")
                under = next((u for u in under_outcomes if u.get("point") == pt), None)
                if under is None:
                    continue
                rows.append({
                    "bookmaker":   book,
                    "line":        float(pt),
                    "over_price":  float(o.get("price")),
                    "under_price": float(under.get("price")),
                })
    return rows


def parse_event_lines(rows: list[dict], event_id: str, home_team: str, away_team: str,
                      gameday: str, commence_time: str = "") -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[(df["over_price"] > -10000) & (df["under_price"] > -10000)].copy()
    df = df[df["line"] <= MAX_LINE].copy()
    if df.empty:
        return pd.DataFrame()

    def _dec(american: float) -> float:
        if american >= 0:
            return 1 + american / 100
        return 1 + 100 / abs(american)

    df["dec_over"]       = df["over_price"].map(_dec)
    df["dec_under"]      = df["under_price"].map(_dec)
    df["raw_prob_over"]  = 1.0 / df["dec_over"]
    df["raw_prob_under"] = 1.0 / df["dec_under"]
    df["total_prob"]     = df["raw_prob_over"] + df["raw_prob_under"]
    df["novig_prob_over"]  = df["raw_prob_over"]  / df["total_prob"]
    df["novig_prob_under"] = df["raw_prob_under"] / df["total_prob"]

    df["event_id"]   = event_id
    df["home_team"]  = normalize_team(home_team)
    df["away_team"]  = normalize_team(away_team)
    df["game_date"]  = gameday
    df["game_time_et"] = commence_to_et(commence_time)

    return df


def compute_consensus_features(event_lines: pd.DataFrame) -> dict:
    """Compute game-level consensus features from all books at all lines."""
    return {
        "consensus_line": float(event_lines["line"].mean()),
        "min_line":       float(event_lines["line"].min()),
        "max_line":       float(event_lines["line"].max()),
    }


# ── Model (train from historical spine) ──────────────────────────────────────

def train_ridge(games: pd.DataFrame) -> tuple:
    """Fit Ridge(α=50) on historical games. Returns (model, scaler)."""
    valid = games.dropna(subset=FEATURE_COLS + ["total_runs"])
    X = valid[FEATURE_COLS].values.astype(float)
    y = valid["total_runs"].values.astype(float)
    sc  = StandardScaler()
    X_s = sc.fit_transform(X)
    mdl = Ridge(alpha=50)
    mdl.fit(X_s, y)
    return mdl, sc


def train_calibration(games: pd.DataFrame, feat_col: str = "y_hat") -> dict:
    """
    Method C per-line calibration: for each line bucket with >= MIN_GAMES_CAL games,
    fit logistic(y_hat) → P(hit_over). Returns {line_val: (scaler, clf)}.
    Also returns global fallback as key None.
    """
    valid = games.dropna(subset=[feat_col, "hit_over", "consensus_line"])
    global_sc  = StandardScaler()
    X_g = global_sc.fit_transform(valid[[feat_col]].values.astype(float))
    clf_g = LogisticRegression(max_iter=500, C=0.5)
    clf_g.fit(X_g, valid["hit_over"].values.astype(int))

    calib = {None: (global_sc, clf_g)}  # global fallback

    valid["line_bucket"] = valid["consensus_line"].round(1)
    for line_val in sorted(valid["line_bucket"].unique()):
        sub = valid[valid["line_bucket"] == line_val]
        if len(sub) < MIN_GAMES_CAL:
            continue
        sc  = StandardScaler()
        X_s = sc.fit_transform(sub[[feat_col]].values.astype(float))
        clf = LogisticRegression(max_iter=500, C=0.5)
        clf.fit(X_s, sub["hit_over"].values.astype(int))
        calib[line_val] = (sc, clf)

    return calib


def predict_p_model(y_hat: float, line: float, calib: dict) -> float:
    """Apply Method C calibration: y_hat → P(over | line)."""
    bucket = round(float(line), 1)
    if bucket in calib:
        sc, clf = calib[bucket]
    else:
        sc, clf = calib[None]
    X = sc.transform(np.array([[y_hat]]))
    return float(clf.predict_proba(X)[0, 1])


# ── Score today's games ───────────────────────────────────────────────────────

def build_game_features(
    event_lines: pd.DataFrame,
    team_feats: dict[str, dict],
    consensus: dict,
    home_team: str,
    away_team: str,
) -> dict | None:
    """Build feature vector for a single game. Returns None if features are missing."""
    home_stats = team_feats.get(home_team, {})
    away_stats = team_feats.get(away_team, {})

    if not home_stats or not away_stats:
        missing = []
        if not home_stats:
            missing.append(home_team)
        if not away_stats:
            missing.append(away_team)
        print(f"    Missing team features for: {missing}")
        return None

    park_factor = PARK_FACTORS.get(home_team, 1.00)

    features = {
        "consensus_line":    consensus["consensus_line"],
        "park_factor":       park_factor,
        "combined_ra_L10":   home_stats.get("ra_L10", np.nan) + away_stats.get("ra_L10", np.nan)
                             if pd.notna(home_stats.get("ra_L10")) and pd.notna(away_stats.get("ra_L10"))
                             else np.nan,
        "home_ra_L20":       home_stats.get("ra_L20", np.nan),
        "combined_ra_L5":    home_stats.get("ra_L5",  np.nan) + away_stats.get("ra_L5",  np.nan)
                             if pd.notna(home_stats.get("ra_L5")) and pd.notna(away_stats.get("ra_L5"))
                             else np.nan,
        "combined_rs_L10":   home_stats.get("rs_L10", np.nan) + away_stats.get("rs_L10", np.nan)
                             if pd.notna(home_stats.get("rs_L10")) and pd.notna(away_stats.get("rs_L10"))
                             else np.nan,
        "home_rs_L10":       home_stats.get("rs_L10", np.nan),
        "away_rs_L3":        away_stats.get("rs_L3",  np.nan),
        "combined_ra_career":home_stats.get("ra_career", np.nan) + away_stats.get("ra_career", np.nan)
                             if pd.notna(home_stats.get("ra_career")) and pd.notna(away_stats.get("ra_career"))
                             else np.nan,
        "away_ra_L5":        away_stats.get("ra_L5",  np.nan),
        "home_ra_L10":       home_stats.get("ra_L10", np.nan),
        "away_ra_L10":       away_stats.get("ra_L10", np.nan),
    }
    return features


def score_game(
    event_lines: pd.DataFrame,
    game_feats: dict,
    ridge_model,
    scaler,
    calib: dict,
    home_team: str,
    away_team: str,
    game_time_et: str,
    gameday: str,
) -> pd.DataFrame:
    """Score all book rows for a game. Returns DataFrame of scored rows."""
    X_vals = [game_feats[f] for f in FEATURE_COLS]
    if any(pd.isna(v) for v in X_vals):
        missing = [FEATURE_COLS[i] for i, v in enumerate(X_vals) if pd.isna(v)]
        print(f"    Missing features for {home_team}: {missing}")
        return pd.DataFrame()

    X    = np.array(X_vals, dtype=float).reshape(1, -1)
    X_s  = scaler.transform(X)
    y_hat = float(ridge_model.predict(X_s)[0])

    rows = []
    for _, r in event_lines.iterrows():
        line = float(r["line"])
        p_over = predict_p_model(y_hat, line, calib)
        p_under = 1.0 - p_over
        edge_under = p_under - float(r["raw_prob_under"])
        edge_over  = p_over  - float(r["raw_prob_over"])

        rows.append({
            "game_date":       gameday,
            "game_time_et":    game_time_et,
            "home_team":       home_team,
            "away_team":       away_team,
            "bookmaker":       r["bookmaker"],
            "line":            line,
            "over_price":      int(r["over_price"]),
            "under_price":     int(r["under_price"]),
            "raw_prob_over":   round(float(r["raw_prob_over"]),  4),
            "raw_prob_under":  round(float(r["raw_prob_under"]), 4),
            "novig_prob_over": round(float(r["novig_prob_over"]),  4),
            "novig_prob_under":round(float(r["novig_prob_under"]), 4),
            "y_hat":           round(y_hat, 4),
            "p_model_over":    round(p_over,   4),
            "p_model_under":   round(p_under,  4),
            "edge_under":      round(edge_under, 4),
            "edge_over":       round(edge_over,  4),
            **{f: round(game_feats[f], 4) if not pd.isna(game_feats[f]) else np.nan
               for f in FEATURE_COLS},
        })

    return pd.DataFrame(rows)


# ── Email ─────────────────────────────────────────────────────────────────────

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


def build_html_email(
    bets: pd.DataFrame,
    all_scored: pd.DataFrame,
    gameday: str,
) -> str:
    he = html_module.escape

    bets_both      = bets[bets["strategy_tag"] == "both"]
    bets_model     = bets[bets["strategy_tag"] == "model"]
    bets_benchmark = bets[bets["strategy_tag"] == "benchmark"]
    n_both      = len(bets_both)
    n_model     = len(bets_model)
    n_benchmark = len(bets_benchmark)

    def fmt(v, fmt_str: str) -> str:
        try:
            return format(float(v), fmt_str)
        except (TypeError, ValueError):
            return "—"

    _ADJ_TH = "background:#3d5a38"
    _ADJ_TH_FIRST = f"{_ADJ_TH};border-left:2px solid #7fff9a"
    _TABLE_HEADER = (
        "<tr>"
        "<th rowspan='2'>Book</th>"
        "<th rowspan='2' style='text-align:center'>Line</th>"
        "<th rowspan='2' style='text-align:center'>Over</th>"
        "<th rowspan='2' style='text-align:center'>Under</th>"
        "<th rowspan='2' style='text-align:center'>Side</th>"
        "<th rowspan='2' style='text-align:center'>Raw P(O)</th>"
        "<th rowspan='2' style='text-align:center'>Raw P(U)</th>"
        "<th rowspan='2' style='text-align:center'>Fair P(O)</th>"
        "<th rowspan='2' style='text-align:center'>Fair P(U)</th>"
        "<th rowspan='2' style='text-align:center'>Model P(O)</th>"
        "<th rowspan='2' style='text-align:center'>Model P(U)</th>"
        "<th rowspan='2' style='text-align:center'>Edge</th>"
        f"<th colspan='3' style='text-align:center;{_ADJ_TH_FIRST}'>Adjusted / Shrinkage Projections</th>"
        "<th rowspan='2' style='text-align:center'>Cons Line</th>"
        "<th rowspan='2' style='text-align:center'>RA L10</th>"
        "</tr>\n"
        "<tr>"
        f"<th style='text-align:center;{_ADJ_TH_FIRST}'>Adj P(O)</th>"
        f"<th style='text-align:center;{_ADJ_TH}'>Adj P(U)</th>"
        f"<th style='text-align:center;{_ADJ_TH}'>Adj Edge</th>"
        "</tr>\n"
    )

    def game_rows(df: pd.DataFrame) -> str:
        if df.empty:
            return "<tr><td colspan='17' style='color:#888;text-align:center;padding:12px'>No bets today</td></tr>\n"
        html = ""
        df = df.copy()
        df["_ts"] = df["game_time_et"].apply(_time_sort_key)
        df = df.sort_values(["_ts", "line"])

        for (home, away), grp in df.groupby(["home_team", "away_team"], sort=False):
            gtime = grp["game_time_et"].iloc[0] if "game_time_et" in grp else ""
            n_g   = len(grp)
            row0  = grp.iloc[0]
            label = f"<span style='color:#7fff9a;font-weight:bold'>{n_g} BET{'S' if n_g!=1 else ''}</span>"

            html += (
                f"<tr style='background:#1e2a35'>"
                f"<td colspan='17' style='padding:7px 10px;font-weight:600;font-size:12px;"
                f"color:#aab8c2;letter-spacing:.3px'>"
                f"{he(gtime)} ET &nbsp;·&nbsp; {he(away)} @ {he(home)} &nbsp;·&nbsp; {label}"
                f"&nbsp;·&nbsp; y_hat={fmt(row0['y_hat'],'.2f')}"
                f"</td></tr>\n"
            )

            for _, r in grp.sort_values(["line", "display_edge"], ascending=[True, False]).iterrows():
                side      = str(r.get("side", "under"))
                edge_val  = float(r.get("display_edge", 0))
                o_bold    = "font-weight:bold;color:#1d4ed8" if side == "over"  else ""
                u_bold    = "font-weight:bold;color:#1d4ed8" if side == "under" else ""

                if edge_val >= HIGH_CONVICTION:
                    marker = "◄◄"
                elif edge_val >= POSITIVE_EDGE_MIN:
                    marker = "◄"
                else:
                    marker = ""

                adj_edge_val = float(r.get("edge_eff_u", 0))
                adj_edge_color = "#276221" if adj_edge_val >= 0 else "#b91c1c"
                html += (
                    f"<tr style='background:#eaf6ea'>"
                    f"<td>{he(full_book_name(str(r['bookmaker'])))}</td>"
                    f"<td style='text-align:center'>{fmt(r['line'],'.1f')}</td>"
                    f"<td style='text-align:center;{o_bold}'>{format_american(r['over_price'])}</td>"
                    f"<td style='text-align:center;{u_bold}'>{format_american(r['under_price'])}</td>"
                    f"<td style='text-align:center;font-weight:bold'>{he(side.upper())}</td>"
                    f"<td style='text-align:center'>{fmt(r['raw_prob_over'],'.1%')}</td>"
                    f"<td style='text-align:center'>{fmt(r['raw_prob_under'],'.1%')}</td>"
                    f"<td style='text-align:center'>{fmt(r['novig_prob_over'],'.1%')}</td>"
                    f"<td style='text-align:center'>{fmt(r['novig_prob_under'],'.1%')}</td>"
                    f"<td style='text-align:center'>{fmt(r['p_model_over'],'.1%')}</td>"
                    f"<td style='text-align:center'>{fmt(r['p_model_under'],'.1%')}</td>"
                    f"<td style='text-align:center;color:#276221;font-weight:bold'>{fmt(edge_val,'+.1%')} {marker}</td>"
                    f"<td style='text-align:center;border-left:2px solid #7fff9a;color:#8aab86'>{fmt(r.get('p_eff_over'),'.1%')}</td>"
                    f"<td style='text-align:center;color:#8aab86'>{fmt(r.get('p_eff_under'),'.1%')}</td>"
                    f"<td style='text-align:center;color:{adj_edge_color};font-weight:bold'>{fmt(adj_edge_val,'+.1%')}</td>"
                    f"<td style='text-align:center;font-size:11px'>{fmt(r.get('consensus_line'),'.2f')}</td>"
                    f"<td style='text-align:center;font-size:11px'>{fmt(r.get('combined_ra_L10'),'.2f')}</td>"
                    f"</tr>\n"
                )
        return html

    rows_both      = game_rows(bets_both)
    rows_model     = game_rows(bets_model)
    rows_benchmark = game_rows(bets_benchmark)

    # All-lines summary (collapsed)
    rows_all = ""
    all_sorted = all_scored.sort_values(["home_team", "line", "edge_under"], ascending=[True, True, False])
    for _, r in all_sorted.iterrows():
        rows_all += (
            f"<tr>"
            f"<td>{he(r['away_team'])} @ {he(r['home_team'])}</td>"
            f"<td style='text-align:center'>{he(full_book_name(str(r['bookmaker'])))}</td>"
            f"<td style='text-align:center'>{fmt(r['line'],'.1f')}</td>"
            f"<td style='text-align:center'>{fmt(r['novig_prob_under'],'.1%')}</td>"
            f"<td style='text-align:center'>{fmt(r['p_model_under'],'.1%')}</td>"
            f"<td style='text-align:center;font-weight:bold'>{fmt(r['edge_under'],'+.1%')}</td>"
            f"</tr>\n"
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<style>
  body {{font-family:{_SANS};color:#222;max-width:1400px;margin:auto;padding:20px}}
  h2 {{color:#2c3e50;margin-bottom:4px}}
  table {{border-collapse:collapse;width:100%;margin-top:8px}}
  th {{background:#2c3e50;color:#fff;padding:7px 8px;text-align:left;font-size:12px;white-space:nowrap}}
  td {{padding:5px 8px;border-bottom:1px solid #e0e0e0;font-size:12px}}
  details {{margin-top:16px;border:1px solid #ddd;border-radius:6px;padding:0 12px 8px}}
  summary {{font-weight:600;font-size:14px;cursor:pointer;padding:10px 0;color:#2c3e50;user-select:none}}
  .footer {{background:#ecf0f1;border-radius:6px;padding:10px 16px;margin-top:16px;font-size:12px;color:#555}}
</style>
</head><body>
<h2>MLB Game Totals — {he(gameday)}</h2>
<p style='margin-top:4px'>
  <span style='color:#276221;font-weight:bold'>{len(bets)} total bets</span>
  &nbsp;·&nbsp; ★★ both={n_both}
  &nbsp;·&nbsp; ★ model={n_model}
  &nbsp;·&nbsp; benchmark={n_benchmark}
  &nbsp;·&nbsp; ◄◄ edge&gt;1pp &nbsp;·&nbsp; ◄ model positive
</p>

<details open>
  <summary>★★ BOTH — 9.5 Under (benchmark + model agree) — {n_both} bets</summary>
  <table>{_TABLE_HEADER}{rows_both}</table>
</details>

<details open>
  <summary>★ MODEL — Strategy B picks (8.5/9.5, edge≥2pp, shrink=0.50) — {n_model} bets</summary>
  <table>{_TABLE_HEADER}{rows_model}</table>
</details>

<details>
  <summary>BENCHMARK — 9.5 Unders (all, edge-agnostic) — {n_benchmark} bets</summary>
  <table>{_TABLE_HEADER}{rows_benchmark}</table>
</details>

<details>
  <summary>▸ All lines scored ({len(all_scored)} rows)</summary>
  <table>
    <tr>
      <th>Matchup</th><th>Book</th><th>Line</th><th>Mkt Under%</th><th>Model Under%</th><th>Edge</th>
    </tr>
    {rows_all}
  </table>
</details>

<div class='footer'>
  Flat 1u per bet · Ridge(α=50) + Method C calibration · Re-trained daily<br>
  Benchmark OOS: +196.4u, +8.45% ROI, MDD=-77.4u, net/MDD=2.54x (n=2,324) ·
  Strategy B OOS: +285.2u (n=1,623, on probation 2026) ·
  Monitor: pause benchmark if 9.5-under hit rate &lt;53% for 30 days
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
    parser.add_argument("--gameday", default=today_et())
    parser.add_argument("--local-spine", action="store_true", help="Use local spine instead of S3")
    args = parser.parse_args()
    gameday    = args.gameday
    target_dt  = pd.Timestamp(gameday)
    use_s3_spine = not args.local_spine

    print(f"MLB Game Totals pipeline | gameday={gameday}")

    # Load spine
    print("Loading historical spine...")
    spine = load_spine(s3=use_s3_spine)
    print(f"  {len(spine):,} rows  |  {spine['game_pk'].nunique():,} games  |  {spine['game_date'].max().date()} most recent")

    # Extract team latest rolling stats (as of target_date)
    print("Building team feature lookup...")
    team_feats = get_team_latest_features(spine, target_dt)
    print(f"  {len(team_feats)} teams with features")

    # Train model on all historical games before target_date
    print("Training Ridge regression + Method C calibration...")
    historical_games = get_game_level_historical(spine, target_dt)
    print(f"  Training on {len(historical_games)} historical games")

    ridge_model, scaler = train_ridge(historical_games)

    # Compute y_hat in-sample for calibration
    X_hist  = historical_games[FEATURE_COLS].values.astype(float)
    X_hist_s = scaler.transform(X_hist)
    historical_games = historical_games.copy()
    historical_games["y_hat"] = ridge_model.predict(X_hist_s)

    calib = train_calibration(historical_games)
    print(f"  Calibration fitted for {len(calib)-1} line buckets (+ global fallback)")

    # Fetch live events
    print("Fetching live Odds API events...")
    events = fetch_live_events(gameday)
    if not events:
        msg = f"No MLB events found for {gameday}"
        publish_sns(f"MLB Game Totals — no events {gameday}", msg)
        print(msg)
        return

    # Fetch totals for each event, score
    all_scored_frames: list[pd.DataFrame] = []
    for ev in events:
        event_id      = ev["id"]
        home_team     = normalize_team(ev.get("home_team", ""))
        away_team     = normalize_team(ev.get("away_team", ""))
        commence_time = ev.get("commence_time", "")

        rows = fetch_event_totals(event_id)
        if not rows:
            print(f"  {away_team[:18]:18} @ {home_team[:18]:18}  — no totals posted")
            continue

        event_lines = parse_event_lines(rows, event_id, home_team, away_team, gameday, commence_time)
        if event_lines.empty:
            continue

        consensus = compute_consensus_features(event_lines)
        game_feats = build_game_features(event_lines, team_feats, consensus, home_team, away_team)
        if game_feats is None:
            print(f"  {away_team[:18]:18} @ {home_team[:18]:18}  — missing team features, skipping")
            continue

        gtime = commence_to_et(commence_time)
        scored = score_game(event_lines, game_feats, ridge_model, scaler, calib,
                            home_team, away_team, gtime, gameday)
        if not scored.empty:
            all_scored_frames.append(scored)
            n_bets_95 = int((scored["line"] == TARGET_LINE).sum())
            edge_max  = scored.loc[scored["line"] == TARGET_LINE, "edge_under"].max() if n_bets_95 > 0 else float("nan")
            print(f"  {away_team[:18]:18} @ {home_team[:18]:18}  "
                  f"y_hat={game_feats['consensus_line']:.2f}→{ridge_model.predict(scaler.transform(np.array([[game_feats[f] for f in FEATURE_COLS]]))[0:1])[0]:.2f}  "
                  f"9.5-line rows: {n_bets_95}  max_edge: {edge_max:.1%}")

    if not all_scored_frames:
        msg = f"No games scored for {gameday}"
        publish_sns(f"MLB Game Totals — no scored games {gameday}", msg)
        print(msg)
        return

    all_scored = pd.concat(all_scored_frames, ignore_index=True)
    print(f"\nTotal scored rows: {len(all_scored)} across {all_scored[['home_team','away_team']].drop_duplicates().shape[0]} games")

    # ── Strategy qualification ────────────────────────────────────────────────
    # Strategy B: shrinkage=0.50 blend of model + market probabilities
    all_scored["p_eff_under"] = (1 - SHRINKAGE) * all_scored["p_model_under"] + SHRINKAGE * all_scored["novig_prob_under"]
    all_scored["p_eff_over"]  = (1 - SHRINKAGE) * all_scored["p_model_over"]  + SHRINKAGE * all_scored["novig_prob_over"]
    all_scored["edge_eff_u"]  = all_scored["p_eff_under"] - all_scored["raw_prob_under"]
    all_scored["edge_eff_o"]  = all_scored["p_eff_over"]  - all_scored["raw_prob_over"]
    all_scored["edge_eff"]    = all_scored[["edge_eff_u", "edge_eff_o"]].max(axis=1)
    all_scored["bet_under"]   = all_scored["edge_eff_u"] >= all_scored["edge_eff_o"]

    all_scored["is_benchmark"] = all_scored["line"] == TARGET_LINE
    all_scored["is_model"]     = (
        all_scored["line"].isin(MODEL_LINES) &
        (all_scored["edge_eff"] >= MODEL_EDGE_MIN)
    )

    def _strategy_tag(row) -> str | None:
        is_b  = bool(row["is_benchmark"])
        is_m  = bool(row["is_model"])
        bet_u = bool(row["bet_under"])
        if is_b and is_m and bet_u:
            return "both"
        if is_m:
            return "model"
        if is_b:
            return "benchmark"
        return None

    def _side(row) -> str | None:
        tag = row["strategy_tag"]
        if tag in ("benchmark", "both"):
            return "under"
        if tag == "model":
            return "under" if row["bet_under"] else "over"
        return None

    all_scored["strategy_tag"] = all_scored.apply(_strategy_tag, axis=1)
    all_scored["side"]         = all_scored.apply(_side, axis=1)
    all_scored["price"]        = all_scored.apply(
        lambda r: r["under_price"] if r["side"] == "under" else r["over_price"] if r["side"] == "over" else None,
        axis=1,
    )
    # display_edge: raw model edge for benchmark (context), effective edge for model/both (qualifier)
    all_scored["display_edge"] = all_scored.apply(
        lambda r: r["edge_under"] if r["strategy_tag"] == "benchmark" else r["edge_eff"],
        axis=1,
    )

    bets = all_scored[all_scored["strategy_tag"].notna()].copy()

    n_both      = int((bets["strategy_tag"] == "both").sum())      if not bets.empty else 0
    n_model     = int((bets["strategy_tag"] == "model").sum())     if not bets.empty else 0
    n_benchmark = int((bets["strategy_tag"] == "benchmark").sum()) if not bets.empty else 0
    n_total     = len(bets)
    n_games     = bets[["home_team", "away_team"]].drop_duplicates().shape[0] if n_total > 0 else 0

    print(f"Qualifying bets: {n_total} total  (★★both={n_both}  ★model={n_model}  benchmark={n_benchmark})")
    print(f"  Across {n_games} games")

    # Save to S3
    try:
        picks_key = f"{DAILY_PREFIX}/{gameday}/picks.csv"
        s3_put_csv(picks_key, all_scored)
        print(f"  Scored → s3://{S3_BUCKET}/{picks_key}")
        if n_total > 0:
            bets_key = f"{DAILY_PREFIX}/{gameday}/qualifying_bets.csv"
            s3_put_csv(bets_key, bets)
            print(f"  Bets   → s3://{S3_BUCKET}/{bets_key}")
    except Exception as e:
        print(f"  S3 write failed (non-fatal): {e}")

    # Build and send email
    subject = (
        f"MLB Game Totals — {n_total} bets "
        f"(★★{n_both} / ★{n_model} / {n_benchmark}) "
        f"· {n_games} game{'s' if n_games!=1 else ''} — {gameday}"
    )
    html_body = build_html_email(bets, all_scored, gameday)
    send_ses(subject, html_body)

    if n_total > 0:
        publish_sns(
            subject,
            f"{n_total} bets: ★★{n_both} (both) / ★{n_model} (model) / {n_benchmark} (benchmark). "
            f"{n_games} games."
        )
    else:
        publish_sns(subject, f"No qualifying bets on {gameday}.")


if __name__ == "__main__":
    main()
