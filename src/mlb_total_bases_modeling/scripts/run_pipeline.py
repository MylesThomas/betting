"""
Live gameday pipeline for MLB batter total-bases props.

Strategy params come from config.yaml strategy block:
  lines: [1.5]
  min_bet_edge:   0.05  → green rows in email (place these bets)
  min_track_edge: 0.03  → yellow rows in email (paper track only)
  dogs_only: true       → only bet +odds side (novig_under < 0.50)

For each batter with a total-bases prop on the given gameday:
  1. Fetch live batter_total_bases events + props from The Odds API
  2. Load rolling features from the spine (S3) — take each player's most recent row
  3. Score with v2 XGBoost regression + Method C calibration (S3)
  4. Compute no-vig market P(over) and edge_under = p_model_under - novig_prob_under (per book)
  5. Tag: tier=play (edge≥min_bet_edge) or tier=track (edge≥min_track_edge)
  6. Send SES HTML email + SNS notification
  7. Save recommendations CSV (plays + tracks) to S3

S3 paths read:
  s3://the-odds-api-mt/mlb/total_bases_model/spine/mlb_total_bases_spine.parquet
  s3://the-odds-api-mt/mlb/total_bases_model/model/mlb_tb_regression_v2.joblib

S3 paths written:
  s3://the-odds-api-mt/mlb/total_bases_model/daily_runs/{gameday}/recommendations.csv

Usage:
  python src/mlb_total_bases_modeling/scripts/run_pipeline.py
  python src/mlb_total_bases_modeling/scripts/run_pipeline.py --gameday 2026-07-04
"""
from __future__ import annotations

import argparse
import html as html_module
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import botocore.exceptions
import joblib
import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

CONFIG_PATH   = Path(__file__).resolve().parents[1] / "config.yaml"

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "baseball_mlb"
MARKETS       = "batter_total_bases,batter_total_bases_alternate"
REGIONS       = "us,us2"
SLEEP_S       = 0.25

S3_BUCKET     = "the-odds-api-mt"
SPINE_KEY     = "mlb/total_bases_model/spine/mlb_total_bases_spine.parquet"
MODEL_KEY     = "mlb/total_bases_model/model/mlb_tb_regression_v2.joblib"
DAILY_PREFIX  = "mlb/total_bases_model/daily_runs"

SES_SOURCE    = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

NOVIG_MIN        = 0.10   # filter bad market data (overridden by config strategy.novig_min)

_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"


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
        s = dt.astimezone(ET).strftime("%I:%M %p")
        return s.lstrip("0")
    except Exception:
        return ""


BOOK_DISPLAY = {
    "draftkings": "DK", "fanduel": "FD", "betmgm": "MGM",
    "betrivers": "BR", "caesars": "CZR", "pointsbetus": "PBU",
    "betparx": "PX", "espnbet": "ESPN", "fliff": "FLF",
    "hardrock": "HR", "mybookieag": "MB", "bovada": "BVD",
    "lowvig": "LV", "betus": "BTU", "betonlineag": "BOL", "pinnacle": "PIN",
}


def display_book(key: str) -> str:
    return BOOK_DISPLAY.get(str(key).lower(), str(key)[:4].upper())

MANUAL_MAP = {
    "daniel vogelbach":   "Dan Vogelbach",
    "michael a taylor":   "Michael Taylor",
    "max muncy (2002)":   "Max Muncy",
    "diego a castillo":   "Diego Castillo",
    "james jarvis":       "Jim Jarvis",
    "donnie walton":      "Donovan Walton",
    "josh kuroda-grauer": "Joshua Kuroda-Grauer",
}


def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    manual_norm = {normalize_name_raw(k): normalize_name_raw(v) for k, v in MANUAL_MAP.items()}
    n = normalize_name_raw(name)
    return manual_norm.get(n, n)


def normalize_name_raw(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[.,'\-]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    name = re.sub(r"\s+", "", name)
    return name.strip()


# ── S3 ────────────────────────────────────────────────────────────────────────

def _s3():
    return boto3.client("s3")


def s3_get_parquet(key: str) -> pd.DataFrame:
    body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def s3_put_csv(key: str, df: pd.DataFrame) -> None:
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=df.to_csv(index=False).encode())


def s3_get_bytes(key: str) -> bytes:
    return _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()


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
        print(f"  Events API error: {r.status_code}")
        return []
    events = r.json()
    # Filter to today's games
    today_events = [e for e in events if e.get("commence_time", "")[:10] == gameday]
    print(f"  Live events for {gameday}: {len(today_events)}")
    return today_events


def fetch_event_props(event_id: str) -> list[dict]:
    r = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds",
        params={"apiKey": ODDS_API_KEY, "markets": MARKETS, "regions": REGIONS},
        timeout=30,
    )
    time.sleep(SLEEP_S)
    if r.status_code != 200 or not r.json():
        return []
    rows = []
    for bm in r.json().get("bookmakers", []):
        book = bm["key"]
        for mkt in bm.get("markets", []):
            mkt_key = mkt["key"]
            over_outcomes  = [o for o in mkt.get("outcomes", []) if o["name"] == "Over"]
            under_outcomes = [o for o in mkt.get("outcomes", []) if o["name"] == "Under"]
            for o in over_outcomes:
                pt     = o.get("point")
                player = o.get("description", "")
                under  = next(
                    (u for u in under_outcomes if u.get("description") == player and u.get("point") == pt), None
                )
                rows.append({
                    "bookmaker":   book,
                    "market_key":  mkt_key,
                    "player_name": player,
                    "line":        pt,
                    "over_price":  o.get("price"),
                    "under_price": under["price"] if under else None,
                })
    return rows


def build_market_consensus(rows: list[dict], event_id: str, home_team: str, away_team: str, gameday: str, commence_time: str = "") -> tuple:
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows)
    df["event_id"]  = event_id
    df["home_team"] = home_team
    df["away_team"] = away_team
    df["game_date"] = gameday
    df = df[
        (df["market_key"] == "batter_total_bases")
        & df["over_price"].notna()
        & df["under_price"].notna()
        & (df["over_price"] > 1.0)
        & (df["under_price"] > 1.0)
    ].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["raw_prob_over"]  = 1.0 / df["over_price"]
    df["raw_prob_under"] = 1.0 / df["under_price"]
    df["total_prob"]     = df["raw_prob_over"] + df["raw_prob_under"]
    df["novig_over"]     = df["raw_prob_over"]  / df["total_prob"]
    df["novig_under"]    = df["raw_prob_under"] / df["total_prob"]
    df["name_norm"]      = df["player_name"].map(normalize_name)

    gtime = commence_to_et(commence_time)

    # Per-book df (one row per player-line-book, used for per-book edge expansion)
    per_book = df[["name_norm", "player_name", "line", "bookmaker",
                   "over_price", "under_price", "novig_over", "novig_under"]].copy()
    per_book["home_team"]   = home_team
    per_book["away_team"]   = away_team
    per_book["game_time_et"] = gtime

    # Best under odds per player-line (for non-play display in email)
    best_idx = df.groupby(["name_norm", "line"])["under_price"].idxmax()
    best_under_df = (
        df.loc[best_idx, ["name_norm", "line", "bookmaker", "under_price"]]
        .rename(columns={"bookmaker": "best_under_book", "under_price": "best_under_price"})
    )

    consensus = (
        df.groupby(["name_norm", "line"])
        .agg(
            player_name        = ("player_name",   "first"),
            avg_raw_prob_over  = ("raw_prob_over",  "mean"),
            avg_raw_prob_under = ("raw_prob_under", "mean"),
            novig_prob_over    = ("novig_over",     "mean"),
            novig_prob_under   = ("novig_under",    "mean"),
            n_books            = ("bookmaker",      "nunique"),
            home_team          = ("home_team",      "first"),
            away_team          = ("away_team",      "first"),
        )
        .reset_index()
    )

    # min_line/max_line per player across all posted lines (v2 model features)
    line_range = (
        consensus.groupby("name_norm")["line"]
        .agg(min_line="min", max_line="max")
        .reset_index()
    )
    consensus = consensus.merge(line_range, on="name_norm", how="left")

    consensus = consensus.merge(best_under_df, on=["name_norm", "line"], how="left")
    consensus["game_time_et"] = gtime
    return consensus, per_book


# ── Rolling features lookup ───────────────────────────────────────────────────

def get_latest_features(spine: pd.DataFrame) -> pd.DataFrame:
    """Return most recent row per player (latest game_date). Used for inference."""
    spine = spine.sort_values("game_date")
    latest = spine.groupby("name_norm").tail(1).copy()
    return latest.set_index("name_norm")


# ── Score ─────────────────────────────────────────────────────────────────────

def score_slate(
    consensus: pd.DataFrame,
    latest_feats: pd.DataFrame,
    model_bundle: dict,
) -> pd.DataFrame:
    """Score each player-line using v2 regression + Method C calibration.

    y_hat = XGBoost prediction of expected total bases (player-game level).
    P(over | line) = per-line LogisticRegression(y_hat) from calib_models.
    edge_under = P(under | model) - novig_prob_under (consensus).
    """
    reg_model    = model_bundle["model"]
    scaler       = model_bundle["scaler"]
    features     = model_bundle["features_numeric"]
    calib_models = model_bundle.get("calib_models", {})

    rows = []
    for _, row in consensus.iterrows():
        player_key = row["name_norm"]
        if player_key not in latest_feats.index:
            continue
        feat_row = latest_feats.loc[player_key]
        line     = float(row["line"])

        if line not in calib_models:
            continue

        rec = {
            "player_name":       row["player_name"],
            "name_norm":         player_key,
            "team":              feat_row.get("team", ""),
            "opponent":          feat_row.get("opponent", ""),
            "line":              line,
            "n_books":           row["n_books"],
            "novig_prob_over":   row["novig_prob_over"],
            "novig_prob_under":  row["novig_prob_under"],
            "avg_raw_prob_over": row["avg_raw_prob_over"],
            "avg_raw_prob_under":row["avg_raw_prob_under"],
            "home_team":         row.get("home_team", ""),
            "away_team":         row.get("away_team", ""),
            "game_time_et":      row.get("game_time_et", ""),
            "best_under_book":   row.get("best_under_book", ""),
            "best_under_price":  row.get("best_under_price", np.nan),
            "min_line":          row.get("min_line", np.nan),
            "max_line":          row.get("max_line", np.nan),
            "tb_Lcareer":        feat_row.get("tb_Lcareer",  np.nan),
            "tb_L10":            feat_row.get("tb_L10",       np.nan),
            "ab_Lcareer":        feat_row.get("ab_Lcareer",  np.nan),
            "ab_L10":            feat_row.get("ab_L10",       np.nan),
        }

        # Build feature vector using player-game-level features
        feat_vals = []
        for f in features:
            # min_line/max_line come from live consensus, not spine
            if f == "min_line":
                feat_vals.append(row.get("min_line", np.nan))
            elif f == "max_line":
                feat_vals.append(row.get("max_line", np.nan))
            else:
                feat_vals.append(feat_row.get(f, np.nan))
        X = np.array(feat_vals, dtype=float).reshape(1, -1)
        if np.any(np.isnan(X)):
            continue

        # y_hat = regression prediction
        X_sc  = scaler.transform(X)
        y_hat = float(reg_model.predict(X_sc)[0])
        rec["y_hat"] = y_hat

        # P(over | line) via Method C per-line logistic calibration
        calib = calib_models[line]
        p_over = float(calib.predict_proba([[y_hat]])[0, 1])
        p_over = float(np.clip(p_over, 0.01, 0.99))

        rec["p_model"]        = p_over
        rec["p_market"]       = row["novig_prob_over"]
        rec["edge_over"]      = p_over - row["novig_prob_over"]
        rec["edge_under"]     = (1 - p_over) - row["novig_prob_under"]
        rec["dec_odds_under"] = 1.0 / row["avg_raw_prob_under"]
        rec["over_odds"]      = 1.0 / row["avg_raw_prob_over"]
        rec["bet_direction"]  = "UNDER"

        rows.append(rec)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Per-book expansion ───────────────────────────────────────────────────────

def expand_to_books(scored: pd.DataFrame, all_book_rows: pd.DataFrame) -> pd.DataFrame:
    """Join p_model (consensus-scored) back onto per-book rows; compute per-book edge."""
    if scored.empty or all_book_rows.empty:
        return pd.DataFrame()
    base_keep = ["name_norm", "line", "p_model", "team", "opponent",
                 "n_books", "novig_prob_over", "novig_prob_under"]
    optional  = ["y_hat", "tb_Lcareer", "tb_L20", "tb_L10", "ab_Lcareer", "ab_L10"]
    keep = base_keep + [c for c in optional if c in scored.columns]
    slim = scored[keep].copy()
    merged = all_book_rows.merge(slim, on=["name_norm", "line"], how="inner")
    merged["edge_under"]     = (1 - merged["p_model"]) - merged["novig_under"]
    merged["dec_odds_under"] = merged["under_price"]  # alias for settle compatibility
    return merged


# ── Email ─────────────────────────────────────────────────────────────────────

def _time_sort_key(t: str) -> int:
    """Convert '7:07 PM' → minutes since midnight for chronological sort."""
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
    tiered_books: pd.DataFrame,
    all_scored: pd.DataFrame,
    gameday: str,
    bet_lines: list,
    min_bet_edge: float,
    min_track_edge: float,
) -> str:
    he = html_module.escape

    play_books  = tiered_books[tiered_books["tier"] == "play"]  if not tiered_books.empty else pd.DataFrame()
    track_books = tiered_books[tiered_books["tier"] == "track"] if not tiered_books.empty else pd.DataFrame()
    n_play_bets  = len(play_books)
    n_track_bets = len(track_books)
    n_play_players  = play_books["name_norm"].nunique()  if not play_books.empty  else 0
    n_track_players = track_books["name_norm"].nunique() if not track_books.empty else 0

    def fmt(v, fmt_str):
        try:
            return format(float(v), fmt_str)
        except (TypeError, ValueError):
            return "—"

    # ── Section 1: per-book rows for qualifying, consensus row for non-qualifying ──
    at_bet = all_scored[all_scored["line"].isin(bet_lines)].copy()

    # Per-book qualifying rows at bet lines
    tiered_bet = (
        tiered_books[tiered_books["line"].isin(bet_lines)].copy()
        if not tiered_books.empty else pd.DataFrame()
    )

    # Non-qualifying: players in at_bet with no qualifying book row
    if not tiered_bet.empty:
        qual_set = set(zip(tiered_bet["name_norm"].astype(str), tiered_bet["line"].astype(float)))
        non_qual = at_bet[~at_bet.apply(
            lambda r: (str(r["name_norm"]), float(r["line"])) in qual_set, axis=1
        )].copy()
    else:
        non_qual = at_bet.copy()

    # Sort tiered: plays first, then tracks; within tier by edge desc
    if not tiered_bet.empty:
        tiered_bet["_tsort"]      = tiered_bet["game_time_et"].map(_time_sort_key)
        tiered_bet["_tier_order"] = tiered_bet["tier"].map({"play": 0, "track": 1})
        tiered_bet = tiered_bet.sort_values(
            ["_tsort", "home_team", "_tier_order", "name_norm", "edge_under"],
            ascending=[True, True, True, True, False],
        )

    non_qual["_tsort"] = non_qual["game_time_et"].map(_time_sort_key)
    non_qual = non_qual.sort_values(
        ["_tsort", "home_team", "line", "edge_under"], ascending=[True, True, True, False]
    )

    # Build sorted unique game list
    game_srcs = []
    if not tiered_bet.empty:
        game_srcs.append(tiered_bet[["_tsort", "game_time_et", "home_team", "away_team"]])
    if not non_qual.empty:
        game_srcs.append(non_qual[["_tsort", "game_time_et", "home_team", "away_team"]])
    all_games = (
        pd.concat(game_srcs)
        .drop_duplicates(["home_team", "away_team"])
        .sort_values("_tsort")
        if game_srcs else pd.DataFrame()
    )

    S1_COLS = 16
    rows_bet = ""

    for _, grow in all_games.iterrows():
        gtime = grow["game_time_et"]
        home  = grow["home_team"]
        away  = grow["away_team"]

        game_tiered = tiered_bet[
            (tiered_bet["home_team"] == home) & (tiered_bet["away_team"] == away)
        ] if not tiered_bet.empty else pd.DataFrame()
        n_gplays  = int((game_tiered["tier"] == "play").sum())  if not game_tiered.empty else 0
        n_gtracks = int((game_tiered["tier"] == "track").sum()) if not game_tiered.empty else 0

        status_parts = []
        if n_gplays:
            status_parts.append(f"<span style='color:#276221'>{n_gplays} play{'s' if n_gplays!=1 else ''}</span>")
        if n_gtracks:
            status_parts.append(f"<span style='color:#b8860b'>{n_gtracks} track{'s' if n_gtracks!=1 else ''}</span>")
        if not status_parts:
            status_parts.append("<span style='color:#888'>no plays</span>")
        rows_bet += (
            f"<tr style='background:#edf1f5'>"
            f"<td colspan='{S1_COLS}' style='padding:7px 10px;font-weight:600;font-size:12px;"
            f"color:#2c3e50;border-top:2px solid #bdc3c7;border-bottom:1px solid #bdc3c7'>"
            f"{he(gtime)} ET &nbsp;·&nbsp; {he(away)} @ {he(home)}"
            f" &nbsp;·&nbsp; {' &nbsp;·&nbsp; '.join(status_parts)}"
            f"</td></tr>\n"
        )

        # One row per qualifying book
        for _, r in game_tiered.iterrows():
            tier = r.get("tier")
            if tier == "play":
                bg     = "background:#eaf6ea"
                status = "<span style='color:#276221;font-weight:bold'>PLAY ✓</span>"
            else:
                bg     = "background:#fffde7"
                status = "<span style='color:#b8860b;font-weight:bold'>TRACK</span>"

            book_abbrev = he(display_book(str(r.get("bookmaker", ""))))
            under_odds  = to_american(r.get("under_price"))
            over_odds   = to_american(r.get("over_price"))
            mkt_under   = fmt(r.get("novig_under"), ".1%")
            model_under = fmt(1 - float(r.get("p_model", np.nan)), ".1%")
            edge_str    = fmt(r.get("edge_under"), "+.1%")
            proj_tb     = fmt(r.get("y_hat"), ".2f")

            rows_bet += (
                f"<tr style='{bg}'>"
                f"<td>{he(str(r.get('player_name', '')))}</td>"
                f"<td style='text-align:center;color:#555'>{he(str(r.get('team','—')))}</td>"
                f"<td style='text-align:center;color:#555'>{he(str(r.get('opponent','—')))}</td>"
                f"<td style='text-align:center'>{fmt(r.get('line'), '.1f')}</td>"
                f"<td style='text-align:center;font-weight:600'>{book_abbrev}</td>"
                f"<td style='text-align:center;color:#555'>{over_odds}</td>"
                f"<td style='text-align:center;font-weight:bold;color:#1d4ed8'>{under_odds}</td>"
                f"<td style='text-align:center'>{mkt_under}</td>"
                f"<td style='text-align:center'>{model_under}</td>"
                f"<td style='text-align:center'>{edge_str}</td>"
                f"<td style='text-align:center;font-size:11px;color:#1565c0;font-weight:bold'>{proj_tb}</td>"
                f"<td style='text-align:center;font-size:11px;color:#555'>{fmt(r.get('ab_Lcareer'), '.1f')}</td>"
                f"<td style='text-align:center;font-size:11px;color:#555'>{fmt(r.get('ab_L10'), '.1f')}</td>"
                f"<td style='text-align:center;font-size:11px;color:#555'>{fmt(r.get('tb_Lcareer'), '.2f')}</td>"
                f"<td style='text-align:center;font-size:11px;color:#555'>{fmt(r.get('tb_L10'), '.2f')}</td>"
                f"<td style='text-align:center'>{status}</td>"
                f"</tr>\n"
            )

        # One consensus row per non-qualifying player-line in this game
        game_non_qual = non_qual[
            (non_qual["home_team"] == home) & (non_qual["away_team"] == away)
        ] if not non_qual.empty else pd.DataFrame()

        for _, r in game_non_qual.iterrows():
            best_book  = str(r.get("best_under_book", "") or "—")
            under_odds = to_american(r.get("best_under_price"))
            over_odds  = to_american(r.get("over_odds"))

            rows_bet += (
                f"<tr>"
                f"<td style='color:#888'>{he(str(r.get('player_name', '')))}</td>"
                f"<td style='text-align:center;color:#aaa'>{he(str(r.get('team','—')))}</td>"
                f"<td style='text-align:center;color:#aaa'>{he(str(r.get('opponent','—')))}</td>"
                f"<td style='text-align:center;color:#aaa'>{fmt(r.get('line'), '.1f')}</td>"
                f"<td style='text-align:center;font-size:11px;color:#aaa'>{he(display_book(best_book))}</td>"
                f"<td style='text-align:center;color:#aaa'>{over_odds}</td>"
                f"<td style='text-align:center;color:#aaa'>{under_odds}</td>"
                f"<td style='text-align:center;color:#aaa'>{fmt(r.get('novig_prob_under'), '.1%')}</td>"
                f"<td style='text-align:center;color:#aaa'>{fmt(1 - float(r.get('p_model', np.nan)), '.1%')}</td>"
                f"<td style='text-align:center;color:#aaa'>{fmt(r.get('edge_under'), '+.1%')}</td>"
                f"<td style='text-align:center;font-size:11px;color:#aaa'>{fmt(r.get('y_hat'), '.2f')}</td>"
                f"<td style='text-align:center;font-size:11px;color:#aaa'>{fmt(r.get('ab_Lcareer'), '.1f')}</td>"
                f"<td style='text-align:center;font-size:11px;color:#aaa'>{fmt(r.get('ab_L10'), '.1f')}</td>"
                f"<td style='text-align:center;font-size:11px;color:#aaa'>{fmt(r.get('tb_Lcareer'), '.2f')}</td>"
                f"<td style='text-align:center;font-size:11px;color:#aaa'>{fmt(r.get('tb_L10'), '.2f')}</td>"
                f"<td style='text-align:center'><span style='color:#aaa'>—</span></td>"
                f"</tr>\n"
            )

    # ── Section 2: all lines (consensus, unchanged) ───────────────────────────
    all_s = all_scored.copy()
    all_s["_tsort"] = all_s["game_time_et"].map(_time_sort_key)
    all_s = all_s.sort_values(["_tsort", "home_team", "line", "edge_under"], ascending=[True, True, True, False])
    rows_all = ""
    for _, r in all_s.iterrows():
        rows_all += (
            f"<tr>"
            f"<td>{he(str(r['player_name']))}</td>"
            f"<td style='text-align:center;color:#555'>{he(str(r.get('team','—')))}</td>"
            f"<td style='text-align:center;color:#555'>{he(str(r.get('opponent','—')))}</td>"
            f"<td style='text-align:center;color:#555'>{he(str(r.get('game_time_et','—')))}</td>"
            f"<td style='text-align:center'>{fmt(r['line'], '.1f')}</td>"
            f"<td style='text-align:center'>{int(r['n_books'])}</td>"
            f"<td style='text-align:center'>{fmt(r['novig_prob_under'], '.1%')}</td>"
            f"<td style='text-align:center'>{fmt(1 - r['p_model'], '.1%')}</td>"
            f"<td style='text-align:center'>{fmt(r['edge_under'], '+.1%')}</td>"
            f"</tr>\n"
        )

    lines_str   = "+".join(str(l) for l in sorted(bet_lines))
    play_label  = f"{n_play_players}p / {n_play_bets}b plays (≥{min_bet_edge*100:.0f}pp)"
    track_label = f"{n_track_players}p / {n_track_bets}b tracks ({min_track_edge*100:.0f}–{min_bet_edge*100:.0f}pp)"
    n_s1_rows   = len(at_bet)

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<style>
  body {{font-family:{_SANS};color:#222;max-width:1200px;margin:auto;padding:20px}}
  h2 {{color:#2c3e50;margin-bottom:4px}}
  table {{border-collapse:collapse;width:100%;margin-top:8px}}
  th {{background:#2c3e50;color:#fff;padding:7px 8px;text-align:left;font-size:12px;white-space:nowrap}}
  td {{padding:5px 8px;border-bottom:1px solid #e0e0e0;font-size:12px}}
  details {{margin-top:16px;border:1px solid #ddd;border-radius:6px;padding:0 12px 8px}}
  summary {{font-weight:600;font-size:14px;cursor:pointer;padding:10px 0;color:#2c3e50;user-select:none}}
  .footer {{background:#ecf0f1;border-radius:6px;padding:10px 16px;margin-top:16px;font-size:12px;color:#555}}
  .legend-play  {{display:inline-block;width:12px;height:12px;background:#eaf6ea;border:1px solid #276221;margin-right:4px;vertical-align:middle}}
  .legend-track {{display:inline-block;width:12px;height:12px;background:#fffde7;border:1px solid #b8860b;margin-right:4px;vertical-align:middle}}
</style>
</head><body>
<h2>MLB Total Bases — {gameday}</h2>
<p style='margin-top:4px'>
  <span style='color:#276221;font-weight:bold'>{play_label}</span>
  &nbsp;·&nbsp;
  <span style='color:#b8860b;font-weight:bold'>{track_label}</span>
  &nbsp;·&nbsp; UNDER {lines_str}
</p>
<p style='font-size:11px;color:#888;margin-top:2px'>
  <span class='legend-play'></span>Green = PLAY (bet) &nbsp;&nbsp;
  <span class='legend-track'></span>Yellow = TRACK (paper only) &nbsp;&nbsp;
  Grey = no edge (context only)
</p>

<details open>
  <summary>▸ Strategy: UNDER {lines_str} &nbsp;<span style='font-weight:normal;color:#666'>({n_s1_rows} players scored · {n_play_bets} bets)</span></summary>
  <table>
    <tr>
      <th>Player</th><th>Team</th><th>Opp</th><th>Line</th>
      <th>Book</th><th>Over</th><th>Under</th>
      <th>Mkt Under%</th><th>Model Under%</th><th>Edge</th>
      <th>Proj TB</th>
      <th>AB/G (C)</th><th>AB/G (L10)</th><th>TB/G (C)</th><th>TB/G (L10)</th>
      <th>Status</th>
    </tr>
    {rows_bet}
  </table>
</details>

<details>
  <summary>▸ All live props &nbsp;<span style='font-weight:normal;color:#666'>({len(all_scored)} total across all lines)</span></summary>
  <table>
    <tr><th>Player</th><th>Team</th><th>Opp</th><th>Game Time ET</th><th>Line</th><th>Books</th><th>Mkt Under%</th><th>Model Under%</th><th>Edge</th></tr>
    {rows_all}
  </table>
</details>

<div class='footer'>
  Flat 1u per book bet · v2 XGBoost regression + Method C calibration &nbsp;·&nbsp;
  Plays (≥{min_bet_edge*100:.0f}pp, dogs, line=1.5): OOS +211.9u, ROI=+1.72%, n=12,323 &nbsp;·&nbsp;
  Tracks ({min_track_edge*100:.0f}pp, dogs, line=1.5): OOS +47.7u, ROI=+0.34%, n=14,110
</div>

<details style='margin-top:16px'>
  <summary style='font-weight:600;font-size:13px;cursor:pointer;padding:8px 0;color:#888;user-select:none'>▸ Model inputs (8 features — v2 XGBoost regression)</summary>
  <table style='margin-top:6px;width:auto'>
    <tr><th style='background:#95a5a6;color:#fff;padding:6px 10px;font-size:12px'>Feature</th><th style='background:#95a5a6;color:#fff;padding:6px 10px;font-size:12px'>Description</th><th style='background:#95a5a6;color:#fff;padding:6px 10px;font-size:12px'>Importance</th></tr>
    <tr><td style='padding:5px 10px;font-size:12px;font-family:{_MONO}'>max_line</td><td style='padding:5px 10px;font-size:12px'>Max line posted for player today — market consensus on ceiling</td><td style='padding:5px 10px;font-size:12px'>45.8%</td></tr>
    <tr style='background:#f9f9f9'><td style='padding:5px 10px;font-size:12px;font-family:{_MONO}'>min_line</td><td style='padding:5px 10px;font-size:12px'>Min line posted for player today — market consensus on floor</td><td style='padding:5px 10px;font-size:12px'>21.4%</td></tr>
    <tr><td style='padding:5px 10px;font-size:12px;font-family:{_MONO}'>tb_Lcareer</td><td style='padding:5px 10px;font-size:12px'>Career avg total bases per game — baseline hitter quality</td><td style='padding:5px 10px;font-size:12px'>8.9%</td></tr>
    <tr style='background:#f9f9f9'><td style='padding:5px 10px;font-size:12px;font-family:{_MONO}'>ab_Lcareer</td><td style='padding:5px 10px;font-size:12px'>Career avg at-bats per game — lineup position / playing time</td><td style='padding:5px 10px;font-size:12px'>6.2%</td></tr>
    <tr><td style='padding:5px 10px;font-size:12px;font-family:{_MONO}'>tb_Lseason</td><td style='padding:5px 10px;font-size:12px'>Season-to-date avg total bases per game</td><td style='padding:5px 10px;font-size:12px'>6.0%</td></tr>
    <tr style='background:#f9f9f9'><td style='padding:5px 10px;font-size:12px;font-family:{_MONO}'>hr_Lcareer</td><td style='padding:5px 10px;font-size:12px'>Career avg home runs per game — power hitter signal</td><td style='padding:5px 10px;font-size:12px'>4.1%</td></tr>
    <tr><td style='padding:5px 10px;font-size:12px;font-family:{_MONO}'>tb_L20</td><td style='padding:5px 10px;font-size:12px'>Last 20 games avg total bases — medium-term form</td><td style='padding:5px 10px;font-size:12px'>3.9%</td></tr>
    <tr style='background:#f9f9f9'><td style='padding:5px 10px;font-size:12px;font-family:{_MONO}'>tb_L10</td><td style='padding:5px 10px;font-size:12px'>Last 10 games avg total bases — recent form</td><td style='padding:5px 10px;font-size:12px'>3.7%</td></tr>
  </table>
  <p style='font-size:11px;color:#888;margin-top:6px'>
    Model: XGBoost regressor → y_hat (projected TB) → Method C calibration per line → P(under) → edge.<br>
    Strategy: UNDER line=1.5 · dogs only (novig_under &lt; 50%) · play ≥5pp · track ≥3pp.
  </p>
</details>
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
    parser.add_argument("--gameday", default=today_et())
    args = parser.parse_args()
    gameday = args.gameday

    print(f"MLB Total Bases pipeline | gameday={gameday}")

    # Load config
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    strat = cfg.get("strategy", {})
    BET_LINES      = strat.get("lines", [1.5])
    MIN_BET_EDGE   = strat.get("min_bet_edge", 0.05)
    MIN_TRACK_EDGE = strat.get("min_track_edge", 0.03)
    DOGS_ONLY      = strat.get("dogs_only", True)
    novig_min      = strat.get("novig_min", NOVIG_MIN)
    print(f"  Strategy: UNDER lines={BET_LINES}, play≥{MIN_BET_EDGE*100:.0f}pp, track≥{MIN_TRACK_EDGE*100:.0f}pp, dogs_only={DOGS_ONLY}")

    # Load spine
    print("Loading spine from S3 ...")
    spine = s3_get_parquet(SPINE_KEY)
    print(f"  {len(spine):,} rows  |  {spine['name_norm'].nunique():,} players")
    latest_feats = get_latest_features(spine)

    # Load v2 regression model (includes calibration models in bundle)
    print("Loading v2 regression model from S3 ...")
    model_bytes  = s3_get_bytes(MODEL_KEY)
    model_bundle = joblib.load(BytesIO(model_bytes))
    calib_lines  = model_bundle.get("calib_lines", [])
    print(f"  Model: {model_bundle.get('combo_name','?')} ({model_bundle.get('model_type','?')})")
    print(f"  Features: {model_bundle['features_numeric']}")
    print(f"  Calibration lines: {calib_lines}")

    # Fetch live events
    print("Fetching live Odds API events ...")
    events = fetch_live_events(gameday)
    if not events:
        msg = f"No MLB events found for {gameday}"
        print(msg)
        publish_sns(f"MLB TB pipeline — no events {gameday}", msg)
        return

    # Fetch props for each event
    all_rows: list[pd.DataFrame] = []
    all_book_rows: list[pd.DataFrame] = []
    for ev in events:
        event_id  = ev["id"]
        home_team = ev.get("home_team", "")
        away_team = ev.get("away_team", "")
        commence_time = ev.get("commence_time", "")
        props = fetch_event_props(event_id)
        consensus, per_book = build_market_consensus(props, event_id, home_team, away_team, gameday, commence_time)
        if not consensus.empty:
            all_rows.append(consensus)
        if not per_book.empty:
            all_book_rows.append(per_book)
        print(f"  {away_team[:15]:15} @ {home_team[:15]:15}  {len(consensus)} players")

    if not all_rows:
        msg = f"No batter_total_bases props found for {gameday}"
        print(msg)
        publish_sns(f"MLB TB pipeline — no props {gameday}", msg)
        return

    consensus_df = pd.concat(all_rows, ignore_index=True)
    consensus_df = consensus_df[(consensus_df["novig_prob_over"] >= novig_min)].copy()
    book_rows_df = pd.concat(all_book_rows, ignore_index=True) if all_book_rows else pd.DataFrame()
    print(f"\nTotal scored props (after novig filter): {len(consensus_df)}")

    # Score
    scored = score_slate(consensus_df, latest_feats, model_bundle)
    if scored.empty:
        msg = f"0 scored props for {gameday} (all players missing from spine)"
        print(msg)
        publish_sns(f"MLB TB pipeline — 0 scored {gameday}", msg)
        return

    print(f"Scored: {len(scored)} player-lines")

    # Expand to per-book, tag tier=play|track
    per_book_scored = expand_to_books(scored, book_rows_df)
    tiered_books = pd.DataFrame()
    if not per_book_scored.empty:
        subset = per_book_scored[
            (per_book_scored["line"].isin(BET_LINES))
            & (per_book_scored["edge_under"] >= MIN_TRACK_EDGE)
        ].copy()
        if not subset.empty:
            subset["tier"] = "track"
            # Play tier: edge≥bet threshold AND dogs only (novig_under < 0.50 = +odds)
            play_mask = subset["edge_under"] >= MIN_BET_EDGE
            if DOGS_ONLY:
                play_mask = play_mask & (subset["novig_under"] < 0.50)
            subset.loc[play_mask, "tier"] = "play"
            tiered_books = subset

    play_books  = tiered_books[tiered_books["tier"] == "play"]  if not tiered_books.empty else pd.DataFrame()
    track_books = tiered_books[tiered_books["tier"] == "track"] if not tiered_books.empty else pd.DataFrame()
    n_play_players  = play_books["name_norm"].nunique()  if not play_books.empty  else 0
    n_track_players = track_books["name_norm"].nunique() if not track_books.empty else 0
    n_play_bets  = len(play_books)
    n_track_bets = len(track_books)
    print(f"Plays:  {n_play_players} players, {n_play_bets} bets  (edge≥{MIN_BET_EDGE*100:.0f}pp)")
    print(f"Tracks: {n_track_players} players, {n_track_bets} bets (edge≥{MIN_TRACK_EDGE*100:.0f}pp)")

    # Save to S3
    scored_key = f"{DAILY_PREFIX}/{gameday}/scored.csv"
    s3_put_csv(scored_key, scored)
    print(f"  Scored → s3://{S3_BUCKET}/{scored_key}")

    if not tiered_books.empty:
        bets_key = f"{DAILY_PREFIX}/{gameday}/recommendations.csv"
        s3_put_csv(bets_key, tiered_books)
        print(f"  Recs   → s3://{S3_BUCKET}/{bets_key}  ({n_play_bets} plays + {n_track_bets} tracks)")

    # Email — always send (even 0 bets, so we know the pipeline ran)
    lines_str = "+".join(str(l) for l in sorted(BET_LINES))
    subject = (
        f"MLB Total Bases — {n_play_bets} play{'s' if n_play_bets != 1 else ''} · "
        f"{n_track_bets} track{'s' if n_track_bets != 1 else ''} — {gameday}"
    )
    html_body = build_html_email(tiered_books, scored, gameday, BET_LINES, MIN_BET_EDGE, MIN_TRACK_EDGE)
    send_ses(subject, html_body)
    if n_play_bets > 0:
        publish_sns(subject, f"{n_play_players} players, {n_play_bets} UNDER {lines_str} plays for {gameday}. Check email.")


if __name__ == "__main__":
    main()
