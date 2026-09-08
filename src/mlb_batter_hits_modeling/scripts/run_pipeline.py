"""
Live gameday pipeline for MLB Batter Hits props.

Strategy: 0.5 UNDER · plus_odds_only (under_price > 2.0) · edge >= 2pp (play) / 1pp (watch)
OOS: 2,263 bets · 55.5% hit · +497u · +22.0% ROI · net/MDD=17.05x

For each batter with a batter_hits prop on the given gameday:
  1. Fetch live events + props from The Odds API (batter_hits + batter_hits_alternate)
  2. Load rolling features from the spine (S3) — most recent row per player
  3. Compute market features (min/max implied probs, consensus line)
  4. Score with OLS (S3) → yhat, then Logistic (S3) → p_model
  5. Compute under_edge = (1-p_model) - (1/under_price)  [raw, vig-inclusive]
  6. Filter to 0.5 UNDER, edge >= edge_min
  7. Send SES HTML email + SNS notification
  8. Save recommendations CSV to S3

S3 paths read:
  s3://the-odds-api-mt/mlb/batter_hits_model/spine/mlb_batter_hits_spine.parquet
  s3://the-odds-api-mt/mlb/batter_hits_model/model/mlb_batter_hits_ols.joblib
  s3://the-odds-api-mt/mlb/batter_hits_model/model/mlb_batter_hits_logit.joblib

S3 paths written:
  s3://the-odds-api-mt/mlb/batter_hits_model/daily_runs/{gameday}/recommendations.csv

Usage:
  python src/mlb_batter_hits_modeling/scripts/run_pipeline.py
  python src/mlb_batter_hits_modeling/scripts/run_pipeline.py --gameday 2026-07-27
  python src/mlb_batter_hits_modeling/scripts/run_pipeline.py --output /tmp/hits_out.json
"""
from __future__ import annotations

import argparse
import html as html_module
import json
import os
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
ET            = ZoneInfo("America/New_York")

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "baseball_mlb"
SLEEP_S       = 0.25

SES_SOURCE    = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()


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


def _dec_to_american(d: float) -> str:
    if not d or pd.isna(d):
        return "N/A"
    if d >= 2.0:
        return f"+{int(round((d - 1) * 100))}"
    return f"-{int(round(100 / (d - 1)))}"


# -----------------------------------------------------------------------
# Data fetching
# -----------------------------------------------------------------------

def fetch_events(cfg: dict, gameday: str) -> list[dict]:
    """Fetch today's MLB games from Odds API."""
    if not ODDS_API_KEY:
        print("  ODDS_API_KEY not set — using S3 cache if available")
        return []
    url = f"{ODDS_API_BASE}/sports/{SPORT}/events"
    resp = requests.get(url, params={"apiKey": ODDS_API_KEY}, timeout=30)
    resp.raise_for_status()
    events = resp.json()
    today_events = [e for e in events if e.get("commence_time", "").startswith(gameday)]
    print(f"  {len(today_events)} games on {gameday}")
    return today_events


def fetch_props_for_event(event_id: str, cfg: dict) -> list[dict]:
    """Fetch batter_hits + batter_hits_alternate props for one event."""
    markets = f"{cfg['market']['market_key']},{cfg['market']['alt_market_key']}"
    url = f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": cfg["market"]["regions"],
        "markets": markets,
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(SLEEP_S)
    return resp.json().get("bookmakers", [])


def parse_bookmakers(event: dict, bookmakers: list[dict], gameday: str) -> list[dict]:
    """Flatten bookmaker props into rows."""
    rows = []
    for bk in bookmakers:
        for market in bk.get("markets", []):
            for outcome in market.get("outcomes", []):
                rows.append({
                    "event_id":    event["id"],
                    "game_date":   gameday,
                    "home_team":   event.get("home_team", ""),
                    "away_team":   event.get("away_team", ""),
                    "commence_time": event.get("commence_time", ""),
                    "bookmaker":   bk["key"],
                    "market_key":  market["key"],
                    "player_name": outcome.get("description", ""),
                    "side":        outcome["name"],
                    "line":        outcome.get("point"),
                    "price":       outcome.get("price"),
                })
    return rows


def build_market_df(events: list[dict], cfg: dict, gameday: str) -> pd.DataFrame:
    """Fetch all props → pivot Over/Under → one row per player/book/line."""
    all_rows = []
    for ev in events:
        if not ODDS_API_KEY:
            break
        bks = fetch_props_for_event(ev["id"], cfg)
        all_rows.extend(parse_bookmakers(ev, bks, gameday))

    if not all_rows:
        # Fallback: load from S3 market_raw cache (for Lambda / testing)
        return _load_market_from_s3_cache(cfg, gameday)

    df = pd.DataFrame(all_rows)
    df = df[df["side"].isin(["Over", "Under"])].copy()
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Pivot: one row per (event_id, player_name, bookmaker, line)
    pivot = df.pivot_table(
        index=["event_id", "game_date", "home_team", "away_team", "commence_time",
               "bookmaker", "player_name", "line"],
        columns="side",
        values="price",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"Over": "over_price", "Under": "under_price"})

    # Normalize player key
    pivot["player_key"] = pivot["player_name"].apply(normalize_name)
    pivot["game_date"] = gameday
    return pivot


def _load_market_from_s3_cache(cfg: dict, gameday: str) -> pd.DataFrame:
    """Load pre-fetched market parquet files from S3 for the given gameday."""
    s3 = boto3.client("s3")
    bucket = cfg["data"]["s3_bucket"]
    prefix = f"{cfg['data']['market_prefix']}/2026/{gameday.replace('-', '')}"
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        frames = []
        for obj in resp.get("Contents", []):
            if obj["Size"] == 0:
                continue
            data = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            frames.append(pd.read_parquet(BytesIO(data)))
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df["player_key"] = df["player_name"].apply(normalize_name)
            print(f"  Loaded {len(df):,} rows from S3 cache for {gameday}")
            return df
    except Exception as e:
        print(f"  S3 cache fallback failed: {e}")
    return pd.DataFrame()


# -----------------------------------------------------------------------
# Feature engineering for today
# -----------------------------------------------------------------------

_SPINE_COLS = [
    "player_key", "game_date", "hits_actual",
    "hits_roll_career", "hits_roll_L1", "hits_roll_L5", "hits_roll_L10",
    "hits_roll_L20", "hits_roll_season", "ba_roll_career", "ba_roll_L5",
    "ab_roll_career",
]

def load_spine_from_s3(cfg: dict) -> pd.DataFrame:
    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket=cfg["data"]["s3_bucket"], Key=cfg["data"]["spine_key"])
    buf = BytesIO(obj["Body"].read())
    import pyarrow.parquet as pq
    available = pq.read_schema(buf).names
    buf.seek(0)
    cols = [c for c in _SPINE_COLS if c in available]
    spine = pd.read_parquet(buf, columns=cols)
    print(f"  Spine: {len(spine):,} rows × {len(cols)} cols")
    return spine


def get_latest_player_features(spine: pd.DataFrame) -> pd.DataFrame:
    """Most recent row per player (their latest rolling features)."""
    rolling_cols = [
        "player_key", "game_date",
        "hits_roll_career", "hits_roll_L1", "hits_roll_L5", "hits_roll_L10",
        "hits_roll_L20", "hits_roll_season", "ba_roll_career", "ba_roll_L5",
        "ab_roll_career",
    ]
    avail = [c for c in rolling_cols if c in spine.columns]
    settled = spine[spine["hits_actual"].notna()][avail].dropna(subset=["hits_roll_career"])
    return (
        settled.sort_values("game_date")
        .drop_duplicates(subset=["player_key"], keep="last")
        .reset_index(drop=True)
    )


def build_market_features(market_df: pd.DataFrame) -> pd.DataFrame:
    """Add min/max implied prob and consensus features at player-game level."""
    market_df = market_df.copy()
    market_df["raw_implied_prob_over"]  = 1.0 / market_df["over_price"]
    market_df["raw_implied_prob_under"] = 1.0 / market_df["under_price"]

    pg = market_df.groupby("player_key").agg(
        min_line=("line", "min"),
        max_line=("line", "max"),
        min_raw_implied_prob_over=("raw_implied_prob_over", "min"),
        max_raw_implied_prob_over=("raw_implied_prob_over", "max"),
        min_raw_implied_prob_under=("raw_implied_prob_under", "min"),
        max_raw_implied_prob_under=("raw_implied_prob_under", "max"),
        consensus_line=("line", "mean"),
    ).reset_index()

    return market_df.merge(pg, on="player_key", how="left", suffixes=("", "_pg"))


# -----------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------

def score(market_df: pd.DataFrame, latest_features: pd.DataFrame,
          ols, logit, cfg: dict) -> pd.DataFrame:
    """Merge features, apply OLS → yhat, logistic → p_model, compute edges."""
    feat_names = cfg["model"]["ols_features"]

    df = market_df.merge(latest_features, on="player_key", how="left",
                         suffixes=("", "_spine"))

    # Override market pg features with correct names
    for col in ["min_line", "max_line", "min_raw_implied_prob_over",
                "max_raw_implied_prob_over", "min_raw_implied_prob_under",
                "max_raw_implied_prob_under", "consensus_line"]:
        pg_col = col + "_pg"
        if pg_col in df.columns:
            df[col] = df[pg_col].fillna(df.get(col, np.nan))

    avail = [f for f in feat_names if f in df.columns]
    X = df[avail].values
    has_all = ~np.any(np.isnan(X.astype(float)), axis=1)

    df["yhat_ols"] = np.nan
    df["p_model"]  = np.nan
    if has_all.any():
        df.loc[has_all, "yhat_ols"] = ols.predict(X[has_all])

        # Assert yhat is book-invariant before computing p_model
        scored_so_far = df[df["yhat_ols"].notna()]
        yhat_spread = (
            scored_so_far.groupby(["player_key", "line"])["yhat_ols"]
            .agg(lambda x: x.max() - x.min())
        )
        if yhat_spread.max() >= 1e-8:
            bad = yhat_spread[yhat_spread >= 1e-8].reset_index()[["player_key", "line"]].values.tolist()
            raise RuntimeError(
                f"yhat_ols is not book-invariant for {len(bad)} (player, line) groups: {bad[:5]}. "
                f"A per-book feature has entered the model inputs — do not send email."
            )

        X2 = df.loc[has_all, ["yhat_ols", "line"]].values
        df.loc[has_all, "p_model"] = logit.predict_proba(X2)[:, 1]

    df["over_edge"]  = df["p_model"] - df["raw_implied_prob_over"]
    df["under_edge"] = (1 - df["p_model"]) - df["raw_implied_prob_under"]

    # Novig (proportional de-vig) and vig
    raw_sum = df["raw_implied_prob_over"] + df["raw_implied_prob_under"]
    df["novig_prob_over"]  = df["raw_implied_prob_over"]  / raw_sum
    df["novig_prob_under"] = df["raw_implied_prob_under"] / raw_sum
    df["vig_pp"] = (raw_sum - 1.0) * 100

    # Delta: model projection vs offered line
    df["delta"] = df["yhat_ols"] - df["line"]

    return df


# -----------------------------------------------------------------------
# Email formatting
# -----------------------------------------------------------------------

_BOOK_DISPLAY = {
    "betonlineag":    "BetOnline",
    "fanduel":        "FanDuel",
    "draftkings":     "DraftKings",
    "betmgm":         "BetMGM",
    "caesars":        "Caesars",
    "betrivers":      "BetRivers",
    "pointsbetus":    "PointsBet",
    "unibet_us":      "Unibet",
    "mybookieag":     "MyBookie",
    "bovada":         "Bovada",
    "pinnacle":       "Pinnacle",
    "bet365":         "Bet365",
    "williamhill_us": "William Hill",
    "lowvig":         "LowVig",
    "ballybet":       "Bally Bet",
    "espnbet":        "ESPN Bet",
    "fliff":          "Fliff",
    "betanysports":   "BetAnySports",
    "fanatics":       "Fanatics",
    "hardrockbet_oh": "Hard Rock Bet",
    "hardrock":       "Hard Rock Bet",
}

_SS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
_EMAIL_CSS = f"""
body{{font-family:{_SS};font-size:13px;color:#222;background:#f5f5f5;margin:0;padding:16px}}
.card{{background:#fff;border-radius:8px;padding:16px 20px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
h1{{font-size:20px;margin:0 0 4px;color:#1a1a2e}}
h2{{font-size:15px;margin:0 0 16px;font-weight:normal;color:#555}}
h3{{font-size:14px;margin:16px 0 8px;color:#1a1a2e}}
.summary-bar{{background:#1a1a2e;color:#fff;border-radius:6px;padding:10px 16px;margin-bottom:16px;font-size:14px;font-weight:bold}}
.game-hdr{{background:#2c3e50;color:#fff;padding:8px 12px;font-weight:bold;font-size:13px;border-radius:4px;margin:12px 0 4px}}
tr.game-hdr-row td{{background:#2c3e50;color:#fff;font-weight:bold;font-size:12px;padding:6px 10px;border:none}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
th{{background:#ecf0f1;padding:5px 7px;text-align:center;font-size:11px;color:#555;border:1px solid #ddd;white-space:nowrap}}
th.group{{background:#2c3e50;color:#fff;font-size:11px;text-align:center;border:1px solid #1a252f}}
td{{padding:4px 7px;border-bottom:1px solid #eee;white-space:nowrap;vertical-align:middle}}
tr.play-row{{background:#fce8e6}}
tr.track-row{{background:#fffde7}}
tr.none-row{{background:#fff}}
tr.none-row:nth-child(even){{background:#fafafa}}
.play-tag{{background:#2e7d32;color:#fff;border-radius:3px;padding:1px 5px;font-size:10px;font-weight:bold}}
.track-tag{{background:#f9a825;color:#333;border-radius:3px;padding:1px 5px;font-size:10px;font-weight:bold}}
.pos{{color:#1a7f37;font-weight:bold}}
.neg{{color:#d32f2f;font-weight:bold}}
.stat-cards{{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap}}
.stat-card{{background:#f8f9fa;border-radius:6px;padding:10px 16px;min-width:100px;text-align:center;border:1px solid #e0e0e0}}
.stat-card .val{{font-size:20px;font-weight:bold;color:#1a1a2e}}
.stat-card .lbl{{font-size:11px;color:#888;margin-top:2px}}
hr.section{{border:none;border-top:2px solid #ecf0f1;margin:24px 0}}
"""


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.1f}%"


def _fmt_f(v, dec=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{dec}f}"


def _game_time_et(commence_time: str) -> str:
    try:
        ct = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        return ct.astimezone(ET).strftime("%-I:%M %p ET")
    except Exception:
        return "TBD"


def _grouped_thead() -> str:
    """Two-row grouped <thead> for the player table."""
    # Row 1: group headers
    g = (
        '<thead>'
        '<tr>'
        '<th class="group" colspan="5">Player / Game</th>'
        '<th class="group" colspan="1">Book</th>'
        '<th class="group" colspan="2">American Odds</th>'
        '<th class="group" colspan="3">Implied</th>'
        '<th class="group" colspan="4">No-Vig</th>'
        '<th class="group" colspan="4">Model Prediction</th>'
        '<th class="group" colspan="2">Edge</th>'
        '<th class="group" colspan="7">Player Stats (Model Inputs)</th>'
        '</tr>'
        # Row 2: individual column names
        '<tr>'
        '<th>Player</th><th>Team</th><th>Opp</th><th>Time</th><th>Line</th>'
        '<th>Book</th>'
        '<th>Over</th><th>Under</th>'
        '<th>Raw Ov</th><th>Raw Un</th><th>Raw Tot</th>'
        '<th>Fair Ov</th><th>Fair Un</th><th>Fair Tot</th><th>Vig</th>'
        '<th>Proj Hits</th><th>vs Line</th><th>P(Over)</th><th>P(Under)</th>'
        '<th>Over Edge</th><th>Under Edge</th>'
        '<th>H/G Career</th><th>H/G Last 5</th><th>H/G Last 10</th><th>H/G Season</th><th>BA Career</th><th>AB/G</th>'
        '<th>Status</th>'
        '</tr>'
        '</thead>'
    )
    return g


def _player_row(r: pd.Series, status: str) -> str:
    cls = {"PLAY - UNDER": "play-row", "TRACK": "track-row"}.get(status, "none-row")

    if status == "PLAY - UNDER":
        tag = '<span class="play-tag">PLAY UNDER</span>'
    elif status == "TRACK":
        tag = '<span class="track-tag">TRACK</span>'
    else:
        tag = ""

    uedge = r.get("under_edge", np.nan)
    uedge_str = f'<span class="pos">{uedge*100:+.1f}pp</span>' if (
        not pd.isna(uedge) and uedge > 0
    ) else (f'<span class="neg">{uedge*100:+.1f}pp</span>' if not pd.isna(uedge) else "—")

    raw_tot = r.get("raw_implied_prob_over", np.nan) + r.get("raw_implied_prob_under", np.nan)

    book_display = _BOOK_DISPLAY.get(str(r.get("bookmaker", "")), str(r.get("bookmaker", "")))

    return (
        f'<tr class="{cls}">'
        f'<td>{html_module.escape(str(r.get("player_name", "")))}</td>'
        f'<td style="font-size:11px;color:#555">{html_module.escape(str(r.get("away_team", "")))}</td>'
        f'<td style="font-size:11px;color:#555">{html_module.escape(str(r.get("home_team", "")))}</td>'
        f'<td style="font-size:11px">{_game_time_et(str(r.get("commence_time", "")))}</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("line"), 1)}</td>'
        f'<td>{html_module.escape(book_display)}</td>'
        f'<td style="text-align:center">{_dec_to_american(r.get("over_price"))}</td>'
        f'<td style="text-align:center">{_dec_to_american(r.get("under_price"))}</td>'
        f'<td style="text-align:center">{_fmt_pct(r.get("raw_implied_prob_over"))}</td>'
        f'<td style="text-align:center">{_fmt_pct(r.get("raw_implied_prob_under"))}</td>'
        f'<td style="text-align:center">{_fmt_pct(raw_tot) if not pd.isna(raw_tot) else "—"}</td>'
        f'<td style="text-align:center">{_fmt_pct(r.get("novig_prob_over"))}</td>'
        f'<td style="text-align:center">{_fmt_pct(r.get("novig_prob_under"))}</td>'
        f'<td style="text-align:center">100.0%</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("vig_pp"), 1)}pp</td>'
        f'<td style="text-align:center" title="Projected hits this game">{_fmt_f(r.get("yhat_ols"), 3)}</td>'
        f'<td style="text-align:center" title="Projection minus line">{_fmt_f(r.get("delta"), 3)}</td>'
        f'<td style="text-align:center">{_fmt_pct(r.get("p_model"))}</td>'
        f'<td style="text-align:center">{_fmt_pct(1 - r.get("p_model", np.nan)) if not pd.isna(r.get("p_model", np.nan)) else "—"}</td>'
        f'<td style="text-align:center">{_fmt_edge(r.get("over_edge", np.nan))}</td>'
        f'<td style="text-align:center">{uedge_str}</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("hits_roll_career"), 3)}</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("hits_roll_L5"), 2)}</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("hits_roll_L10"), 2)}</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("hits_roll_season"), 2)}</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("ba_roll_career"), 3)}</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("ab_roll_career"), 1)}</td>'
        f'<td style="text-align:center">{tag}</td>'
        f'</tr>'
    )


def _fmt_edge(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    cls = "pos" if v > 0 else "neg"
    return f'<span class="{cls}">{v*100:+.1f}pp</span>'


_MODEL_INPUTS_INFO = [
    # (feature_col, shown_as, what_it_measures, role)
    ("min_raw_implied_prob_under", "Min Raw Under Prob", "Lowest raw P(under) offered across all books — consensus floor for the under", "OLS input (market signal)"),
    ("max_raw_implied_prob_over",  "Max Raw Over Prob",  "Highest raw P(over) offered across all books — consensus ceiling for the over", "OLS input (market signal)"),
    ("hits_roll_career",           "H/G Career",         "Career hits per game average (all prior games)", "OLS input · displayed"),
    ("hits_roll_L10",              "H/G Last 10",        "Hits per game average over last 10 games", "OLS input · displayed"),
    ("hits_roll_L20",              "H/G Last 20",        "Hits per game average over last 20 games", "OLS input"),
    ("hits_roll_season",           "H/G Season",         "Hits per game average this season", "OLS input · displayed"),
    ("ba_roll_career",             "BA Career",          "Career batting average", "OLS input · displayed"),
    ("ab_roll_career",             "AB/G",               "Career at-bats per game", "OLS input · displayed"),
    ("consensus_line",             "Consensus Line",     "Average line offered across all books for this player-game", "OLS input"),
    ("max_line",                   "Max Line",           "Highest line offered across all books for this player-game", "OLS input"),
    ("yhat_ols",                   "Proj Hits",          "OLS stage 1 output: projected hits this game", "Logistic input · displayed"),
    ("offered_line",               "Line",               "The specific line for this book row", "Logistic input · displayed"),
    ("hits_roll_L5",               "H/G Last 5",         "Hits per game average over last 5 games (display only — not in OLS)", "Displayed only"),
]


def _model_inputs_table() -> str:
    rows = "".join(
        f'<tr><td><code>{html_module.escape(feat)}</code></td>'
        f'<td>{html_module.escape(shown)}</td>'
        f'<td>{html_module.escape(desc)}</td>'
        f'<td>{html_module.escape(role)}</td></tr>'
        for feat, shown, desc, role in _MODEL_INPUTS_INFO
    )
    return (
        f'<h3 style="margin-top:24px">Model Inputs Reference</h3>'
        f'<table style="font-size:11px">'
        f'<thead><tr>'
        f'<th style="text-align:left">Feature</th>'
        f'<th style="text-align:left">Shown as</th>'
        f'<th style="text-align:left">What it measures</th>'
        f'<th style="text-align:left">Role</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table>'
    )


_N_COLS = 28  # total columns in _grouped_thead() — must match exactly


def _section1_plays(scored: pd.DataFrame, cfg: dict, gameday: str) -> str:
    strat      = cfg["strategy"]
    edge_min   = strat["edge_min"]
    edge_show  = strat["edge_show"]
    strat_line = strat["line"]

    display = scored[
        (scored["line"] == strat_line) &
        scored["p_model"].notna() &
        scored["under_price"].notna()
    ].copy()

    if display.empty:
        return '<p style="color:#888">No players scored today at line 0.5.</p>'

    plus_only = strat.get("plus_odds_only", False)

    def _status(r):
        is_plus = (not plus_only) or (r.get("under_price", 0) > 2.0)
        if is_plus and r["under_edge"] >= edge_min:
            return "PLAY - UNDER"
        if r["under_edge"] >= edge_show:
            return "TRACK"
        return ""
    display["_status"] = display.apply(_status, axis=1)

    # One row per player per game: best-edge book.
    display = (
        display
        .sort_values("under_edge", ascending=False)
        .drop_duplicates(subset=["player_name", "home_team", "away_team"])
    )

    # Sort: by game time, then plays → tracks → no-edge, then edge desc within tier.
    display["_tier_order"] = display["_status"].map({"PLAY - UNDER": 0, "TRACK": 1, "": 2})
    display = display.sort_values(
        ["commence_time", "home_team", "away_team", "_tier_order", "under_edge"],
        ascending=[True, True, True, True, False],
    )

    n_plays    = (display["_status"] == "PLAY - UNDER").sum()
    play_games = display[display["_status"] == "PLAY - UNDER"].groupby(
        ["home_team", "away_team"]
    ).ngroups

    odds_label = " · +odds only" if plus_only else ""
    summary_bar = (
        f'<div class="summary-bar">'
        f'{n_plays} play{"s" if n_plays != 1 else ""} today across '
        f'{play_games} game{"s" if play_games != 1 else ""} · '
        f'Strategy: 0.5 UNDER{odds_label} · edge≥{edge_min*100:.0f}pp · best book per player'
        f'</div>'
    )

    # --- Full-detail table: plays + tracks only ---
    qualifying = display[display["_status"].isin(["PLAY - UNDER", "TRACK"])]
    non_qual   = display[display["_status"] == ""]

    tbody_rows = ""
    for (commence_time, home, away), grp in qualifying.groupby(
        ["commence_time", "home_team", "away_team"], sort=False
    ):
        time_str   = _game_time_et(str(commence_time))
        n_play_grp = (grp["_status"] == "PLAY - UNDER").sum()
        n_total    = len(grp)
        play_label = f"{n_play_grp} play" if n_play_grp == 1 else f"{n_play_grp} plays"

        tbody_rows += (
            f'<tr class="game-hdr-row">'
            f'<td colspan="{_N_COLS}">'
            f'{time_str} &mdash; {html_module.escape(away)} @ {html_module.escape(home)}'
            f' &nbsp;·&nbsp; {play_label}'
            f' &nbsp;·&nbsp; {n_total} player{"s" if n_total != 1 else ""}'
            f'</td></tr>'
        )
        tbody_rows += "".join(_player_row(r, r["_status"]) for _, r in grp.iterrows())

    main_table = (
        f'<table>{_grouped_thead()}<tbody>{tbody_rows}</tbody></table>'
        if tbody_rows else
        '<p style="color:#888">No qualifying plays or tracks today.</p>'
    )

    # --- Compact table: non-qualifying players (no full column detail) ---
    compact_section = ""
    if not non_qual.empty:
        c_rows = ""
        for (commence_time, home, away), grp in non_qual.groupby(
            ["commence_time", "home_team", "away_team"], sort=False
        ):
            time_str = _game_time_et(str(commence_time))
            c_rows += (
                f'<tr style="background:#e8eaf6;font-weight:bold;font-size:11px">'
                f'<td colspan="6">{time_str} &mdash; {html_module.escape(away)} @ {html_module.escape(home)}'
                f' &nbsp;·&nbsp; {len(grp)} players</td></tr>'
            )
            for _, r in grp.iterrows():
                uedge = r.get("under_edge", float("nan"))
                uedge_str = f'{uedge*100:+.1f}pp' if not pd.isna(uedge) else "—"
                uedge_cls = "pos" if (not pd.isna(uedge) and uedge > 0) else "neg"
                book_display = _BOOK_DISPLAY.get(str(r.get("bookmaker", "")), str(r.get("bookmaker", "")))
                c_rows += (
                    f'<tr style="color:#555">'
                    f'<td>{html_module.escape(str(r.get("player_name", "")))}</td>'
                    f'<td style="font-size:11px">{html_module.escape(str(r.get("away_team", "")))} @ {html_module.escape(str(r.get("home_team", "")))}</td>'
                    f'<td>{html_module.escape(book_display)}</td>'
                    f'<td style="text-align:center">{_dec_to_american(r.get("under_price"))}</td>'
                    f'<td style="text-align:center">{_fmt_pct(1 - r.get("p_model", float("nan"))) if not pd.isna(r.get("p_model", float("nan"))) else "—"}</td>'
                    f'<td style="text-align:center"><span class="{uedge_cls}">{uedge_str}</span></td>'
                    f'</tr>'
                )
        compact_section = (
            f'<h3 style="margin-top:20px;color:#555;font-size:12px">'
            f'Other Players (no qualifying edge) — {len(non_qual)}</h3>'
            f'<table style="font-size:11px;width:auto">'
            f'<thead><tr>'
            f'<th style="text-align:left">Player</th>'
            f'<th style="text-align:left">Game</th>'
            f'<th style="text-align:left">Book (best)</th>'
            f'<th>Under</th><th>P(Under)</th><th>Under Edge</th>'
            f'</tr></thead>'
            f'<tbody>{c_rows}</tbody></table>'
        )

    return summary_bar + main_table + compact_section + _model_inputs_table()


def _section2_yesterday(settle_html: str | None) -> str:
    if settle_html:
        return settle_html
    return (
        '<div class="card">'
        '<h1>Yesterday\'s Results</h1>'
        '<p style="color:#888">Settlement results will appear here after the 8:30 AM settle run.</p>'
        '</div>'
    )


def _section3_alltime(cfg: dict) -> str:
    """Load production settled history from S3 and render summary cards + season table."""
    try:
        s3     = boto3.client("s3")
        obj    = s3.get_object(Bucket=cfg["data"]["s3_bucket"], Key=cfg["data"]["settled_key"])
        hist   = pd.read_parquet(BytesIO(obj["Body"].read()))
        prod   = hist[hist["result"].isin(["win", "loss", "push"])]
        n_bets = len(prod)
        n_win  = (prod["result"] == "win").sum()
        n_loss = (prod["result"] == "loss").sum()
        net    = prod["pnl"].sum() if "pnl" in prod.columns else 0.0
        roi    = net / max(n_bets, 1) * 100
        win_pct = n_win / max(n_bets, 1) * 100

        cards = (
            f'<div class="stat-cards">'
            f'<div class="stat-card"><div class="val {"pos" if net >= 0 else "neg"}">{net:+.1f}u</div><div class="lbl">All-Time P&L</div></div>'
            f'<div class="stat-card"><div class="val">{n_win}W / {n_loss}L</div><div class="lbl">Record</div></div>'
            f'<div class="stat-card"><div class="val">{win_pct:.1f}%</div><div class="lbl">Win %</div></div>'
            f'<div class="stat-card"><div class="val {"pos" if roi >= 0 else "neg"}">{roi:+.1f}%</div><div class="lbl">ROI</div></div>'
            f'<div class="stat-card"><div class="val">{n_bets}</div><div class="lbl">Total Bets</div></div>'
            f'</div>'
        )

        # Season breakdown
        if "game_date" in prod.columns and n_bets > 0:
            prod = prod.copy()
            prod["season"] = pd.to_datetime(prod["game_date"]).dt.year
            by_season = prod.groupby("season").apply(lambda g: pd.Series({
                "bets":    len(g),
                "wins":    (g["result"] == "win").sum(),
                "losses":  (g["result"] == "loss").sum(),
                "units":   g["pnl"].sum(),
                "roi_pct": g["pnl"].sum() / len(g) * 100,
            })).reset_index()
            season_rows = "".join(
                f'<tr>'
                f'<td>{int(r["season"])}</td>'
                f'<td style="text-align:center">{int(r["bets"])}</td>'
                f'<td style="text-align:center">{int(r["wins"])}W / {int(r["losses"])}L</td>'
                f'<td style="text-align:center">{r["wins"]/max(r["bets"],1)*100:.1f}%</td>'
                f'<td style="text-align:center" class="{"pos" if r["units"]>=0 else "neg"}">{r["units"]:+.1f}u</td>'
                f'<td style="text-align:center" class="{"pos" if r["roi_pct"]>=0 else "neg"}">{r["roi_pct"]:+.1f}%</td>'
                f'</tr>'
                for _, r in by_season.iterrows()
            )
            season_table = (
                f'<h3>By Season</h3>'
                f'<table><thead><tr>'
                f'<th>Season</th><th>Bets</th><th>Record</th><th>Win %</th><th>Units</th><th>ROI</th>'
                f'</tr></thead><tbody>{season_rows}</tbody></table>'
            )
        else:
            season_table = ""

    except Exception:
        cards = (
            '<div class="stat-cards">'
            '<div class="stat-card"><div class="val">+0.0u</div><div class="lbl">All-Time P&L</div></div>'
            '<div class="stat-card"><div class="val">0W / 0L</div><div class="lbl">Record</div></div>'
            '<div class="stat-card"><div class="val">—</div><div class="lbl">Win %</div></div>'
            '<div class="stat-card"><div class="val">—</div><div class="lbl">ROI</div></div>'
            '</div>'
        )
        season_table = ""

    return (
        f'<div class="card">'
        f'<h1>All-Time Production Results</h1>'
        f'<h2>Live bets since pipeline launched — not backtest</h2>'
        f'{cards}{season_table}'
        f'</div>'
    )


def _section4_backtest(cfg: dict) -> str:
    bt     = cfg.get("backtest", {})
    n_bets = bt.get("n_bets",    "—")
    hit    = bt.get("hit_rate",  None)
    units  = bt.get("net_units", "—")
    roi    = bt.get("roi_pct",   "—")
    mdd    = bt.get("max_dd",    "—")
    nmdd   = bt.get("net_mdd",   "—")
    period = bt.get("period",    "—")

    hit_str = f"{hit*100:.1f}%" if hit else "—"
    strat   = cfg.get("strategy", {})

    oos_row = (
        f'<tr>'
        f'<td>0.5 UNDER · edge≥{strat.get("edge_min",0)*100:.0f}pp · <strong>OOF</strong></td>'
        f'<td>{period}</td>'
        f'<td style="text-align:center">{n_bets:,}</td>'
        f'<td style="text-align:center">{hit_str}</td>'
        f'<td style="text-align:center" class="pos">+{units}u</td>'
        f'<td style="text-align:center" class="pos">+{roi}%</td>'
        f'<td style="text-align:center">-{mdd}u</td>'
        f'<td style="text-align:center">{nmdd}x</td>'
        f'</tr>'
    )

    is_cfg  = bt.get("in_sample", {})
    is_row  = ""
    if is_cfg:
        is_hit  = is_cfg.get("hit_rate", None)
        is_row = (
            f'<tr style="opacity:0.7">'
            f'<td>0.5 UNDER · edge≥{strat.get("edge_min",0)*100:.0f}pp · <em>IS upper bound</em></td>'
            f'<td>{is_cfg.get("period", "—")}</td>'
            f'<td style="text-align:center">{is_cfg.get("n_bets","—"):,}</td>'
            f'<td style="text-align:center">{f"{is_hit*100:.1f}%" if is_hit else "—"}</td>'
            f'<td style="text-align:center" class="pos">+{is_cfg.get("net_units","—")}u</td>'
            f'<td style="text-align:center" class="pos">+{is_cfg.get("roi_pct","—")}%</td>'
            f'<td style="text-align:center">—</td>'
            f'<td style="text-align:center">—</td>'
            f'</tr>'
        )

    summary_row = oos_row + is_row

    by_season = bt.get("out_of_sample_by_season", [])
    if by_season:
        season_rows = "".join(
            f'<tr>'
            f'<td>{s["season"]}</td>'
            f'<td style="text-align:center">{s["n_bets"]:,}</td>'
            f'<td style="text-align:center">{s["hit_rate"]*100:.1f}%</td>'
            f'<td style="text-align:center" class="{"pos" if s["net_units"] >= 0 else "neg"}">{s["net_units"]:+.1f}u</td>'
            f'<td style="text-align:center" class="{"pos" if s["roi_pct"] >= 0 else "neg"}">{s["roi_pct"]:+.1f}%</td>'
            f'</tr>'
            for s in by_season
        )
        season_table = (
            f'<h3>Per-Season (OOF)</h3>'
            f'<table style="width:auto"><thead><tr>'
            f'<th>Season</th><th>Bets</th><th>Hit %</th><th>Net Units</th><th>ROI</th>'
            f'</tr></thead><tbody>{season_rows}</tbody></table>'
        )
    else:
        season_table = ""

    return (
        f'<div class="card">'
        f'<h1>Historical Backtest — Research Phase (OOS)</h1>'
        f'<h2>Frozen at research time · OOF cross-validation · not updated during season</h2>'
        f'<table style="width:auto"><thead><tr>'
        f'<th>Strategy</th><th>Period</th><th>Bets</th><th>Hit %</th><th>Net Units</th><th>ROI</th><th>Max DD</th><th>Net/MDD</th>'
        f'</tr></thead><tbody>{summary_row}</tbody></table>'
        f'{season_table}'
        f'</div>'
    )


def build_email_html(
    scored: pd.DataFrame,
    cfg: dict,
    gameday: str,
    n_plays: int,
    settle_html: str | None = None,
) -> str:
    s1 = (
        f'<div class="card">'
        f'<h1>MLB Batter Hits — {gameday}</h1>'
        f'<h2>0.5 UNDER · edge≥{cfg["strategy"]["edge_min"]*100:.0f}pp</h2>'
        f'{_section1_plays(scored, cfg, gameday)}'
        f'</div>'
    )
    s2 = _section2_yesterday(settle_html)
    s3 = _section3_alltime(cfg)
    s4 = _section4_backtest(cfg)

    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<style>{_EMAIL_CSS}</style></head><body>'
        f'{s1}<hr class="section">{s2}<hr class="section">{s3}<hr class="section">{s4}'
        f'</body></html>'
    )


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
    print(f"  Email sent: {subject[:80]}")


def _send_sns(subject: str, n_plays: int) -> None:
    if not SNS_TOPIC_ARN:
        return
    boto3.client("sns", region_name="us-east-2").publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=subject[:100],
        Message=f"{n_plays} plays today. Check email.",
    )


def _save_to_s3(df: pd.DataFrame, cfg: dict, gameday: str) -> None:
    key = f"{cfg['data']['daily_prefix']}/{gameday}/recommendations.csv"
    s3 = boto3.client("s3")
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=cfg["data"]["s3_bucket"], Key=key, Body=buf.read())
    print(f"  Saved {len(df)} recs → s3://{cfg['data']['s3_bucket']}/{key}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main(gameday: str | None = None, output: str | None = None, no_email: bool = False) -> dict:
    cfg = _load_config()
    if not gameday:
        gameday = datetime.now(ET).strftime("%Y-%m-%d")
    print(f"\nMLB Batter Hits pipeline | {gameday}")

    # Load models from S3
    print("Loading models from S3...")
    s3 = boto3.client("s3")
    bucket = cfg["data"]["s3_bucket"]

    ols   = joblib.load(BytesIO(s3.get_object(Bucket=bucket, Key=cfg["data"]["ols_model_key"])["Body"].read()))
    logit = joblib.load(BytesIO(s3.get_object(Bucket=bucket, Key=cfg["data"]["logit_model_key"])["Body"].read()))
    print("  OLS + Logistic models loaded")

    # Load spine
    print("Loading spine from S3...")
    spine = load_spine_from_s3(cfg)
    latest_features = get_latest_player_features(spine)
    print(f"  {len(latest_features):,} players with features")

    # Fetch today's market
    print(f"Fetching market data for {gameday}...")
    events = fetch_events(cfg, gameday)
    if events:
        market_df = build_market_df(events, cfg, gameday)
    else:
        market_df = _load_market_from_s3_cache(cfg, gameday)

    if market_df.empty:
        print("  No market data available — exiting")
        result = {"n_play_bets": 0, "n_track_bets": 0, "gameday": gameday}
        if output:
            Path(output).write_text(json.dumps(result))
        return result

    print(f"  {len(market_df):,} market rows, {market_df['player_name'].nunique()} players")

    market_df = build_market_features(market_df)

    # Score
    print("Scoring...")
    scored = score(market_df, latest_features, ols, logit, cfg)
    has_pmodel = scored["p_model"].notna().sum()
    print(f"  {has_pmodel:,} rows with p_model ({has_pmodel/len(scored)*100:.1f}%)")

    # Filter plays and tracks
    strat     = cfg["strategy"]
    plus_only = strat.get("plus_odds_only", False)
    plus_mask = (scored["under_price"] > 2.0) if plus_only else pd.Series(True, index=scored.index)

    plays = scored[
        (scored["line"] == strat["line"]) &
        plus_mask &
        scored["under_edge"].notna() &
        (scored["under_edge"] >= strat["edge_min"]) &
        scored["under_price"].notna()
    ]
    tracks = scored[
        (scored["line"] == strat["line"]) &
        scored["under_edge"].notna() &
        (scored["under_edge"] >= strat["edge_show"]) &
        (scored["under_edge"] < strat["edge_min"]) &
        scored["under_price"].notna()
    ]

    n_plays  = len(plays)
    n_tracks = len(tracks)
    print(f"  PLAYS: {n_plays}  TRACKS: {n_tracks}")

    # Email
    html_body = build_email_html(scored, cfg, gameday, n_plays)
    play_word = "play" if n_plays == 1 else "plays"
    subject = f"MLB Batter Hits — {n_plays} {play_word} today — {gameday}"

    if not no_email:
        _send_ses(subject, html_body)
        _send_sns(subject, n_plays)

    # Save to S3
    if n_plays + n_tracks > 0:
        save_df = pd.concat([plays, tracks]).copy()
        save_df["tier"] = "play"
        save_df.loc[save_df["player_key"].isin(tracks["player_key"]), "tier"] = "track"
        _save_to_s3(save_df, cfg, gameday)

    result = {
        "n_play_bets":  n_plays,
        "n_track_bets": n_tracks,
        "gameday":      gameday,
        "subject":      subject,
        "html":         html_body,
    }
    if output:
        Path(output).write_text(json.dumps(result))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-email", action="store_true")
    args = parser.parse_args()
    main(gameday=args.gameday, output=args.output, no_email=args.no_email)
