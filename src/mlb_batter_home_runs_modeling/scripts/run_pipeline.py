"""
Live gameday pipeline for MLB Batter Home Runs props.

Strategy: 0.5 OVER · plus_odds_only (over_price >= 2.0) · edge >= 10pp
OOS: 173 bets · 38.2% hit · +37.4u · +21.6% ROI (shrinkage=0, all books)

For each batter with a batter_home_runs prop on the given gameday:
  1. Fetch live events + props from The Odds API (batter_home_runs)
  2. Load rolling features from the spine (S3) — most recent row per player
  3. Compute market features (min/max implied probs)
  4. Score with saved logistic regression → p_model_over
  5. Compute edge_over = p_model_over - (1/over_price)  [raw, vig-inclusive]
  6. Filter to 0.5 OVER, plus-odds, edge >= 10pp
  7. Send SES HTML email + SNS notification
  8. Save recommendations CSV to S3

Model features:
  hr_roll_L5, hr_roll_L10, hr_roll_L20, hr_roll_career, ab_roll_career,
  opp_hr_rate_career, min_raw_implied_prob_under, max_raw_implied_prob_under, is_home

S3 paths read:
  s3://the-odds-api-mt/mlb/batter_home_runs_model/spine/mlb_batter_hr_spine.parquet
  s3://the-odds-api-mt/mlb/batter_home_runs_model/model/mlb_batter_hr_model.joblib

S3 paths written:
  s3://the-odds-api-mt/mlb/batter_home_runs_model/daily_runs/{gameday}/recommendations.csv

Usage:
  python src/mlb_batter_home_runs_modeling/scripts/run_pipeline.py
  python src/mlb_batter_home_runs_modeling/scripts/run_pipeline.py --gameday 2026-03-27
  python src/mlb_batter_home_runs_modeling/scripts/run_pipeline.py --output /tmp/hr_out.json
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

S3_BUCKET     = "the-odds-api-mt"
SPINE_KEY     = "mlb/batter_home_runs_model/spine/mlb_batter_hr_spine.parquet"
MODEL_KEY     = "mlb/batter_home_runs_model/model/mlb_batter_hr_model.joblib"
DAILY_PREFIX  = "mlb/batter_home_runs_model/daily_runs"
SETTLED_KEY   = "mlb/batter_home_runs_model/settled/mlb_batter_hr_settled_bets.parquet"

STRATEGY_LINE     = 0.5
STRATEGY_EDGE_MIN = 0.10
STRATEGY_EDGE_SHOW = 0.05
STRATEGY_ODDS_MIN = 2.0  # plus-odds only

MODEL_FEATURES = [
    "hr_roll_L5", "hr_roll_L10", "hr_roll_L20", "hr_roll_career",
    "ab_roll_career", "opp_hr_rate_career",
    "min_raw_implied_prob_under", "max_raw_implied_prob_under",
    "is_home",
]

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
    "williamhill_us": "William Hill",
    "lowvig":         "LowVig",
    "ballybet":       "Bally Bet",
    "espnbet":        "ESPN Bet",
    "fliff":          "Fliff",
    "betanysports":   "BetAnySports",
    "fanatics":       "Fanatics",
    "hardrockbet":    "Hard Rock Bet",
    "hardrockbet_oh": "Hard Rock Bet",
    "betparx":        "BetParx",
}


# ── Utilities ─────────────────────────────────────────────────────────────────

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


def dec_to_american(d) -> str:
    if not d or pd.isna(d):
        return "N/A"
    if d >= 2.0:
        return f"+{int(round((d - 1) * 100))}"
    return f"-{int(round(100 / (d - 1)))}"


def game_time_et(commence_time: str) -> str:
    try:
        ct = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        return ct.astimezone(ET).strftime("%-I:%M %p ET")
    except Exception:
        return "TBD"


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_events(gameday: str) -> list[dict]:
    if not ODDS_API_KEY:
        print("  ODDS_API_KEY not set")
        return []
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT}/events",
        params={"apiKey": ODDS_API_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    events = [e for e in resp.json() if e.get("commence_time", "").startswith(gameday)]
    print(f"  {len(events)} games on {gameday}")
    return events


def fetch_props_for_event(event_id: str) -> list[dict]:
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds",
        params={
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "batter_home_runs",
            "oddsFormat": "decimal",
        },
        timeout=30,
    )
    resp.raise_for_status()
    time.sleep(SLEEP_S)
    return resp.json().get("bookmakers", [])


def build_market_df(events: list[dict], gameday: str) -> pd.DataFrame:
    all_rows = []
    for ev in events:
        if not ODDS_API_KEY:
            break
        for bk in fetch_props_for_event(ev["id"]):
            for market in bk.get("markets", []):
                for outcome in market.get("outcomes", []):
                    all_rows.append({
                        "event_id":      ev["id"],
                        "game_date":     gameday,
                        "home_team":     ev.get("home_team", ""),
                        "away_team":     ev.get("away_team", ""),
                        "commence_time": ev.get("commence_time", ""),
                        "bookmaker":     bk["key"],
                        "player_name":   outcome.get("description", ""),
                        "side":          outcome["name"],
                        "line":          outcome.get("point"),
                        "price":         outcome.get("price"),
                    })

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df[df["side"].isin(["Over", "Under"])].copy()
    df["line"]  = pd.to_numeric(df["line"],  errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    pivot = df.pivot_table(
        index=["event_id", "game_date", "home_team", "away_team", "commence_time",
               "bookmaker", "player_name", "line"],
        columns="side", values="price", aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"Over": "over_price", "Under": "under_price"})
    pivot["player_key"] = pivot["player_name"].apply(normalize_name)
    return pivot


# ── Spine + feature loading ────────────────────────────────────────────────────

_SPINE_COLS = [
    "player_key", "player_name", "game_date", "home_team", "away_team", "is_home",
    "hr_roll_L5", "hr_roll_L10", "hr_roll_L20", "hr_roll_career",
    "ab_roll_career", "opp_hr_rate_career", "hr_actual",
]


def load_spine_from_s3() -> pd.DataFrame:
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)
    buf = BytesIO(obj["Body"].read())
    import pyarrow.parquet as pq
    available = pq.read_schema(buf).names
    buf.seek(0)
    cols = [c for c in _SPINE_COLS if c in available]
    spine = pd.read_parquet(buf, columns=cols)
    # De-duplicate: one row per player-game
    spine = spine.drop_duplicates(subset=["player_key", "game_date"])
    print(f"  Spine: {len(spine):,} player-game rows × {len(cols)} cols")
    return spine


def get_latest_player_features(spine: pd.DataFrame) -> pd.DataFrame:
    """Most recent settled row per player."""
    avail = [c for c in _SPINE_COLS if c in spine.columns]
    settled = spine[spine.get("hr_actual", pd.Series(index=spine.index)).notna()][avail]
    return (
        settled.sort_values("game_date")
        .drop_duplicates(subset=["player_key"], keep="last")
        .reset_index(drop=True)
    )


def get_latest_opp_hr_rates(spine: pd.DataFrame) -> dict[str, float]:
    """Latest opp_hr_rate_career per pitching team (team name as key)."""
    if "opp_hr_rate_career" not in spine.columns:
        return {}
    sp = spine.copy()
    # opponent is the pitching team: away_team when batter is home, home_team when away
    sp["opponent_team"] = np.where(
        sp["is_home"].fillna(0) == 1,
        sp["away_team"].fillna(""),
        sp["home_team"].fillna(""),
    )
    latest = (
        sp[["opponent_team", "game_date", "opp_hr_rate_career"]]
        .dropna(subset=["opp_hr_rate_career", "opponent_team"])
        .query("opponent_team != ''")
        .sort_values("game_date")
        .drop_duplicates(subset=["opponent_team"], keep="last")
        .set_index("opponent_team")["opp_hr_rate_career"]
        .to_dict()
    )
    return latest


def get_player_team_map(spine: pd.DataFrame) -> dict[str, str]:
    """Latest team per player_key (team = home_team if is_home else away_team)."""
    sp = spine.copy()
    sp["player_team"] = np.where(
        sp["is_home"].fillna(0) == 1,
        sp["home_team"].fillna(""),
        sp["away_team"].fillna(""),
    )
    return (
        sp[["player_key", "game_date", "player_team"]]
        .dropna(subset=["player_team"])
        .query("player_team != ''")
        .sort_values("game_date")
        .drop_duplicates(subset=["player_key"], keep="last")
        .set_index("player_key")["player_team"]
        .to_dict()
    )


# ── Scoring ────────────────────────────────────────────────────────────────────

def build_features(
    market_df: pd.DataFrame,
    latest_features: pd.DataFrame,
    opp_hr_rates: dict[str, float],
    player_team_map: dict[str, str],
) -> pd.DataFrame:
    """Merge rolling features + compute market features + determine is_home."""
    df = market_df.copy()

    # Merge rolling player features
    roll_cols = ["player_key", "hr_roll_L5", "hr_roll_L10", "hr_roll_L20",
                 "hr_roll_career", "ab_roll_career"]
    avail = [c for c in roll_cols if c in latest_features.columns]
    df = df.merge(latest_features[avail], on="player_key", how="left")

    # Determine is_home and opponent per player-event
    df["player_team"] = df["player_key"].map(player_team_map)
    df["is_home"] = np.where(
        df["player_team"] == df["home_team"], 1.0,
        np.where(df["player_team"] == df["away_team"], 0.0, np.nan),
    )
    df["opponent_team"] = np.where(
        df["is_home"] == 1.0, df["away_team"],
        np.where(df["is_home"] == 0.0, df["home_team"], None)
    )
    df["opp_hr_rate_career"] = df["opponent_team"].map(opp_hr_rates)

    # Market features at player-game level (book-invariant)
    df["raw_implied_prob_over"]  = 1.0 / df["over_price"]
    df["raw_implied_prob_under"] = 1.0 / df["under_price"]

    pg = df.groupby(["player_key", "event_id"]).agg(
        min_raw_implied_prob_over=("raw_implied_prob_over",  "min"),
        max_raw_implied_prob_over=("raw_implied_prob_over",  "max"),
        min_raw_implied_prob_under=("raw_implied_prob_under", "min"),
        max_raw_implied_prob_under=("raw_implied_prob_under", "max"),
        min_line=("line", "min"),
        max_line=("line", "max"),
        consensus_line=("line", "mean"),
    ).reset_index()

    df = df.merge(pg, on=["player_key", "event_id"], how="left", suffixes=("", "_pg"))
    for col in ["min_raw_implied_prob_over", "max_raw_implied_prob_over",
                "min_raw_implied_prob_under", "max_raw_implied_prob_under",
                "min_line", "max_line", "consensus_line"]:
        pg_col = col + "_pg"
        if pg_col in df.columns:
            df[col] = df[pg_col].fillna(df.get(col, np.nan))
            df.drop(columns=[pg_col], inplace=True)

    return df


def score(df: pd.DataFrame, model) -> pd.DataFrame:
    """Apply logistic regression → p_model_over; compute edges."""
    df = df.copy()
    avail = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[avail].values.astype(float)
    has_all = ~np.any(np.isnan(X), axis=1)

    df["p_model_over"] = np.nan
    if has_all.any():
        df.loc[has_all, "p_model_over"] = model.predict_proba(X[has_all])[:, 1]

        # Verify book-invariance
        scored = df[df["p_model_over"].notna()]
        yhat_spread = scored.groupby(["player_key", "event_id"])["p_model_over"].agg(
            lambda x: x.max() - x.min()
        )
        if yhat_spread.max() >= 1e-8:
            bad = yhat_spread[yhat_spread >= 1e-8].reset_index()
            raise RuntimeError(
                f"p_model_over is NOT book-invariant for {len(bad)} groups — "
                f"a per-book feature entered the model. Do NOT send email.\n{bad.head(3)}"
            )

    df["edge_over"]  = df["p_model_over"] - df["raw_implied_prob_over"]
    df["edge_under"] = (1 - df["p_model_over"]) - df["raw_implied_prob_under"]

    raw_sum = df["raw_implied_prob_over"].fillna(0) + df["raw_implied_prob_under"].fillna(0)
    df["novig_prob_over"]  = (df["raw_implied_prob_over"]  / raw_sum).where(raw_sum > 0)
    df["novig_prob_under"] = (df["raw_implied_prob_under"] / raw_sum).where(raw_sum > 0)
    df["vig_pp"] = (raw_sum - 1.0) * 100

    return df


# ── Email HTML ─────────────────────────────────────────────────────────────────

_SS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
_CSS = f"""
body{{font-family:{_SS};font-size:13px;color:#222;background:#f5f5f5;margin:0;padding:16px}}
.card{{background:#fff;border-radius:8px;padding:16px 20px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
h1{{font-size:20px;margin:0 0 4px;color:#1a1a2e}}
h2{{font-size:15px;margin:0 0 16px;font-weight:normal;color:#555}}
h3{{font-size:14px;margin:16px 0 8px;color:#1a1a2e}}
.summary-bar{{background:#1a1a2e;color:#fff;border-radius:6px;padding:10px 16px;margin-bottom:16px;font-size:14px;font-weight:bold}}
tr.game-hdr-row td{{background:#2c3e50;color:#fff;font-weight:bold;font-size:12px;padding:6px 10px;border:none}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
th{{background:#ecf0f1;padding:5px 7px;text-align:center;font-size:11px;color:#555;border:1px solid #ddd;white-space:nowrap}}
th.group{{background:#2c3e50;color:#fff;font-size:11px;text-align:center;border:1px solid #1a252f}}
td{{padding:4px 7px;border-bottom:1px solid #eee;white-space:nowrap;vertical-align:middle}}
tr.play-row{{background:#e8f5e9}}
tr.track-row{{background:#fffde7}}
.play-tag{{background:#2e7d32;color:#fff;border-radius:3px;padding:1px 5px;font-size:10px;font-weight:bold}}
.track-tag{{background:#f9a825;color:#333;border-radius:3px;padding:1px 5px;font-size:10px;font-weight:bold}}
.pos{{color:#1a7f37;font-weight:bold}}
.neg{{color:#d32f2f;font-weight:bold}}
.stat-cards{{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap}}
.stat-card{{background:#f8f9fa;border-radius:6px;padding:10px 16px;min-width:100px;text-align:center;border:1px solid #e0e0e0}}
.stat-card .val{{font-size:20px;font-weight:bold;color:#1a1a2e}}
.stat-card .lbl{{font-size:11px;color:#888;margin-top:2px}}
"""

_N_COLS = 22


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.1f}%"


def _fmt_f(v, dec=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{dec}f}"


def _fmt_edge(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    cls = "pos" if v > 0 else "neg"
    return f'<span class="{cls}">{v*100:+.1f}pp</span>'


def _thead() -> str:
    return (
        '<thead>'
        '<tr>'
        '<th class="group" colspan="5">Player / Game</th>'
        '<th class="group" colspan="1">Book</th>'
        '<th class="group" colspan="2">American Odds</th>'
        '<th class="group" colspan="2">Implied</th>'
        '<th class="group" colspan="2">No-Vig</th>'
        '<th class="group" colspan="2">Model</th>'
        '<th class="group" colspan="2">Edge</th>'
        '<th class="group" colspan="5">Stats (Model Inputs)</th>'
        '<th class="group" colspan="1">Status</th>'
        '</tr>'
        '<tr>'
        '<th>Player</th><th>Team</th><th>Opp</th><th>Time</th><th>Line</th>'
        '<th>Book</th>'
        '<th>Over</th><th>Under</th>'
        '<th>Raw Ov%</th><th>Raw Un%</th>'
        '<th>NV Ov%</th><th>NV Un%</th>'
        '<th>P(HR)</th><th>P(0 HR)</th>'
        '<th>Over Edge</th><th>Under Edge</th>'
        '<th>HR/G L5</th><th>HR/G L20</th><th>HR/G Career</th><th>AB/G</th><th>Opp HR/G</th>'
        '<th>Status</th>'
        '</tr>'
        '</thead>'
    )


def _player_row(r: pd.Series, status: str) -> str:
    cls = {"PLAY": "play-row", "TRACK": "track-row"}.get(status, "")
    tag = ({'PLAY': '<span class="play-tag">PLAY OVER</span>',
             'TRACK': '<span class="track-tag">TRACK</span>'}.get(status, ""))
    book_disp = _BOOK_DISPLAY.get(str(r.get("bookmaker", "")), str(r.get("bookmaker", "")))
    p_model = r.get("p_model_over", np.nan)
    return (
        f'<tr class="{cls}">'
        f'<td>{html_module.escape(str(r.get("player_name", r.get("player_key", ""))))}</td>'
        f'<td style="font-size:11px">{html_module.escape(str(r.get("player_team", "")))}</td>'
        f'<td style="font-size:11px">{html_module.escape(str(r.get("opponent_team", "")))}</td>'
        f'<td style="font-size:11px">{game_time_et(str(r.get("commence_time", "")))}</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("line"), 1)}</td>'
        f'<td>{html_module.escape(book_disp)}</td>'
        f'<td style="text-align:center">{dec_to_american(r.get("over_price"))}</td>'
        f'<td style="text-align:center">{dec_to_american(r.get("under_price"))}</td>'
        f'<td style="text-align:center">{_fmt_pct(r.get("raw_implied_prob_over"))}</td>'
        f'<td style="text-align:center">{_fmt_pct(r.get("raw_implied_prob_under"))}</td>'
        f'<td style="text-align:center">{_fmt_pct(r.get("novig_prob_over"))}</td>'
        f'<td style="text-align:center">{_fmt_pct(r.get("novig_prob_under"))}</td>'
        f'<td style="text-align:center">{_fmt_pct(p_model)}</td>'
        f'<td style="text-align:center">{_fmt_pct(1 - p_model) if not pd.isna(p_model) else "—"}</td>'
        f'<td style="text-align:center">{_fmt_edge(r.get("edge_over"))}</td>'
        f'<td style="text-align:center">{_fmt_edge(r.get("edge_under"))}</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("hr_roll_L5"), 3)}</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("hr_roll_L20"), 3)}</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("hr_roll_career"), 3)}</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("ab_roll_career"), 1)}</td>'
        f'<td style="text-align:center">{_fmt_f(r.get("opp_hr_rate_career"), 3)}</td>'
        f'<td style="text-align:center">{tag}</td>'
        f'</tr>'
    )


def build_email_html(scored: pd.DataFrame, gameday: str, n_plays: int) -> str:
    display = scored[
        (scored["line"] == STRATEGY_LINE) & scored["p_model_over"].notna()
    ].copy()

    def _status(r):
        if (r.get("over_price", 0) >= STRATEGY_ODDS_MIN
                and r.get("edge_over", -99) >= STRATEGY_EDGE_MIN
                and not pd.isna(r.get("over_price"))):
            return "PLAY"
        if r.get("edge_over", -99) >= STRATEGY_EDGE_SHOW:
            return "TRACK"
        return ""

    display["_status"] = display.apply(_status, axis=1)
    display["_tier"] = display["_status"].map({"PLAY": 0, "TRACK": 1, "": 2})
    display = display.sort_values(
        ["commence_time", "home_team", "_tier", "edge_over"],
        ascending=[True, True, True, False],
    )

    n_plays_actual = (display["_status"] == "PLAY").sum()
    play_games = display[display["_status"] == "PLAY"].groupby("event_id").ngroups

    summary = (
        f'<div class="summary-bar">'
        f'{n_plays_actual} play{"s" if n_plays_actual != 1 else ""} today'
        f' · Strategy: 0.5 OVER · edge≥10pp · plus-odds'
        f'</div>'
    )

    qualifying = display[display["_status"].isin(["PLAY", "TRACK"])]
    tbody = ""
    for eid, grp in qualifying.groupby("event_id", sort=False):
        home = grp["home_team"].iloc[0]
        away = grp["away_team"].iloc[0]
        ct   = grp["commence_time"].iloc[0]
        n_p  = (grp["_status"] == "PLAY").sum()
        tbody += (
            f'<tr class="game-hdr-row">'
            f'<td colspan="{_N_COLS}">'
            f'{game_time_et(str(ct))} &mdash; {html_module.escape(away)} @ {html_module.escape(home)}'
            f' &nbsp;·&nbsp; {n_p} play{"s" if n_p != 1 else ""}'
            f'</td></tr>'
        )
        tbody += "".join(_player_row(r, r["_status"]) for _, r in grp.iterrows())

    main_table = (
        f'<table>{_thead()}<tbody>{tbody}</tbody></table>' if tbody
        else '<p style="color:#888">No qualifying plays or tracks today.</p>'
    )

    # Compact non-qualifying players
    non_q = display[display["_status"] == ""]
    compact = ""
    if not non_q.empty:
        c_rows = "".join(
            f'<tr><td>{html_module.escape(str(r.get("player_name", r.get("player_key","")))) }</td>'
            f'<td>{html_module.escape(str(r.get("away_team","")))} @ {html_module.escape(str(r.get("home_team","")))}</td>'
            f'<td>{_BOOK_DISPLAY.get(str(r.get("bookmaker","")), str(r.get("bookmaker","")))}</td>'
            f'<td>{dec_to_american(r.get("over_price"))}</td>'
            f'<td>{_fmt_pct(r.get("p_model_over"))}</td>'
            f'<td>{_fmt_edge(r.get("edge_over"))}</td></tr>'
            for _, r in non_q.iterrows()
        )
        compact = (
            f'<h3 style="color:#555;font-size:12px">Other Players (no qualifying edge) — {len(non_q)}</h3>'
            f'<table style="font-size:11px;width:auto"><thead><tr>'
            f'<th>Player</th><th>Game</th><th>Book</th><th>Over $</th><th>P(HR)</th><th>Over Edge</th>'
            f'</tr></thead><tbody>{c_rows}</tbody></table>'
        )

    s1 = (
        f'<div class="card"><h1>MLB Batter Home Runs — {gameday}</h1>'
        f'<h2>0.5 OVER · edge≥10pp · plus-odds only</h2>'
        f'{summary}{main_table}{compact}</div>'
    )
    s2 = _section_alltime()
    s3 = _section_backtest()
    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<style>{_CSS}</style></head><body>'
        f'{s1}<hr style="border:none;border-top:2px solid #ecf0f1;margin:24px 0">'
        f'{s2}<hr style="border:none;border-top:2px solid #ecf0f1;margin:24px 0">'
        f'{s3}</body></html>'
    )


def _section_alltime() -> str:
    try:
        s3   = boto3.client("s3")
        obj  = s3.get_object(Bucket=S3_BUCKET, Key=SETTLED_KEY)
        hist = pd.read_parquet(BytesIO(obj["Body"].read()))
        prod = hist[hist["result"].isin(["win", "loss", "push"])]
        n    = len(prod)
        wins = (prod["result"] == "win").sum()
        net  = prod["pnl"].sum() if "pnl" in prod.columns else 0.0
        roi  = net / max(n, 1) * 100
        wp   = wins / max(n, 1) * 100
        cards = (
            f'<div class="stat-cards">'
            f'<div class="stat-card"><div class="val {"pos" if net >= 0 else "neg"}">{net:+.1f}u</div><div class="lbl">All-Time P&L</div></div>'
            f'<div class="stat-card"><div class="val">{wins}W / {n-wins}L</div><div class="lbl">Record</div></div>'
            f'<div class="stat-card"><div class="val">{wp:.1f}%</div><div class="lbl">Win %</div></div>'
            f'<div class="stat-card"><div class="val {"pos" if roi >= 0 else "neg"}">{roi:+.1f}%</div><div class="lbl">ROI</div></div>'
            f'</div>'
        )
    except Exception:
        cards = '<p style="color:#888">No settled history yet.</p>'
    return f'<div class="card"><h1>All-Time Production Results</h1><h2>Live bets since launch</h2>{cards}</div>'


def _section_backtest() -> str:
    return (
        f'<div class="card"><h1>Historical Backtest (OOS)</h1>'
        f'<h2>0.5 OVER · edge≥10pp · plus-odds · OOF cross-validation</h2>'
        f'<div class="stat-cards">'
        f'<div class="stat-card"><div class="val pos">+37.4u</div><div class="lbl">Net Units</div></div>'
        f'<div class="stat-card"><div class="val">173</div><div class="lbl">Bets</div></div>'
        f'<div class="stat-card"><div class="val">38.2%</div><div class="lbl">Win %</div></div>'
        f'<div class="stat-card"><div class="val pos">+21.6%</div><div class="lbl">ROI</div></div>'
        f'<div class="stat-card"><div class="val neg">-25.4u</div><div class="lbl">Max DD</div></div>'
        f'</div></div>'
    )


# ── Notifications + storage ────────────────────────────────────────────────────

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
        Message=f"{n_plays} HR plays today. Check email.",
    )


def _save_to_s3(df: pd.DataFrame, gameday: str) -> None:
    key = f"{DAILY_PREFIX}/{gameday}/recommendations.csv"
    s3  = boto3.client("s3")
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.read())
    print(f"  Saved {len(df)} recs → s3://{S3_BUCKET}/{key}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(gameday: str | None = None, output: str | None = None) -> dict:
    if not gameday:
        gameday = datetime.now(ET).strftime("%Y-%m-%d")
    print(f"\nMLB Batter Home Runs pipeline | {gameday}")

    # Load model
    print("Loading model from S3...")
    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
    model = joblib.load(BytesIO(obj["Body"].read()))
    print("  Model loaded")

    # Load spine
    print("Loading spine from S3...")
    spine = load_spine_from_s3()
    latest_features = get_latest_player_features(spine)
    opp_hr_rates    = get_latest_opp_hr_rates(spine)
    player_team_map = get_player_team_map(spine)
    print(f"  {len(latest_features):,} players with features")
    print(f"  {len(opp_hr_rates)} teams with opp_hr_rate_career")

    # Fetch market
    print(f"Fetching market for {gameday}...")
    events    = fetch_events(gameday)
    market_df = build_market_df(events, gameday) if events else pd.DataFrame()

    if market_df.empty:
        print("  No market data — exiting")
        result = {"n_play_bets": 0, "n_track_bets": 0, "gameday": gameday}
        if output:
            Path(output).write_text(json.dumps(result))
        return result

    print(f"  {len(market_df):,} market rows, {market_df['player_name'].nunique()} players")

    # Build features + score
    print("Building features...")
    df = build_features(market_df, latest_features, opp_hr_rates, player_team_map)
    match_rate = df["hr_roll_career"].notna().mean()
    print(f"  Feature match rate: {match_rate:.1%}")

    print("Scoring...")
    scored = score(df, model)
    n_scored = scored["p_model_over"].notna().sum()
    print(f"  {n_scored:,} rows with p_model ({n_scored/len(scored)*100:.1f}%)")

    # Qualify plays and tracks
    plays = scored[
        (scored["line"] == STRATEGY_LINE) &
        (scored["over_price"] >= STRATEGY_ODDS_MIN) &
        scored["edge_over"].notna() &
        (scored["edge_over"] >= STRATEGY_EDGE_MIN) &
        scored["over_price"].notna()
    ]
    tracks = scored[
        (scored["line"] == STRATEGY_LINE) &
        scored["edge_over"].notna() &
        (scored["edge_over"] >= STRATEGY_EDGE_SHOW) &
        (scored["edge_over"] < STRATEGY_EDGE_MIN)
    ]

    n_plays  = len(plays)
    n_tracks = len(tracks)
    print(f"  PLAYS: {n_plays}  TRACKS: {n_tracks}")

    # Email
    html_body  = build_email_html(scored, gameday, n_plays)
    play_word  = "play" if n_plays == 1 else "plays"
    subject    = f"MLB Batter HRs — {n_plays} {play_word} today — {gameday}"
    _send_ses(subject, html_body)
    _send_sns(subject, n_plays)

    if n_plays + n_tracks > 0:
        _save_to_s3(pd.concat([plays, tracks]).copy(), gameday)

    result = {"n_play_bets": n_plays, "n_track_bets": n_tracks,
              "gameday": gameday, "html": html_body}
    if output:
        Path(output).write_text(json.dumps({k: v for k, v in result.items() if k != "html"}))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=None)
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()
    main(gameday=args.gameday, output=args.output)
