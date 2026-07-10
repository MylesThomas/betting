"""
Live gameday pipeline for MLB pitcher strikeouts props.

For each pitcher with a pitcher_strikeouts prop on the given gameday:
  1. Fetches today's probable starters from MLB Stats API (home/away, opponent)
  2. Fetches live events + props from The Odds API
  3. Loads rolling features from the spine (S3)
  4. Scores with OLS + bootstrap P(over): 10,000 residual draws per row
  5. Computes no-vig market P(over) for each unique line per pitcher
  6. Filters to edge >= 0.07 primary; shows edge >= 0.05 backup
  7. Sends SES HTML email + SNS notification
  8. Saves recommendations CSV to S3

Strategy: BOTH directions, edge >= 3%, no shrinkage, any odds tier
  Lines 3.5–6.5 only · shrinkage=0.0
Out-of-sample (v5, both): 7,104 bets · 59.36% WR · +528.97u · +7.45% ROI (2025–2026)
  Model: OLS, 7 features — v5 adds over_price_bucket_fine + under_price_bucket_fine
  In-sample/out-of-sample ratio: 0.99x (nearly identical — no overfitting)
Research: knowledge-base/raw/20260703-mlb-pitcher-strikeouts-v2.html

S3 paths read:
  s3://the-odds-api-mt/mlb/strikeouts_model/spine/mlb_strikeouts_spine.parquet
  s3://the-odds-api-mt/mlb/strikeouts_model/model/mlb_strikeouts_model.joblib
  s3://the-odds-api-mt/mlb/strikeouts_model/model/mlb_strikeouts_residuals.npy

S3 paths written:
  s3://the-odds-api-mt/mlb/strikeouts_model/daily_runs/{gameday}/recommendations.csv

Run:
    python src/mlb_strikeouts_modeling/scripts/run_pipeline.py
    python src/mlb_strikeouts_modeling/scripts/run_pipeline.py --gameday 2026-07-04
"""
from __future__ import annotations

import argparse
import html as html_module
import io
import json
import os
import sys
import time
import warnings
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import botocore.exceptions
import joblib
import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

ET = ZoneInfo("America/New_York")

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
MLB_API_BASE  = "https://statsapi.mlb.com/api/v1"
SPORT         = "baseball_mlb"
MARKET        = "pitcher_strikeouts"
REGIONS       = "us"
SLEEP_S       = 0.25

S3_BUCKET    = "the-odds-api-mt"
SPINE_KEY    = "mlb/strikeouts_model/spine/mlb_strikeouts_spine.parquet"
MODEL_KEY    = "mlb/strikeouts_model/model/mlb_strikeouts_model.joblib"
RESIDUALS_KEY= "mlb/strikeouts_model/model/mlb_strikeouts_residuals_clean.npy"
DAILY_PREFIX = "mlb/strikeouts_model/daily_runs"
SETTLED_KEY  = "mlb/strikeouts_model/settled/settled_bets.parquet"

SES_SOURCE    = os.environ.get("SETTLEMENT_SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SETTLEMENT_SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()

FEATURES = [
    "k_roll_career", "k_roll_c5",
    "opp_k_against_season", "is_home",
    "consensus_line",
    "over_price_bucket_fine",   # v5: 9-tier granular bin of avg American over odds at consensus line
    "under_price_bucket_fine",  # v5: 9-tier granular bin of avg American under odds at consensus line
]
# novig_prob_over is NOT a model feature (v3): it varies per book, making raw K projection
# book-dependent. raw_p_over/raw_p_under ARE used for edge (v6): edge = p_model - raw_p (breakeven).
# consensus_line IS a feature: it is player-level (modal line across all books), book-independent.
# over_price_bucket_fine / under_price_bucket_fine (v5): player-game level, book-independent.
SHRINKAGE               = 0.0
N_BOOT                  = 10_000
RNG                     = np.random.default_rng(42)
EDGE_THRESHOLD_UNDER    = 0.03   # UNDER bets: edge >= 3% vs raw implied prob (v6)
EDGE_THRESHOLD_OVER     = 0.03   # OVER bets:  edge >= 3% vs raw implied prob (v6)
EDGE_THRESHOLD_SHOW     = 0.02   # show backup bets (2–3% vs raw) in email
MIN_BOOKS               = 2
LINE_MIN                = 3.5    # exclude extreme alt lines that generate false edges
LINE_MAX                = 6.5

_BOOK_LEVEL_GROUPBY = ["player_key", "line", "bookmaker"]

_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"

NAME_MAP = {
    "louie varland": "louis varland",
}

BOOK_ABBREV = {
    "draftkings":     "DK",
    "fanduel":        "FD",
    "betmgm":         "MGM",
    "caesars":        "CZR",
    "pointsbet":      "PB",
    "bet365":         "B365",
    "williamhill_us": "WH",
    "bovada":         "BOV",
    "betonlineag":    "BOL",
    "mybookieag":     "MYB",
    "lowvig":         "LV",
    "pinnacle":       "PIN",
    "fliff":          "FL",
    "espnbet":        "ESPN",
    "fanatics":       "FAN",
    "betrivers":      "BR",
    "hardrockbet":    "HR",
    "ballybet":       "BB",
    "superbook":      "SB",
}

BOOK_DISPLAY = {
    "draftkings":     "DraftKings",
    "fanduel":        "FanDuel",
    "betmgm":         "BetMGM",
    "caesars":        "Caesars",
    "pointsbet":      "PointsBet",
    "bet365":         "Bet365",
    "williamhill_us": "William Hill",
    "bovada":         "Bovada",
    "betonlineag":    "BetOnline",
    "mybookieag":     "MyBookie",
    "lowvig":         "LowVig",
    "pinnacle":       "Pinnacle",
    "fliff":          "Fliff",
    "espnbet":        "ESPN Bet",
    "fanatics":       "Fanatics",
    "betrivers":      "BetRivers",
    "hardrockbet":    "Hard Rock Bet",
    "ballybet":       "Bally Bet",
    "superbook":      "SuperBook",
}


def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _normalize_name(name: str) -> str:
    import unicodedata, re
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name.strip().lower())
    name = re.sub(r"\s+", " ", name).strip()
    return NAME_MAP.get(name, name)


def _normalize_team(name: str) -> str:
    return str(name).strip().lower()


def american_profit(odds: float) -> float:
    return odds / 100.0 if odds >= 0 else 100.0 / abs(odds)


def fmt_odds(price) -> str:
    try:
        return f"{int(float(price)):+d}"
    except Exception:
        return "—"


# ── S3 ────────────────────────────────────────────────────────────────────────

def _s3():
    return boto3.client("s3")


def s3_get_parquet(key: str) -> pd.DataFrame:
    body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def s3_get_bytes(key: str) -> bytes:
    return _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()


def s3_put_csv(key: str, df: pd.DataFrame) -> None:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())


def load_settled() -> pd.DataFrame:
    try:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=SETTLED_KEY)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except Exception:
        return pd.DataFrame()


# ── MLB Stats API ─────────────────────────────────────────────────────────────

def fetch_probable_starters(gameday: str) -> dict[str, dict]:
    """
    Returns {player_key: {team, opponent, is_home, team_raw, opp_raw}}
    from MLB Stats API schedule endpoint with probablePitcher hydration.
    """
    try:
        r = requests.get(
            f"{MLB_API_BASE}/schedule",
            params={"sportId": 1, "date": gameday, "hydrate": "probablePitcher"},
            timeout=20,
        )
        r.raise_for_status()
    except Exception as e:
        print(f"  MLB API schedule failed: {e}")
        return {}

    result = {}
    for game in r.json().get("dates", [{}])[0].get("games", []):
        home = game.get("teams", {}).get("home", {})
        away = game.get("teams", {}).get("away", {})
        home_name = home.get("team", {}).get("name", "")
        away_name = away.get("team", {}).get("name", "")
        for side, team_info, opp_info, is_home in [
            ("home", home, away, 1),
            ("away", away, home, 0),
        ]:
            prob = team_info.get("probablePitcher", {})
            if not prob:
                continue
            name = _normalize_name(prob.get("fullName", ""))
            result[name] = {
                "player_id": prob.get("id"),
                "is_home":   is_home,
                "team_raw":  home_name if is_home else away_name,
                "opp_raw":   away_name if is_home else home_name,
                "opp_key":   _normalize_team(away_name if is_home else home_name),
            }
    print(f"  Probable starters from MLB API: {len(result)}")
    return result


# ── Odds API ──────────────────────────────────────────────────────────────────

def _api_get(path: str, params: dict) -> dict:
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY not set")
    resp = requests.get(
        f"{ODDS_API_BASE}{path}",
        params={**params, "apiKey": ODDS_API_KEY},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_events(gameday: str) -> list[dict]:
    data = _api_get(f"/sports/{SPORT}/events", {
        "dateFormat":        "iso",
        "commenceTimeFrom": f"{gameday}T00:00:00Z",
        "commenceTimeTo":   f"{gameday}T23:59:59Z",
    })
    events = [e for e in data if gameday in e.get("commence_time", "")]
    print(f"  Events on {gameday}: {len(events)}")
    return events


def fetch_props(events: list[dict], gameday: str) -> pd.DataFrame:
    rows = []
    for ev in events:
        time.sleep(SLEEP_S)
        try:
            data = _api_get(f"/sports/{SPORT}/events/{ev['id']}/odds", {
                "regions":    REGIONS,
                "markets":    MARKET,
                "oddsFormat": "american",
            })
        except requests.HTTPError as e:
            print(f"  Skipping {ev.get('id')} — {e}")
            continue
        for bk in data.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] != MARKET:
                    continue
                for outcome in mkt.get("outcomes", []):
                    rows.append({
                        "event_id":       ev["id"],
                        "game_date":      gameday,
                        "commence_time":  ev.get("commence_time", ""),
                        "home_team":      ev.get("home_team", ""),
                        "away_team":      ev.get("away_team", ""),
                        "bookmaker":      bk["key"],
                        "player":         outcome["description"],
                        "side":           outcome["name"].lower(),
                        "line":           float(outcome["point"]),
                        "odds":           float(outcome["price"]),
                    })
    return pd.DataFrame(rows)


# ── Feature assembly ──────────────────────────────────────────────────────────

def build_spine_lookups(spine: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      - player_latest: one row per player_key with most recent rolling features
      - opp_rate: {opp_key -> opp_k_against_season} from most recent spine entry
    """
    spine = spine.copy()
    spine["game_date_dt"] = pd.to_datetime(spine["game_date"])

    feat_cols = FEATURES + ["player_id", "game_date", "game_date_dt"]
    available = [c for c in feat_cols if c in spine.columns]

    player_latest = (
        spine.dropna(subset=["k_roll_s5"])
        .sort_values("game_date_dt")
        .groupby("player_key", as_index=False)
        .last()
    )[["player_key"] + [c for c in available if c != "game_date_dt"] + ["game_date_dt"]]

    # Opponent K-against rate: latest per opp_key in current season
    if "opp_key" in spine.columns and "opp_k_against_season" in spine.columns:
        cur_year = spine["game_date_dt"].dt.year.max()
        opp_rate = (
            spine[spine["game_date_dt"].dt.year == cur_year]
            .dropna(subset=["opp_key", "opp_k_against_season"])
            .sort_values("game_date_dt")
            .groupby("opp_key")["opp_k_against_season"]
            .last()
            .to_dict()
        )
    else:
        opp_rate = {}

    return player_latest, opp_rate


def assemble_bet_rows(
    props: pd.DataFrame,
    player_latest: pd.DataFrame,
    probables: dict[str, dict],
    opp_rate: dict[str, float],
    gameday: str,
    model,
    residuals: np.ndarray,
) -> pd.DataFrame:
    """
    Produces one row per _BOOK_LEVEL_GROUPBY (player_key, line, bookmaker) with model features
    and P(over) scored per book. Model features are all player-level (not book-level) so yhat is
    identical across all book rows for the same player-game. novig_over is used only for edge
    comparison (p_market_over = novig_over) and is NOT passed to model.predict().
    """
    if props.empty:
        return pd.DataFrame(), []

    props = props.copy()
    props["player_key"] = props["player"].apply(_normalize_name)

    def _parse_game_time(ct: str) -> str:
        try:
            dt_utc = datetime.strptime(ct[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            dt_et  = dt_utc.astimezone(ET)
            return dt_et.strftime("%-I:%M %p ET")
        except Exception:
            return ""

    game_time_map = (
        props.drop_duplicates("player_key")[["player_key", "commence_time"]]
        .set_index("player_key")["commence_time"]
        .apply(_parse_game_time)
        .to_dict()
    )

    over_props  = props[props["side"] == "over"].copy()
    under_props = props[props["side"] == "under"].copy()
    if over_props.empty or under_props.empty:
        return pd.DataFrame(), []

    # Player-level features (not book-level)
    consensus_line = (
        over_props.groupby("player_key")["line"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.median())
        .rename("consensus_line")
        .reset_index()
    )
    n_books_per_player = (
        over_props.groupby("player_key")["bookmaker"]
        .nunique()
        .rename("n_books_total")
        .reset_index()
    )

    # ── v5 features: odds bucket at consensus line (player-game level, book-independent) ──
    # odds from the Odds API are American odds (integer). Average over all books at modal line.
    def _granular_bin(odds: float) -> int:
        if odds < -300:  return 0
        if odds < -200:  return 1
        if odds < -110:  return 2
        if odds < -100:  return 3
        if odds <= 100:  return 4
        if odds <= 200:  return 5
        if odds <= 300:  return 6
        if odds <= 500:  return 7
        return 8

    cl_map = consensus_line.set_index("player_key")["consensus_line"].to_dict()
    # Filter each side to rows where line == modal (consensus) line
    over_at_cl  = over_props[over_props.apply(
        lambda r: r["line"] == cl_map.get(r["player_key"], r["line"]), axis=1
    )].copy()
    under_at_cl = under_props[under_props.apply(
        lambda r: r["line"] == cl_map.get(r["player_key"], r["line"]), axis=1
    )].copy()
    # avg American odds at consensus line per player
    over_am_avg = (
        over_at_cl.groupby("player_key")["odds"].mean()
        .rename("avg_over_am")
        .reset_index()
    )
    under_am_avg = (
        under_at_cl.groupby("player_key")["odds"].mean()
        .rename("avg_under_am")
        .reset_index()
    )
    price_buckets = over_am_avg.merge(under_am_avg, on="player_key", how="outer")
    price_buckets["over_price_bucket_fine"]  = price_buckets["avg_over_am"].apply(_granular_bin)
    price_buckets["under_price_bucket_fine"] = price_buckets["avg_under_am"].apply(_granular_bin)
    price_buckets = price_buckets[["player_key", "over_price_bucket_fine", "under_price_bucket_fine"]]

    # Build per-book paired rows at _BOOK_LEVEL_GROUPBY grain
    over_props["profit_o"] = over_props["odds"].apply(american_profit)

    under_props_merge = under_props[["player_key", "line", "bookmaker", "odds"]].copy()
    under_props_merge["profit_u"] = under_props_merge["odds"].apply(american_profit)
    under_props_merge = under_props_merge.rename(columns={"odds": "odds_u"})

    # Inner join on _BOOK_LEVEL_GROUPBY: one row per player-line-book
    book_rows = over_props.merge(under_props_merge, on=_BOOK_LEVEL_GROUPBY, how="inner")

    # Per-book implied probs and novig
    book_rows["raw_p_over"]  = 1 / (1 + book_rows["profit_o"])
    book_rows["raw_p_under"] = 1 / (1 + book_rows["profit_u"])
    denom = book_rows["raw_p_over"] + book_rows["raw_p_under"]
    book_rows["novig_over"]  = book_rows["raw_p_over"]  / denom
    book_rows["novig_under"] = book_rows["raw_p_under"] / denom

    # novig_prob_over = per-book novig (for display/CSV only — NOT a model feature in v3)
    # The model uses consensus_line (player-level) instead; novig_over is used only for edge comparison.
    book_rows["novig_prob_over"] = book_rows["novig_over"]

    # Book abbreviation for display
    book_rows["book_abbrev"] = book_rows["bookmaker"].apply(
        lambda b: BOOK_ABBREV.get(str(b).lower(), str(b).upper()[:4])
    )

    # Join player-level features
    book_rows = book_rows.merge(consensus_line,     on="player_key", how="left")
    book_rows = book_rows.merge(n_books_per_player, on="player_key", how="left")
    book_rows = book_rows.merge(price_buckets,      on="player_key", how="left")

    # Merge spine features
    book_rows = book_rows.merge(player_latest, on="player_key", how="left", suffixes=("", "_spine"))

    today_dt = pd.to_datetime(gameday)
    if "game_date_dt" in book_rows.columns:
        book_rows["days_since_last"] = (today_dt - book_rows["game_date_dt"]).dt.days
    else:
        book_rows["days_since_last"] = np.nan

    book_rows["game_time_et"] = book_rows["player_key"].map(game_time_map)

    # Enrich from MLB API probables
    for col in ("is_home", "opp_key", "player_id", "team_raw", "opp_raw"):
        book_rows[col] = book_rows["player_key"].map(
            {k: v.get(col) for k, v in probables.items()}
        )

    book_rows["opp_k_against_season"] = book_rows["opp_key"].map(opp_rate)

    unmatched = book_rows.loc[book_rows["team_raw"].isna(), "player_key"].unique().tolist()
    if unmatched:
        print(f"  WARNING: {len(unmatched)} player(s) with no probables match, dropping: {unmatched}")
        book_rows = book_rows[book_rows["team_raw"].notna()].copy()

    book_rows["player_id"] = book_rows["player_key"].map(
        {k: v.get("player_id") for k, v in probables.items()}
    ).combine_first(book_rows.get("player_id", pd.Series(dtype=float)))

    book_rows["game_month"] = today_dt.month
    book_rows["game_date"]  = gameday

    # Filter to main line range
    book_rows = book_rows[book_rows["line"].between(LINE_MIN, LINE_MAX)].copy()

    # Drop players with insufficient books or missing core features (player-level filters)
    book_rows = book_rows[book_rows["n_books_total"] >= MIN_BOOKS].copy()
    book_rows = book_rows.dropna(subset=["k_roll_career", "consensus_line"]).copy()

    if book_rows.empty:
        return pd.DataFrame(), unmatched

    book_rows["opp_k_against_season"] = book_rows["opp_k_against_season"].fillna(
        book_rows["opp_k_against_season"].median() if book_rows["opp_k_against_season"].notna().any() else 5.5
    )
    book_rows["is_home"] = book_rows["is_home"].fillna(0).astype(int)

    # Score: FEATURES are all player-level (no book-level inputs) → yhat is identical across
    # all book rows for the same player-game. p_market_over (novig_over) varies per book — used
    # only for edge comparison, not as a model input.
    feat_df = book_rows[FEATURES].copy()
    feat_df = feat_df.fillna(feat_df.median())
    book_rows["yhat"] = model.predict(feat_df)

    # Bootstrap P(over) once per (player_key, line) — not per book row.
    # yhat is book-independent so all books share the same mean_adj for a given player-line.
    # Running bootstrap per-row would give different Monte Carlo estimates for the same bet
    # just because the RNG advances differently, producing spurious Model% variance across books.
    pl_pairs = book_rows[["player_key", "line", "yhat"]].drop_duplicates(["player_key", "line"]).copy()
    line_arr = pl_pairs["line"].values
    yhat_arr = pl_pairs["yhat"].values
    mean_adj = line_arr + (1 - SHRINKAGE) * (yhat_arr - line_arr)

    samples  = RNG.choice(residuals, size=(len(mean_adj), N_BOOT), replace=True)
    sims     = mean_adj[:, None] + samples
    pl_pairs["p_model_over"]  = np.clip((sims > line_arr[:, None]).mean(axis=1), 0.01, 0.99)
    pl_pairs["p_model_under"] = 1.0 - pl_pairs["p_model_over"]

    book_rows = book_rows.merge(
        pl_pairs[["player_key", "line", "p_model_over", "p_model_under"]],
        on=["player_key", "line"],
        how="left",
    )
    # Edge vs raw implied prob (actual breakeven from posted odds) — not novig.
    # novig compares against a price never actually offered; raw_p is what you get paid.
    book_rows["p_market_over"]  = book_rows["raw_p_over"]
    book_rows["p_market_under"] = book_rows["raw_p_under"]

    book_rows["edge_over"]  = book_rows["p_model_over"]  - book_rows["p_market_over"]
    book_rows["edge_under"] = book_rows["p_model_under"] - book_rows["p_market_under"]

    # Simple strategy (v6, per book row):
    #   OVER:  edge >= EDGE_THRESHOLD_OVER (3%) vs raw implied prob, any odds tier
    #   UNDER: edge >= EDGE_THRESHOLD_UNDER (3%), any odds tier
    #   SHOW:  either direction >= EDGE_THRESHOLD_SHOW (2%) in email
    #   When both over and under qualify, pick the larger edge.
    over_q  = book_rows["edge_over"]  >= EDGE_THRESHOLD_SHOW
    under_q = book_rows["edge_under"] >= EDGE_THRESHOLD_SHOW

    book_rows["side"] = np.where(
        over_q & (~under_q | (book_rows["edge_over"] >= book_rows["edge_under"])), "over",
        np.where(under_q, "under", None),
    )
    book_rows["edge"] = np.where(
        book_rows["side"] == "over",  book_rows["edge_over"],
        np.where(book_rows["side"] == "under", book_rows["edge_under"], np.nan)
    )
    book_rows["is_primary"] = np.where(
        book_rows["side"] == "over",   book_rows["edge_over"]  >= EDGE_THRESHOLD_OVER,
        np.where(book_rows["side"] == "under", book_rows["edge_under"] >= EDGE_THRESHOLD_UNDER, False)
    )
    book_rows["player"]       = book_rows["player_key"]
    book_rows["max_abs_edge"] = np.maximum(
        book_rows["edge_over"].abs(), book_rows["edge_under"].abs()
    )

    return book_rows.sort_values("max_abs_edge", ascending=False).reset_index(drop=True), unmatched


# ── HTML email ────────────────────────────────────────────────────────────────

def build_html(bets: pd.DataFrame, gameday: str, n_scored: int,
               settled_yesterday: pd.DataFrame | None = None,
               settled_all: pd.DataFrame | None = None) -> str:
    he       = html_module.escape
    now_str  = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    n_primary = int(bets["is_primary"].sum())
    n_total   = int(len(bets))

    def fmt(v, fstr=""):
        try:
            return format(float(v), fstr) if fstr else str(v)
        except (TypeError, ValueError):
            return "—"

    # Derive home/away for each row so we can group by game
    bets = bets.copy()
    bets["_home"] = bets.apply(
        lambda r: r.get("team_raw", "") if r.get("is_home") else r.get("opp_raw", ""), axis=1
    )
    bets["_away"] = bets.apply(
        lambda r: r.get("opp_raw", "") if r.get("is_home") else r.get("team_raw", ""), axis=1
    )
    NCOLS = 25  # Player Team Opp Time(ET) Line | Book | Over Under | RawO RawU RawT | FairO FairU FairT Vig | Pred Delta PredO PredU | OvrEdge UndEdge | k_career k_c5 opp_k Status

    bets["max_abs_edge"] = bets["max_abs_edge"] if "max_abs_edge" in bets.columns else (
        bets[["edge_over", "edge_under"]].abs().max(axis=1)
    )
    # Sort: game start time first, then matchup (keeps same-time games together),
    # then within each game: primary first, then by max abs edge desc.
    _SORT_COLS = ["commence_time", "_home", "_away", "is_primary", "max_abs_edge"]
    _SORT_ASC  = [True, True, True, False, False]
    bets = bets.sort_values(_SORT_COLS, ascending=_SORT_ASC)

    # Collapse no-edge rows to one per (player_key, line) — keeps the book with highest abs edge.
    # PLAY/WATCH rows stay per-book so you can see which book to bet on.
    no_edge_mask = bets["side"].isna()
    bets = pd.concat([
        bets[~no_edge_mask],
        bets[no_edge_mask].drop_duplicates(subset=["player_key", "line"], keep="first"),
    ]).sort_values(
        _SORT_COLS, ascending=_SORT_ASC
    )

    rows_html = ""
    seen_games: set = set()

    for _, r in bets.iterrows():
        home = str(r.get("_home", ""))
        away = str(r.get("_away", ""))
        gk   = (home, away)

        if gk not in seen_games:
            seen_games.add(gk)
            game_group = bets[(bets["_home"] == home) & (bets["_away"] == away)]
            n_plays    = int(game_group["is_primary"].sum())
            play_label = f"{n_plays} PLAY{'S' if n_plays != 1 else ''}" if n_plays > 0 else "no plays"
            play_color = "#276221" if n_plays > 0 else "#888"
            game_time  = str(game_group["game_time_et"].iloc[0]) if "game_time_et" in game_group.columns and len(game_group) > 0 else ""
            time_part  = f"<span style='font-weight:400;color:#666'>{he(game_time)}</span> &nbsp;·&nbsp; " if game_time else ""
            rows_html += (
                f"<tr style='background:#edf1f5'>"
                f"<td colspan='{NCOLS}' style='padding:7px 10px;font-weight:600;font-size:12px;"
                f"color:#2c3e50;border-top:2px solid #bdc3c7;border-bottom:1px solid #bdc3c7'>"
                f"{time_part}{he(away)} @ {he(home)}"
                f" &nbsp;·&nbsp; <span style='color:{play_color}'>{play_label}</span>"
                f"</td></tr>\n"
            )

        is_primary = bool(r.get("is_primary"))
        side       = r.get("side")  # None / "over" / "under"
        has_side   = side in ("over", "under")
        is_over    = side == "over"

        if is_primary:
            side_dir = "OVER" if side == "over" else "UNDER"
            bg       = "background:#eaf6ea"
            status   = f"<span style='color:#276221;font-weight:bold'>PLAY - {side_dir}</span>"
        elif has_side:
            bg     = "background:#fffbeb"
            status = "<span style='color:#b45309;font-weight:bold'>WATCH</span>"
        else:
            bg     = "background:#fff"
            status = "<span style='color:#bbb'>—</span>"

        dim = "color:#bbb;font-size:11px"  # style for no-edge rows

        # ── Player / Game group ──────────────────────────────────────────────
        team = he(str(r.get("team_raw") or "—"))
        opp  = he(str(r.get("opp_raw")  or "—"))
        name_style = "font-weight:600" if has_side else f"font-weight:600;{dim}"
        team_cell  = f"<td style='text-align:center;font-size:11px;{'color:#555' if has_side else dim}'>{team}</td>"
        opp_cell   = f"<td style='text-align:center;font-size:11px;{'color:#555' if has_side else dim}'>{opp}</td>"
        game_time_val = he(str(r.get("game_time_et") or "—"))
        time_cell  = f"<td style='text-align:center;font-size:11px;{'color:#666' if has_side else dim}'>{game_time_val}</td>"
        line_cell  = f"<td style='text-align:center;{'font-size:12px' if has_side else dim}'>{fmt(r.get('line'), '.1f')}</td>"

        # ── Book group ───────────────────────────────────────────────────────
        book_full  = BOOK_DISPLAY.get(str(r.get("bookmaker", "")).lower(),
                                      str(r.get("bookmaker", "")).title())
        book_disp  = he(book_full)
        if has_side:
            book_color = "#276221;font-weight:600" if is_primary else "#555"
            book_cell  = f"<td style='font-size:11px;color:{book_color};text-align:center'>{book_disp}</td>"
        else:
            book_cell = f"<td style='font-size:11px;text-align:center;{dim}'>{book_disp}</td>"

        # ── American Odds group ──────────────────────────────────────────────
        over_odds  = r.get("odds")
        under_odds = r.get("odds_u")
        if has_side:
            over_color = "#276221;font-weight:bold" if is_over  else "#555"
            undr_color = "#1d4ed8;font-weight:bold" if not is_over else "#555"
            over_cell  = f"<td style='text-align:center;font-size:12px;color:{over_color}'>{fmt_odds(over_odds)}</td>"
            undr_cell  = f"<td style='text-align:center;font-size:12px;color:{undr_color}'>{fmt_odds(under_odds)}</td>"
        else:
            over_cell = f"<td style='text-align:center;font-size:12px;{dim}'>{fmt_odds(over_odds)}</td>"
            undr_cell = f"<td style='text-align:center;font-size:12px;{dim}'>{fmt_odds(under_odds)}</td>"

        # ── Implied + No-Vig groups ───────────────────────────────────────────
        raw_o = r.get("raw_p_over")
        raw_u = r.get("raw_p_under")
        raw_t = float(raw_o) + float(raw_u) if pd.notna(raw_o) and pd.notna(raw_u) else None

        def _pct_cell(val, bold: bool = False) -> str:
            s = fmt(val, '.1%') if pd.notna(val) else '—'
            if not has_side:
                return f"<td style='text-align:center;font-size:11px;{dim}'>{s}</td>"
            style = "text-align:center;font-size:11px;font-weight:bold" if bold else "text-align:center;font-size:11px;color:#999"
            return f"<td style='{style}'>{s}</td>"

        raw_over_cell   = _pct_cell(raw_o)
        raw_under_cell  = _pct_cell(raw_u)
        raw_total_cell  = f"<td style='text-align:center;font-size:11px;{'color:#555' if has_side else dim}'>{fmt(raw_t, '.1%') if raw_t is not None else '—'}</td>"
        fair_over_cell  = _pct_cell(r.get("novig_over"),  is_over)
        fair_under_cell = _pct_cell(r.get("novig_under"), not is_over)
        fair_total_cell = f"<td style='text-align:center;font-size:11px;{'color:#555' if has_side else dim}'>100.0%</td>"
        vig_val  = raw_t - 1.0 if raw_t is not None else None
        vig_cell = f"<td style='text-align:center;font-size:11px;{'color:#555' if has_side else dim}'>{fmt(vig_val, '.1%') if vig_val is not None else '—'}</td>"

        # ── Model Prediction group ───────────────────────────────────────────
        yhat  = r.get("yhat")
        line  = r.get("line")
        delta = float(yhat) - float(line) if pd.notna(yhat) and pd.notna(line) else None
        delta_color = "#276221" if (delta is not None and delta > 0) else "#c0392b" if (delta is not None and delta < 0) else "#555"
        pred_cell       = f"<td style='text-align:center;font-weight:600;{'color:#2c3e50' if has_side else dim}'>{fmt(yhat, '.2f')}</td>"
        delta_cell      = f"<td style='text-align:center;font-weight:600;{'color:' + delta_color if has_side else dim}'>{(f'{delta:+.2f}' if delta is not None else '—')}</td>"
        pred_over_cell  = _pct_cell(r.get("p_model_over"),  is_over)
        pred_under_cell = _pct_cell(r.get("p_model_under"), not is_over)

        # ── Edge group ───────────────────────────────────────────────────────
        edge_over  = float(r.get("edge_over",  0) or 0)
        edge_under = float(r.get("edge_under", 0) or 0)

        def _edge_cell(val: float, active: bool) -> str:
            if has_side:
                style = "text-align:center;font-weight:bold" if active else "text-align:center;color:#999;font-size:11px"
            else:
                style = f"text-align:center;{dim}"
            return f"<td style='{style}'>{val:+.1%}</td>"

        # ── Model Inputs group ───────────────────────────────────────────────
        stat_style = f"text-align:center;font-size:11px;{'color:#555' if has_side else 'color:#bbb'}"

        rows_html += (
            f"<tr style='{bg}'>"
            f"<td style='{name_style}'>{he(str(r.get('player', '—')))}</td>"
            + team_cell + opp_cell + time_cell + line_cell
            + book_cell
            + over_cell + undr_cell
            + raw_over_cell + raw_under_cell + raw_total_cell
            + fair_over_cell + fair_under_cell + fair_total_cell + vig_cell
            + pred_cell + delta_cell + pred_over_cell + pred_under_cell
            + _edge_cell(edge_over,  is_over)
            + _edge_cell(edge_under, not is_over)
            + f"<td style='{stat_style}'>{fmt(r.get('k_roll_career'), '.1f')}</td>"
            f"<td style='{stat_style}'>{fmt(r.get('k_roll_c5'), '.1f')}</td>"
            f"<td style='{stat_style}'>{fmt(r.get('opp_k_against_season'), '.1f')}</td>"
            f"<td style='text-align:center'>{status}</td>"
            f"</tr>\n"
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{font-family:{_SANS};color:#222;max-width:1200px;margin:auto;padding:20px}}
  h2 {{color:#2c3e50;margin-bottom:4px}}
  table {{border-collapse:collapse;width:100%;margin-top:8px}}
  th {{background:#2c3e50;color:#fff;padding:7px 8px;text-align:left;font-size:12px;white-space:nowrap}}
  td {{padding:5px 8px;border-bottom:1px solid #e0e0e0;font-size:12px}}
  details {{margin-top:16px;border:1px solid #ddd;border-radius:6px;padding:0 12px 8px}}
  summary {{font-weight:600;font-size:14px;cursor:pointer;padding:10px 0;color:#2c3e50;user-select:none}}
  .footer {{background:#ecf0f1;border-radius:6px;padding:10px 16px;margin-top:16px;font-size:12px;color:#555}}
  .legend span {{display:inline-block;padding:2px 10px;border-radius:4px;margin-right:8px;font-size:12px;font-weight:600}}
</style>
</head><body>

<h2>MLB Pitcher Strikeouts — {gameday}</h2>
<p style='margin-top:4px'>
  <strong>{n_primary} PLAY{'S' if n_primary != 1 else ''}</strong>
  &nbsp;·&nbsp; {n_total - n_primary} watch
  &nbsp;·&nbsp; {n_scored} pitchers scored
  &nbsp;·&nbsp; {now_str}
</p>

<p class='legend'>
  <span style='background:#eaf6ea;color:#276221'>PLAY - OVER / PLAY - UNDER</span> edge ≥{EDGE_THRESHOLD_OVER*100:.0f}pp, any odds
  <span style='background:#fffbeb;color:#b45309;margin-left:8px'>WATCH</span> OVER or UNDER {EDGE_THRESHOLD_SHOW*100:.0f}–{EDGE_THRESHOLD_UNDER*100:.0f}pp
</p>

<details open>
  <summary>▸ Props by game &nbsp;<span style='font-weight:normal;color:#666'>({n_total} rows scored)</span></summary>
  <table>
    <thead>
    <tr>
      <th colspan="5" style="text-align:center;border-right:1px solid #6a8fb0">Player / Game</th>
      <th colspan="1" style="text-align:center;border-right:1px solid #6a8fb0">Book</th>
      <th colspan="2" style="text-align:center;border-right:1px solid #6a8fb0">American Odds</th>
      <th colspan="3" style="text-align:center;border-right:1px solid #6a8fb0">Implied</th>
      <th colspan="4" style="text-align:center;border-right:1px solid #6a8fb0">No-Vig</th>
      <th colspan="4" style="text-align:center;border-right:1px solid #6a8fb0">Model Prediction</th>
      <th colspan="2" style="text-align:center;border-right:1px solid #6a8fb0">Edge</th>
      <th colspan="4" style="text-align:center">Model Inputs</th>
    </tr>
    <tr>
      <th>Player</th><th>Team</th><th>Opp</th><th>Time (ET)</th><th>Line</th>
      <th>Book</th>
      <th>Over</th><th>Under</th>
      <th>Raw Over</th><th>Raw Under</th><th>Raw Total</th>
      <th>Fair Over</th><th>Fair Under</th><th>Fair Total</th><th>Vig</th>
      <th>Prediction</th><th>Delta</th><th>Pred Over</th><th>Pred Under</th>
      <th>Over Edge</th><th>Under Edge</th>
      <th>k_roll_career</th><th>k_roll_c5</th><th>opp_k_against_season</th><th>Status</th>
    </tr>
    </thead>
    {rows_html}
  </table>
</details>

<details>
  <summary>▸ Model inputs (OLS — 7 features, v5)</summary>
  <table style='margin-top:6px;width:auto'>
    <tr>
      <th>Feature</th><th>Shown as</th><th>What it measures</th><th>Role</th>
    </tr>
    <tr><td style='font-family:{_MONO}'>consensus_line</td><td>—</td><td>Market's consensus K line — modal line across all books; player-level, book-independent; dominant predictor</td><td>Primary signal</td></tr>
    <tr style='background:#f9f9f9'><td style='font-family:{_MONO}'>k_roll_career</td><td>—</td><td>Career avg Ks — stable long-run baseline for pitcher quality</td><td>Baseline</td></tr>
    <tr><td style='font-family:{_MONO}'>k_roll_c5</td><td>k_roll_c5</td><td>Avg Ks over last 5 starts (career window, crosses seasons) — captures recent form</td><td>Recent form</td></tr>
    <tr style='background:#f9f9f9'><td style='font-family:{_MONO}'>opp_k_against_season</td><td>opp_K_rate</td><td>Opponent team Ks-against rate this season — harder lineups suppress Ks</td><td>Matchup</td></tr>
    <tr><td style='font-family:{_MONO}'>is_home</td><td>—</td><td>1 if pitcher at home — small but consistent home advantage</td><td>Context</td></tr>
    <tr style='background:#f9f9f9'><td style='font-family:{_MONO}'>over_price_bucket_fine</td><td>—</td><td>9-tier bin of avg American over odds at consensus line (player-game level, book-independent)</td><td>v5 odds signal</td></tr>
    <tr><td style='font-family:{_MONO}'>under_price_bucket_fine</td><td>—</td><td>9-tier bin of avg American under odds at consensus line (player-game level, book-independent)</td><td>v5 odds signal</td></tr>
  </table>
  <p style='font-size:11px;color:#555;margin-top:8px'>
    Model: OLS (StandardScaler + LinearRegression) · out-of-sample RMSE=2.127 · residual σ=2.146 · shrinkage=0.0 · bootstrap N=10,000<br>
    Prediction: yhat → P(over) = fraction of 10k simulated K counts above line (no shrinkage, full model signal)<br>
    novig_prob_over removed (v3): per-book pricing varies → raw K projection was book-dependent (nonsensical).<br>
    consensus_line retained: player-level modal line, same for all books — legitimate, stable predictor.<br>
    v5 adds over_price_bucket_fine + under_price_bucket_fine: granular odds tier encoding, book-independent.<br>
    Research: knowledge-base/raw/20260703-mlb-pitcher-strikeouts-v2.html
  </p>
</details>

<div class='footer'>
  Out-of-sample (v5, 2025–2026): 7,104 bets &nbsp;·&nbsp; 59.36% WR &nbsp;·&nbsp; +528.97u &nbsp;·&nbsp; +7.45% ROI &nbsp;|&nbsp; In-sample/out-of-sample ratio: 0.99x (no overfitting)
</div>

{build_section2_html(settled_yesterday if settled_yesterday is not None else pd.DataFrame(),
                     (pd.Timestamp(gameday) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))}
{build_section3_html(settled_all if settled_all is not None else pd.DataFrame())}

</body></html>"""


def build_warning_html(unmatched: list[str], probables: dict, gameday: str) -> str:
    he = html_module.escape
    rows = "".join(
        f"<tr><td style='padding:6px 10px;font-family:{_MONO}'>{he(name)}</td></tr>"
        for name in sorted(unmatched)
    )
    prob_rows = "".join(
        f"<tr><td style='padding:4px 10px;font-family:{_MONO};color:#555'>{he(k)}</td></tr>"
        for k in sorted(probables.keys())
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{font-family:{_SANS};color:#222;max-width:800px;margin:auto;padding:20px}}
  h2 {{color:#b45309}}
  table {{border-collapse:collapse;margin-top:8px}}
  th {{background:#b45309;color:#fff;padding:7px 10px;text-align:left;font-size:12px}}
  td {{border-bottom:1px solid #e0e0e0;font-size:12px}}
</style>
</head><body>
<h2>⚠ MLB Strikeouts — Unmatched Players — {gameday}</h2>
<p>The following players had props on The Odds API but could <strong>not be matched</strong>
to a probable starter from the MLB Stats API. They were <strong>excluded</strong> from today's
recommendations. Add a <code>NAME_MAP</code> entry in <code>run_pipeline.py</code> to fix.</p>
<h3 style='margin-bottom:4px'>Unmatched (Odds API names, normalised)</h3>
<table><tr><th>player_key</th></tr>{rows}</table>
<h3 style='margin-top:16px;margin-bottom:4px'>Available probables keys (MLB API)</h3>
<table><tr><th>player_key</th></tr>{prob_rows}</table>
</body></html>"""


def build_section2_html(settled_yesterday: pd.DataFrame, yesterday: str) -> str:
    he = html_module.escape
    if settled_yesterday.empty:
        return f"<details open><summary>▸ Yesterday's results &nbsp;<span style='font-weight:normal;color:#666'>({yesterday})</span></summary><p style='padding:8px;color:#888'>No settled bets for {yesterday}.</p></details>\n"

    df = settled_yesterday[settled_yesterday["outcome"].isin(["WIN", "LOSS", "PUSH", "DNP"])].copy()
    wins   = int((df["outcome"] == "WIN").sum())
    losses = int((df["outcome"] == "LOSS").sum())
    pushes = int((df["outcome"] == "PUSH").sum())
    dnps   = int((df["outcome"] == "DNP").sum())
    day_pnl = float(df["pnl"].sum())
    pnl_color = "#276221" if day_pnl >= 0 else "#c0392b"
    record = f"{wins}W–{losses}L{f'–{pushes}P' if pushes else ''}{f'–{dnps}DNP' if dnps else ''}"

    def _oc_color(oc: str) -> str:
        return {"WIN": "#276221", "LOSS": "#c0392b", "PUSH": "#b45309", "DNP": "#888"}.get(oc, "#222")

    rows = ""
    for _, r in df.sort_values(["outcome", "player_key"]).iterrows():
        side     = str(r.get("side", "")).upper()
        side_c   = "#276221" if side == "OVER" else "#1d4ed8"
        oc       = str(r.get("outcome", "—"))
        raw_odds = r.get("odds") if side == "OVER" else r.get("odds_u")
        odds_str = f"{int(float(raw_odds)):+d}" if pd.notna(raw_odds) else "—"
        actual_k = "—" if pd.isna(r.get("actual_k")) else int(r["actual_k"])
        pnl_val  = float(r.get("pnl") or 0)
        edge_val = float(r.get("edge") or 0)
        pnl_c    = "#276221" if pnl_val > 0 else "#c0392b" if pnl_val < 0 else "#888"
        book     = he(str(r.get("book_abbrev") or r.get("bookmaker") or "—"))
        rows += (
            f"<tr>"
            f"<td style='padding:5px 8px;font-weight:600'>{he(str(r.get('player_key', '—')))}</td>"
            f"<td style='padding:5px 8px;text-align:center;font-weight:bold;color:{side_c}'>{side}</td>"
            f"<td style='padding:5px 8px;text-align:center'>{r.get('line', '—')}</td>"
            f"<td style='padding:5px 8px;text-align:center'>{book}</td>"
            f"<td style='padding:5px 8px;text-align:center'>{odds_str}</td>"
            f"<td style='padding:5px 8px;text-align:center'>{edge_val*100:+.1f}pp</td>"
            f"<td style='padding:5px 8px;text-align:center'>{actual_k}</td>"
            f"<td style='padding:5px 8px;text-align:center;font-weight:bold;color:{_oc_color(oc)}'>{oc}</td>"
            f"<td style='padding:5px 8px;text-align:center;font-weight:bold;color:{pnl_c}'>{pnl_val:+.2f}u</td>"
            f"</tr>\n"
        )

    return f"""<details open>
  <summary>▸ Yesterday's results &nbsp;<span style='font-weight:normal;color:#666'>({yesterday} &nbsp;·&nbsp; {record} &nbsp;·&nbsp; <span style='color:{pnl_color};font-weight:bold'>{day_pnl:+.2f}u</span>)</span></summary>
  <table style='margin-top:6px'>
    <tr>
      <th>Player</th><th>Side</th><th>Line</th><th>Book</th><th>Odds</th><th>Edge</th><th>Actual K</th><th>Outcome</th><th>P&amp;L</th>
    </tr>
    {rows}
  </table>
</details>
"""


def build_section3_html(settled_all: pd.DataFrame) -> str:
    if settled_all.empty:
        return "<details><summary>▸ All-time results</summary><p style='padding:8px;color:#888'>No history yet.</p></details>\n"

    settled = settled_all[settled_all["outcome"].isin(["WIN", "LOSS"])].copy()
    if settled.empty:
        return ""

    total_bets = len(settled)
    wins       = int((settled["outcome"] == "WIN").sum())
    losses     = int((settled["outcome"] == "LOSS").sum())
    total_pnl  = float(settled["pnl"].dropna().sum())
    roi        = total_pnl / total_bets * 100 if total_bets else 0.0
    wr         = wins / total_bets * 100 if total_bets else 0.0
    pnl_color  = "#276221" if total_pnl >= 0 else "#c0392b"

    def _card(label: str, value: str, color: str = "#2c3e50") -> str:
        return (
            f"<div style='background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;"
            f"padding:10px 16px;min-width:100px;display:inline-block;margin:0 8px 8px 0'>"
            f"<div style='font-size:11px;color:#666;text-transform:uppercase'>{label}</div>"
            f"<div style='font-size:18px;font-weight:700;color:{color}'>{value}</div>"
            f"</div>"
        )

    settled["_season"] = pd.to_datetime(settled["game_date"]).dt.year
    season_rows = ""
    for season, grp in settled.groupby("_season"):
        s_bets   = len(grp)
        s_wins   = int((grp["outcome"] == "WIN").sum())
        s_losses = int((grp["outcome"] == "LOSS").sum())
        s_pnl    = float(grp["pnl"].dropna().sum())
        s_roi    = s_pnl / s_bets * 100 if s_bets else 0.0
        s_wr     = s_wins / s_bets * 100 if s_bets else 0.0
        s_c      = "#276221" if s_pnl >= 0 else "#c0392b"
        season_rows += (
            f"<tr>"
            f"<td style='padding:5px 8px'>{season}</td>"
            f"<td style='padding:5px 8px;text-align:center'>{s_bets}</td>"
            f"<td style='padding:5px 8px;text-align:center'>{s_wins}W–{s_losses}L</td>"
            f"<td style='padding:5px 8px;text-align:center'>{s_wr:.1f}%</td>"
            f"<td style='padding:5px 8px;text-align:center;font-weight:bold;color:{s_c}'>{s_pnl:+.2f}u</td>"
            f"<td style='padding:5px 8px;text-align:center;font-weight:bold;color:{s_c}'>{s_roi:+.1f}%</td>"
            f"</tr>\n"
        )

    return f"""<details>
  <summary>▸ All-time results</summary>
  <div style='margin:8px 0 16px'>
    {_card("All-time P&L", f"{total_pnl:+.2f}u", pnl_color)}
    {_card("Record", f"{wins}W–{losses}L")}
    {_card("Win %", f"{wr:.1f}%")}
    {_card("ROI", f"{roi:+.1f}%", pnl_color)}
    {_card("Bets", str(total_bets))}
  </div>
  <table style='width:auto'>
    <tr><th>Season</th><th>Bets</th><th>Record</th><th>Win %</th><th>Units</th><th>ROI</th></tr>
    {season_rows}
  </table>
  <p style='font-size:11px;color:#555;margin-top:8px'>
    Flat-bet 1u &nbsp;·&nbsp; UNDER or OVER ≥{EDGE_THRESHOLD_UNDER*100:.0f}pp &nbsp;·&nbsp;
    OOS baseline (v5, 2025–2026): +528.97u · +7.45% ROI
  </p>
</details>
"""


# ── Send / notify ─────────────────────────────────────────────────────────────

def send_email(subject: str, html_body: str) -> None:
    to_list = [a.strip() for a in SES_TO_RAW.split(",") if a.strip()]
    if not SES_SOURCE or not to_list:
        print(f"  SES not configured (SES_SOURCE={SES_SOURCE!r}), skipping email")
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


def publish_sns(subject: str, message: str) -> None:
    if not SNS_TOPIC_ARN:
        return
    boto3.client("sns").publish(TopicArn=SNS_TOPIC_ARN, Subject=subject[:100], Message=message)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=today_et())
    args   = parser.parse_args()
    gameday = args.gameday

    print(f"\nMLB Strikeouts Pipeline | gameday={gameday}", flush=True)

    # 1. Load model artifacts from S3
    print("Loading model artifacts from S3...", flush=True)
    model     = joblib.load(BytesIO(s3_get_bytes(MODEL_KEY)))
    residuals = np.load(BytesIO(s3_get_bytes(RESIDUALS_KEY)))
    sigma     = residuals.std()
    residuals = np.clip(residuals, -5 * sigma, 5 * sigma)
    print(f"  Model loaded · Residuals: {len(residuals):,} · σ={sigma:.3f}")

    # 2. Load spine
    print("Loading spine from S3...", flush=True)
    spine = s3_get_parquet(SPINE_KEY)
    player_latest, opp_rate = build_spine_lookups(spine)
    print(f"  Spine players: {len(player_latest):,} · Opp rates: {len(opp_rate)}")

    # 3. Fetch today's probable starters
    print("Fetching probable starters from MLB API...", flush=True)
    probables = fetch_probable_starters(gameday)

    # 4. Fetch events + props from Odds API
    print("Fetching events...", flush=True)
    events = fetch_events(gameday)
    if not events:
        msg = f"No MLB events found for {gameday}"
        print(msg)
        send_email(f"MLB Strikeouts — {gameday} — No games", f"<p>{msg}</p>")
        return

    print("Fetching props...", flush=True)
    props = fetch_props(events, gameday)
    print(f"  Raw prop rows: {len(props):,}")

    if props.empty:
        msg = f"No {MARKET} props found for {gameday}"
        print(msg)
        send_email(f"MLB Strikeouts — {gameday} — No props", f"<p>{msg}</p>")
        return

    # 5. Assemble features + score
    print("Scoring bets...", flush=True)
    bets, unmatched = assemble_bet_rows(props, player_latest, probables, opp_rate, gameday, model, residuals)
    n_scored = bets["player_key"].nunique() if not bets.empty else 0

    if unmatched:
        warn_subject  = f"⚠ MLB Strikeouts {gameday} — {len(unmatched)} unmatched player(s)"
        warn_html     = build_warning_html(unmatched, probables, gameday)
        send_email(warn_subject, warn_html)
    print(f"  Pitchers scored: {n_scored}")

    if bets.empty:
        msg = f"No pitchers could be scored for {gameday} (missing spine/props data)"
        print(msg)
        send_email(f"MLB Strikeouts — {gameday} — No pitchers scored", f"<p>{msg}</p>")
        return

    n_primary = int(bets["is_primary"].sum())
    n_watch   = int((bets["side"].notna() & ~bets["is_primary"]).sum())
    print(f"  Pitchers in email: {len(bets)} ({n_primary} plays · {n_watch} watch · {len(bets)-n_primary-n_watch} no edge)")
    print(f"  Primary bets   (UNDER≥{EDGE_THRESHOLD_UNDER:.0%} or OVER≥{EDGE_THRESHOLD_OVER:.0%}): {n_primary}")

    # 6. Save to S3
    save_cols = [
        "player", "player_key", "player_id", "game_date",
        "line", "consensus_line", "bookmaker", "book_abbrev", "side",
        "odds", "odds_u", "raw_p_over", "raw_p_under", "novig_over", "novig_under",
        "n_books_total",
        "k_roll_s5", "k_roll_c5", "k_roll_career", "opp_k_against_season",
        "is_home", "novig_prob_over",
        "yhat", "p_model_over", "p_model_under",
        "p_market_over", "p_market_under",
        "edge_over", "edge_under", "edge", "is_primary",
    ]
    rec_key = f"{DAILY_PREFIX}/{gameday}/recommendations.csv"
    s3_put_csv(rec_key, bets[[c for c in save_cols if c in bets.columns]])
    print(f"  Saved → s3://{S3_BUCKET}/{rec_key}")

    # 7. Load settled history for sections 2 & 3
    yesterday = (pd.Timestamp(gameday) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print("Loading settled history...", flush=True)
    settled_all = load_settled()
    settled_yesterday = (
        settled_all[settled_all["game_date"] == yesterday].copy()
        if not settled_all.empty and "game_date" in settled_all.columns
        else pd.DataFrame()
    )
    print(f"  All-time bets: {len(settled_all)} · Yesterday ({yesterday}): {len(settled_yesterday)}")

    # 8. Email + SNS
    subject   = f"MLB Strikeouts {gameday} — {n_primary} plays · {n_watch} watch · {n_scored} w/ posted lines"
    html_body = build_html(bets, gameday, n_scored, settled_yesterday, settled_all)
    send_email(subject, html_body)

    print(f"\nDone. {n_primary} primary bets, {len(bets)} total.")


if __name__ == "__main__":
    main()
