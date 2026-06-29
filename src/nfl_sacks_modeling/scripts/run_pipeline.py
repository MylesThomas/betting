"""
Live gameday pipeline for NFL sacks props.

For each player with a sacks prop on the given gameday:
  1. Fetches live event IDs, sacks props, and game lines from The Odds API
  2. Joins with rolling features from the spine (downloaded from S3)
  3. Scores with the LR model (downloaded from S3)
  4. Filters to prob < threshold (Under 0.5 sacks bets)
  5. Uploads HTML + CSV bet sheet to S3 and sends SNS notification

S3 paths read:
  s3://the-odds-api-mt/nfl/sacks_model/spine/nfl_sacks_historical_spine.parquet
  s3://the-odds-api-mt/nfl/sacks_model/model/lr_model.pkl

S3 paths written:
  s3://the-odds-api-mt/nfl/sacks_model/daily_runs/{gameday}/bet_sheet.csv
  s3://the-odds-api-mt/nfl/sacks_model/daily_runs/{gameday}/bet_sheet.html

Run:
  python src/nfl_sacks_modeling/scripts/run_pipeline.py --gameday 2026-09-10
  python src/nfl_sacks_modeling/scripts/run_pipeline.py  # defaults to today ET
"""

import argparse
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
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

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "americanfootball_nfl"
REGIONS       = "us"
SLEEP_S       = 0.20

S3_BUCKET     = "the-odds-api-mt"
S3_PREFIX     = "nfl/sacks_model"
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()
SES_SOURCE    = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SES_TO", "").strip()

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

_BIN_EDGES  = list(range(0, 110, 10))
_BIN_LABELS = [f"{i}-{i+10}" for i in range(0, 100, 10)]

BOOK_CONFIGS = [
    ("fanduel",     "Over",  0.50, "fanduel_over_0p5"),
    ("betonlineag", "Over",  0.50, "betonline_over_0p5"),
    ("betonlineag", "Under", 0.50, "betonline_under_0p5"),
    ("draftkings",  "Over",  0.25, "draftkings_over_0p25"),
    ("draftkings",  "Under", 0.25, "draftkings_under_0p25"),
]

TEAM_NAME_MAP = {
    "Arizona Cardinals":     "ARI", "Atlanta Falcons":       "ATL",
    "Baltimore Ravens":      "BAL", "Buffalo Bills":         "BUF",
    "Carolina Panthers":     "CAR", "Chicago Bears":         "CHI",
    "Cincinnati Bengals":    "CIN", "Cleveland Browns":      "CLE",
    "Dallas Cowboys":        "DAL", "Denver Broncos":        "DEN",
    "Detroit Lions":         "DET", "Green Bay Packers":     "GB",
    "Houston Texans":        "HOU", "Indianapolis Colts":    "IND",
    "Jacksonville Jaguars":  "JAX", "Kansas City Chiefs":    "KC",
    "Las Vegas Raiders":     "LV",  "Los Angeles Chargers":  "LAC",
    "Los Angeles Rams":      "LA",  "Miami Dolphins":        "MIA",
    "Minnesota Vikings":     "MIN", "New England Patriots":  "NE",
    "New Orleans Saints":    "NO",  "New York Giants":       "NYG",
    "New York Jets":         "NYJ", "Philadelphia Eagles":   "PHI",
    "Pittsburgh Steelers":   "PIT", "San Francisco 49ers":   "SF",
    "Seattle Seahawks":      "SEA", "Tampa Bay Buccaneers":  "TB",
    "Tennessee Titans":      "TEN", "Washington Commanders": "WAS",
}

ABBR_TO_TEAM = {v: k for k, v in TEAM_NAME_MAP.items()}


# ── Name normalisation ─────────────────────────────────────────────────────────

_DOTS_RE = re.compile(r"(?<=[A-Za-z])\.")


def _load_name_norm_config() -> tuple[re.Pattern, dict[str, str]]:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f).get("player_name_normalization", {})
    suffixes  = cfg.get("strip_suffixes", [])
    pattern   = r"\s+(" + "|".join(re.escape(s) for s in suffixes) + r")$"
    suffix_re = re.compile(pattern, re.IGNORECASE)
    aliases   = {k.lower(): v.lower() for k, v in cfg.get("aliases", {}).items()}
    return suffix_re, aliases


_SUFFIX_RE, _NAME_ALIASES = _load_name_norm_config()


def _normalize(name: str) -> str:
    name = _SUFFIX_RE.sub("", name.strip())
    name = _DOTS_RE.sub("", name)
    name = name.lower()
    return _NAME_ALIASES.get(name, name)


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["nfl_sacks_model"]


def feature_lists(cfg: dict) -> tuple[list[str], list[str]]:
    windows = cfg["rolling_windows"]
    rolling = [
        f"{feat}_L{('career' if w >= 999 else w)}"
        for feat in ["sack_rate", "qbhit_rate", "snap_pct"]
        for w in windows
    ]
    market_num = [
        "prop_median_impl_over", "prop_median_impl_under",
        "prop_mean_impl_over",   "prop_mean_impl_under",
        "prop_min_impl_over",    "prop_max_impl_over",
        "prop_min_impl_under",   "prop_max_impl_under",
        "prop_book_spread_over", "prop_book_spread_under",
        "prop_n_books",
        "fanduel_over_0p5_implied",
        "betonline_over_0p5_implied", "betonline_under_0p5_implied",
        "draftkings_over_0p25_implied", "draftkings_under_0p25_implied",
    ]
    market_cat = [
        "prop_median_impl_over_bin", "prop_mean_impl_over_bin",
        "prop_median_impl_under_bin", "prop_mean_impl_under_bin",
    ]
    numeric = rolling + ["game_total", "team_spread", "games_played_ytd"] + market_num
    categorical = ["pos_group", "pos_side"] + market_cat
    return numeric, categorical


# ── Helpers ────────────────────────────────────────────────────────────────────

def implied_prob(price: float) -> float:
    if pd.isna(price):
        return float("nan")
    return abs(price) / (abs(price) + 100.0) if price < 0 else 100.0 / (price + 100.0)


def american_to_decimal(price: float) -> float:
    if pd.isna(price):
        return float("nan")
    return 1 + 100 / abs(price) if price < 0 else 1 + price / 100


def decimal_to_american(d: float) -> float:
    if pd.isna(d) or d <= 1.0:
        return float("nan")
    return (d - 1) * 100 if d >= 2.0 else -100 / (d - 1)


def impl_bin(val: float):
    if pd.isna(val):
        return float("nan")
    return pd.cut([val * 100], bins=_BIN_EDGES, labels=_BIN_LABELS,
                  right=True, include_lowest=True).tolist()[0]


def api_get(url: str, params: dict) -> dict | list:
    resp = requests.get(url, params=params, timeout=30)
    remaining = int(resp.headers.get("x-requests-remaining", 999_999))
    resp.raise_for_status()
    time.sleep(SLEEP_S)
    return resp.json(), remaining


# ── Past-date helpers ─────────────────────────────────────────────────────────

def _infer_season(gameday: str) -> int:
    """Map YYYY-MM-DD to NFL season year (Aug–Dec → same year; Jan–Jul → previous year)."""
    d = date.fromisoformat(gameday)
    return d.year if d.month >= 8 else d.year - 1


def _game_lines_from_parquet(df: pd.DataFrame) -> dict:
    """Reconstruct {team_abbr: {game_total, team_spread}} from a game_lines parquet."""
    totals = df[(df["market"] == "totals") & (df["outcome_name"] == "Over")]["point"]
    game_total = float(np.nanmedian(totals)) if len(totals) else float("nan")

    result = {}
    for team_full, grp in df[df["market"] == "spreads"].groupby("outcome_name"):
        abbr = TEAM_NAME_MAP.get(team_full)
        if abbr:
            result[abbr] = {
                "game_total":  game_total,
                "team_spread": float(np.nanmedian(grp["point"])),
            }
    return result


def _load_past_date_s3(gameday: str, season: int) -> tuple[list[dict], list[dict], dict]:
    """
    Load sacks props + game lines from S3 for a past gameday.
    Returns (events, all_prop_rows, all_game_lines) in the same format
    that the live API path produces.
    """
    s3_map_key = f"nfl/event_id_maps/event_id_map_{season}.csv"
    print(f"  Loading event_id_map from s3://{S3_BUCKET}/{s3_map_key}")
    try:
        body = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=s3_map_key)["Body"].read()
    except Exception as exc:
        raise FileNotFoundError(f"No event_id_map for season {season} in S3: {exc}") from exc

    eid_map = pd.read_csv(BytesIO(body))
    games = eid_map[eid_map["gameday"] == gameday]
    if games.empty:
        return [], [], {}

    s3 = boto3.client("s3")
    events: list[dict] = []
    all_prop_rows: list[dict] = []
    all_game_lines: dict = {}

    for _, g in games.iterrows():
        game_id   = g["nfl_game_id"]
        home_abbr = g["home_team"]   # abbr in event_id_map
        away_abbr = g["away_team"]
        home_full = ABBR_TO_TEAM.get(home_abbr, home_abbr)
        away_full = ABBR_TO_TEAM.get(away_abbr, away_abbr)
        gametime  = str(g.get("gametime", "00:00"))[:5]  # "HH:MM"

        dt_et = datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        commence_utc = dt_et.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        events.append({
            "id":            game_id,
            "home_team":     home_full,
            "away_team":     away_full,
            "commence_time": commence_utc,
        })

        # Sacks props
        props_key = f"nfl/player_props/player_sacks/{season}/{game_id}.parquet"
        try:
            body = s3.get_object(Bucket=S3_BUCKET, Key=props_key)["Body"].read()
            props_df = pd.read_parquet(BytesIO(body))
            for _, row in props_df.iterrows():
                all_prop_rows.append({
                    "event_id":     game_id,
                    "home_team":    home_full,
                    "away_team":    away_full,
                    "bookmaker":    row["bookmaker"],
                    "outcome_name": row["outcome_name"],
                    "outcome_desc": row["outcome_desc"],
                    "point":        row["point"],
                    "price":        row["price"],
                })
        except s3.exceptions.NoSuchKey:
            print(f"  WARNING: no sacks props parquet in S3 for {game_id}")

        # Game lines
        lines_key = f"nfl/game_lines/{season}/{game_id}.parquet"
        try:
            body = s3.get_object(Bucket=S3_BUCKET, Key=lines_key)["Body"].read()
            lines_df = pd.read_parquet(BytesIO(body))
            all_game_lines[game_id] = _game_lines_from_parquet(lines_df)
        except s3.exceptions.NoSuchKey:
            print(f"  WARNING: no game_lines parquet in S3 for {game_id}")
            all_game_lines[game_id] = {}

    return events, all_prop_rows, all_game_lines


# ── Odds API fetches ────────────────────────────────────────────────────────────

def fetch_events_for_date(gameday: str) -> list[dict]:
    """Get NFL event IDs for a specific calendar date (ET)."""
    # Convert gameday to UTC window: cover 6am–6am+1 UTC to span all ET game times
    day_start = datetime.strptime(gameday, "%Y-%m-%d").replace(tzinfo=ET).astimezone(UTC)
    day_end   = day_start + timedelta(hours=28)  # generous window
    events, remaining = api_get(
        f"{ODDS_API_BASE}/sports/{SPORT}/events",
        {
            "apiKey":             ODDS_API_KEY,
            "dateFormat":         "iso",
            "commenceTimeFrom":   day_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commenceTimeTo":     day_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    print(f"  Events for {gameday}: {len(events)} found  (credits remaining: {remaining})")
    return events


def fetch_sacks_props(event_id: str, home: str, away: str) -> list[dict]:
    """Fetch player_sacks odds for a single event (live endpoint)."""
    try:
        data, _ = api_get(
            f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds",
            {
                "apiKey":     ODDS_API_KEY,
                "markets":    "player_sacks",
                "regions":    REGIONS,
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in (404, 422):
            return []
        raise

    rows = []
    for book in data.get("bookmakers", []):
        for mkt in book.get("markets", []):
            if mkt.get("key") != "player_sacks":
                continue
            for outcome in mkt.get("outcomes", []):
                rows.append({
                    "event_id":     event_id,
                    "home_team":    home,
                    "away_team":    away,
                    "bookmaker":    book["key"],
                    "outcome_name": outcome.get("name", ""),
                    "outcome_desc": outcome.get("description", ""),
                    "point":        outcome.get("point"),
                    "price":        outcome.get("price"),
                })
    return rows


def fetch_game_lines(event_id: str, home: str, away: str) -> dict:
    """Fetch spreads + totals for a single event. Returns {team → {game_total, team_spread}}."""
    try:
        data, _ = api_get(
            f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds",
            {
                "apiKey":     ODDS_API_KEY,
                "markets":    "spreads,totals",
                "regions":    REGIONS,
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in (404, 422):
            return {}
        raise

    totals_points = []
    spreads = {}  # team_name → list of spread points
    for book in data.get("bookmakers", []):
        for mkt in book.get("markets", []):
            key = mkt.get("key")
            if key == "totals":
                for outcome in mkt.get("outcomes", []):
                    if outcome.get("name") == "Over":
                        totals_points.append(outcome.get("point", float("nan")))
            elif key == "spreads":
                for outcome in mkt.get("outcomes", []):
                    tname = outcome.get("name", "")
                    if tname not in spreads:
                        spreads[tname] = []
                    spreads[tname].append(outcome.get("point", float("nan")))

    game_total = float(np.nanmedian(totals_points)) if totals_points else float("nan")
    result = {}
    for team_name in [home, away]:
        abbr = TEAM_NAME_MAP.get(team_name)
        if abbr:
            spread_vals = spreads.get(team_name, [])
            team_spread = float(np.nanmedian(spread_vals)) if spread_vals else float("nan")
            result[abbr] = {"game_total": game_total, "team_spread": team_spread}
    return result


# ── Spine rolling features ──────────────────────────────────────────────────────

def load_spine_s3() -> pd.DataFrame:
    print(f"  Downloading spine from s3://{S3_BUCKET}/{S3_PREFIX}/spine/nfl_sacks_historical_spine.parquet...")
    key = f"{S3_PREFIX}/spine/nfl_sacks_historical_spine.parquet"
    body = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def compute_player_rolling(
    spine: pd.DataFrame,
    player_names: list[str],
    windows: list[int],
    target_season: int | None = None,
    target_week: int | None = None,
) -> pd.DataFrame:
    """
    For each player, compute rolling features representing their history
    going INTO the next game (i.e., averages over all completed games).

    target_season / target_week: when set, spine is cut to games BEFORE that
    week so games_played_ytd and rolling windows are correct for past dates.
    In live mode (spine is already a current snapshot) these are left as None.

    Player matching: Odds API name (outcome_desc) → spine player name.
    """
    name_map = {_normalize(name): name for name in spine["player"].unique()}
    pid_map  = spine.groupby("player")["pfr_player_id"].first().to_dict()
    team_map = spine.groupby("pfr_player_id")["team"].last().to_dict()
    pos_map  = spine.groupby("pfr_player_id")["position"].last().to_dict()
    pg_map   = spine.groupby("pfr_player_id")["pos_group"].last().to_dict()
    ps_map   = spine.groupby("pfr_player_id")["pos_side"].last().to_dict()

    cur_season = target_season if target_season is not None else int(spine["season"].max())

    rows = []
    for name in player_names:
        spine_name = name_map.get(_normalize(name))
        if spine_name is None:
            rows.append({"player": name, "matched": False})
            continue

        pid = pid_map.get(spine_name)
        g = spine[spine["pfr_player_id"] == pid].sort_values(["season", "week"]).reset_index(drop=True)

        # Cut to games played BEFORE the target game (past-date mode only)
        if target_season is not None and target_week is not None:
            g = g[
                (g["season"] < target_season) |
                ((g["season"] == target_season) & (g["week"] < target_week))
            ].reset_index(drop=True)

        row = {
            "player":           name,
            "pfr_player_id":    pid,
            "team":             team_map.get(pid, ""),
            "position":         pos_map.get(pid, ""),
            "pos_group":        pg_map.get(pid, "OTH"),
            "pos_side":         ps_map.get(pid, "other"),
            "games_played_ytd": len(g[g["season"] == cur_season]),
            "matched":          True,
        }

        for feat, src_col in [("sack_rate", "sacks"), ("qbhit_rate", "qb_hits"), ("snap_pct", "defense_pct")]:
            series = g[src_col].values.astype(float)
            n = len(series)
            for w in windows:
                wlabel = "career" if w >= 999 else str(w)
                win = min(n, 10_000 if w >= 999 else w)
                row[f"{feat}_L{wlabel}"] = float(np.mean(series[-win:])) if win > 0 else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


# ── Props aggregation ───────────────────────────────────────────────────────────

def aggregate_props(raw_rows: list[dict]) -> pd.DataFrame:
    """
    Aggregate sacks prop rows per player (same logic as build_sacks_features.py).
    Returns one row per (event_id, player_name) with all prop feature columns.
    """
    if not raw_rows:
        return pd.DataFrame()

    df = pd.DataFrame(raw_rows)
    df["implied_prob"] = df["price"].apply(implied_prob)
    df["point"] = pd.to_numeric(df["point"], errors="coerce")

    results = []
    for (event_id, player_name), g in df.groupby(["event_id", "outcome_desc"]):
        over_0_5  = g[(g["outcome_name"] == "Over")  & (g["point"] == 0.5)]
        under_0_5 = g[(g["outcome_name"] == "Under") & (g["point"] == 0.5)]
        over_impl  = over_0_5["implied_prob"]
        under_impl = under_0_5["implied_prob"]
        over_dec   = over_0_5["price"].apply(american_to_decimal)
        under_dec  = under_0_5["price"].apply(american_to_decimal)

        best_price_over  = over_0_5.loc[over_impl.idxmin(),  "price"] if len(over_impl)  else float("nan")
        best_price_under = under_0_5.loc[under_impl.idxmin(), "price"] if len(under_impl) else float("nan")

        median_over  = float(over_impl.median())  if len(over_impl)  else float("nan")
        median_under = float(under_impl.median()) if len(under_impl) else float("nan")
        mean_over    = float(over_impl.mean())    if len(over_impl)  else float("nan")
        mean_under   = float(under_impl.mean())   if len(under_impl) else float("nan")

        book_data = {}
        for book, side, line, prefix in BOOK_CONFIGS:
            brows = g[(g["bookmaker"] == book) & (g["outcome_name"] == side) & (g["point"] == line)]
            if len(brows):
                row = brows.iloc[-1]
                book_data[f"{prefix}_implied"] = row["implied_prob"]
            else:
                book_data[f"{prefix}_implied"] = float("nan")

        results.append({
            "event_id":                   event_id,
            "player_name":                player_name,
            "prop_median_impl_over":      median_over,
            "prop_median_impl_under":     median_under,
            "prop_mean_impl_over":        mean_over,
            "prop_mean_impl_under":       mean_under,
            "prop_min_impl_over":         float(over_impl.min())  if len(over_impl)  else float("nan"),
            "prop_max_impl_over":         float(over_impl.max())  if len(over_impl)  else float("nan"),
            "prop_min_impl_under":        float(under_impl.min()) if len(under_impl) else float("nan"),
            "prop_max_impl_under":        float(under_impl.max()) if len(under_impl) else float("nan"),
            "prop_best_price_over":       best_price_over,
            "prop_best_price_under":      best_price_under,
            "prop_book_spread_over":      float(over_impl.max()  - over_impl.min())  if len(over_impl)  else float("nan"),
            "prop_book_spread_under":     float(under_impl.max() - under_impl.min()) if len(under_impl) else float("nan"),
            "prop_n_books":               g["bookmaker"].nunique(),
            "prop_median_price_over":     decimal_to_american(float(over_dec.median()))  if len(over_dec)  else float("nan"),
            "prop_median_price_under":    decimal_to_american(float(under_dec.median())) if len(under_dec) else float("nan"),
            "prop_median_impl_over_bin":  impl_bin(median_over),
            "prop_mean_impl_over_bin":    impl_bin(mean_over),
            "prop_median_impl_under_bin": impl_bin(median_under),
            "prop_mean_impl_under_bin":   impl_bin(mean_under),
            **book_data,
        })

    return pd.DataFrame(results)


# ── Feature matrix ──────────────────────────────────────────────────────────────

def build_feature_matrix(
    prop_df: pd.DataFrame,
    player_features: pd.DataFrame,
    game_lines: dict,   # event_id → {team_abbr → {game_total, team_spread}}
    events: list[dict], # event metadata
) -> pd.DataFrame:
    """Join prop features + player rolling features + game lines."""
    event_map = {e["id"]: e for e in events}
    home_map  = {e["id"]: TEAM_NAME_MAP.get(e["home_team"], "") for e in events}
    away_map  = {e["id"]: TEAM_NAME_MAP.get(e["away_team"], "") for e in events}

    rows = []
    for _, prop_row in prop_df.iterrows():
        event_id    = prop_row["event_id"]
        player_name = prop_row["player_name"]

        pf = player_features[player_features["player"] == player_name]
        if pf.empty or not pf.iloc[0].get("matched", False):
            continue

        pf = pf.iloc[0]
        team = pf["team"]

        # Game lines for this player's team
        glines = game_lines.get(event_id, {}).get(team, {})

        row = {
            "event_id":          event_id,
            "player":            player_name,
            "team":              team,
            "opponent":          away_map[event_id] if team == home_map[event_id] else home_map[event_id],
            "home_team":         home_map[event_id],
            "away_team":         away_map[event_id],
            "commence_time_utc": event_map[event_id].get("commence_time", ""),
            "position":          pf["position"],
            "pos_group":         pf["pos_group"],
            "pos_side":          pf["pos_side"],
            "games_played_ytd":  pf["games_played_ytd"],
            "game_total":        glines.get("game_total", float("nan")),
            "team_spread":       glines.get("team_spread", float("nan")),
            **{k: prop_row[k] for k in prop_row.index if k not in ("event_id", "player_name")},
        }
        for col in player_features.columns:
            if col not in row and col not in ("player", "matched"):
                row[col] = pf[col]

        rows.append(row)

    return pd.DataFrame(rows)


# ── Scoring ─────────────────────────────────────────────────────────────────────

def load_model_s3(cfg: dict) -> dict:
    artifact = cfg.get("model_artifact", "lr_model.pkl")
    key      = f"{S3_PREFIX}/model/{artifact}"
    print(f"  Downloading model from s3://{S3_BUCKET}/{key}...")
    body = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return joblib.load(BytesIO(body))


def score(df: pd.DataFrame, artifact: dict, cfg: dict) -> pd.DataFrame:
    pipe      = artifact["pipeline"]
    n_cols    = artifact["n_cols"]
    c_cols    = artifact["c_cols"]

    inf       = cfg["inference"]
    threshold = inf["threshold"]
    direction = inf["direction"]
    min_edge  = inf["min_edge"]

    all_cols = n_cols + c_cols
    X = df[[c for c in all_cols if c in df.columns]]
    for col in all_cols:
        if col not in X.columns:
            X[col] = float("nan")
    X = X[all_cols]

    df = df.copy()
    df["p_over"] = pipe.predict_proba(X)[:, 1]

    mkt_over  = df.get("prop_median_impl_over",  pd.Series(float("nan"), index=df.index))
    mkt_under = df.get("prop_median_impl_under", pd.Series(float("nan"), index=df.index))

    bet_under = (
        (direction in ("under", "both")) &
        (df["p_over"] < threshold) &
        ((1 - df["p_over"]) - mkt_under >= min_edge)
    )
    bet_over = (
        (direction in ("over", "both")) &
        (df["p_over"] > (1 - threshold)) &
        (df["p_over"] - mkt_over >= min_edge)
    )

    df["bet"]      = bet_under | bet_over
    df["bet_side"] = ""
    df.loc[bet_under, "bet_side"] = "Under 0.5 sacks"
    df.loc[bet_over,  "bet_side"] = "Over 0.5 sacks"
    return df


# ── Output ──────────────────────────────────────────────────────────────────────

def build_bet_sheet_html(df: pd.DataFrame, gameday: str, n_events: int) -> str:
    bets = df[df["bet"]].copy()
    bets = bets.sort_values("p_over")

    def fmt_odds(p):
        if pd.isna(p):
            return "N/A"
        return f"+{int(p)}" if p > 0 else str(int(p))

    rows_html = ""
    for _, r in bets.iterrows():
        p_over    = r["p_over"]
        bet_side  = r.get("bet_side", "Under 0.5 sacks")
        is_over   = str(bet_side).startswith("Over")
        mkt_impl  = r.get("prop_median_impl_over", float("nan"))
        edge      = (p_over - mkt_impl) if is_over else (mkt_impl - p_over)
        odds_col  = "prop_median_price_over" if is_over else "prop_median_price_under"
        odds      = fmt_odds(r.get(odds_col, float("nan")))
        prob_pct  = f"{p_over:.1%}"
        mkt_pct   = f"{mkt_impl:.1%}" if not pd.isna(mkt_impl) else "N/A"
        edge_str  = f"+{edge:.1%}" if not pd.isna(edge) else "N/A"
        rows_html += f"""
        <tr>
          <td>{r['player']}</td>
          <td>{r['team']} vs {r['opponent']}</td>
          <td>{r['pos_group']}/{r['pos_side']}</td>
          <td style="color:#1d6fa4;font-weight:600">{prob_pct}</td>
          <td>{mkt_pct}</td>
          <td style="color:#2a7d2e;font-weight:600">{edge_str}</td>
          <td>{odds}</td>
          <td style="font-weight:600">{bet_side}</td>
        </tr>"""

    no_bets_msg = "" if len(bets) > 0 else '<p style="color:#888">No qualifying bets today.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NFL Sacks Bet Sheet — {gameday}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; background: #f5f5f5; color: #1a1a1a; }}
h1 {{ font-size: 20px; margin-bottom: 4px; }}
.meta {{ color: #666; font-size: 13px; margin-bottom: 20px; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
th {{ background: #1d2d44; color: #fff; padding: 10px 12px; text-align: left; font-size: 13px; }}
td {{ padding: 9px 12px; font-size: 13px; border-bottom: 1px solid #f0f0f0; }}
tr:hover td {{ background: #f9f9f9; }}
.summary {{ margin-top: 20px; background: #fff; padding: 16px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08); font-size: 13px; }}
</style>
</head>
<body>
<h1>NFL Sacks Bet Sheet — {gameday}</h1>
<div class="meta">
  Generated: {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')} &nbsp;|&nbsp;
  Games checked: {n_events} &nbsp;|&nbsp;
  Players with props: {len(df)} &nbsp;|&nbsp;
  Qualifying bets: {len(bets)}
</div>
{no_bets_msg}
{"" if len(bets) == 0 else f"""
<table>
<thead><tr>
  <th>Player</th><th>Matchup</th><th>Pos</th>
  <th>Model P(Over)</th><th>Market P(Over)</th><th>Edge</th>
  <th>Odds</th><th>Bet</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""}
<div class="summary">
  <strong>Strategy:</strong> direction=both · thresh=0.45 · edge≥0.03<br>
  <strong>OOS performance (2025):</strong> 222 bets · 73.9% hit · +0.1348 EV/unit · +29.92u<br>
  <strong>Model:</strong> lr_norm_2024_2025_v4.pkl — Logistic Regression trained on 2024+2025
</div>
</body>
</html>"""


def upload_to_s3(gameday: str, df: pd.DataFrame, html: str) -> tuple[str, str]:
    prefix = f"{S3_PREFIX}/daily_runs/{gameday}"
    csv_key  = f"{prefix}/bet_sheet.csv"
    html_key = f"{prefix}/bet_sheet.html"

    s3 = boto3.client("s3")
    s3.put_object(Bucket=S3_BUCKET, Key=csv_key,  Body=df.to_csv(index=False).encode())
    s3.put_object(Bucket=S3_BUCKET, Key=html_key, Body=html.encode(), ContentType="text/html")
    return f"s3://{S3_BUCKET}/{csv_key}", f"s3://{S3_BUCKET}/{html_key}"


def send_plays_notification(gameday: str, scored: pd.DataFrame, csv_uri: str, html_uri: str) -> None:
    SCRIPTS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPTS_DIR))
    from settle_sacks import build_plays_html

    n_bets    = int(scored["bet"].sum())
    n_players = len(scored)
    bets      = scored[scored["bet"]].sort_values("p_over")

    subject = (
        f"NFL Sacks — {gameday} — {n_bets} bet{'s' if n_bets != 1 else ''}"
        if n_bets else f"NFL Sacks — {gameday} — No qualifying bets"
    )

    html_body = build_plays_html(scored, gameday)

    lines = [f"NFL Sacks — {gameday}", f"Players with props: {n_players}", f"Qualifying bets: {n_bets}", ""]
    for _, r in bets.iterrows():
        p_over   = r.get("p_over", float("nan"))
        bet_side = r.get("bet_side", "Under 0.5 sacks")
        is_over  = str(bet_side).startswith("Over")
        mkt      = r.get("prop_median_impl_over", float("nan"))
        edge     = (p_over - mkt) if is_over else (mkt - p_over)
        edge_s   = f"+{edge:.1%}" if not pd.isna(edge) else "N/A"
        odds_col = "prop_median_price_over" if is_over else "prop_median_price_under"
        odds     = r.get(odds_col, float("nan"))
        odds_s   = f"{int(odds):+d}" if not pd.isna(odds) else "N/A"
        lines.append(f"  {r['player']:<28} {r['team']:>3} vs {r['opponent']:<3}  "
                     f"model={p_over:.1%}  edge={edge_s}  odds={odds_s}  [{bet_side}]")
    lines += ["", f"CSV : {csv_uri}", f"HTML: {html_uri}"]
    text_body = "\n".join(lines)

    to_list = [a.strip() for a in SES_TO_RAW.split(",") if a.strip()]
    if SES_SOURCE and to_list:
        try:
            boto3.client("ses", region_name="us-east-2").send_email(
                Source=SES_SOURCE,
                Destination={"ToAddresses": to_list},
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": {
                        "Html": {"Data": html_body,  "Charset": "UTF-8"},
                        "Text": {"Data": text_body,  "Charset": "UTF-8"},
                    },
                },
            )
            print(f"  SES plays email sent ({n_bets} bets)")
        except Exception as e:
            print(f"  SES send failed: {e}")

    if SNS_TOPIC_ARN:
        boto3.client("sns").publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject[:100],
            Message=text_body[:256_000],
        )
        print(f"  SNS published ({n_bets} bets)")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", type=str, default=None,
                        help="YYYY-MM-DD (default: today in ET)")
    args = parser.parse_args()
    gameday = args.gameday or datetime.now(ET).strftime("%Y-%m-%d")

    print(f"\nNFL Sacks Pipeline — {gameday}")
    print(f"{'='*60}")

    cfg = load_config()
    windows = cfg["rolling_windows"]
    n_cols, c_cols = feature_lists(cfg)

    is_past = date.fromisoformat(gameday) < date.today()

    # ── 1. Events + props + game lines ────────────────────────────────────────
    if is_past:
        season = _infer_season(gameday)
        print(f"\nPast-date mode — loading from S3 (season={season})...")
        events, all_prop_rows, all_game_lines = _load_past_date_s3(gameday, season)
        if not events:
            print(f"  No games found in event_id_map for {gameday}.")
            return
        for ev in events:
            home_abbr = TEAM_NAME_MAP.get(ev["home_team"], ev["home_team"])
            away_abbr = TEAM_NAME_MAP.get(ev["away_team"], ev["away_team"])
            n_players_ev = len(set(
                r["outcome_desc"] for r in all_prop_rows if r["event_id"] == ev["id"]
            ))
            print(f"  {home_abbr} vs {away_abbr}: {n_players_ev} players with sacks props")
    else:
        if not ODDS_API_KEY:
            sys.exit("ODDS_API_KEY not set — add to .env or environment")
        print("\nFetching events...")
        events = fetch_events_for_date(gameday)
        if not events:
            print(f"  No NFL games found for {gameday}.")
            msg = f"No NFL games on {gameday} — pipeline skipped."
            if SNS_TOPIC_ARN:
                boto3.client("sns").publish(TopicArn=SNS_TOPIC_ARN,
                                            Subject=f"NFL sacks — no games on {gameday}",
                                            Message=msg)
            print(msg)
            return

        print("\nFetching props and game lines...")
        all_prop_rows: list[dict] = []
        all_game_lines: dict = {}

        for ev in events:
            eid   = ev["id"]
            home  = ev.get("home_team", "")
            away  = ev.get("away_team", "")

            prop_rows = fetch_sacks_props(eid, home, away)
            all_prop_rows.extend(prop_rows)

            glines = fetch_game_lines(eid, home, away)
            all_game_lines[eid] = glines

            n_players_ev = len(set(r["outcome_desc"] for r in prop_rows))
            print(f"  {home} vs {away}: {n_players_ev} players with sacks props")

    if not all_prop_rows:
        print(f"\n  No sacks props found for {gameday}.")
        return

    # ── 3. Aggregate props ────────────────────────────────────────────────────
    print("\nAggregating props...")
    prop_df = aggregate_props(all_prop_rows)
    print(f"  {len(prop_df)} player-event prop rows  ({prop_df['player_name'].nunique()} unique players)")

    # ── 4. Spine + rolling features ───────────────────────────────────────────
    print("\nLoading spine and computing rolling features...")
    spine = load_spine_s3()

    # In past-date mode, cut spine to games before this week so rolling
    # features and games_played_ytd reflect history going INTO the game.
    roll_season, roll_week = None, None
    if is_past and events:
        roll_season = _infer_season(gameday)
        roll_week   = int(events[0]["id"].split("_")[1])  # "2025_01_DAL_PHI" → 1

    player_features = compute_player_rolling(
        spine, prop_df["player_name"].tolist(), windows,
        target_season=roll_season, target_week=roll_week,
    )
    matched = player_features["matched"].sum()
    print(f"  Spine players matched: {matched}/{len(player_features)}")

    # ── 5. Feature matrix ─────────────────────────────────────────────────────
    print("\nBuilding feature matrix...")
    feature_df = build_feature_matrix(prop_df, player_features, all_game_lines, events)
    print(f"  Feature rows: {len(feature_df)}")

    if feature_df.empty:
        print("  No scorable rows — check player name matching and spine coverage.")
        return

    # ── 6. Score ──────────────────────────────────────────────────────────────
    print("\nLoading model and scoring...")
    artifact  = load_model_s3(cfg)
    scored    = score(feature_df, artifact, cfg)
    n_bets    = int(scored["bet"].sum())
    inf       = cfg["inference"]
    print(f"  Scored {len(scored)} players  →  {n_bets} bets "
          f"(direction={inf['direction']} thresh={inf['threshold']} edge≥{inf['min_edge']})")

    # ── 7. Output ──────────────────────────────────────────────────────────────
    print("\nGenerating bet sheet and uploading...")
    html = build_bet_sheet_html(scored, gameday, len(events))
    csv_uri, html_uri = upload_to_s3(gameday, scored, html)
    print(f"  CSV : {csv_uri}")
    print(f"  HTML: {html_uri}")

    # ── 8. Notify ──────────────────────────────────────────────────────────────
    send_plays_notification(gameday, scored, csv_uri, html_uri)

    print(f"\n{'='*60}")
    print(f"  Done. {n_bets} qualifying bets on {gameday}.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
