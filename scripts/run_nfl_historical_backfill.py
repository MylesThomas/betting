"""
Backfill all NFL markets (DraftKings + Bovada) for every game, season by season.

Starts at 2025 and walks back through 2020, stopping automatically when credits
fall below CREDIT_STOP_THRESHOLD. Fully idempotent — safe to re-run after any
interruption.

Usage:
  python scripts/run_nfl_historical_backfill.py [--dry-run] [--start-season 2024]

Output:
  data/nfl/event_id_map_{season}.csv          (local cache, rebuilt if missing)
  s3://the-odds-api-mt/nfl/all_markets/{season}/{nfl_game_id}.parquet

Credit budget:
  ~83 calls/game × 285 games/season ≈ 23,655 calls/season
  With 90k remaining: ~3-4 full seasons before hitting the stop threshold.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────
ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "americanfootball_nfl"
BOOKMAKERS    = ["bovada"]
REGIONS       = "us2"
SLEEP_S       = 0.05

S3_BUCKET          = "the-odds-api-mt"
CREDIT_STOP_THRESHOLD = 200   # stop when fewer than this many credits remain
LOCAL_TMP          = Path.home() / "Downloads" / "tmp"

SEASONS = [2025, 2024, 2023, 2022, 2021, 2020]

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

TEAM_NAME_MAP = {
    "ARI": "Arizona Cardinals",  "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",   "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",  "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",     "DEN": "Denver Broncos",
    "DET": "Detroit Lions",      "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",     "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams",
    "LV":  "Las Vegas Raiders",  "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",  "NE":  "New England Patriots",
    "NO":  "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets",      "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers","SF":  "San Francisco 49ers",
    "SEA": "Seattle Seahawks",   "TB":  "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",   "WAS": "Washington Commanders",
}

ALL_MARKETS = [

    # ── game lines ────────────────────────────────────────────────────────────
    "h2h",
    "spreads",
    "totals",
    # "alternate_spreads",       # CUT: alt lines
    # "alternate_totals",        # CUT: alt lines
    # "team_totals",
    # "alternate_team_totals",   # CUT: alt lines

    # ── quarter lines ─────────────────────────────────────────────────────────
    # "h2h_q1", "h2h_q2", "h2h_q3", "h2h_q4",
    # "h2h_3_way_q1", "h2h_3_way_q2", "h2h_3_way_q3", "h2h_3_way_q4",  # CUT: NFL draws are rare
    # "spreads_q1", "spreads_q2", "spreads_q3", "spreads_q4",
    # "alternate_spreads_q1", "alternate_spreads_q2", "alternate_spreads_q3",  # CUT: alt lines
    # "totals_q1", "totals_q2", "totals_q3", "totals_q4",
    # "alternate_totals_q1", "alternate_totals_q2", "alternate_totals_q3",     # CUT: alt lines
    # "team_totals_q1", "team_totals_q2", "team_totals_q3", "team_totals_q4",
    # "alternate_team_totals_q1", "alternate_team_totals_q2",                  # CUT: alt lines

    # ── half lines ────────────────────────────────────────────────────────────
    # "h2h_h1", "h2h_h2",
    # "h2h_3_way_h1", "h2h_3_way_h2",  # CUT: NFL draws are rare
    # "spreads_h1", "spreads_h2",
    # "alternate_spreads_h1", "alternate_spreads_h2",  # CUT: alt lines
    # "totals_h1", "totals_h2",
    # "alternate_totals_h1", "alternate_totals_h2",    # CUT: alt lines
    # "team_totals_h1", "team_totals_h2",
    # "alternate_team_totals_h1", "alternate_team_totals_h2",  # CUT: alt lines

    # ── passing (QB context for WR target share / air yards) ─────────────────
    "player_pass_yds",
    # "player_pass_yds_alternate",          # CUT: alt lines
    "player_pass_attempts",
    # "player_pass_attempts_alternate",     # CUT: alt lines
    "player_pass_completions",
    # "player_pass_completions_alternate",  # CUT: alt lines
    "player_pass_tds",
    # "player_pass_tds_alternate",          # CUT: alt lines
    "player_pass_interceptions",
    # "player_pass_interceptions_alternate", # CUT: alt lines

    # ── rushing ───────────────────────────────────────────────────────────────
    "player_rush_yds",
    # "player_rush_yds_alternate",      # CUT: alt lines
    "player_rush_attempts",
    # "player_rush_attempts_alternate", # CUT: alt lines

    # ── receiving (core WR signal) ────────────────────────────────────────────
    "player_reception_yds",
    # "player_reception_yds_alternate", # CUT: alt lines
    "player_receptions",
    # "player_receptions_alternate",    # CUT: alt lines

    # ── combo yards (usage share across play types) ───────────────────────────
    "player_pass_rush_yds",       # QB total yards (pass + rush)
    # "player_pass_rush_yds_alternate",          # CUT: alt lines
    "player_rush_reception_yds",  # RB/WR total yards (rush + receiving)
    # "player_rush_reception_yds_alternate",     # CUT: alt lines

    # ── touchdowns ────────────────────────────────────────────────────────────
    # "player_tds_over",    # TD count over/under
    # "player_anytime_td",  # TD probability
    # "player_1st_td",    # CUT: ordering/luck
    # "player_last_td",   # CUT: ordering/luck

    # ── defensive ─────────────────────────────────────────────────────────────
    "player_tackles_assists",
    "player_sacks",

]


class CreditExhausted(Exception):
    pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def snapshot_utc(gameday: str, gametime_et: str) -> str:
    dt_et  = datetime.strptime(f"{gameday} {gametime_et}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    dt_utc = dt_et.astimezone(UTC) - timedelta(minutes=30)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_remaining(headers: dict) -> int:
    val = headers.get("x-requests-remaining", "")
    try:
        return int(val)
    except (ValueError, TypeError):
        return 999_999


def guard_credits(remaining: int):
    if remaining < CREDIT_STOP_THRESHOLD:
        raise CreditExhausted(f"Credits remaining ({remaining}) below threshold ({CREDIT_STOP_THRESHOLD})")


# ── Step 1: build event_id map ─────────────────────────────────────────────────

def build_event_id_map(season: int, dry_run: bool) -> pd.DataFrame | None:
    import nfl_data_py as nfl

    out_path = REPO_ROOT / "data" / "nfl" / f"event_id_map_{season}.csv"
    if out_path.exists():
        print(f"  [event_id_map] loaded from cache: {out_path.name}")
        return pd.read_csv(out_path)

    sched = nfl.import_schedules([season])
    sched = sched[sched["gameday"].notna()].copy()
    sched["home_full"] = sched["home_team"].map(TEAM_NAME_MAP)
    sched["away_full"] = sched["away_team"].map(TEAM_NAME_MAP)

    game_days = sorted(sched["gameday"].unique())
    print(f"  [event_id_map] {len(sched)} games / {len(game_days)} game days — fetching IDs...")

    full_name_to_event: dict[tuple, str] = {}

    for gameday in game_days:
        if dry_run:
            print(f"    DRY {gameday}")
            continue

        snapshot = f"{gameday}T12:00:00Z"
        for attempt in range(3):
            try:
                resp = requests.get(
                    f"{ODDS_API_BASE}/historical/sports/{SPORT}/odds",
                    params={"apiKey": ODDS_API_KEY, "markets": "h2h", "regions": REGIONS,
                            "oddsFormat": "american", "dateFormat": "iso", "date": snapshot},
                    timeout=60,
                )
                break
            except requests.exceptions.Timeout:
                if attempt == 2:
                    raise
                wait = 2 ** (attempt + 1)
                print(f"    timeout on {gameday}, retrying in {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)
        time.sleep(SLEEP_S)

        remaining = parse_remaining(resp.headers)
        if resp.status_code in (404, 422):
            print(f"    {gameday} → no data")
            continue
        resp.raise_for_status()
        guard_credits(remaining)

        events = resp.json().get("data", [])
        for e in events:
            full_name_to_event[(e.get("home_team", ""), e.get("away_team", ""))] = e["id"]
        print(f"    {gameday} → {len(events)} events  remaining={remaining}")

    if dry_run:
        return None

    sched["odds_api_event_id"] = sched.apply(
        lambda r: full_name_to_event.get((r["home_full"], r["away_full"])), axis=1
    )
    out = sched[["game_id", "odds_api_event_id", "home_team", "away_team",
                 "gameday", "gametime", "game_type", "week"]].rename(columns={"game_id": "nfl_game_id"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    matched = out["odds_api_event_id"].notna().sum()
    print(f"  [event_id_map] matched {matched}/{len(out)} → saved {out_path.name}")
    return out


# ── Step 2: fetch all markets per game ─────────────────────────────────────────

def s3_key(season: int, nfl_game_id: str) -> str:
    return f"nfl/all_markets/{season}/{nfl_game_id}.parquet"


def already_in_s3(s3_client, season: int, nfl_game_id: str) -> bool:
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key(season, nfl_game_id))
        return True
    except Exception:
        return False


def fetch_market(event_id: str, market: str, snapshot: str) -> tuple[list[dict], int]:
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
                params={
                    "apiKey":     ODDS_API_KEY,
                    "markets":    market,
                    "bookmakers": ",".join(BOOKMAKERS),
                    "regions":    REGIONS,
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                    "date":       snapshot,
                },
                timeout=60,
            )
            break
        except requests.exceptions.Timeout:
            if attempt == 2:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    timeout on {market}, retrying in {wait}s (attempt {attempt+1}/3)...")
            time.sleep(wait)
    time.sleep(SLEEP_S)
    remaining = parse_remaining(resp.headers)

    if resp.status_code in (404, 422):
        return [], remaining
    resp.raise_for_status()

    data = resp.json().get("data", {})
    rows = []
    if data:
        for book in data.get("bookmakers", []):
            for mkt in book.get("markets", []):
                for outcome in mkt.get("outcomes", []):
                    rows.append({
                        "odds_api_event_id": data.get("id", ""),
                        "home_team":         data.get("home_team", ""),
                        "away_team":         data.get("away_team", ""),
                        "commence_time":     data.get("commence_time", ""),
                        "snapshot_time":     snapshot,
                        "market":            mkt["key"],
                        "bookmaker":         book["key"],
                        "last_update":       book.get("last_update", ""),
                        "outcome_name":      outcome.get("name", ""),
                        "outcome_desc":      outcome.get("description", ""),
                        "price":             outcome.get("price"),
                        "point":             outcome.get("point"),
                    })
    return rows, remaining


def process_game(s3_client, season: int, row: pd.Series, dry_run: bool,
                 games_done: int, games_total: int) -> tuple[int, pd.DataFrame | None]:
    """Returns (credits_remaining, df). df is None if skipped/empty."""
    nfl_game_id = row["nfl_game_id"]
    event_id    = str(row.get("odds_api_event_id", ""))
    gametime    = str(row.get("gametime", "13:00"))

    if not event_id or event_id == "nan":
        print(f"  [{games_done}/{games_total}] SKIP {nfl_game_id} — no event_id")
        return -1, None

    if already_in_s3(s3_client, season, nfl_game_id):
        print(f"  [{games_done}/{games_total}] SKIP {nfl_game_id} — already in S3")
        return -1, None

    if dry_run:
        print(f"  [{games_done}/{games_total}] DRY  {nfl_game_id}")
        return -1, None

    snapshot  = snapshot_utc(row["gameday"], gametime)
    all_rows  = []
    remaining = -1

    for market in ALL_MARKETS:
        rows, remaining = fetch_market(event_id, market, snapshot)
        all_rows.extend(rows)
        guard_credits(remaining)

    if not all_rows:
        print(f"  [{games_done}/{games_total}] EMPTY {nfl_game_id}  remaining={remaining}")
        return remaining, None

    df = pd.DataFrame(all_rows)
    df["nfl_game_id"] = nfl_game_id
    df["season"]      = season

    # S3
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key(season, nfl_game_id), Body=buf.getvalue())

    # Local — write immediately so partial runs aren't lost
    local_dir = LOCAL_TMP / "nfl_all_markets" / str(season)
    local_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(local_dir / f"{nfl_game_id}.parquet", index=False)

    n_markets = df["market"].nunique()
    n_books   = df["bookmaker"].nunique()
    print(f"  [{games_done}/{games_total}] OK   {nfl_game_id}  "
          f"markets={n_markets}  books={n_books}  remaining={remaining}")
    return remaining, df


def process_season(s3_client, season: int, dry_run: bool) -> int:
    """Returns last known credits remaining. Raises CreditExhausted if budget hit."""
    import nfl_data_py as nfl

    print(f"\n{'='*60}")
    print(f"  SEASON {season}")
    print(f"{'='*60}")

    games = build_event_id_map(season, dry_run)
    if games is None:
        return -1

    # Attach gametime if not already in map (older cache files may lack it)
    if "gametime" not in games.columns:
        sched = nfl.import_schedules([season])[["game_id", "gametime"]].rename(
            columns={"game_id": "nfl_game_id"})
        games = games.merge(sched, on="nfl_game_id", how="left")

    games = games.sort_values(["gameday", "gametime"]).reset_index(drop=True)
    total = len(games)
    remaining = -1
    season_frames = []

    print(f"\n  Fetching markets for {total} games → s3://{S3_BUCKET}/nfl/all_markets/{season}/\n")

    for i, (_, row) in enumerate(games.iterrows(), 1):
        remaining, df = process_game(s3_client, season, row, dry_run, i, total)
        if df is not None:
            season_frames.append(df)

    # Write combined season parquet once all games are done
    if season_frames and not dry_run:
        combined_path = LOCAL_TMP / "nfl_all_markets" / f"nfl_all_markets_{season}_combined.parquet"
        pd.concat(season_frames, ignore_index=True).to_parquet(combined_path, index=False)
        print(f"\n  Combined → {combined_path}  ({len(season_frames)} games)")

    return remaining


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--start-season", type=int, default=2025,
                        help="Start from this season (default 2025, walks back to 2020)")
    args = parser.parse_args()

    if not ODDS_API_KEY and not args.dry_run:
        sys.exit("ODDS_API_KEY not set — add it to .env")

    seasons_to_run = [s for s in SEASONS if s <= args.start_season]

    print(f"Seasons   : {seasons_to_run}")
    print(f"Markets   : {len(ALL_MARKETS)} per game")
    print(f"Bookmakers: {BOOKMAKERS}")
    print(f"Stop at   : <{CREDIT_STOP_THRESHOLD:,} credits remaining")
    print(f"S3 bucket : {S3_BUCKET}")
    if args.dry_run:
        print("\n--- DRY RUN ---")

    s3_client = boto3.client("s3")
    completed = []
    stopped_at = None
    t_start = datetime.now()

    try:
        for season in seasons_to_run:
            process_season(s3_client, season, args.dry_run)
            completed.append(season)

    except CreditExhausted as e:
        stopped_at = str(e)
    except KeyboardInterrupt:
        stopped_at = "interrupted by user"

    elapsed = datetime.now() - t_start
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Elapsed  : {elapsed}")
    print(f"  Completed: {completed}")
    if stopped_at:
        print(f"  Stopped  : {stopped_at}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
