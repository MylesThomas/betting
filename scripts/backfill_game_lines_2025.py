"""
Backfill game totals + spreads for all 272 NFL REG 2025 games.
Snapshot: -30 min before kickoff (same as player props).
Markets: totals, spreads

Cache (checked in order):
  1. ~/Downloads/tmp/nfl_game_lines/2025/{nfl_game_id}.parquet  (local)
  2. s3://the-odds-api-mt/nfl/game_lines/2025/{nfl_game_id}.parquet

Output:
  Same two locations above.

Run:
  python scripts/backfill_game_lines_2025.py [--dry-run]
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

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "americanfootball_nfl"
REGIONS       = "us"
MARKETS       = "totals,spreads"
SLEEP_S       = 0.15

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "nfl/game_lines/2025"
LOCAL_DIR = Path.home() / "Downloads" / "tmp" / "nfl_game_lines" / "2025"

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Full Odds API team name → NFL abbreviation (all 32 teams)
TEAM_NAME_MAP = {
    "Arizona Cardinals":      "ARI",
    "Atlanta Falcons":        "ATL",
    "Baltimore Ravens":       "BAL",
    "Buffalo Bills":          "BUF",
    "Carolina Panthers":      "CAR",
    "Chicago Bears":          "CHI",
    "Cincinnati Bengals":     "CIN",
    "Cleveland Browns":       "CLE",
    "Dallas Cowboys":         "DAL",
    "Denver Broncos":         "DEN",
    "Detroit Lions":          "DET",
    "Green Bay Packers":      "GB",
    "Houston Texans":         "HOU",
    "Indianapolis Colts":     "IND",
    "Jacksonville Jaguars":   "JAX",
    "Kansas City Chiefs":     "KC",
    "Las Vegas Raiders":      "LV",
    "Los Angeles Chargers":   "LAC",
    "Los Angeles Rams":       "LA",
    "Miami Dolphins":         "MIA",
    "Minnesota Vikings":      "MIN",
    "New England Patriots":   "NE",
    "New Orleans Saints":     "NO",
    "New York Giants":        "NYG",
    "New York Jets":          "NYJ",
    "Philadelphia Eagles":    "PHI",
    "Pittsburgh Steelers":    "PIT",
    "San Francisco 49ers":    "SF",
    "Seattle Seahawks":       "SEA",
    "Tampa Bay Buccaneers":   "TB",
    "Tennessee Titans":       "TEN",
    "Washington Commanders":  "WAS",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def snapshot_utc(gameday: str, gametime_et: str, offset_min: int = -30) -> str:
    dt_et = datetime.strptime(f"{gameday} {gametime_et}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    dt_utc = dt_et.astimezone(UTC) + timedelta(minutes=offset_min)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_remaining(headers: dict) -> int:
    try:
        return int(headers.get("x-requests-remaining", 999_999))
    except (ValueError, TypeError):
        return 999_999


def fetch_lines(event_id: str, snapshot: str) -> tuple[list[dict], int]:
    resp = requests.get(
        f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
        params={
            "apiKey":     ODDS_API_KEY,
            "markets":    MARKETS,
            "regions":    REGIONS,
            "oddsFormat": "american",
            "dateFormat": "iso",
            "date":       snapshot,
        },
        timeout=60,
    )
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
                        "market":       mkt["key"],
                        "bookmaker":    book["key"],
                        "outcome_name": outcome.get("name", ""),
                        "point":        outcome.get("point"),
                        "price":        outcome.get("price"),
                        "snapshot":     snapshot,
                    })
    return rows, remaining


def s3_key(nfl_game_id: str) -> str:
    return f"{S3_PREFIX}/{nfl_game_id}.parquet"


def in_local_cache(nfl_game_id: str) -> bool:
    return (LOCAL_DIR / f"{nfl_game_id}.parquet").exists()


def in_s3(s3_client, nfl_game_id: str) -> bool:
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key(nfl_game_id))
        return True
    except Exception:
        return False


def write_game(s3_client, df: pd.DataFrame, nfl_game_id: str):
    local_path = LOCAL_DIR / f"{nfl_game_id}.parquet"
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(local_path, index=False)

    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key(nfl_game_id), Body=buf.getvalue())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not ODDS_API_KEY and not args.dry_run:
        sys.exit("ODDS_API_KEY not set — add it to .env")

    map_path = REPO_ROOT / "data" / "nfl" / "event_id_map_2025.csv"
    games    = pd.read_csv(map_path)
    reg      = games[games["game_type"] == "REG"].sort_values(["gameday", "gametime"]).reset_index(drop=True)

    print(f"REG games   : {len(reg)}")
    print(f"Markets     : {MARKETS}")
    print(f"Regions     : {REGIONS}")
    print(f"S3          : s3://{S3_BUCKET}/{S3_PREFIX}/")
    print(f"Local cache : {LOCAL_DIR}")
    if args.dry_run:
        print("\n--- DRY RUN ---\n")

    s3_client = boto3.client("s3")
    skipped = fetched = 0
    unknown_names: set[str] = set()

    for i, row in reg.iterrows():
        game_id  = row["nfl_game_id"]
        event_id = str(row["odds_api_event_id"])
        gameday  = row["gameday"]
        gametime = str(row.get("gametime", "13:00"))
        week     = int(row["week"])
        n        = i + 1
        total    = len(reg)
        is_london = int(gametime.split(":")[0]) < 10

        if in_local_cache(game_id):
            skipped += 1
            print(f"  [{n:>3}/{total}] wk{week:>2}  SKIP (local)  {game_id}")
            continue
        if not args.dry_run and in_s3(s3_client, game_id):
            skipped += 1
            print(f"  [{n:>3}/{total}] wk{week:>2}  SKIP (s3)     {game_id}")
            continue
        if args.dry_run:
            print(f"  [{n:>3}/{total}] wk{week:>2}  DRY           {game_id}")
            continue

        snap = snapshot_utc(gameday, gametime, offset_min=-30)
        rows, remaining = fetch_lines(event_id, snap)

        if is_london and not rows:
            snap = snapshot_utc(gameday, gametime, offset_min=-120)
            rows, remaining = fetch_lines(event_id, snap)

        if not rows:
            print(f"  [{n:>3}/{total}] wk{week:>2}  WARN: 0 rows  {game_id}  (skipping — investigate)")
            continue

        df = pd.DataFrame(rows)
        df["nfl_game_id"] = game_id
        df["week"]        = week
        df["gameday"]     = gameday
        df["gametime_et"] = gametime
        df["home_team"]   = row["home_team"]
        df["away_team"]   = row["away_team"]
        df["season"]      = 2025

        # Validate team name mapping on spreads
        sp_names = set(df.loc[df["market"] == "spreads", "outcome_name"].unique())
        for name in sp_names:
            if name not in TEAM_NAME_MAP:
                unknown_names.add(name)

        col_order = ["nfl_game_id", "season", "week", "gameday", "gametime_et",
                     "home_team", "away_team", "market", "bookmaker",
                     "outcome_name", "point", "price", "snapshot"]
        df = df[[c for c in col_order if c in df.columns]]

        write_game(s3_client, df, game_id)
        fetched += 1

        n_books = df["bookmaker"].nunique()
        mkts    = sorted(df["market"].unique())
        print(f"  [{n:>3}/{total}] wk{week:>2}  {game_id:<27}  books={n_books}  markets={mkts}  remaining={remaining}")

    print(f"\n{'='*60}")
    print(f"  DONE  |  fetched={fetched}  skipped={skipped}")
    if unknown_names:
        print(f"\n  UNKNOWN team names in spreads (add to TEAM_NAME_MAP):")
        for name in sorted(unknown_names):
            print(f"    '{name}'")
    else:
        print(f"  All spread team names resolved via TEAM_NAME_MAP ✓")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
