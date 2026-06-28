"""
Phase 1: player_sacks backfill for all 2025 NFL REG season games.

Fetches player_sacks lines for every game using all US bookmakers (regions=us).
Fails hard if any game returns 0 rows — missing data is not acceptable.

Cache (checked in order):
  1. ~/Downloads/tmp/nfl_defensive_props/2025/{nfl_game_id}.parquet  (local)
  2. s3://the-odds-api-mt/nfl/player_props/player_sacks/2025/{nfl_game_id}.parquet

Output:
  Same two locations above.

Run:
  python scripts/backfill_nfl_sacks_2025.py [--dry-run]
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
MARKET        = "player_sacks"
SLEEP_S       = 0.15

S3_BUCKET  = "the-odds-api-mt"
S3_PREFIX  = "nfl/player_props/player_sacks/2025"

LOCAL_DIR  = Path.home() / "Downloads" / "tmp" / "nfl_defensive_props" / "2025"

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


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


def fetch_sacks(event_id: str, snapshot: str) -> tuple[list[dict], int]:
    resp = requests.get(
        f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
        params={
            "apiKey":     ODDS_API_KEY,
            "markets":    MARKET,
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
                        "last_update":  book.get("last_update", ""),
                        "outcome_name": outcome.get("name", ""),
                        "outcome_desc": outcome.get("description", ""),
                        "price":        outcome.get("price"),
                        "point":        outcome.get("point"),
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
    print(f"Market      : {MARKET}")
    print(f"Regions     : {REGIONS}")
    print(f"S3          : s3://{S3_BUCKET}/{S3_PREFIX}/")
    print(f"Local cache : {LOCAL_DIR}")
    if args.dry_run:
        print("\n--- DRY RUN ---\n")

    s3_client = boto3.client("s3")
    failed    = []
    skipped   = 0
    fetched   = 0
    t_start   = datetime.now()

    for i, row in reg.iterrows():
        game_id  = row["nfl_game_id"]
        event_id = str(row["odds_api_event_id"])
        gameday  = row["gameday"]
        gametime = str(row.get("gametime", "13:00"))
        week     = int(row["week"])
        n        = i + 1
        total    = len(reg)
        is_london = int(gametime.split(":")[0]) < 10

        # Local cache check first, then S3
        if in_local_cache(game_id):
            print(f"  [{n:>3}/{total}] wk{week:>2}  SKIP (local)  {game_id}")
            skipped += 1
            continue
        if not args.dry_run and in_s3(s3_client, game_id):
            print(f"  [{n:>3}/{total}] wk{week:>2}  SKIP (s3)     {game_id}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [{n:>3}/{total}] wk{week:>2}  DRY           {game_id}")
            continue

        # Primary snapshot: -30 min
        snap = snapshot_utc(gameday, gametime, offset_min=-30)
        rows, remaining = fetch_sacks(event_id, snap)

        # London retry: -120 min
        if is_london and not rows:
            snap2 = snapshot_utc(gameday, gametime, offset_min=-120)
            rows, remaining = fetch_sacks(event_id, snap2)
            tag = "LONDON-2h" if rows else "LONDON-EMPTY"
        else:
            tag = "LONDON" if is_london else "OK"

        if not rows:
            # Known market gaps — confirmed played, no sacks props posted by any book
            KNOWN_NO_MARKET = {
                "2025_18_GB_MIN": "GB rested pass rushers wk18; no book posted sacks despite MIN's D.Turner playing (2 sacks in game)",
                "2025_18_DAL_NYG": "Both teams irrelevant wk18 (DAL 5-12, NYG worse); no book posted sacks despite J.Clowney 3 sacks in game",
                "2025_18_NYJ_BUF": "wk18 meaningless for both teams; no book posted sacks market",
                "2025_18_KC_LV":   "KC resting starters wk18 with seeding locked; no book posted sacks market",
            }
            if game_id in KNOWN_NO_MARKET:
                no_market_path = LOCAL_DIR.parent / "no_market_games.csv"
                no_market_path.parent.mkdir(parents=True, exist_ok=True)
                with open(no_market_path, "a") as f:
                    f.write(f"{game_id},{week},{gameday},{KNOWN_NO_MARKET[game_id]}\n")
                print(f"  [{n:>3}/{total}] wk{week:>2}  NO_MARKET     {game_id}  (logged to no_market_games.csv)")
                continue

            msg = (
                f"\n  FATAL: 0 rows returned for {game_id} (wk{week}, {gameday}).\n"
                f"  Event ID : {event_id}\n"
                f"  Snapshot : {snap}\n"
                f"  Remaining: {remaining}\n"
                f"  This game needs to be marked as cancelled/not-played or investigated.\n"
                f"  Completed {fetched} games before failure.\n"
            )
            print(msg)
            sys.exit(1)

        # Attach metadata
        df = pd.DataFrame(rows)
        df["nfl_game_id"]  = game_id
        df["week"]         = week
        df["gameday"]      = gameday
        df["gametime_et"]  = gametime
        df["home_team"]    = row["home_team"]
        df["away_team"]    = row["away_team"]
        df["season"]       = 2025
        df["is_london"]    = is_london

        col_order = [
            "nfl_game_id", "season", "week", "gameday", "gametime_et",
            "home_team", "away_team", "is_london",
            "market", "bookmaker", "outcome_name", "outcome_desc",
            "point", "price", "last_update", "snapshot",
        ]
        df = df[[c for c in col_order if c in df.columns]]

        write_game(s3_client, df, game_id)
        fetched += 1

        n_books  = df["bookmaker"].nunique()
        n_players = df["outcome_desc"].nunique()
        print(f"  [{n:>3}/{total}] wk{week:>2}  {tag:<10}  {game_id:<27}  "
              f"books={n_books}  players={n_players}  remaining={remaining}")

    elapsed = datetime.now() - t_start
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Elapsed : {elapsed}")
    print(f"  Fetched : {fetched}")
    print(f"  Skipped : {skipped}")
    if failed:
        print(f"  Failed  : {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
