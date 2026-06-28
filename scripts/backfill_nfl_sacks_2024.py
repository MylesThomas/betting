"""
Phase 1: player_sacks backfill for all 2024 NFL REG season games.

Fetches player_sacks lines for every game using all US bookmakers (regions=us).
0-row responses are soft-logged (no hard exit) since 2024 market coverage is
expected to be thinner than 2025 — some games may have had no sacks props posted.

Cache (checked in order):
  1. ~/Downloads/tmp/nfl_defensive_props/2024/{nfl_game_id}.parquet  (local)
  2. s3://the-odds-api-mt/nfl/player_props/player_sacks/2024/{nfl_game_id}.parquet

Output:
  Same two locations above.

Run:
  python scripts/backfill_nfl_sacks_2024.py [--dry-run]
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
S3_PREFIX  = "nfl/player_props/player_sacks/2024"

LOCAL_DIR  = Path.home() / "Downloads" / "tmp" / "nfl_defensive_props" / "2024"

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

    map_path = REPO_ROOT / "data" / "nfl" / "event_id_map_2024.csv"
    games    = pd.read_csv(map_path)
    reg      = games[games["game_type"] == "REG"].sort_values(["gameday", "gametime"]).reset_index(drop=True)

    print(f"REG games   : {len(reg)}")
    print(f"Market      : {MARKET}")
    print(f"Regions     : {REGIONS}")
    print(f"S3          : s3://{S3_BUCKET}/{S3_PREFIX}/")
    print(f"Local cache : {LOCAL_DIR}")
    if args.dry_run:
        print("\n--- DRY RUN ---\n")

    s3_client  = boto3.client("s3")
    no_market  = []
    skipped    = 0
    fetched    = 0
    t_start    = datetime.now()

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

        snap = snapshot_utc(gameday, gametime, offset_min=-30)
        rows, remaining = fetch_sacks(event_id, snap)

        if is_london and not rows:
            snap2 = snapshot_utc(gameday, gametime, offset_min=-120)
            rows, remaining = fetch_sacks(event_id, snap2)
            tag = "LONDON-2h" if rows else "LONDON-EMPTY"
        else:
            tag = "LONDON" if is_london else "OK"

        if not rows:
            KNOWN_NO_MARKET = {
                "2024_01_BAL_KC":  "wk1 opener — sacks props market not yet live",
                "2024_01_GB_PHI":  "wk1 — no sacks props posted",
                "2024_01_WAS_TB":  "wk1 — no sacks props posted",
                "2024_01_LA_DET":  "wk1 — no sacks props posted",
                "2024_08_MIN_LA":  "wk8 — no sacks props posted",
                "2024_11_WAS_PHI": "wk11 — no sacks props posted",
                "2024_14_GB_DET":  "wk14 — no sacks props posted",
                "2024_14_LAC_KC":  "wk14 — no sacks props posted",
                "2024_16_HOU_KC":  "wk16 — no sacks props posted",
                "2024_16_PIT_BAL": "wk16 — no sacks props posted",
                "2024_18_CAR_ATL": "wk18 — irrelevant game, no market",
                "2024_18_JAX_IND": "wk18 — irrelevant game, no market",
                "2024_18_BUF_NE":  "wk18 — irrelevant game, no market",
                "2024_18_HOU_TEN": "wk18 — irrelevant game, no market",
                "2024_18_SF_ARI":  "wk18 — irrelevant game, no market",
                "2024_18_MIN_DET": "wk18 — irrelevant game, no market",
            }
            if game_id in KNOWN_NO_MARKET:
                no_market_path = LOCAL_DIR.parent / "no_market_games_2024.csv"
                no_market_path.parent.mkdir(parents=True, exist_ok=True)
                with open(no_market_path, "a") as f:
                    f.write(f"{game_id},{week},{gameday},{KNOWN_NO_MARKET[game_id]}\n")
                print(f"  [{n:>3}/{total}] wk{week:>2}  NO_MARKET     {game_id}  (logged)")
                no_market.append(game_id)
                continue

            msg = (
                f"\n  FATAL: 0 rows returned for {game_id} (wk{week}, {gameday}).\n"
                f"  Event ID : {event_id}\n"
                f"  Snapshot : {snap}\n"
                f"  Remaining: {remaining}\n"
                f"  If this game had no sacks props, add it to KNOWN_NO_MARKET.\n"
                f"  Completed {fetched} games before failure.\n"
            )
            print(msg)
            sys.exit(1)

        df = pd.DataFrame(rows)
        df["nfl_game_id"]  = game_id
        df["week"]         = week
        df["gameday"]      = gameday
        df["gametime_et"]  = gametime
        df["home_team"]    = row["home_team"]
        df["away_team"]    = row["away_team"]
        df["season"]       = 2024
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

        n_books   = df["bookmaker"].nunique()
        n_players = df["outcome_desc"].nunique()
        print(f"  [{n:>3}/{total}] wk{week:>2}  {tag:<10}  {game_id:<27}  "
              f"books={n_books}  players={n_players}  remaining={remaining}")

    elapsed = datetime.now() - t_start
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Elapsed    : {elapsed}")
    print(f"  Fetched    : {fetched}")
    print(f"  Skipped    : {skipped}")
    print(f"  No market  : {len(no_market)}  ({', '.join(no_market[:5])}{'...' if len(no_market) > 5 else ''})")
    print(f"  Credits est used: ~{fetched * 10}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
