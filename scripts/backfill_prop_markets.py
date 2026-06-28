"""
Backfill 3 NFL prop markets with all-book coverage for model training.

Markets × seasons:
  player_tackles_assists — 2024, 2025  (no coverage before 2024)
  player_rush_attempts   — 2023, 2024, 2025
  player_reception_yds   — 2023, 2024, 2025

Regions: us, us2  (all US books)
Output:  s3://the-odds-api-mt/nfl/props_backfill/{season}/{nfl_game_id}.parquet

Each parquet contains rows for all applicable markets for that game.
Idempotent — skips games already in S3.

Usage:
  python scripts/backfill_prop_markets.py [--dry-run]
"""

from __future__ import annotations

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
REGIONS       = "us,us2"
SLEEP_S       = 0.1
CREDIT_STOP   = 50_000   # safety floor — well above zero given 5M plan

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "nfl/props_backfill"

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Markets to fetch per season
MARKET_SEASONS: dict[str, list[int]] = {
    "player_tackles_assists": [2024, 2025],
    "player_rush_attempts":   [2023, 2024, 2025],
    "player_reception_yds":   [2023, 2024, 2025],
}

ALL_SEASONS = sorted({s for seasons in MARKET_SEASONS.values() for s in seasons})


class CreditExhausted(Exception):
    pass


def snapshot_utc(gameday: str, gametime: str) -> str:
    gt = gametime if isinstance(gametime, str) and len(gametime) == 5 else "13:00"
    dt_et  = datetime.strptime(f"{gameday} {gt}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    dt_utc = dt_et.astimezone(UTC) - timedelta(minutes=30)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_remaining(headers: dict) -> int:
    try:
        return int(headers.get("x-requests-remaining", 999_999))
    except (ValueError, TypeError):
        return 999_999


def s3_key(season: int, nfl_game_id: str) -> str:
    return f"{S3_PREFIX}/{season}/{nfl_game_id}.parquet"


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
            time.sleep(2 ** (attempt + 1))

    time.sleep(SLEEP_S)
    remaining = parse_remaining(resp.headers)

    if resp.status_code in (404, 422):
        return [], remaining
    resp.raise_for_status()

    rows = []
    data = resp.json().get("data", {})
    for book in data.get("bookmakers", []):
        for mkt in book.get("markets", []):
            for outcome in mkt.get("outcomes", []):
                rows.append({
                    "market":       mkt["key"],
                    "bookmaker":    book["key"],
                    "last_update":  book.get("last_update", ""),
                    "outcome_name": outcome.get("name", ""),
                    "outcome_desc": outcome.get("description", ""),
                    "point":        outcome.get("point"),
                    "price":        outcome.get("price"),
                })

    if remaining < CREDIT_STOP:
        raise CreditExhausted(f"Credits remaining ({remaining:,}) below safety floor ({CREDIT_STOP:,})")

    return rows, remaining


def process_game(s3_client, season: int, row: pd.Series,
                 markets_for_season: list[str], dry_run: bool,
                 idx: int, total: int) -> int:
    nfl_game_id = row["nfl_game_id"]
    event_id    = str(row.get("odds_api_event_id", ""))

    if not event_id or event_id == "nan":
        print(f"  [{idx}/{total}] SKIP {nfl_game_id} — no event_id")
        return -1

    if already_in_s3(s3_client, season, nfl_game_id):
        print(f"  [{idx}/{total}] SKIP {nfl_game_id} — already in S3")
        return -1

    if dry_run:
        print(f"  [{idx}/{total}] DRY  {nfl_game_id}  markets={markets_for_season}")
        return -1

    snapshot = snapshot_utc(row["gameday"], str(row.get("gametime", "13:00")))
    all_rows  = []
    remaining = -1

    for market in markets_for_season:
        rows, remaining = fetch_market(event_id, market, snapshot)
        all_rows.extend(rows)

    if not all_rows:
        print(f"  [{idx}/{total}] EMPTY {nfl_game_id}  remaining={remaining:,}")
        return remaining

    df = pd.DataFrame(all_rows)
    df["nfl_game_id"] = nfl_game_id
    df["season"]      = season
    df["snapshot"]    = snapshot

    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key(season, nfl_game_id), Body=buf.getvalue())

    markets_found = sorted(df["market"].unique())
    books_found   = df["bookmaker"].nunique()
    print(f"  [{idx}/{total}] OK   {nfl_game_id}  "
          f"markets={len(markets_found)}  books={books_found}  remaining={remaining:,}")
    return remaining


def load_event_map(season: int) -> pd.DataFrame:
    path = REPO_ROOT / "data" / "nfl" / f"event_id_map_{season}.csv"
    if not path.exists():
        sys.exit(f"Missing event_id_map for {season}: {path}\n"
                 f"Run the historical backfill script first to build it.")
    df = pd.read_csv(path)
    df = df[df["odds_api_event_id"].notna()].copy()
    return df.sort_values(["gameday", "gametime"]).reset_index(drop=True)


def process_season(s3_client, season: int, dry_run: bool):
    markets = [m for m, seasons in MARKET_SEASONS.items() if season in seasons]
    print(f"\n{'='*65}")
    print(f"  SEASON {season}  |  markets: {markets}")
    print(f"{'='*65}")

    games = load_event_map(season)
    total = len(games)
    print(f"  {total} games with event IDs\n")

    for i, (_, row) in enumerate(games.iterrows(), 1):
        process_game(s3_client, season, row, markets, dry_run, i, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not ODDS_API_KEY and not args.dry_run:
        sys.exit("ODDS_API_KEY not set")

    print("=" * 65)
    print("  NFL PROP MARKET BACKFILL")
    print("=" * 65)
    for market, seasons in MARKET_SEASONS.items():
        print(f"  {market:<30}  seasons={seasons}")
    print(f"\n  Regions  : {REGIONS}")
    print(f"  S3       : s3://{S3_BUCKET}/{S3_PREFIX}/{{season}}/{{game_id}}.parquet")
    print(f"  Seasons  : {ALL_SEASONS}")
    if args.dry_run:
        print("\n  *** DRY RUN ***")
    print()

    s3_client = boto3.client("s3")

    try:
        for season in ALL_SEASONS:
            process_season(s3_client, season, args.dry_run)
    except CreditExhausted as e:
        print(f"\nSTOPPED: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    print("\nDone.")


if __name__ == "__main__":
    main()
