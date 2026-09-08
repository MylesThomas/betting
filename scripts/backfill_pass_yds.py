"""
Backfill player_pass_yds prop lines for all books across 2023–2025 NFL seasons.

Output: s3://the-odds-api-mt/nfl/pass_yds_model/{season}/{nfl_game_id}.parquet

Idempotent — skips games already in S3.

Usage:
  python scripts/backfill_pass_yds.py [--dry-run] [--season 2024]
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
MARKET        = "player_pass_yds"
SLEEP_S       = 0.1
CREDIT_STOP   = 50_000

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "nfl/pass_yds_model"
SEASONS   = [2023, 2024, 2025]

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


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


def books_in_s3(s3_client, season: int, nfl_game_id: str) -> list[str]:
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key(season, nfl_game_id))
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        return sorted(df["bookmaker"].unique().tolist())
    except Exception:
        return []


def fetch_market(event_id: str, snapshot: str) -> tuple[list[dict], int]:
    for attempt in range(3):
        try:
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
                 dry_run: bool, idx: int, total: int) -> int:
    nfl_game_id = row["nfl_game_id"]
    event_id    = str(row.get("odds_api_event_id", ""))

    if not event_id or event_id == "nan":
        print(f"  [{idx}/{total}] SKIP {nfl_game_id} — no event_id")
        return -1

    if already_in_s3(s3_client, season, nfl_game_id):
        existing_books = books_in_s3(s3_client, season, nfl_game_id)
        has_betonline  = "betonlineag" in existing_books
        flag           = "" if has_betonline else "  *** NO BETONLINE ***"
        print(f"  [{idx}/{total}] SKIP {nfl_game_id} — books={len(existing_books)} {existing_books}{flag}")
        return -1

    if dry_run:
        print(f"  [{idx}/{total}] DRY  {nfl_game_id}")
        return -1

    snapshot = snapshot_utc(row["gameday"], str(row.get("gametime", "13:00")))
    rows, remaining = fetch_market(event_id, snapshot)

    if not rows:
        print(f"  [{idx}/{total}] EMPTY {nfl_game_id}  remaining={remaining:,}")
        return remaining

    df = pd.DataFrame(rows)
    df["nfl_game_id"] = nfl_game_id
    df["season"]      = season
    df["snapshot"]    = snapshot

    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key(season, nfl_game_id), Body=buf.getvalue())

    books_found = df["bookmaker"].nunique()
    book_list   = sorted(df["bookmaker"].unique())
    print(f"  [{idx}/{total}] OK   {nfl_game_id}  books={books_found} {book_list}  remaining={remaining:,}")
    return remaining


def process_season(s3_client, season: int, dry_run: bool):
    path = REPO_ROOT / "data" / "nfl" / f"event_id_map_{season}.csv"
    if not path.exists():
        sys.exit(f"Missing event_id_map for {season}: {path}")

    games = pd.read_csv(path)
    games = games[games["odds_api_event_id"].notna()].copy()
    games = games.sort_values(["gameday", "gametime"]).reset_index(drop=True)
    total = len(games)

    print(f"\n{'='*65}")
    print(f"  SEASON {season}  |  {total} games")
    print(f"{'='*65}\n")

    for i, (_, row) in enumerate(games.iterrows(), 1):
        process_game(s3_client, season, row, dry_run, i, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--season", type=int, choices=SEASONS)
    args = parser.parse_args()

    if not ODDS_API_KEY and not args.dry_run:
        sys.exit("ODDS_API_KEY not set")

    seasons = [args.season] if args.season else SEASONS

    print("=" * 65)
    print("  NFL PASSING YARDS BACKFILL")
    print("=" * 65)
    print(f"  Market  : {MARKET}")
    print(f"  Regions : {REGIONS}")
    print(f"  Seasons : {seasons}")
    print(f"  S3      : s3://{S3_BUCKET}/{S3_PREFIX}/{{season}}/{{game_id}}.parquet")
    if args.dry_run:
        print("\n  *** DRY RUN ***")
    print()

    s3_client = boto3.client("s3")

    try:
        for season in seasons:
            process_season(s3_client, season, args.dry_run)
    except CreditExhausted as e:
        print(f"\nSTOPPED: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print("\nDone.")


if __name__ == "__main__":
    main()
