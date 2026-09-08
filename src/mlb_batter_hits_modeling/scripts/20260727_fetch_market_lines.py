"""
Backfill batter_hits Odds API lines for 2024-2026.
Idempotent — skips event IDs already saved in S3.

Uses ThreadPoolExecutor to fetch multiple events in parallel.

Output paths:
  S3:    s3://the-odds-api-mt/mlb/batter_hits_model/market_raw/{season}/{event_id}.parquet
  Local: ~/Downloads/tmp/mlb_batter_hits_market_raw.parquet  (merged all seasons)

Usage:
  python src/mlb_batter_hits_modeling/scripts/20260727_fetch_market_lines.py
  python src/mlb_batter_hits_modeling/scripts/20260727_fetch_market_lines.py --seasons 2026
  python src/mlb_batter_hits_modeling/scripts/20260727_fetch_market_lines.py --dry-run
  python src/mlb_batter_hits_modeling/scripts/20260727_fetch_market_lines.py --rebuild-local
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "baseball_mlb"
PROP_MARKETS  = {"batter_hits", "batter_hits_alternate"}
MARKETS       = "batter_hits,batter_hits_alternate"
REGIONS       = "us,us2"
CREDIT_STOP   = 50_000
MAX_WORKERS   = 8   # parallel threads for event fetching

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "mlb/batter_hits_model/market_raw"
LOCAL_OUT = Path.home() / "Downloads/tmp/mlb_batter_hits_market_raw.parquet"

ET  = ZoneInfo("America/New_York")

SEASON_DATES: dict[int, tuple[date, date]] = {
    2024: (date(2024, 3, 20), date(2024, 10, 1)),
    2025: (date(2025, 3, 18), date(2025, 10, 1)),
    2026: (date(2026, 3, 25), date.today()),
}

_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


class CreditExhausted(Exception):
    pass


def parse_remaining(headers: dict) -> int:
    try:
        return int(headers.get("x-requests-remaining", 999_999))
    except (ValueError, TypeError):
        return 999_999


def snapshot_utc(commence_time_str: str) -> str:
    commence_dt = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
    snapshot_dt = commence_dt - timedelta(hours=1)
    return snapshot_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def s3_key(season: int, event_id: str) -> str:
    return f"{S3_PREFIX}/{season}/{event_id}.parquet"


def list_existing_event_ids(s3c, season: int) -> set[str]:
    """Bulk-list all event IDs already saved in S3 for a season (1 paginated call vs N head_objects)."""
    existing: set[str] = set()
    paginator = s3c.get_paginator("list_objects_v2")
    prefix = f"{S3_PREFIX}/{season}/"
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            # Key format: mlb/batter_hits_model/market_raw/{season}/{event_id}.parquet
            key = obj["Key"]
            event_id = key.split("/")[-1].replace(".parquet", "")
            existing.add(event_id)
    return existing


def get_events_for_date(d: date) -> list[dict]:
    dt_str = f"{d.isoformat()}T18:00:00Z"
    for attempt in range(3):
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/historical/sports/{SPORT}/events",
                params={"apiKey": ODDS_API_KEY, "date": dt_str},
                timeout=15,
            )
            if r.status_code != 200:
                return []
            events = r.json().get("data", [])
            seen: set[str] = set()
            unique = []
            for ev in events:
                if ev["id"] not in seen:
                    seen.add(ev["id"])
                    unique.append(ev)
            return unique
        except Exception:
            if attempt == 2:
                return []
            time.sleep(2 ** (attempt + 1))
    return []


def fetch_and_save_event(ev: dict, season: int, s3c) -> tuple[int, int]:
    """Fetch odds for one event and save to S3. Returns (rows_written, credits_remaining)."""
    event_id  = ev["id"]
    home_team = ev.get("home_team", "")
    away_team = ev.get("away_team", "")
    commence_time_str = ev.get("commence_time", "")

    if commence_time_str:
        commence_dt = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        game_date = commence_dt.astimezone(ET).strftime("%Y-%m-%d")
    else:
        game_date = ""

    snapshot = snapshot_utc(commence_time_str) if commence_time_str else ""

    for attempt in range(3):
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
                params={
                    "apiKey":  ODDS_API_KEY,
                    "markets": MARKETS,
                    "regions": REGIONS,
                    "date":    snapshot,
                },
                timeout=30,
            )
            break
        except Exception:
            if attempt == 2:
                empty = pd.DataFrame(columns=["player_name","bookmaker","market_key","line",
                                               "over_price","under_price","event_id","game_date",
                                               "home_team","away_team","snapshot","season"])
                buf = BytesIO()
                empty.to_parquet(buf, index=False)
                s3c.put_object(Bucket=S3_BUCKET, Key=s3_key(season, event_id), Body=buf.getvalue())
                return 0, -1
            time.sleep(2 ** (attempt + 1))

    remaining = parse_remaining(r.headers)
    data = r.json().get("data") or {}
    bookmakers = data.get("bookmakers", [])

    if not bookmakers:
        empty = pd.DataFrame(columns=["player_name","bookmaker","market_key","line",
                                       "over_price","under_price","event_id","game_date",
                                       "home_team","away_team","snapshot","season"])
        buf = BytesIO()
        empty.to_parquet(buf, index=False)
        s3c.put_object(Bucket=S3_BUCKET, Key=s3_key(season, event_id), Body=buf.getvalue())
        return 0, remaining

    prop_rows: list[dict] = []
    for bm in bookmakers:
        book = bm["key"]
        for mkt in bm.get("markets", []):
            mkt_key  = mkt["key"]
            outcomes = mkt.get("outcomes", [])
            if mkt_key not in PROP_MARKETS:
                continue
            over_outcomes  = [o for o in outcomes if o["name"] == "Over"]
            under_outcomes = [o for o in outcomes if o["name"] == "Under"]
            for o in over_outcomes:
                pt     = o.get("point")
                player = o.get("description", "")
                under_match = next(
                    (u for u in under_outcomes
                     if u.get("description") == player and u.get("point") == pt),
                    None,
                )
                prop_rows.append({
                    "bookmaker":   book,
                    "market_key":  mkt_key,
                    "player_name": player,
                    "line":        pt,
                    "over_price":  o.get("price"),
                    "under_price": under_match["price"] if under_match else None,
                })

    df = pd.DataFrame(prop_rows)
    df["event_id"]  = event_id
    df["game_date"] = game_date
    df["home_team"] = home_team
    df["away_team"] = away_team
    df["snapshot"]  = snapshot
    df["season"]    = season

    buf = BytesIO()
    df.to_parquet(buf, index=False)
    s3c.put_object(Bucket=S3_BUCKET, Key=s3_key(season, event_id), Body=buf.getvalue())

    tprint(f"  {game_date}  {away_team[:14]:14} @ {home_team[:14]:14}"
           f"  players={df['player_name'].nunique():3}  rows={len(prop_rows):4}"
           f"  credits={remaining:,}")

    return len(prop_rows), remaining


def process_season(season: int, dry_run: bool = False) -> int:
    """Collect all events for the season, skip already-fetched, parallel-fetch the rest."""
    start, end = SEASON_DATES[season]
    s3c = boto3.client("s3")

    print(f"\n  Collecting event list for {season} ...")
    all_events: list[dict] = []
    d = start
    while d <= end:
        events = get_events_for_date(d)
        all_events.extend(events)
        d += timedelta(days=1)

    # Deduplicate across days (API can return the same event_id on adjacent days)
    seen: set[str] = set()
    unique_events = []
    for ev in all_events:
        if ev["id"] not in seen:
            seen.add(ev["id"])
            unique_events.append(ev)

    print(f"  Total unique events for {season}: {len(unique_events):,}")

    # Bulk-list existing S3 keys (1 API call instead of N head_objects)
    existing_ids = list_existing_event_ids(s3c, season)
    pending = [ev for ev in unique_events if ev["id"] not in existing_ids]
    skipped = len(unique_events) - len(pending)
    print(f"  Already in S3: {skipped:,}   Pending: {len(pending):,}")

    if dry_run:
        for ev in pending[:5]:
            print(f"    DRY  {ev.get('commence_time','')[:10]}  {ev.get('away_team','')} @ {ev.get('home_team','')}")
        if len(pending) > 5:
            print(f"    ... and {len(pending)-5} more")
        return 0

    if not pending:
        print(f"  Season {season}: all events already fetched.")
        return 0

    total_rows = 0
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_and_save_event, ev, season, s3c): ev for ev in pending}
        for future in as_completed(futures):
            rows, remaining = future.result()
            total_rows += rows
            done += 1
            if remaining != -1 and remaining < CREDIT_STOP:
                raise CreditExhausted(f"Credits below {CREDIT_STOP}: {remaining}")
            if done % 100 == 0:
                tprint(f"  [{season}] Progress: {done}/{len(pending)} events fetched")

    print(f"  Season {season}: {done} events fetched, {total_rows:,} prop rows total")
    return total_rows


def rebuild_local_from_s3(seasons: list[int]) -> None:
    s3c    = boto3.client("s3")
    frames = []
    paginator = s3c.get_paginator("list_objects_v2")
    for season in seasons:
        prefix = f"{S3_PREFIX}/{season}/"
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Size"] == 0:
                    continue
                resp = s3c.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
                df = pd.read_parquet(BytesIO(resp["Body"].read()))
                if len(df) > 0:
                    frames.append(df)
        sys.stdout.write(f"\r  Loaded {len(frames):,} files…")
        sys.stdout.flush()
    print()
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(LOCAL_OUT, index=False)
        print(f"Rebuilt → {LOCAL_OUT}  ({len(combined):,} rows, {combined['player_name'].nunique():,} players)")
    else:
        print("No data found in S3.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(SEASON_DATES.keys()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--rebuild-local", action="store_true")
    args = parser.parse_args()

    if args.rebuild_local:
        rebuild_local_from_s3(sorted(args.seasons))
        return

    if args.start_date:
        sd = date.fromisoformat(args.start_date)
        for s in args.seasons:
            orig_start, orig_end = SEASON_DATES[s]
            if sd > orig_start:
                SEASON_DATES[s] = (sd, orig_end)

    for season in sorted(args.seasons):
        print(f"\n=== Season {season} ===")
        process_season(season, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nRebuilding local merged parquet from S3...")
        rebuild_local_from_s3(sorted(args.seasons))


if __name__ == "__main__":
    main()
