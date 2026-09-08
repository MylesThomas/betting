"""
Step 1 — Fetch batter_strikeouts Odds API lines + coverage check.

Fetches historical batter_strikeouts odds for 2024–2026, stores per-event
parquets to S3 (idempotent), then prints a coverage report:

  - Games found vs games with coverage (by season)
  - Line distribution (0.5, 1.5, other)
  - Books per event
  - Players per game

Usage:
    # Full backfill + coverage report:
    python src/mlb_batter_strikeouts_modeling/scripts/20260711_fetch_coverage_check.py

    # Dry-run (list events, no API calls):
    python src/mlb_batter_strikeouts_modeling/scripts/20260711_fetch_coverage_check.py --dry-run

    # Rebuild merged local parquet from S3 (no API calls), then print coverage:
    python src/mlb_batter_strikeouts_modeling/scripts/20260711_fetch_coverage_check.py --rebuild-local

    # Single season:
    python src/mlb_batter_strikeouts_modeling/scripts/20260711_fetch_coverage_check.py --seasons 2026
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections import Counter
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
MARKETS       = "batter_strikeouts"
REGIONS       = "us,us2"
SLEEP_S       = 0.15
CREDIT_STOP   = 50_000

S3_BUCKET  = "the-odds-api-mt"
S3_PREFIX  = "mlb/batter_strikeouts_model/market_raw"
LOCAL_OUT  = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_market_raw.parquet"

ET  = ZoneInfo("America/New_York")

SEASON_DATES: dict[int, tuple[date, date]] = {
    2024: (date(2024, 3, 20), date(2024, 10, 1)),
    2025: (date(2025, 3, 18), date(2025, 10, 1)),
    2026: (date(2026, 3, 25), date.today()),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CreditExhausted(Exception):
    pass


def parse_remaining(headers: dict) -> int:
    try:
        return int(headers.get("x-requests-remaining", 999_999))
    except (ValueError, TypeError):
        return 999_999


def snapshot_utc(commence_time_str: str, minutes_before: int = 60) -> str:
    commence_dt = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
    snapshot_dt = commence_dt - timedelta(minutes=minutes_before)
    return snapshot_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def s3_key(season: int, event_id: str) -> str:
    return f"{S3_PREFIX}/{season}/{event_id}.parquet"


def already_in_s3(s3c, season: int, event_id: str) -> bool:
    try:
        s3c.head_object(Bucket=S3_BUCKET, Key=s3_key(season, event_id))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def get_events_for_date(d: date) -> list[dict]:
    dt_str = f"{d.isoformat()}T13:00:00Z"  # 9am ET — before any game starts
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


def fetch_event_odds(event_id: str, snapshot: str) -> tuple[list[dict], int]:
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
                return [], -1
            time.sleep(2 ** (attempt + 1))

    remaining = parse_remaining(r.headers)
    data      = r.json().get("data") or {}
    bookmakers = data.get("bookmakers", [])

    if not bookmakers:
        return [], remaining

    prop_rows: list[dict] = []
    for bm in bookmakers:
        book = bm["key"]
        for mkt in bm.get("markets", []):
            if mkt["key"] != "batter_strikeouts":
                continue
            outcomes      = mkt.get("outcomes", [])
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
                    "market_key":  "batter_strikeouts",
                    "player_name": player,
                    "line":        pt,
                    "over_price":  o.get("price"),
                    "under_price": under_match["price"] if under_match else None,
                })

    return prop_rows, remaining


# ---------------------------------------------------------------------------
# Per-season processing
# ---------------------------------------------------------------------------

def process_season(season: int, dry_run: bool = False) -> pd.DataFrame:
    start, end = SEASON_DATES[season]
    s3c = boto3.client("s3")
    all_rows: list[pd.DataFrame] = []

    d = start
    while d <= end:
        print(f"checking {d}...")
        events = get_events_for_date(d)
        time.sleep(SLEEP_S)

        if not events:
            print(f"  {d}  no events")
            d += timedelta(days=1)
            continue

        n_skip = sum(1 for ev in events if already_in_s3(s3c, season, ev["id"]))
        if n_skip == len(events):
            print(f"  {d}  {len(events)} events — all cached (skipped)")
            d += timedelta(days=1)
            continue

        for ev in events:
            event_id  = ev["id"]
            home_team = ev.get("home_team", "")
            away_team = ev.get("away_team", "")
            commence_time_str = ev.get("commence_time", "")
            if commence_time_str:
                commence_dt = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
                game_date = commence_dt.astimezone(ET).strftime("%Y-%m-%d")
            else:
                game_date = d.isoformat()

            if already_in_s3(s3c, season, event_id):
                print(f"  {game_date}  {away_team[:15]:15} @ {home_team[:15]:15}  cached")
                continue

            if dry_run:
                print(f"  DRY  {game_date}  {away_team} @ {home_team}")
                continue

            snapshot = snapshot_utc(commence_time_str, minutes_before=60)
            prop_rows, remaining = fetch_event_odds(event_id, snapshot)
            time.sleep(SLEEP_S)

            if remaining != -1 and remaining < CREDIT_STOP:
                raise CreditExhausted(f"Credits below {CREDIT_STOP}: {remaining}")

            if not prop_rows and commence_time_str:
                print(f"  {game_date}  {away_team[:15]:15} @ {home_team[:15]:15}  no lines at -60min, retrying at -30min...")
                snapshot = snapshot_utc(commence_time_str, minutes_before=30)
                prop_rows, remaining = fetch_event_odds(event_id, snapshot)
                time.sleep(SLEEP_S)
                if remaining != -1 and remaining < CREDIT_STOP:
                    raise CreditExhausted(f"Credits below {CREDIT_STOP}: {remaining}")

            if not prop_rows:
                # Write empty marker so this event is permanently skipped on resume
                empty = pd.DataFrame(columns=["player_name","bookmaker","market_key","line",
                                               "over_price","under_price","event_id","game_date",
                                               "home_team","away_team","snapshot","season"])
                buf = BytesIO()
                empty.to_parquet(buf, index=False)
                s3c.put_object(Bucket=S3_BUCKET, Key=s3_key(season, event_id), Body=buf.getvalue())
                print(f"  {game_date}  {away_team[:15]:15} @ {home_team[:15]:15}  no lines (marked)")
                d += timedelta(days=1)
                continue

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
            all_rows.append(df)
            print(
                f"  {game_date}  {away_team[:15]:15} @ {home_team[:15]:15}"
                f"  props={len(prop_rows):3}  credits={remaining:,}"
            )

        d += timedelta(days=1)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Rebuild local parquet from S3 (no API calls)
# ---------------------------------------------------------------------------

def rebuild_local_from_s3(seasons: list[int]) -> pd.DataFrame:
    s3c    = boto3.client("s3")
    frames = []
    for season in seasons:
        prefix = f"{S3_PREFIX}/{season}/"
        paginator = s3c.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Size"] == 0:
                    continue
                resp = s3c.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
                df = pd.read_parquet(BytesIO(resp["Body"].read()))
                if len(df) > 0:
                    frames.append(df)
                sys.stdout.write(f"\r  Downloaded {len(frames):,} files…")
                sys.stdout.flush()
    print()
    if not frames:
        print("No data found in S3.")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(LOCAL_OUT, index=False)
    print(f"Rebuilt → {LOCAL_OUT}  ({len(combined):,} rows)")
    return combined


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def coverage_report(df: pd.DataFrame) -> None:
    if df.empty:
        print("\nNo data to report.")
        return

    print("\n" + "="*60)
    print("COVERAGE REPORT — batter_strikeouts")
    print("="*60)

    # Total rows + players
    print(f"\nTotal rows:    {len(df):,}")
    print(f"Unique players:{df['player_name'].nunique():,}")
    print(f"Seasons:       {sorted(df['season'].unique())}")
    print(f"Date range:    {df['game_date'].min()} → {df['game_date'].max()}")

    # Coverage by season: games with props / total games
    print("\n--- Coverage by season ---")
    for season, grp in df.groupby("season"):
        n_events    = grp["event_id"].nunique()
        n_dates     = grp["game_date"].nunique()
        n_players   = grp["player_name"].nunique()
        rows_per_ev = len(grp) / n_events if n_events else 0
        print(f"  {season}: {n_events:,} events · {n_dates} game-days · "
              f"{n_players:,} unique players · {rows_per_ev:.1f} rows/event")

    # Line distribution
    print("\n--- Line distribution ---")
    line_counts = Counter(df["line"].dropna())
    total_lines = sum(line_counts.values())
    for line, cnt in sorted(line_counts.items()):
        pct = cnt / total_lines * 100
        print(f"  {line:>5.1f}  {cnt:>7,}  ({pct:4.1f}%)")

    # Books per event
    books_per_event = df.groupby("event_id")["bookmaker"].nunique()
    print(f"\n--- Books per event ---")
    print(f"  mean={books_per_event.mean():.1f}  "
          f"median={books_per_event.median():.0f}  "
          f"p25={books_per_event.quantile(0.25):.0f}  "
          f"p75={books_per_event.quantile(0.75):.0f}  "
          f"min={books_per_event.min()}  max={books_per_event.max()}")

    # Players per event
    players_per_event = df.groupby("event_id")["player_name"].nunique()
    print(f"\n--- Players per event ---")
    print(f"  mean={players_per_event.mean():.1f}  "
          f"median={players_per_event.median():.0f}  "
          f"min={players_per_event.min()}  max={players_per_event.max()}")

    # Top books by volume
    print("\n--- Top books by row count ---")
    book_counts = df["bookmaker"].value_counts().head(10)
    for book, cnt in book_counts.items():
        print(f"  {book:<25} {cnt:>7,}")

    # Odds sample — over/under distribution for 0.5 and 1.5 lines
    for line_val in [0.5, 1.5]:
        sub = df[df["line"] == line_val].dropna(subset=["over_price", "under_price"])
        if len(sub) == 0:
            continue
        print(f"\n--- Odds sample (line={line_val}, n={len(sub):,}) ---")
        print(f"  over_price:  mean={sub['over_price'].mean():.0f}  "
              f"median={sub['over_price'].median():.0f}  "
              f"p10={sub['over_price'].quantile(0.10):.0f}  "
              f"p90={sub['over_price'].quantile(0.90):.0f}")
        print(f"  under_price: mean={sub['under_price'].mean():.0f}  "
              f"median={sub['under_price'].median():.0f}  "
              f"p10={sub['under_price'].quantile(0.10):.0f}  "
              f"p90={sub['under_price'].quantile(0.90):.0f}")

    print("\n" + "="*60)
    print(f"Local parquet: {LOCAL_OUT}")
    print("="*60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MLB batter_strikeouts odds + coverage check.")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(SEASON_DATES.keys()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Override season start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Override season end date YYYY-MM-DD")
    parser.add_argument("--rebuild-local", action="store_true",
                        help="Rebuild local parquet from S3 then print coverage (no API calls)")
    args = parser.parse_args()

    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set. Add to .env or export.", file=sys.stderr)
        sys.exit(1)

    if args.rebuild_local:
        df = rebuild_local_from_s3(sorted(args.seasons))
        coverage_report(df)
        return

    if args.start_date:
        sd = date.fromisoformat(args.start_date)
        for s in args.seasons:
            orig_start, orig_end = SEASON_DATES[s]
            if sd > orig_start:
                SEASON_DATES[s] = (sd, orig_end)

    if args.end_date:
        ed = date.fromisoformat(args.end_date)
        for s in args.seasons:
            orig_start, orig_end = SEASON_DATES[s]
            if ed < orig_end:
                SEASON_DATES[s] = (orig_start, ed)

    all_dfs: list[pd.DataFrame] = []
    for season in sorted(args.seasons):
        print(f"\n=== Season {season} ({SEASON_DATES[season][0]} → {SEASON_DATES[season][1]}) ===")
        df = process_season(season, dry_run=args.dry_run)
        if not df.empty:
            all_dfs.append(df)
            print(f"  Season {season}: {len(df):,} new rows fetched")

    if args.dry_run:
        return

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(LOCAL_OUT, index=False)
        coverage_report(combined)
    elif LOCAL_OUT.exists():
        print("\nNo new rows fetched (all already in S3). Loading existing local file for report.")
        coverage_report(pd.read_parquet(LOCAL_OUT))
    else:
        print("\nNo data fetched and no local file found. Run without --rebuild-local after a full backfill.")


if __name__ == "__main__":
    main()
