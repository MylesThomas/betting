"""
Backfill pitcher_walks Odds API lines for 2024-2026.
Also fetches h2h and spreads per game and stores consensus team odds
as game-level columns on every prop row — used as features in the spine.

Idempotent — skips event IDs already saved in S3.

Output schema per row:
  game_date, event_id, home_team, away_team, player_name,
  bookmaker, market_key, line, over_price, under_price,
  snapshot, season,
  consensus_home_moneyline, consensus_away_moneyline,
  home_run_line_point, away_run_line_point,
  home_run_line_odds, away_run_line_odds

  Notes:
    - All prices stored as American odds (requested directly — no decimal conversion needed).
    - consensus_home/away_moneyline: median American odds across all books.
    - home/away_run_line_point: almost always -1.5/+1.5.

Output paths:
  S3:    s3://the-odds-api-mt/mlb/pitcher_walks_model/market_raw/
         {season}/{event_id}.parquet
  Local: ~/Downloads/tmp/mlb_pitcher_walks_market_raw.parquet  (merged all seasons)

Usage:
  # Full backfill (all seasons):
  python src/mlb_pitcher_walks_modeling/scripts/fetch_market_lines.py

  # Single season:
  python src/mlb_pitcher_walks_modeling/scripts/fetch_market_lines.py --seasons 2026

  # Dry-run (print events that would be fetched, no API calls for odds):
  python src/mlb_pitcher_walks_modeling/scripts/fetch_market_lines.py --dry-run

  # Skip ahead to a specific date (useful for daily top-up after initial backfill):
  python src/mlb_pitcher_walks_modeling/scripts/fetch_market_lines.py --start-date 2026-07-01

  # Rebuild merged local parquet from S3 (no new API calls):
  python src/mlb_pitcher_walks_modeling/scripts/fetch_market_lines.py --rebuild-local
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
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
PROP_MARKETS  = {"pitcher_walks"}
GAME_MARKETS  = {"h2h", "spreads"}
MARKETS       = "pitcher_walks,h2h,spreads"
REGIONS       = "us,us2"
ODDS_FORMAT   = "american"   # request American directly — no decimal conversion needed
SLEEP_S       = 0.15
CREDIT_STOP   = 50_000

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "mlb/pitcher_walks_model/market_raw"
LOCAL_OUT = Path.home() / "Downloads/tmp/mlb_pitcher_walks_market_raw.parquet"

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

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


def snapshot_utc(game_date: str) -> str:
    """Pre-game snapshot: 2pm ET on game_date."""
    dt_et  = datetime.strptime(f"{game_date} 14:00", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    dt_utc = dt_et.astimezone(UTC)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


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
    dt_str = f"{d.isoformat()}T18:00:00Z"
    for attempt in range(3):
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/historical/sports/{SPORT}/events",
                params={"apiKey": ODDS_API_KEY, "date": dt_str},
                timeout=30,
            )
            if r.status_code != 200:
                return []
            return r.json().get("data", [])
        except Exception:
            if attempt == 2:
                return []
            time.sleep(2 ** (attempt + 1))
    return []


def fetch_event_odds(event_id: str, snapshot: str, home_team: str, away_team: str) -> tuple[list[dict], dict, int]:
    """Fetch all markets (props + game lines) for one event.

    Returns:
        prop_rows: list of dicts, one per (bookmaker, player, line) for pitcher_walks
        game_odds: dict with consensus h2h/spread values for this game
        remaining: API credits remaining
    """
    for attempt in range(3):
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
                params={
                    "apiKey":      ODDS_API_KEY,
                    "markets":     MARKETS,
                    "regions":     REGIONS,
                    "oddsFormat":  ODDS_FORMAT,
                    "date":        snapshot,
                },
                timeout=30,
            )
            break
        except Exception:
            if attempt == 2:
                return [], {}, -1
            time.sleep(2 ** (attempt + 1))

    remaining = parse_remaining(r.headers)
    data      = r.json().get("data") or {}
    bookmakers = data.get("bookmakers", [])

    if not bookmakers:
        return [], {}, remaining

    prop_rows: list[dict] = []

    home_h2h_prices:     list[float] = []
    away_h2h_prices:     list[float] = []
    home_runline_points: list[float] = []
    away_runline_points: list[float] = []
    home_runline_prices: list[float] = []
    away_runline_prices: list[float] = []

    for bm in bookmakers:
        book = bm["key"]
        for mkt in bm.get("markets", []):
            mkt_key  = mkt["key"]
            outcomes = mkt.get("outcomes", [])

            if mkt_key in PROP_MARKETS:
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

            elif mkt_key == "h2h":
                for o in outcomes:
                    if o.get("name") == home_team:
                        home_h2h_prices.append(o["price"])
                    elif o.get("name") == away_team:
                        away_h2h_prices.append(o["price"])

            elif mkt_key == "spreads":
                for o in outcomes:
                    if o.get("name") == home_team:
                        if o.get("point") is not None:
                            home_runline_points.append(o["point"])
                        home_runline_prices.append(o["price"])
                    elif o.get("name") == away_team:
                        if o.get("point") is not None:
                            away_runline_points.append(o["point"])
                        away_runline_prices.append(o["price"])

    def _median_or_none(lst: list[float]) -> float | None:
        return round(statistics.median(lst), 1) if lst else None

    game_odds = {
        "consensus_home_moneyline": _median_or_none(home_h2h_prices),
        "consensus_away_moneyline": _median_or_none(away_h2h_prices),
        "home_run_line_point":      _median_or_none(home_runline_points),
        "away_run_line_point":      _median_or_none(away_runline_points),
        "home_run_line_odds":       _median_or_none(home_runline_prices),
        "away_run_line_odds":       _median_or_none(away_runline_prices),
    }

    return prop_rows, game_odds, remaining


# ---------------------------------------------------------------------------
# Per-season processing
# ---------------------------------------------------------------------------

def process_season(season: int, dry_run: bool = False) -> pd.DataFrame:
    start, end = SEASON_DATES[season]
    s3c = boto3.client("s3")
    all_rows: list[pd.DataFrame] = []

    d = start
    while d <= end:
        events = get_events_for_date(d)
        time.sleep(SLEEP_S)

        for ev in events:
            event_id  = ev["id"]
            home_team = ev.get("home_team", "")
            away_team = ev.get("away_team", "")
            game_date = ev.get("commence_time", "")[:10]

            if already_in_s3(s3c, season, event_id):
                continue

            if dry_run:
                print(f"  DRY  {game_date}  {away_team} @ {home_team}")
                continue

            snapshot = snapshot_utc(game_date)
            prop_rows, game_odds, remaining = fetch_event_odds(event_id, snapshot, home_team, away_team)
            time.sleep(SLEEP_S)

            if remaining != -1 and remaining < CREDIT_STOP:
                raise CreditExhausted(f"Credits below {CREDIT_STOP}: {remaining}")

            if not prop_rows:
                continue

            df = pd.DataFrame(prop_rows)
            df["event_id"]  = event_id
            df["game_date"] = game_date
            df["home_team"] = home_team
            df["away_team"] = away_team
            df["snapshot"]  = snapshot
            df["season"]    = season

            for col, val in game_odds.items():
                df[col] = val

            buf = BytesIO()
            df.to_parquet(buf, index=False)
            s3c.put_object(Bucket=S3_BUCKET, Key=s3_key(season, event_id), Body=buf.getvalue())
            all_rows.append(df)
            print(
                f"  {game_date}  {away_team[:15]:15} @ {home_team[:15]:15}"
                f"  props={len(prop_rows):3}  game_odds={'✓' if game_odds.get('consensus_home_moneyline') else '✗'}"
                f"  credits={remaining:,}"
            )

        d += timedelta(days=1)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Rebuild local parquet from S3 (no API calls)
# ---------------------------------------------------------------------------

def rebuild_local_from_s3(seasons: list[int]) -> None:
    s3c    = boto3.client("s3")
    frames = []
    for season in seasons:
        prefix = f"{S3_PREFIX}/{season}/"
        paginator = s3c.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                resp = s3c.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
                frames.append(pd.read_parquet(BytesIO(resp["Body"].read())))
                sys.stdout.write(f"\r  Downloaded {len(frames):,} files…")
                sys.stdout.flush()
    print()
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(LOCAL_OUT, index=False)
        print(f"Rebuilt → {LOCAL_OUT}  ({len(combined):,} rows, {combined['player_name'].nunique():,} players)")
    else:
        print("No data found in S3.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MLB pitcher_walks market lines from Odds API.")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(SEASON_DATES.keys()),
                        help="Seasons to fetch (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print events that would be fetched without making odds API calls")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Override season start date (YYYY-MM-DD) — useful for daily top-up")
    parser.add_argument("--rebuild-local", action="store_true",
                        help="Rebuild merged local parquet from S3 (no new API calls)")
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

    all_dfs: list[pd.DataFrame] = []
    for season in sorted(args.seasons):
        print(f"\n=== Season {season} ({SEASON_DATES[season][0]} → {SEASON_DATES[season][1]}) ===")
        df = process_season(season, dry_run=args.dry_run)
        if not df.empty:
            all_dfs.append(df)
            print(f"  Season {season}: {len(df):,} rows, {df['player_name'].nunique():,} players")

    if all_dfs and not args.dry_run:
        combined = pd.concat(all_dfs, ignore_index=True)
        LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(LOCAL_OUT, index=False)
        print(f"\nMerged saved → {LOCAL_OUT}")
        print(f"Total rows: {len(combined):,}  |  Players: {combined['player_name'].nunique():,}")


if __name__ == "__main__":
    main()
