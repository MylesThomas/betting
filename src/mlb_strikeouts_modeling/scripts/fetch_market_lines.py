"""
Backfill pitcher_strikeouts (+ alternate) Odds API lines for 2024-2026.
Idempotent — skips game IDs already saved in S3.

Output schema per row:
  game_date, event_id, home_team, away_team, player_name,
  bookmaker, line, over_price, under_price, market_key, snapshot, season

Output paths:
  S3:    s3://the-odds-api-mt/mlb/strikeouts_model/market_raw/
         {season}/{event_id}.parquet
  Local: ~/Downloads/tmp/mlb_strikeouts_market_raw.parquet  (merged)

Usage:
  python src/mlb_strikeouts_modeling/scripts/fetch_market_lines.py
  python src/mlb_strikeouts_modeling/scripts/fetch_market_lines.py --seasons 2026 --dry-run
"""
from __future__ import annotations

import argparse
import os
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
MARKETS       = "pitcher_strikeouts,pitcher_strikeouts_alternate"
REGIONS       = "us,us2"
SLEEP_S       = 0.15
CREDIT_STOP   = 50_000

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "mlb/strikeouts_model/market_raw"
LOCAL_OUT = Path.home() / "Downloads/tmp/mlb_strikeouts_market_raw.parquet"

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SEASON_DATES = {
    2024: (date(2024, 3, 20), date(2024, 10, 1)),
    2025: (date(2025, 3, 18), date(2025, 10, 1)),
    2026: (date(2026, 3, 25), date(2026, 7, 3)),
}


class CreditExhausted(Exception):
    pass


def parse_remaining(headers: dict) -> int:
    try:
        return int(headers.get("x-requests-remaining", 999_999))
    except (ValueError, TypeError):
        return 999_999


def snapshot_utc(game_date: str) -> str:
    """Return UTC snapshot at 2pm ET on the game date — pre-game props window."""
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
    if r.status_code != 200 or not r.json().get("data"):
        return [], remaining

    rows = []
    data = r.json()["data"]
    for bm in data.get("bookmakers", []):
        book = bm["key"]
        for mkt in bm.get("markets", []):
            mkt_key = mkt["key"]
            over_outcomes  = [o for o in mkt.get("outcomes", []) if o["name"] == "Over"]
            under_outcomes = [o for o in mkt.get("outcomes", []) if o["name"] == "Under"]
            for o in over_outcomes:
                pt     = o.get("point")
                player = o.get("description", "")
                under_match = next(
                    (u for u in under_outcomes
                     if u.get("description") == player and u.get("point") == pt), None
                )
                rows.append({
                    "bookmaker":   book,
                    "market_key":  mkt_key,
                    "player_name": player,
                    "line":        pt,
                    "over_price":  o.get("price"),
                    "under_price": under_match["price"] if under_match else None,
                })
    return rows, remaining


def process_season(season: int, dry_run: bool = False) -> pd.DataFrame:
    start, end = SEASON_DATES[season]
    s3c = boto3.client("s3")
    all_rows = []

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
            rows, remaining = fetch_event_odds(event_id, snapshot)
            time.sleep(SLEEP_S)

            if remaining < CREDIT_STOP:
                raise CreditExhausted(f"Credits below {CREDIT_STOP}: {remaining}")

            if not rows:
                continue

            df = pd.DataFrame(rows)
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
            print(f"  {game_date}  {away_team[:15]:15} @ {home_team[:15]:15}  {len(rows):4} rows  credits={remaining:,}")

        d += timedelta(days=1)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(SEASON_DATES.keys()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Override season start date (YYYY-MM-DD) to skip ahead")
    args = parser.parse_args()

    if args.start_date:
        from datetime import date as date_cls
        sd = date_cls.fromisoformat(args.start_date)
        for s in (args.seasons or list(SEASON_DATES.keys())):
            orig_start, orig_end = SEASON_DATES[s]
            if sd > orig_start:
                SEASON_DATES[s] = (sd, orig_end)

    all_dfs = []
    for season in sorted(args.seasons):
        print(f"\n=== Season {season} ===")
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
