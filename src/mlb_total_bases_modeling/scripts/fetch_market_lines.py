"""
Backfill batter_total_bases (+ alternate) Odds API lines for 2024-2026.
Saves all rows to one parquet locally and to S3 at the end.
Checkpoints every 200 games to ~/Downloads/tmp/ so progress isn't lost on crash.

Output schema per row:
  game_date, event_id, home_team, away_team, player_name,
  bookmaker, line, over_price, under_price, market_key, snapshot, season

Usage:
  python src/mlb_total_bases_modeling/scripts/fetch_market_lines.py
  python src/mlb_total_bases_modeling/scripts/fetch_market_lines.py --seasons 2026
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
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
MARKETS       = "batter_total_bases,batter_total_bases_alternate"
REGIONS       = "us,us2"
SLEEP_S       = 0.05   # minimal sleep — Odds API allows ~10 req/s
CREDIT_STOP   = 50_000
CHECKPOINT_N  = 200   # save to local parquet every N games

S3_BUCKET  = "the-odds-api-mt"
S3_KEY     = "mlb/total_bases_model/market_raw/mlb_total_bases_market_raw.parquet"
LOCAL_OUT  = Path.home() / "Downloads/tmp/mlb_total_bases_market_raw.parquet"
CHECKPOINT = Path.home() / "Downloads/tmp/mlb_total_bases_market_raw_checkpoint.parquet"

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SEASON_DATES = {
    2024: (date(2024, 3, 20), date(2024, 10, 1)),
    2025: (date(2025, 3, 18), date(2025, 10, 1)),
    2026: (date(2026, 3, 25), date(2026, 7, 3)),
}


def parse_remaining(headers: dict) -> int:
    try:
        return int(headers.get("x-requests-remaining", 999_999))
    except (ValueError, TypeError):
        return 999_999


def snapshot_utc(game_date: str) -> str:
    from datetime import datetime
    dt_et  = datetime.strptime(f"{game_date} 14:00", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    return dt_et.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_events_for_date(d: date) -> list[dict]:
    dt_str = f"{d.isoformat()}T18:00:00Z"
    r = requests.get(
        f"{ODDS_API_BASE}/historical/sports/{SPORT}/events",
        params={"apiKey": ODDS_API_KEY, "date": dt_str},
        timeout=30,
    )
    if r.status_code != 200:
        return []
    return r.json().get("data", [])


def fetch_event_odds(event_id: str, snapshot: str) -> tuple[list[dict], int]:
    r = requests.get(
        f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
        params={"apiKey": ODDS_API_KEY, "markets": MARKETS, "regions": REGIONS, "date": snapshot},
        timeout=30,
    )
    remaining = parse_remaining(r.headers)
    if r.status_code != 200 or not r.json().get("data"):
        return [], remaining

    rows = []
    for bm in r.json()["data"].get("bookmakers", []):
        book = bm["key"]
        for mkt in bm.get("markets", []):
            mkt_key = mkt["key"]
            over_outcomes  = [o for o in mkt.get("outcomes", []) if o["name"] == "Over"]
            under_outcomes = [o for o in mkt.get("outcomes", []) if o["name"] == "Under"]
            for o in over_outcomes:
                pt     = o.get("point")
                player = o.get("description", "")
                under  = next(
                    (u for u in under_outcomes
                     if u.get("description") == player and u.get("point") == pt), None
                )
                rows.append({
                    "bookmaker":   book,
                    "market_key":  mkt_key,
                    "player_name": player,
                    "line":        pt,
                    "over_price":  o.get("price"),
                    "under_price": under["price"] if under else None,
                })
    return rows, remaining


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(SEASON_DATES.keys()))
    parser.add_argument("--no-s3", action="store_true")
    args = parser.parse_args()

    # Load checkpoint if it exists (resume from where we left off)
    seen_events: set[str] = set()
    existing_rows: list[pd.DataFrame] = []
    checkpoint_max_date: dict[int, date] = {}  # season → latest date in checkpoint
    if CHECKPOINT.exists():
        ck = pd.read_parquet(CHECKPOINT)
        seen_events = set(ck["event_id"].unique())
        existing_rows.append(ck)
        for season, grp in ck.groupby("season"):
            checkpoint_max_date[int(season)] = date.fromisoformat(grp["game_date"].max())
        print(f"Resuming from checkpoint: {len(seen_events):,} events already done")
        for s, d in sorted(checkpoint_max_date.items()):
            print(f"  Season {s}: last date in checkpoint = {d}")

    all_rows: list[pd.DataFrame] = list(existing_rows)
    games_since_ck = 0

    for season in sorted(args.seasons):
        start, end = SEASON_DATES[season]
        # If checkpoint exists for this season, skip to day after last known date
        if season in checkpoint_max_date:
            start = checkpoint_max_date[season] + timedelta(days=1)
            print(f"\n=== Season {season} === (skipping to {start})", flush=True)
        else:
            print(f"\n=== Season {season} ===", flush=True)
        d = start
        while d <= end:
            events = get_events_for_date(d)
            time.sleep(SLEEP_S)
            for ev in events:
                event_id  = ev["id"]
                if event_id in seen_events:
                    continue
                home_team = ev.get("home_team", "")
                away_team = ev.get("away_team", "")
                game_date = ev.get("commence_time", "")[:10]
                snapshot  = snapshot_utc(game_date)
                rows, remaining = fetch_event_odds(event_id, snapshot)
                time.sleep(SLEEP_S)
                if remaining < CREDIT_STOP:
                    print(f"CREDIT STOP at {remaining}", flush=True)
                    sys.exit(1)
                if not rows:
                    seen_events.add(event_id)
                    continue
                df = pd.DataFrame(rows)
                df["event_id"]  = event_id
                df["game_date"] = game_date
                df["home_team"] = home_team
                df["away_team"] = away_team
                df["snapshot"]  = snapshot
                df["season"]    = season
                all_rows.append(df)
                seen_events.add(event_id)
                games_since_ck += 1
                print(f"  {game_date}  {away_team[:15]:15} @ {home_team[:15]:15}  "
                      f"{len(rows):4} rows  credits={remaining:,}", flush=True)
                if games_since_ck >= CHECKPOINT_N:
                    combined = pd.concat(all_rows, ignore_index=True)
                    combined.to_parquet(CHECKPOINT, index=False)
                    games_since_ck = 0
            d += timedelta(days=1)

    if not all_rows:
        print("No rows fetched.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(LOCAL_OUT, index=False)
    print(f"\nSaved locally → {LOCAL_OUT} ({len(combined):,} rows)")

    if not args.no_s3:
        buf = BytesIO()
        combined.to_parquet(buf, index=False)
        boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=buf.getvalue())
        print(f"Saved to S3 → s3://{S3_BUCKET}/{S3_KEY}")

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("Checkpoint deleted.")


if __name__ == "__main__":
    main()
