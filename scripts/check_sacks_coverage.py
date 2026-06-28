"""
Check historical sacks props coverage for a given NFL season.

Samples ~10 games spread across the season and reports how many had sacks
props data on The Odds API. Use this before committing to a full backfill.

Requires: data/nfl/event_id_map_{season}.csv
If the map doesn't exist for the target season, run build_nfl_event_id_map.py first.

Cost: ~10 API calls × 10 credits = ~100 credits per season check.

Run:
  python scripts/check_sacks_coverage.py --season 2023
  python scripts/check_sacks_coverage.py --season 2022
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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
SLEEP_S       = 0.20

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

CHECK_WEEKS = list(range(1, 19))   # all regular season weeks


def snapshot_utc(gameday: str, gametime_et: str, offset_min: int = -30) -> str:
    dt_et  = datetime.strptime(f"{gameday} {gametime_et}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    dt_utc = dt_et.astimezone(UTC) + timedelta(minutes=offset_min)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_sacks(event_id: str, snapshot: str) -> tuple[list[dict], int]:
    for attempt in range(2):
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
                timeout=30,
            )
            time.sleep(SLEEP_S)
            remaining = int(resp.headers.get("x-requests-remaining", 999_999))

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
                                "bookmaker":    book["key"],
                                "outcome_desc": outcome.get("description", ""),
                                "point":        outcome.get("point"),
                            })
            return rows, remaining

        except requests.exceptions.Timeout:
            if attempt == 0:
                time.sleep(2)
                continue
            return [], 999_999   # treat timeout as no data, continue

    return [], 999_999


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args   = parser.parse_args()
    season = args.season

    if not ODDS_API_KEY:
        sys.exit("ODDS_API_KEY not set — add it to .env")

    map_path = REPO_ROOT / "data" / "nfl" / f"event_id_map_{season}.csv"
    if not map_path.exists():
        sys.exit(
            f"No event_id_map found at {map_path}.\n"
            f"Run: python scripts/build_nfl_event_id_map.py --season {season}"
        )

    games = pd.read_csv(map_path)
    reg   = games[games["game_type"] == "REG"].dropna(subset=["odds_api_event_id"])
    reg   = reg.sort_values(["week", "gameday"]).reset_index(drop=True)

    sample = reg[reg["week"].isin(CHECK_WEEKS)].copy()

    print(f"\nSacks props coverage check — {season} season")
    print(f"{'='*65}")
    print(f"  REG games in map : {len(reg)}")
    print(f"  Checking weeks   : {CHECK_WEEKS}  ({len(sample)} games)")
    print(f"  Credit estimate  : ~{len(sample) * 10} credits")
    print(f"{'='*65}\n")

    results = []
    for _, row in sample.iterrows():
        game_id  = row["nfl_game_id"]
        event_id = str(row["odds_api_event_id"])
        gameday  = row["gameday"]
        gametime = str(row.get("gametime", "13:00"))
        week     = int(row["week"])

        snap     = snapshot_utc(gameday, gametime, offset_min=-30)
        rows, remaining = fetch_sacks(event_id, snap)

        # Try 2h early for London games
        is_london = int(gametime.split(":")[0]) < 10
        if is_london and not rows:
            snap  = snapshot_utc(gameday, gametime, offset_min=-120)
            rows, remaining = fetch_sacks(event_id, snap)

        n_books   = len(set(r["bookmaker"]   for r in rows))
        n_players = len(set(r["outcome_desc"] for r in rows))
        has_data  = len(rows) > 0
        tag       = "OK" if has_data else "NO_DATA"

        print(f"  wk{week:>2}  {tag:<8}  {game_id:<30}  "
              f"books={n_books}  players={n_players}  remaining={remaining}")

        results.append({
            "week": week, "game_id": game_id, "has_data": has_data,
            "n_books": n_books, "n_players": n_players,
        })

    df = pd.DataFrame(results)
    n_with   = df["has_data"].sum()
    n_total  = len(df)
    pct      = n_with / n_total * 100
    avg_bks  = df.loc[df["has_data"], "n_books"].mean()
    avg_ply  = df.loc[df["has_data"], "n_players"].mean()

    print(f"\n{'='*65}")
    print(f"  Season        : {season}")
    print(f"  Coverage      : {n_with}/{n_total} sampled games had sacks props ({pct:.0f}%)")
    if n_with > 0:
        print(f"  Avg books     : {avg_bks:.1f}  (when props exist)")
        print(f"  Avg players   : {avg_ply:.1f}  (when props exist)")

    if pct >= 80:
        verdict = "FULL BACKFILL recommended"
    elif pct >= 50:
        verdict = "PARTIAL coverage — backfill may be usable, expect gaps"
    elif pct >= 20:
        verdict = "THIN coverage — probably too sparse to be useful"
    else:
        verdict = "NO coverage — sacks props market didn't exist for this season"

    print(f"  Verdict       : {verdict}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
