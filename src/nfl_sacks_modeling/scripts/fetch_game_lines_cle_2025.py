"""
Fetch game totals + spreads for all 17 CLE REG 2025 games.
Snapshot: -30 min before kickoff (same as sacks props).
Markets: totals, spreads

Output:
  ~/Downloads/tmp/cle_game_lines_2025.parquet

Run:
  python nfl_sacks_modeling/scripts/fetch_game_lines_cle_2025.py
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "americanfootball_nfl"
REGIONS       = "us"
MARKETS       = "totals,spreads"
SLEEP_S       = 0.15

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

OUT = Path.home() / "Downloads" / "tmp" / "cle_game_lines_2025.parquet"


def snapshot_utc(gameday: str, gametime_et: str, offset_min: int = -30) -> str:
    dt_et = datetime.strptime(f"{gameday} {gametime_et}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    dt_utc = dt_et.astimezone(UTC) + timedelta(minutes=offset_min)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_lines(event_id: str, snapshot: str) -> tuple[list[dict], int]:
    resp = requests.get(
        f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
        params={
            "apiKey":     ODDS_API_KEY,
            "markets":    MARKETS,
            "regions":    REGIONS,
            "oddsFormat": "american",
            "dateFormat": "iso",
            "date":       snapshot,
        },
        timeout=60,
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
                        "market":       mkt["key"],
                        "bookmaker":    book["key"],
                        "outcome_name": outcome.get("name", ""),
                        "point":        outcome.get("point"),
                        "price":        outcome.get("price"),
                        "snapshot":     snapshot,
                    })
    return rows, remaining


def main():
    if not ODDS_API_KEY:
        sys.exit("ODDS_API_KEY not set")

    map_path = REPO_ROOT / "data" / "nfl" / "event_id_map_2025.csv"
    games    = pd.read_csv(map_path)
    cle_reg  = games[
        (games["game_type"] == "REG") &
        (games["nfl_game_id"].str.contains("CLE"))
    ].sort_values(["gameday", "gametime"]).reset_index(drop=True)

    assert len(cle_reg) == 17, f"Expected 17 CLE REG games, got {len(cle_reg)}"

    all_rows = []

    for i, row in cle_reg.iterrows():
        game_id  = row["nfl_game_id"]
        event_id = str(row["odds_api_event_id"])
        gameday  = row["gameday"]
        gametime = str(row.get("gametime", "13:00"))
        week     = int(row["week"])
        is_london = int(gametime.split(":")[0]) < 10

        snap = snapshot_utc(gameday, gametime, offset_min=-30)
        rows, remaining = fetch_lines(event_id, snap)

        if is_london and not rows:
            snap = snapshot_utc(gameday, gametime, offset_min=-120)
            rows, remaining = fetch_lines(event_id, snap)

        for r in rows:
            r["nfl_game_id"] = game_id
            r["week"]        = week
            r["gameday"]     = gameday
            r["home_team"]   = row["home_team"]
            r["away_team"]   = row["away_team"]

        n = len(rows)
        print(f"  [{i+1:>2}/17] wk{week:>2}  {game_id:<27}  rows={n}  remaining={remaining}")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT, index=False)
    print(f"\nSaved → {OUT}")
    print(f"Markets: {df['market'].unique().tolist()}")
    print(f"Books:   {sorted(df['bookmaker'].unique().tolist())}")


if __name__ == "__main__":
    main()
