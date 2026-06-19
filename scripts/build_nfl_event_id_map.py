"""
Build a nfl_game_id → Odds API event_id lookup for the full 2025-26 NFL season.

Strategy: one historical odds snapshot per unique game day (h2h market, cheapest).
64 game days → 64 API calls for 285 games.

Output: data/nfl/event_id_map_2025.csv
Columns: nfl_game_id, odds_api_event_id, home_team, away_team, gameday, game_type
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "americanfootball_nfl"
SEASON        = 2025
OUT_PATH      = REPO_ROOT / "data" / "nfl" / "event_id_map_2025.csv"

# nfl-data-py abbr → Odds API full name
TEAM_NAME_MAP = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC":  "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LA":  "Los Angeles Rams",
    "LV":  "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE":  "New England Patriots",
    "NO":  "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF":  "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB":  "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}


def fetch_events_for_date(api_key: str, gameday: str) -> list[dict]:
    """
    Fetch all NFL events from the historical odds endpoint at 12:00 UTC on gameday.
    Uses h2h (cheapest market) — we only need event metadata (id, teams, time).
    Returns list of {id, home_team, away_team, commence_time}.
    """
    snapshot = f"{gameday}T12:00:00Z"
    resp = requests.get(
        f"{ODDS_API_BASE}/historical/sports/{SPORT}/odds",
        params={
            "apiKey":     api_key,
            "markets":    "h2h",
            "oddsFormat": "american",
            "dateFormat": "iso",
            "date":       snapshot,
        },
        timeout=30,
    )
    if resp.status_code in (404, 422):
        return []
    resp.raise_for_status()

    data = resp.json()
    events = data.get("data", []) if isinstance(data, dict) else data
    remaining = resp.headers.get("x-requests-remaining", "?")
    used      = resp.headers.get("x-requests-used", "?")

    print(f"  {gameday} → {len(events)} events  |  credits used={used}  remaining={remaining}")
    return [
        {
            "odds_api_event_id": e["id"],
            "home_team_full":    e.get("home_team", ""),
            "away_team_full":    e.get("away_team", ""),
            "commence_time":     e.get("commence_time", ""),
        }
        for e in events
    ]


def main():
    if not ODDS_API_KEY:
        sys.exit("ODDS_API_KEY not set — add it to .env or export it before running.")

    import nfl_data_py as nfl

    sched = nfl.import_schedules([SEASON])
    sched = sched[sched["gameday"].notna()].copy()
    sched["home_full"] = sched["home_team"].map(TEAM_NAME_MAP)
    sched["away_full"] = sched["away_team"].map(TEAM_NAME_MAP)

    unmapped = sched[sched["home_full"].isna() | sched["away_full"].isna()]["home_team"].unique()
    if len(unmapped):
        print(f"WARNING: unmapped team abbrs: {unmapped}")

    game_days = sorted(sched["gameday"].unique())
    print(f"\n{len(sched)} games across {len(game_days)} game days — fetching event IDs...\n")

    # Build full name → event_id lookup from API responses
    full_name_to_event: dict[tuple, str] = {}

    for gameday in game_days:
        events = fetch_events_for_date(ODDS_API_KEY, gameday)
        for ev in events:
            key = (ev["home_team_full"], ev["away_team_full"])
            full_name_to_event[key] = ev["odds_api_event_id"]
        time.sleep(0.3)

    # Match schedule rows to event IDs
    def lookup(row):
        return full_name_to_event.get((row["home_full"], row["away_full"]), None)

    sched["odds_api_event_id"] = sched.apply(lookup, axis=1)

    matched   = sched["odds_api_event_id"].notna().sum()
    unmatched = sched["odds_api_event_id"].isna().sum()
    print(f"\nMatched: {matched}/{len(sched)}  |  Unmatched: {unmatched}")

    if unmatched:
        print("\nUnmatched games:")
        print(sched[sched["odds_api_event_id"].isna()][
            ["game_id", "gameday", "home_team", "away_team", "game_type"]
        ].to_string(index=False))

    out = sched[[
        "game_id", "odds_api_event_id", "home_team", "away_team",
        "gameday", "game_type", "week",
    ]].rename(columns={"game_id": "nfl_game_id"})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
