"""
Fetch player_reception_yds for one SEA game and print all player names
returned by the Odds API — so we can find what JSN is listed as.
"""
import os
import requests
import json

ODDS_API_KEY  = os.environ["ODDS_API_KEY"]
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "americanfootball_nfl"
MARKET        = "player_reception_yds"
BOOKMAKERS    = "draftkings,fanduel,betmgm,williamhill_us"

# Wk1 2025 — SEA @ DEN, Sep 7 2025
GAME_DATE     = "2025-09-07"
SNAPSHOT_TIME = f"{GAME_DATE}T17:00:00Z"  # noon ET pre-game snapshot

url = f"{ODDS_API_BASE}/historical/sports/{SPORT}/odds"
params = {
    "apiKey":     ODDS_API_KEY,
    "markets":    MARKET,
    "bookmakers": BOOKMAKERS,
    "oddsFormat": "american",
    "dateFormat": "iso",
    "date":       SNAPSHOT_TIME,
}

resp = requests.get(url, params=params, timeout=30)
print(f"Status: {resp.status_code}")
print(f"Requests remaining: {resp.headers.get('x-requests-remaining', 'N/A')}")

data = resp.json()
events = data.get("data", []) if isinstance(data, dict) else data
print(f"Events returned: {len(events)}\n")

# Find SEA game
sea_event = None
for e in events:
    if "Seattle" in e.get("home_team", "") or "Seattle" in e.get("away_team", ""):
        sea_event = e
        break

if not sea_event:
    print("No SEA game found in this snapshot. Try a different time.")
    print("All events returned:")
    for e in events:
        print(f"  {e.get('home_team')} vs {e.get('away_team')}")
else:
    print(f"Found: {sea_event['home_team']} vs {sea_event['away_team']}\n")
    names = set()
    for bookie in sea_event.get("bookmakers", []):
        for mkt in bookie.get("markets", []):
            if mkt["key"] != MARKET:
                continue
            for outcome in mkt.get("outcomes", []):
                names.add(outcome.get("description", "MISSING"))

    print(f"All player names for {MARKET}:")
    for n in sorted(names):
        print(f"  {n}")
