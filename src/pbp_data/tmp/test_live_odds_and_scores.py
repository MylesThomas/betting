"""
Test script to debug The Odds API integration.
Tries to fetch live odds and figure out correct API usage.
"""
import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).resolve()
while not (project_root / '.gitignore').exists():
    project_root = project_root.parent
    if project_root == project_root.parent:
        raise FileNotFoundError("Could not find project root (no .gitignore)")

env_path = project_root / '.env'
load_dotenv(env_path)

ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '')

if not ODDS_API_KEY:
    print("❌ ODDS_API_KEY not found in .env")
    sys.exit(1)

print(f"✅ API Key loaded: {ODDS_API_KEY[:8]}...")
print()

# Test 1: Get all NBA games
print("="*80)
print("TEST 1: Fetching all NBA games from Odds API")
print("="*80)

url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
params = {
    'apiKey': ODDS_API_KEY,
    'regions': 'us',
    'markets': 'player_points',
    'oddsFormat': 'american'
}

try:
    response = requests.get(url, params=params, timeout=10, verify=False)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {len(data)} games with odds")
        print()
        
        for game in data[:3]:  # Show first 3
            print(f"Game ID: {game.get('id')}")
            print(f"  Teams: {game.get('home_team')} vs {game.get('away_team')}")
            print(f"  Commence: {game.get('commence_time')}")
            
            # Check if player props available
            bookmakers = game.get('bookmakers', [])
            if bookmakers:
                print(f"  Bookmakers: {len(bookmakers)}")
                for bm in bookmakers[:2]:
                    print(f"    - {bm.get('key')}: {len(bm.get('markets', []))} markets")
            print()
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Exception: {e}")

print()
print("="*80)
print("TEST 2: Fetching ESPN scoreboard to compare game IDs")
print("="*80)

espn_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
try:
    response = requests.get(espn_url, timeout=10, verify=False)
    if response.status_code == 200:
        data = response.json()
        games = data.get('events', [])
        print(f"✅ Found {len(games)} games on ESPN")
        print()
        
        for game in games[:3]:
            game_id = game.get('id')
            status = game.get('status', {}).get('type', {}).get('name', 'Unknown')
            competitors = game.get('competitions', [{}])[0].get('competitors', [])
            
            teams = []
            for comp in competitors:
                teams.append(comp.get('team', {}).get('displayName', ''))
            
            print(f"ESPN Game ID: {game_id}")
            print(f"  Teams: {' vs '.join(teams)}")
            print(f"  Status: {status}")
            print()
            
    else:
        print(f"❌ Error: {response.status_code}")
        
except Exception as e:
    print(f"❌ Exception: {e}")

print()
print("="*80)
print("TEST 3: Get all games WITHOUT player_points market")
print("="*80)

url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
params = {
    'apiKey': ODDS_API_KEY,
    'regions': 'us',
    'oddsFormat': 'american'
}

try:
    response = requests.get(url, params=params, timeout=10, verify=False)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {len(data)} games with odds")
        print()
        
        # Save first game for inspection
        if data:
            print("First game structure:")
            print(json.dumps(data[0], indent=2)[:1000])
            
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Exception: {e}")

print()
print("="*80)
print("TEST 4: Try player props endpoint")
print("="*80)

# The Odds API might have a separate endpoint for player props
url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
params = {
    'apiKey': ODDS_API_KEY,
}

try:
    response = requests.get(url, params=params, timeout=10, verify=False)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {len(data)} events")
        print()
        
        # Show events with their IDs
        for event in data[:3]:
            print(f"Event ID: {event.get('id')}")
            print(f"  Teams: {event.get('home_team')} vs {event.get('away_team')}")
            print()
            
            # Now try to get player props for this event
            event_id = event.get('id')
            prop_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
            prop_params = {
                'apiKey': ODDS_API_KEY,
                'regions': 'us',
                'markets': 'player_points',
                'oddsFormat': 'american'
            }
            
            print(f"  Trying player_points for event {event_id}...")
            try:
                prop_response = requests.get(prop_url, params=prop_params, timeout=10, verify=False)
                print(f"    Status: {prop_response.status_code}")
                
                if prop_response.status_code == 200:
                    prop_data = prop_response.json()
                    bookmakers = prop_data.get('bookmakers', [])
                    print(f"    ✅ Found {len(bookmakers)} bookmakers with player props!")
                    break  # Found working example
                else:
                    print(f"    ❌ {prop_response.text[:100]}")
                    
            except Exception as e2:
                print(f"    ❌ Exception: {e2}")
            
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Exception: {e}")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print("Need to:")
print("1. Get all events from /v4/sports/basketball_nba/events")
print("2. Match ESPN games to Odds API events by team names")
print("3. Use Odds API event IDs (not ESPN IDs) to fetch player props")
