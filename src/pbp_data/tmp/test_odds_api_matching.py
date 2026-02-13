"""
Test script to figure out how to match ESPN game IDs to The Odds API event IDs.

PROBLEM:
The Odds API uses different event IDs than ESPN.
ESPN game ID: 401810644
Odds API event ID: 1c3809071b6a6e313c87dee15e46bdc1

SOLUTION:
1. Get all NBA events from The Odds API
2. Match to ESPN games by team names
3. Use The Odds API event ID to fetch player props
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

# ============================================================================
# STEP 1: Get ESPN scoreboard
# ============================================================================

print("="*80)
print("STEP 1: Fetch ESPN scoreboard")
print("="*80)

espn_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
try:
    response = requests.get(espn_url, timeout=10, verify=False)
    response.raise_for_status()
    
    data = response.json()
    espn_games = data.get('events', [])
    print(f"✅ Found {len(espn_games)} games on ESPN\n")
    
    # Build lookup dict: {(away_team, home_team): espn_game_id}
    espn_lookup = {}
    for game in espn_games:
        espn_game_id = game.get('id')
        status = game.get('status', {}).get('type', {}).get('name', 'Unknown')
        competitors = game.get('competitions', [{}])[0].get('competitors', [])
        
        home_team = None
        away_team = None
        for comp in competitors:
            team_name = comp.get('team', {}).get('displayName', '')
            if comp.get('homeAway') == 'home':
                home_team = team_name
            else:
                away_team = team_name
        
        if home_team and away_team:
            espn_lookup[(away_team, home_team)] = espn_game_id
            print(f"ESPN Game ID: {espn_game_id}")
            print(f"  Away: {away_team}")
            print(f"  Home: {home_team}")
            print(f"  Status: {status}")
            print()
            
except Exception as e:
    print(f"❌ Exception: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: Get The Odds API events
# ============================================================================

print("="*80)
print("STEP 2: Fetch The Odds API events")
print("="*80)

url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
params = {
    'apiKey': ODDS_API_KEY,
}

try:
    response = requests.get(url, params=params, timeout=10, verify=False)
    response.raise_for_status()
    
    odds_events = response.json()
    print(f"✅ Found {len(odds_events)} events from The Odds API\n")
    
    # Build lookup dict: {(away_team, home_team): odds_event_id}
    odds_lookup = {}
    for event in odds_events:
        odds_event_id = event.get('id')
        home_team = event.get('home_team')
        away_team = event.get('away_team')
        commence_time = event.get('commence_time')
        
        if home_team and away_team:
            odds_lookup[(away_team, home_team)] = odds_event_id
            print(f"Odds API Event ID: {odds_event_id}")
            print(f"  Away: {away_team}")
            print(f"  Home: {home_team}")
            print(f"  Time: {commence_time}")
            print()
            
except Exception as e:
    print(f"❌ Exception: {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: Match ESPN games to Odds API events
# ============================================================================

print("="*80)
print("STEP 3: Match ESPN games to Odds API events")
print("="*80)

matches = []
for (away, home), espn_id in espn_lookup.items():
    # Try exact match
    if (away, home) in odds_lookup:
        odds_id = odds_lookup[(away, home)]
        matches.append({
            'espn_game_id': espn_id,
            'odds_event_id': odds_id,
            'away_team': away,
            'home_team': home,
            'match_type': 'exact'
        })
        print(f"✅ EXACT MATCH:")
        print(f"   ESPN ID: {espn_id}")
        print(f"   Odds ID: {odds_id}")
        print(f"   Teams: {away} @ {home}")
        print()
    else:
        print(f"⚠️  NO MATCH for {away} @ {home}")
        print(f"   ESPN ID: {espn_id}")
        print(f"   Looking for similar team names in Odds API...")
        
        # Try fuzzy matching
        for (odds_away, odds_home), odds_id in odds_lookup.items():
            # Check if team names are similar (case-insensitive substring match)
            if (away.lower() in odds_away.lower() or odds_away.lower() in away.lower()) and \
               (home.lower() in odds_home.lower() or odds_home.lower() in home.lower()):
                matches.append({
                    'espn_game_id': espn_id,
                    'odds_event_id': odds_id,
                    'away_team': away,
                    'home_team': home,
                    'odds_away_team': odds_away,
                    'odds_home_team': odds_home,
                    'match_type': 'fuzzy'
                })
                print(f"   ⚠️  FUZZY MATCH:")
                print(f"      Odds ID: {odds_id}")
                print(f"      Odds teams: {odds_away} @ {odds_home}")
                print()
                break
        else:
            print(f"   ❌ No match found")
        print()

# ============================================================================
# STEP 4: Test fetching player props with matched IDs
# ============================================================================

print("="*80)
print("STEP 4: Test fetching player props")
print("="*80)

if matches:
    match = matches[0]  # Test with first match
    print(f"Testing with: {match['away_team']} @ {match['home_team']}")
    print(f"  ESPN ID: {match['espn_game_id']}")
    print(f"  Odds ID: {match['odds_event_id']}")
    print()
    
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{match['odds_event_id']}/odds"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'player_points',
        'oddsFormat': 'american'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10, verify=False)
        response.raise_for_status()
        
        data = response.json()
        bookmakers = data.get('bookmakers', [])
        
        print(f"✅ SUCCESS! Found {len(bookmakers)} bookmakers with player props")
        
        # Show sample data
        if bookmakers:
            first_bm = bookmakers[0]
            print(f"\nSample data from {first_bm.get('title')}:")
            markets = first_bm.get('markets', [])
            if markets:
                print(f"  Markets: {len(markets)}")
                first_market = markets[0]
                print(f"  Market: {first_market.get('key')}")
                outcomes = first_market.get('outcomes', [])
                print(f"  Outcomes: {len(outcomes)}")
                if outcomes:
                    print(f"\n  Sample outcomes (first 3):")
                    for outcome in outcomes[:3]:
                        print(f"    - {outcome.get('description')}: {outcome.get('name')} {outcome.get('point')} @ {outcome.get('price')}")
        
    except Exception as e:
        print(f"❌ Exception: {e}")
else:
    print("❌ No matches found - cannot test")

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"ESPN games: {len(espn_lookup)}")
print(f"Odds API events: {len(odds_lookup)}")
print(f"Matches found: {len(matches)}")
print()
print("SOLUTION for 10_live_betting_signal_generator.py:")
print("1. Fetch The Odds API events first: /v4/sports/basketball_nba/events")
print("2. Build lookup: {(away_team, home_team): odds_event_id}")
print("3. For each ESPN game, match by team names to get Odds event ID")
print("4. Use Odds event ID to fetch player props")
print("="*80)

# ============================================================================
# CODE TO ADD TO 10_live_betting_signal_generator.py
# ============================================================================

print()
print("="*80)
print("CODE SNIPPET FOR 10_live_betting_signal_generator.py")
print("="*80)
print()

code_snippet = '''
def fetch_odds_api_events() -> Dict[Tuple[str, str], str]:
    """
    Fetch all NBA events from The Odds API and return a lookup dict.
    
    Returns:
        Dict mapping (away_team, home_team) -> odds_event_id
    """
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    params = {'apiKey': ODDS_API_KEY}
    
    try:
        response = requests.get(url, params=params, timeout=10, verify=False)
        response.raise_for_status()
        
        events = response.json()
        
        # Build lookup: {(away_team, home_team): event_id}
        lookup = {}
        for event in events:
            away = event.get('away_team')
            home = event.get('home_team')
            event_id = event.get('id')
            
            if away and home and event_id:
                lookup[(away, home)] = event_id
        
        return lookup
        
    except Exception as e:
        print(f"⚠️  Error fetching Odds API events: {e}")
        return {}


def match_espn_to_odds_event(espn_game: Dict, odds_lookup: Dict[Tuple[str, str], str]) -> Optional[str]:
    """
    Match ESPN game to Odds API event by team names.
    
    Args:
        espn_game: ESPN game dict with 'home_team' and 'away_team'
        odds_lookup: Dict mapping (away_team, home_team) -> odds_event_id
    
    Returns:
        Odds API event ID or None if no match
    """
    away = espn_game.get('away_team')
    home = espn_game.get('home_team')
    
    if not away or not home:
        return None
    
    # Try exact match
    if (away, home) in odds_lookup:
        return odds_lookup[(away, home)]
    
    # Try fuzzy match (in case team names differ slightly)
    for (odds_away, odds_home), event_id in odds_lookup.items():
        if (away.lower() in odds_away.lower() or odds_away.lower() in away.lower()) and \\
           (home.lower() in odds_home.lower() or odds_home.lower() in home.lower()):
            return event_id
    
    return None


# UPDATE fetch_live_odds to use Odds API event ID instead of ESPN game ID:

def fetch_live_odds(game: Dict, odds_lookup: Dict, test_mode: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch live player prop odds from The Odds API.
    
    Args:
        game: ESPN game dict with team names
        odds_lookup: Dict mapping (away_team, home_team) -> odds_event_id
        test_mode: If True, return fake data for testing
    
    Returns:
        DataFrame with player props or None if unavailable
    """
    if test_mode:
        return generate_fake_live_odds()
    
    if not ODDS_API_KEY:
        print("⚠️  ODDS_API_KEY not set in environment")
        return None
    
    # Match ESPN game to Odds API event
    odds_event_id = match_espn_to_odds_event(game, odds_lookup)
    
    if not odds_event_id:
        print(f"⚠️  No matching Odds API event for {game['away_team']} @ {game['home_team']}")
        return None
    
    try:
        url = f"{ODDS_API_BASE_URL}/sports/basketball_nba/events/{odds_event_id}/odds"
        params = {
            'apiKey': ODDS_API_KEY,
            'regions': 'us',
            'markets': 'player_points',
            'oddsFormat': 'american'
        }
        
        response = requests.get(url, params=params, timeout=10, verify=False)
        
        if response.status_code == 404:
            return None
        
        response.raise_for_status()
        data = response.json()
        
        # Parse to DataFrame (existing logic)
        # ... rest of parsing code ...
        
    except Exception as e:
        print(f"⚠️  Error fetching live odds: {e}")
        return None


# IN MAIN LOOP, add this BEFORE the game loop:

    # Fetch Odds API events once per iteration (for matching)
    print("="*80)
    print("STEP 3: Fetching Odds API events for matching...")
    print("="*80)
    odds_lookup = fetch_odds_api_events()
    print(f"✅ Found {len(odds_lookup)} Odds API events")
    print()
    
    # Then in the game loop, pass odds_lookup to fetch_live_odds:
    for game in live_games:
        # ...
        live_odds_df = fetch_live_odds(game, odds_lookup, test_mode=test_mode)
        # ...
'''

print(code_snippet)
print()
print("="*80)
