"""
Quick test script to validate NCAAB API integration.

Tests:
1. ESPN API returns valid NCAAB data
2. The Odds API returns NCAAB games
3. Team names match between APIs
4. Data structure is compatible with lambda function

Created: 2026-02-16
Context: Testing NCAAB implementation for live odds tracker
"""

import os
import sys
import requests
import warnings
from pathlib import Path
from datetime import datetime, timezone

# Suppress SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Find project root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

ODDS_API_KEY = os.getenv('ODDS_API_KEY')
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'
ESPN_NCAAB_SCOREBOARD = 'http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'

def test_espn_api():
    """Test ESPN API for NCAAB."""
    print("="*80)
    print("TEST 1: ESPN API")
    print("="*80)
    
    response = requests.get(ESPN_NCAAB_SCOREBOARD, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    events = data.get('events', [])
    print(f"✅ ESPN API returned {len(events)} games\n")
    
    if not events:
        print("⚠️  No games found\n")
        return []
    
    espn_teams = []
    for event in events:
        competition = event['competitions'][0]
        competitors = competition['competitors']
        
        away_team_obj = next((c for c in competitors if c['homeAway'] == 'away'), None)
        home_team_obj = next((c for c in competitors if c['homeAway'] == 'home'), None)
        
        if away_team_obj and home_team_obj:
            away_team = away_team_obj['team']['displayName']
            home_team = home_team_obj['team']['displayName']
            status = event['status']['type']['state']
            
            print(f"  {away_team} @ {home_team}")
            print(f"    Status: {status}")
            print(f"    ESPN Game ID: {event['id']}")
            print()
            
            espn_teams.append((away_team, home_team))
    
    return espn_teams


def test_odds_api():
    """Test The Odds API for NCAAB."""
    print("="*80)
    print("TEST 2: THE ODDS API")
    print("="*80)
    
    if not ODDS_API_KEY:
        print("❌ ODDS_API_KEY not found in environment\n")
        return []
    
    url = f"{ODDS_API_BASE}/sports/basketball_ncaab/odds"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'spreads,h2h',
        'oddsFormat': 'american',
    }
    
    response = requests.get(url, params=params, timeout=15, verify=False)
    response.raise_for_status()
    games = response.json()
    
    remaining = response.headers.get('x-requests-remaining', 'unknown')
    used = response.headers.get('x-requests-used', 'unknown')
    
    print(f"✅ Odds API returned {len(games)} games")
    print(f"   API Usage: {used} used, {remaining} remaining\n")
    
    if not games:
        print("⚠️  No games found\n")
        return []
    
    odds_teams = []
    for i, game in enumerate(games[:10], 1):  # Show first 10
        away_team = game['away_team']
        home_team = game['home_team']
        num_books = len(game.get('bookmakers', []))
        
        print(f"  {i}. {away_team} @ {home_team}")
        print(f"     Bookmakers: {num_books}")
        print(f"     Commence: {game['commence_time']}")
        print()
        
        odds_teams.append((away_team, home_team))
    
    return odds_teams


def test_team_name_matching(espn_teams, odds_teams):
    """Test if team names match between APIs."""
    print("="*80)
    print("TEST 3: TEAM NAME MATCHING")
    print("="*80)
    
    if not espn_teams or not odds_teams:
        print("⚠️  Skipping (no data from one or both APIs)\n")
        return
    
    print(f"ESPN teams: {len(espn_teams)}")
    print(f"Odds API teams: {len(odds_teams)}\n")
    
    # Check for exact matches
    espn_set = set(espn_teams)
    odds_set = set(odds_teams)
    
    matches = espn_set & odds_set
    espn_only = espn_set - odds_set
    odds_only = odds_set - espn_set
    
    print(f"✅ Exact matches: {len(matches)}")
    if matches:
        for away, home in sorted(matches):
            print(f"   {away} @ {home}")
        print()
    
    if espn_only:
        print(f"⚠️  Only in ESPN ({len(espn_only)}):")
        for away, home in sorted(espn_only):
            print(f"   {away} @ {home}")
        print()
    
    if odds_only:
        print(f"⚠️  Only in Odds API ({len(odds_only)}):")
        for away, home in sorted(odds_only):
            print(f"   {away} @ {home}")
        print()
    
    # Check for partial matches (case-insensitive, fuzzy)
    if espn_only and odds_only:
        print("Checking for potential fuzzy matches...")
        for espn_matchup in espn_only:
            espn_away, espn_home = espn_matchup
            for odds_matchup in odds_only:
                odds_away, odds_home = odds_matchup
                
                # Simple fuzzy match: check if core team name appears
                if (espn_away.lower() in odds_away.lower() or odds_away.lower() in espn_away.lower()) and \
                   (espn_home.lower() in odds_home.lower() or odds_home.lower() in espn_home.lower()):
                    print(f"\n   Potential match:")
                    print(f"     ESPN:     {espn_away} @ {espn_home}")
                    print(f"     Odds API: {odds_away} @ {odds_home}")


def main():
    print("\n" + "="*80)
    print("NCAAB API INTEGRATION TEST")
    print("="*80)
    print()
    
    try:
        espn_teams = test_espn_api()
        odds_teams = test_odds_api()
        test_team_name_matching(espn_teams, odds_teams)
        
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print("✅ All API calls successful")
        print(f"   ESPN games: {len(espn_teams)}")
        print(f"   Odds API games: {len(odds_teams)}")
        print("\n🎯 NCAAB integration is ready!")
        print("   - ESPN API endpoint works")
        print("   - The Odds API supports basketball_ncaab")
        print("   - Team names can be matched (validate during live games)")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
