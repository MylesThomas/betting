"""
Fetch NCAAF Championship Game odds from The Odds API.

Context:
Fetch real moneyline odds for the CFP National Championship game
on January 19, 2026 between Miami and Indiana.

Purpose:
- Call The Odds API for NCAAF game odds
- Filter to championship game on 1/19/2026
- Extract moneyline odds for both teams at each bookmaker
- Output in format ready to hardcode into generate_ncaaf_futures_viz.py

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 tmp/fetch_cfp_championship_odds.py
"""

# SSL Fix for macOS - must be imported BEFORE requests
import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
import os
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Monkey-patch requests to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# Load environment variables
load_dotenv()

API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4'


def fetch_ncaaf_games():
    """Fetch NCAAF game odds from The Odds API"""
    url = f"{BASE_URL}/sports/americanfootball_ncaaf/odds/"
    
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'h2h',  # head-to-head (moneyline)
        'oddsFormat': 'american'
    }
    
    print("🏈 Fetching NCAAF game odds from The Odds API...")
    print()
    
    response = requests.get(url, params=params)
    
    # Print API usage
    remaining = response.headers.get('x-requests-remaining')
    used = response.headers.get('x-requests-used')
    if remaining:
        print(f"📊 API Usage: {used} used, {remaining} remaining")
        print()
    
    response.raise_for_status()
    return response.json()


def find_championship_game(games):
    """Find the championship game (Miami vs Indiana on Jan 19, 2026)"""
    print("🔍 Looking for CFP Championship Game (Jan 19, 2026)...")
    print()
    
    championship_game = None
    
    for game in games:
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        commence_time = game.get('commence_time', '')
        
        # Parse game date
        game_date = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
        
        # Check if this is the championship game
        # (Miami vs Indiana on Jan 19, 2026)
        is_miami_indiana = (
            ('Miami' in home_team or 'Miami' in away_team) and
            ('Indiana' in home_team or 'Indiana' in away_team)
        )
        
        is_jan_19 = (
            game_date.year == 2026 and 
            game_date.month == 1 and 
            (game_date.day == 19 or game_date.day == 20)  # Account for timezone conversion
        )
        
        print(f"Game: {away_team} @ {home_team}")
        print(f"Date: {game_date.strftime('%Y-%m-%d %I:%M %p ET')}")
        print(f"Match: {'✅' if is_miami_indiana else '❌'} | Date: {'✅' if is_jan_19 else '❌'}")
        print()
        
        if is_miami_indiana and is_jan_19:
            championship_game = game
            print(f"🎯 FOUND CHAMPIONSHIP GAME!")
            print()
            break
    
    return championship_game


def extract_moneyline_odds(game):
    """Extract moneyline odds for both teams from all bookmakers"""
    if not game:
        print("❌ Championship game not found!")
        return None
    
    home_team = game['home_team']
    away_team = game['away_team']
    
    print("=" * 80)
    print(f"CFP NATIONAL CHAMPIONSHIP - MONEYLINE ODDS")
    print("=" * 80)
    print(f"Game: {away_team} @ {home_team}")
    print(f"Date: January 19, 2026")
    print()
    
    bookmaker_odds = []
    
    for bookmaker in game.get('bookmakers', []):
        book_key = bookmaker['key']
        book_title = bookmaker['title']
        
        # Find h2h (moneyline) market
        for market in bookmaker.get('markets', []):
            if market['key'] == 'h2h':
                outcomes = market['outcomes']
                
                # Extract odds for each team
                team_odds = {}
                for outcome in outcomes:
                    team_name = outcome['name']
                    odds = outcome['price']
                    team_odds[team_name] = odds
                
                bookmaker_odds.append({
                    'book_key': book_key,
                    'book_title': book_title,
                    'away_team': away_team,
                    'away_odds': team_odds.get(away_team),
                    'home_team': home_team,
                    'home_odds': team_odds.get(home_team)
                })
    
    # Print table
    print(f"{'Bookmaker':<20} {away_team:<25} {home_team:<25}")
    print("-" * 80)
    
    for book in bookmaker_odds:
        away_odds_str = f"{book['away_odds']:+d}" if book['away_odds'] else "N/A"
        home_odds_str = f"{book['home_odds']:+d}" if book['home_odds'] else "N/A"
        print(f"{book['book_title']:<20} {away_odds_str:<25} {home_odds_str:<25}")
    
    print()
    
    # Generate hardcoded data format
    print("=" * 80)
    print("HARDCODED DATA FOR generate_ncaaf_futures_viz.py")
    print("=" * 80)
    print()
    print("NCAAF_FUTURES_DATA = [")
    
    # Determine which team is favorite (negative odds)
    # Group by team
    teams = set()
    for book in bookmaker_odds:
        teams.add(book['away_team'])
        teams.add(book['home_team'])
    
    for team in sorted(teams):
        print(f"    # {team}")
        for book in bookmaker_odds:
            if book['away_team'] == team:
                odds = book['away_odds']
                record = '15-0' if 'Indiana' in team else '13-2'  # Update as needed
                print(f"    {{'bookmaker': '{book['book_key']}', 'team': '{team}', 'odds': {odds}, 'record': '{record}'}},")
            elif book['home_team'] == team:
                odds = book['home_odds']
                record = '15-0' if 'Indiana' in team else '13-2'  # Update as needed
                print(f"    {{'bookmaker': '{book['book_key']}', 'team': '{team}', 'odds': {odds}, 'record': '{record}'}},")
        print()
    
    print("]")
    print()
    
    return bookmaker_odds


def main():
    """Main function"""
    
    if not API_KEY:
        print("❌ ERROR: ODDS_API_KEY not found in .env file")
        return
    
    # Fetch games
    games = fetch_ncaaf_games()
    
    print(f"📊 Found {len(games)} NCAAF games")
    print()
    
    # Find championship game
    championship_game = find_championship_game(games)
    
    # Extract and display odds
    extract_moneyline_odds(championship_game)


if __name__ == "__main__":
    main()

