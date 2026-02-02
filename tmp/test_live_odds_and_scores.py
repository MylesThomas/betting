"""
Test script to explore live NBA odds and scores APIs

PURPOSE:
Determine what data we can pull from:
1. The Odds API - live spreads and moneylines
2. ESPN API - live scores, clock, game status

Run this during live NBA games to see actual responses.

CONTEXT:
Building a live odds tracker that runs every 1 minute to capture spreads + moneylines
during games. Need to understand API response structure and data availability.

Outputs saved to ~/Downloads/tmp/ for inspection.

USAGE:
    python tmp/test_live_odds_and_scores.py --test odds     # Test Odds API only
    python tmp/test_live_odds_and_scores.py --test game     # Test ESPN only
    python tmp/test_live_odds_and_scores.py --test both     # Test both (default)
"""

import os
import sys
import json
import requests
import argparse
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Suppress SSL warnings for testing
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


# =============================================================================
# CONFIGURATION
# =============================================================================

ODDS_API_KEY = os.getenv('ODDS_API_KEY')
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'

ESPN_NBA_SCOREBOARD = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'

# Sports
SPORT_NBA = 'basketball_nba'

# Output directory
OUTPUT_DIR = Path.home() / 'Downloads' / 'tmp'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EMOJI MAP
# =============================================================================

EMOJI = {
    'success': '✅',
    'error': '❌',
    'info': 'ℹ️',
    'chart': '📊',
    'nba': '🏀',
    'time': '⏰',
    'money': '💰',
    'fire': '🔥',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_current_time_et():
    """Get current time in ET for display."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo('America/New_York'))


def format_timestamp(dt):
    """Format datetime for display."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
    return dt.strftime('%Y-%m-%d %I:%M %p %Z')


def ml_to_implied_prob(odds: int) -> float:
    """
    Convert American odds to implied probability.
    
    Examples:
        -150 → 60% (favorite)
        +130 → 43.5% (underdog)
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def remove_vig(away_prob: float, home_prob: float) -> tuple:
    """
    Remove vig to get true probabilities (sum to 100%).
    
    Example:
        Away: 60%, Home: 43.5% → 103.5% total (vig)
        True: Away 58%, Home 42%
    """
    total = away_prob + home_prob
    return (away_prob / total, home_prob / total)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_odds_api_live_games():
    """
    Test 1: Fetch live NBA games from Odds API.
    
    Goal: See what data structure we get for live odds.
    """
    print(f"\n{'='*80}")
    print(f"{EMOJI['nba']} TEST 1: THE ODDS API - LIVE NBA GAMES")
    print(f"{'='*80}\n")
    
    url = f"{ODDS_API_BASE}/sports/{SPORT_NBA}/odds"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'spreads,h2h',  # Spreads + moneylines
        'oddsFormat': 'american',
        # NO bookmakers filter - get ALL available books
    }
    
    print(f"{EMOJI['info']} Endpoint: {url}")
    print(f"{EMOJI['info']} Markets: spreads, h2h (moneylines)")
    print(f"{EMOJI['info']} Bookmakers: ALL (not filtered)\n")
    
    try:
        response = requests.get(url, params=params, timeout=10, verify=False)
        response.raise_for_status()
        
        games = response.json()
        
        print(f"{EMOJI['success']} API Response: {response.status_code}")
        print(f"{EMOJI['chart']} Total games returned: {len(games)}\n")
        
        # Check API usage
        remaining = response.headers.get('x-requests-remaining', 'unknown')
        used = response.headers.get('x-requests-used', 'unknown')
        print(f"{EMOJI['money']} API Usage: {used} used, {remaining} remaining\n")
        
        if not games:
            print(f"{EMOJI['info']} No games found. Either:")
            print("   1. No NBA games currently live")
            print("   2. All games are upcoming (not started yet)")
            print("   3. All games are finished\n")
            return None
        
        # Filter to likely live games (started within last 4 hours)
        now = datetime.now(timezone.utc)
        live_games = []
        
        for game in games:
            commence_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
            time_since_start = (now - commence_time).total_seconds() / 3600  # hours
            
            # Game started and not too long ago (likely still live)
            if 0 < time_since_start < 4:
                live_games.append(game)
        
        print(f"{EMOJI['fire']} Likely LIVE games: {len(live_games)}\n")
        
        if not live_games:
            print(f"{EMOJI['info']} No live games detected right now.")
            print(f"\nShowing first game anyway (for structure reference):\n")
            live_games = [games[0]]
        
        # Print detailed info for first live game (to analyze all bookmakers)
        for idx, game in enumerate(live_games[:1], 1):  # Show 1 game only (full detail)
            print(f"{'-'*80}")
            print(f"GAME {idx}: {game['away_team']} @ {game['home_team']}")
            print(f"{'-'*80}")
            
            commence_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
            time_since_start = (now - commence_time).total_seconds() / 60  # minutes
            
            print(f"Game ID: {game['id']}")
            print(f"Scheduled Start: {format_timestamp(commence_time)}")
            print(f"Time Since Start: {time_since_start:.0f} minutes ago")
            print(f"Bookmakers Available: {len(game.get('bookmakers', []))}\n")
            
            # Show odds from ALL bookmakers (analyze which update fastest)
            bookmakers = game.get('bookmakers', [])
            print(f"\nAnalyzing {len(bookmakers)} bookmakers...\n")
            
            # Collect update times for analysis
            book_updates = []
            
            for book in bookmakers:
                last_update = datetime.fromisoformat(book['last_update'].replace('Z', '+00:00'))
                seconds_ago = (now - last_update).total_seconds()
                
                book_updates.append({
                    'key': book['key'],
                    'last_update': last_update,
                    'seconds_ago': seconds_ago,
                })
                
                print(f"\n📗 {book['key'].upper()}")
                print(f"   Last Update: {format_timestamp(book['last_update'])} ({seconds_ago:.0f}s ago)")
                
                # Find spreads market
                spreads_market = next((m for m in book['markets'] if m['key'] == 'spreads'), None)
                h2h_market = next((m for m in book['markets'] if m['key'] == 'h2h'), None)
                
                if spreads_market:
                    print(f"\n   SPREADS:")
                    for outcome in spreads_market['outcomes']:
                        spread_val = outcome.get('point', 'N/A')
                        price = outcome.get('price', 'N/A')
                        print(f"      {outcome['name']}: {spread_val:+.1f} @ {price:+d}")
                
                if h2h_market:
                    print(f"\n   MONEYLINES:")
                    outcomes = h2h_market['outcomes']
                    
                    for outcome in outcomes:
                        price = outcome['price']
                        implied_prob = ml_to_implied_prob(price)
                        print(f"      {outcome['name']}: {price:+d} (implied: {implied_prob:.1%})")
                    
                    # Calculate no-vig probabilities
                    if len(outcomes) == 2:
                        prob1 = ml_to_implied_prob(outcomes[0]['price'])
                        prob2 = ml_to_implied_prob(outcomes[1]['price'])
                        true_prob1, true_prob2 = remove_vig(prob1, prob2)
                        
                        vig = (prob1 + prob2 - 1) * 100
                        print(f"\n   VIG ANALYSIS:")
                        print(f"      Total prob (with vig): {(prob1 + prob2):.1%}")
                        print(f"      Vig: {vig:.2f}%")
                        print(f"      True probabilities:")
                        print(f"         {outcomes[0]['name']}: {true_prob1:.1%}")
                        print(f"         {outcomes[1]['name']}: {true_prob2:.1%}")
            
            # Summary: Which books updated most recently?
            print(f"\n\n{'='*60}")
            print(f"BOOKMAKER UPDATE SPEED ANALYSIS")
            print(f"{'='*60}\n")
            
            # Sort by most recent update
            book_updates_sorted = sorted(book_updates, key=lambda x: x['seconds_ago'])
            
            print(f"Ranked by update recency (most recent first):\n")
            for i, book in enumerate(book_updates_sorted, 1):
                freshness = "🟢 FRESH" if book['seconds_ago'] < 30 else "🟡 STALE" if book['seconds_ago'] < 60 else "🔴 OLD"
                print(f"  {i}. {book['key']:20s} - {book['seconds_ago']:5.0f}s ago  {freshness}")
            
            print(f"\n{EMOJI['info']} Recommendation: Use books that update <30s ago for live tracking")
            print(f"{EMOJI['info']} Run this test multiple times during live games to find consistently fast books\n")
        
        # Save full response for inspection
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = OUTPUT_DIR / f'live_odds_response_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(games, f, indent=2)
        print(f"\n\n{EMOJI['success']} Full API response saved to: {output_file}")
        
        return games
        
    except requests.exceptions.RequestException as e:
        print(f"{EMOJI['error']} API Error: {e}")
        return None


def test_espn_scoreboard():
    """
    Test 2: Fetch live NBA scores from ESPN API.
    
    Goal: Get game status, scores, period, time remaining.
    """
    print(f"\n\n{'='*80}")
    print(f"{EMOJI['nba']} TEST 2: ESPN API - NBA SCOREBOARD")
    print(f"{'='*80}\n")
    
    url = ESPN_NBA_SCOREBOARD
    
    print(f"{EMOJI['info']} Endpoint: {url}")
    print(f"{EMOJI['info']} No API key required (free, public endpoint)\n")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"{EMOJI['success']} API Response: {response.status_code}\n")
        
        events = data.get('events', [])
        print(f"{EMOJI['chart']} Total games: {len(events)}\n")
        
        if not events:
            print(f"{EMOJI['info']} No games found on ESPN scoreboard.\n")
            return None
        
        # Find live games
        live_games = [e for e in events if e['status']['type']['state'] == 'in']
        
        print(f"{EMOJI['fire']} LIVE games: {len(live_games)}\n")
        
        if not live_games:
            print(f"{EMOJI['info']} No live games right now.")
            print(f"\nShowing first game anyway (for structure reference):\n")
            live_games = [events[0]]
        
        # Print detailed info for live games
        for idx, event in enumerate(live_games[:2], 1):  # Show max 2 games
            print(f"{'-'*80}")
            
            competition = event['competitions'][0]
            competitors = competition['competitors']
            
            away_team = next(c for c in competitors if c['homeAway'] == 'away')
            home_team = next(c for c in competitors if c['homeAway'] == 'home')
            
            print(f"GAME {idx}: {away_team['team']['displayName']} @ {home_team['team']['displayName']}")
            print(f"{'-'*80}")
            
            status = event['status']
            
            print(f"\nGame ID: {event['id']}")
            print(f"Status: {status['type']['description']}")
            print(f"State: {status['type']['state']} ({'LIVE' if status['type']['state'] == 'in' else 'NOT LIVE'})")
            
            print(f"\nSCORE:")
            print(f"   {away_team['team']['displayName']}: {away_team['score']}")
            print(f"   {home_team['team']['displayName']}: {home_team['score']}")
            
            if status['type']['state'] == 'in':
                print(f"\nGAME CLOCK:")
                print(f"   Period: {status.get('period', 'N/A')}")
                print(f"   Display Clock: {status.get('displayClock', 'N/A')}")
                
                # Calculate time remaining in minutes
                clock_str = status.get('displayClock', '0:00')
                try:
                    parts = clock_str.split(':')
                    if len(parts) == 2:
                        mins = int(parts[0])
                        secs = int(parts[1])
                        time_remaining = mins + secs / 60
                        print(f"   Time Remaining (decimal): {time_remaining:.2f} minutes")
                except:
                    print(f"   Time Remaining (decimal): Unable to parse")
            
            # Check for additional metadata
            if 'situation' in competition:
                print(f"\nSITUATION:")
                situation = competition['situation']
                print(f"   Possession: {situation.get('possession', 'N/A')}")
                print(f"   Down Distance: {situation.get('downDistanceText', 'N/A')}")
        
        # Save full response
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = OUTPUT_DIR / f'espn_scoreboard_response_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n\n{EMOJI['success']} Full API response saved to: {output_file}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"{EMOJI['error']} API Error: {e}")
        return None


def test_team_name_matching():
    """
    Test 3: Check if team names match between Odds API and ESPN.
    
    Goal: Identify any name mismatches we need to handle.
    """
    print(f"\n\n{'='*80}")
    print(f"{EMOJI['chart']} TEST 3: TEAM NAME MATCHING")
    print(f"{'='*80}\n")
    
    print("This test requires both APIs to have live games.")
    print(f"Check the saved JSON files in {OUTPUT_DIR} to compare team names.\n")
    
    print(f"{EMOJI['info']} Common mismatches to watch for:")
    print("   - Odds API: 'Los Angeles Lakers'")
    print("   - ESPN: 'Los Angeles Lakers' (usually match, but check)")
    print("   - Some books abbreviate: 'LA Lakers', 'L.A. Lakers'")
    print("\nRecommendation: Build team name normalization map if needed.\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run tests based on CLI flags."""
    parser = argparse.ArgumentParser(description='Test live NBA odds and scores APIs')
    parser.add_argument(
        '--test',
        choices=['odds', 'game', 'both'],
        default='both',
        help='Which API to test: odds (Odds API), game (ESPN), both (default)'
    )
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"{EMOJI['nba']} LIVE ODDS & SCORES API TESTING")
    print(f"{'='*80}")
    
    current_time = get_current_time_et()
    print(f"{EMOJI['time']} Current Time: {current_time.strftime('%Y-%m-%d %I:%M %p ET')}")
    print(f"{EMOJI['info']} Output Directory: {OUTPUT_DIR}")
    print(f"{EMOJI['info']} Test Mode: {args.test}\n")
    
    # Test Odds API
    if args.test in ['odds', 'both']:
        if not ODDS_API_KEY:
            print(f"{EMOJI['error']} ERROR: ODDS_API_KEY not found in environment.")
            print("Set it with: export ODDS_API_KEY='your_key_here'")
            if args.test == 'odds':
                return
            print("\nSkipping Odds API test, continuing with ESPN...\n")
        else:
            print(f"{EMOJI['success']} Odds API Key: {ODDS_API_KEY[:10]}...{ODDS_API_KEY[-4:]}\n")
            odds_games = test_odds_api_live_games()
    
    # Test ESPN Scoreboard
    if args.test in ['game', 'both']:
        espn_data = test_espn_scoreboard()
    
    # Team name matching info
    if args.test == 'both':
        test_team_name_matching()
    
    print(f"\n{'='*80}")
    print(f"{EMOJI['success']} TESTING COMPLETE")
    print(f"{'='*80}\n")
    
    print("NEXT STEPS:")
    print(f"1. Check {OUTPUT_DIR} for saved JSON responses")
    print("2. Verify team name matching between APIs")
    print("3. Design Parquet schema for DuckDB storage")
    print("4. Implement live tracker using these API patterns\n")


if __name__ == '__main__':
    main()
