"""
Test The Odds API for NFL games on 2025-11-23
Getting pregame betting lines for the 12 Sunday games

This is notebook-friendly code - simple and straightforward
"""

import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import json
import ssl
import urllib3

# =============================================================================
# GLOBAL CONFIG
# =============================================================================

# SSL fix (needed for macOS)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4'
SPORT = 'americanfootball_nfl'
TARGET_DATE = '2025-11-23'  # Sunday with 12 games
MARKETS = 'h2h,spreads'  # Game outcome markets (moneyline and spreads)
REGIONS = 'us'
ODDS_FORMAT = 'american'

# Output directory for CSVs
OUTPUT_DIR = 'data/01_input/the-odds-api/nfl/game_lines'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_api_key():
    """Verify API key is loaded"""
    if not API_KEY or API_KEY == 'your_api_key_here':
        print("❌ ERROR: No valid API key found!")
        print("Make sure ODDS_API_KEY is set in your .env file")
        return False
    print(f"✅ API key loaded")
    return True


def get_historical_nfl_events(date=TARGET_DATE):
    """
    Get historical NFL events/games for a specific date
    
    NOTE: This costs 1 credit per request
    
    Args:
        date: Date string in format 'YYYY-MM-DD'
    
    Returns:
        List of event IDs and game info
    """
    url = f"{BASE_URL}/historical/sports/{SPORT}/events"
    
    # Use noon ET as the snapshot time
    timestamp = f"{date}T17:00:00Z"  # 12pm ET
    
    params = {
        'apiKey': API_KEY,
        'date': timestamp,
        'dateFormat': 'iso'
    }
    
    print(f"\n{'='*80}")
    print(f"Fetching historical NFL events for {date}...")
    print(f"Snapshot time: {timestamp}")
    print(f"{'='*80}")
    
    try:
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        
        data = response.json()
        
        # Print API usage
        cost = response.headers.get('x-requests-last', 'N/A')
        remaining = response.headers.get('x-requests-remaining', 'N/A')
        print(f"\n📊 API Usage: Cost {cost} credits, {remaining} remaining")
        
        events = data.get('data', [])
        print(f"✅ Found {len(events)} events")
        
        return events
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"Status code: {e.response.status_code}")
        if e.response.status_code == 401:
            print("Invalid API key!")
        elif e.response.status_code == 422:
            print("Historical API may require a paid plan or valid date")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def get_historical_event_odds(event_id, date=TARGET_DATE):
    """
    Get historical odds for a specific event
    
    NOTE: This costs 10 credits per event!
    
    Args:
        event_id: Event ID from get_historical_nfl_events
        date: Date string for timestamp
    
    Returns:
        Odds data for the event
    """
    url = f"{BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds"
    
    timestamp = f"{date}T17:00:00Z"  # 12pm ET
    
    params = {
        'apiKey': API_KEY,
        'date': timestamp,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        
        data = response.json()
        
        # Print API usage
        cost = response.headers.get('x-requests-last', 'N/A')
        remaining = response.headers.get('x-requests-remaining', 'N/A')
        
        event_data = data.get('data', {})
        game_str = f"{event_data.get('away_team', '?')} @ {event_data.get('home_team', '?')}"
        print(f"  ✓ {game_str:<50s} Cost: {cost:>3s} credits, Remaining: {remaining}")
        
        return event_data
        
    except requests.exceptions.HTTPError as e:
        print(f"  ❌ Error for event {event_id}: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def get_historical_nfl_games(date=TARGET_DATE):
    """
    Get all historical NFL games with their odds for a specific date
    
    WARNING: This is expensive! Costs 1 + (10 * num_games) credits
    For 12 games on Nov 23: 1 + (10 * 12) = 121 credits
    
    Args:
        date: Date string in format 'YYYY-MM-DD'
    
    Returns:
        List of game data with odds
    """
    # Step 1: Get all events (costs 1 credit)
    events = get_historical_nfl_events(date)
    
    if not events:
        print(f"\n❌ No events found for {date}")
        return []
    
    print(f"\n{'='*80}")
    print(f"Fetching odds for {len(events)} games (costs 10 credits each)...")
    print(f"TOTAL COST: {10 * len(events)} credits")
    print(f"{'='*80}\n")
    
    # Step 2: Get odds for each event (costs 10 credits each)
    games_with_odds = []
    
    for i, event in enumerate(events, 1):
        event_id = event.get('id')
        print(f"[{i}/{len(events)}]", end=" ")
        
        odds_data = get_historical_event_odds(event_id, date)
        
        if odds_data:
            games_with_odds.append(odds_data)
    
    print(f"\n✅ Successfully fetched {len(games_with_odds)} games with odds")
    
    return games_with_odds


def get_all_upcoming_nfl_games():
    """
    Get all upcoming NFL games (next few weeks)
    
    Returns:
        List of game events
    """
    url = f"{BASE_URL}/sports/{SPORT}/odds"
    
    params = {
        'apiKey': API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': 'iso'
    }
    
    print(f"\n{'='*80}")
    print(f"Fetching upcoming NFL games...")
    print(f"{'='*80}")
    
    try:
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        
        data = response.json()
        
        # Print API usage
        used = response.headers.get('x-requests-used', 'N/A')
        remaining = response.headers.get('x-requests-remaining', 'N/A')
        print(f"\n📊 API Usage: {used} used, {remaining} remaining")
        print(f"✅ Found {len(data)} total upcoming games")
        
        return data
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"Status code: {e.response.status_code}")
        if e.response.status_code == 401:
            print("Invalid API key!")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def get_nfl_games_live(date=TARGET_DATE):
    """
    Get NFL games from live/upcoming endpoint (filtered by date)
    
    NOTE: Only returns future games, not past games
    
    Args:
        date: Date string in format 'YYYY-MM-DD'
    
    Returns:
        List of game events
    """
    # First get all upcoming games
    all_games = get_all_upcoming_nfl_games()
    
    if not all_games:
        return []
    
    # Filter by target date
    target_games = []
    for game in all_games:
        game_time = game.get('commence_time', '')
        if game_time.startswith(date):
            target_games.append(game)
    
    print(f"\n✅ Found {len(target_games)} games on {date}")
    
    if len(target_games) == 0:
        print(f"\n⚠️  No games found on {date}")
        print(f"\n📅 Available game dates:")
        game_dates = set()
        for game in all_games:
            game_date = game.get('commence_time', '')[:10]
            game_dates.add(game_date)
        
        for gd in sorted(game_dates):
            games_on_date = [g for g in all_games if g.get('commence_time', '').startswith(gd)]
            print(f"   {gd}: {len(games_on_date)} games")
    
    return target_games


def get_nfl_games(date=TARGET_DATE, use_historical=False):
    """
    Get NFL games for a specific date
    
    Args:
        date: Date string in format 'YYYY-MM-DD'
        use_historical: If True, use historical API (costs credits!). If False, use live/upcoming API (free)
    
    Returns:
        List of game events
    """
    if use_historical:
        print(f"\n⚠️  Using HISTORICAL API (costs credits!)")
        return get_historical_nfl_games(date)
    else:
        print(f"\n Using LIVE/UPCOMING API (free)")
        return get_nfl_games_live(date)


def parse_game_lines(games):
    """
    Parse game betting lines into a clean DataFrame
    
    Args:
        games: List of game data from API
    
    Returns:
        DataFrame with betting lines
    """
    lines_list = []
    
    for game in games:
        game_id = game.get('id')
        game_time = game.get('commence_time')
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        # Parse each bookmaker
        for bookmaker in game.get('bookmakers', []):
            bookmaker_key = bookmaker.get('key')
            bookmaker_title = bookmaker.get('title')
            last_update = bookmaker.get('last_update')
            
            # Parse each market (h2h, spreads)
            for market in bookmaker.get('markets', []):
                market_key = market.get('key')
                outcomes = market.get('outcomes', [])
                
                # Organize outcomes by team
                outcome_dict = {o['name']: o for o in outcomes}
                
                if market_key == 'h2h':
                    # Moneyline odds
                    home_ml = outcome_dict.get(home_team, {}).get('price')
                    away_ml = outcome_dict.get(away_team, {}).get('price')
                    
                    lines_list.append({
                        'game_id': game_id,
                        'game_time': game_time,
                        'away_team': away_team,
                        'home_team': home_team,
                        'bookmaker': bookmaker_title,
                        'bookmaker_key': bookmaker_key,
                        'last_update': last_update,
                        'market': 'moneyline',
                        'away_line': away_ml,
                        'home_line': home_ml,
                        'spread_points': None
                    })
                    
                elif market_key == 'spreads':
                    # Spread odds
                    home_spread = outcome_dict.get(home_team, {}).get('point')
                    home_spread_odds = outcome_dict.get(home_team, {}).get('price')
                    away_spread = outcome_dict.get(away_team, {}).get('point')
                    away_spread_odds = outcome_dict.get(away_team, {}).get('price')
                    
                    lines_list.append({
                        'game_id': game_id,
                        'game_time': game_time,
                        'away_team': away_team,
                        'home_team': home_team,
                        'bookmaker': bookmaker_title,
                        'bookmaker_key': bookmaker_key,
                        'last_update': last_update,
                        'market': 'spread',
                        'away_line': f"{away_spread:+.1f} ({away_spread_odds:+d})" if away_spread else None,
                        'home_line': f"{home_spread:+.1f} ({home_spread_odds:+d})" if home_spread else None,
                        'spread_points': away_spread
                    })
    
    df = pd.DataFrame(lines_list)
    
    if not df.empty:
        # Convert game_time to datetime
        df['game_time'] = pd.to_datetime(df['game_time'])
        # Sort by game time then bookmaker
        df = df.sort_values(['game_time', 'bookmaker', 'market'])
    
    return df


def display_games_summary(df):
    """Display a nice summary of all games"""
    if df.empty:
        print("No data to display")
        return
    
    print(f"\n{'='*80}")
    print(f"NFL GAMES ON {TARGET_DATE}")
    print(f"{'='*80}\n")
    
    # Get unique games
    games = df[['game_time', 'away_team', 'home_team']].drop_duplicates()
    
    for idx, (_, row) in enumerate(games.iterrows(), 1):
        time_str = row['game_time'].strftime('%I:%M %p ET')
        print(f"{idx:2d}. {row['away_team']:25s} @ {row['home_team']:25s} ({time_str})")
    
    print(f"\nTotal games: {len(games)}")


def display_betting_lines(df, game_num=1):
    """
    Display betting lines for a specific game
    
    Args:
        df: DataFrame with all betting lines
        game_num: Which game to display (1-12)
    """
    if df.empty:
        print("No data to display")
        return
    
    # Get unique games sorted by time
    games = df[['game_time', 'away_team', 'home_team']].drop_duplicates().sort_values('game_time')
    
    if game_num > len(games):
        print(f"Only {len(games)} games available")
        return
    
    # Get the selected game
    game = games.iloc[game_num - 1]
    away = game['away_team']
    home = game['home_team']
    
    # Filter data for this game
    game_lines = df[(df['away_team'] == away) & (df['home_team'] == home)]
    
    print(f"\n{'='*80}")
    print(f"GAME {game_num}: {away} @ {home}")
    print(f"Time: {game['game_time'].strftime('%A %B %d, %Y at %I:%M %p ET')}")
    print(f"{'='*80}\n")
    
    # Display moneyline
    print("MONEYLINE:")
    print(f"{'Bookmaker':<20s} {'Away ML':<15s} {'Home ML':<15s}")
    print("-" * 50)
    
    ml_lines = game_lines[game_lines['market'] == 'moneyline']
    for _, row in ml_lines.iterrows():
        away_ml = f"{row['away_line']:+d}" if pd.notna(row['away_line']) else 'N/A'
        home_ml = f"{row['home_line']:+d}" if pd.notna(row['home_line']) else 'N/A'
        print(f"{row['bookmaker']:<20s} {away_ml:<15s} {home_ml:<15s}")
    
    # Display spreads
    print("\n\nSPREADS:")
    print(f"{'Bookmaker':<20s} {'Away Spread':<20s} {'Home Spread':<20s}")
    print("-" * 60)
    
    spread_lines = game_lines[game_lines['market'] == 'spread']
    for _, row in spread_lines.iterrows():
        away_spread = row['away_line'] if pd.notna(row['away_line']) else 'N/A'
        home_spread = row['home_line'] if pd.notna(row['home_line']) else 'N/A'
        print(f"{row['bookmaker']:<20s} {away_spread:<20s} {home_spread:<20s}")


def export_to_csv(df, date=TARGET_DATE, filename=None):
    """Export betting lines to CSV in proper data structure"""
    if df.empty:
        print("No data to export")
        return
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"nfl_game_lines_{date}_{timestamp}.csv"
    
    # Create output directory if needed
    # data/01_input/the-odds-api/nfl/game_lines/
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    
    print(f"\n💾 Saved to: {filepath}")
    print(f"   Rows: {len(df)}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_test(date=TARGET_DATE, use_historical=False):
    """
    Main function to test the NFL Odds API
    
    Args:
        date: Date string 'YYYY-MM-DD' (default: 2025-11-23)
        use_historical: If True, use historical API (costs ~121 credits for 12 games!)
                       If False, use live/upcoming API (free, ~2 credits)
    """
    print("="*80)
    print(f"NFL ODDS API TEST - {date}")
    print("="*80)
    
    print(f"\nMODE: {'HISTORICAL (costs credits)' if use_historical else 'LIVE/UPCOMING (free)'}")
    print(f"TARGET DATE: {date}")
    
    if use_historical:
        print(f"\n⚠️  WARNING: Historical API costs credits!")
        print(f"   - Getting events list: 1 credit")
        print(f"   - Getting odds per game: 10 credits each")
        print(f"   - For 12 games: ~121 credits total")
    
    # Step 1: Check API key
    if not check_api_key():
        return None
    
    # Step 2: Fetch games
    games = get_nfl_games(date, use_historical=use_historical)
    
    if not games:
        if not use_historical:
            print("\n⚠️  No games found on target date using live API.")
            print("   The live API only returns upcoming games, not historical games.")
            print("   For historical data, call: run_test(date='2025-11-23', use_historical=True)")
            print("\nContinuing with first available game as example...")
            
            # Get all upcoming games instead
            all_games = get_all_upcoming_nfl_games()
            if all_games:
                games = all_games[:1]  # Just use first game as example
                print(f"\nUsing example game: {games[0]['away_team']} @ {games[0]['home_team']}")
            else:
                print("\n❌ No games available at all.")
                return None
        else:
            print("\n❌ No historical games found.")
            return None
    
    # Step 3: Parse betting lines
    print(f"\n{'='*80}")
    print("Parsing betting lines...")
    print(f"{'='*80}")
    
    df = parse_game_lines(games)
    
    print(f"\n✅ Parsed {len(df)} betting lines from {df['bookmaker'].nunique()} bookmakers")
    
    # Step 4: Display results
    display_games_summary(df)
    
    # Display first few games in detail
    num_games_to_show = min(3, df['game_id'].nunique())
    print(f"\n\nSample betting lines for first {num_games_to_show} games:")
    for i in range(1, num_games_to_show + 1):
        display_betting_lines(df, game_num=i)
    
    # Step 5: Export to CSV
    export_to_csv(df, date=date)
    
    print(f"\n{'='*80}")
    print("✅ TEST COMPLETE")
    print(f"{'='*80}\n")
    
    # Quick data check
    print("QUICK STATS:")
    print(f"  Total games:      {df['game_id'].nunique()}")
    print(f"  Bookmakers:       {df['bookmaker'].nunique()}")
    print(f"  Markets:          {df['market'].nunique()}")
    print(f"  Total data rows:  {len(df)}")
    
    # Show available bookmakers
    print(f"\n  Available bookmakers:")
    for book in sorted(df['bookmaker'].unique()):
        print(f"    - {book}")
    
    return df


if __name__ == "__main__":
    # =============================================================================
    # USAGE EXAMPLES
    # =============================================================================
    #
    # OPTION 1: Get live/upcoming games (FREE)
    # df = run_test(date='2025-11-30', use_historical=False)
    #
    # OPTION 2: Get historical games for Nov 23 (COSTS ~121 CREDITS)
    # df = run_test(date='2025-11-23', use_historical=True)
    #
    # =============================================================================
    
    # Default: Run with live/upcoming API (free)
    df = run_test(date=TARGET_DATE, use_historical=False)
    
    # To get Nov 23 historical data, uncomment this line (costs ~121 credits):
    # df = run_test(date='2025-11-23', use_historical=True)

