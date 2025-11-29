"""
Fetch Ad-hoc Odds Data from The Odds API

Flexible script to fetch current or historical betting odds for any sport and market(s).
Saves results to data/01_input/the-odds-api/adhoc/

Usage (CURRENT ODDS - default, cheap):
    python fetch_adhoc_odds.py --sport americanfootball_nfl --markets spreads,h2h
    python fetch_adhoc_odds.py -s basketball_nba -m player_points
    python fetch_adhoc_odds.py -s icehockey_nhl -m h2h,spreads,totals

Usage (HISTORICAL ODDS - expensive):
    python fetch_adhoc_odds.py --sport americanfootball_nfl --markets spreads,h2h --historical --date 2025-11-28
    python fetch_adhoc_odds.py -s basketball_nba -m player_points --historical -d 2025-11-01 --hour 17

Common Sports:
    - americanfootball_nfl
    - basketball_nba
    - icehockey_nhl
    - baseball_mlb
    - soccer_epl
    
Common Markets:
    - h2h (moneyline)
    - spreads
    - totals (over/under)
    - player_points, player_rebounds, player_assists (props)
    - outrights
"""

import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import json
import ssl
import urllib3
import time
import argparse
from pathlib import Path

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
REGIONS = 'us'
ODDS_FORMAT = 'american'

# Output directory
OUTPUT_DIR = '/Users/thomasmyles/dev/betting/data/01_input/the-odds-api/adhoc'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default snapshot hour (UTC) - 12pm ET = 5pm UTC
DEFAULT_SNAPSHOT_HOUR = 17

RATE_LIMIT_DELAY = 0.5  # seconds between API calls

# API usage tracking
credits_remaining = None
credits_used = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_api_key():
    """Verify API key is loaded"""
    if not API_KEY or API_KEY == 'your_api_key_here':
        print("❌ ERROR: No valid API key found!")
        print("Make sure ODDS_API_KEY is set in your .env file")
        return False
    return True


def get_historical_events(sport, date_str, snapshot_hour=None):
    """
    Get historical events/games for a specific sport and date
    
    NOTE: This costs 1 credit per request
    
    Args:
        sport: Sport key (e.g., 'americanfootball_nfl')
        date_str: Date string in format 'YYYY-MM-DD'
        snapshot_hour: Optional hour (UTC) for snapshot
    
    Returns:
        Dict with events list, cost, remaining credits
    """
    global credits_remaining, credits_used
    
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    hour_to_use = snapshot_hour if snapshot_hour is not None else DEFAULT_SNAPSHOT_HOUR
    timestamp = date_obj.replace(hour=hour_to_use, minute=0, second=0).isoformat() + 'Z'
    
    url = f"{BASE_URL}/historical/sports/{sport}/events"
    
    params = {
        'apiKey': API_KEY,
        'date': timestamp,
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        
        data = response.json()
        
        # Get API usage
        credits_remaining = int(response.headers.get('x-requests-remaining', 0))
        credits_used = int(response.headers.get('x-requests-used', 0))
        cost = int(response.headers.get('x-requests-last', 0))
        
        events = data.get('data', [])
        
        return {
            'events': events,
            'cost': cost,
            'remaining': credits_remaining
        }
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            # No data for this date
            return {'events': [], 'cost': 0, 'remaining': credits_remaining}
        print(f"❌ HTTP Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def get_current_odds(sport, markets):
    """
    Get current/live odds for all upcoming games in a sport
    
    NOTE: This costs 1 credit total (much cheaper than historical!)
    
    Args:
        sport: Sport key (e.g., 'americanfootball_nfl')
        markets: Comma-separated market keys (e.g., 'h2h,spreads')
    
    Returns:
        List of games with odds data
    """
    global credits_remaining, credits_used
    
    url = f"{BASE_URL}/sports/{sport}/odds"
    
    params = {
        'apiKey': API_KEY,
        'regions': REGIONS,
        'markets': markets,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        
        data = response.json()
        
        # Get API usage
        credits_remaining = int(response.headers.get('x-requests-remaining', 0))
        credits_used = int(response.headers.get('x-requests-used', 0))
        cost = int(response.headers.get('x-requests-last', 0))
        
        games = data if isinstance(data, list) else []
        
        return {
            'games': games,
            'cost': cost,
            'remaining': credits_remaining
        }
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def get_historical_event_odds(sport, event_id, date_str, markets, snapshot_hour=None):
    """
    Get historical odds for a specific event
    
    NOTE: This costs 10 credits per event!
    
    Args:
        sport: Sport key
        event_id: Event ID from get_historical_events
        date_str: Date string for timestamp
        markets: Comma-separated market keys (e.g., 'h2h,spreads')
        snapshot_hour: Optional hour (UTC) for snapshot
    
    Returns:
        Dict with data, cost, remaining
    """
    global credits_remaining, credits_used
    
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    hour_to_use = snapshot_hour if snapshot_hour is not None else DEFAULT_SNAPSHOT_HOUR
    timestamp = date_obj.replace(hour=hour_to_use, minute=0, second=0).isoformat() + 'Z'
    
    url = f"{BASE_URL}/historical/sports/{sport}/events/{event_id}/odds"
    
    params = {
        'apiKey': API_KEY,
        'date': timestamp,
        'regions': REGIONS,
        'markets': markets,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        
        data = response.json()
        
        # Get API usage
        credits_remaining = int(response.headers.get('x-requests-remaining', 0))
        credits_used = int(response.headers.get('x-requests-used', 0))
        cost = int(response.headers.get('x-requests-last', 0))
        
        event_data = data.get('data', {})
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
        
        return {
            'data': event_data,
            'cost': cost,
            'remaining': credits_remaining
        }
        
    except requests.exceptions.HTTPError as e:
        print(f"  ❌ Error for event {event_id[:8]}: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def parse_odds_data(games, markets_list):
    """
    Parse game betting lines into a clean DataFrame
    
    Args:
        games: List of game data from API
        markets_list: List of market keys being fetched
    
    Returns:
        DataFrame with betting lines
    """
    lines_list = []
    
    for game in games:
        game_id = game.get('id')
        game_time = game.get('commence_time')
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        sport = game.get('sport_key')
        sport_title = game.get('sport_title')
        
        # Parse each bookmaker
        for bookmaker in game.get('bookmakers', []):
            bookmaker_key = bookmaker.get('key')
            bookmaker_title = bookmaker.get('title')
            last_update = bookmaker.get('last_update')
            
            # Parse each market
            for market in bookmaker.get('markets', []):
                market_key = market.get('key')
                outcomes = market.get('outcomes', [])
                
                # Organize outcomes by name/team
                outcome_dict = {o.get('name', o.get('description', '')): o for o in outcomes}
                
                # Different parsing logic based on market type
                if market_key == 'h2h':
                    # Moneyline odds
                    away_odds = outcome_dict.get(away_team, {}).get('price')
                    home_odds = outcome_dict.get(home_team, {}).get('price')
                    
                    lines_list.append({
                        'sport': sport,
                        'sport_title': sport_title,
                        'game_id': game_id,
                        'game_time': game_time,
                        'away_team': away_team,
                        'home_team': home_team,
                        'bookmaker': bookmaker_title,
                        'bookmaker_key': bookmaker_key,
                        'last_update': last_update,
                        'market': 'h2h',
                        'away_odds': away_odds,
                        'home_odds': home_odds,
                        'away_value': None,
                        'home_value': None
                    })
                
                elif market_key == 'spreads':
                    # Spread odds
                    home_spread = outcome_dict.get(home_team, {}).get('point')
                    home_spread_odds = outcome_dict.get(home_team, {}).get('price')
                    away_spread = outcome_dict.get(away_team, {}).get('point')
                    away_spread_odds = outcome_dict.get(away_team, {}).get('price')
                    
                    lines_list.append({
                        'sport': sport,
                        'sport_title': sport_title,
                        'game_id': game_id,
                        'game_time': game_time,
                        'away_team': away_team,
                        'home_team': home_team,
                        'bookmaker': bookmaker_title,
                        'bookmaker_key': bookmaker_key,
                        'last_update': last_update,
                        'market': 'spread',
                        'away_odds': away_spread_odds,
                        'home_odds': home_spread_odds,
                        'away_value': away_spread,
                        'home_value': home_spread
                    })
                
                elif market_key == 'totals':
                    # Totals (over/under)
                    over_outcome = outcome_dict.get('Over', {})
                    under_outcome = outcome_dict.get('Under', {})
                    
                    lines_list.append({
                        'sport': sport,
                        'sport_title': sport_title,
                        'game_id': game_id,
                        'game_time': game_time,
                        'away_team': away_team,
                        'home_team': home_team,
                        'bookmaker': bookmaker_title,
                        'bookmaker_key': bookmaker_key,
                        'last_update': last_update,
                        'market': 'totals',
                        'away_odds': over_outcome.get('price'),  # Using away for Over
                        'home_odds': under_outcome.get('price'),  # Using home for Under
                        'away_value': over_outcome.get('point'),
                        'home_value': under_outcome.get('point')
                    })
                
                else:
                    # Generic parsing for player props and other markets
                    # Store all outcomes as separate rows
                    for outcome_name, outcome in outcome_dict.items():
                        lines_list.append({
                            'sport': sport,
                            'sport_title': sport_title,
                            'game_id': game_id,
                            'game_time': game_time,
                            'away_team': away_team,
                            'home_team': home_team,
                            'bookmaker': bookmaker_title,
                            'bookmaker_key': bookmaker_key,
                            'last_update': last_update,
                            'market': market_key,
                            'outcome_name': outcome_name,
                            'outcome_description': outcome.get('description', ''),
                            'odds': outcome.get('price'),
                            'point': outcome.get('point')
                        })
    
    if not lines_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(lines_list)
    
    if not df.empty:
        # Convert game_time to datetime
        df['game_time'] = pd.to_datetime(df['game_time'])
        # Sort by game time then bookmaker
        df = df.sort_values(['game_time', 'bookmaker', 'market'])
    
    return df


def fetch_current_odds(sport, markets):
    """
    Fetch current/live odds for all upcoming games
    
    Args:
        sport: Sport key (e.g., 'americanfootball_nfl')
        markets: Comma-separated market keys (e.g., 'h2h,spreads')
    
    Returns:
        DataFrame with all current odds
    """
    print(f"\n{'='*80}")
    print(f"FETCHING CURRENT ODDS")
    print(f"{'='*80}")
    print(f"Sport: {sport}")
    print(f"Markets: {markets}")
    print(f"Mode: Live/Current (real-time)")
    
    # Get current odds (single API call!)
    result = get_current_odds(sport, markets)
    
    if result is None:
        print(f"❌ API Error")
        return pd.DataFrame()
    
    games = result['games']
    
    if not games:
        print(f"  ⚠️  No upcoming games found")
        return pd.DataFrame()
    
    print(f"\n  Found {len(games)} upcoming games:")
    for game in games[:10]:  # Show first 10
        commence_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
        time_str = commence_time.strftime('%m/%d %I:%M %p')
        print(f"    • {game['away_team']} @ {game['home_team']} ({time_str})")
    
    if len(games) > 10:
        print(f"    ... and {len(games) - 10} more")
    
    print(f"\n  💰 Cost: {result['cost']} credits (single API call)")
    print(f"     Remaining: {result['remaining']:,} credits")
    
    # Parse and return DataFrame
    markets_list = [m.strip() for m in markets.split(',')]
    df = parse_odds_data(games, markets_list)
    
    if df.empty:
        print(f"  ⚠️  No lines parsed")
        return df
    
    return df


def fetch_historical_odds(sport, date_str, markets, snapshot_hour=None):
    """
    Fetch historical odds for a specific date
    
    Args:
        sport: Sport key (e.g., 'americanfootball_nfl')
        date_str: Date in YYYY-MM-DD format
        markets: Comma-separated market keys (e.g., 'h2h,spreads')
        snapshot_hour: Optional hour (UTC) for snapshot
    
    Returns:
        DataFrame with all odds for that date
    """
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    day_of_week = date_obj.strftime('%A')
    
    print(f"\n{'='*80}")
    print(f"FETCHING HISTORICAL ODDS")
    print(f"{'='*80}")
    print(f"Date: {date_str} ({day_of_week})")
    print(f"Sport: {sport}")
    print(f"Markets: {markets}")
    if snapshot_hour:
        print(f"Snapshot hour: {snapshot_hour}:00 UTC")
    else:
        print(f"Snapshot hour: {DEFAULT_SNAPSHOT_HOUR}:00 UTC (default)")
    
    # Get events for that date
    result = get_historical_events(sport, date_str, snapshot_hour)
    
    if result is None:
        print(f"❌ API Error for {date_str}")
        return pd.DataFrame()
    
    events = result['events']
    
    if not events:
        print(f"  ⚠️  No events found")
        return pd.DataFrame()
    
    print(f"  Found {len(events)} events:")
    for event in events:
        commence_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
        time_str = commence_time.strftime('%I:%M %p')
        print(f"    • {event['away_team']} @ {event['home_team']} ({time_str})")
    
    estimated_cost = 1 + (len(events) * 10)
    print(f"\n  💰 Estimated cost: 1 + ({len(events)} × 10) = {estimated_cost} credits")
    print(f"     Current balance: {result['remaining']:,} credits")
    
    # Ask for confirmation
    response = input(f"\n  Continue? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print("  ❌ Cancelled by user")
        return pd.DataFrame()
    
    # Fetch odds for each event
    games_with_odds = []
    
    for i, event in enumerate(events, 1):
        game_desc = f"{event['away_team']} @ {event['home_team']}"
        print(f"  [{i}/{len(events)}] {game_desc}...", end=" ")
        
        odds_result = get_historical_event_odds(sport, event['id'], date_str, markets, snapshot_hour)
        
        if odds_result and odds_result['data']:
            games_with_odds.append(odds_result['data'])
            print(f"✓ (Remaining: {odds_result['remaining']:,})")
        else:
            print("❌ Failed")
    
    if not games_with_odds:
        print(f"  ⚠️  No odds data retrieved")
        return pd.DataFrame()
    
    # Parse and return DataFrame
    markets_list = [m.strip() for m in markets.split(',')]
    df = parse_odds_data(games_with_odds, markets_list)
    
    if df.empty:
        print(f"  ⚠️  No lines parsed")
        return df
    
    return df


def save_odds_data(df, sport, markets, date_str=None, is_historical=False):
    """
    Save odds data to CSV file
    
    Args:
        df: DataFrame with odds data
        sport: Sport key
        markets: Markets string
        date_str: Date string (for historical) or None (for current)
        is_historical: Whether this is historical data
    
    Returns:
        Path to saved file
    """
    if df.empty:
        print("  ⚠️  No data to save")
        return None
    
    # Create filename
    markets_clean = markets.replace(',', '_')
    if is_historical and date_str:
        filename = f"{sport}_{date_str}_{markets_clean}.csv"
    else:
        # For current odds, use timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{sport}_current_{timestamp}_{markets_clean}.csv"
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    print(f"\n{'='*80}")
    print("SAVED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"File: {filepath}")
    print(f"Events: {df['game_id'].nunique()}")
    print(f"Total lines: {len(df)}")
    print(f"Bookmakers: {df['bookmaker'].nunique()}")
    if 'market' in df.columns:
        print(f"Markets: {df['market'].unique().tolist()}")
    if 'last_update' in df.columns and not df.empty:
        latest_update = pd.to_datetime(df['last_update']).max()
        print(f"Latest odds update: {latest_update}")
    print(f"Credits remaining: {credits_remaining:,}")
    
    return filepath


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fetch ad-hoc odds data from The Odds API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (CURRENT ODDS - default, cheap):
  # NFL current spreads and moneyline
  python fetch_adhoc_odds.py -s americanfootball_nfl -m spreads,h2h
  
  # NBA current player props
  python fetch_adhoc_odds.py -s basketball_nba -m player_points
  
  # NHL current all markets
  python fetch_adhoc_odds.py -s icehockey_nhl -m h2h,spreads,totals

Examples (HISTORICAL ODDS - expensive):
  # NFL spreads and moneyline for Nov 28
  python fetch_adhoc_odds.py -s americanfootball_nfl -m spreads,h2h --historical -d 2025-11-28
  
  # NBA player props for Dec 1
  python fetch_adhoc_odds.py -s basketball_nba -m player_points --historical -d 2025-12-01
  
  # Custom snapshot time (6 AM ET = 11 UTC) for London NFL game
  python fetch_adhoc_odds.py -s americanfootball_nfl -m spreads,h2h --historical -d 2025-10-05 --hour 11

Common Sports:
  americanfootball_nfl, basketball_nba, icehockey_nhl, baseball_mlb,
  soccer_epl, soccer_uefa_champs_league, mma_mixed_martial_arts

Common Markets:
  h2h (moneyline), spreads, totals, player_points, player_rebounds,
  player_assists, player_threes, outrights
        """
    )
    
    parser.add_argument('-s', '--sport', required=True,
                       help='Sport key (e.g., americanfootball_nfl, basketball_nba)')
    parser.add_argument('-m', '--markets', required=True,
                       help='Comma-separated market keys (e.g., spreads,h2h)')
    parser.add_argument('--historical', action='store_true',
                       help='Use historical API (expensive, requires --date)')
    parser.add_argument('-d', '--date', default=None,
                       help='Date in YYYY-MM-DD format (required for --historical)')
    parser.add_argument('--hour', type=int, default=None,
                       help='Snapshot hour in UTC for historical (default: 17 = 12pm ET)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.historical and not args.date:
        print("❌ ERROR: --date is required when using --historical mode")
        print("\nExamples:")
        print("  Historical: python fetch_adhoc_odds.py -s americanfootball_nfl -m spreads,h2h --historical -d 2025-11-28")
        print("  Current:    python fetch_adhoc_odds.py -s americanfootball_nfl -m spreads,h2h")
        exit(1)
    
    if args.date:
        # Validate date format
        try:
            datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print(f"❌ Invalid date format: {args.date}")
            print("Please use YYYY-MM-DD format (e.g., 2025-11-28)")
            exit(1)
    
    # Check API key
    if not check_api_key():
        exit(1)
    
    print("="*80)
    print("AD-HOC ODDS FETCHER")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Fetch odds based on mode
    if args.historical:
        print(f"\n🕐 Mode: HISTORICAL (snapshot at specific time)")
        df = fetch_historical_odds(args.sport, args.date, args.markets, args.hour)
        
        # Save data
        if not df.empty:
            save_odds_data(df, args.sport, args.markets, date_str=args.date, is_historical=True)
            print("\n✅ Complete!")
        else:
            print("\n⚠️  No data retrieved")
    else:
        print(f"\n⚡ Mode: CURRENT/LIVE (real-time odds)")
        df = fetch_current_odds(args.sport, args.markets)
        
        # Save data
        if not df.empty:
            save_odds_data(df, args.sport, args.markets, is_historical=False)
            print("\n✅ Complete!")
        else:
            print("\n⚠️  No data retrieved")

