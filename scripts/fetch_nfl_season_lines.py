"""
Fetch NFL Game Lines for 2024-25 Season

Fetches historical betting lines (h2h + spreads) for each game day.
Captures CLOSING lines by fetching on game day, not early opening lines.

Similar approach to fetch_historical_props.py for NBA

Usage:
    python fetch_nfl_season_lines.py               # Interactive mode
    python fetch_nfl_season_lines.py --london      # Fetch only London games
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import ssl
import urllib3
import time
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse

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
MARKETS = 'spreads'  # Only spreads (not moneyline aka h2h)
REGIONS = 'us'
ODDS_FORMAT = 'american'

# Season dates
SEASON_START = '2025-09-04'  # Thursday, Sept 4, 2025
TODAY = datetime.now().strftime('%Y-%m-%d')

# Output directory - save each date as separate CSV
OUTPUT_DIR = 'data/01_input/the-odds-api/nfl/game_lines/historical'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Snapshot times (UTC)
EVENT_LIST_HOUR = 17  # 12pm ET (5 PM UTC)
ODDS_SNAPSHOT_HOUR = 17  # 12pm ET - capture closing lines before games start

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


def get_historical_nfl_events(date_str):
    """
    Get historical NFL events/games for a specific date
    
    NOTE: This costs 1 credit per request
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD'
    
    Returns:
        Dict with events list, cost, remaining credits
    """
    global credits_remaining, credits_used
    
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    timestamp = date_obj.replace(hour=EVENT_LIST_HOUR, minute=0, second=0).isoformat() + 'Z'
    
    url = f"{BASE_URL}/historical/sports/{SPORT}/events"
    
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
            # No data for this date (common for non-game days)
            return {'events': [], 'cost': 0, 'remaining': credits_remaining}
        print(f"❌ HTTP Error for {date_str}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error for {date_str}: {e}")
        return None


def get_historical_event_odds(event_id, date_str, snapshot_hour=None):
    """
    Get historical odds for a specific event
    
    NOTE: This costs 10 credits per event!
    
    Args:
        event_id: Event ID from get_historical_nfl_events
        date_str: Date string for timestamp
        snapshot_hour: Optional hour (UTC) for snapshot. Defaults to ODDS_SNAPSHOT_HOUR
    
    Returns:
        Dict with data, cost, remaining
    """
    global credits_remaining, credits_used
    
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    hour_to_use = snapshot_hour if snapshot_hour is not None else ODDS_SNAPSHOT_HOUR
    timestamp = date_obj.replace(hour=hour_to_use, minute=0, second=0).isoformat() + 'Z'
    
    url = f"{BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds"
    
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
                
                if market_key == 'spreads':
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
                        'away_spread': away_spread,
                        'away_odds': away_spread_odds,
                        'home_spread': home_spread,
                        'home_odds': home_spread_odds
                    })
    
    df = pd.DataFrame(lines_list)
    
    if not df.empty:
        # Convert game_time to datetime
        df['game_time'] = pd.to_datetime(df['game_time'])
        # Sort by game time then bookmaker
        df = df.sort_values(['game_time', 'bookmaker', 'market'])
    
    return df


def fetch_date_lines(date_str, save=True):
    """
    Fetch all game lines for a specific date
    
    Args:
        date_str: Date in YYYY-MM-DD format
        save: Save results to file
    
    Returns:
        DataFrame with all lines for that date
    """
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    day_of_week = date_obj.strftime('%A')
    
    print(f"\n{'='*80}")
    print(f"🌐 FETCHING LINES FOR {date_str} ({day_of_week}) - API CALL")
    print(f"{'='*80}")
    
    # Get events for that date
    print(f"  📡 API CALL 1: Checking for events on {date_str}... (1 credit)", end=" ")
    result = get_historical_nfl_events(date_str)
    
    if result is None:
        print(f"❌ API Error")
        return pd.DataFrame()
    
    print(f"✓ (Remaining: {result['remaining']:,})")
    all_events = result['events']
    
    if not all_events:
        print(f"  ℹ️  No games found on this date")
        # Save empty file so we don't check this date again
        if save:
            filename = f"nfl_game_lines_{date_str}.csv"
            filepath = os.path.join(OUTPUT_DIR, filename)
            pd.DataFrame(columns=['game_id', 'game_time', 'away_team', 'home_team', 'bookmaker', 
                                 'bookmaker_key', 'last_update', 'market', 'away_spread', 
                                 'away_odds', 'home_spread', 'home_odds']).to_csv(filepath, index=False)
            print(f"  💾 Saved empty file → next run will SKIP this date (0 credits)")
        return pd.DataFrame()
    
    # Filter to only games that START on this specific date (in ET timezone)
    # NFL games typically Thu night, Sun 1pm/4pm/8pm, Mon night
    et_tz = ZoneInfo('America/New_York')
    
    # Define window in ET timezone: 6 AM to 11:59 PM ET on the target date
    start_of_window_et = datetime(date_obj.year, date_obj.month, date_obj.day, 6, 0, 0, tzinfo=et_tz)
    end_of_window_et = datetime(date_obj.year, date_obj.month, date_obj.day, 23, 59, 59, tzinfo=et_tz)
    
    events = []
    for event in all_events:
        # Convert UTC time to ET
        commence_time_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
        commence_time_et = commence_time_utc.astimezone(et_tz)
        
        # Check if game starts within the ET window for this date
        if start_of_window_et <= commence_time_et <= end_of_window_et:
            events.append(event)
    
    if not events:
        filtered_count = len(all_events)
        print(f"  ℹ️  Found {filtered_count} events in API, but none start on {date_str} (all future games)")
        # Save empty file so we don't check this date again
        if save:
            filename = f"nfl_game_lines_{date_str}.csv"
            filepath = os.path.join(OUTPUT_DIR, filename)
            pd.DataFrame(columns=['game_id', 'game_time', 'away_team', 'home_team', 'bookmaker', 
                                 'bookmaker_key', 'last_update', 'market', 'away_spread', 
                                 'away_odds', 'home_spread', 'home_odds']).to_csv(filepath, index=False)
            print(f"  💾 Saved empty file → next run will SKIP this date (0 credits)")
        return pd.DataFrame()
    
    print(f"  ✓ Found {len(events)} games starting on {date_str}:")
    for event in events:
        commence_time_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
        commence_time_et = commence_time_utc.astimezone(et_tz)
        time_et = commence_time_et.strftime('%I:%M %p ET')
        print(f"    • {event['away_team']} @ {event['home_team']} ({time_et})")
    
    estimated_cost = 1 + (len(events) * 10)
    print(f"  💰 Total cost for this date: 1 (events) + {len(events)}×10 (odds) = {estimated_cost} credits")
    
    # Fetch odds for each event
    games_with_odds = []
    
    for i, event in enumerate(events, 1):
        game_desc = f"{event['away_team']} @ {event['home_team']}"
        print(f"  📡 API CALL {i+1}: [{i}/{len(events)}] {game_desc}... (10 credits)", end=" ")
        
        odds_result = get_historical_event_odds(event['id'], date_str)
        
        if odds_result and odds_result['data']:
            games_with_odds.append(odds_result['data'])
            print(f"✓ (Remaining: {odds_result['remaining']:,})")
        else:
            print("❌ Failed")
    
    if not games_with_odds:
        print(f"  ⚠️  No odds data retrieved for {date_str}")
        return pd.DataFrame()
    
    # Parse and return DataFrame
    df = parse_game_lines(games_with_odds)
    
    if df.empty:
        print(f"  ⚠️  No lines parsed for {date_str}")
        return df
    
    if save:
        filename = f"nfl_game_lines_{date_str}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath, index=False)
        
        num_games = df['game_id'].nunique()
        print(f"\n  💾 Saved {num_games} games to {filename}")
        print(f"     Total lines: {len(df)} (spread lines only)")
        print(f"     Bookmakers: {df['bookmaker'].nunique()}")
    
    return df


def fetch_full_season():
    """
    Fetch lines for all dates from season start to today
    
    Returns:
        Summary dict with statistics
    """
    print("="*80)
    print("NFL SEASON BETTING LINES FETCH")
    print("="*80)
    print(f"Season: 2024-25")
    print(f"Date range: {SEASON_START} to {TODAY}")
    
    # Generate all dates in range
    start_date = datetime.strptime(SEASON_START, '%Y-%m-%d').date()
    end_date = datetime.strptime(TODAY, '%Y-%m-%d').date()
    
    all_dates = []
    current = start_date
    while current <= end_date:
        all_dates.append(current)
        current += timedelta(days=1)
    
    total_dates = len(all_dates)
    
    print(f"Total days to check: {total_dates}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Check API key
    if not check_api_key():
        return None
    
    # Track statistics
    stats = {
        'total_dates': total_dates,
        'processed': 0,
        'skipped_no_games': 0,
        'skipped_existing': 0,
        'successful': 0,
        'failed': 0,
        'total_games_collected': 0,
        'failed_dates': [],
        'credits_start': credits_used,
        'credits_start_remaining': None
    }
    
    print(f"\n{'='*80}")
    print("STARTING FETCH")
    print(f"{'='*80}")
    print(f"📋 Strategy: Check filesystem first, only call API if file missing")
    print(f"   • File exists → Skip (0 credits)")
    print(f"   • File missing → API call (1 + 10*games credits)")
    print(f"{'='*80}")
    
    first_fetch = True
    
    for i, date_obj in enumerate(all_dates, 1):
        date_str = date_obj.strftime('%Y-%m-%d')
        day_of_week = date_obj.strftime('%A')
        
        print(f"\n[{i}/{total_dates}] {date_str} ({day_of_week})", end=" ")
        print(f"| Progress: {(i/total_dates)*100:.1f}% | ✓ {stats['successful']} | ⏭ {stats['skipped_existing']} | ∅ {stats['skipped_no_games']}")
        
        # ===== CHECK FILE SYSTEM FIRST (NO API CALL) =====
        filename = f"nfl_game_lines_{date_str}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        print(f"  🔍 Checking if {filename} exists...", end=" ")
        
        if os.path.exists(filepath):
            try:
                existing_df = pd.read_csv(filepath)
                num_games = existing_df['game_id'].nunique() if 'game_id' in existing_df.columns else 0
                if num_games > 0:
                    print(f"✓ Found ({num_games} games) - SKIPPING API CALL")
                else:
                    print(f"✓ Found (empty file - no games this date) - SKIPPING API CALL")
            except:
                print(f"✓ Found - SKIPPING API CALL")
            
            stats['skipped_existing'] += 1
            continue
        
        # File doesn't exist - need to fetch from API
        print(f"✗ Not found - WILL CALL API")
        
        # Track credits before fetch
        credits_before = credits_remaining
        
        # Fetch lines for this date
        try:
            df = fetch_date_lines(date_str, save=True)
            
            # Log starting credits after first API call
            if first_fetch and credits_remaining is not None:
                stats['credits_start_remaining'] = credits_remaining
                print(f"\n{'='*80}")
                print(f"💰 STARTING CREDITS: {credits_remaining:,}")
                print(f"{'='*80}")
                first_fetch = False
            
            if df.empty:
                stats['skipped_no_games'] += 1
            else:
                num_games = df['game_id'].nunique()
                stats['successful'] += 1
                stats['total_games_collected'] += num_games
                
                # Calculate credits used
                if credits_before and credits_remaining:
                    credits_used_for_date = credits_before - credits_remaining
                    print(f"  💰 Credits used: {credits_used_for_date}")
            
            stats['processed'] += 1
            
        except KeyboardInterrupt:
            print("\n\n⚠️  User interrupted (Ctrl+C)")
            print("Stopping fetch process...")
            break
            
        except Exception as e:
            print(f"\n❌ Error processing {date_str}: {e}")
            stats['failed'] += 1
            stats['failed_dates'].append(date_str)
            
            # Continue on error (unlike NBA script which stops)
            # NFL data is more sparse, so we want to keep going
            continue
    
    # Show final summary
    print(f"\n{'='*80}")
    print("FETCH SUMMARY")
    print(f"{'='*80}")
    print(f"Date range: {SEASON_START} to {TODAY}")
    print(f"")
    print(f"Total days checked: {stats['total_dates']}")
    print(f"Processed this session: {stats['processed']}")
    print(f"  ✅ Successful: {stats['successful']}")
    print(f"  ⏭️  Skipped (already exist): {stats['skipped_existing']}")
    print(f"  ∅  No games: {stats['skipped_no_games']}")
    print(f"  ❌ Failed: {stats['failed']}")
    print(f"")
    print(f"Total games collected: {stats['total_games_collected']}")
    
    if credits_remaining is not None:
        print(f"")
        print("💰 CREDITS SUMMARY")
        if stats['credits_start_remaining']:
            actual_spent = stats['credits_start_remaining'] - credits_remaining
            print(f"   Started with: {stats['credits_start_remaining']:,} credits")
            print(f"   Remaining now: {credits_remaining:,} credits")
            print(f"   Used: {actual_spent:,} credits")
        else:
            print(f"   Remaining: {credits_remaining:,} credits")
        
        print(f"   Capacity left: ~{credits_remaining // 10} more games")
    
    if stats['failed_dates']:
        print(f"")
        print(f"❌ Failed dates: {', '.join(stats['failed_dates'])}")
    
    print(f"{'='*80}")
    
    return stats


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def fetch_london_games():
    """
    Fetch all London games (games before 1pm ET) from the season
    Saves to a single CSV file: 2025_game_lines_london.csv
    
    Uses 6 AM ET (10 AM UTC) snapshot to get closing lines before 9:30 AM ET kickoff
    """
    print("="*80)
    print("LONDON GAMES MODE - Fetching games before 1pm ET")
    print("="*80)
    
    # Hardcoded London game dates for 2025 season
    # Oct 5, 2025:  Minnesota Vikings @ Cleveland Browns (2:30 PM BST / 9:30 AM ET) - Tottenham Hotspur Stadium
    # Oct 12, 2025: Denver Broncos @ New York Jets (2:30 PM BST / 9:30 AM ET) - Tottenham Hotspur Stadium
    # Oct 19, 2025: Los Angeles Rams @ Jacksonville Jaguars (2:30 PM BST / 9:30 AM ET) - Wembley Stadium
    LONDON_DATES = ['2025-10-05', '2025-10-12', '2025-10-19']
    
    # Expected matchups for validation
    EXPECTED_MATCHUPS = {
        '2025-10-05': ('Minnesota Vikings', 'Cleveland Browns'),
        '2025-10-12': ('Denver Broncos', 'New York Jets'),
        '2025-10-19': ('Los Angeles Rams', 'Jacksonville Jaguars')
    }
    
    print(f"London game dates: {', '.join(LONDON_DATES)}")
    print(f"Using 6 AM ET (10 AM UTC) snapshot for closing lines\n")
    
    all_london_games = []
    et_tz = ZoneInfo('America/New_York')
    LONDON_SNAPSHOT_HOUR = 10  # 10 AM UTC = 6 AM ET (before 9:30 AM ET kickoff) [trying this bc we couldn't find the first 2 games]
    
    for date_str in LONDON_DATES:
        print(f"\n{'='*80}")
        print(f"CHECKING {date_str}")
        print(f"{'='*80}")
        
        expected = EXPECTED_MATCHUPS.get(date_str)
        if expected:
            print(f"Expected: {expected[0]} @ {expected[1]}")
        
        # Get events for this date using early snapshot
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        timestamp = date_obj.replace(hour=LONDON_SNAPSHOT_HOUR, minute=0, second=0).isoformat() + 'Z'
        
        url = f"{BASE_URL}/historical/sports/{SPORT}/events"
        params = {
            'apiKey': API_KEY,
            'date': timestamp,
            'dateFormat': 'iso'
        }
        
        try:
            response = requests.get(url, params=params, verify=False)
            response.raise_for_status()
            data = response.json()
            all_events = data.get('data', [])
            
            if not all_events:
                print(f"  ⚠️  No events found at {timestamp}")
                continue
            
            # Find the expected London game
            london_game = None
            for event in all_events:
                if expected:
                    away = event.get('away_team', '')
                    home = event.get('home_team', '')
                    if away == expected[0] and home == expected[1]:
                        london_game = event
                        break
            
            if not london_game:
                print(f"  ⚠️  Expected game not found in events list")
                print(f"  Found {len(all_events)} total events:")
                for event in all_events[:5]:  # Show first 5
                    print(f"    • {event.get('away_team')} @ {event.get('home_team')}")
                continue
            
            # Fetch odds for the London game using early snapshot
            commence_time_utc = datetime.fromisoformat(london_game['commence_time'].replace('Z', '+00:00'))
            commence_time_et = commence_time_utc.astimezone(et_tz)
            time_et = commence_time_et.strftime('%I:%M %p ET')
            
            print(f"  ✓ Found: {london_game['away_team']} @ {london_game['home_team']} ({time_et})")
            print(f"    Fetching odds at 6 AM ET snapshot...", end=" ")
            
            odds_result = get_historical_event_odds(london_game['id'], date_str, snapshot_hour=LONDON_SNAPSHOT_HOUR)
            
            if odds_result and odds_result['data']:
                all_london_games.append(odds_result['data'])
                print(f"✓ (Remaining: {odds_result['remaining']:,})")
            else:
                print(f"❌ Failed")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Parse and save all London games to single CSV
    if all_london_games:
        print(f"\n{'='*80}")
        print(f"PARSING LONDON GAMES DATA")
        print(f"{'='*80}")
        
        df = parse_game_lines(all_london_games)
        
        output_file = os.path.join(OUTPUT_DIR, '2025_game_lines_london.csv')
        df.to_csv(output_file, index=False)
        
        print(f"\n✅ SUCCESS!")
        print(f"   Total London games: {df['game_id'].nunique()}")
        print(f"   Total lines: {len(df)}")
        print(f"   Saved to: {output_file}")
        print(f"   Credits remaining: {credits_remaining:,}")
    else:
        print(f"\n⚠️  No London games found")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fetch NFL betting lines')
    parser.add_argument('--london', action='store_true', 
                       help='Fetch only London games (games before 1pm ET)')
    parser.add_argument('--prod-run', action='store_true',
                       help='Production mode: fetch full season non-interactively (for automation)')
    args = parser.parse_args()
    
    print("="*80)
    print("NFL HISTORICAL GAME LINES FETCHER")
    print("="*80)
    print(f"Season: 2024-25")
    print(f"Markets: {MARKETS}")
    print(f"Output: {OUTPUT_DIR}")
    
    if not check_api_key():
        exit(1)
    
    # London mode - fetch all London games
    if args.london:
        fetch_london_games()
        exit(0)
    
    # Production mode - fetch full season non-interactively
    if args.prod_run:
        print(f"\n🏈 PRODUCTION MODE")
        print(f"Fetching all dates from {SEASON_START} to {TODAY}")
        print(f"Skipping dates with existing files")
        
        stats = fetch_full_season()
        
        if stats and stats['successful'] > 0:
            print(f"\n✅ Fetch completed - {stats['successful']} dates with games")
        elif stats and stats['skipped_existing'] > 0:
            print(f"\n✅ All dates already fetched!")
        else:
            print(f"\n⚠️  No new data fetched")
        
        exit(0)
    
    # Interactive mode for regular fetching
    # Ask user for mode
    print("\n" + "="*80)
    print("MODE SELECTION")
    print("="*80)
    print("1. Test mode (fetch one date only - Nov 24, 2025)")
    print("2. Full season (fetch all dates from Sept 4 to today)")
    print("="*80)
    
    while True:
        choice = input("\nSelect mode (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    if choice == '1':
        # Test mode - fetch one date
        default_date = '2025-11-24'  # Sunday with games
        print(f"\n🧪 TEST MODE")
        date_input = input(f"Enter date to fetch (YYYY-MM-DD) or press Enter for default [{default_date}]: ").strip()
        
        test_date = date_input if date_input else default_date
        
        # Validate date format
        try:
            datetime.strptime(test_date, '%Y-%m-%d')
        except ValueError:
            print(f"❌ Invalid date format: {test_date}")
            print("Please use YYYY-MM-DD format (e.g., 2025-11-24)")
            exit(1)
        
        print(f"\nFetching {test_date}...")
        print("This will use approximately 1 + (num_games × 10) credits")
        
        df = fetch_date_lines(test_date, save=True)
        
        if not df.empty:
            print(f"\n✅ Test successful!")
            print(f"   Games fetched: {df['game_id'].nunique()}")
            print(f"   Total lines: {len(df)}")
            print(f"   Credits remaining: {credits_remaining:,}")
            print(f"\nRun again and select mode 2 to fetch the full season.")
        else:
            print(f"\n⚠️  No games found on {test_date}")
    
    else:
        # Full season mode
        print(f"\n🏈 FULL SEASON MODE")
        print(f"This will check all dates from {SEASON_START} to {TODAY}")
        print(f"It will skip dates that already have files.")
        
        stats = fetch_full_season()
        
        if stats and stats['successful'] > 0:
            print(f"\n✅ Fetch completed - {stats['successful']} dates with games")
        elif stats and stats['skipped_existing'] > 0:
            print(f"\n✅ All dates already fetched!")
        else:
            print(f"\n⚠️  No new data collected")
