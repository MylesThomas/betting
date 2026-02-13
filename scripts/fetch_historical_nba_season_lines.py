"""
Fetch NBA Game Lines (Moneyline + Spreads) for Historical Seasons

Fetches historical betting lines (h2h/moneyline + spreads) for each game day
from the 2020-21 season through current 2025-26 season.

Captures CLOSING lines by fetching on game day, not early opening lines.

S3 Storage Structure:
    s3://the-odds-api-mt/nba/historical/
        2020-21/
            nba_game_lines_2020-12-22.csv
            nba_game_lines_2020-12-23.csv
            ...
        2021-22/
            nba_game_lines_2021-10-19.csv
            ...
        2022-23/
        2023-24/
        2024-25/
        2025-26/

Context:
- The Odds API has NBA data starting from 2020-21 season
- Includes both regular season AND playoff games
- Similar to fetch_historical_nfl_season_lines.py but for NBA
- Files organized by season in S3 for easier analysis

Usage:
    # Current season (2025-26, default)
    python scripts/fetch_historical_nba_season_lines.py --season 2025-26 --prod-run
    
    # Historical season
    python scripts/fetch_historical_nba_season_lines.py --season 2024-25 --prod-run
    python scripts/fetch_historical_nba_season_lines.py --season 2023-24 --prod-run
    python scripts/fetch_historical_nba_season_lines.py --season 2022-23 --prod-run
    python scripts/fetch_historical_nba_season_lines.py --season 2021-22 --prod-run
    python scripts/fetch_historical_nba_season_lines.py --season 2020-21 --prod-run
    
    # Interactive mode (shows cost estimate first)
    python scripts/fetch_historical_nba_season_lines.py --season 2024-25
    
    # Test single date
    python scripts/fetch_historical_nba_season_lines.py --test-date 2024-10-22
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import ssl
import urllib3
import time
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import boto3
from io import StringIO
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from season_utils import get_current_nba_season

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
SPORT = 'basketball_nba'
MARKETS = 'h2h,spreads'  # Both moneyline and spreads
REGIONS = 'us'
ODDS_FORMAT = 'american'

# S3 Configuration
S3_BUCKET = 'the-odds-api-mt'
S3_PREFIX = 'nba/historical_game_lines'  # S3 path: s3://the-odds-api-mt/nba/historical_game_lines/YYYY-YY/

# Local backup directory (optional)
OUTPUT_DIR = 'data/01_input/the-odds-api/nba/game_lines/historical_game_lines'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# S3 client (lazy initialization)
_s3_client = None

# Snapshot times (UTC)
EVENT_LIST_HOUR = 17  # 12pm ET (5 PM UTC) - for listing games
# ODDS_SNAPSHOT_HOUR = 17  # 12pm ET (5 PM UTC) - safe time before most games start (7-10pm ET)
# (ODDS_SNAPSHOT_HOUR is not used, as we fetch lines at the last full hour before tipoff)

RATE_LIMIT_DELAY = 0.5  # seconds between API calls

# API usage tracking
credits_remaining = None
credits_used = None

# Season date ranges
# NBA seasons run Oct-June (regular season ends ~April, playoffs through June)
SEASON_DATES = {
    '2020-21': ('2020-12-22', '2021-07-21'),  # COVID delayed season
    '2021-22': ('2021-10-19', '2022-06-17'),  # Full season + playoffs
    '2022-23': ('2022-10-18', '2023-06-13'),  # Full season + playoffs
    '2023-24': ('2023-10-24', '2024-06-18'),  # Full season + playoffs
    '2024-25': ('2024-10-22', '2025-06-23'),  # Full season + playoffs (estimated)
    '2025-26': ('2025-10-21', '2026-06-22'),  # Current season (estimated)
}


# =============================================================================
# S3 HELPER FUNCTIONS
# =============================================================================

def get_s3_client():
    """Get or create S3 client (lazy initialization)"""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3')
    return _s3_client


def get_season_from_date(date_str: str) -> str:
    """
    Determine NBA season from a date.
    
    NBA seasons run Oct-June (e.g., 2024-25 season = Oct 2024 - June 2025)
    
    Args:
        date_str: Date in YYYY-MM-DD format
    
    Returns:
        Season string like '2024-25'
    """
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    year = date_obj.year
    month = date_obj.month
    
    # If month >= 10 (Oct, Nov, Dec), it's the start of the season
    # Season year is current year (e.g., Oct 2024 = 2024-25 season)
    if month >= 10:
        season_start_year = year
    else:
        # If month < 10 (Jan-Sep), it's the end of the season
        # Season year is previous year (e.g., April 2025 = 2024-25 season)
        season_start_year = year - 1
    
    season_end_year = (season_start_year + 1) % 100  # Get last 2 digits
    return f"{season_start_year}-{season_end_year:02d}"


def get_s3_key_from_date(date_str: str) -> str:
    """
    Generate S3 key for a given date.
    
    Args:
        date_str: Date in YYYY-MM-DD format
    
    Returns:
        S3 key like: nba/historical/2024-25/nba_game_lines_2024-10-22.csv
    """
    season = get_season_from_date(date_str)
    filename = f"nba_game_lines_{date_str}.csv"
    return f"{S3_PREFIX}/{season}/{filename}"


def file_exists_in_s3(date_str: str) -> bool:
    """
    Check if file exists in S3 for a given date.
    
    Args:
        date_str: Date in YYYY-MM-DD format
    
    Returns:
        True if file exists in S3, False otherwise
    """
    s3_key = get_s3_key_from_date(date_str)
    try:
        s3_client = get_s3_client()
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return True
    except:
        return False


def save_df_to_s3(df: pd.DataFrame, date_str: str) -> bool:
    """
    Save DataFrame to S3 as CSV.
    
    Args:
        df: DataFrame to save
        date_str: Date in YYYY-MM-DD format
    
    Returns:
        True if successful, False otherwise
    """
    if df.empty:
        # For empty DataFrames, still save an empty file
        pass
    
    s3_key = get_s3_key_from_date(date_str)
    
    try:
        # Convert DataFrame to CSV in memory
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        # Upload to S3
        s3_client = get_s3_client()
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        return True
    except Exception as e:
        print(f"  ❌ S3 upload failed: {e}")
        return False


def save_df_locally(df: pd.DataFrame, date_str: str) -> str:
    """
    Save DataFrame to local file (backup).
    
    Args:
        df: DataFrame to save
        date_str: Date in YYYY-MM-DD format
    
    Returns:
        Path to saved file
    """
    filename = f"nba_game_lines_{date_str}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    return filepath


# =============================================================================
# API HELPER FUNCTIONS
# =============================================================================

def check_api_key():
    """Verify API key is loaded"""
    if not API_KEY or API_KEY == 'your_api_key_here':
        print("❌ ERROR: No valid API key found!")
        print("Make sure ODDS_API_KEY is set in your .env file")
        return False
    return True


def get_historical_nba_events(date_str):
    """
    Get historical NBA events/games for a specific date
    
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
        credits_remaining = int(float(response.headers.get('x-requests-remaining', 0)))
        credits_used = int(float(response.headers.get('x-requests-used', 0)))
        cost = int(float(response.headers.get('x-requests-last', 0)))
        
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


def get_historical_event_odds(event_id, game_commence_time_et):
    """
    Get historical odds for a specific event at the last full hour before tipoff.
    
    Works in Eastern Time for clarity, converts to UTC for API.
    Fetches odds at the last round hour before the game starts.
    
    Examples (all in ET):
        - Game at 8:10 PM ET → Fetch odds at 8:00 PM ET (10 min before)
        - Game at 12:10 PM ET → Fetch odds at 12:00 PM ET (10 min before)
        - Game at 7:00 PM ET → Fetch odds at 6:00 PM ET (1 hour before)
    
    NOTE: This costs 10 credits per event per market
    Since we fetch h2h + spreads, that's 2 markets = 20 credits per event
    
    Args:
        event_id: Event ID from get_historical_nba_events
        game_commence_time_et: Game tipoff time in ET (pandas Timestamp with ET timezone)
    
    Returns:
        Dict with data, cost, remaining
    """
    global credits_remaining, credits_used
    
    # Calculate odds time: last full hour before tipoff (in ET)
    odds_time_et = game_commence_time_et.replace(minute=0, second=0, microsecond=0)
    
    # If game is exactly on the hour, go back 1 hour
    if odds_time_et == game_commence_time_et:
        odds_time_et = odds_time_et - pd.Timedelta(hours=1)
    
    # Convert to UTC for API call
    odds_time_utc = odds_time_et.tz_convert('UTC')
    # Format as ISO with 'Z' suffix (API requires this format, not +00:00)
    timestamp = odds_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    
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
        credits_remaining = int(float(response.headers.get('x-requests-remaining', 0)))
        credits_used = int(float(response.headers.get('x-requests-used', 0)))
        cost = int(float(response.headers.get('x-requests-last', 0)))
        
        event_data = data.get('data', {})
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
        
        return {
            'data': event_data,
            'cost': cost,
            'remaining': credits_remaining
        }
        
    except requests.exceptions.HTTPError as e:
        # Get response text for debugging errors
        try:
            error_detail = e.response.json()
            error_msg = error_detail.get('message', str(e))
            error_code = error_detail.get('error_code', '')
        except:
            error_msg = e.response.text[:100]
            error_code = ''
        
        # 404 means no odds data for this event (common for some games)
        # Don't print full error, just note it
        if e.response.status_code == 404:
            print(f"\n  ⚠️  No odds data for event {event_id[:8]} (404)")
        else:
            print(f"\n  ❌ HTTP {e.response.status_code}: {error_msg}")
        
        return None
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        return None


def parse_game_lines(games, odds_fetch_times=None):
    """
    Parse game betting lines into a clean DataFrame
    
    Handles both h2h (moneyline) and spreads markets
    
    Args:
        games: List of game data from API
        odds_fetch_times: Dict mapping game_id to the timestamp when odds were fetched
    
    Returns:
        DataFrame with betting lines
    """
    lines_list = []
    
    for game in games:
        game_id = game.get('id')
        game_time = game.get('commence_time')
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        # Get the odds fetch time for this game
        odds_pull_time = None
        if odds_fetch_times and game_id in odds_fetch_times:
            odds_pull_time = odds_fetch_times[game_id]
        
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
                        'odds_pull_time': odds_pull_time,
                        'away_team': away_team,
                        'home_team': home_team,
                        'bookmaker': bookmaker_title,
                        'bookmaker_key': bookmaker_key,
                        'last_update': last_update,
                        'market': 'spread',
                        'away_line': away_spread,
                        'away_odds': away_spread_odds,
                        'home_line': home_spread,
                        'home_odds': home_spread_odds
                    })
                
                elif market_key == 'h2h':
                    # Moneyline odds
                    home_ml_odds = outcome_dict.get(home_team, {}).get('price')
                    away_ml_odds = outcome_dict.get(away_team, {}).get('price')
                    
                    lines_list.append({
                        'game_id': game_id,
                        'game_time': game_time,
                        'odds_pull_time': odds_pull_time,
                        'away_team': away_team,
                        'home_team': home_team,
                        'bookmaker': bookmaker_title,
                        'bookmaker_key': bookmaker_key,
                        'last_update': last_update,
                        'market': 'moneyline',
                        'away_line': None,  # No line for moneyline
                        'away_odds': away_ml_odds,
                        'home_line': None,
                        'home_odds': home_ml_odds
                    })
    
    df = pd.DataFrame(lines_list)
    
    if not df.empty:
        # Convert game_time to datetime
        df['game_time'] = pd.to_datetime(df['game_time'])
        # Convert odds_pull_time to datetime if present
        if 'odds_pull_time' in df.columns:
            df['odds_pull_time'] = pd.to_datetime(df['odds_pull_time'])
        # Sort by game time then bookmaker then market
        df = df.sort_values(['game_time', 'bookmaker', 'market'])
    
    return df


def fetch_date_lines(date_str, save=True, local_backup=True, force=False):
    """
    Fetch all game lines for a specific date
    
    Args:
        date_str: Date in YYYY-MM-DD format
        save: Save results to S3
        local_backup: Also save to local disk
        force: Force overwrite existing files (skip existence check)
    
    Returns:
        DataFrame with all lines for that date
    """
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    day_of_week = date_obj.strftime('%A')
    
    print(f"\n{'='*80}")
    print(f"🏀 FETCHING GAME LINES FOR {date_str} ({day_of_week})")
    print(f"{'='*80}")
    
    # Check if file already exists in S3 (skip if force=True)
    if not force and file_exists_in_s3(date_str):
        s3_key = get_s3_key_from_date(date_str)
        print(f"  ✅ File already exists in S3: s3://{S3_BUCKET}/{s3_key}")
        print(f"     Skipping (0 credits used)")
        # Try to load from S3 to return
        try:
            s3_client = get_s3_client()
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            existing_df = pd.read_csv(obj['Body'])
            if not existing_df.empty:
                num_games = existing_df['game_id'].nunique()
                print(f"     {num_games} games, {len(existing_df)} lines")
            return existing_df
        except:
            print(f"     (Could not read from S3, will re-fetch)")
            pass
    
    # Get events for that date
    print(f"  📡 API CALL 1: Checking for events on {date_str}... (1 credit)", end=" ")
    result = get_historical_nba_events(date_str)
    
    if result is None:
        print(f"❌ API Error")
        return pd.DataFrame()
    
    print(f"✓ (Remaining: {result['remaining']:,})")
    all_events = result['events']
    
    if not all_events:
        print(f"  ℹ️  No games found on this date")
        # Save empty file so we don't check this date again
        if save:
            empty_df = pd.DataFrame(columns=['game_id', 'game_time', 'away_team', 'home_team', 
                                            'bookmaker', 'bookmaker_key', 'last_update', 'market',
                                            'away_line', 'away_odds', 'home_line', 'home_odds'])
            save_df_to_s3(empty_df, date_str)
            if local_backup:
                save_df_locally(empty_df, date_str)
            print(f"  💾 Saved empty file to S3 → next run will SKIP this date (0 credits)")
        return pd.DataFrame()
    
    # Filter to only games on this specific date (in ET timezone)
    et_tz = ZoneInfo('America/New_York')
    
    # Define window in ET timezone: 6 AM to 11:59 PM ET on the target date
    start_of_window_et = datetime(date_obj.year, date_obj.month, date_obj.day, 6, 0, 0, tzinfo=et_tz)
    end_of_window_et = datetime(date_obj.year, date_obj.month, date_obj.day, 23, 59, 59, tzinfo=et_tz)
    
    events = []
    for event in all_events:
        commence_time_str = event.get('commence_time')
        if commence_time_str:
            commence_time = pd.to_datetime(commence_time_str).tz_convert(et_tz)
            if start_of_window_et <= commence_time <= end_of_window_et:
                events.append(event)
    
    if not events:
        print(f"  ℹ️  No games in the {date_str} ET window (found {len(all_events)} but different date)")
        if save:
            empty_df = pd.DataFrame(columns=['game_id', 'game_time', 'away_team', 'home_team', 
                                            'bookmaker', 'bookmaker_key', 'last_update', 'market',
                                            'away_line', 'away_odds', 'home_line', 'home_odds'])
            save_df_to_s3(empty_df, date_str)
            if local_backup:
                save_df_locally(empty_df, date_str)
        return pd.DataFrame()
    
    print(f"\n  🎯 Found {len(events)} games on {date_str}:")
    for event in events:
        away = event.get('away_team', '?')
        home = event.get('home_team', '?')
        commence = event.get('commence_time', '')
        if commence:
            game_time_et = pd.to_datetime(commence).tz_convert(et_tz).strftime('%I:%M %p ET')
        else:
            game_time_et = '?'
        print(f"     - {away} @ {home} ({game_time_et})")
    
    # Cost estimate: 2 markets (h2h + spreads) × 10 credits = 20 credits per game
    estimated_cost = len(events) * 20
    print(f"\n  💰 Estimated cost: {len(events)} games × 20 (h2h+spreads) = {estimated_cost} credits")
    
    # Fetch odds for each event individually (per-game, not per-day)
    # Each game gets odds from the last full hour before its specific tipoff
    games_with_odds = []
    odds_fetch_times = {}  # Track when odds were fetched for each game
    failed_games = []  # Track games that failed to fetch odds
    
    for i, event in enumerate(events, 1):
        game_desc = f"{event['away_team']} @ {event['home_team']}"
        
        # Get game tipoff time (comes from API in UTC)
        commence_time_str = event.get('commence_time')
        if not commence_time_str:
            print(f"  ⚠️  No commence time for {game_desc}, skipping")
            continue
        
        # Convert to ET for all calculations
        game_commence_time_utc = pd.to_datetime(commence_time_str)
        game_commence_time_et = game_commence_time_utc.tz_convert(et_tz)
        tipoff_time_et = game_commence_time_et.strftime('%I:%M %p ET')
        
        # Calculate odds time in ET (last full hour before tipoff)
        odds_time_et = game_commence_time_et.replace(minute=0, second=0, microsecond=0)
        if odds_time_et == game_commence_time_et:
            odds_time_et = odds_time_et - pd.Timedelta(hours=1)
        
        mins_before = int((game_commence_time_et - odds_time_et).total_seconds() / 60)
        odds_time_str = odds_time_et.strftime('%I:%M %p ET')
        
        # Store the odds fetch time for this game (in UTC for consistency)
        odds_fetch_times[event['id']] = odds_time_et.tz_convert('UTC').isoformat()
        
        print(f"  📡 [{i}/{len(events)}] {game_desc} (Tip:{tipoff_time_et}, Odds:{odds_time_str}, {mins_before}m before)... ", end="")
        
        # Fetch odds for this specific game at its calculated time
        odds_result = get_historical_event_odds(event['id'], game_commence_time_et)
        
        if odds_result and odds_result['data']:
            games_with_odds.append(odds_result['data'])
            print(f"✓ (Rem:{odds_result['remaining']:,})")
        else:
            # No odds data for this game (404 or other error)
            # Track the failure
            failed_games.append({
                'game_id': event['id'],
                'game_date': date_str,
                'game_time': commence_time_str,
                'away_team': event['away_team'],
                'home_team': event['home_team'],
                'odds_pull_time_attempted': odds_time_et.tz_convert('UTC').isoformat(),
                'reason': 'No odds data available (likely 404)'
            })
            # Continue to next game - we'll save what we got
            if not odds_result:
                print("⚠️ skip")
            else:
                print("❌")
    
    # Parse successful games
    if games_with_odds:
        df = parse_game_lines(games_with_odds, odds_fetch_times)
    else:
        df = pd.DataFrame()
    
    # Show summary of what we got
    if not df.empty:
        successful_games = df['game_id'].nunique()
    else:
        successful_games = 0
    
    num_failed = len(failed_games)
    
    if num_failed > 0:
        print(f"\n  ℹ️  Retrieved {successful_games}/{len(events)} games ({num_failed} games had no odds data)")
    
    # Save successful games to main file
    if save and not df.empty:
        # Save to S3
        s3_key = get_s3_key_from_date(date_str)
        if save_df_to_s3(df, date_str):
            num_games = df['game_id'].nunique()
            num_spread_lines = len(df[df['market'] == 'spread'])
            num_ml_lines = len(df[df['market'] == 'moneyline'])
            print(f"\n  💾 Saved {num_games} games to S3: s3://{S3_BUCKET}/{s3_key}")
            print(f"     Spread lines: {num_spread_lines}")
            print(f"     Moneyline lines: {num_ml_lines}")
            print(f"     Bookmakers: {df['bookmaker'].nunique()}")
            
            # Optional local backup
            if local_backup:
                local_path = save_df_locally(df, date_str)
                print(f"     Local backup: {local_path}")
    
    # Save failed games log if any failures occurred
    # Append to season-level failures file
    if save and failed_games:
        failed_df = pd.DataFrame(failed_games)
        season_year_str = get_season_from_date(date_str)
        failed_s3_key = f"nba/historical/{season_year_str}/failed_game_pulls.csv"
        
        try:
            # Check if failures file already exists
            existing_df = None
            try:
                response = s3_client.get_object(Bucket=S3_BUCKET, Key=failed_s3_key)
                existing_df = pd.read_csv(response['Body'])
            except Exception as e:
                # File doesn't exist yet or other error, that's fine
                if 'NoSuchKey' not in str(e):
                    print(f"  ⚠️  Note: {e}")
                pass
            
            # Append to existing data or create new
            if existing_df is not None:
                combined_df = pd.concat([existing_df, failed_df], ignore_index=True)
                # Remove duplicates (same game_id might be tried multiple times)
                combined_df = combined_df.drop_duplicates(subset=['game_id'], keep='last')
            else:
                combined_df = failed_df
            
            # Save back to S3
            csv_buffer = StringIO()
            combined_df.to_csv(csv_buffer, index=False)
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=failed_s3_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )
            print(f"  📝 Logged {len(failed_games)} failed games to: s3://{S3_BUCKET}/{failed_s3_key}")
            print(f"     Total failures for season: {len(combined_df)}")
        except Exception as e:
            print(f"  ⚠️  Failed to save failures log: {e}")
    
    return df


def check_past_season_complete(season_start, season_end):
    """
    Check if a past season is already complete in S3.
    
    Args:
        season_start: Start date (YYYY-MM-DD)
        season_end: End date (YYYY-MM-DD)
    
    Returns:
        tuple: (is_complete: bool, total_dates: int, files_found: int)
    """
    s3 = get_s3_client()
    
    # Determine season from dates
    start_year = int(season_start.split('-')[0])
    end_year = int(season_end.split('-')[0])
    if end_year > start_year:
        season = f"{start_year}-{str(end_year)[-2:]}"
    else:
        season = f"{start_year}-{str(start_year + 1)[-2:]}"
    
    # Count expected dates
    start_date = datetime.strptime(season_start, '%Y-%m-%d')
    end_date = datetime.strptime(season_end, '%Y-%m-%d')
    
    # Don't count future dates
    today = datetime.today()
    if end_date > today:
        end_date = today
    
    expected_dates = (end_date - start_date).days + 1
    
    # Count files in S3
    prefix = f"{S3_PREFIX}/{season}/"
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        if 'Contents' not in response:
            return False, expected_dates, 0
        
        csv_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.csv')]
        return len(csv_files) >= expected_dates, expected_dates, len(csv_files)
    except Exception as e:
        print(f"⚠️  Error checking S3: {e}")
        return False, expected_dates, 0


def fetch_full_season(season_start, season_end, dry_run=True, force=False):
    """
    Fetch all game lines for a full season
    
    Args:
        season_start: Start date (YYYY-MM-DD)
        season_end: End date (YYYY-MM-DD)
        dry_run: If True, only estimate costs without fetching
        force: If True, skip "season complete" check and fetch missing dates
    
    Returns:
        Summary dict with stats
    """
    start_date = datetime.strptime(season_start, '%Y-%m-%d')
    end_date = datetime.strptime(season_end, '%Y-%m-%d')
    
    # Determine season
    start_year = int(season_start.split('-')[0])
    end_year = int(season_end.split('-')[0])
    if end_year > start_year:
        season = f"{start_year}-{str(end_year)[-2:]}"
    else:
        season = f"{start_year}-{str(start_year + 1)[-2:]}"
    
    # Check if past season and complete (unless --force is used)
    current_season = get_current_nba_season()
    is_past_season = season < current_season
    
    if is_past_season and not force:
        print(f"\n📅 Checking if past season {season} is already complete...")
        is_complete, expected_dates, files_found = check_past_season_complete(season_start, season_end)
        
        if is_complete:
            print(f"\n{'='*80}")
            print(f"✅ PAST SEASON COMPLETE - SKIPPING")
            print(f"{'='*80}")
            print(f"Season: {season}")
            print(f"Found: {files_found}/{expected_dates} files in S3")
            print(f"S3 Path: s3://{S3_BUCKET}/{S3_PREFIX}/{season}/")
            print(f"\nNo fetch needed - all historical data exists!")
            print(f"\n💡 TIP: Use --force to re-check and fetch any missing dates")
            print(f"{'='*80}\n")
            return {'skipped': True, 'reason': 'Past season complete', 'files_found': files_found}
        else:
            print(f"   Found {files_found}/{expected_dates} files - will fetch missing dates")
    elif is_past_season and force:
        print(f"\n🔄 Force mode enabled - will check all dates for {season} season")
    else:
        print(f"\n🔄 Current season {season} - checking for updates...")
    
    # Don't fetch data for future dates
    today = datetime.today()
    if end_date > today:
        original_end = end_date.strftime('%Y-%m-%d')
        end_date = today
        print(f"\n⚠️  Adjusted end date from {original_end} to {today.strftime('%Y-%m-%d')} (today)")
        print(f"   Cannot fetch historical data for future dates")
    
    # Get all dates in range
    all_dates = []
    current = start_date
    while current <= end_date:
        all_dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    print(f"\n{'='*80}")
    print(f"📅 SEASON DATE RANGE: {season_start} to {end_date.strftime('%Y-%m-%d')}")
    print(f"{'='*80}")
    print(f"Total days in range: {len(all_dates)}")
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - Estimating costs...")
        print("\nEstimated costs (assuming ~82 games per team × 30 teams / 2 = ~1,230 games):")
        print("  - Regular season: ~1,230 games × 20 credits = 24,600 credits")
        print("  - Playoffs: ~90 games × 20 credits = 1,800 credits")
        print("  - Total estimate: ~26,400 credits per season")
        print("\nTo proceed with actual fetching, run with --prod-run flag")
        return
    
    # Production run
    print("\n🚀 PRODUCTION RUN - Fetching data...")
    
    total_games = 0
    total_lines = 0
    dates_with_games = 0
    
    for i, date_str in enumerate(all_dates, 1):
        print(f"\n[{i}/{len(all_dates)}] Processing {date_str}...")
        
        df = fetch_date_lines(date_str, save=True)
        
        if not df.empty:
            dates_with_games += 1
            total_games += df['game_id'].nunique()
            total_lines += len(df)
        
        # Rate limiting
        time.sleep(0.2)
    
    print(f"\n{'='*80}")
    print(f"✅ SEASON FETCH COMPLETE")
    print(f"{'='*80}")
    print(f"Dates processed: {len(all_dates)}")
    print(f"Dates with games: {dates_with_games}")
    print(f"Total games: {total_games}")
    print(f"Total betting lines: {total_lines}")
    if credits_remaining is not None:
        print(f"Credits remaining: {credits_remaining:,}")
    else:
        print(f"Credits remaining: Unknown (all files existed, no API calls made)")



def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Fetch NBA game lines from The Odds API and save to S3')
    parser.add_argument('--season', type=str, 
                       choices=['2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26'],
                       help='Season (e.g., 2024-25 or 2025-26)')
    parser.add_argument('--prod-run', action='store_true',
                       help='Actually fetch data (otherwise just shows cost estimate)')
    parser.add_argument('--test-date', type=str,
                       help='Test single date (YYYY-MM-DD format)')
    parser.add_argument('--no-local-backup', action='store_true',
                       help='Skip local backup (S3 only)')
    parser.add_argument('--force', action='store_true',
                       help='Force fetch even if past season appears complete (useful for adding playoff data)')
    
    args = parser.parse_args()
    
    if not check_api_key():
        return
    
    # Test single date
    if args.test_date:
        print(f"\n🧪 TEST MODE - Single date: {args.test_date}")
        print(f"S3 Bucket: s3://{S3_BUCKET}/{S3_PREFIX}/")
        local_backup = not args.no_local_backup
        df = fetch_date_lines(args.test_date, save=True, local_backup=local_backup)
        if not df.empty:
            print(f"\n✅ Successfully fetched {len(df)} lines for {df['game_id'].nunique()} games")
            s3_key = get_s3_key_from_date(args.test_date)
            print(f"   S3: s3://{S3_BUCKET}/{s3_key}")
        return
    
    # Season fetch
    if not args.season:
        print("Please specify --season (e.g., --season 2024-25)")
        return
    
    if args.season not in SEASON_DATES:
        print(f"Season {args.season} not available. Available: {', '.join(SEASON_DATES.keys())}")
        return
    
    season_start, season_end = SEASON_DATES[args.season]
    
    fetch_full_season(season_start, season_end, dry_run=not args.prod_run, force=args.force)


if __name__ == '__main__':
    main()

