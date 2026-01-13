"""
Fetch NFL Player Props Historical Data (Modular - Any Market)

Purpose:
--------
Modular script to download historical NFL player props for specified markets
from The Odds API and save to S3 in an organized structure.

Context:
--------
User confirmed historical data exists starting 2023-24 NFL season.
This script can fetch any player prop market (rushing, passing, receiving, etc.)
and organize it by season and market type in S3.

S3 Storage Structure:
--------------------
s3://the-odds-api-mt/nfl/historical_player_props/
    2023-24/
        player_rush_yds/
            2023-09-07.csv
            2023-09-10.csv
            ...
        player_rush_attempts/
            2023-09-07.csv
            ...
        player_pass_yds/
            2023-09-07.csv
            ...
    2024-25/
        player_rush_yds/
            2024-09-05.csv
            ...

Usage:
------
# Single market, single season
python scripts/fetch_nfl_player_props_historical.py \
    --season 2023-24 \
    --markets player_rush_yds \
    --prod-run

# Multiple markets, single season  
python scripts/fetch_nfl_player_props_historical.py \
    --season 2024-25 \
    --markets player_rush_yds player_rush_attempts player_pass_yds \
    --prod-run

# Test mode (dry run with cost estimate)
python scripts/fetch_nfl_player_props_historical.py \
    --season 2024-25 \
    --markets player_rush_yds

# Specific date range
python scripts/fetch_nfl_player_props_historical.py \
    --season 2024-25 \
    --markets player_rush_yds \
    --start-date 2024-09-05 \
    --end-date 2024-12-31 \
    --prod-run

# Test single date
python scripts/fetch_nfl_player_props_historical.py \
    --test-date 2024-01-13 \
    --markets player_rush_yds

Author: Thomas Myles
Date: 2026-01-12
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

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# SSL fix
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

# =============================================================================
# GLOBAL CONFIG
# =============================================================================

API_KEY = os.getenv('ODDS_API_KEY') or os.getenv('THE_ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4'
SPORT_KEY = 'americanfootball_nfl'
REGIONS = 'us'
ODDS_FORMAT = 'american'

# S3 Configuration
S3_BUCKET = 'the-odds-api-mt'
S3_PREFIX_BASE = 'nfl/historical_player_props'  # Base path

# Local backup directory (not used by default - S3 only)
# OUTPUT_DIR_BASE = 'data/01_input/the-odds-api/nfl/historical_player_props'

# S3 client (lazy initialization)
_s3_client = None

# Snapshot time (UTC) - 12pm ET = 5 PM UTC
EVENT_LIST_HOUR = 17  # 12pm ET (5 PM UTC) - for listing games and fetching props

RATE_LIMIT_DELAY = 0.5  # seconds between API calls

# API usage tracking
credits_remaining = None
credits_used = None

# Season date ranges (NFL seasons run Sept-Feb)
SEASON_DATES = {
    '2021-22': ('2021-09-09', '2022-02-13'),  # Regular season + playoffs
    '2022-23': ('2022-09-08', '2023-02-12'),  # Regular season + playoffs
    '2023-24': ('2023-09-07', '2024-02-11'),  # Regular season + playoffs
    '2024-25': ('2024-09-05', '2025-02-09'),  # Regular season + playoffs
    '2025-26': ('2025-09-04', '2026-02-08'),  # Regular season + playoffs (estimated)
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_s3_client():
    """Get S3 client (lazy initialization)"""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3')
    return _s3_client


def get_s3_key(season, market, date_str):
    """
    Get S3 key for a specific season/market/date
    
    Args:
        season: Season like '2023-24'
        market: Market key like 'player_rush_yds'
        date_str: Date in YYYY-MM-DD format
    
    Returns:
        S3 key string
    """
    return f"{S3_PREFIX_BASE}/{season}/{market}/{date_str}.csv"


def file_exists_in_s3(season, market, date_str):
    """Check if file already exists in S3"""
    s3_client = get_s3_client()
    s3_key = get_s3_key(season, market, date_str)
    
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return True
    except:
        return False


def upload_to_s3(df, season, market, date_str):
    """
    Upload DataFrame to S3
    
    Args:
        df: DataFrame to upload
        season: Season like '2023-24'
        market: Market key like 'player_rush_yds'
        date_str: Date in YYYY-MM-DD format
    
    Returns:
        bool: Success or failure
    """
    if df.empty:
        return False
    
    s3_client = get_s3_client()
    s3_key = get_s3_key(season, market, date_str)
    
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        print(f"  💾 Uploaded to s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        print(f"  ❌ S3 upload failed: {e}")
        return False


def save_local_backup(df, season, market, date_str):
    """Save local backup copy (NOT USED - disabled by default, S3 only)"""
    # Local backup disabled - use S3 only
    # If you need local backup, uncomment the code below
    # if df.empty:
    #     return
    # output_dir = Path(OUTPUT_DIR_BASE) / season / market
    # output_dir.mkdir(parents=True, exist_ok=True)
    # output_path = output_dir / f"{date_str}.csv"
    # df.to_csv(output_path, index=False)
    # print(f"  💾 Local backup: {output_path}")
    pass


def get_historical_nfl_events(date_str):
    """
    Get list of NFL events for a specific date
    
    Args:
        date_str: Date in YYYY-MM-DD format
    
    Returns:
        dict: {
            'events': list of event dicts,
            'remaining': credits remaining,
            'used': credits used
        }
        or None if error
    """
    global credits_remaining, credits_used
    
    # Convert date to UTC timestamp at 12pm ET (5 PM UTC)
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    timestamp_str = date_obj.replace(hour=EVENT_LIST_HOUR, minute=0, second=0).isoformat() + 'Z'
    
    url = f"{BASE_URL}/historical/sports/{SPORT_KEY}/events"
    params = {
        'apiKey': API_KEY,
        'date': timestamp_str,
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params, verify=False, timeout=30)
        
        # Track API usage (handle float strings like '4535056.0')
        credits_remaining = int(float(response.headers.get('x-requests-remaining', 0)))
        credits_used = int(float(response.headers.get('x-requests-used', 0)))
        
        if response.status_code == 200:
            data = response.json()
            events = data.get('data', [])
            return {
                'events': events,
                'remaining': credits_remaining,
                'used': credits_used
            }
        elif response.status_code == 422:
            # No data available for this date/time
            return {
                'events': [],
                'remaining': credits_remaining,
                'used': credits_used
            }
        else:
            print(f"  ❌ API Error: HTTP {response.status_code}")
            if response.status_code == 422:
                print(f"     (422 = No historical data available for {date_str})")
            return None
            
    except Exception as e:
        print(f"  ❌ Request failed: {e}")
        return None


def get_historical_event_props(event_id, date_str, market):
    """
    Get player props for a specific event
    
    Args:
        event_id: Event ID from events endpoint
        date_str: Date in YYYY-MM-DD format
        market: Market key (e.g., 'player_rush_yds')
    
    Returns:
        dict: {
            'data': props data dict,
            'remaining': credits remaining,
            'used': credits used
        }
        or None if error
    """
    global credits_remaining, credits_used
    
    # Convert date to UTC timestamp at 12pm ET (5 PM UTC)
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    timestamp_str = date_obj.replace(hour=EVENT_LIST_HOUR, minute=0, second=0).isoformat() + 'Z'
    
    url = f"{BASE_URL}/historical/sports/{SPORT_KEY}/events/{event_id}/odds"
    params = {
        'apiKey': API_KEY,
        'date': timestamp_str,
        'regions': REGIONS,
        'markets': market,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params, verify=False, timeout=30)
        
        # Track API usage (handle float strings like '4535056.0')
        credits_remaining = int(float(response.headers.get('x-requests-remaining', 0)))
        credits_used = int(float(response.headers.get('x-requests-used', 0)))
        
        if response.status_code == 200:
            data = response.json()
            return {
                'data': data.get('data', {}),
                'remaining': credits_remaining,
                'used': credits_used
            }
        elif response.status_code == 422:
            # Props not available for this event
            return {
                'data': {},
                'remaining': credits_remaining,
                'used': credits_used
            }
        else:
            return None
            
    except Exception as e:
        return None


def parse_player_props(event_data, market):
    """
    Parse player props from API response
    
    Args:
        event_data: Event data dict from API
        market: Market key
    
    Returns:
        DataFrame with player props
    """
    rows = []
    
    # Extract game info
    away_team = event_data.get('away_team', 'Unknown')
    home_team = event_data.get('home_team', 'Unknown')
    commence_time = event_data.get('commence_time', '')
    
    # Parse bookmakers
    bookmakers = event_data.get('bookmakers', [])
    
    for bookmaker in bookmakers:
        bookmaker_name = bookmaker.get('key', 'unknown')
        bookmaker_last_update = bookmaker.get('last_update', '')
        
        # Find the market we're looking for
        markets = bookmaker.get('markets', [])
        for market_data in markets:
            if market_data.get('key') != market:
                continue
            
            market_last_update = market_data.get('last_update', '')
            
            # Parse outcomes (each outcome is Over/Under for a player)
            outcomes = market_data.get('outcomes', [])
            
            # Group by player (Over/Under pairs)
            player_lines = {}
            for outcome in outcomes:
                player_name = outcome.get('description', '')
                outcome_name = outcome.get('name', '')  # 'Over' or 'Under'
                point = outcome.get('point')
                price = outcome.get('price')
                
                if player_name not in player_lines:
                    player_lines[player_name] = {}
                
                if outcome_name == 'Over':
                    player_lines[player_name]['over_odds'] = price
                    player_lines[player_name]['prop_line'] = point
                elif outcome_name == 'Under':
                    player_lines[player_name]['under_odds'] = price
                    if 'prop_line' not in player_lines[player_name]:
                        player_lines[player_name]['prop_line'] = point
            
            # Create row for each player
            for player_name, line_data in player_lines.items():
                rows.append({
                    'player': player_name,
                    'away_team': away_team,
                    'home_team': home_team,
                    'game_time': commence_time,
                    'market': market,
                    'prop_line': line_data.get('prop_line'),
                    'over_odds': line_data.get('over_odds'),
                    'under_odds': line_data.get('under_odds'),
                    'bookmaker': bookmaker_name,
                    'bookmaker_last_update': bookmaker_last_update,
                    'market_last_update': market_last_update
                })
    
    if rows:
        df = pd.DataFrame(rows)
        return df
    else:
        return pd.DataFrame()


def fetch_date_market_props(date_str, market, season, save=True, skip_if_exists=True):
    """
    Fetch player props for a specific date and market
    
    Args:
        date_str: Date in YYYY-MM-DD format
        market: Market key (e.g., 'player_rush_yds')
        season: Season like '2023-24'
        save: Save to S3
        skip_if_exists: Skip if file already exists
    
    Returns:
        DataFrame with all props for that date/market
    """
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    day_of_week = date_obj.strftime('%A')
    
    print(f"\n{'='*80}")
    print(f"🏈 FETCHING: {date_str} ({day_of_week}) | Market: {market}")
    print(f"{'='*80}")
    
    # Check if already exists
    if skip_if_exists and file_exists_in_s3(season, market, date_str):
        s3_key = get_s3_key(season, market, date_str)
        print(f"  ✅ File already exists: s3://{S3_BUCKET}/{s3_key}")
        print(f"     Skipping (0 credits used)")
        return pd.DataFrame()
    
    # Get events for that date
    print(f"  📡 API CALL: Checking for events... (1 credit)", end=" ")
    result = get_historical_nfl_events(date_str)
    
    if result is None:
        print(f"❌ API Error")
        return pd.DataFrame()
    
    print(f"✓ (Remaining: {result['remaining']:,})")
    all_events = result['events']
    
    if not all_events:
        print(f"  ℹ️  No games on {date_str} (or no historical data available)")
        return pd.DataFrame()
    
    print(f"  🏈 Found {len(all_events)} games")
    
    # Fetch props for each event
    all_props = []
    credits_for_date = 1  # Started with 1 for events list
    
    for i, event in enumerate(all_events, 1):
        event_id = event.get('id')
        away_team = event.get('away_team', 'Unknown')
        home_team = event.get('home_team', 'Unknown')
        
        print(f"  📡 API CALL {i}/{len(all_events)}: {away_team} @ {home_team}... ", end="")
        
        props_result = get_historical_event_props(event_id, date_str, market)
        credits_for_date += 1
        
        if props_result is None:
            print(f"❌ API Error")
            continue
        
        event_props = props_result['data']
        
        if not event_props or not event_props.get('bookmakers'):
            print(f"⚠️  No props")
            continue
        
        # Parse props
        props_df = parse_player_props(event_props, market)
        
        if not props_df.empty:
            all_props.append(props_df)
            print(f"✅ {len(props_df)} props")
        else:
            print(f"⚠️  No props")
        
        # Rate limit
        time.sleep(RATE_LIMIT_DELAY)
    
    # Combine all props
    if all_props:
        final_df = pd.concat(all_props, ignore_index=True)
        final_df['fetch_date'] = date_str
        final_df['season'] = season
        
        print(f"\n  📊 TOTAL: {len(final_df)} prop lines from {len(all_props)} games")
        print(f"  💰 Credits used: {credits_for_date}")
        print(f"  💰 Credits remaining: {credits_remaining:,}")
        
        # Save to S3 only
        if save:
            upload_to_s3(final_df, season, market, date_str)
        
        return final_df
    else:
        print(f"\n  ℹ️  No props data for {date_str}")
        return pd.DataFrame()


def get_season_dates(season, start_date=None, end_date=None):
    """
    Get list of dates to fetch for a season
    
    Args:
        season: Season like '2023-24'
        start_date: Optional start date override (YYYY-MM-DD)
        end_date: Optional end date override (YYYY-MM-DD)
    
    Returns:
        list: List of date strings in YYYY-MM-DD format
    """
    if season not in SEASON_DATES:
        raise ValueError(f"Season {season} not configured")
    
    season_start, season_end = SEASON_DATES[season]
    
    # Override if provided
    if start_date:
        season_start = start_date
    if end_date:
        season_end = end_date
    
    start = datetime.strptime(season_start, '%Y-%m-%d').date()
    end = datetime.strptime(season_end, '%Y-%m-%d').date()
    
    # Only fetch dates up to today
    today = datetime.now().date()
    end = min(end, today)
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates


def fetch_full_season(season, markets, start_date=None, end_date=None, prod_run=False):
    """
    Fetch props for full season across multiple markets
    
    Args:
        season: Season like '2023-24'
        markets: List of market keys
        start_date: Optional start date
        end_date: Optional end date
        prod_run: If True, actually fetch. If False, estimate cost only.
    """
    print(f"\n{'='*80}")
    print(f"🏈 NFL HISTORICAL PLAYER PROPS - SEASON {season}")
    print(f"{'='*80}")
    print(f"Markets: {', '.join(markets)}")
    
    # Get dates
    dates = get_season_dates(season, start_date, end_date)
    print(f"\nDate range: {dates[0]} to {dates[-1]}")
    print(f"Total dates: {len(dates)}")
    
    # Estimate cost
    print(f"\n{'─'*80}")
    print("💰 COST ESTIMATE")
    print(f"{'─'*80}")
    
    # Check how many dates already have data
    dates_to_fetch = []
    for date_str in dates:
        needs_fetch = False
        for market in markets:
            if not file_exists_in_s3(season, market, date_str):
                needs_fetch = True
                break
        if needs_fetch:
            dates_to_fetch.append(date_str)
    
    print(f"Dates already in S3: {len(dates) - len(dates_to_fetch)}")
    print(f"Dates to fetch: {len(dates_to_fetch)}")
    print(f"Markets per date: {len(markets)}")
    
    # Estimate: 1 credit for events + ~12 credits per game per market
    # NFL typically has 10-16 games on game days (Thu/Sun/Mon)
    avg_games_per_day = 3  # Conservative (many days have 0 games)
    credits_per_date_per_market = 1 + (avg_games_per_day * 1)  # 1 for events, 1 per game
    total_estimated_credits = len(dates_to_fetch) * len(markets) * credits_per_date_per_market
    
    print(f"\nEstimated API credits: ~{total_estimated_credits:,}")
    print(f"  (Assumes avg {avg_games_per_day} games per date)")
    
    if not prod_run:
        print(f"\n{'─'*80}")
        print("⚠️  DRY RUN MODE - No data will be fetched")
        print("   Add --prod-run flag to actually fetch data")
        print(f"{'─'*80}")
        return
    
    print(f"\n{'─'*80}")
    print("🚀 STARTING FETCH")
    print(f"{'─'*80}")
    
    # Fetch each date/market combo
    total_fetched = 0
    for date_str in dates_to_fetch:
        for market in markets:
            df = fetch_date_market_props(date_str, market, season, save=True, skip_if_exists=True)
            if not df.empty:
                total_fetched += len(df)
        
        # Rate limit between dates
        time.sleep(1)
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE")
    print(f"{'='*80}")
    print(f"Total prop lines fetched: {total_fetched:,}")
    print(f"Credits remaining: {credits_remaining:,}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Fetch NFL historical player props for specific markets'
    )
    parser.add_argument(
        '--season',
        type=str,
        default='2024-25',
        help='Season to fetch (e.g., 2023-24, 2024-25)'
    )
    parser.add_argument(
        '--markets',
        nargs='+',
        required=True,
        help='Market keys to fetch (e.g., player_rush_yds player_rush_attempts)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD) - overrides season start'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD) - overrides season end'
    )
    parser.add_argument(
        '--test-date',
        type=str,
        help='Test mode - fetch single date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--prod-run',
        action='store_true',
        help='Production run - actually fetch data (default is dry run)'
    )
    
    args = parser.parse_args()
    
    if not API_KEY:
        print("❌ ERROR: API key not found!")
        print("   Set ODDS_API_KEY or THE_ODDS_API_KEY in .env file")
        sys.exit(1)
    
    # Test mode - single date
    if args.test_date:
        print(f"\n🧪 TEST MODE - Single date: {args.test_date}")
        for market in args.markets:
            df = fetch_date_market_props(
                args.test_date, 
                market, 
                args.season, 
                save=False, 
                skip_if_exists=False
            )
            if not df.empty:
                print(f"\n📊 Sample data for {market}:")
                print(df.head(10))
        return
    
    # Full season mode
    fetch_full_season(
        args.season,
        args.markets,
        start_date=args.start_date,
        end_date=args.end_date,
        prod_run=args.prod_run
    )


if __name__ == "__main__":
    main()

