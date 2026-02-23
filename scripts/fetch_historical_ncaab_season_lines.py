"""
Fetch Historical NCAAB Game Lines (Spreads + Totals)

Fetches historical betting lines (spreads + totals) for NCAAB games to support
score prediction modeling analysis. Captures closing lines by fetching at game time.

For each game, we need both spreads AND totals to calculate market-implied scores:
    Given: Spread = -5.5 (Team A favorite), Total = 145.5
    Market implies: Team A = 75.5, Team B = 70.0
    Math: team_a = (total - spread) / 2, team_b = (total + spread) / 2

Data saved to: s3://ncaab-betting-mt/data/01_input/the-odds-api/ncaab/game_lines/

Output Format:
- One CSV per date: YYYY-MM-DD.csv
- Columns: date, event_id, home_team, away_team, commence_time_et, 
           consensus_spread, consensus_total, [bookmaker spreads/totals]

Usage:
    # Fetch all available dates (2025-26 season to date)
    python scripts/fetch_historical_ncaab_season_lines.py --s3
    
    # Fetch past season (e.g., 2024-25)
    python scripts/fetch_historical_ncaab_season_lines.py --season 2024-25 --s3 --skip-existing
    
    # Fetch specific date
    python scripts/fetch_historical_ncaab_season_lines.py --date 2025-11-09 --s3
    
    # Fetch season with custom date range
    python scripts/fetch_historical_ncaab_season_lines.py --season 2024-25 --start-date 2024-12-01 --end-date 2025-03-31 --s3
    
    # Test mode (save locally)
    python scripts/fetch_historical_ncaab_season_lines.py --date 2025-11-09 --test
    
    # Skip dates already in S3 (resume interrupted run - RECOMMENDED)
    python scripts/fetch_historical_ncaab_season_lines.py --s3 --skip-existing

Important:
    - Uses --skip-existing to avoid re-fetching dates already in S3
    - Saves empty files for dates with no games (prevents re-fetching)
    - Each date costs 1 credit (events) + 20 credits per game (odds)
    - Shows S3 path after each successful upload

Cost Estimate:
    ~1,500 games × 10 credits per game = 15,000 credits
    (Season runs Nov-April, ~150 game days, ~10-15 games per day on average)

Context:
User request - "we would need historical game outcomes, spreads and o/u's for ncaab"
Building a v0 model to predict NCAAB scores and compare against market-implied scores
from betting lines.

Author: Thomas Myles
Date: 2026-01-15
"""

import sys
import os
import requests
import pandas as pd
import boto3
import urllib3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import argparse
import time
from io import StringIO

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Find project root
def find_project_root():
    """Find project root by looking for .gitignore file."""
    current = Path.cwd()
    while current != current.parent:
        if (current / '.gitignore').exists():
            return current
        current = current.parent
    return Path.cwd()

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from config_loader import get_config

# Load config
CONFIG = get_config()
load_dotenv()

# =============================================================================
# API CONFIGURATION
# =============================================================================

API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4'
SPORT = 'basketball_ncaab'
MARKETS = 'spreads,totals'  # BOTH needed for implied score calculation
REGIONS = 'us'
ODDS_FORMAT = 'american'

# S3 Configuration
S3_BUCKET = 'ncaab-betting-mt'
S3_PATH = 'data/01_input/the-odds-api/ncaab/game_lines/'

# Local test directory
LOCAL_TEST_DIR = Path.home() / 'Downloads' / 'tmp'

# Timing (UTC) - fetch at noon ET (5pm UTC) to capture closing lines
EVENT_LIST_HOUR = 17  # 12pm ET
ODDS_SNAPSHOT_HOUR = 17  # 12pm ET

RATE_LIMIT_DELAY = 0.5  # seconds between API calls

# API usage tracking
credits_remaining = None
credits_used = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_api_key():
    """Verify API key is loaded."""
    if not API_KEY or API_KEY == 'your_api_key_here':
        print("❌ ERROR: No valid API key found!")
        print("Make sure ODDS_API_KEY is set in your .env file")
        return False
    return True


def get_historical_ncaab_events(date_str):
    """
    Get historical NCAAB events for a specific date.
    
    Cost: 1 credit per request
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD'
    
    Returns:
        Dict with events list, cost, remaining credits (or None on error)
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
            # No data for this date
            return {'events': [], 'cost': 0, 'remaining': credits_remaining}
        print(f"❌ HTTP Error for {date_str}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error for {date_str}: {e}")
        return None


def get_historical_event_odds(event_id, date_str):
    """
    Get historical odds for a specific event.
    
    Cost: 10 credits per event (for spreads + totals)
    
    Args:
        event_id: Event ID from get_historical_ncaab_events
        date_str: Date string for timestamp
    
    Returns:
        Dict with data, cost, remaining (or None on error)
    """
    global credits_remaining, credits_used
    
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    timestamp = date_obj.replace(hour=ODDS_SNAPSHOT_HOUR, minute=0, second=0).isoformat() + 'Z'
    
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
        print(f"  ❌ Error for event {event_id[:8]}: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def parse_game_lines(games):
    """
    Parse game lines from API response into standardized format.
    
    Extracts spreads and totals from each bookmaker, calculates consensus values.
    
    Args:
        games: List of game dicts from API
    
    Returns:
        pd.DataFrame with columns: date, event_id, home_team, away_team,
                                    commence_time_et, consensus_spread, consensus_total,
                                    [bookmaker_spread, bookmaker_total columns]
    """
    rows = []
    
    for game in games:
        event_id = game.get('id')
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        commence_time = game.get('commence_time')
        
        # Convert commence time to ET
        if commence_time:
            commence_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            commence_et = commence_dt.astimezone(ZoneInfo('America/New_York'))
            game_date = commence_et.date()
            commence_time_et = commence_et.strftime('%Y-%m-%d %H:%M:%S')
        else:
            game_date = None
            commence_time_et = None
        
        # Extract bookmakers data
        bookmakers_data = game.get('bookmakers', [])
        
        # Collect all spreads and totals
        spreads = []  # Home team spreads
        totals = []
        bookmaker_spreads = {}
        bookmaker_totals = {}
        
        for book in bookmakers_data:
            book_name = book.get('key')
            markets = book.get('markets', [])
            
            for market in markets:
                market_key = market.get('key')
                outcomes = market.get('outcomes', [])
                
                if market_key == 'spreads':
                    # Find home team spread
                    for outcome in outcomes:
                        if outcome.get('name') == home_team:
                            spread_value = outcome.get('point')
                            if spread_value is not None:
                                spreads.append(spread_value)
                                bookmaker_spreads[book_name] = spread_value
                            break
                
                elif market_key == 'totals':
                    # Get the total (Over/Under line)
                    for outcome in outcomes:
                        if outcome.get('name') == 'Over':
                            total_value = outcome.get('point')
                            if total_value is not None:
                                totals.append(total_value)
                                bookmaker_totals[book_name] = total_value
                            break
        
        # Calculate consensus (average)
        consensus_spread = sum(spreads) / len(spreads) if spreads else None
        consensus_total = sum(totals) / len(totals) if totals else None
        
        # Build row
        row = {
            'date': game_date,
            'event_id': event_id,
            'home_team': home_team,
            'away_team': away_team,
            'commence_time_et': commence_time_et,
            'consensus_spread': round(consensus_spread, 2) if consensus_spread else None,
            'consensus_total': round(consensus_total, 2) if consensus_total else None,
            'num_books_spread': len(spreads),
            'num_books_total': len(totals),
        }
        
        # Add individual bookmaker columns
        for book_name, spread in bookmaker_spreads.items():
            col_name = f"{book_name}_spread"
            row[col_name] = spread
        
        for book_name, total in bookmaker_totals.items():
            col_name = f"{book_name}_total"
            row[col_name] = total
        
        rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Reorder columns - fixed columns first, then bookmaker columns
    fixed_cols = ['date', 'event_id', 'home_team', 'away_team', 'commence_time_et',
                  'consensus_spread', 'consensus_total', 'num_books_spread', 'num_books_total']
    bookmaker_cols = sorted([c for c in df.columns if c not in fixed_cols])
    df = df[fixed_cols + bookmaker_cols]
    
    return df


def check_s3_file_exists(date_str):
    """
    Check if a file already exists in S3 for a given date.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
    
    Returns:
        bool: True if file exists
    """
    s3_key = f"{S3_PATH}{date_str}.csv"
    
    try:
        s3_client = boto3.client('s3')
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return True
    except:
        return False


def save_to_s3(df, date_str):
    """
    Upload game lines to S3.
    Uploads file with headers even if no games found.
    
    Args:
        df: DataFrame with game lines
        date_str: Date string in YYYY-MM-DD format
    
    Returns:
        bool: True if uploaded successfully
    """
    s3_key = f"{S3_PATH}{date_str}.csv"
    
    # Convert to CSV (even if empty, with headers)
    csv_buffer = StringIO()
    if df.empty:
        # Create empty dataframe with expected columns
        empty_df = pd.DataFrame(columns=[
            'date', 'event_id', 'home_team', 'away_team', 'commence_time_et',
            'consensus_spread', 'consensus_total', 'num_books_spread', 'num_books_total'
        ])
        empty_df.to_csv(csv_buffer, index=False)
    else:
        df.to_csv(csv_buffer, index=False)
    
    try:
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        if df.empty:
            print(f"   ✅ Uploaded empty file (headers only)")
            print(f"      s3://{S3_BUCKET}/{s3_key}")
        else:
            print(f"   ✅ Uploaded to S3")
            print(f"      s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        print(f"   ❌ S3 upload failed: {e}")
        return False


def save_to_local(df, date_str):
    """
    Save game lines locally (test mode).
    
    Args:
        df: DataFrame with game lines
        date_str: Date string in YYYY-MM-DD format
    
    Returns:
        bool: True if saved successfully
    """
    LOCAL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    filename = f"ncaab_game_lines_{date_str}.csv"
    output_path = LOCAL_TEST_DIR / filename
    
    try:
        df.to_csv(output_path, index=False)
        print(f"   💾 Saved locally to: {output_path}")
        return True
    except Exception as e:
        print(f"   ❌ Local save failed: {e}")
        return False


def fetch_date(date_str, upload_s3=False, test_mode=False, skip_existing=False):
    """
    Fetch game lines for a single date.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        upload_s3: Whether to upload to S3
        test_mode: Whether to save locally
        skip_existing: If True, skip if file already exists in S3
    
    Returns:
        pd.DataFrame with game lines (or None if skipped)
    """
    # Check if already exists and should skip
    if skip_existing and upload_s3:
        if check_s3_file_exists(date_str):
            print(f"\n⏭️  Skipping {date_str} (already exists in S3)")
            return None
    
    print(f"\n📥 Fetching NCAAB game lines for {date_str}...")
    
    # Step 1: Get events for this date
    events_result = get_historical_ncaab_events(date_str)
    
    if not events_result:
        print("   ❌ Failed to fetch events")
        return pd.DataFrame()
    
    events = events_result['events']
    
    if not events:
        print(f"   ℹ️  No games found for {date_str}")
        # Still save empty file to S3 so we don't re-fetch this date
        if upload_s3:
            save_to_s3(pd.DataFrame(), date_str)
        return pd.DataFrame()
    
    print(f"   ✅ Found {len(events)} games (will filter to games on {date_str})")
    print(f"   💰 API Credits: {events_result['cost']} used, {events_result['remaining']} remaining")
    
    # Step 2: Get odds for each event
    games_with_odds = []
    total_cost = events_result['cost']
    
    for i, event in enumerate(events, 1):
        event_id = event.get('id')
        home = event.get('home_team', '')
        away = event.get('away_team', '')
        
        print(f"   📊 [{i}/{len(events)}] {away} @ {home}...", end=' ')
        
        odds_result = get_historical_event_odds(event_id, date_str)
        
        if odds_result and odds_result['data']:
            games_with_odds.append(odds_result['data'])
            total_cost += odds_result['cost']
            print(f"✅ ({odds_result['cost']} credits)")
        else:
            print("❌ No odds")
    
    if not games_with_odds:
        print(f"   ⚠️  No games with odds data for {date_str}")
        # Still save empty file to S3 so we don't re-fetch this date
        if upload_s3:
            save_to_s3(pd.DataFrame(), date_str)
        return pd.DataFrame()
    
    # Step 3: Parse into DataFrame
    df = parse_game_lines(games_with_odds)
    
    # Step 4: Filter to only games happening on this date (in ET)
    # The API returns all events with lines available on the query date,
    # but we only want games actually played on this date
    if not df.empty:
        # Convert date_str to date object for comparison
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        # Track filtered out games before filtering
        filtered_out_df = df[df['date'] != target_date].copy()
        
        # Filter: keep only games where date (game date in ET) matches target date
        df_before_filter = len(df)
        df = df[df['date'] == target_date].copy()
        
        games_filtered_out = df_before_filter - len(df)
        print(f"   🔍 Filtering: {df_before_filter} games → {len(df)} games on {date_str}")
        
        # Show which games were filtered out
        if games_filtered_out > 0:
            print(f"\n   ⏭️  Filtered out {games_filtered_out} game(s) not on {date_str}:")
            for _, row in filtered_out_df.iterrows():
                actual_date = row['date']
                home = row['home_team']
                away = row['away_team']
                print(f"      {away} @ {home} (actual date: {actual_date})")
    
    print(f"\n   📊 Summary:")
    print(f"      Games with lines on {date_str}: {len(df)}")
    print(f"      Total API cost: {total_cost} credits")
    print(f"      Credits remaining: {credits_remaining}")
    
    # Show sample
    if not df.empty:
        print(f"\n   Sample lines:")
        for _, row in df.head(3).iterrows():
            spread = row['consensus_spread']
            total = row['consensus_total']
            home = row['home_team']
            away = row['away_team']
            print(f"     {away} @ {home}: Spread {spread:+.1f}, Total {total:.1f}")
    else:
        print(f"\n   ℹ️  No games scheduled for {date_str}")
    
    # Save results (including empty file if no games on this date)
    if upload_s3:
        save_to_s3(df, date_str)
    
    if test_mode:
        save_to_local(df, date_str)
    
    return df


def parse_season_to_dates(season_str):
    """
    Parse season string to start/end dates.
    
    NCAAB seasons typically run:
    - Start: First Monday of November
    - End: First week of April (regular season + conf tournaments)
    
    Args:
        season_str: Season string like "2024-25" or "2025-26"
    
    Returns:
        Tuple of (start_date, end_date) as date objects
    """
    parts = season_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid season format: {season_str}. Use format like '2024-25'")
    
    start_year = int(parts[0])
    end_year = int(parts[1])
    
    # Validate year sequence
    if end_year != start_year + 1 and end_year != (start_year % 100) + 1:
        raise ValueError(f"Invalid season years: {season_str}")
    
    # Convert 2-digit end year to 4-digit if needed
    if end_year < 100:
        end_year = (start_year // 100) * 100 + end_year
    
    # NCAAB season dates (approximate)
    # Start: November 4th (typical first Monday of November)
    # End: April 7th (after conference tournaments, before NCAA tournament)
    start_date = datetime(start_year, 11, 4).date()
    end_date = datetime(end_year, 4, 7).date()
    
    return start_date, end_date


def get_season_dates(start_date_str=None, end_date_str=None):
    """
    Get list of dates for the season.
    
    Args:
        start_date_str: Optional start date (YYYY-MM-DD)
        end_date_str: Optional end date (YYYY-MM-DD)
    
    Returns:
        List of date strings in YYYY-MM-DD format
    """
    # Default: 2025-26 season
    if start_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    else:
        start_date = datetime(2025, 11, 3).date()  # Season start
    
    if end_date_str:
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    else:
        end_date = datetime.now().date()  # Today
    
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates


def fetch_season(upload_s3=False, test_mode=False, skip_existing=False, start_date=None, end_date=None):
    """
    Fetch game lines for entire season (or date range).
    
    Args:
        upload_s3: Whether to upload to S3
        test_mode: Whether to save locally
        skip_existing: If True, skip dates already in S3
        start_date: Optional start date override
        end_date: Optional end date override
    
    Returns:
        pd.DataFrame with all game lines
    """
    print(f"\n🔄 Fetching NCAAB season game lines...")
    
    dates = get_season_dates(start_date, end_date)
    
    print(f"   Date range: {dates[0]} to {dates[-1]}")
    print(f"   Total dates: {len(dates)}")
    if skip_existing:
        print(f"   Mode: Skip existing files in S3")
    
    all_dfs = []
    dates_processed = 0
    dates_skipped = 0
    games_found = 0
    
    for date_str in dates:
        df = fetch_date(date_str, upload_s3=upload_s3, test_mode=test_mode, skip_existing=skip_existing)
        
        if df is None:
            dates_skipped += 1
        elif not df.empty:
            all_dfs.append(df)
            games_found += len(df)
        
        dates_processed += 1
        
        # Progress update every 20 dates
        if dates_processed % 20 == 0:
            skip_msg = f", {dates_skipped} skipped" if skip_existing else ""
            print(f"\n   📊 Progress: {dates_processed}/{len(dates)} dates processed{skip_msg}, {games_found} games found")
    
    print(f"\n✅ Season fetch complete!")
    print(f"   Total dates processed: {dates_processed}")
    if skip_existing:
        print(f"   Dates skipped (already in S3): {dates_skipped}")
        print(f"   Dates fetched: {dates_processed - dates_skipped}")
    print(f"   Total games found: {games_found}")
    print(f"   Final credits remaining: {credits_remaining}")
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Fetch historical NCAAB game lines (spreads + totals)')
    parser.add_argument('--date', type=str, default=None,
                       help='Specific date to fetch (YYYY-MM-DD)')
    parser.add_argument('--season', type=str, default=None,
                       help='Season to fetch (e.g., "2024-25" or "2025-26"). Auto-sets Nov-April date range.')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for season fetch (YYYY-MM-DD). Overrides --season start date.')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for season fetch (YYYY-MM-DD). Overrides --season end date.')
    parser.add_argument('--s3', action='store_true',
                       help='Upload results to S3')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip dates that already exist in S3')
    parser.add_argument('--test', action='store_true',
                       help='Save results locally to ~/Downloads/tmp')
    
    args = parser.parse_args()
    
    # Check API key
    if not check_api_key():
        return
    
    # Handle season argument
    season_start = args.start_date
    season_end = args.end_date
    
    if args.season:
        try:
            auto_start, auto_end = parse_season_to_dates(args.season)
            # Use season dates as defaults, but allow overrides
            if not season_start:
                season_start = auto_start.strftime('%Y-%m-%d')
            if not season_end:
                season_end = auto_end.strftime('%Y-%m-%d')
            season_display = args.season
        except ValueError as e:
            print(f"❌ Error: {e}")
            return
    else:
        season_display = "2025-26 (default)"
    
    print("=" * 80)
    print("NCAAB GAME LINES FETCHER (Spreads + Totals)")
    print("=" * 80)
    print(f"Sport: NCAAB (College Basketball)")
    print(f"Season: {season_display}")
    print(f"Markets: Spreads + Totals")
    print(f"S3 Upload: {'✅ Enabled' if args.s3 else '❌ Disabled'}")
    print(f"Skip Existing: {'✅ Enabled' if args.skip_existing else '❌ Disabled'}")
    print(f"Test Mode: {'✅ Enabled (saving to ~/Downloads/tmp)' if args.test else '❌ Disabled'}")
    
    # Single date mode
    if args.date:
        df = fetch_date(args.date, upload_s3=args.s3, test_mode=args.test, skip_existing=args.skip_existing)
        
        if df is not None and not df.empty:
            print(f"\n📋 Results ({len(df)} games):")
            print(df[['home_team', 'away_team', 'consensus_spread', 'consensus_total']].to_string(index=False))
    
    # Season mode
    else:
        df = fetch_season(
            upload_s3=args.s3, 
            test_mode=args.test, 
            skip_existing=args.skip_existing,
            start_date=season_start,
            end_date=season_end
        )
        
        if not df.empty:
            print(f"\n📊 Season Summary:")
            print(f"   Total games: {len(df)}")
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"   Avg books per game (spread): {df['num_books_spread'].mean():.1f}")
            print(f"   Avg books per game (total): {df['num_books_total'].mean():.1f}")
    
    print("\n" + "=" * 80)
    print("✅ Fetch complete!")
    print("=" * 80)
    
    return df


if __name__ == '__main__':
    main()

