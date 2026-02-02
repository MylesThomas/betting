"""
Collect live NBA odds and scores data continuously during games.

PURPOSE:
Run during live NBA games to build up a parquet dataset with:
- Live odds from The Odds API (spreads + moneylines from multiple bookmakers)
- Live scores and game status from ESPN
- Timestamp of each collection

USAGE:
    python tmp/collect_live_game_data.py
    
    # Optional: specify collection interval (default 60 seconds)
    python tmp/collect_live_game_data.py --interval 30
    
    # Press Ctrl+C to stop collection

DATA STORAGE:
- Checks if parquet file exists in ~/Downloads/tmp/
- If exists: reads existing data, appends new records, saves back
- If not: creates new parquet file
- Filename: live_nba_data_YYYYMMDD.parquet (one file per day)

CONTEXT:
Building a live odds tracker. Need real data from actual games to validate:
- API response consistency
- Bookmaker update frequencies  
- How odds shift during games
- Data schema design for production tracker
"""

import os
import sys
import json
import time
import requests
import argparse
import warnings
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


# =============================================================================
# CONFIGURATION
# =============================================================================

ODDS_API_KEY = os.getenv('ODDS_API_KEY')
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'
ESPN_NBA_SCOREBOARD = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'

SPORT_NBA = 'basketball_nba'

# Output directory and file pattern
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
    'save': '💾',
    'refresh': '🔄',
    'fire': '🔥',
    'money': '💰',
    'warning': '⚠️',
}


# =============================================================================
# HELPER FUNCTIONS (in execution flow order)
# =============================================================================

def get_output_filepaths():
    """
    Get parquet filepaths for today's date.
    
    Returns:
        Tuple of (odds_file, espn_file, joined_file)
    """
    today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d')
    odds_file = OUTPUT_DIR / f'odds_api_{today}.parquet'
    espn_file = OUTPUT_DIR / f'espn_scoreboard_{today}.parquet'
    joined_file = OUTPUT_DIR / f'live_nba_data_{today}.parquet'
    return odds_file, espn_file, joined_file


def get_current_time_et():
    """Get current time in ET."""
    return datetime.now(ZoneInfo('America/New_York'))


def fetch_odds_api_data():
    """
    Fetch live NBA odds from The Odds API.
    
    Returns:
        Tuple of (games_list, usage_info_dict) or (None, None) if error
        
    Usage info includes:
        - requests_used: Total API calls used
        - requests_remaining: Remaining API calls
        - status_code: HTTP status
    """
    url = f"{ODDS_API_BASE}/sports/{SPORT_NBA}/odds"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'spreads,h2h',  # Both markets in 1 call = 1 credit
        'oddsFormat': 'american',
    }
    
    try:
        response = requests.get(url, params=params, timeout=10, verify=False)
        response.raise_for_status()
        
        usage_info = {
            'requests_used': response.headers.get('x-requests-used', 'unknown'),
            'requests_remaining': response.headers.get('x-requests-remaining', 'unknown'),
            'status_code': response.status_code,
        }
        
        return response.json(), usage_info
    except requests.exceptions.RequestException as e:
        print(f"{EMOJI['error']} Odds API Error: {e}")
        return None, None


def fetch_espn_scoreboard_data():
    """
    Fetch live NBA scoreboard from ESPN.
    
    Returns:
        ESPN scoreboard data, or None if error
    """
    try:
        response = requests.get(ESPN_NBA_SCOREBOARD, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"{EMOJI['error']} ESPN API Error: {e}")
        return None


def parse_odds_data(odds_games, collection_timestamp):
    """
    Parse Odds API response into flat records.
    
    One record per game-bookmaker-market combination.
    
    Args:
        odds_games: List of games from Odds API
        collection_timestamp: When we collected this data
        
    Returns:
        List of dicts (one per bookmaker per game)
    """
    records = []
    
    for game in odds_games:
        game_id = game['id']
        away_team = game['away_team']
        home_team = game['home_team']
        commence_time = game['commence_time']
        
        bookmakers = game.get('bookmakers', [])
        
        for book in bookmakers:
            book_key = book['key']
            book_last_update = book['last_update']
            
            # Find spreads and h2h markets
            spreads_market = next((m for m in book['markets'] if m['key'] == 'spreads'), None)
            h2h_market = next((m for m in book['markets'] if m['key'] == 'h2h'), None)
            
            # Extract spread data
            away_spread = None
            away_spread_price = None
            home_spread = None
            home_spread_price = None
            
            if spreads_market:
                for outcome in spreads_market['outcomes']:
                    if outcome['name'] == away_team:
                        away_spread = outcome.get('point')
                        away_spread_price = outcome.get('price')
                    elif outcome['name'] == home_team:
                        home_spread = outcome.get('point')
                        home_spread_price = outcome.get('price')
            
            # Extract moneyline data
            away_ml = None
            home_ml = None
            
            if h2h_market:
                for outcome in h2h_market['outcomes']:
                    if outcome['name'] == away_team:
                        away_ml = outcome['price']
                    elif outcome['name'] == home_team:
                        home_ml = outcome['price']
            
        records.append({
            'query_time': collection_timestamp,
            'collection_timestamp': collection_timestamp,
            'game_id': game_id,
            'away_team': away_team,
            'home_team': home_team,
            'commence_time': commence_time,
            'bookmaker': book_key,
            'bookmaker_last_update': book_last_update,
            'away_spread': away_spread,
            'away_spread_price': away_spread_price,
            'home_spread': home_spread,
            'home_spread_price': home_spread_price,
            'away_ml': away_ml,
            'home_ml': home_ml,
        })
    
    return records


def parse_espn_data(espn_data, collection_timestamp):
    """
    Parse ESPN scoreboard response into flat records.
    
    One record per game.
    
    Args:
        espn_data: ESPN scoreboard response
        collection_timestamp: When we collected this data
        
    Returns:
        List of dicts (one per game)
        
    Note:
        ESPN API game_status values (from status['type']['state']):
        - 'pre': game hasn't started yet
        - 'in': game is currently live/in progress
        - 'post': game has finished
    """
    records = []
    
    events = espn_data.get('events', [])
    
    for event in events:
        competition = event['competitions'][0]
        competitors = competition['competitors']
        
        away_team = next(c for c in competitors if c['homeAway'] == 'away')
        home_team = next(c for c in competitors if c['homeAway'] == 'home')
        
        status = event['status']
        
        # Parse clock if live
        period = status.get('period')
        display_clock = status.get('displayClock')
        time_remaining_minutes = None
        
        if display_clock and status['type']['state'] == 'in':
            try:
                parts = display_clock.split(':')
                if len(parts) == 2:
                    mins = int(parts[0])
                    secs = int(parts[1])
                    time_remaining_minutes = mins + secs / 60
            except:
                pass
        
        records.append({
            'query_time': collection_timestamp,
            'collection_timestamp': collection_timestamp,
            'espn_game_id': event['id'],
            'away_team_espn': away_team['team']['displayName'],
            'home_team_espn': home_team['team']['displayName'],
            'away_score': int(away_team['score']) if away_team['score'] else None,
            'home_score': int(home_team['score']) if home_team['score'] else None,
            'game_status': status['type']['state'],
            'game_status_description': status['type']['description'],
            'period': period,
            'display_clock': display_clock,
            'time_remaining_minutes': time_remaining_minutes,
        })
    
    return records


def append_to_parquet(new_records, filepath):
    """
    Append new records to parquet file.
    
    If file exists: read existing data, append, save back
    If not: create new file with just the new records
    
    Args:
        new_records: List of dicts to append
        filepath: Path to parquet file
    """
    new_df = pd.DataFrame(new_records)
    
    if filepath.exists():
        # Read existing data
        existing_df = pd.read_parquet(filepath)
        # Append new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Save back
        combined_df.to_parquet(filepath, index=False)
        return len(existing_df), len(combined_df)
    else:
        # Create new file
        new_df.to_parquet(filepath, index=False)
        return 0, len(new_df)


def worker_collect_odds(collection_timestamp, odds_filepath):
    """
    Worker 1: Fetch odds data and save to parquet.
    
    Args:
        collection_timestamp: ISO timestamp for this collection
        odds_filepath: Path to save odds data
        
    Returns:
        Tuple of (success, num_records, error_message, usage_info)
    """
    try:
        odds_games, usage_info = fetch_odds_api_data()
        
        if not odds_games:
            return False, 0, "No games returned from API", usage_info
        
        odds_records = parse_odds_data(odds_games, collection_timestamp)
        
        if odds_records:
            prev_count, new_count = append_to_parquet(odds_records, odds_filepath)
            return True, len(odds_records), None, usage_info
        else:
            return False, 0, "No records parsed", usage_info
            
    except Exception as e:
        return False, 0, str(e), None


def worker_collect_espn(collection_timestamp, espn_filepath):
    """
    Worker 2: Fetch ESPN data and save to parquet.
    
    Args:
        collection_timestamp: ISO timestamp for this collection
        espn_filepath: Path to save ESPN data
        
    Returns:
        Tuple of (success, num_records, error_message, num_live_games)
    """
    try:
        espn_data = fetch_espn_scoreboard_data()
        
        if not espn_data:
            return False, 0, "No data returned from API", 0
        
        espn_records = parse_espn_data(espn_data, collection_timestamp)
        
        if espn_records:
            prev_count, new_count = append_to_parquet(espn_records, espn_filepath)
            num_live = sum(1 for r in espn_records if r['game_status'] == 'in')
            return True, len(espn_records), None, num_live
        else:
            return False, 0, "No records parsed", 0
            
    except Exception as e:
        return False, 0, str(e), 0


def join_and_save_data(collection_timestamp, odds_filepath, espn_filepath, joined_filepath):
    """
    Join odds and ESPN data, save to combined file.
    
    Joins on team names (with fuzzy matching if needed).
    
    Args:
        collection_timestamp: ISO timestamp for this collection
        odds_filepath: Path to odds data
        espn_filepath: Path to ESPN data
        joined_filepath: Path to save joined data
        
    Returns:
        Number of joined records for this collection, or 0 if failed
    """
    try:
        # Read the latest records (matching this collection timestamp)
        if not odds_filepath.exists() or not espn_filepath.exists():
            return 0
        
        odds_df = pd.read_parquet(odds_filepath)
        espn_df = pd.read_parquet(espn_filepath)
        
        # Filter to just this collection timestamp
        odds_current = odds_df[odds_df['collection_timestamp'] == collection_timestamp].copy()
        espn_current = espn_df[espn_df['collection_timestamp'] == collection_timestamp].copy()
        
        if len(odds_current) == 0 or len(espn_current) == 0:
            return 0
        
        # Join on away_team and home_team
        # Note: This is a simple exact match. In production, might need fuzzy matching
        joined = odds_current.merge(
            espn_current,
            left_on=['away_team', 'home_team'],
            right_on=['away_team_espn', 'home_team_espn'],
            how='left',
            suffixes=('', '_espn')
        )
        
        if len(joined) > 0:
            # Append to joined file
            prev_count, new_count = append_to_parquet(joined.to_dict('records'), joined_filepath)
            return len(joined)
        
        return 0
        
    except Exception as e:
        print(f"  {EMOJI['error']} Join error: {e}")
        return 0


def collect_once(odds_filepath, espn_filepath, joined_filepath):
    """
    Perform one collection cycle using concurrent workers.
    
    Workers run in parallel:
    - Worker 1: Fetch odds, save to odds_filepath
    - Worker 2: Fetch ESPN, save to espn_filepath
    - Main: Join both, save to joined_filepath
    
    OPTIMIZATION: Check ESPN first (free) to see if any games are live.
    If no live games, skip the Odds API call (saves API credits).
    
    Args:
        odds_filepath: Path for odds data
        espn_filepath: Path for ESPN data
        joined_filepath: Path for joined data
        
    Returns:
        Dict with collection results
    """
    # Capture exact query time (when we start API calls)
    query_time = datetime.now(timezone.utc)
    collection_timestamp = query_time.isoformat()
    
    results = {
        'odds_success': False,
        'odds_records': 0,
        'odds_error': None,
        'odds_usage': None,
        'espn_success': False,
        'espn_records': 0,
        'espn_error': None,
        'espn_live_games': 0,
        'joined_records': 0,
        'skipped_odds_api': False,
    }
    
    # Step 1: Check ESPN first (free API) to see if any games are live
    espn_success, espn_records, espn_error, num_live = worker_collect_espn(collection_timestamp, espn_filepath)
    results['espn_success'] = espn_success
    results['espn_records'] = espn_records
    results['espn_error'] = espn_error
    results['espn_live_games'] = num_live
    
    # Step 2: Only call Odds API if there are live games
    if espn_success and num_live > 0:
        # Games are live - fetch odds
        odds_success, odds_records, odds_error, usage_info = worker_collect_odds(collection_timestamp, odds_filepath)
        results['odds_success'] = odds_success
        results['odds_records'] = odds_records
        results['odds_error'] = odds_error
        results['odds_usage'] = usage_info
        
        # Join data if both succeeded
        if odds_success:
            joined_count = join_and_save_data(
                collection_timestamp, 
                odds_filepath, 
                espn_filepath, 
                joined_filepath
            )
            results['joined_records'] = joined_count
    else:
        # No live games - skip Odds API call
        results['skipped_odds_api'] = True
        if espn_success and num_live == 0:
            print(f"  {EMOJI['info']} No live games - skipping Odds API call (saved 1 API credit)")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run continuous data collection."""
    parser = argparse.ArgumentParser(description='Collect live NBA odds and scores')
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Collection interval in seconds (default: 60)'
    )
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"{EMOJI['nba']} LIVE NBA DATA COLLECTOR")
    print(f"{'='*80}\n")
    
    if not ODDS_API_KEY:
        print(f"{EMOJI['error']} ERROR: ODDS_API_KEY not found in environment.")
        print("Set it with: export ODDS_API_KEY='your_key_here'\n")
        return
    
    odds_filepath, espn_filepath, joined_filepath = get_output_filepaths()
    
    print(f"{EMOJI['info']} Collection Interval: {args.interval} seconds")
    print(f"{EMOJI['info']} Collection Strategy: Concurrent workers (parallel API calls)")
    print(f"\n{EMOJI['save']} Output Files:")
    print(f"  Worker 1 (Odds API):   {odds_filepath.name}")
    print(f"  Worker 2 (ESPN):       {espn_filepath.name}")
    print(f"  Joined Data:           {joined_filepath.name}")
    print(f"\n{EMOJI['info']} Press Ctrl+C to stop\n")
    
    print(f"{'='*80}\n")
    
    # Calculate and wait for next wall clock interval before starting
    now = time.time()
    next_run = ((now // args.interval) + 1) * args.interval
    sleep_time = next_run - now
    
    if sleep_time > 0:
        from datetime import datetime as dt
        next_time_utc = dt.fromtimestamp(next_run, tz=ZoneInfo('UTC'))
        next_time_et = next_time_utc.astimezone(ZoneInfo('America/New_York'))
        
        print(f"{EMOJI['time']} Waiting {sleep_time:.0f}s for next wall clock interval...")
        print(f"{EMOJI['info']} First collection at: {next_time_et.strftime('%I:%M:%S %p ET')}\n")
        time.sleep(sleep_time)
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            current_time = get_current_time_et()
            
            print(f"{EMOJI['refresh']} Collection #{iteration} - {current_time.strftime('%I:%M:%S %p ET')}")
            
            # Collect data (concurrent workers)
            results = collect_once(odds_filepath, espn_filepath, joined_filepath)
            
            # Report odds worker results
            if results['skipped_odds_api']:
                # Skipped due to no live games
                pass  # Already printed the skip message
            elif results['odds_success']:
                print(f"  {EMOJI['success']} Worker 1 (Odds API): {results['odds_records']} records")
                
                # API usage info
                if results['odds_usage']:
                    usage = results['odds_usage']
                    print(f"    {EMOJI['money']} API Credits: {usage['requests_used']} used | {usage['requests_remaining']} remaining")
                    print(f"    {EMOJI['info']} Note: 1 API call = spreads + h2h markets (both in 1 request)")
                
                # Show details from latest collection
                if odds_filepath.exists():
                    odds_df = pd.read_parquet(odds_filepath)
                    unique_games = odds_df['game_id'].nunique()
                    unique_books = odds_df['bookmaker'].nunique()
                    total_rows = len(odds_df)
                    print(f"    {EMOJI['nba']} Total games tracked: {unique_games} | Bookmakers: {unique_books} | Rows: {total_rows}")
            else:
                print(f"  {EMOJI['error']} Worker 1 (Odds API): FAILED - {results['odds_error']}")
                if results['odds_usage']:
                    usage = results['odds_usage']
                    print(f"    {EMOJI['money']} API Credits: {usage['requests_used']} used | {usage['requests_remaining']} remaining")
            
            # Report ESPN worker results
            if results['espn_success']:
                print(f"  {EMOJI['success']} Worker 2 (ESPN API): {results['espn_records']} records")
                print(f"    {EMOJI['fire']} Live games: {results['espn_live_games']}/{results['espn_records']}")
                
                # Show total rows
                if espn_filepath.exists():
                    espn_df = pd.read_parquet(espn_filepath)
                    total_rows = len(espn_df)
                    print(f"    {EMOJI['chart']} Total rows: {total_rows}")
            else:
                print(f"  {EMOJI['error']} Worker 2 (ESPN API): FAILED - {results['espn_error']}")
            
            # Report joined data results
            if results['joined_records'] > 0:
                print(f"  {EMOJI['chart']} Joined: {results['joined_records']} records")
                
                # Show consensus lines for each game (split by live vs upcoming)
                if odds_filepath.exists() and espn_filepath.exists():
                    try:
                        odds_df = pd.read_parquet(odds_filepath)
                        espn_df = pd.read_parquet(espn_filepath)
                        
                        # Get latest collection only
                        latest_timestamp = odds_df['collection_timestamp'].max()
                        latest_odds = odds_df[odds_df['collection_timestamp'] == latest_timestamp]
                        latest_espn = espn_df[espn_df['collection_timestamp'] == latest_timestamp]
                        
                        # Calculate consensus for each game
                        consensus = latest_odds.groupby(['away_team', 'home_team']).agg({
                            'away_spread': 'median',
                            'home_spread': 'median',
                            'away_ml': 'median',
                            'home_ml': 'median',
                        }).reset_index()
                        
                        # Join with ESPN to get game status
                        consensus = consensus.merge(
                            latest_espn[['away_team_espn', 'home_team_espn', 'game_status', 'away_score', 'home_score']],
                            left_on=['away_team', 'home_team'],
                            right_on=['away_team_espn', 'home_team_espn'],
                            how='left'
                        )
                        
                        # Split into live vs upcoming
                        # ESPN API game_status: 'pre' = not started, 'in' = live, 'post' = finished
                        live_games = consensus[consensus['game_status'] == 'in']
                        upcoming_games = consensus[consensus['game_status'] != 'in']
                        
                        # Display live games
                        if len(live_games) > 0:
                            print(f"\n  {EMOJI['fire']} LIVE GAMES ({len(live_games)}):")
                            for _, game in live_games.iterrows():
                                matchup = f"{game['away_team']} @ {game['home_team']}"
                                away_spread = f"{game['away_spread']:+.1f}" if pd.notna(game['away_spread']) else "N/A"
                                home_spread = f"{game['home_spread']:+.1f}" if pd.notna(game['home_spread']) else "N/A"
                                away_ml = f"{int(game['away_ml']):+d}" if pd.notna(game['away_ml']) else "N/A"
                                home_ml = f"{int(game['home_ml']):+d}" if pd.notna(game['home_ml']) else "N/A"
                                
                                print(f"    {matchup}")
                                print(f"      Spread: {away_spread} | {home_spread}  |  ML: {away_ml} | {home_ml}")
                                
                                # Show score on separate line if available
                                if pd.notna(game['away_score']) and pd.notna(game['home_score']):
                                    print(f"      {EMOJI['nba']} Score: {int(game['away_score'])}-{int(game['home_score'])}")
                                
                                # Debug: Check how many bookmakers have ML for this game
                                game_odds = latest_odds[
                                    (latest_odds['away_team'] == game['away_team']) & 
                                    (latest_odds['home_team'] == game['home_team'])
                                ]
                                ml_count = game_odds['away_ml'].notna().sum()
                                total_books = len(game_odds)
                                if ml_count == 0:
                                    print(f"      {EMOJI['warning']} No ML offered: 0/{total_books} bookmakers")
                        
                        # Display upcoming games
                        if len(upcoming_games) > 0:
                            print(f"\n  {EMOJI['chart']} UPCOMING GAMES ({len(upcoming_games)}):")
                            for _, game in upcoming_games.iterrows():
                                matchup = f"{game['away_team']} @ {game['home_team']}"
                                away_spread = f"{game['away_spread']:+.1f}" if pd.notna(game['away_spread']) else "N/A"
                                home_spread = f"{game['home_spread']:+.1f}" if pd.notna(game['home_spread']) else "N/A"
                                away_ml = f"{int(game['away_ml']):+d}" if pd.notna(game['away_ml']) else "N/A"
                                home_ml = f"{int(game['home_ml']):+d}" if pd.notna(game['home_ml']) else "N/A"
                                
                                print(f"    {matchup}")
                                print(f"      Spread: {away_spread} | {home_spread}  |  ML: {away_ml} | {home_ml}")
                    except Exception as e:
                        print(f"  {EMOJI['error']} Error displaying consensus: {e}")
                        
            elif results['odds_success'] and results['espn_success']:
                print(f"  {EMOJI['error']} Joined: No matching games (possible team name mismatch)")
            
            print()
            
            # Wait until next wall clock interval
            # This ensures collections happen at :00, :30, etc. (not drifting)
            now = time.time()
            next_run = ((now // args.interval) + 1) * args.interval
            sleep_time = next_run - now
            
            if sleep_time > 0:
                from datetime import datetime as dt
                next_time_utc = dt.fromtimestamp(next_run, tz=ZoneInfo('UTC'))
                next_time_et = next_time_utc.astimezone(ZoneInfo('America/New_York'))
                
                print(f"{EMOJI['time']} Sleeping {sleep_time:.0f}s until next collection at {next_time_et.strftime('%I:%M:%S %p ET')}...\n")
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print(f"\n\n{EMOJI['success']} Collection stopped by user\n")
        
        print(f"{'='*80}")
        print(f"{EMOJI['save']} FINAL DATA SUMMARY")
        print(f"{'='*80}\n")
        
        if odds_filepath.exists():
            odds_df = pd.read_parquet(odds_filepath)
            print(f"File 1 - Odds Data: {len(odds_df)} total records")
            print(f"  Unique games: {odds_df['game_id'].nunique()}")
            print(f"  Unique bookmakers: {odds_df['bookmaker'].nunique()}")
            print(f"  Collection cycles: {odds_df['collection_timestamp'].nunique()}")
        
        if espn_filepath.exists():
            espn_df = pd.read_parquet(espn_filepath)
            print(f"\nFile 2 - ESPN Data: {len(espn_df)} total records")
            print(f"  Unique games: {espn_df['espn_game_id'].nunique()}")
            print(f"  Collection cycles: {espn_df['collection_timestamp'].nunique()}")
        
        if joined_filepath.exists():
            joined_df = pd.read_parquet(joined_filepath)
            print(f"\nFile 3 - Joined Data: {len(joined_df)} total records")
            print(f"  Collection cycles: {joined_df['collection_timestamp'].nunique()}")
        
        print(f"\n{EMOJI['info']} All files saved in: {OUTPUT_DIR}\n")


if __name__ == '__main__':
    main()
