"""
Fetch NBA Player Props (All Markets)

Fetches historical player props from The Odds API for all available markets.
Data is saved to S3 for later analysis.

CRITICAL TIMING NOTE - NBA API Data Delay:
When using --fetch-games flag, be aware of NBA API publishing delays:
  ✅ Props (The Odds API): Available immediately for historical dates
  ❌ Player game logs (NBA API): Takes 12+ HOURS after games finish
  
Example: Games ending at 1am ET on 2026-01-12
  - Props available: Immediately (anytime on 2026-01-12 or later)
  - Game logs available: After 2pm ET on 2026-01-12 (12+ hours later)
  
If you run this script too early, the NBA API will return empty/invalid JSON
and cause timeouts on retries. This is EXPECTED - wait 14+ hours after games finish.

═══════════════════════════════════════════════════════════════════════════════
WORKFLOW: 2025-26 Season Setup
═══════════════════════════════════════════════════════════════════════════════

STEP 1: Prerequisites
---------------------
# A. Fetch NBA calendar (local - used for game dates)
python scripts/nba_calendar_builder.py --season 2025-26

# B. Set API key in .env
export ODDS_API_KEY=your_key_here


STEP 2: Fetch Props (All Markets)
----------------------------------
# Test mode (fetch most recent game date)
python scripts/fetch_nba_player_props.py --mode 1

# Full season (all past dates, props only)
python scripts/fetch_nba_player_props.py --mode 2

# Full season with game results for backtesting
python scripts/fetch_nba_player_props.py --mode 2 --fetch-games


STEP 3: Verify S3 Upload
-------------------------
# Check props data
aws s3 ls s3://the-odds-api-mt/nba/historical_player_props/2025-26/

# Check game results (if --fetch-games used)
aws s3 ls s3://nba-api-mt/player_game_logs/2025-26/


═══════════════════════════════════════════════════════════════════════════════
S3 STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

Props (OUTPUT):
    s3://the-odds-api-mt/nba/historical_player_props/2025-26/
        ├── 2025-01-04.csv
        ├── 2025-01-05.csv
        └── ...

Game Results (OUTPUT if --fetch-games):
    s3://nba-api-mt/player_game_logs/2025-26/
        ├── 2025-01-04.csv
        ├── 2025-01-05.csv
        └── ...


═══════════════════════════════════════════════════════════════════════════════
DATA COLUMNS
═══════════════════════════════════════════════════════════════════════════════

Props CSV (ALL MARKETS):
    - player, away_team, home_team, game_time
    - market (player_points, player_rebounds, player_assists, player_threes, etc.)
    - prop_line, over_odds, under_odds, bookmaker
    - bookmaker_last_update, market_last_update
    - fetch_date, season

Game Results CSV (if --fetch-games):
    - PLAYER_ID, PLAYER_NAME, TEAM_NAME, GAME_DATE, MATCHUP
    - PTS, REB, AST, FG3M (actual stats for all markets)
    - MIN, FGM, FGA, WL, PLUS_MINUS


═══════════════════════════════════════════════════════════════════════════════
COST ESTIMATION
═══════════════════════════════════════════════════════════════════════════════

Per game date:
    - 1 credit for event list
    - ~10 credits per game per market batch
    - 9 markets fetched (points, rebounds, assists, threes, blocks, steals, etc.)
    - Typical day (10-15 games) = ~100-150 credits per game date

Full season (2025-26):
    - ~163 game dates
    - Estimated: 16,000-24,000 credits (~$20-30 for 500k plan)


Created: 2025-01-05
Author: Thomas Myles
"""

import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pathlib import Path
import time
import logging
import ssl
import urllib3
from zoneinfo import ZoneInfo
import boto3
from io import StringIO
import sys
from nba_api.stats.endpoints import playergamelogs #leaguegamefinder is wrong

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from season_utils import get_current_nba_season

# Load environment variables
load_dotenv()

# Add src to path for config loader
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(description='Fetch NBA player props (all markets) from The Odds API')
parser.add_argument('--season', type=str, default='2025-26',
                    help='Season to fetch (e.g., 2025-26)')
parser.add_argument('--mode', type=int, choices=[1, 2], default=None,
                    help='Mode: 1=test (one recent date), 2=full season')
parser.add_argument('--date', type=str, default=None,
                    help='Specific date to fetch (YYYY-MM-DD)')
parser.add_argument('--s3', action='store_true',
                    help='Upload to S3 (default: True for full season mode)')
parser.add_argument('--fetch-games', action='store_true',
                    help='Also fetch player game results from NBA API and save to nba-api-mt')
args = parser.parse_args()

# ============================================================================
# SSL FIX FOR MACOS
# ============================================================================
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests with timeout
original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    kwargs.setdefault('timeout', 120)  # 120 second timeout for NBA API
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(log_prefix='fetch_player_props'):
    """Configure logging to file and console"""
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_filename = f"{log_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = log_dir / log_filename
    
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized - log file: {log_filepath}")
    return log_filepath

setup_logging()

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
API_KEY = os.getenv('ODDS_API_KEY') or os.getenv('THE_ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4'
SPORT_KEY = 'basketball_nba'
DEFAULT_REGION = 'us'
DEFAULT_MARKETS = 'player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals,player_double_double,player_triple_double,player_points_rebounds_assists'  # All available markets
ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'
SEASON = args.season
PROJECT_ROOT = Path(__file__).parent.parent
RATE_LIMIT_DELAY = 0.6  # seconds between API calls

# S3 Configuration - Props
S3_BUCKET_PROPS = 'the-odds-api-mt'
S3_PREFIX_PROPS = f'nba/historical_player_props/{SEASON}'

# S3 Configuration - Game Results
S3_BUCKET_GAMES = 'nba-api-mt'
S3_PREFIX_GAMES = f'player_game_logs/{SEASON}'

# Local paths (for NBA calendar only)
NBA_CALENDAR_PATH = PROJECT_ROOT / 'data' / '01_input' / 'nba_calendar'

# Timestamps
EVENT_LIST_HOUR = 12  # Noon UTC to get day's games
ODDS_SNAPSHOT_HOUR = 15  # 3 PM UTC (10 AM ET)

# API usage tracking
credits_remaining = None
credits_used = None

# Log configuration
logging.info("="*80)
logging.info("CONFIGURATION")
logging.info("="*80)
logging.info(f"API_KEY: {'*' * 8 + API_KEY[-4:] if API_KEY and len(API_KEY) > 4 else 'NOT SET'}")
logging.info(f"SEASON: {SEASON}")
logging.info(f"MARKETS: {DEFAULT_MARKETS}")
logging.info(f"S3_BUCKET_PROPS: {S3_BUCKET_PROPS}")
logging.info(f"S3_PREFIX_PROPS: {S3_PREFIX_PROPS}")
logging.info(f"S3_BUCKET_GAMES: {S3_BUCKET_GAMES}")
logging.info(f"S3_PREFIX_GAMES: {S3_PREFIX_GAMES}")
logging.info(f"FETCH_GAMES: {args.fetch_games}")
logging.info(f"OUTPUT: S3 only (no local files)")
logging.info("="*80)


# ============================================================================
# API FUNCTIONS (from fetch_historical_nba_prop_markets.py)
# ============================================================================

def get_historical_events(date_str, sport=SPORT_KEY):
    """Get list of events for a specific date"""
    global credits_remaining, credits_used
    
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    timestamp = date_obj.replace(hour=EVENT_LIST_HOUR, minute=0, second=0).isoformat() + 'Z'
    
    endpoint = f'historical/sports/{sport}/events'
    params = {
        'api_key': API_KEY,
        'date': timestamp,
        'dateFormat': DATE_FORMAT
    }
    
    logging.info(f"Fetching events for {date_str} (timestamp: {timestamp})")
    
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        
        credits_remaining = int(float(response.headers.get('x-requests-remaining', 0)))
        credits_used = int(float(response.headers.get('x-requests-used', 0)))
        last_cost = int(float(response.headers.get('x-requests-last', 0)))
        
        logging.info(f"API call successful - Cost: {last_cost} credits, Remaining: {credits_remaining:,}")
        
        data = response.json()
        
        if 'data' in data:
            events = data['data']
            logging.info(f"Found {len(events)} events for {date_str}")
            return events
        else:
            logging.warning(f"No events found for {date_str}")
            return []
            
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error fetching events for {date_str}: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error fetching events: {e}", exc_info=True)
        return []


def get_historical_event_odds(sport, event_id, date_str, markets=DEFAULT_MARKETS, regions=DEFAULT_REGION):
    """Get historical odds for a specific event"""
    global credits_remaining, credits_used
    
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    timestamp = date_obj.replace(hour=ODDS_SNAPSHOT_HOUR, minute=0, second=0).isoformat() + 'Z'
    
    endpoint = f'historical/sports/{sport}/events/{event_id}/odds'
    params = {
        'api_key': API_KEY,
        'date': timestamp,
        'regions': regions,
        'markets': markets,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FORMAT
    }
    
    logging.debug(f"Fetching odds for event {event_id[:8]} - markets: {markets}")
    
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        
        credits_remaining = int(float(response.headers.get('x-requests-remaining', 0)))
        credits_used = int(float(response.headers.get('x-requests-used', 0)))
        last_cost = int(float(response.headers.get('x-requests-last', 0)))
        
        logging.info(f"Event {event_id[:8]} - Cost: {last_cost} credits, Remaining: {credits_remaining:,}")
        
        data = response.json()
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
        
        return data
        
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error fetching odds for event {event_id[:8]}: {e}")
        if e.response.status_code == 422:
            logging.warning(f"Props not available for event {event_id[:8]} at date {date_str}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching odds for event {event_id[:8]}: {e}", exc_info=True)
        return None


def parse_player_props(odds_data):
    """Parse player props from odds data (all markets)"""
    if not odds_data or 'data' not in odds_data:
        return []
    
    event_data = odds_data['data']
    props_list = []
    
    away_team = event_data.get('away_team')
    home_team = event_data.get('home_team')
    game_time = event_data.get('commence_time')
    
    for bookmaker in event_data.get('bookmakers', []):
        bookmaker_name = bookmaker['key']
        bookmaker_last_update = bookmaker['last_update']
        
        for market in bookmaker.get('markets', []):
            market_key = market['key']
            market_last_update = market['last_update']
            
            # Group outcomes by player, market, and line
            player_line_props = {}
            for outcome in market.get('outcomes', []):
                player = outcome.get('description', 'Unknown')
                line = outcome.get('point')
                odds = outcome.get('price')
                bet_type = outcome.get('name')
                
                # Key by player, market, and line
                key = (player, market_key, line)
                
                if key not in player_line_props:
                    player_line_props[key] = {
                        'player': player,
                        'away_team': away_team,
                        'home_team': home_team,
                        'game_time': game_time,
                        'market': market_key,
                        'prop_line': line,
                        'bookmaker': bookmaker_name,
                        'bookmaker_last_update': bookmaker_last_update,
                        'market_last_update': market_last_update
                    }
                
                if bet_type == 'Over':
                    player_line_props[key]['over_odds'] = odds
                elif bet_type == 'Under':
                    player_line_props[key]['under_odds'] = odds
            
            props_list.extend(player_line_props.values())
    
    return props_list


# ============================================================================
# NBA API GAME RESULTS FUNCTIONS
# ============================================================================

def fetch_games_for_date(date_str, max_retries=3):
    """
    Fetch player game results for a specific date from NBA API
    
    IMPORTANT: NBA API player game logs have a 12+ HOUR publishing delay!
    If games ended at 1am ET, data won't be available until ~2pm ET same day.
    Running this too early will result in JSONDecodeError or empty results.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        max_retries: Number of retry attempts on timeout/connection errors
    
    Returns:
        DataFrame with player game logs for that date
    """
    logging.info(f"📡 Fetching NBA game results for {date_str}...")
    
    # Retry logic for flaky NBA API
    for attempt in range(max_retries):
        try:
            # Parse date
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Fetch player game logs for the season
            season_str = SEASON  # e.g., '2025-26'
            
            game_logs = playergamelogs.PlayerGameLogs(
                season_nullable=season_str,
                season_type_nullable='Regular Season',
                date_from_nullable=date_str,
                date_to_nullable=date_str
            )
            
            games = game_logs.get_data_frames()[0]
            
            if games.empty:
                logging.info(f"   No games found for {date_str}")
                return pd.DataFrame()
            
            # Key columns for our analysis (only keep columns that exist)
            cols_to_keep = [
                'SEASON_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME',
                'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL',
                'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
                'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS'
            ]
            
            # Only select columns that actually exist in the dataframe
            available_cols = [col for col in cols_to_keep if col in games.columns]
            
            if not available_cols:
                logging.warning(f"   ⚠️  No expected columns found. Available columns: {list(games.columns)}")
                # Return all columns if none of our expected ones exist
                pass
            else:
                games = games[available_cols]
            
            num_games = games['GAME_ID'].nunique()
            num_players = len(games)
            
            logging.info(f"   ✅ Found {num_games} games with {num_players} player performances")
            
            # Rate limiting for NBA API
            time.sleep(RATE_LIMIT_DELAY)
            
            return games
            
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                logging.warning(f"   ⚠️  Request timed out (attempt {attempt + 1}/{max_retries})")
                logging.warning(f"   Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logging.error(f"   ❌ Failed after {max_retries} attempts: {e}", exc_info=True)
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"❌ Error fetching games for {date_str}: {e}", exc_info=True)
            return pd.DataFrame()


# ============================================================================
# S3 FUNCTIONS
# ============================================================================

def check_s3_file_exists(bucket, key):
    """Check if a file exists in S3"""
    try:
        s3_client = boto3.client('s3')
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False


def save_props_to_s3(df, date_str):
    """Save props DataFrame to the-odds-api-mt bucket"""
    s3_key = f"{S3_PREFIX_PROPS}/{date_str}.csv"
    
    try:
        s3_client = boto3.client('s3')
        
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_client.put_object(
            Bucket=S3_BUCKET_PROPS,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        logging.info(f"✅ Saved props to S3: s3://{S3_BUCKET_PROPS}/{s3_key}")
        return s3_key
        
    except Exception as e:
        logging.warning(f"⚠️  Failed to save props to S3: {e}")
        return None


def save_games_to_s3(df, date_str):
    """Save game results DataFrame to nba-api-mt bucket"""
    s3_key = f"{S3_PREFIX_GAMES}/{date_str}.csv"
    
    try:
        s3_client = boto3.client('s3')
        
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_client.put_object(
            Bucket=S3_BUCKET_GAMES,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        logging.info(f"✅ Saved game results to S3: s3://{S3_BUCKET_GAMES}/{s3_key}")
        return s3_key
        
    except Exception as e:
        logging.warning(f"⚠️  Failed to save game results to S3: {e}")
        return None


# ============================================================================
# MAIN FETCH FUNCTION
# ============================================================================

def fetch_date_props(date_str, upload_s3=True, fetch_games=False, skip_if_exists=True):
    """
    Fetch player props for a specific date
    
    Args:
        date_str: Date in YYYY-MM-DD format
        upload_s3: Upload to S3 (default: True)
        fetch_games: Also fetch game results from NBA API
        skip_if_exists: Skip fetching if files exist in S3 (default: True)
    
    Returns:
        Tuple of (props_df, games_df)
    """
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    day_of_week = date_obj.strftime('%A')
    
    # Check if props already exist in S3
    props_s3_key = f"{S3_PREFIX_PROPS}/{date_str}.csv"
    props_exist = skip_if_exists and upload_s3 and check_s3_file_exists(S3_BUCKET_PROPS, props_s3_key)
    
    if props_exist:
        logging.info("="*80)
        logging.info(f"⏭️  PROPS ALREADY EXIST FOR {date_str} - SKIPPING PROPS FETCH")
        logging.info("="*80)
        df = pd.DataFrame()  # Empty dataframe since we're not fetching
    else:
        logging.info("="*80)
        logging.info(f"FETCHING PROPS FOR {date_str} ({day_of_week})")
        logging.info("="*80)
        
        # Get events for that date
        all_events = get_historical_events(date_str)
        
        if not all_events:
            logging.error(f"No events found for {date_str}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Filter to games that start on this date (ET timezone)
        et_tz = ZoneInfo('America/New_York')
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        start_of_window_et = datetime(date_obj.year, date_obj.month, date_obj.day, 6, 0, 0, tzinfo=et_tz)
        end_of_window_et = datetime(date_obj.year, date_obj.month, date_obj.day, 23, 59, 59, tzinfo=et_tz)
        
        events = []
        for event in all_events:
            commence_time_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            commence_time_et = commence_time_utc.astimezone(et_tz)
            
            if start_of_window_et <= commence_time_et <= end_of_window_et:
                events.append(event)
        
        logging.info(f"Found {len(events)} games that start on {date_str}")
        
        if not events:
            logging.warning(f"No games actually start on {date_str}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Fetch props for each event
        all_props = []
        for i, event in enumerate(events, 1):
            game_desc = f"{event['away_team']} @ {event['home_team']}"
            logging.info(f"Processing game {i}/{len(events)}: {game_desc}")
            
            odds_data = get_historical_event_odds(
                sport=SPORT_KEY,
                event_id=event['id'],
                date_str=date_str,
                markets=DEFAULT_MARKETS
            )
            
            if odds_data:
                props = parse_player_props(odds_data)
                all_props.extend(props)
                logging.info(f"  ✅ Found {len(props)} player props")
            else:
                logging.warning(f"  ⚠️  No props available")
        
        if not all_props:
            logging.warning(f"No props data collected for {date_str}")
            return pd.DataFrame(), pd.DataFrame()
        
        df = pd.DataFrame(all_props)
        
        # Add metadata
        df['fetch_date'] = datetime.now().isoformat()
        df['season'] = SEASON
        
        # Show summary
        logging.info("="*80)
        logging.info("SUMMARY:")
        logging.info(f"  Total props: {len(df)}")
        logging.info(f"  Unique players: {df['player'].nunique()}")
        logging.info(f"  Markets: {df['market'].nunique()}")
        logging.info(f"  Bookmakers: {df['bookmaker'].nunique()}")
        
        # Show market breakdown
        market_counts = df['market'].value_counts()
        logging.info("\nProps by market:")
        for market, count in market_counts.items():
            logging.info(f"  {market}: {count}")
        
        logging.info("="*80)
        
        # Upload to S3
        if upload_s3:
            save_props_to_s3(df, date_str)
        else:
            logging.info("⚠️  Skipping S3 upload (--no-s3 flag)")
    
    # Fetch game results if requested (independent of props fetch)
    games_df = pd.DataFrame()
    if fetch_games:
        # Check if game logs already exist in S3
        games_s3_key = f"{S3_PREFIX_GAMES}/{date_str}.csv"
        if skip_if_exists and upload_s3 and check_s3_file_exists(S3_BUCKET_GAMES, games_s3_key):
            logging.info("")
            logging.info("="*80)
            logging.info(f"⏭️  GAME RESULTS ALREADY EXIST FOR {date_str} - SKIPPING GAME FETCH")
            logging.info("="*80)
        else:
            logging.info("")
            logging.info("="*80)
            logging.info(f"FETCHING GAME RESULTS FOR {date_str}")
            logging.info("="*80)
            
            games_df = fetch_games_for_date(date_str)
            
            if not games_df.empty:
                # Upload to S3
                if upload_s3:
                    save_games_to_s3(games_df, date_str)
                else:
                    logging.info("⚠️  Skipping game results S3 upload")
            else:
                error_msg = f"❌ CRITICAL: Game fetch returned 0 rows for {date_str}"
                logging.error(error_msg)
                logging.error(f"Expected games file: s3://{S3_BUCKET_GAMES}/{games_s3_key}")
                # Exit with error code to signal failure to Lambda
                sys.exit(1)
    
    return df, games_df


# ============================================================================
# FULL SEASON MODE
# ============================================================================

def check_past_season_complete(season, expected_game_dates):
    """
    Check if a past season is already complete in S3.
    
    Args:
        season: NBA season (e.g., "2024-25")
        expected_game_dates: Number of expected game dates
    
    Returns:
        tuple: (is_complete: bool, props_found: int, gamelogs_found: int)
    """
    s3 = boto3.client('s3')
    
    # Check player props
    props_prefix = f"nba/historical_player_props/{season}/"
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_PROPS, Prefix=props_prefix)
        props_files = 0
        if 'Contents' in response:
            props_files = len([obj for obj in response['Contents'] if obj['Key'].endswith('.csv')])
    except Exception as e:
        logging.warning(f"Error checking props S3: {e}")
        props_files = 0
    
    # Check game logs
    gamelogs_prefix = f"player_game_logs/{season}/"
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_GAMES, Prefix=gamelogs_prefix)
        gamelogs_files = 0
        if 'Contents' in response:
            gamelogs_files = len([obj for obj in response['Contents'] if obj['Key'].endswith('.csv')])
    except Exception as e:
        logging.warning(f"Error checking gamelogs S3: {e}")
        gamelogs_files = 0
    
    # Consider complete if both have at least the expected number of files
    is_complete = props_files >= expected_game_dates and gamelogs_files >= expected_game_dates
    
    return is_complete, props_files, gamelogs_files


def fetch_full_season(upload_s3=True, fetch_games=False):
    """Fetch props for all dates in season"""
    logging.info("="*80)
    logging.info("FULL SEASON FETCH MODE")
    logging.info("="*80)
    logging.info(f"Upload to S3: {upload_s3}")
    logging.info(f"Fetch game results: {fetch_games}")
    
    # Load games CSV
    games_csv = NBA_CALENDAR_PATH / f'all_games_{SEASON.replace("-", "_")}.csv'
    
    if not games_csv.exists():
        logging.error(f"Games CSV not found: {games_csv}")
        logging.error("Run nba_calendar_builder.py first")
        return None
    
    games_df = pd.read_csv(games_csv)
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE']).dt.date
    game_dates = sorted(games_df['GAME_DATE'].unique())
    
    logging.info(f"Season has {len(game_dates)} game dates")
    
    # Filter to past dates only
    today = datetime.now().date()
    past_dates = [d for d in game_dates if d < today]
    
    logging.info(f"Past dates to fetch: {len(past_dates)}")
    logging.info(f"Future dates: {len(game_dates) - len(past_dates)}")
    
    # Check if past season and complete
    current_season = get_current_nba_season()
    is_past_season = SEASON < current_season
    
    if is_past_season:
        logging.info(f"\n📅 Checking if past season {SEASON} is already complete...")
        is_complete, props_found, gamelogs_found = check_past_season_complete(SEASON, len(past_dates))
        
        if is_complete:
            logging.info("="*80)
            logging.info("✅ PAST SEASON COMPLETE - SKIPPING")
            logging.info("="*80)
            logging.info(f"Season: {SEASON}")
            logging.info(f"Props files: {props_found}/{len(past_dates)} in S3")
            logging.info(f"Gamelogs files: {gamelogs_found}/{len(past_dates)} in S3")
            logging.info(f"Props S3: s3://{S3_BUCKET_PROPS}/nba/historical_player_props/{SEASON}/")
            logging.info(f"Gamelogs S3: s3://{S3_BUCKET_GAMES}/player_game_logs/{SEASON}/")
            logging.info("\nNo fetch needed - all historical data exists!")
            logging.info("="*80)
            return {'skipped': True, 'reason': 'Past season complete', 'props_found': props_found, 'gamelogs_found': gamelogs_found}
        else:
            logging.info(f"   Props: {props_found}/{len(past_dates)}, Gamelogs: {gamelogs_found}/{len(past_dates)} - will fetch missing dates")
    else:
        logging.info(f"\n🔄 Current season {SEASON} - checking for updates...")
    
    # Track stats
    stats = {
        'processed': 0,
        'successful': 0,
        'skipped_existing': 0,
        'failed': 0,
        'total_props': 0
    }
    
    for i, date_obj in enumerate(past_dates, 1):
        date_str = date_obj.strftime('%Y-%m-%d')
        day_of_week = date_obj.strftime('%A')
        
        logging.info(f"\n{'='*80}")
        logging.info(f"DATE {i}/{len(past_dates)}: {date_str} ({day_of_week})")
        logging.info(f"Progress: {(i/len(past_dates))*100:.1f}%")
        logging.info(f"{'='*80}")
        
        # Fetch (will upload to S3, skips internally if files exist)
        try:
            props_df, games_df = fetch_date_props(date_str, upload_s3=upload_s3, fetch_games=fetch_games)
            
            if not props_df.empty:
                stats['successful'] += 1
                stats['total_props'] += len(props_df)
            
            stats['processed'] += 1
            
        except KeyboardInterrupt:
            logging.warning("\n⚠️  User interrupted (Ctrl+C)")
            break
        except Exception as e:
            logging.error(f"❌ Error processing {date_str}: {e}", exc_info=True)
            stats['failed'] += 1
            break
    
    # Summary
    logging.info("\n" + "="*80)
    logging.info("FULL SEASON SUMMARY")
    logging.info("="*80)
    logging.info(f"Processed: {stats['processed']}")
    logging.info(f"Successful: {stats['successful']}")
    logging.info(f"Skipped (existing): {stats['skipped_existing']}")
    logging.info(f"Failed: {stats['failed']}")
    logging.info(f"Total props collected: {stats['total_props']:,}")
    logging.info("="*80)
    
    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    logging.info("="*80)
    logging.info("NBA PLAYER PROPS FETCHER (ALL MARKETS)")
    logging.info("="*80)
    
    if not API_KEY:
        logging.error("API key not configured! Set ODDS_API_KEY in .env")
        return
    
    # Specific date mode
    if args.date:
        date_str = args.date
        logging.info(f"Fetching specific date: {date_str}")
        upload_s3 = True if args.s3 is None else args.s3
        props_df, games_df = fetch_date_props(date_str, upload_s3=upload_s3, fetch_games=args.fetch_games)
        if not props_df.empty:
            logging.info("✅ Props fetch complete!")
        if not games_df.empty:
            logging.info("✅ Game results fetch complete!")
        return
    
    # Get mode
    if args.mode is not None:
        choice = str(args.mode)
    else:
        print("\n" + "="*80)
        print("MODE SELECTION")
        print("="*80)
        print("1. Test mode (fetch most recent game date)")
        print("2. Full season mode (fetch all past dates)")
        print("="*80)
        
        while True:
            choice = input("Select mode (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")
    
    if choice == '1':
        # Test mode - get most recent game date
        games_csv = NBA_CALENDAR_PATH / f'all_games_{SEASON.replace("-", "_")}.csv'
        
        if not games_csv.exists():
            logging.error(f"Games CSV not found: {games_csv}")
            return
        
        games_df = pd.read_csv(games_csv)
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE']).dt.date
        
        # Get most recent past date
        today = datetime.now().date()
        past_dates = games_df[games_df['GAME_DATE'] < today]['GAME_DATE']
        
        if len(past_dates) == 0:
            logging.error("No past game dates found")
            return
        
        test_date = past_dates.max()
        test_date_str = test_date.strftime('%Y-%m-%d')
        
        logging.info(f"TEST MODE: Fetching {test_date_str}")
        
        upload_s3 = True if args.s3 is None else args.s3
        props_df, games_df = fetch_date_props(test_date_str, upload_s3=upload_s3, fetch_games=args.fetch_games)
        
        if not props_df.empty:
            logging.info("="*80)
            logging.info("✅ Test successful! Ready for full season.")
            logging.info("="*80)
        if not games_df.empty:
            logging.info("✅ Game results also fetched successfully!")
    else:
        # Full season mode
        logging.info("FULL SEASON MODE selected")
        upload_s3 = args.s3 if args.s3 is not None else True  # Default to True for full season
        stats = fetch_full_season(upload_s3=upload_s3, fetch_games=args.fetch_games)
        
        if stats:
            if stats.get('skipped'):
                logging.info(f"\n✅ Season {SEASON} already complete - skipped fetching")
            elif stats.get('successful', 0) > 0:
                logging.info("\n✅ Full season fetch completed!")


if __name__ == "__main__":
    main()

