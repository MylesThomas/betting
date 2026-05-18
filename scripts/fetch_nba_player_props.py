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
# A. Ensure season dates are in config/season_dates.yaml
#    Script uses season_start and regular_season_end to generate date range

# B. Set API key in .env
export ODDS_API_KEY=your_key_here


STEP 2: Fetch Props (All Markets)
----------------------------------
# Test mode (fetch most recent game date)
python scripts/fetch_nba_player_props.py --mode 1 --season 2025-26

# Full season (all past dates, props only)
python scripts/fetch_nba_player_props.py --mode 2 --season 2025-26 --s3

# Full season with game results for backtesting
python scripts/fetch_nba_player_props.py --mode 2 --fetch-games --s3 --season 2025-26

NOTE: --fetch-games now fetches BOTH game results AND game lines:
  1. Player props (The Odds API) → the-odds-api-mt/nba/historical_player_props/
  2. Game results (NBA API) → nba-api-mt/player_game_logs/
  3. Game lines (The Odds API) → the-odds-api-mt/nba/historical_game_lines/


STEP 3: Verify S3 Upload
-------------------------
# Check props data
aws s3 ls s3://the-odds-api-mt/nba/historical_player_props/2025-26/

# Check game results (if --fetch-games used)
aws s3 ls s3://nba-api-mt/player_game_logs/2025-26/


═══════════════════════════════════════════════════════════════════════════════
CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════

Season Dates (config/season_dates.yaml):
    - season_start: First day of regular season
    - regular_season_end: Last day of regular season
    - Script generates all dates in this range and fetches for each
    - Days with no games will return empty results (e.g., All-Star Sunday)


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

Game Lines (OUTPUT if --fetch-games):
    s3://the-odds-api-mt/nba/historical_game_lines/2025-26/
        ├── nba_game_lines_2025-01-04.csv
        ├── nba_game_lines_2025-01-05.csv
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
    - If --fetch-games: +20 credits per game for game lines (h2h + spreads)
    - Typical day (10-15 games) = ~100-150 credits for props + ~200-300 for game lines
    - Total with --fetch-games: ~300-450 credits per game date

Full season (2025-26):
    - ~163 game dates
    - Props only: 16,000-24,000 credits
    - With --fetch-games: 48,000-73,000 credits (~$50-75 for 500k plan)


Created: 2025-01-05
Updated: 2026-02-10 (switched to config-based date range, removed calendar dependency)
Author: Thomas Myles
"""

import argparse
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
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
import yaml
from nba_api.stats.endpoints import playergamelogs #leaguegamefinder is wrong
from functools import wraps

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from season_utils import get_current_nba_season

# Import game lines fetching functions
sys.path.insert(0, str(project_root / 'scripts'))
from fetch_historical_nba_season_lines import (
    fetch_date_lines,
    file_exists_in_s3 as game_lines_file_exists_in_s3,
    get_s3_key_from_date as get_game_lines_s3_key,
    S3_BUCKET as GAME_LINES_S3_BUCKET
)

# Load environment variables
load_dotenv()

# Add src to path for config loader
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# ============================================================================
# TIMING DECORATOR
# ============================================================================

def timed(func):
    """Decorator to time function execution and log results"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_time = time.time()
        logging.info(f"⏱️  [{func_name}] Starting...")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logging.info(f"⏱️  [{func_name}] Completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logging.error(f"⏱️  [{func_name}] Failed after {elapsed:.2f}s: {e}")
            raise
    
    return wrapper


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
                    help='Also fetch game results (NBA API) AND game lines (The Odds API)')
parser.add_argument('--force', action='store_true',
                    help='Force overwrite existing S3 files (skip existence check)')
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
    kwargs.setdefault('timeout', 3)  # 3 second timeout - fail fast
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

# ============================================================================
# ESPN API HELPER FUNCTIONS (Fallback for NBA API when Lambda IPs blocked)
# ============================================================================

def parse_made(fg_string):
    """Parse '7-12' to get made (7)"""
    if not fg_string or '-' not in str(fg_string):
        return 0
    return int(str(fg_string).split('-')[0])


def parse_attempts(fg_string):
    """Parse '7-12' to get attempts (12)"""
    if not fg_string or '-' not in str(fg_string):
        return 0
    return int(str(fg_string).split('-')[1])


def calculate_pct(fg_string):
    """Parse '7-12' to calculate percentage (0.583)"""
    made = parse_made(fg_string)
    attempts = parse_attempts(fg_string)
    if attempts == 0:
        return 0.0
    return round(made / attempts, 3)


def safe_int(value):
    """Safely parse int, handling empty/None/dash values"""
    if not value or value == '':
        return 0
    try:
        # If it contains a dash, take the first number (made, not attempted)
        if '-' in str(value):
            return parse_made(value)
        return int(value)
    except (ValueError, TypeError):
        return 0


def parse_minutes(min_string):
    """Parse '35:24' to get total minutes as float (35.4)"""
    if not min_string or ':' not in str(min_string):
        return 0.0
    try:
        parts = str(min_string).split(':')
        minutes = int(parts[0])
        seconds = int(parts[1]) if len(parts) > 1 else 0
        return round(minutes + seconds / 60, 2)
    except (ValueError, TypeError):
        return 0.0


def parse_espn_box_score(box_data, date_str, away_abbr, home_abbr):
    """
    Parse ESPN box score into NBA API format (29 key columns).
    
    ESPN stats order: [0]=MIN, [1]=PTS, [2]=FG, [3]=3PT, [4]=FT, [5]=REB, [6]=AST,
                      [7]=TO, [8]=STL, [9]=BLK, [10]=OREB, [11]=DREB, [12]=PF, [13]=+/-
    """
    players = []
    
    boxscore = box_data.get('boxscore', {})
    if not boxscore:
        return players
    
    teams_data = boxscore.get('players', [])
    
    for team_data in teams_data:
        team_info = team_data.get('team', {})
        team_abbr = team_info.get('abbreviation', '')
        team_name = team_info.get('displayName', '')
        
        # Determine if home or away
        is_home = (team_abbr == home_abbr)
        matchup = f"{team_abbr} vs. {away_abbr}" if is_home else f"{team_abbr} @ {home_abbr}"
        
        # Get statistics section
        statistics = team_data.get('statistics', [])
        if not statistics:
            continue
        
        # Usually statistics[0] has the player stats
        stat_section = statistics[0]
        athletes = stat_section.get('athletes', [])
        
        for athlete_data in athletes:
            athlete = athlete_data.get('athlete', {})
            stats = athlete_data.get('stats', [])
            
            if not stats:
                continue
            
            # ESPN stats order (confirmed from API labels):
            # [0]=MIN, [1]=PTS, [2]=FG, [3]=3PT, [4]=FT, [5]=REB, [6]=AST,
            # [7]=TO, [8]=STL, [9]=BLK, [10]=OREB, [11]=DREB, [12]=PF, [13]=+/-
            
            player_dict = {
                # IDs and names
                'PLAYER_ID': athlete.get('id', ''),
                'PLAYER_NAME': athlete.get('displayName', ''),
                'TEAM_ID': team_info.get('id', ''),
                'TEAM_NAME': team_name,
                'TEAM_ABBREVIATION': team_abbr,
                
                # Game info
                'GAME_ID': box_data.get('header', {}).get('id', ''),
                'GAME_DATE': date_str,
                'MATCHUP': matchup,
                
                # Stats (using correct ESPN order)
                'MIN': parse_minutes(stats[0]) if len(stats) > 0 else 0,
                'PTS': safe_int(stats[1]) if len(stats) > 1 else 0,
                'REB': safe_int(stats[5]) if len(stats) > 5 else 0,
                'AST': safe_int(stats[6]) if len(stats) > 6 else 0,
                'TOV': safe_int(stats[7]) if len(stats) > 7 else 0,
                'STL': safe_int(stats[8]) if len(stats) > 8 else 0,
                'BLK': safe_int(stats[9]) if len(stats) > 9 else 0,
                'OREB': safe_int(stats[10]) if len(stats) > 10 else 0,
                'DREB': safe_int(stats[11]) if len(stats) > 11 else 0,
                'PF': safe_int(stats[12]) if len(stats) > 12 else 0,
                'PLUS_MINUS': safe_int(stats[13]) if len(stats) > 13 else 0,
                
                # Field goals (index 2: "7-12" format)
                'FGM': parse_made(stats[2]) if len(stats) > 2 else 0,
                'FGA': parse_attempts(stats[2]) if len(stats) > 2 else 0,
                'FG_PCT': calculate_pct(stats[2]) if len(stats) > 2 else 0.0,
                
                # 3-pointers (index 3: "2-5" format)
                'FG3M': parse_made(stats[3]) if len(stats) > 3 else 0,
                'FG3A': parse_attempts(stats[3]) if len(stats) > 3 else 0,
                'FG3_PCT': calculate_pct(stats[3]) if len(stats) > 3 else 0.0,
                
                # Free throws (index 4: "4-4" format)
                'FTM': parse_made(stats[4]) if len(stats) > 4 else 0,
                'FTA': parse_attempts(stats[4]) if len(stats) > 4 else 0,
                'FT_PCT': calculate_pct(stats[4]) if len(stats) > 4 else 0.0,
                
                # Win/Loss (placeholder)
                'WL': 'TBD',
            }
            
            players.append(player_dict)
    
    return players


def fetch_games_from_espn(date_str):
    """
    Fetch game results from ESPN API (fallback when NBA API blocked).
    Returns DataFrame with same 29 columns as NBA API.
    """
    logging.info(f"   📡 ESPN API: Fetching scoreboard...")
    
    # Convert date format: 2026-02-12 → 20260212
    espn_date = date_str.replace('-', '')
    
    # Get scoreboard
    scoreboard_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={espn_date}"
    response = requests.get(scoreboard_url, timeout=10)
    response.raise_for_status()
    scoreboard = response.json()
    
    events = scoreboard.get('events', [])
    logging.info(f"   📡 ESPN API: Found {len(events)} games")
    
    if not events:
        return pd.DataFrame()
    
    # Fetch box score for each game
    all_players = []
    
    for i, event in enumerate(events, 1):
        game_id = event['id']
        competition = event['competitions'][0]
        
        # Get team info
        away_team = competition['competitors'][1]  # Away is index 1
        home_team = competition['competitors'][0]  # Home is index 0
        
        away_abbr = away_team['team']['abbreviation']
        home_abbr = home_team['team']['abbreviation']
        
        logging.debug(f"   ESPN game {i}/{len(events)}: {away_abbr} @ {home_abbr}")
        
        # Fetch detailed box score
        box_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
        box_response = requests.get(box_url, timeout=10)
        box_response.raise_for_status()
        box_data = box_response.json()
        
        # Parse players
        players = parse_espn_box_score(box_data, date_str, away_abbr, home_abbr)
        all_players.extend(players)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_players)
    
    return df


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# CRITICAL: Print before setup_logging() to debug Lambda hanging
logging.debug("DEBUG: Script reached logging section")

def setup_logging(log_prefix='fetch_player_props'):
    """Configure logging to file and console"""
    logging.debug(f"DEBUG: setup_logging() called with prefix={log_prefix}")
    log_dir = Path(__file__).parent.parent / 'logs'
    logging.debug(f"DEBUG: log_dir={log_dir}")
    log_dir.mkdir(exist_ok=True)
    logging.debug(f"DEBUG: log_dir created")
    
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
    logging.debug(f"DEBUG: Logging initialized successfully")
    return log_filepath

logging.debug("DEBUG: About to call setup_logging()")
setup_logging()
logging.debug("DEBUG: setup_logging() completed")

# Print to stdout for Lambda visibility (logging goes to file)
logging.info(f"🚀 Script starting at {datetime.now().isoformat()}")
logging.info(f"📍 Working directory: {os.getcwd()}")

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
logging.debug("DEBUG: Loading API_KEY")
API_KEY = os.getenv('ODDS_API_KEY') or os.getenv('THE_ODDS_API_KEY')
logging.info(f"🔑 API_KEY loaded: {'Yes' if API_KEY else 'No'}")
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

# Local paths
SEASON_DATES_CONFIG_PATH = PROJECT_ROOT / 'config' / 'season_dates.yaml'

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
# CONFIG LOADING FUNCTIONS
# ============================================================================

def load_season_dates_config():
    """
    Load season dates from YAML config file.
    
    Returns:
        Dict with NBA season dates
    """
    if not SEASON_DATES_CONFIG_PATH.exists():
        logging.warning(f"Season dates config not found: {SEASON_DATES_CONFIG_PATH}")
        return {}
    
    with open(SEASON_DATES_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('nba', {})


def _get_nba_season_type(date_str: str, season: str) -> str:
    """Return correct NBA season type for the given date using season_dates.yaml."""
    nba_dates = load_season_dates_config()
    season_block = nba_dates[season]
    playin_start = season_block.get('playin_start')
    playin_end = season_block.get('playin_end')
    playoff_start = season_block.get('playoff_start')
    playoff_end = season_block.get('playoff_end')
    if playin_start and playin_end and playin_start <= date_str <= playin_end:
        return 'PlayIn'
    if playoff_start and playoff_end and playoff_start <= date_str <= playoff_end:
        return 'Playoffs'
    return 'Regular Season'


def get_season_date_range(season):
    """
    Get the date range for a season from config.
    
    Args:
        season: Season string (e.g., '2025-26')
    
    Returns:
        Tuple of (season_start_date, regular_season_end_date) as date objects
        Returns (None, None) if not found in config
    """
    config = load_season_dates_config()
    
    if season not in config:
        logging.warning(f"Season {season} not found in config/season_dates.yaml")
        return None, None
    
    season_config = config[season]
    season_start = datetime.strptime(season_config['season_start'], '%Y-%m-%d').date()
    regular_season_end = datetime.strptime(season_config['regular_season_end'], '%Y-%m-%d').date()
    
    logging.info(f"📅 Season date range from config: {season_start} to {regular_season_end}")
    
    return season_start, regular_season_end


def generate_season_date_range(season):
    """
    Generate all dates in the season range from config.
    
    Args:
        season: Season string (e.g., '2025-26')
    
    Returns:
        List of date objects for all days in the season
    """
    season_start, season_end = get_season_date_range(season)
    
    if not season_start or not season_end:
        logging.error(f"Could not load season dates from config for {season}")
        return []
    
    # Generate all dates in range
    date_list = []
    current_date = season_start
    while current_date <= season_end:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    
    logging.info(f"📅 Generated {len(date_list)} dates from {season_start} to {season_end}")
    
    return date_list


# ============================================================================
# API FUNCTIONS (from fetch_historical_nba_prop_markets.py)
# ============================================================================

@timed
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


@timed
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


def _normalize_odds_response_for_parser(api_json: dict | None) -> dict | None:
    """Historical odds return {'data': event}; live /events/{{id}}/odds return the event object at top level."""
    if not api_json:
        return None
    if "data" in api_json:
        return api_json
    if "bookmakers" in api_json or "away_team" in api_json:
        return {"data": api_json}
    return None


@timed
def get_live_nba_events_list(sport=SPORT_KEY):
    """Upcoming NBA events (same feed as run_live_arb_finder / find_nba_arb_opportunities)."""
    global credits_remaining, credits_used
    endpoint = f"sports/{sport}/events"
    params = {"apiKey": API_KEY}
    logging.info(f"Fetching live event list: {endpoint}")
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        credits_remaining = int(float(response.headers.get("x-requests-remaining", 0)))
        credits_used = int(float(response.headers.get("x-requests-used", 0)))
        last_cost = int(float(response.headers.get("x-requests-last", 0)))
        logging.info(f"Live events list - Cost: {last_cost} credits, Remaining: {credits_remaining:,}")
        data = response.json()
        if isinstance(data, list):
            return data
        logging.warning("Live events response was not a list")
        return []
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error fetching live NBA events: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error fetching live NBA events: {e}", exc_info=True)
        return []


@timed
def get_live_event_player_props_odds(sport, event_id, markets=DEFAULT_MARKETS, regions=DEFAULT_REGION):
    """Player props for one event via non-historical /events/{{id}}/odds (used for same-day ET slates)."""
    global credits_remaining, credits_used
    endpoint = f"sports/{sport}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
    }
    logging.debug(f"Fetching live odds for event {event_id[:8]} - markets: {markets}")
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        credits_remaining = int(float(response.headers.get("x-requests-remaining", 0)))
        credits_used = int(float(response.headers.get("x-requests-used", 0)))
        last_cost = int(float(response.headers.get("x-requests-last", 0)))
        logging.info(f"Event {event_id[:8]} (live) - Cost: {last_cost} credits, Remaining: {credits_remaining:,}")
        time.sleep(RATE_LIMIT_DELAY)
        raw = response.json()
        return _normalize_odds_response_for_parser(raw)
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error fetching live odds for event {event_id[:8]}: {e}")
        if e.response is not None and e.response.status_code == 422:
            logging.warning(f"Props not available (live) for event {event_id[:8]}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching live odds for event {event_id[:8]}: {e}", exc_info=True)
        return None


def filter_events_by_et_calendar_date(events: list, target_date: date) -> list:
    """Keep events whose tipoff falls on target_date in America/New_York (full calendar day)."""
    et_tz = ZoneInfo("America/New_York")
    out = []
    for event in events:
        commence_time_utc = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        commence_time_et = commence_time_utc.astimezone(et_tz)
        if commence_time_et.date() == target_date:
            out.append(event)
    return out


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

@timed
def fetch_games_for_date(date_str, max_retries=1):
    """
    Fetch player game results for a specific date.
    Tries NBA API first, falls back to ESPN API if NBA is blocked.
    
    IMPORTANT: NBA API player game logs have a 12+ HOUR publishing delay!
    If games ended at 1am ET, data won't be available until ~2pm ET same day.
    
    NOTE: NBA API blocks Lambda IPs - ESPN fallback ensures reliability
    
    Args:
        date_str: Date in YYYY-MM-DD format
        max_retries: Number of retry attempts on timeout/connection errors
    
    Returns:
        DataFrame with player game logs for that date
    """
    logging.info(f"📡 Fetching NBA game results for {date_str}...")
    
    # =========================================================================
    # ATTEMPT 1: NBA API (preferred source)
    # =========================================================================
    for attempt in range(max_retries):
        try:
            # Fetch player game logs for the season
            season_str = SEASON  # e.g., '2025-26'

            season_type = _get_nba_season_type(date_str, season_str)
            logging.info(f"   Season type: {season_type} (from season_dates.yaml)")
            game_logs = playergamelogs.PlayerGameLogs(
                season_nullable=season_str,
                season_type_nullable=season_type,
                date_from_nullable=date_str,
                date_to_nullable=date_str,
            )

            games = game_logs.get_data_frames()[0]

            if games.empty:
                logging.info(f"   No games found for {date_str} (regular season or playoffs)")
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
            
            # NBA API returns MIN as string "MM:SS" (e.g. "35:24"). Parse to float minutes
            # so S3 CSV has numeric MIN; otherwise downstream (lambda backtest) drops rows.
            if 'MIN' in games.columns:
                min_vals = games['MIN']
                if min_vals.dtype == object or min_vals.dtype.kind in ('U', 'O', 'S'):
                    games['MIN'] = min_vals.apply(
                        lambda x: parse_minutes(x) if pd.notna(x) and str(x).strip() else 0.0
                    )
            
            num_games = games['GAME_ID'].nunique()
            num_players = len(games)
            
            logging.info(f"   ✅ Source: NBA API - Found {num_games} games with {num_players} players")
            
            # Rate limiting for NBA API
            time.sleep(RATE_LIMIT_DELAY)
            
            return games
            
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 1  # 1 second between retries - fail fast
                logging.warning(f"   ⚠️  NBA API timeout (attempt {attempt + 1}/{max_retries})")
                logging.warning(f"   Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logging.warning(f"   ⚠️  NBA API failed after {max_retries} attempts")
                
        except Exception as e:
            logging.warning(f"   ⚠️  NBA API error: {e}")
            break  # Don't retry on non-timeout errors
    
    # =========================================================================
    # ATTEMPT 2: ESPN API (fallback when NBA API blocked)
    # =========================================================================
    logging.info(f"   🔄 Falling back to ESPN API...")
    try:
        games_df = fetch_games_from_espn(date_str)
        
        if not games_df.empty:
            num_games = games_df['GAME_ID'].nunique()
            num_players = len(games_df)
            logging.info(f"   ✅ Source: ESPN API - Found {num_games} games with {num_players} players")
            return games_df
        else:
            logging.warning(f"   ⚠️  ESPN API returned no data")
            
    except Exception as e:
        logging.error(f"   ❌ ESPN API also failed: {e}")
    
    # Both sources failed
    logging.error(f"❌ Both NBA API and ESPN API failed for {date_str}")
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


@timed
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


@timed
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

@timed
def fetch_date_props(date_str, upload_s3=True, fetch_games=False, skip_if_exists=True, force=False):
    """
    Fetch player props for a specific date
    
    Args:
        date_str: Date in YYYY-MM-DD format
        upload_s3: Upload to S3 (default: True)
        fetch_games: Also fetch game results from NBA API
        skip_if_exists: Skip fetching if files exist in S3 (default: True)
        force: Force overwrite existing files (overrides skip_if_exists)
    
    Returns:
        Tuple of (props_df, games_df, game_lines_df)
    """
    # Force flag overrides skip_if_exists
    if force:
        skip_if_exists = False
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    day_of_week = date_obj.strftime('%A')
    
    # Check if props already exist in S3
    props_s3_key = f"{S3_PREFIX_PROPS}/{date_str}.csv"
    props_exist = skip_if_exists and upload_s3 and check_s3_file_exists(S3_BUCKET_PROPS, props_s3_key)
    
    if props_exist:
        logging.info("="*80)
        logging.info(f"⏭️  PROPS ALREADY EXIST FOR {date_str} - SKIPPING PROPS FETCH")
        logging.info(f"   S3: s3://{S3_BUCKET_PROPS}/{props_s3_key}")
        logging.info("="*80)
        df = pd.DataFrame()  # Empty dataframe since we're not fetching
    else:
        logging.info("="*80)
        logging.info(f"FETCHING PLAYER PROPS FOR {date_str} ({day_of_week})")
        logging.info(f"Source: The Odds API - 9 markets (points, rebounds, assists, threes, blocks, steals, double_double, triple_double, PRA)")
        logging.info("="*80)
        
        # Get events for that date (historical snapshot); fall back to live feed for "today" ET
        et_tz = ZoneInfo("America/New_York")
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        today_et = datetime.now(et_tz).date()

        all_events = get_historical_events(date_str)
        events = filter_events_by_et_calendar_date(all_events, target_date)
        use_live_odds = False

        if not events and target_date == today_et:
            logging.warning(
                f"No historical events for {date_str} after ET calendar filter; "
                f"trying live /sports/{SPORT_KEY}/events (same-day ingest)"
            )
            live_list = get_live_nba_events_list()
            events = filter_events_by_et_calendar_date(live_list, target_date)
            if events:
                use_live_odds = True
                logging.info(f"Live feed: {len(events)} game(s) on {date_str} (ET)")

        if not events:
            logging.error(f"No events found for {date_str} (historical + optional live fallback)")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        logging.info(f"Found {len(events)} games on {date_str} (ET calendar day)")

        # Fetch props for each event
        all_props = []
        for i, event in enumerate(events, 1):
            game_desc = f"{event['away_team']} @ {event['home_team']}"
            logging.info(f"Processing game {i}/{len(events)}: {game_desc}")

            if use_live_odds:
                odds_data = get_live_event_player_props_odds(
                    sport=SPORT_KEY,
                    event_id=event["id"],
                    markets=DEFAULT_MARKETS,
                )
            else:
                odds_data = get_historical_event_odds(
                    sport=SPORT_KEY,
                    event_id=event["id"],
                    date_str=date_str,
                    markets=DEFAULT_MARKETS,
                )
            
            if odds_data:
                props = parse_player_props(odds_data)
                all_props.extend(props)
                logging.info(f"  ✅ Found {len(props)} player props")
            else:
                logging.warning(f"  ⚠️  No props available")
        
        if not all_props:
            logging.warning(f"No props data collected for {date_str}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
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
    
    # =========================================================================
    # FETCH GAME LINES (The Odds API) - Spread, Moneyline
    # =========================================================================
    game_lines_df = pd.DataFrame()
    if fetch_games:
        # Check if game lines already exist in S3
        if skip_if_exists and upload_s3 and game_lines_file_exists_in_s3(date_str):
            game_lines_s3_key = get_game_lines_s3_key(date_str)
            logging.info("")
            logging.info("="*80)
            logging.info(f"⏭️  GAME LINES ALREADY EXIST FOR {date_str} - SKIPPING LINES FETCH")
            logging.info(f"   Source: The Odds API (spreads, moneylines)")
            logging.info(f"   S3: s3://{GAME_LINES_S3_BUCKET}/{game_lines_s3_key}")
            logging.info("="*80)
        else:
            logging.info("")
            logging.info("="*80)
            logging.info(f"FETCHING GAME LINES FOR {date_str}")
            logging.info(f"Source: The Odds API (spreads, moneylines)")
            logging.info("="*80)
            
            # Time the game lines fetch
            game_lines_start = time.time()
            logging.info(f"⏱️  [fetch_date_lines] Starting...")
            
            # Use fetch_date_lines from fetch_historical_nba_season_lines.py
            # It handles S3 upload internally when save=True
            game_lines_df = fetch_date_lines(date_str, save=upload_s3, local_backup=False, force=force)
            
            game_lines_elapsed = time.time() - game_lines_start
            logging.info(f"⏱️  [fetch_date_lines] Completed in {game_lines_elapsed:.2f}s")
            
            if not game_lines_df.empty:
                num_games = game_lines_df['game_id'].nunique()
                num_spread_lines = len(game_lines_df[game_lines_df['market'] == 'spread'])
                num_ml_lines = len(game_lines_df[game_lines_df['market'] == 'moneyline'])
                logging.info(f"✅ Fetched game lines: {num_games} games, {num_spread_lines} spread lines, {num_ml_lines} ML lines")
            else:
                logging.info("ℹ️  No games on this date (no game lines)")
    
    # =========================================================================
    # FETCH GAME RESULTS (NBA API) - Actual player stats (LAST, can fail)
    # =========================================================================
    games_df = pd.DataFrame()
    if fetch_games:
        # Check if game logs already exist in S3
        games_s3_key = f"{S3_PREFIX_GAMES}/{date_str}.csv"
        if skip_if_exists and upload_s3 and check_s3_file_exists(S3_BUCKET_GAMES, games_s3_key):
            logging.info("")
            logging.info("="*80)
            logging.info(f"⏭️  GAME RESULTS ALREADY EXIST FOR {date_str} - SKIPPING GAME FETCH")
            logging.info(f"   Source: NBA API (actual player stats)")
            logging.info(f"   S3: s3://{S3_BUCKET_GAMES}/{games_s3_key}")
            logging.info("="*80)
        else:
            logging.info("")
            logging.info("="*80)
            logging.info(f"FETCHING GAME RESULTS FOR {date_str}")
            logging.info(f"Source: NBA API (actual player stats)")
            logging.info(f"⚠️  Note: NBA API may block Lambda IPs - failure is non-critical")
            logging.info("="*80)
            
            games_df = fetch_games_for_date(date_str)
            
            if not games_df.empty:
                # Upload to S3
                if upload_s3:
                    save_games_to_s3(games_df, date_str)
                else:
                    logging.info("⚠️  Skipping game results S3 upload")
            else:
                # NBA API failed - log warning but continue (don't fail entire Lambda)
                error_msg = f"⚠️  WARNING: Game fetch returned 0 rows for {date_str}"
                logging.warning(error_msg)
                logging.warning(f"   NBA API may be blocking Lambda IPs or data not ready")
                logging.warning(f"   Continuing without game results...")
                # Don't exit - props and game lines are more important
    
    return df, games_df, game_lines_df


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


def fetch_full_season(upload_s3=True, fetch_games=False, force=False):
    """Fetch props for all dates in season"""
    logging.info("="*80)
    logging.info("FULL SEASON FETCH MODE")
    logging.info("="*80)
    logging.info(f"Upload to S3: {upload_s3}")
    logging.info(f"Fetch game results & game lines: {fetch_games}")
    if force:
        logging.info(f"Force mode: ON (will overwrite existing files)")
    
    # Generate all dates in season from config
    all_season_dates = generate_season_date_range(SEASON)
    
    if not all_season_dates:
        logging.error("Failed to generate season dates from config")
        return None
    
    # Filter to past dates only
    today = datetime.now().date()
    past_dates = [d for d in all_season_dates if d < today]
    
    logging.info(f"Past dates to fetch: {len(past_dates)}")
    logging.info(f"Future dates: {len(all_season_dates) - len(past_dates)}")
    
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
        
        # Fetch (will upload to S3, skips internally if files exist unless force=True)
        try:
            props_df, games_df, game_lines_df = fetch_date_props(date_str, upload_s3=upload_s3, fetch_games=fetch_games, force=force)
            
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
        if args.force:
            logging.info("⚠️  FORCE MODE: Will overwrite existing S3 files")
        upload_s3 = True if args.s3 is None else args.s3
        props_df, games_df, game_lines_df = fetch_date_props(date_str, upload_s3=upload_s3, fetch_games=args.fetch_games, force=args.force)
        if not props_df.empty:
            logging.info("✅ Props fetch complete!")
        if not games_df.empty:
            logging.info("✅ Game results fetch complete!")
        if not game_lines_df.empty:
            logging.info("✅ Game lines fetch complete!")
        return
    
    # Get mode
    if args.mode is not None:
        choice = str(args.mode)
    else:
        logging.info("\n" + "="*80)
        logging.info("MODE SELECTION")
        logging.info("="*80)
        logging.info("1. Test mode (fetch most recent game date)")
        logging.info("2. Full season mode (fetch all past dates)")
        logging.info("="*80)
        
        while True:
            choice = input("Select mode (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            logging.info("Invalid choice. Please enter 1 or 2.")
    
    if choice == '1':
        # Test mode - get most recent date in season
        all_season_dates = generate_season_date_range(SEASON)
        
        if not all_season_dates:
            logging.error("Failed to generate season dates from config")
            return
        
        # Get most recent past date
        today = datetime.now().date()
        past_dates = [d for d in all_season_dates if d < today]
        
        if len(past_dates) == 0:
            logging.error("No past game dates found")
            return
        
        test_date = max(past_dates)
        test_date_str = test_date.strftime('%Y-%m-%d')
        
        logging.info(f"TEST MODE: Fetching {test_date_str}")
        if args.force:
            logging.info("⚠️  FORCE MODE: Will overwrite existing S3 files")
        
        upload_s3 = True if args.s3 is None else args.s3
        props_df, games_df, game_lines_df = fetch_date_props(test_date_str, upload_s3=upload_s3, fetch_games=args.fetch_games, force=args.force)
        
        if not props_df.empty:
            logging.info("="*80)
            logging.info("✅ Test successful! Ready for full season.")
            logging.info("="*80)
        if not games_df.empty:
            logging.info("✅ Game results also fetched successfully!")
        if not game_lines_df.empty:
            logging.info("✅ Game lines also fetched successfully!")
    else:
        # Full season mode
        logging.info("FULL SEASON MODE selected")
        if args.force:
            logging.info("⚠️  FORCE MODE: Will overwrite existing S3 files")
        upload_s3 = args.s3 if args.s3 is not None else True  # Default to True for full season
        stats = fetch_full_season(upload_s3=upload_s3, fetch_games=args.fetch_games, force=args.force)
        
        if stats:
            if stats.get('skipped'):
                logging.info(f"\n✅ Season {SEASON} already complete - skipped fetching")
            elif stats.get('successful', 0) > 0:
                logging.info("\n✅ Full season fetch completed!")


if __name__ == "__main__":
    main()

