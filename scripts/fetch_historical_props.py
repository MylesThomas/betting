"""
Fetch Historical NBA Player Props from The Odds API

This script fetches historical player prop data for specific dates
Tests with opening day first, then can be used for full season

Steps I took:
1. Ran this file for opening day 2024-10-22
2. Updated code to iterate through the rest of the dates in the nba_calendar/game_dates_2024_25.csv
3. Ran this file for the rest of the dates

TIMESTAMPS CAPTURED:
- bookmaker_last_update: When bookmaker's general odds were last updated
- market_last_update: When the specific player_threes market was last updated
These timestamps allow filtering stale lines and detecting arbitrage windows

KNOWN LIMITATION:
This script only fetches Regular Season games (based on nba_calendar data).
Special games are NOT included:
- NBA Cup Championship (Bucks vs. Thunder, Dec 17, 2024)
- Other non-regular season games

To fetch these games manually, try using fetch_date_props('2024-12-17') directly (Have not tested this yet)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
from pathlib import Path
import time
import logging
import ssl
import urllib3
from zoneinfo import ZoneInfo

# Load environment variables
load_dotenv()

# ============================================================================
# SSL FIX FOR MACOS
# ============================================================================
# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests to disable SSL verification
original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

GLOBAL_LOG_PREFIX = 'fetch_historical_props_test'

def setup_logging(log_prefix=GLOBAL_LOG_PREFIX):
    """
    Configure logging to both file and console
    
    Args:
        log_prefix: Prefix for log filename (e.g., 'test', 'full_season', 'debug')
    """
    # Create logs directory in repo root (one level up from api_setup)
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = f"{log_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = log_dir / log_filename
    
    # Configure logging format
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Get root logger
    logger = logging.getLogger()
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized - log file: {log_filepath}")
    return log_filepath

# Initialize logging
setup_logging()

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4'
SPORT_KEY = 'basketball_nba'
DEFAULT_REGION = 'us'
DEFAULT_MARKET = 'player_threes'
ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'
# Use absolute path to save to correct location
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'data' / '01_input' / 'the-odds-api' / 'nba' / 'historical_props'
RATE_LIMIT_DELAY = 0.5  # seconds between API calls

# Event timestamp offsets (for capturing the right snapshot)
EVENT_LIST_HOUR = 12  # Noon UTC to get day's games
ODDS_SNAPSHOT_HOUR = 15  # 3 PM UTC (10 AM ET, before any games start)
# Note: NBA games can start as early as noon ET (matinees, weekend games)
# Some games start 7-10 PM ET for evening schedule
# We fetch at 3 PM UTC (10 AM ET) to capture all pre-game lines before earliest tipoffs

# API usage tracking
credits_remaining = None
credits_used = None

# Log all configuration at startup
logging.info("="*60)
logging.info("CONFIGURATION")
logging.info("="*60)
logging.info(f"API_KEY: {'*' * 8 + API_KEY[-4:] if API_KEY and len(API_KEY) > 4 else 'NOT SET'}")
logging.info(f"BASE_URL: {BASE_URL}")
logging.info(f"SPORT_KEY: {SPORT_KEY}")
logging.info(f"DEFAULT_REGION: {DEFAULT_REGION}")
logging.info(f"DEFAULT_MARKET: {DEFAULT_MARKET}")
logging.info(f"ODDS_FORMAT: {ODDS_FORMAT}")
logging.info(f"DATE_FORMAT: {DATE_FORMAT}")
logging.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
logging.info(f"RATE_LIMIT_DELAY: {RATE_LIMIT_DELAY}s")
logging.info(f"EVENT_LIST_HOUR: {EVENT_LIST_HOUR} UTC")
logging.info(f"ODDS_SNAPSHOT_HOUR: {ODDS_SNAPSHOT_HOUR} UTC (10 AM ET)")
logging.info(f"LOG_PREFIX: {GLOBAL_LOG_PREFIX}")
logging.info("="*60)


def get_historical_events(date_str, sport=SPORT_KEY):
    """
    Get list of events for a specific date
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        sport: Sport key (default: SPORT_KEY from config)
    
    Returns:
        List of event objects with IDs
    """
    global credits_remaining, credits_used
    
    # Convert date string to ISO8601 timestamp
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    timestamp = date_obj.replace(hour=EVENT_LIST_HOUR, minute=0, second=0).isoformat() + 'Z'
    
    endpoint = f'historical/sports/{sport}/events'
    params = {
        'api_key': API_KEY,  # Note: underscore, not camelCase!
        'date': timestamp,
        'dateFormat': DATE_FORMAT
    }
    
    logging.info(f"Fetching events for {date_str} (timestamp: {timestamp})")
    
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        
        # Track credits
        credits_remaining = int(response.headers.get('x-requests-remaining', 0))
        credits_used = int(response.headers.get('x-requests-used', 0))
        last_cost = int(response.headers.get('x-requests-last', 0))
        
        logging.info(f"API call successful - Cost: {last_cost} credits, Remaining: {credits_remaining:,}")
        
        data = response.json()
        
        if 'data' in data:
            events = data['data']
            logging.info(f"Found {len(events)} events for {date_str}")
            logging.debug(f"Event IDs: {[e['id'][:8] for e in events]}")
            return events
        else:
            logging.warning(f"No events found for {date_str}")
            return []
            
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error fetching events for {date_str}: {e}")
        if e.response.status_code == 401:
            logging.error("Invalid API key - check .env file")
        elif e.response.status_code == 422:
            logging.error("Date out of range or invalid format")
        return []
    except Exception as e:
        logging.error(f"Unexpected error fetching events: {e}", exc_info=True)
        return []


def get_historical_event_odds(sport, event_id, date_str, markets=DEFAULT_MARKET, regions=DEFAULT_REGION):
    """
    Get historical odds for a specific event
    
    Args:
        sport: Sport key
        event_id: Event ID from historical events endpoint
        date_str: Date string in YYYY-MM-DD format
        markets: Comma-separated markets (default: DEFAULT_MARKET from config)
        regions: Comma-separated regions (default: DEFAULT_REGION from config)
    
    Returns:
        Historical odds data
    """
    global credits_remaining, credits_used
    
    # Convert to ISO timestamp
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    timestamp = date_obj.replace(hour=ODDS_SNAPSHOT_HOUR, minute=0, second=0).isoformat() + 'Z'
    
    endpoint = f'historical/sports/{sport}/events/{event_id}/odds'
    params = {
        'api_key': API_KEY,  # Note: underscore, not camelCase!
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
        
        # Track credits
        credits_remaining = int(response.headers.get('x-requests-remaining', 0))
        credits_used = int(response.headers.get('x-requests-used', 0))
        last_cost = int(response.headers.get('x-requests-last', 0))
        
        logging.info(f"Event {event_id[:8]} - Cost: {last_cost} credits, Remaining: {credits_remaining:,}")
        
        data = response.json()
        
        # Rate limiting - be respectful to API
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
    """Parse player props from odds data into clean format"""
    if not odds_data or 'data' not in odds_data:
        logging.debug("No data to parse in odds_data")
        return []
    
    event_data = odds_data['data']
    props_list = []
    
    game_info = f"{event_data.get('away_team')} @ {event_data.get('home_team')}"
    game_time = event_data.get('commence_time')
    
    logging.debug(f"Parsing props for game: {game_info}")
    
    for bookmaker in event_data.get('bookmakers', []):
        bookmaker_name = bookmaker['key']
        bookmaker_last_update = bookmaker['last_update']
        
        for market in bookmaker.get('markets', []):
            market_key = market['key']
            market_last_update = market['last_update']
            
            # Group outcomes by player
            player_props = {}
            for outcome in market.get('outcomes', []):
                player = outcome.get('description', 'Unknown')
                line = outcome.get('point')
                odds = outcome.get('price')
                bet_type = outcome.get('name')  # 'Over' or 'Under'
                
                if player not in player_props:
                    player_props[player] = {
                        'player': player,
                        'game': game_info,
                        'game_time': game_time,
                        'market': market_key,
                        'line': line,
                        'bookmaker': bookmaker_name,
                        'bookmaker_last_update': bookmaker_last_update,  # When bookmaker updated
                        'market_last_update': market_last_update  # When this specific market updated
                    }
                
                if bet_type == 'Over':
                    player_props[player]['over_odds'] = odds
                elif bet_type == 'Under':
                    player_props[player]['under_odds'] = odds
            
            props_list.extend(player_props.values())
    
    return props_list


def fetch_date_props(date_str, markets=DEFAULT_MARKET, save=True):
    """
    Fetch all player props for a specific date
    
    Args:
        date_str: Date in YYYY-MM-DD format
        markets: Markets to fetch (default: DEFAULT_MARKET from config)
        save: Save results to file
    
    Returns:
        DataFrame with all props for that date
    """
    # Get day of week
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    day_of_week = date_obj.strftime('%A')
    
    logging.info("="*60)
    logging.info(f"FETCHING PROPS FOR {date_str} ({day_of_week})")
    logging.info("="*60)
    
    # Get events for that date
    all_events = get_historical_events(date_str)
    
    if not all_events:
        logging.error(f"No events found for {date_str}")
        return pd.DataFrame()
    
    # Filter to only games that START on this specific date (in ET timezone)
    # (API returns all games with lines posted, including future games)
    # NBA games typically run from ~6 AM ET to ~11 PM ET
    et_tz = ZoneInfo('America/New_York')
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
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
    
    filtered_count = len(all_events) - len(events)
    logging.info(f"Found {len(all_events)} total events in API response")
    logging.info(f"Filtered out {filtered_count} future/past games")
    logging.info(f"Kept {len(events)} games that actually start on {date_str}:")
    
    if not events:
        logging.warning(f"No games actually start on {date_str} (all were future/past games)")
        logging.debug(f"Sample filtered event times: {[e.get('commence_time') for e in all_events[:5]]}")
        return pd.DataFrame()
    
    for event in events:
        # Parse and convert times
        commence_time_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
        commence_time_et = commence_time_utc.astimezone(ZoneInfo('America/New_York'))
        
        time_utc = commence_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
        time_et = commence_time_et.strftime('%Y-%m-%d %I:%M:%S %p ET')
        
        logging.info(f"  • {event['away_team']} @ {event['home_team']}")
        logging.info(f"    Start: {time_et} ({time_utc})")
        logging.info(f"    Event ID: {event['id']}")
        logging.info(f"    Sport: {event.get('sport_key', 'N/A')} | Sport Title: {event.get('sport_title', 'N/A')}")
    
    estimated_cost = len(events) * 10
    logging.info(f"Estimated cost: {len(events)} games × ~10 credits = ~{estimated_cost} credits")
    
    # Get day of week for confirmation prompt
    date_obj_confirm = datetime.strptime(date_str, '%Y-%m-%d')
    day_of_week_confirm = date_obj_confirm.strftime('%A')
    
    # Auto-approve if cost is under 150 credits
    if estimated_cost < 150:
        logging.info(f"✅ Auto-approved (cost {estimated_cost} < 150 credits)")
        logging.info("-"*60)
    else:
        # Ask for confirmation for larger fetches
        print(f"\n⚠️  Proceed with fetching {len(events)} games for {date_str} ({day_of_week_confirm})?")
        print(f"   Markets: {markets}")
        print(f"   Estimated cost: ~{estimated_cost} credits")
        response = input(f"   Continue? (y/n): ")
        
        if response.lower() != 'y':
            logging.warning("User cancelled fetch operation")
            return pd.DataFrame()
        
        logging.info("User confirmed - starting fetch operation")
        logging.info("-"*60)
    
    # Fetch odds for each event
    all_props = []
    for i, event in enumerate(events, 1):
        game_desc = f"{event['away_team']} @ {event['home_team']}"
        logging.info(f"Processing game {i}/{len(events)}: {game_desc}")
        
        odds_data = get_historical_event_odds(
            sport=SPORT_KEY,
            event_id=event['id'],
            date_str=date_str,
            markets=markets
        )
        
        if odds_data:
            props = parse_player_props(odds_data)
            all_props.extend(props)
            logging.info(f"  ✅ Found {len(props)} player props for {game_desc}")
        else:
            logging.warning(f"  ⚠️  No props available for {game_desc}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_props)
    
    if df.empty:
        logging.warning(f"No props data collected for {date_str}")
        return df
    
    if save:
        output_dir = OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"props_{date_str}_{markets.replace(',', '_')}.csv"
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)
        
        logging.info("="*60)
        logging.info(f"Saved to {filepath}")
        logging.info("="*60)
    
    # Show summary
    logging.info("SUMMARY:")
    logging.info(f"  Total props: {len(df)}")
    logging.info(f"  Unique players: {df['player'].nunique()}")
    logging.info(f"  Bookmakers: {', '.join(df['bookmaker'].unique())}")
    logging.info(f"  Markets: {', '.join(df['market'].unique())}")
    
    logging.debug("Sample props:")
    logging.debug(f"\n{df[['player', 'game', 'market', 'line', 'over_odds', 'under_odds']].head(10).to_string()}")
    
    return df


def fetch_full_season(markets=DEFAULT_MARKET):
    """
    Fetch props for all dates in 2024-25 NBA season
    Iterates through every day and checks if games are scheduled
    
    Args:
        markets: Markets to fetch (default: DEFAULT_MARKET)
    
    Returns:
        Summary dict with statistics
    """
    logging.info("="*60)
    logging.info("FULL SEASON FETCH MODE")
    logging.info("="*60)
    logging.warning("Note: Only fetching Regular Season games (per nba_calendar data)")
    logging.warning("NBA Cup championship and other special games are NOT included")
    logging.warning("Example: Bucks vs Thunder on Dec 17, 2024 will be skipped")
    
    # Load all games CSV to check which dates have games
    games_csv = Path(__file__).parent / 'nba_calendar' / 'all_games_2024_25.csv'
    
    if not games_csv.exists():
        logging.error(f"Games CSV not found: {games_csv}")
        logging.error("Run nba_calendar_builder.py first to generate calendar")
        return None
    
    # Load games and extract unique game dates
    games_df = pd.read_csv(games_csv)
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE']).dt.date
    game_dates_set = set(games_df['GAME_DATE'].unique())
    
    # Get season date range
    opening_day = games_df['GAME_DATE'].min()
    last_game = games_df['GAME_DATE'].max()
    
    # Generate ALL dates in season (not just game dates)
    all_dates = []
    current_date = opening_day
    while current_date <= last_game:
        all_dates.append(current_date)
        current_date += timedelta(days=1)
    
    total_dates = len(all_dates)
    expected_game_dates = len(game_dates_set)
    
    logging.info(f"Season date range: {opening_day} to {last_game}")
    logging.info(f"Total days in season: {total_dates}")
    logging.info(f"Days with games (per CSV): {expected_game_dates}")
    logging.info(f"Days without games: {total_dates - expected_game_dates}")
    
    # Track statistics
    stats = {
        'total_dates': total_dates,
        'expected_game_dates': expected_game_dates,
        'processed': 0,
        'skipped_no_games': 0,  # Skipped because CSV says no games
        'skipped_existing': 0,   # Skipped because already fetched
        'skipped_user': 0,
        'failed': 0,
        'no_data': 0,
        'successful': 0,
        'total_props_collected': 0,
        'failed_dates': [],
        'credits_start': None,  # Will be set after first API call
        'credits_start_remaining': None  # Track starting remaining credits
    }
    
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("\n" + "="*60)
    logging.info("STARTING FULL SEASON FETCH")
    logging.info("="*60)
    
    # Track if this is first fetch (to log starting credits)
    first_fetch = True
    
    for i, date_obj in enumerate(all_dates, 1):
        # Convert date object to string and get day of week
        date_str = date_obj.strftime('%Y-%m-%d')
        day_of_week = date_obj.strftime('%A')  # Full day name (e.g., Wednesday)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"PROCESSING DATE {i}/{total_dates}: {date_str} ({day_of_week})")
        logging.info(f"Progress: {(i/total_dates)*100:.1f}% | {stats['successful']} successful | {stats['skipped_existing']} skipped | {stats['skipped_no_games']} no games")
        logging.info(f"{'='*60}")
        
        # Check if this date has games according to CSV
        # NOTE: This skips non-regular season games (e.g., NBA Cup championship on 2024-12-17)
        if date_obj not in game_dates_set:
            logging.info(f"⊘ No games scheduled on {date_str} (per all_games_2024_25.csv)")
            logging.info(f"  Skipping API call - no regular season games on this date")
            if date_str == '2024-12-17':
                logging.info(f"  ℹ️  Note: NBA Cup championship (Bucks vs Thunder) not in calendar")
            stats['skipped_no_games'] += 1
            continue
        
        # Date has games - proceed with fetch logic
        logging.info(f"✓ Games scheduled on {date_str} (per CSV) - checking for data...")
        
        # Check if file already exists
        filename = f"props_{date_str}_{markets.replace(',', '_')}.csv"
        filepath = output_dir / filename
        
        if filepath.exists():
            # Check how many rows in existing file
            try:
                existing_df = pd.read_csv(filepath)
                num_props = len(existing_df)
                logging.info(f"✓ File already exists: {filepath}")
                logging.info(f"  Contains {num_props} props | Skipping {date_str}")
            except Exception:
                logging.info(f"✓ File already exists: {filepath}")
                logging.info(f"  Skipping {date_str} (already fetched)")
            
            stats['skipped_existing'] += 1
            continue
        
        # Track credits before fetch
        credits_before_fetch = credits_remaining
        
        # Fetch props for this date
        try:
            df = fetch_date_props(date_str, markets=markets, save=True)
            
            # Log starting credits after first API call
            if first_fetch and credits_remaining is not None:
                stats['credits_start'] = credits_used
                stats['credits_start_remaining'] = credits_remaining
                
                logging.info("\n" + "="*60)
                logging.info("💰 STARTING CREDITS")
                logging.info("="*60)
                logging.info(f"Credits remaining at start: {credits_remaining:,}")
                logging.info(f"Estimated capacity: ~{credits_remaining // 10} games")
                logging.info("="*60 + "\n")
                first_fetch = False
            
            # Calculate credits used for this specific date
            credits_used_for_date = 0
            if credits_before_fetch and credits_remaining:
                credits_used_for_date = credits_before_fetch - credits_remaining
            
            if df.empty:
                logging.warning(f"❌ No data collected for {date_str}")
                logging.warning(f"   This could mean: no games on this date, or props not available")
                if credits_used_for_date > 0:
                    logging.info(f"   Credits used: {credits_used_for_date}")
                stats['no_data'] += 1
            else:
                # Log detailed success info
                num_games = df['game'].nunique()
                logging.info("="*60)
                logging.info(f"✅ FETCH COMPLETE FOR {date_str}")
                logging.info("="*60)
                logging.info(f"   File saved: {filepath}")
                logging.info(f"   Total props: {len(df)}")
                logging.info(f"   Unique players: {df['player'].nunique()}")
                logging.info(f"   Games fetched: {num_games}")
                logging.info(f"   Bookmakers: {', '.join(df['bookmaker'].unique())}")
                logging.info(f"   💰 Credits used: {credits_used_for_date} (1 event list + {num_games} games × ~10)")
                logging.info(f"   💰 Credits remaining: {credits_remaining:,}")
                logging.info("="*60)
                
                stats['successful'] += 1
                stats['total_props_collected'] += len(df)
            
            stats['processed'] += 1
            
        except KeyboardInterrupt:
            logging.warning("\n⚠️  User interrupted (Ctrl+C)")
            logging.info("Stopping fetch process...")
            break
            
        except Exception as e:
            logging.error(f"❌ Error processing {date_str}: {e}", exc_info=True)
            stats['failed'] += 1
            stats['failed_dates'].append(date_str)
            
            # Stop on error so user can investigate
            logging.error("\n" + "="*60)
            logging.error("FETCH STOPPED DUE TO ERROR")
            logging.error("="*60)
            logging.error(f"Failed on date: {date_str}")
            logging.error("Please investigate the error above before continuing")
            logging.error("You can re-run this script and it will skip already-fetched dates")
            break
    
    # Show final summary
    logging.info("\n" + "="*60)
    logging.info("FULL SEASON FETCH SUMMARY")
    logging.info("="*60)
    logging.info(f"Season: 2024-25")
    logging.info(f"Date range: {opening_day} to {last_game}")
    logging.info("")
    logging.info(f"Total days in season: {stats['total_dates']}")
    logging.info(f"Expected game days: {stats['expected_game_dates']}")
    logging.info(f"Days without games: {stats['skipped_no_games']}")
    logging.info("")
    logging.info(f"Processed this session: {stats['processed']}")
    logging.info(f"  ✅ Successful: {stats['successful']}")
    logging.info(f"  ⏭️  Skipped (already exist): {stats['skipped_existing']}")
    logging.info(f"  ⚠️  No data from API: {stats['no_data']}")
    logging.info(f"  ❌ Failed: {stats['failed']}")
    logging.info("")
    logging.info(f"Total props collected this session: {stats['total_props_collected']:,}")
    
    if credits_remaining is not None:
        logging.info("")
        logging.info("💰 CREDITS SUMMARY & VERIFICATION")
        if stats['credits_start_remaining']:
            actual_spent = stats['credits_start_remaining'] - credits_remaining
            logging.info(f"   Started with: {stats['credits_start_remaining']:,} credits")
            logging.info(f"   Remaining now: {credits_remaining:,} credits")
            logging.info(f"   Actual used (API tracking): {actual_spent:,} credits")
            
            # Verification
            if credits_used and stats['credits_start']:
                tracked_spent = credits_used - stats['credits_start']
                logging.info(f"   Our tracking: {tracked_spent:,} credits")
                
                if actual_spent == tracked_spent:
                    logging.info(f"   ✅ Verification: Our tracking matches API perfectly!")
                else:
                    diff = abs(actual_spent - tracked_spent)
                    logging.warning(f"   ⚠️  Mismatch: {diff} credit difference (not critical)")
        else:
            logging.info(f"   Remaining: {credits_remaining:,} credits")
        
        logging.info(f"   Capacity left: ~{credits_remaining // 10} more games")
    
    if stats['failed_dates']:
        logging.info("")
        logging.info(f"❌ Failed dates: {', '.join(stats['failed_dates'])}")
        logging.info("   Re-run this script to retry failed dates")
    
    # Check how many game dates still need fetching
    remaining = stats['expected_game_dates'] - stats['skipped_existing'] - stats['successful']
    if remaining > 0:
        logging.info("")
        logging.info(f"📋 Game dates remaining to fetch: {remaining}")
        logging.info(f"   Estimated credits needed: ~{remaining * 10} (assuming ~10 credits per game date)")
    
    logging.info("="*60)
    
    return stats


def main():
    """Main entry point - choose between test mode and full season"""
    logging.info("="*60)
    logging.info("HISTORICAL PLAYER PROPS FETCHER")
    logging.info("="*60)
    
    if not API_KEY or API_KEY == 'your_api_key_here':
        logging.error("API key not configured!")
        logging.error("Make sure your API key is in the .env file!")
        return
    
    logging.info(f"API configured - Base URL: {BASE_URL}")
    logging.info(f"Default market: {DEFAULT_MARKET}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    
    # Ask user what they want to do
    print("\n" + "="*60)
    print("MODE SELECTION")
    print("="*60)
    print("1. Test mode (fetch opening day only)")
    print("2. Full season mode (fetch all 163 game days)")
    print("="*60)
    
    while True:
        choice = input("Select mode (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    if choice == '1':
        # Test mode - opening day
        opening_day = '2024-10-22'
        
        logging.info(f"TEST MODE: Fetching props for opening day ({opening_day})")
        logging.info("This is a small test to validate the API")
        
        df = fetch_date_props(
            date_str=opening_day,
            markets=DEFAULT_MARKET,
            save=True
        )
        
        if not df.empty:
            logging.info("="*60)
            logging.info("✅ Test successful! Ready to fetch full season data.")
            logging.info("="*60)
            logging.info("Run again and select mode 2 for full season fetch")
        else:
            logging.error("Test failed - no data collected")
    
    else:
        # Full season mode
        logging.info("FULL SEASON MODE selected")
        stats = fetch_full_season(markets=DEFAULT_MARKET)
        
        if stats and stats['successful'] > 0:
            logging.info("\n✅ Full season fetch completed successfully!")
        elif stats and stats['failed'] > 0:
            logging.warning("\n⚠️  Fetch stopped due to error - see logs above")
        else:
            logging.info("\n✅ All dates already fetched!")


if __name__ == "__main__":
    main()

