"""
Live Betting Signal Generator

Purpose:
Scan live NBA games for profitable betting opportunities using Monte Carlo
simulation to detect edges between our model and live betting markets.

Process:
1. Fetch live games from ESPN API
2. Validate PBP data freshness (Gate 2: <1min old)
3. Fetch live odds from The Odds API
4. Filter stale odds (Gate 3: <1min at bookmaker level)
5. Get active players (Gate 1: in live game)
6. For each player with fresh odds and pregame line (Gates 3 & 4):
   - Run Monte Carlo simulation (1000 iterations)
   - Analyze all (bookmaker × line × side) combinations
   - Return best bet by expected value
7. Display profitable signals
8. Save live odds data to S3

Performance Gates (all must pass before running MC):
- Gate 1: Player is in a live game (ESPN boxscore exists)
- Gate 2: PBP data is fresh (<1 min old)
- Gate 3: Player has live odds from at least one fresh bookmaker
- Gate 4: Player has pregame line in S3 (for calibration)

Usage:
    # Run once
    python src/pbp_data/10_live_betting_signal_generator.py
    
    # Run continuously (every 60 seconds)
    python src/pbp_data/10_live_betting_signal_generator.py --loop --interval 60
    
    # With custom parameters
    python src/pbp_data/10_live_betting_signal_generator.py --min-edge 0.20 --n-sims 2000
    
    # Test mode with fake data
    python src/pbp_data/10_live_betting_signal_generator.py --test-with-fake-data

Output:
    - Console: Profitable betting signals with specific bookmakers
    - S3: Live odds snapshots (s3://nba-betting-mt/data/01_input/live_player_odds/)
"""

import sys
import requests
import boto3
import pandas as pd
import numpy as np
import json
import os
import argparse
import time
import pytz
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pbp_data.monte_carlo_utils import (
    load_player_profile,
    monte_carlo_simulate_bet,
    find_vegas_adjustment,
    get_consensus_prop_line,
    get_data_paths
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# The Odds API configuration
ODDS_API_KEY = os.environ.get('THE_ODDS_API_KEY', '')  # Set in environment
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# S3 configuration
S3_BUCKET = "nba-betting-mt"
S3_LIVE_ODDS_PREFIX = "data/01_input/live_player_odds/player_points"
S3_SIGNALS_PREFIX = "data/04_output/live_betting_signals/player_points"

# Local data directories
LOCAL_DATA_DIR = Path.home() / "Downloads" / "tmp" / "live_betting_data"
LOCAL_ODDS_DIR = LOCAL_DATA_DIR / "odds"
LOCAL_SIGNALS_DIR = LOCAL_DATA_DIR / "signals"

# Create directories
LOCAL_ODDS_DIR.mkdir(exist_ok=True, parents=True)
LOCAL_SIGNALS_DIR.mkdir(exist_ok=True, parents=True)

# Betting parameters
MIN_EDGE_THRESHOLD = 0.10  # 15% minimum edge
N_SIMULATIONS = 1000  # Default simulations (balance speed vs accuracy)
MAX_PLAYERS_PER_GAME = 20  # Only check top scorers to save time
MAX_DATA_AGE_SECONDS = 60  # Maximum age for fresh data (PBP and odds)

# ESPN API
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

# Initialize boto3
s3_client = boto3.client('s3')


# =============================================================================
# PERFORMANCE TIMING DECORATOR
# =============================================================================

def timed(func):
    """Decorator to time function execution and log results."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"      ⏱️  {func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper


# =============================================================================
# HELPER FUNCTIONS - ODDS CONVERSION
# =============================================================================

def adjust_for_longshot_bias(market_prob: float, odds: int) -> float:
    """
    Adjust market probability for bookmaker's longshot bias.
    
    Context (Longshot Fallacy):
    For money line bets, bookmakers disproportionately load vig onto longshots.
    This is because bettors systematically overbet underdogs, so bookmakers
    exploit this by offering worse odds on longshots than fair pricing would suggest.
    
    Example:
    - True probabilities: Favorite 90%, Longshot 10%
    - Fair American odds: Favorite -900, Longshot +900
    - Bookmaker adds 10% vig (skewed toward longshot):
      - Favorite implied: 92% → -1150 (slightly worse)
      - Longshot implied: 18% → +456 (massively worse, cut from +900 to +456)
    
    Result:
    - Longshots = terrible value (huge vig)
    - Favorites = less terrible value (moderate vig)
    
    This adjustment corrects for this systematic bias. For O/U bets at -110/-110,
    the adjustment is minimal. For moneyline bets at -300/+250, it's significant.
    
    Adjustment methodology:
    - For probabilities near 50% (±10pp): minimal adjustment (normal vig)
    - For longshots (<40%): reduce implied probability (they load more vig)
    - For favorites (>60%): slight increase (they load less vig)
    
    Args:
        market_prob: Market-implied probability (with vig, from odds)
        odds: American odds (e.g., -110, +200)
    
    Returns:
        Adjusted probability accounting for longshot bias
    """
    # For near 50/50 bets (-110 range), no adjustment needed
    if 0.45 <= market_prob <= 0.55:
        return market_prob
    
    # For longshots (underdogs), bookmakers typically add 5-10% extra vig
    # We need to REDUCE the implied probability to get closer to true probability
    if market_prob < 0.40:
        # Heavy underdog: reduce by ~15%
        longshot_adjustment = 0.85
    elif market_prob < 0.45:
        # Moderate underdog: reduce by ~7%
        longshot_adjustment = 0.93
    elif market_prob > 0.60:
        # Heavy favorite: slight increase (~3%)
        longshot_adjustment = 1.03
    elif market_prob > 0.55:
        # Moderate favorite: minimal increase (~1.5%)
        longshot_adjustment = 1.015
    else:
        longshot_adjustment = 1.0
    
    adjusted_prob = market_prob * longshot_adjustment
    
    # Clamp to valid probability range
    return max(0.01, min(0.99, adjusted_prob))


def american_odds_to_prob(odds: int) -> float:
    """
    Convert American odds to implied probability.
    
    Args:
        odds: American odds (e.g., -110, +200)
    
    Returns:
        Implied probability (0.0 to 1.0)
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def american_odds_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds < 0:
        return 1 + (100 / abs(odds))
    else:
        return 1 + (odds / 100)


def remove_vig(prob_over: float, prob_under: float) -> Tuple[float, float]:
    """
    Remove bookmaker's vig to get true probabilities.
    
    Args:
        prob_over: Implied probability of over (with vig)
        prob_under: Implied probability of under (with vig)
    
    Returns:
        Tuple of (true_prob_over, true_prob_under)
    """
    total = prob_over + prob_under
    return prob_over / total, prob_under / total


def calculate_ev(model_prob: float, odds: int, bet_amount: float = 100) -> float:
    """
    Calculate expected value of a bet.
    
    Args:
        model_prob: True probability from our model (0.0 to 1.0)
        odds: American odds (e.g., -110, +200)
        bet_amount: Amount to bet (default $100)
    
    Returns:
        Expected value in dollars
    """
    decimal_odds = american_odds_to_decimal(odds)
    
    # EV = (probability of win × amount won) - (probability of loss × amount lost)
    win_amount = (decimal_odds - 1) * bet_amount
    ev = (model_prob * win_amount) - ((1 - model_prob) * bet_amount)
    
    return ev


# =============================================================================
# TEST MODE - FAKE DATA GENERATION
# =============================================================================

def generate_fake_live_games() -> List[Dict]:
    """Generate fake live games for testing."""
    return [
        {
            'game_id': '401810642',
            'away_team': 'Memphis Grizzlies',
            'home_team': 'Denver Nuggets',
            'away_score': 44,
            'home_score': 53,
            'quarter': 2,
            'clock': '3:28',
            'game_date': '2026-02-11'
        }
    ]


def generate_fake_active_players(game_id: str) -> List[Dict]:
    """
    Generate fake active players for testing.
    
    Note: These players MUST have:
    1. Historical data in minute_by_minute.parquet (for player profile)
    2. Pregame lines in S3 player props data (for baseline calibration)
    
    Using real players from actual games ensures the test works end-to-end.
    
    Test scenarios:
    - Jokic: Normal -110 odds (tests standard calculation)
    - Murray: Skewed odds -300/+250 (tests longshot bias adjustment)
    """
    return [
        {
            'player_name': 'Nikola Jokic',
            'player_id': '3112335',
            'team': 'Denver Nuggets',
            'current_points': 9.0,  # Player's current points (Q2, ~13 min into game)
            'minutes_played': 12.5
        },
        {
            'player_name': 'Jamal Murray',
            'player_id': '3102529',
            'team': 'Denver Nuggets',
            'current_points': 6.0,  # Player's current points
            'minutes_played': 13.2
        },
        {
            'player_name': 'LeBron James',
            'player_id': '1966',
            'team': 'Los Angeles Lakers',
            'current_points': 8.0,  # Test with skewed odds
            'minutes_played': 11.0
        }
    ]


def generate_fake_live_odds() -> pd.DataFrame:
    """
    Generate realistic fake live odds for testing.
    
    Simulates real market structure:
    - Multiple bookmakers (DraftKings, FanDuel, BetMGM, BetRivers, BetOnlineAG, Bovada)
    - Main lines at -110/-110 (most books)
    - Bovada with multiple alt lines (5-9 different prop lines)
    - Skewed odds for some players (tests longshot bias)
    
    This tests the multi-bookmaker, multi-line analysis logic.
    """
    now = datetime.now(timezone.utc).isoformat()
    odds_data = []
    
    # Nikola Jokic - standard market (pregame 26.5, live 24.5)
    # Multiple bookmakers, mostly -110
    for bookmaker in ['draftkings', 'fanduel', 'betmgm', 'betrivers', 'betonlineag']:
        odds_data.extend([
            {'bookmaker': bookmaker, 'player_name': 'Nikola Jokic', 'line': 24.5, 'side': 'Over', 'odds': -110, 'timestamp': now},
            {'bookmaker': bookmaker, 'player_name': 'Nikola Jokic', 'line': 24.5, 'side': 'Under', 'odds': -110, 'timestamp': now},
        ])
    
    # Bovada alt lines for Jokic (pregame was 26.5, live main is 24.5)
    jokic_bovada_lines = [
        (20.5, -250, 185),   # Heavy favorite over
        (21.5, -200, 150),
        (22.5, -165, 125),
        (23.5, -130, 100),
        (24.5, -110, -120),  # Main line
        (25.5, 110, -145),
        (26.5, 130, -170),
        (27.5, 155, -210),
        (28.5, 190, -260),
    ]
    for line, over_odds, under_odds in jokic_bovada_lines:
        odds_data.extend([
            {'bookmaker': 'bovada', 'player_name': 'Nikola Jokic', 'line': line, 'side': 'Over', 'odds': over_odds, 'timestamp': now},
            {'bookmaker': 'bovada', 'player_name': 'Nikola Jokic', 'line': line, 'side': 'Under', 'odds': under_odds, 'timestamp': now},
        ])
    
    # Jamal Murray - standard market (pregame 22.5, live 20.5)
    for bookmaker in ['draftkings', 'fanduel', 'betmgm', 'betrivers', 'betonlineag']:
        odds_data.extend([
            {'bookmaker': bookmaker, 'player_name': 'Jamal Murray', 'line': 20.5, 'side': 'Over', 'odds': -110, 'timestamp': now},
            {'bookmaker': bookmaker, 'player_name': 'Jamal Murray', 'line': 20.5, 'side': 'Under', 'odds': -110, 'timestamp': now},
        ])
    
    # Bovada alt lines for Murray (pregame was 22.5, live main is 20.5)
    murray_bovada_lines = [
        (16.5, -250, 185),
        (17.5, -200, 150),
        (18.5, -165, 125),
        (19.5, -130, 100),
        (20.5, -110, -120),  # Main line
        (21.5, 110, -145),
        (22.5, 130, -170),
        (23.5, 155, -210),
    ]
    for line, over_odds, under_odds in murray_bovada_lines:
        odds_data.extend([
            {'bookmaker': 'bovada', 'player_name': 'Jamal Murray', 'line': line, 'side': 'Over', 'odds': over_odds, 'timestamp': now},
            {'bookmaker': 'bovada', 'player_name': 'Jamal Murray', 'line': line, 'side': 'Under', 'odds': under_odds, 'timestamp': now},
        ])
    
    # LeBron James - heavily skewed market (pregame 25.5, live 23.5)
    # Market thinks he's unlikely to hit (he's having a bad game)
    # Main line is -290/+240 (tests longshot bias adjustment)
    for bookmaker in ['draftkings', 'fanduel', 'betmgm', 'betrivers']:
        # Slight variation across books
        over_odds = -290 if bookmaker == 'draftkings' else -280
        under_odds = 240 if bookmaker == 'draftkings' else 230
        odds_data.extend([
            {'bookmaker': bookmaker, 'player_name': 'LeBron James', 'line': 23.5, 'side': 'Over', 'odds': over_odds, 'timestamp': now},
            {'bookmaker': bookmaker, 'player_name': 'LeBron James', 'line': 23.5, 'side': 'Under', 'odds': under_odds, 'timestamp': now},
        ])
    
    # Bovada alt lines for LeBron (pregame was 25.5, live main is 23.5)
    lebron_bovada_lines = [
        (19.5, -350, 250),
        (20.5, -300, 220),
        (21.5, -270, 200),
        (22.5, -240, 180),
        (23.5, -210, 160),  # Main line (still skewed but less extreme)
        (24.5, -170, 130),
        (25.5, -140, 110),
        (26.5, -110, -120),
    ]
    for line, over_odds, under_odds in lebron_bovada_lines:
        odds_data.extend([
            {'bookmaker': 'bovada', 'player_name': 'LeBron James', 'line': line, 'side': 'Over', 'odds': over_odds, 'timestamp': now},
            {'bookmaker': 'bovada', 'player_name': 'LeBron James', 'line': line, 'side': 'Under', 'odds': under_odds, 'timestamp': now},
        ])
    
    return pd.DataFrame(odds_data)


# =============================================================================
# STEP 1: FETCH LIVE GAMES
# =============================================================================

@timed
def fetch_live_games(test_mode: bool = False) -> List[Dict]:
    """
    Fetch currently live NBA games from ESPN API.
    
    Args:
        test_mode: If True, return fake data for testing
    
    Returns:
        List of game dictionaries with game_id, teams, score, clock, etc.
    """
    if test_mode:
        return generate_fake_live_games()
    
    try:
        response = requests.get(ESPN_SCOREBOARD_URL, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        live_games = []
        
        for event in data.get('events', []):
            status = event['status']['type']['name']
            
            # Only include in-progress games
            if status == 'STATUS_IN_PROGRESS':
                competition = event['competitions'][0]
                
                game_info = {
                    'game_id': event['id'],
                    'away_team': competition['competitors'][1]['team']['displayName'],
                    'home_team': competition['competitors'][0]['team']['displayName'],
                    'away_score': int(competition['competitors'][1]['score']),
                    'home_score': int(competition['competitors'][0]['score']),
                    'quarter': event['status']['period'],
                    'clock': event['status']['displayClock'],
                    'game_date': event['date'][:10],  # YYYY-MM-DD
                }
                
                live_games.append(game_info)
        
        return live_games
    
    except Exception as e:
        print(f"❌ Error fetching live games: {e}")
        return []


# =============================================================================
# STEP 2: VALIDATE PBP FRESHNESS (GATE 2)
# =============================================================================

@timed
def fetch_and_validate_pbp(game_id: str, max_age_seconds: int = MAX_DATA_AGE_SECONDS, test_mode: bool = False) -> Optional[Dict]:
    """
    Fetch play-by-play data and validate it's fresh (Gate 2).
    
    Args:
        game_id: ESPN game ID
        max_age_seconds: Maximum age in seconds for data to be considered fresh
        test_mode: If True, always return valid (for testing)
    
    Returns:
        PBP data dict if fresh, None if stale or API fails
    """
    if test_mode:
        # In test mode, return a minimal valid structure
        return {'plays': [{'clock': {'timestamp': datetime.now(timezone.utc).isoformat()}}]}
    
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        # Check if we have plays
        plays = data.get('plays', [])
        if not plays:
            print(f"      ⚠️  No plays found in PBP (Gate 2 failed)")
            return None
        
        # Get last play timestamp
        last_play = plays[-1]
        last_play_timestamp = last_play.get('wallclock')
        
        if not last_play_timestamp:
            print(f"      ⚠️  No timestamp on last play (Gate 2 failed)")
            return None
        
        # Parse timestamp
        try:
            # ESPN timestamp format: "2024-02-11T21:45:30Z"
            last_play_time = datetime.fromisoformat(last_play_timestamp.replace('Z', '+00:00'))
            age_seconds = (datetime.now(timezone.utc) - last_play_time).total_seconds()
            
            if age_seconds > max_age_seconds:
                print(f"      ⚠️  PBP data stale ({age_seconds:.0f}s old) (Gate 2 failed)")
                return None
            
            print(f"      ✅ PBP data fresh ({age_seconds:.0f}s old) (Gate 2 passed)")
            return data
        
        except Exception as e:
            print(f"      ⚠️  Could not parse timestamp: {e} (Gate 2 failed)")
            return None
    
    except Exception as e:
        print(f"      ❌ Failed to fetch PBP: {e} (Gate 2 failed)")
        return None


# =============================================================================
# STEP 3: GET ACTIVE PLAYERS (GATE 1)
# =============================================================================

@timed
def get_active_players(game_id: str, test_mode: bool = False) -> List[Dict]:
    """
    Get active players from a live game's boxscore.
    
    Args:
        game_id: ESPN game ID
        test_mode: If True, return fake data for testing
    
    Returns:
        List of player dictionaries with name, points, minutes, etc.
    """
    if test_mode:
        return generate_fake_active_players(game_id)
    
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        players = []
        
        # Parse boxscore for both teams
        boxscore = data.get('boxscore', {})
        teams = boxscore.get('teams', [])
        
        for team in teams:
            team_name = team['team']['displayName']
            statistics = team.get('statistics', [])
            
            # Find athletes in statistics
            for stat_group in statistics:
                for athlete in stat_group.get('athletes', []):
                    # Get stats
                    stats = athlete.get('stats', [])
                    
                    # Parse points and minutes
                    points = 0
                    minutes = 0
                    
                    for stat_val in stats:
                        if 'PTS' in str(stat_val):
                            try:
                                points = float(stat_val) if stat_val != '--' else 0
                            except:
                                points = 0
                        if 'MIN' in str(stat_val):
                            try:
                                # Minutes might be "25:30" format
                                if ':' in str(stat_val):
                                    mins, secs = str(stat_val).split(':')
                                    minutes = float(mins) + float(secs) / 60
                                else:
                                    minutes = float(stat_val) if stat_val != '--' else 0
                            except:
                                minutes = 0
                    
                    # Only include players who are playing
                    if minutes > 0:
                        player_info = {
                            'player_name': athlete['athlete']['displayName'],
                            'player_id': athlete['athlete'].get('id'),
                            'team': team_name,
                            'current_points': points,
                            'minutes_played': minutes,
                        }
                        players.append(player_info)
        
        # Sort by points (descending) and limit to top scorers
        players.sort(key=lambda x: x['current_points'], reverse=True)
        return players[:MAX_PLAYERS_PER_GAME]
    
    except Exception as e:
        print(f"⚠️  Error getting active players for game {game_id}: {e}")
        return []


# =============================================================================
# STEP 3: FETCH LIVE ODDS
# =============================================================================

@timed
def fetch_live_odds(game_id: str, test_mode: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch live player prop odds from The Odds API.
    
    Args:
        game_id: ESPN game ID
        test_mode: If True, return fake data for testing
    
    Returns:
        DataFrame with player props or None if unavailable
    """
    if test_mode:
        return generate_fake_live_odds()
    
    if not ODDS_API_KEY:
        print("⚠️  THE_ODDS_API_KEY not set in environment")
        return None
    
    try:
        # Get player props for NBA
        url = f"{ODDS_API_BASE_URL}/sports/basketball_nba/events/{game_id}/odds"
        params = {
            'apiKey': ODDS_API_KEY,
            'regions': 'us',
            'markets': 'player_points',
            'oddsFormat': 'american'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 404:
            # Game not found in The Odds API (might use different ID system)
            return None
        
        response.raise_for_status()
        data = response.json()
        
        # Parse odds data
        odds_records = []
        
        for bookmaker in data.get('bookmakers', []):
            bookmaker_name = bookmaker['key']
            
            for market in bookmaker.get('markets', []):
                if market['key'] == 'player_points':
                    for outcome in market.get('outcomes', []):
                        odds_records.append({
                            'bookmaker': bookmaker_name,
                            'player_name': outcome.get('description'),
                            'line': outcome.get('point'),
                            'side': outcome.get('name'),  # Over or Under
                            'odds': outcome.get('price'),
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
        
        if odds_records:
            return pd.DataFrame(odds_records)
        else:
            return None
    
    except Exception as e:
        print(f"⚠️  Error fetching live odds: {e}")
        return None


def filter_stale_odds(odds_df: pd.DataFrame, max_age_seconds: int = MAX_DATA_AGE_SECONDS) -> pd.DataFrame:
    """
    Filter out stale odds at the bookmaker level (Gate 3 pre-check).
    If ANY odds from a bookmaker are older than max_age_seconds, remove ALL odds from that bookmaker.
    
    Args:
        odds_df: DataFrame with live odds
        max_age_seconds: Maximum age in seconds for fresh data
    
    Returns:
        DataFrame with only fresh bookmakers
    """
    if len(odds_df) == 0:
        return odds_df
    
    now = datetime.now(timezone.utc)
    
    # Convert timestamp to datetime
    odds_df = odds_df.copy()
    odds_df['timestamp_dt'] = pd.to_datetime(odds_df['timestamp'], utc=True)
    odds_df['age_seconds'] = (now - odds_df['timestamp_dt']).dt.total_seconds()
    
    # Find bookmakers with ANY stale data
    stale_bookmakers = odds_df[odds_df['age_seconds'] > max_age_seconds]['bookmaker'].unique()
    
    if len(stale_bookmakers) > 0:
        print(f"   ⚠️  Filtering out stale bookmakers: {', '.join(stale_bookmakers)}")
    
    # Keep only fresh bookmakers
    fresh_df = odds_df[~odds_df['bookmaker'].isin(stale_bookmakers)].copy()
    fresh_df = fresh_df.drop(columns=['timestamp_dt', 'age_seconds'])
    
    num_fresh_books = len(fresh_df['bookmaker'].unique()) if len(fresh_df) > 0 else 0
    print(f"   ✅ {num_fresh_books} bookmaker(s) with fresh odds (<{max_age_seconds}s)")
    
    return fresh_df


# =============================================================================
# STEP 4: SAVE ODDS TO S3
# =============================================================================

def save_live_odds_to_s3(odds_df: pd.DataFrame) -> bool:
    """
    Save live odds snapshot to S3.
    
    Args:
        odds_df: DataFrame with live odds data
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate timestamp filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        s3_key = f"{S3_LIVE_ODDS_PREFIX}/{timestamp}.parquet"
        
        # Convert to parquet in memory
        parquet_buffer = odds_df.to_parquet(index=False)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=parquet_buffer
        )
        
        return True
    
    except Exception as e:
        print(f"⚠️  Error saving odds to S3: {e}")
        return False


# =============================================================================
# STEP 4B: SAVE SIGNALS TO PARQUET
# =============================================================================

def get_et_date() -> str:
    """
    Get current date in ET timezone (YYYY-MM-DD format).
    Games past midnight ET will be in next day's file.
    """
    et_tz = pytz.timezone('US/Eastern')
    et_now = datetime.now(et_tz)
    return et_now.strftime('%Y-%m-%d')


def save_signals_to_parquet(signals: List[Dict]) -> bool:
    """
    Save signals to parquet files (local and S3).
    Uses DuckDB to append to existing file for the day (ET timezone).
    
    File structure:
    - S3: s3://nba-betting-mt/data/04_output/live_betting_signals/player_points/YYYYMMDD.parquet
    - Local: ~/Downloads/tmp/live_betting_data/signals/YYYYMMDD.parquet
    
    Args:
        signals: List of signal dictionaries
    
    Returns:
        True if successful
    """
    if not signals:
        return True
    
    try:
        import duckdb
        
        # Get ET date for file naming
        game_date = get_et_date()
        date_str = game_date.replace('-', '')
        
        # Prepare data
        signals_df = pd.DataFrame(signals)
        
        # Add timestamp in ET
        et_tz = pytz.timezone('US/Eastern')
        signals_df['save_timestamp_et'] = datetime.now(et_tz)
        signals_df['save_timestamp_utc'] = datetime.now(timezone.utc)
        signals_df['game_date_et'] = game_date
        
        # File paths
        local_file = LOCAL_SIGNALS_DIR / f"{date_str}.parquet"
        s3_key = f"{S3_SIGNALS_PREFIX}/{date_str}.parquet"
        
        # Use DuckDB to append or create
        con = duckdb.connect(database=':memory:')
        
        if local_file.exists():
            # Load existing data and append
            con.execute(f"""
                CREATE TABLE existing AS 
                SELECT * FROM read_parquet('{local_file}')
            """)
            
            con.execute("""
                CREATE TABLE new_signals AS 
                SELECT * FROM signals_df
            """)
            
            con.execute(f"""
                COPY (
                    SELECT * FROM existing
                    UNION ALL
                    SELECT * FROM new_signals
                ) TO '{local_file}' (FORMAT PARQUET)
            """)
        else:
            # Create new file
            con.execute(f"""
                COPY signals_df TO '{local_file}' (FORMAT PARQUET)
            """)
        
        con.close()
        
        # Upload to S3
        with open(local_file, 'rb') as f:
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=f.read()
            )
        
        print(f"      💾 Saved {len(signals)} signal(s) to {date_str}.parquet (local + S3)")
        return True
    
    except Exception as e:
        print(f"      ⚠️  Failed to save signals to parquet: {e}")
        return False


# =============================================================================
# STEP 5: EDGE DETECTION & EV CALCULATION
# =============================================================================

def detect_profitable_bet(
    model_prob_over: float,
    over_odds: int,
    under_odds: int,
    min_edge: float = MIN_EDGE_THRESHOLD
) -> Dict:
    """
    Detect if there's a profitable betting opportunity.
    
    Applies longshot bias adjustment to better estimate true market efficiency.
    Returns both raw and adjusted probabilities for transparency.
    
    Args:
        model_prob_over: Model's probability of going over (0.0 to 1.0)
        over_odds: Live odds for over (American format)
        under_odds: Live odds for under (American format)
        min_edge: Minimum edge required to signal (default 0.15)
    
    Returns:
        Dictionary with action, edge, ev, and other details (before/after adjustment)
    """
    # Convert odds to probabilities (with vig)
    market_prob_over_raw = american_odds_to_prob(over_odds)
    market_prob_under_raw = american_odds_to_prob(under_odds)
    
    # Remove vig to get fair probabilities (standard method)
    market_prob_over_fair, market_prob_under_fair = remove_vig(
        market_prob_over_raw, market_prob_under_raw
    )
    
    # Apply longshot bias adjustment
    market_prob_over_adjusted = adjust_for_longshot_bias(market_prob_over_raw, over_odds)
    market_prob_under_adjusted = adjust_for_longshot_bias(market_prob_under_raw, under_odds)
    
    # Renormalize after adjustment
    total_adjusted = market_prob_over_adjusted + market_prob_under_adjusted
    market_prob_over_adjusted_fair = market_prob_over_adjusted / total_adjusted
    market_prob_under_adjusted_fair = market_prob_under_adjusted / total_adjusted
    
    # Calculate edges (BEFORE adjustment)
    edge_over_before = model_prob_over - market_prob_over_fair
    edge_under_before = (1 - model_prob_over) - market_prob_under_fair
    
    # Calculate edges (AFTER longshot adjustment)
    edge_over_after = model_prob_over - market_prob_over_adjusted_fair
    edge_under_after = (1 - model_prob_over) - market_prob_under_adjusted_fair
    
    # Calculate expected values (use adjusted probabilities for better estimate)
    ev_over = calculate_ev(model_prob_over, over_odds, bet_amount=100)
    ev_under = calculate_ev(1 - model_prob_over, under_odds, bet_amount=100)
    
    # Calculate EV breakdown components for display
    def get_ev_breakdown(model_prob: float, odds: int, bet_amount: float = 100):
        decimal_odds = american_odds_to_decimal(odds)
        win_amount = (decimal_odds - 1) * bet_amount
        prob_win = model_prob
        prob_lose = 1 - model_prob
        expected_win = prob_win * win_amount
        expected_loss = prob_lose * bet_amount
        ev = expected_win - expected_loss
        return {
            'decimal_odds': decimal_odds,
            'win_amount': win_amount,
            'prob_win': prob_win,
            'prob_lose': prob_lose,
            'expected_win': expected_win,
            'expected_loss': expected_loss,
            'bet_amount': bet_amount
        }
    
    # Determine recommendation (use adjusted edge for decision)
    if edge_over_after >= min_edge and ev_over > 0:
        ev_breakdown = get_ev_breakdown(model_prob_over, over_odds)
        return {
            'action': 'BET_OVER',
            'edge_before': edge_over_before,
            'edge_after': edge_over_after,
            'ev': ev_over,
            'model_prob': model_prob_over,
            'model_prob_over': model_prob_over,
            'market_prob_fair_before': market_prob_over_fair,
            'market_prob_fair_after': market_prob_over_adjusted_fair,
            'market_prob_implied': market_prob_over_raw,
            'odds': over_odds,
            'bet_side': 'OVER',
            'over_odds': over_odds,
            'under_odds': under_odds,
            'ev_breakdown': ev_breakdown
        }
    elif edge_under_after >= min_edge and ev_under > 0:
        ev_breakdown = get_ev_breakdown(1 - model_prob_over, under_odds)
        return {
            'action': 'BET_UNDER',
            'edge_before': edge_under_before,
            'edge_after': edge_under_after,
            'ev': ev_under,
            'model_prob': 1 - model_prob_over,
            'model_prob_over': model_prob_over,
            'market_prob_fair_before': market_prob_under_fair,
            'market_prob_fair_after': market_prob_under_adjusted_fair,
            'market_prob_implied': market_prob_under_raw,
            'odds': under_odds,
            'bet_side': 'UNDER',
            'over_odds': over_odds,
            'under_odds': under_odds,
            'ev_breakdown': ev_breakdown
        }
    else:
        return {
            'action': 'PASS',
            'edge_before': max(edge_over_before, edge_under_before),
            'edge_after': max(edge_over_after, edge_under_after),
            'ev': max(ev_over, ev_under)
        }


# =============================================================================
# STEP 6: ANALYZE PLAYER BETTING OPPORTUNITY
# =============================================================================

def analyze_player_betting_opportunity(
    player: Dict,
    game: Dict,
    live_odds_df: Optional[pd.DataFrame],
    n_sims: int = N_SIMULATIONS,
    test_mode: bool = False,
    market: str = "player_points"
) -> Optional[Dict]:
    """
    Analyze all betting opportunities for a player across all bookmakers and lines.
    
    Performance gates (all must pass before running MC):
    - Gate 1: Player in live game (already true by this point)
    - Gate 2: PBP data fresh (checked at game level)
    - Gate 3: Player has live odds from at least one bookmaker
    - Gate 4: Player has pregame line for calibration
    
    Process:
    1. Check Gates 3 & 4 (cheap checks)
    2. Load profile and calculate vegas adjustment
    3. Run MC simulation ONCE
    4. Analyze all (bookmaker × line × side) combinations
    5. Return best bet by expected value
    
    Args:
        player: Player info (name, points, minutes)
        game: Game info (game_id, quarter, clock)
        live_odds_df: DataFrame with live odds (already filtered for freshness)
        n_sims: Number of Monte Carlo simulations
        test_mode: If True, use fake pregame lines
        market: Betting market to analyze (default "player_points")
    
    Returns:
        Signal dictionary with best bet if profitable, None otherwise
    """
    player_name = player['player_name']
    current_points = player['current_points']
    
    try:
        # =====================================================================
        # GATE 3: Check if player has live odds
        # =====================================================================
        if live_odds_df is None or len(live_odds_df) == 0:
            print(f"      ⚪ No live odds available (Gate 3 failed)")
            return None
        
        player_odds = live_odds_df[
            live_odds_df['player_name'].str.contains(player_name, case=False, na=False)
        ]
        
        if len(player_odds) == 0:
            print(f"      ⚪ No live odds for player (Gate 3 failed)")
            return None
        
        num_bookmakers = len(player_odds['bookmaker'].unique())
        num_lines = len(player_odds['line'].unique())
        print(f"      ✅ Found odds: {num_bookmakers} bookmaker(s), {num_lines} line(s) (Gate 3 passed)")
        
        # =====================================================================
        # GATE 4: Check if player has pregame line
        # =====================================================================
        if test_mode:
            # Use fake pregame lines for testing
            fake_pregame_lines = {
                'Nikola Jokic': 26.5,
                'Jamal Murray': 22.5,
                'LeBron James': 25.5
            }
            pregame_line = fake_pregame_lines.get(player_name)
            if pregame_line:
                print(f"      ✅ Pregame line (TEST): {pregame_line} (Gate 4 passed)")
        else:
            pregame_line = get_consensus_prop_line(
                player_name,
                game['game_date'],
                market=market
            )
            if pregame_line:
                print(f"      ✅ Pregame line: {pregame_line} (Gate 4 passed)")
        
        if not pregame_line:
            print(f"      ⚪ No pregame line found (Gate 4 failed)")
            return None
        
        # =====================================================================
        # ALL GATES PASSED - Proceed with expensive operations
        # =====================================================================
        
        # Load player profile
        player_profile = load_player_profile(player_name)
        
        # Calculate Vegas adjustment for calibration
        vegas_adjustment = find_vegas_adjustment(
            player_profile,
            pregame_line,
            n_simulations=5000
        )
        
        # Calculate current game minute
        quarter = game['quarter']
        clock = game['clock']
        
        try:
            if ':' in clock:
                mins, secs = map(int, clock.split(':'))
                time_remaining = mins + secs / 60.0
            else:
                time_remaining = float(clock) / 60.0
        except:
            time_remaining = 0
        
        quarter_start = (quarter - 1) * 12
        game_minute = quarter_start + (12 - time_remaining)
        
        # Run Monte Carlo simulation ONCE
        print(f"      🎲 Running Monte Carlo ({n_sims:,} sims)...")
        mc_start = time.time()
        model_prob_over = monte_carlo_simulate_bet(
            player_profile=player_profile,
            current_minute=game_minute,
            current_points=current_points,
            prop_line=pregame_line,
            n_simulations=n_sims,
            vegas_adjustment=vegas_adjustment,
            score_differential=None,
            debug=False
        )
        mc_elapsed = time.time() - mc_start
        print(f"         ⏱️  MC completed: {mc_elapsed:.2f}s")
        
        # =====================================================================
        # Analyze all (bookmaker × line × side) combinations
        # =====================================================================
        all_bets = []
        combinations_checked = 0
        
        for bookmaker in player_odds['bookmaker'].unique():
            book_odds = player_odds[player_odds['bookmaker'] == bookmaker]
            
            for line_value in book_odds['line'].unique():
                line_odds = book_odds[book_odds['line'] == line_value]
                
                # Get over/under odds for this specific book + line
                over_row = line_odds[line_odds['side'] == 'Over']
                under_row = line_odds[line_odds['side'] == 'Under']
                
                if len(over_row) == 0 or len(under_row) == 0:
                    continue
                
                over_odds = int(over_row.iloc[0]['odds'])
                under_odds = int(under_row.iloc[0]['odds'])
                
                # Analyze this specific (bookmaker × line) combination
                # detect_profitable_bet checks BOTH over and under internally
                signal = detect_profitable_bet(
                    model_prob_over=model_prob_over,
                    over_odds=over_odds,
                    under_odds=under_odds,
                    min_edge=MIN_EDGE_THRESHOLD
                )
                
                combinations_checked += 1
                
                if signal['action'] != 'PASS':
                    # Package signal with all context
                    signal.update({
                        'bookmaker': bookmaker,
                        'live_line': line_value,
                        'player_name': player_name,
                        'team': player['team'],
                        'current_points': current_points,
                        'minutes_played': player['minutes_played'],
                        'game_minute': game_minute,
                        'game_state': f"Q{quarter} {clock}",
                        'pregame_line': pregame_line,
                        'game_id': game['game_id'],
                        'game_info': f"{game['away_team']} @ {game['home_team']}"
                    })
                    all_bets.append(signal)
        
        print(f"         📊 Analyzed {combinations_checked} (bookmaker × line) combinations")
        
        # Return bet with highest EV
        if not all_bets:
            print(f"      ⚪ No profitable signals found")
            return None
        
        print(f"         ✅ Found {len(all_bets)} profitable bet(s)")
        best_bet = max(all_bets, key=lambda x: x['ev'])
        print(f"      ✅ PROFITABLE SIGNAL (best EV: ${best_bet['ev']:.2f} on {best_bet['bookmaker']} {best_bet['bet_side']} {best_bet['live_line']})")
        return best_bet
    
    except Exception as e:
        print(f"   ⚠️  Error analyzing {player_name}: {e}")
        return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution loop with performance gates."""
    
    parser = argparse.ArgumentParser(description="Live betting signal generator")
    parser.add_argument("--min-edge", type=float, default=0.15, help="Minimum edge threshold (default 0.15)")
    parser.add_argument("--n-sims", type=int, default=1000, help="Number of MC simulations (default 1000)")
    parser.add_argument("--test-with-fake-data", action="store_true", help="Run in test mode with fake data")
    parser.add_argument("--loop", action="store_true", help="Run continuously (scan every N seconds)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between scans when in loop mode (default 60)")
    args = parser.parse_args()
    
    global MIN_EDGE_THRESHOLD, N_SIMULATIONS
    MIN_EDGE_THRESHOLD = args.min_edge
    N_SIMULATIONS = args.n_sims
    test_mode = args.test_with_fake_data
    loop_mode = args.loop
    interval = args.interval
    
    print("="*80)
    print("LIVE BETTING SIGNAL GENERATOR")
    print("="*80)
    print()
    print("📋 Process Overview (with Performance Gates):")
    print("   1. Fetch live games from ESPN")
    print("   2. Validate PBP data freshness (Gate 2: <1min)")
    print("   3. Fetch live odds from The Odds API")
    print("   4. Filter stale odds (Gate 3: <1min at bookmaker level)")
    print("   5. Get active players (Gate 1: in live game)")
    print("   6. For each player:")
    print("      - Check Gates 3 & 4 (has odds + pregame line)")
    print("      - Run MC simulation (only if all gates pass)")
    print("      - Analyze all (bookmaker × line × side) combinations")
    print("      - Return best bet by EV")
    print("   7. Display profitable signals with specific bookmakers")
    print("   8. Save live odds to S3")
    print()
    print(f"⚙️  Configuration:")
    print(f"   - Mode: {'TEST (Fake Data)' if test_mode else 'LIVE (Real Data)'}")
    print(f"   - Loop Mode: {'ON (every {}s)'.format(interval) if loop_mode else 'OFF (run once)'}")
    print(f"   - Min Edge Threshold: {MIN_EDGE_THRESHOLD:.1%}")
    print(f"   - MC Simulations: {N_SIMULATIONS:,}")
    print(f"   - Max Players Per Game: {MAX_PLAYERS_PER_GAME}")
    print(f"   - Max Data Age: {MAX_DATA_AGE_SECONDS}s")
    print()
    
    # Sync to top of minute if in loop mode
    if loop_mode:
        now = datetime.now()
        seconds_into_minute = now.second + now.microsecond / 1_000_000
        
        if seconds_into_minute > 0:
            # Wait until top of next minute
            sleep_seconds = interval - seconds_into_minute
            next_minute = now.replace(second=0, microsecond=0) + timedelta(seconds=interval)
            
            print(f"⏰ Syncing to minute boundary...")
            print(f"   Current time: {now.strftime('%H:%M:%S.%f')[:-3]}")
            print(f"   Next scan at: {next_minute.strftime('%H:%M:%S')} (waiting {sleep_seconds:.1f}s)")
            print()
            time.sleep(sleep_seconds)
    
    iteration = 0
    
    while True:
        iteration += 1
        iteration_start_time = datetime.now()
        
        if loop_mode:
            print("="*80)
            print(f"ITERATION #{iteration} - {iteration_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            print()
        
        # =====================================================================
        # STEP 1: FETCH LIVE GAMES
        # =====================================================================
        print("="*80)
        print(f"STEP 1: Fetching live games {'(TEST MODE)' if test_mode else 'from ESPN'}...")
        print("="*80)
        
        live_games = fetch_live_games(test_mode=test_mode)
        
        if not live_games:
            print("❌ No live games found")
            if not loop_mode:
                return
            else:
                # Calculate time to next minute boundary
                now = datetime.now()
                elapsed_this_iteration = (now - iteration_start_time).total_seconds()
                current_minute = now.replace(second=0, microsecond=0)
                next_target = current_minute + timedelta(seconds=interval)
                
                if next_target <= now:
                    next_target = next_target + timedelta(seconds=interval)
                
                sleep_seconds = (next_target - now).total_seconds()
                
                print(f"   ⏳ Waiting {sleep_seconds:.1f}s (current: {now.strftime('%H:%M:%S.%f')[:-3]}, iteration: {elapsed_this_iteration:.1f}s)")
                print()
                time.sleep(sleep_seconds)
                continue
        
        print(f"✅ Found {len(live_games)} live game(s)")
        for game in live_games:
            print(f"   🏀 {game['away_team']} ({game['away_score']}) @ {game['home_team']} ({game['home_score']})")
            print(f"      Q{game['quarter']} - {game['clock']}")
        print()
        
        # Collect all signals
        all_signals = []
        
        for game in live_games:
            game_id = game['game_id']
            
            # =================================================================
            # STEP 2: VALIDATE PBP FRESHNESS (GATE 2)
            # =================================================================
            print("="*80)
            print(f"STEP 2: Validating PBP data for {game['away_team']} @ {game['home_team']} (Gate 2)...")
            print("="*80)
            
            pbp_data = fetch_and_validate_pbp(game_id, max_age_seconds=MAX_DATA_AGE_SECONDS, test_mode=test_mode)
            
            if not pbp_data:
                print("   ⏭️  Skipping game (PBP data stale or unavailable)")
                print()
                continue
            
            print()
            
            # =================================================================
            # STEP 3: FETCH LIVE ODDS
            # =================================================================
            print("="*80)
            print(f"STEP 3: Fetching live odds {'(TEST MODE)' if test_mode else 'from The Odds API'}...")
            print("="*80)
            
            live_odds_df = fetch_live_odds(game_id, test_mode=test_mode)
            
            if live_odds_df is None or len(live_odds_df) == 0:
                print("⚠️  No live odds available for this game")
                print()
                continue
            
            print(f"✅ Fetched odds for {live_odds_df['player_name'].nunique()} player(s)")
            
            # Filter stale odds (Gate 3 pre-check at bookmaker level)
            live_odds_df = filter_stale_odds(live_odds_df, max_age_seconds=MAX_DATA_AGE_SECONDS)
            
            if len(live_odds_df) == 0:
                print("⚠️  No fresh odds available (all bookmakers stale)")
                print()
                continue
            
            print()
            
            # =================================================================
            # STEP 4: SAVE ODDS TO S3
            # =================================================================
            print("="*80)
            print(f"STEP 4: Saving live odds to S3...")
            print("="*80)
            
            if test_mode:
                print("⏭️  Skipped (test mode)")
            elif save_live_odds_to_s3(live_odds_df):
                print("✅ Saved to S3")
            else:
                print("⚠️  Failed to save to S3")
            print()
            
            # =================================================================
            # STEP 5: GET ACTIVE PLAYERS (GATE 1)
            # =================================================================
            print("="*80)
            print(f"STEP 5: Getting active players {'(TEST MODE)' if test_mode else ''} (Gate 1)...")
            print("="*80)
            
            players = get_active_players(game_id, test_mode=test_mode)
            
            if not players:
                print("⚠️  No active players found")
                print()
                continue
            
            print(f"✅ Found {len(players)} active player(s)")
            for p in players:
                print(f"   - {p['player_name']} ({p['team']}): {p['current_points']} pts, {p['minutes_played']:.1f} min")
            print()
            
            # =================================================================
            # STEP 6: ANALYZE EACH PLAYER (GATES 3 & 4, THEN MC)
            # =================================================================
            print("="*80)
            print(f"STEP 6: Analyzing players (Gates 3 & 4, then MC if passed)...")
            print("="*80)
            
            for player in players:
                print(f"   🔄 Analyzing {player['player_name']}...")
                
                signal = analyze_player_betting_opportunity(
                    player, game, live_odds_df, n_sims=N_SIMULATIONS, test_mode=test_mode
                )
                
                if signal:
                    all_signals.append(signal)
            
            print()
        
        # =====================================================================
        # STEP 7: DISPLAY SIGNALS
        # =====================================================================
        print("="*80)
        print("PROFITABLE BETTING SIGNALS")
        print("="*80)
        print()
        
        if not all_signals:
            print("❌ No profitable betting opportunities found")
        else:
            print(f"🎯 Found {len(all_signals)} profitable signal(s):")
            print()
            
            for i, signal in enumerate(all_signals, 1):
                # Build odds display showing both sides
                if signal['bet_side'] == 'OVER':
                    odds_display = f"OVER {signal['live_line']} @ {signal['over_odds']:+d} (UNDER {signal['live_line']} @ {signal['under_odds']:+d})"
                    model_display = f"{signal['model_prob']:.1%} (OVER)"
                else:
                    odds_display = f"UNDER {signal['live_line']} @ {signal['under_odds']:+d} (OVER {signal['live_line']} @ {signal['over_odds']:+d})"
                    model_display = f"{signal['model_prob']:.1%} (1 - {signal['model_prob_over']:.1%} over = UNDER)"
                
                # Get EV breakdown
                ev = signal['ev_breakdown']
                
                print(f"Signal #{i} | {signal['player_name']} | {signal['team']}")
                print(f"{'─'*80}")
                print(f"Game:         {signal['game_info']} ({signal['game_state']})")
                print(f"Model Inputs: {signal['current_points']} pts | {signal['game_minute']:.1f} game min | {signal['minutes_played']:.1f} played")
                print(f"Lines:        {signal['pregame_line']} (pregame) → {signal['live_line']} (live)")
                print(f"Bet:          {odds_display} on {signal['bookmaker']}")
                print(f"Probabilities:")
                print(f"  Model:      {model_display}")
                print(f"  Market:     {signal['market_prob_fair_before']:.1%} → {signal['market_prob_fair_after']:.1%} (longshot adj)")
                print(f"  Edge:       {signal['edge_before']:.1%} → {signal['edge_after']:.1%}")
                print(f"EV Breakdown:")
                print(f"  Decimal Odds:     {ev['decimal_odds']:.3f}")
                print(f"  Win Amount:       ${ev['win_amount']:.2f} (per ${ev['bet_amount']:.0f} bet)")
                print(f"  Expected Win:     {ev['prob_win']:.1%} × ${ev['win_amount']:.2f} = ${ev['expected_win']:.2f}")
                print(f"  Expected Loss:    {ev['prob_lose']:.1%} × ${ev['bet_amount']:.0f} = ${ev['expected_loss']:.2f}")
                print(f"  Expected Value:   ${ev['expected_win']:.2f} - ${ev['expected_loss']:.2f} = ${signal['ev']:.2f}")
                print(f"{'─'*80}")
                print()
            
            # Save all signals to parquet (local + S3)
            print("="*80)
            print("SAVING SIGNALS TO PARQUET...")
            print("="*80)
            save_signals_to_parquet(all_signals)
            print()
        
        print("="*80)
        print("✅ SCAN COMPLETE")
        print("="*80)
        
        # Calculate and show iteration timing
        iteration_end_time = datetime.now()
        elapsed_seconds = (iteration_end_time - iteration_start_time).total_seconds()
        
        print()
        print(f"⏱️  Iteration Summary:")
        print(f"   Total time: {elapsed_seconds:.1f}s")
        print(f"   Games processed: {len(live_games)}")
        print(f"   Signals found: {len(all_signals)}")
        
        if loop_mode and elapsed_seconds > interval * 0.9:
            print(f"   ⚠️  WARNING: Iteration took {elapsed_seconds:.1f}s (>{interval*0.9:.0f}s)")
            print(f"       Consider reducing --n-sims or increasing --interval")
        
        print()
        
        # Exit or continue loop
        if not loop_mode:
            break
        else:
            # Calculate time to next interval boundary
            # Round up to next interval boundary
            current_minute = iteration_end_time.replace(second=0, microsecond=0)
            next_target = current_minute + timedelta(seconds=interval)
            
            # If we're already past the next target, go to the one after
            if next_target <= iteration_end_time:
                next_target = next_target + timedelta(seconds=interval)
            
            sleep_seconds = (next_target - iteration_end_time).total_seconds()
            
            print(f"⏳ Next scan at {next_target.strftime('%H:%M:%S')}")
            print(f"   Current time: {iteration_end_time.strftime('%H:%M:%S.%f')[:-3]} (sleeping {sleep_seconds:.1f}s)")
            print()
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
