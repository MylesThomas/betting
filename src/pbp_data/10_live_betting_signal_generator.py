"""
Live Betting Signal Generator

Purpose:
Scan live NBA games for profitable betting opportunities using Monte Carlo
simulation to detect edges between our model and live betting markets.

Process:
1. Fetch live games from ESPN API
2. Validate PBP data freshness (Gate 2: <1min old)
3. Fetch live odds from The Odds API (exclude bookmakers that don't update in-game, e.g. Bovada; see EXCLUDED_BOOKMAKERS)
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
import urllib3
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
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Disable SSL warnings (we use verify=False for ESPN API)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pbp_data.monte_carlo_utils import (
    load_player_profile,
    monte_carlo_simulate_bet,
    monte_carlo_get_distribution,
    find_vegas_adjustment,
    get_consensus_prop_line,
    get_data_paths
)
from player_team_history.name_normalization import (
    normalize_from_odds_api,
    normalize_from_espn_api
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# The Odds API configuration
ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '')  # Set in .env file
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
MIN_EDGE_THRESHOLD = 0.10  # 10% minimum edge
N_SIMULATIONS = 500  # Default simulations (balance speed vs accuracy; lower = faster iterations)
MAX_PLAYERS_PER_GAME = 30  # Max players to run MC on per game (try all; some skip for 0 min / no pregame)
# Cap model probability to avoid overconfident extreme edges (e.g. 84% UNDER on low-minute players)
MODEL_PROB_FLOOR = 0.15
MODEL_PROB_CAP = 0.85
MAX_PBP_AGE_SECONDS = 300  # Maximum age for PBP data (5 minutes - ESPN can lag during timeouts, halftime, etc.)
MAX_ODDS_AGE_SECONDS = 60  # Maximum age for odds data (1 minute - must be fresh ie. within this 1 min interval)
# Bookmakers to exclude: API returns them with recent last_update but lines are often pregame, not live (e.g. Bovada).
# Set via env EXCLUDED_BOOKMAKERS=bovada or bovada,mybookieag (comma-separated, case-insensitive).
EXCLUDED_BOOKMAKERS = [
    k.strip().lower() for k in os.environ.get('EXCLUDED_BOOKMAKERS', 'bovada').split(',') if k.strip()
]
# Flag and skip lines that are 5+ points off current or pregame (likely stale/wrong market).
LINE_OFF_THRESHOLD_POINTS = 5

# ESPN API
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

# Initialize boto3
s3_client = boto3.client('s3')

# Create a requests session with SSL certificate verification disabled
# (ESPN API has certificate issues, but we trust the endpoint)
SESSION = requests.Session()
SESSION.verify = False
SESSION.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})

# Per-iteration step timings (name -> seconds). Cleared at start of each iteration.
_iteration_timings = {}


@contextmanager
def timed_step(name: str):
    """Record elapsed time for a named step. Accumulates when same name is used (e.g. per-game steps)."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    _iteration_timings[name] = _iteration_timings.get(name, 0) + elapsed


# =============================================================================
# TEAM NAME NORMALIZATION (Odds API → ESPN format)
# =============================================================================

# Map Odds API team names to ESPN team names (ESPN is source of truth)
ODDS_TO_ESPN_TEAM_MAPPING = {
    # Odds API uses "LA", ESPN uses "Los Angeles"
    'LA Lakers': 'Los Angeles Lakers',
    'LA Clippers': 'Los Angeles Clippers',
    
    # Trail Blazers variations
    'Portland Trail Blazers': 'Portland Trail Blazers',
    'Portland Trailblazers': 'Portland Trail Blazers',
    
    # All other teams (map to themselves for completeness)
    'Atlanta Hawks': 'Atlanta Hawks',
    'Boston Celtics': 'Boston Celtics',
    'Brooklyn Nets': 'Brooklyn Nets',
    'Charlotte Hornets': 'Charlotte Hornets',
    'Chicago Bulls': 'Chicago Bulls',
    'Cleveland Cavaliers': 'Cleveland Cavaliers',
    'Dallas Mavericks': 'Dallas Mavericks',
    'Denver Nuggets': 'Denver Nuggets',
    'Detroit Pistons': 'Detroit Pistons',
    'Golden State Warriors': 'Golden State Warriors',
    'Houston Rockets': 'Houston Rockets',
    'Indiana Pacers': 'Indiana Pacers',
    'Memphis Grizzlies': 'Memphis Grizzlies',
    'Miami Heat': 'Miami Heat',
    'Milwaukee Bucks': 'Milwaukee Bucks',
    'Minnesota Timberwolves': 'Minnesota Timberwolves',
    'New Orleans Pelicans': 'New Orleans Pelicans',
    'New York Knicks': 'New York Knicks',
    'Oklahoma City Thunder': 'Oklahoma City Thunder',
    'Orlando Magic': 'Orlando Magic',
    'Philadelphia 76ers': 'Philadelphia 76ers',
    'Phoenix Suns': 'Phoenix Suns',
    'Sacramento Kings': 'Sacramento Kings',
    'San Antonio Spurs': 'San Antonio Spurs',
    'Toronto Raptors': 'Toronto Raptors',
    'Utah Jazz': 'Utah Jazz',
    'Washington Wizards': 'Washington Wizards',
}


def normalize_odds_team_to_espn(odds_team_name: str) -> str:
    """
    Normalize Odds API team name to match ESPN format (ground truth).
    
    ESPN is our ground truth - external APIs conform to ESPN names.
    
    Handles common mismatches like "LA Lakers" → "Los Angeles Lakers".
    
    Args:
        odds_team_name: Team name from Odds API
    
    Returns:
        Normalized team name matching ESPN format
        
    Examples:
        >>> normalize_odds_team_to_espn('LA Lakers')
        'Los Angeles Lakers'
        
        >>> normalize_odds_team_to_espn('Milwaukee Bucks')
        'Milwaukee Bucks'
    """
    return ODDS_TO_ESPN_TEAM_MAPPING.get(odds_team_name, odds_team_name)



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
        response = SESSION.get(ESPN_SCOREBOARD_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        live_games = []
        
        for event in data.get('events', []):
            status = event['status']['type']['name']
            
            # Only include in-progress games
            if status == 'STATUS_IN_PROGRESS':
                competition = event['competitions'][0]
                
                # Convert game date from UTC to ET
                # ESPN returns dates in UTC (e.g., "2026-02-13T02:30:00Z")
                # We need ET date for pregame line lookup
                game_date_utc = pd.to_datetime(event['date'])
                game_date_et = game_date_utc.tz_convert('US/Eastern')
                
                game_info = {
                    'game_id': event['id'],
                    'away_team': competition['competitors'][1]['team']['displayName'],
                    'home_team': competition['competitors'][0]['team']['displayName'],
                    'away_score': int(competition['competitors'][1]['score']),
                    'home_score': int(competition['competitors'][0]['score']),
                    'quarter': event['status']['period'],
                    'clock': event['status']['displayClock'],
                    'game_date': game_date_et.strftime('%Y-%m-%d'),  # YYYY-MM-DD in ET
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
def fetch_and_validate_pbp(game_id: str, max_age_seconds: int = MAX_PBP_AGE_SECONDS, test_mode: bool = False) -> Optional[Dict]:
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
        # In test mode, return realistic fake PBP data with scoring plays
        # Match the current_points from generate_fake_active_players:
        # Jokic: 9 pts, Murray: 6 pts, LeBron: 8 pts
        plays = []
        
        # Nikola Jokic scoring plays (9 pts total: 3+3+2+1)
        plays.extend([
            {'text': 'Nikola Jokic makes 3-pt jump shot', 'period': {'number': 1}, 'clock': {'displayValue': '10:30'}},
            {'text': 'Nikola Jokic makes 3-pt jump shot', 'period': {'number': 1}, 'clock': {'displayValue': '8:15'}},
            {'text': 'Nikola Jokic makes 2-pt layup', 'period': {'number': 2}, 'clock': {'displayValue': '9:45'}},
            {'text': 'Nikola Jokic makes free throw', 'period': {'number': 2}, 'clock': {'displayValue': '9:45'}},
        ])
        
        # Jamal Murray scoring plays (6 pts total: 3+2+1)
        plays.extend([
            {'text': 'Jamal Murray makes 3-pt jump shot', 'period': {'number': 1}, 'clock': {'displayValue': '9:20'}},
            {'text': 'Jamal Murray makes 2-pt jumper', 'period': {'number': 2}, 'clock': {'displayValue': '10:15'}},
            {'text': 'Jamal Murray makes free throw', 'period': {'number': 2}, 'clock': {'displayValue': '8:30'}},
        ])
        
        # LeBron James scoring plays (8 pts total: 3+3+2)
        plays.extend([
            {'text': 'LeBron James makes 3-pt jump shot', 'period': {'number': 1}, 'clock': {'displayValue': '11:00'}},
            {'text': 'LeBron James makes 3-pt jump shot', 'period': {'number': 1}, 'clock': {'displayValue': '6:45'}},
            {'text': 'LeBron James makes 2-pt dunk', 'period': {'number': 2}, 'clock': {'displayValue': '11:30'}},
        ])
        
        # Add some non-scoring plays for realism
        plays.extend([
            {'text': 'Anthony Davis defensive rebound', 'period': {'number': 1}, 'clock': {'displayValue': '7:00'}},
            {'text': 'Nikola Jokic missed 2-pt jump shot', 'period': {'number': 2}, 'clock': {'displayValue': '5:00'}},
        ])
        
        return {'plays': plays}
    
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
        response = SESSION.get(url, timeout=10)
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

def get_active_players_from_odds(live_odds_df: pd.DataFrame) -> List[str]:
    """
    Get active player names from live odds data.
    
    If bookmakers are offering odds on a player, they're active/playing.
    This is more reliable than ESPN boxscore which can lag.
    
    Args:
        live_odds_df: DataFrame with live odds (from fetch_live_odds)
    
    Returns:
        List of unique player names with odds available
    """
    if live_odds_df is None or len(live_odds_df) == 0:
        return []
    
    # Get unique player names from odds data
    player_names = live_odds_df['player_name'].unique().tolist()
    return player_names


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
        response = SESSION.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        players = []
        
        # Parse boxscore for both teams
        # CRITICAL: ESPN API has TWO keys in boxscore - DO NOT MIX THEM UP:
        #   1. boxscore['teams'] = TEAM-level stats (FG%, turnovers, etc.) - NO player data
        #   2. boxscore['players'] = PLAYER-level stats (individual athletes with points, minutes, etc.)
        # 
        # This API works for BOTH live and completed games (STATUS_FINAL).
        # ALWAYS use boxscore['players'] to get individual player statistics.
        boxscore = data.get('boxscore', {})
        
        # Grab both for visibility (even though we only use 'players')
        teams_data = boxscore.get('teams', [])
        players_data = boxscore.get('players', [])
        
        # Print what we found in each
        print(f"      📊 Boxscore data fetched:")
        print(f"         - boxscore['teams']: {len(teams_data)} team(s) with TEAM-level stats (FG%, turnovers, etc.)")
        if teams_data:
            team_stats = teams_data[0].get('statistics', [])
            has_athletes = any('athletes' in stat for stat in team_stats)
            print(f"           → {len(team_stats)} stat group(s), has athletes: {has_athletes}")
        
        print(f"         - boxscore['players']: {len(players_data)} team(s) with PLAYER-level stats")
        if players_data:
            player_stats = players_data[0].get('statistics', [])
            has_athletes = any('athletes' in stat for stat in player_stats)
            if has_athletes and player_stats:
                num_athletes = len(player_stats[0].get('athletes', []))
                print(f"           → {len(player_stats)} stat group(s), athletes in first group: {num_athletes}")
        
        # Use players_data (NOT teams_data) for actual parsing
        for team in players_data:
            team_name = team['team']['displayName']
            statistics = team.get('statistics', [])
            
            # Find athletes in statistics
            for stat_group in statistics:
                # Get stat labels to know which position is which
                labels = stat_group.get('labels', [])
                
                for athlete in stat_group.get('athletes', []):
                    # Get stats array (matches order of labels)
                    stats = athlete.get('stats', [])
                    
                    if not stats or len(stats) == 0:
                        continue
                    
                    # Parse stats by matching with labels
                    points = 0
                    minutes = 0
                    
                    for i, label in enumerate(labels):
                        if i >= len(stats):
                            break
                        
                        stat_val = stats[i]
                        
                        if label == 'PTS':
                            try:
                                points = float(stat_val) if stat_val != '--' else 0
                            except:
                                points = 0
                        elif label == 'MIN':
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
                        raw_name = athlete['athlete']['displayName']
                        # Normalize ESPN player name to match odds data normalization
                        normalized_name = normalize_from_espn_api(raw_name)
                        
                        # Skip if normalization failed
                        if not normalized_name:
                            continue
                        
                        player_info = {
                            'player_name': normalized_name,  # Use normalized name
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

def fetch_odds_api_events() -> Dict[Tuple[str, str], str]:
    """
    Fetch all NBA events from The Odds API and return a lookup dict.
    
    Normalizes Odds API team names to ESPN format (ground truth) before creating lookup.
    This way we can match directly with ESPN team names without transformation.
    
    Returns:
        Dict mapping (away_team, home_team) in ESPN format -> odds_event_id
    """
    if not ODDS_API_KEY:
        return {}
    
    url = f"{ODDS_API_BASE_URL}/sports/basketball_nba/events"
    params = {'apiKey': ODDS_API_KEY}
    
    try:
        response = SESSION.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        events = response.json()
        
        # Build lookup with ESPN team names (ground truth)
        lookup = {}
        for event in events:
            odds_away = event.get('away_team')
            odds_home = event.get('home_team')
            event_id = event.get('id')
            
            if odds_away and odds_home and event_id:
                # Normalize Odds API team names to ESPN format
                espn_away = normalize_odds_team_to_espn(odds_away)
                espn_home = normalize_odds_team_to_espn(odds_home)
                
                # Store with ESPN team names as keys
                lookup[(espn_away, espn_home)] = event_id
        
        return lookup
        
    except Exception as e:
        print(f"⚠️  Error fetching Odds API events: {e}")
        return {}


def match_espn_to_odds_event(game: Dict, odds_lookup: Dict[Tuple[str, str], str], verbose: bool = False) -> Optional[str]:
    """
    Match ESPN game to Odds API event by team names.
    
    Since odds_lookup already uses ESPN team names (normalized when fetched),
    we can do direct matching without transformation.
    
    Args:
        game: ESPN game dict with 'home_team' and 'away_team' (ESPN format)
        odds_lookup: Dict mapping (away_team, home_team) in ESPN format -> odds_event_id
        verbose: If True, print debug info when matching fails
    
    Returns:
        Odds API event ID or None if no match
    """
    espn_away = game.get('away_team')
    espn_home = game.get('home_team')
    
    if not espn_away or not espn_home:
        return None
    
    # Try exact match (should work now since lookup uses ESPN names)
    if (espn_away, espn_home) in odds_lookup:
        return odds_lookup[(espn_away, espn_home)]
    
    # Fallback: fuzzy match in case of minor variations
    for (lookup_away, lookup_home), event_id in odds_lookup.items():
        if (espn_away.lower() in lookup_away.lower() or lookup_away.lower() in espn_away.lower()) and \
           (espn_home.lower() in lookup_home.lower() or lookup_home.lower() in espn_home.lower()):
            if verbose:
                print(f"   ⚠️  Fuzzy match: ESPN ({espn_away} @ {espn_home}) ↔ Lookup ({lookup_away} @ {lookup_home})")
            return event_id
    
    # No match found - print debug info
    if verbose:
        print(f"   ❌ No match for: {espn_away} @ {espn_home}")
        print(f"      Available games in lookup:")
        for (lookup_away, lookup_home), event_id in odds_lookup.items():
            print(f"        - {lookup_away} @ {lookup_home}")
    
    return None


@timed
def fetch_live_odds(game: Dict, odds_lookup: Dict[Tuple[str, str], str], test_mode: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch live player prop odds from The Odds API.
    
    Args:
        game: ESPN game dict with 'home_team', 'away_team', 'game_id'
        odds_lookup: Dict mapping (away_team, home_team) -> odds_event_id
        test_mode: If True, return fake data for testing
    
    Returns:
        DataFrame with player props or None if unavailable
    """
    if test_mode:
        return generate_fake_live_odds()
    
    if not ODDS_API_KEY:
        print("⚠️  ODDS_API_KEY not set in environment")
        return None
    
    # Match ESPN game to Odds API event
    odds_event_id = match_espn_to_odds_event(game, odds_lookup, verbose=True)
    
    if not odds_event_id:
        print(f"⚠️  No matching Odds API event for {game.get('away_team')} @ {game.get('home_team')}")
        return None
    
    try:
        # Get player props for NBA using Odds API event ID
        url = f"{ODDS_API_BASE_URL}/sports/basketball_nba/events/{odds_event_id}/odds"
        params = {
            'apiKey': ODDS_API_KEY,
            'regions': 'us',
            'markets': 'player_points', # Hard-coding this for now 
            'oddsFormat': 'american'
        }
        
        response = SESSION.get(url, params=params, timeout=10)
        
        if response.status_code == 404:
            # Game not found in The Odds API (might use different ID system)
            return None
        
        response.raise_for_status()
        data = response.json()
        
        # Parse odds data
        odds_records = []
        
        fetch_time_utc = datetime.now(timezone.utc)
        excluded_skipped = []
        for bookmaker in data.get('bookmakers', []):
            bookmaker_name = bookmaker['key']
            if bookmaker_name.lower() in EXCLUDED_BOOKMAKERS:
                excluded_skipped.append(bookmaker_name)
                continue
            # API provides last_update at bookmaker level (when that book's odds were last updated)
            bookmaker_last_update = bookmaker.get('last_update')  # ISO string or None

            for market in bookmaker.get('markets', []):
                if market['key'] == 'player_points':
                    for outcome in market.get('outcomes', []):
                        raw_player_name = outcome.get('description')
                        # Normalize player name from Odds API
                        normalized_name = normalize_from_odds_api(raw_player_name)
                        
                        # Skip if normalization failed (invalid name)
                        if not normalized_name:
                            continue
                        
                        odds_records.append({
                            'bookmaker': bookmaker_name,
                            'raw_player_name': raw_player_name,
                            'player_name': normalized_name,  # Use normalized name
                            'line': outcome.get('point'),
                            'side': outcome.get('name'),  # Over or Under
                            'odds': outcome.get('price'),
                            'timestamp': fetch_time_utc.isoformat(),
                            'bookmaker_last_update': bookmaker_last_update,
                        })

        if excluded_skipped:
            print(f"   ⚠️  Excluded bookmaker(s) (live lines not updated in-game): {', '.join(excluded_skipped)}")
        if odds_records:
            return pd.DataFrame(odds_records)
        else:
            return None
    
    except Exception as e:
        print(f"⚠️  Error fetching live odds: {e}")
        return None


def filter_stale_odds(odds_df: pd.DataFrame, max_age_seconds: int = MAX_ODDS_AGE_SECONDS) -> pd.DataFrame:
    """
    Filter out stale odds at the bookmaker level (Gate 3 pre-check).
    Gate 3 = API/bookmaker fetch freshness (we got this payload recently). It does NOT guarantee
    each line has been updated for current game state; already-decided lines (current_points > line)
    are skipped later when analyzing, and we log those with last_update + age.
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
            'ev_over': ev_over,
            'ev_under': ev_under,
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
            'ev_over': ev_over,
            'ev_under': ev_under,
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

def load_pregame_props_lookup(game_date: str, market: str = "player_points") -> Dict[str, float]:
    """
    Load ALL pregame props for the given date and return normalized lookup dict.
    
    Loads entire S3 file once, normalizes all player names, calculates median lines.
    This is called ONCE at script start to avoid repeated S3 reads.
    
    Args:
        game_date: Game date as string "YYYY-MM-DD"
        market: Market type (default: "player_points")
    
    Returns:
        Dict mapping normalized_player_name → median_pregame_line
        Empty dict if file not found or error
    """
    try:
        import duckdb
        import boto3
        
        # S3 path for pregame props
        s3_path = f"s3://the-odds-api-mt/nba/historical_player_props/2025-26/{game_date}.csv"
        
        # Query S3 to load all player_points props
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(f"SET s3_region='us-east-2';")
        
        # Get AWS credentials from boto3 (reads from ~/.aws/credentials or environment)
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials:
            con.execute(f"SET s3_access_key_id='{credentials.access_key}';")
            con.execute(f"SET s3_secret_access_key='{credentials.secret_key}';")
            if credentials.token:  # For temporary credentials
                con.execute(f"SET s3_session_token='{credentials.token}';")
        
        query = f"""
        SELECT player, prop_line
        FROM read_csv_auto('{s3_path}')
        WHERE market = ?
        """
        
        df = con.execute(query, [market]).df()
        con.close()
        
        if df.empty:
            return {}
        
        # Normalize all player names in S3 data using same normalization as live odds
        df['normalized_name'] = df['player'].apply(normalize_from_odds_api)
        
        # Group by normalized name and calculate median line (consensus)
        lookup = df.groupby('normalized_name')['prop_line'].median().to_dict()
        
        return lookup
        
    except Exception as e:
        print(f"⚠️  Could not load pregame props from S3: {e}")
        return {}


def analyze_player_betting_opportunity(
    player: Dict,
    game: Dict,
    live_odds_df: Optional[pd.DataFrame],
    pbp_data: Dict,  # Add PBP data to get current points
    pregame_props_lookup: Dict[str, float],  # Pregame props loaded once at script start
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
        pbp_data: Play-by-play data for extracting current points
        pregame_props_lookup: Dict mapping player_name -> pregame line (preloaded once per iteration)
        n_sims: Number of Monte Carlo simulations
        test_mode: If True, use fake pregame lines
        market: Betting market to analyze (default "player_points")
    
    Returns:
        Signal dictionary with best bet if profitable, None otherwise
    """
    player_name = player['player_name']
    
    # Get current points from PBP data (using text parsing with proper normalization)
    # NOTE: ESPN live PBP API structure differs from cached data:
    # - Live: participants[0].athlete only has 'id' (no displayName)
    # - Cached: participants[0].athlete has full object with 'displayName'
    # Solution: Parse player names from play 'text' field with normalization
    try:
        plays = pbp_data.get('plays', [])
        pbp_points = 0
        
        # Normalize player name from Odds API (already normalized when fetched)
        normalized_player = normalize_from_odds_api(player_name)
        
        if not normalized_player:
            pbp_points = 0
        else:
            for play in plays:
                if play.get('scoringPlay', False):
                    play_text = play.get('text', '')
                    score_val = play.get('scoreValue', 0)
                    
                    # Extract scorer name from play text (always before first "makes")
                    # Example: "P.J. Washington makes 3-foot dunk" → "P.J. Washington"
                    if ' makes ' in play_text.lower():
                        scorer_raw = play_text.split(' makes ')[0].strip()
                        scorer_normalized = normalize_from_espn_api(scorer_raw)
                        
                        # Compare normalized names (Odds API vs ESPN API, both normalized to NBA API format)
                        if scorer_normalized and scorer_normalized == normalized_player:
                            pbp_points += score_val
    except Exception as e:
        print(f"      ⚠️  Could not extract current points from PBP: {e}")
        pbp_points = 0
    
    # Get current points from boxscore for validation
    boxscore_points = player.get('boxscore_points', None)
    
    # Validate PBP vs boxscore points
    if boxscore_points is not None:
        points_diff = abs(pbp_points - boxscore_points)
        if points_diff > 0:
            print(f"      ⚠️  Points mismatch: PBP={pbp_points}, Boxscore={boxscore_points} (diff={points_diff})")
        else:
            print(f"      ✅ Points validated: {pbp_points} pts (PBP matches boxscore)")
    
    # Use PBP points (more reliable as it's what we base game state on)
    current_points = pbp_points
    
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
        # Normalize player name for all lookups (pregame_props_lookup and minute_by_minute both use normalized names)
        normalized_name = normalize_from_odds_api(player_name)
        
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
            pregame_line = pregame_props_lookup.get(normalized_name)
            
            if pregame_line:
                print(f"      ✅ Pregame line: {pregame_line} (Gate 4 passed)")
            else:
                print(f"      ⚪ No pregame line found for '{player_name}' (normalized: '{normalized_name}') (Gate 4 failed)")
        
        if not pregame_line:
            return None
        
        # =====================================================================
        # ALL GATES PASSED - Proceed with expensive operations
        # =====================================================================
        
        # Load player profile (use normalized name for minute_by_minute lookup)
        player_profile = load_player_profile(normalized_name)
        
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
        
        # Run Monte Carlo simulation ONCE to get full distribution
        # This is calibrated with vegas_adjustment (which was set using pregame line)
        print(f"      🎲 Running Monte Carlo ({n_sims:,} sims) - calibrated with pregame line {pregame_line}...")
        mc_start = time.time()
        
        from pbp_data.monte_carlo_utils import monte_carlo_get_distribution
        
        simulated_finals = monte_carlo_get_distribution(
            player_profile=player_profile,
            current_minute=game_minute,
            current_points=current_points,
            n_simulations=n_sims,
            vegas_adjustment=vegas_adjustment,
            score_differential=None,
            debug=False
        )
        
        mc_elapsed = time.time() - mc_start
        print(f"         ⏱️  MC completed: {mc_elapsed:.2f}s")
        print(f"         📊 Distribution: {len(simulated_finals)} simulations, range [{min(simulated_finals):.1f}, {max(simulated_finals):.1f}]")
        
        # =====================================================================
        # Analyze all (bookmaker × line × side) combinations
        # Calculate probability for each line from the same distribution
        # =====================================================================
        all_bets = []
        combinations_checked = 0
        combos_analyzed = []  # (bookmaker, line, over_odds, under_odds) for logging
        combo_maths = []  # (p_over, ev_over, ev_under) per combo, for print after list
        
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
                
                # Skip already-decided / stale lines: if player already has more than the line, OVER has hit.
                # Books shouldn't still offer this; if the API returns it, treat as stale and skip.
                if current_points > line_value:
                    row = line_odds.iloc[0]
                    last_update_raw = row.get('bookmaker_last_update') or row.get('timestamp')
                    try:
                        if last_update_raw:
                            if isinstance(last_update_raw, str):
                                last_dt = datetime.fromisoformat(last_update_raw.replace('Z', '+00:00'))
                            else:
                                last_dt = pd.to_datetime(last_update_raw, utc=True).to_pydatetime()
                            if last_dt.tzinfo is None:
                                last_dt = last_dt.replace(tzinfo=timezone.utc)
                            now_utc = datetime.now(timezone.utc)
                            age_sec = (now_utc - last_dt).total_seconds()
                            age_str = f"{age_sec:.0f}s ago" if age_sec < 60 else f"{age_sec / 60:.1f} min ago"
                            print(f"         🔍 DEBUG: Skipping stale line {line_value} (player has {current_points} pts) | last_update={last_dt.strftime('%H:%M:%S')} UTC, {age_str}")
                        else:
                            print(f"         🔍 DEBUG: Skipping stale line {line_value} (player has {current_points} pts) | no last_update")
                    except Exception:
                        print(f"         🔍 DEBUG: Skipping stale line {line_value} (player has {current_points} pts)")
                    continue
                
                # Calculate probability for this line from the distribution
                # Count how many simulations went over this line
                hits_over = sum(1 for final_pts in simulated_finals if final_pts > line_value)
                raw_prob_over = hits_over / len(simulated_finals)
                
                # Apply same calibration steps as monte_carlo_simulate_bet
                # (minimum probability floor, confidence limits, empirical calibration, conservative factor)
                from pbp_data.monte_carlo_utils import (
                    apply_confidence_limits,
                    apply_calibration,
                    get_game_state,
                    CONSERVATIVE_FACTOR
                )
                
                game_state = get_game_state(game_minute)
                
                # Apply minimum probability floor
                MIN_PROB = 0.001
                if raw_prob_over < MIN_PROB and game_minute < 63:
                    prob_over_floored = MIN_PROB
                else:
                    prob_over_floored = raw_prob_over
                
                # Apply confidence limits
                prob_over_limited = apply_confidence_limits(
                    prob_over_floored, game_minute, current_points, line_value
                )
                
                # Apply empirical calibration
                prob_over_calibrated = apply_calibration(
                    prob_over_limited, game_state['quarter']
                )
                
                # Apply conservative bias
                model_prob_over = prob_over_calibrated * CONSERVATIVE_FACTOR
                # Cap to avoid extreme displayed edges (overconfident on low-minute / noisy cases)
                model_prob_over = max(MODEL_PROB_FLOOR, min(MODEL_PROB_CAP, model_prob_over))
                
                # Analyze this specific (bookmaker × line) combination
                # detect_profitable_bet checks BOTH over and under internally
                signal = detect_profitable_bet(
                    model_prob_over=model_prob_over,
                    over_odds=over_odds,
                    under_odds=under_odds,
                    min_edge=MIN_EDGE_THRESHOLD
                )
                
                combinations_checked += 1
                combos_analyzed.append((bookmaker, line_value, over_odds, under_odds))
                p_over = model_prob_over
                win_over = (american_odds_to_decimal(over_odds) - 1) * 100
                win_under = (american_odds_to_decimal(under_odds) - 1) * 100
                ev_over = p_over * win_over + (1 - p_over) * (-100)
                ev_under = p_over * (-100) + (1 - p_over) * win_under
                combo_maths.append((p_over, ev_over, ev_under))
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
        for i, (bm, line, over_odds, under_odds) in enumerate(combos_analyzed, 1):
            print(f"            {i}. {bm} {line}  (OVER {over_odds:+d} / UNDER {under_odds:+d})")
            if i <= len(combo_maths):
                p_over, ev_o, ev_u = combo_maths[i - 1]
                print(f"               P(over)={p_over:.1%} → EV(OVER) ${ev_o:.2f}, EV(UNDER) ${ev_u:.2f}")
        print()
        # Return bet with highest EV
        if not all_bets:
            print(f"      ⚪ No profitable signals found")
            return None
        
        print(f"         ✅ Found {len(all_bets)} profitable bet(s)")
        for i, bet in enumerate(sorted(all_bets, key=lambda x: -x['ev']), 1):
            p_over = bet['model_prob_over']
            p_under = 1 - p_over
            win_over = (american_odds_to_decimal(bet['over_odds']) - 1) * 100
            win_under = (american_odds_to_decimal(bet['under_odds']) - 1) * 100
            if bet['bet_side'] == 'OVER':
                outcome_over, outcome_under = win_over, -100
            else:
                outcome_over, outcome_under = -100, win_under
            term1 = p_over * outcome_over
            term2 = p_under * outcome_under
            print(f"            {i}. {bet['bookmaker']} {bet['bet_side']} {bet['live_line']}  P(over)={p_over:.1%}×(${outcome_over:+.2f}) + P(under)={p_under:.1%}×(${outcome_under:+.2f}) = ${bet['ev']:.2f}")
            print(f"               EV({bet['bet_side']}) = {term1:.2f} + ({term2:.2f}) = ${bet['ev']:.2f}")
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
    global MIN_EDGE_THRESHOLD, N_SIMULATIONS, MAX_PLAYERS_PER_GAME
    # Parser defaults from module constants
    _d_min_edge = MIN_EDGE_THRESHOLD
    _d_n_sims = N_SIMULATIONS
    _d_max_players = MAX_PLAYERS_PER_GAME

    parser = argparse.ArgumentParser(description="Live betting signal generator")
    parser.add_argument("--min-edge", type=float, default=_d_min_edge, help=f"Minimum edge threshold (default {_d_min_edge})")
    parser.add_argument("--n-sims", type=int, default=_d_n_sims, help=f"Number of MC simulations (default {_d_n_sims}; lower = faster)")
    parser.add_argument("--max-players", type=int, default=_d_max_players, help=f"Max players to analyze per game (default {_d_max_players}; try all, some skip)")
    parser.add_argument("--test-with-fake-data", action="store_true", help="Run in test mode with fake data")
    parser.add_argument("--loop", action="store_true", help="Run continuously (scan every N seconds)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between scans when in loop mode (default 60)")
    args = parser.parse_args()

    MIN_EDGE_THRESHOLD = args.min_edge
    N_SIMULATIONS = args.n_sims
    MAX_PLAYERS_PER_GAME = args.max_players
    test_mode = args.test_with_fake_data
    loop_mode = args.loop
    interval = args.interval
    
    print("="*80)
    print("LIVE BETTING SIGNAL GENERATOR")
    print("="*80)
    print()
    print("📋 Process Overview (with Performance Gates):")
    print("   1. Fetch live games from ESPN")
    print("   2. Load pregame props from S3 (once for all games)")
    print("   3. Fetch Odds API events for team name matching")
    print("   4. For each game:")
    print("      a. Validate PBP data freshness (Gate 2: <5min)")
    print("      b. Fetch live odds from The Odds API")
    print("      c. Filter stale odds (Gate 3: <1min at bookmaker level)")
    print("      d. Get active players (Gate 1: in live game)")
    print("      e. For each player:")
    print("         - Check Gates 3 & 4 (has odds + pregame line)")
    print("         - Run MC simulation (only if all gates pass)")
    print("         - Analyze all (bookmaker × line × side) combinations")
    print("         - Return best bet by EV")
    print("   5. Display profitable signals with specific bookmakers")
    print("   6. Save signals to parquet")
    print()
    print(f"⚙️  Configuration:")
    print(f"   - Mode: {'TEST (Fake Data)' if test_mode else 'LIVE (Real Data)'}")
    print(f"   - Loop Mode: {'ON (every {}s)'.format(interval) if loop_mode else 'OFF (run once)'}")
    print(f"   - Min Edge Threshold: {MIN_EDGE_THRESHOLD:.1%}")
    print(f"   - MC Simulations: {N_SIMULATIONS:,}")
    print(f"   - Max Players Per Game: {MAX_PLAYERS_PER_GAME}")
    print(f"   - Model prob cap: [{MODEL_PROB_FLOOR:.0%}, {MODEL_PROB_CAP:.0%}] (avoids extreme edges)")
    print(f"   - Max PBP Age: {MAX_PBP_AGE_SECONDS}s (ESPN can lag)")
    print(f"   - Max Odds Age: {MAX_ODDS_AGE_SECONDS}s (must be fresh)")
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
        _iteration_timings.clear()
        
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
        
        with timed_step("Step 1: Fetch live games"):
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
        
        # =====================================================================
        # STEP 1.5: LOAD PREGAME PROPS (ONCE FOR ALL GAMES)
        # =====================================================================
        if not test_mode:
            print("="*80)
            print("STEP 1.5: Loading pregame props from S3...")
            print("="*80)
            
            with timed_step("Step 1.5: Load pregame props"):
                et_tz = pytz.timezone('US/Eastern')
                game_date_et = datetime.now(et_tz).strftime('%Y-%m-%d')
                pregame_props_lookup = load_pregame_props_lookup(game_date_et)
                if len(pregame_props_lookup) == 0 or datetime.now(et_tz).hour < 6:
                    yesterday_et = (datetime.now(et_tz) - timedelta(days=1)).strftime('%Y-%m-%d')
                    print(f"   ⚠️  Trying yesterday's pregame lines ({yesterday_et}) as fallback...")
                    pregame_props_yesterday = load_pregame_props_lookup(yesterday_et)
                    if len(pregame_props_yesterday) > 0:
                        pregame_props_lookup.update(pregame_props_yesterday)
                        print(f"   ✅ Loaded {len(pregame_props_yesterday)} additional player(s) from {yesterday_et}")
                        print(f"   📊 Total pregame lines: {len(pregame_props_lookup)} player(s)")
            
            print(f"✅ Loaded pregame lines for {len(pregame_props_lookup)} player(s) from {game_date_et}")
            print()
        else:
            # In test mode, we don't need pregame props (will use fake data)
            pregame_props_lookup = {}
        
        # =====================================================================
        # STEP 1.6: FETCH ODDS API EVENTS FOR MATCHING
        # =====================================================================
        if not test_mode:
            print("="*80)
            print("STEP 1.6: Fetching Odds API events for matching...")
            print("="*80)
            with timed_step("Step 1.6: Fetch Odds API events"):
                odds_lookup = fetch_odds_api_events()
            print(f"✅ Found {len(odds_lookup)} Odds API event(s)")
            print()
        else:
            # In test mode, we don't need the lookup
            odds_lookup = {}
        
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
            
            with timed_step("Step 2: PBP validate"):
                pbp_data = fetch_and_validate_pbp(game_id, max_age_seconds=MAX_PBP_AGE_SECONDS, test_mode=test_mode)
            
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
            
            with timed_step("Step 3: Fetch live odds"):
                live_odds_df = fetch_live_odds(game, odds_lookup, test_mode=test_mode)
            
            if live_odds_df is None or len(live_odds_df) == 0:
                print("⚠️  No live odds available for this game")
                print()
                continue
            
            print(f"✅ Fetched odds for {live_odds_df['player_name'].nunique()} player(s)")
            
            with timed_step("Step 3b: Filter stale odds"):
                live_odds_df = filter_stale_odds(live_odds_df, max_age_seconds=MAX_ODDS_AGE_SECONDS)
            
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
            else:
                with timed_step("Step 4: Save odds to S3"):
                    if save_live_odds_to_s3(live_odds_df):
                        print("✅ Saved to S3")
                    else:
                        print("⚠️  Failed to save to S3")
            print()
            
            # =================================================================
            # STEP 5: GET ACTIVE PLAYERS FROM ODDS (GATE 1) + FETCH BOXSCORE
            # =================================================================
            print("="*80)
            print(f"STEP 5: Getting active players from odds data (Gate 1)...")
            print("="*80)
            
            with timed_step("Step 5: Active players + boxscore + PBP"):
                active_player_names = get_active_players_from_odds(live_odds_df)
            
            if not active_player_names:
                print("⚠️  No players with odds available")
                print()
                continue
            
            # Cap players analyzed per game to control iteration time (MC is the bottleneck)
            total_with_odds = len(active_player_names)
            active_player_names = active_player_names[:MAX_PLAYERS_PER_GAME]
            if total_with_odds > MAX_PLAYERS_PER_GAME:
                print(f"✅ Found {total_with_odds} player(s) with live odds, analyzing first {len(active_player_names)} (--max-players={MAX_PLAYERS_PER_GAME})")
            else:
                print(f"✅ Found {len(active_player_names)} player(s) with live odds")
            
            # Fetch boxscore for validation (optional - won't block if fails)
            print()
            print("   📊 Fetching boxscore for points validation...")
            with timed_step("Step 5: Active players + boxscore + PBP"):
                boxscore_players = get_active_players(game_id, test_mode=test_mode)
            
            # Build lookup: player_name -> {current_points, team, minutes_played}
            boxscore_lookup = {}
            if boxscore_players:
                for p in boxscore_players:
                    boxscore_lookup[p['player_name']] = {
                        'current_points': p['current_points'],
                        'team': p['team'],
                        'minutes_played': p['minutes_played']
                    }
                print(f"   ✅ Boxscore fetched: {len(boxscore_players)} players")
            else:
                print(f"   ⚠️  Boxscore unavailable (will use PBP points only)")
            
            # Build PBP points lookup for display
            # NOTE: ESPN live PBP API structure differs from cached data:
            # - Live: participants[0].athlete only has 'id' (no displayName)
            # - Cached: participants[0].athlete has full object with 'displayName'
            # Solution: Parse player names from play 'text' field
            # Play text format: "Player Name makes 3-foot dunk (Assist Name assists)"
            # Scorer is always at START of text before "makes"
            print("   📊 Extracting PBP points...")
            with timed_step("Step 5: Active players + boxscore + PBP"):
                pbp_lookup = {}
                try:
                    plays = pbp_data['plays']
                    
                    for player_name in active_player_names:
                        points = 0
                        
                        # Normalize player name from Odds API
                        # (already normalized when fetched, but be explicit)
                        normalized_player = normalize_from_odds_api(player_name)
                        
                        if not normalized_player:
                            continue
                        
                        for play in plays:
                            if play.get('scoringPlay', False):
                                play_text = play.get('text', '')
                                score_val = play.get('scoreValue', 0)
                                
                                # Extract scorer name from play text (always before first "makes")
                                # Example: "P.J. Washington makes 3-foot dunk" → "P.J. Washington"
                                if ' makes ' in play_text.lower():
                                    scorer_raw = play_text.split(' makes ')[0].strip()
                                    scorer_normalized = normalize_from_espn_api(scorer_raw)
                                    
                                    # Compare normalized names
                                    if scorer_normalized and scorer_normalized == normalized_player:
                                        points += score_val
                        
                        pbp_lookup[player_name] = points
                except Exception as e:
                    print(f"   ⚠️  Could not extract PBP points: {e}")
            
            if pbp_lookup:
                print(f"   ✅ PBP points extracted for {len(pbp_lookup)} players")
                zero_point_players = [name for name, pts in pbp_lookup.items() if pts == 0]
                if zero_point_players and len(zero_point_players) <= 3:
                    print(f"   🔍 Players with 0 points: {', '.join(zero_point_players)}")
                    print(f"      (This is normal if they haven't scored yet)")
            
            # Show players with both PBP and boxscore points (ALL players)
            for name in active_player_names:
                pbp_pts = pbp_lookup.get(name, '?')
                boxscore_info = boxscore_lookup.get(name, {})
                boxscore_pts = boxscore_info.get('current_points', '?')
                
                # Show validation status
                if pbp_pts != '?' and boxscore_pts != '?':
                    if pbp_pts == boxscore_pts:
                        status = '✅'
                    else:
                        status = f'⚠️ (diff: {abs(pbp_pts - boxscore_pts)})'
                else:
                    status = '❓'
                
                print(f"   - {name}: PBP={pbp_pts} pts, Boxscore={boxscore_pts} pts {status}")
            print()
            
            # =================================================================
            # STEP 6: ANALYZE EACH PLAYER (GATES 3 & 4, THEN MC)
            # =================================================================
            print("="*80)
            print(f"STEP 6: Analyzing players (Gates 3 & 4, then MC if passed)...")
            print("="*80)
            et_tz = pytz.timezone("US/Eastern")
            fetch_dt = pd.to_datetime(live_odds_df["timestamp"].iloc[0], utc=True)
            if fetch_dt.tzinfo is None:
                fetch_dt = fetch_dt.replace(tzinfo=timezone.utc)
            fetch_et = fetch_dt.astimezone(et_tz)
            now_et = datetime.now(et_tz)
            last_updates = live_odds_df["bookmaker_last_update"].dropna()
            if len(last_updates):
                bookmaker_dt = pd.to_datetime(last_updates, utc=True).max()
                if hasattr(bookmaker_dt, "to_pydatetime"):
                    bookmaker_dt = bookmaker_dt.to_pydatetime()
                if bookmaker_dt.tzinfo is None:
                    bookmaker_dt = bookmaker_dt.replace(tzinfo=timezone.utc)
                bookmaker_et = bookmaker_dt.astimezone(et_tz)
                bookmaker_str = bookmaker_et.strftime("%b %d %I:%M:%S%p") + " ET"
            else:
                bookmaker_str = "(not available)"
            print(f"   Odds fetch: {fetch_et.strftime('%b %d %I:%M:%S%p')} ET  |  Now: {now_et.strftime('%b %d %I:%M:%S%p')} ET  |  Bookmaker last update: {bookmaker_str}")
            print()
            with timed_step("Step 6: Analyze players (MC)"):
                for player_name in active_player_names:
                    print("=" * 60)
                    print(f"   🔄 Analyzing {player_name}...")
                    
                    # Get boxscore data for this player
                    boxscore_info = boxscore_lookup.get(player_name, {})
                    
                    # Build player dict with all required fields
                    player = {
                        'player_name': player_name,
                        'boxscore_points': boxscore_info.get('current_points'),  # None if not available
                        'team': boxscore_info.get('team', 'Unknown'),
                        'minutes_played': boxscore_info.get('minutes_played', 0),
                    }
                    
                    # Skip players with 0 minutes (possible DNP; don't suggest live points bets)
                    if not player['minutes_played']:
                        print(f"      ⏭️  Skipping (0 min played – possible DNP)")
                        continue
                    
                    signal = analyze_player_betting_opportunity(
                        player, game, live_odds_df, pbp_data, pregame_props_lookup, n_sims=N_SIMULATIONS, test_mode=test_mode
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
        
        with timed_step("Step 7: Display + save signals"):
            if not all_signals:
                print("❌ No profitable betting opportunities found")
            else:
                print(f"🎯 Found {len(all_signals)} profitable signal(s):")
                print()
                # Summary: book, line/odds, model prob vs implied
                print("   Book            | Line & odds              | Model prob | Implied (book)  | Player")
                print("   ----------------|---------------------------|------------|-----------------|-------")
                for signal in all_signals:
                    if signal['bet_side'] == 'OVER':
                        line_odds = f"OVER {signal['live_line']} @ {signal['over_odds']:+d}"
                    else:
                        line_odds = f"UNDER {signal['live_line']} @ {signal['under_odds']:+d}"
                    model_pct = signal['model_prob']
                    implied = signal['market_prob_implied']
                    print(f"   {signal['bookmaker']:<14} | {line_odds:<25} | {model_pct:>9.1%} | {implied:>14.1%} | {signal['player_name']}")
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
        if _iteration_timings:
            print(f"   Step timings (slowest first):")
            for name, secs in sorted(_iteration_timings.items(), key=lambda x: -x[1]):
                print(f"      {name}: {secs:.2f}s")
        
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
