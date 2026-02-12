"""
Live Betting Signal Generator

Purpose:
Scan live NBA games for profitable betting opportunities using Monte Carlo
simulation to detect edges between our model and live betting markets.

Process:
1. Fetch live games from ESPN API
2. Get active players in each game
3. Load player profiles and pregame lines
4. Run Monte Carlo simulations (10k iterations)
5. Fetch live odds from The Odds API
6. Calculate expected value and detect profitable edges
7. Log profitable signals to console
8. Save live odds data to S3

Usage:
    python src/pbp_data/10_live_betting_signal_generator.py
    
    # With custom parameters
    python src/pbp_data/10_live_betting_signal_generator.py --min-edge 0.20 --n-sims 5000

Output:
    - Console: Profitable betting signals
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
from datetime import datetime, timezone
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
S3_LIVE_ODDS_PREFIX = "data/01_input/live_player_odds/the-odds-api"

# Betting parameters
MIN_EDGE_THRESHOLD = 0.15  # 15% minimum edge
N_SIMULATIONS = 10000
MAX_PLAYERS_PER_GAME = 10  # Only check top scorers to save time

# ESPN API
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

# Initialize boto3
s3_client = boto3.client('s3')


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
    Generate fake live odds for testing.
    
    Using standard -110 odds (typical sportsbook juice) but with DIFFERENT lines
    than pregame to test the line adjustment logic.
    
    Pregame lines (from fake_pregame_lines):
    - Nikola Jokic: 26.5
    - Jamal Murray: 22.5
    
    Live lines (have moved based on game flow):
    - Nikola Jokic: 24.5 (moved down 2 pts - he's behind pace)
    - Jamal Murray: 20.5 (moved down 2 pts - he's behind pace)
    
    Odds explanation:
    - -110 = bet $110 to win $100 (52.4% implied prob with vig)
    - This tests the calculation functions with realistic odds
    """
    odds_data = [
        # Nikola Jokic - standard -110 odds (tests normal calculation)
        {'bookmaker': 'draftkings', 'player_name': 'Nikola Jokic', 'line': 24.5, 'side': 'Over', 'odds': -110, 'timestamp': datetime.now(timezone.utc).isoformat()},
        {'bookmaker': 'draftkings', 'player_name': 'Nikola Jokic', 'line': 24.5, 'side': 'Under', 'odds': -110, 'timestamp': datetime.now(timezone.utc).isoformat()},
        {'bookmaker': 'fanduel', 'player_name': 'Nikola Jokic', 'line': 24.5, 'side': 'Over', 'odds': -110, 'timestamp': datetime.now(timezone.utc).isoformat()},
        {'bookmaker': 'fanduel', 'player_name': 'Nikola Jokic', 'line': 24.5, 'side': 'Under', 'odds': -110, 'timestamp': datetime.now(timezone.utc).isoformat()},
        
        # Jamal Murray - standard -110 odds (tests normal calculation)
        {'bookmaker': 'draftkings', 'player_name': 'Jamal Murray', 'line': 20.5, 'side': 'Over', 'odds': -110, 'timestamp': datetime.now(timezone.utc).isoformat()},
        {'bookmaker': 'draftkings', 'player_name': 'Jamal Murray', 'line': 20.5, 'side': 'Under', 'odds': -110, 'timestamp': datetime.now(timezone.utc).isoformat()},
        {'bookmaker': 'fanduel', 'player_name': 'Jamal Murray', 'line': 20.5, 'side': 'Over', 'odds': -110, 'timestamp': datetime.now(timezone.utc).isoformat()},
        {'bookmaker': 'fanduel', 'player_name': 'Jamal Murray', 'line': 20.5, 'side': 'Under', 'odds': -110, 'timestamp': datetime.now(timezone.utc).isoformat()},
        
        # LeBron James - heavily skewed odds (tests longshot bias adjustment)
        # Market heavily favors OVER (-300), making UNDER a longshot (+250)
        # This will show the adjustment in action
        {'bookmaker': 'draftkings', 'player_name': 'LeBron James', 'line': 23.5, 'side': 'Over', 'odds': -300, 'timestamp': datetime.now(timezone.utc).isoformat()},
        {'bookmaker': 'draftkings', 'player_name': 'LeBron James', 'line': 23.5, 'side': 'Under', 'odds': +250, 'timestamp': datetime.now(timezone.utc).isoformat()},
        {'bookmaker': 'fanduel', 'player_name': 'LeBron James', 'line': 23.5, 'side': 'Over', 'odds': -280, 'timestamp': datetime.now(timezone.utc).isoformat()},
        {'bookmaker': 'fanduel', 'player_name': 'LeBron James', 'line': 23.5, 'side': 'Under', 'odds': +230, 'timestamp': datetime.now(timezone.utc).isoformat()},
    ]
    return pd.DataFrame(odds_data)


# =============================================================================
# STEP 1: FETCH LIVE GAMES
# =============================================================================

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
# STEP 2: GET ACTIVE PLAYERS
# =============================================================================

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
    test_mode: bool = False
) -> Optional[Dict]:
    """
    Analyze a single player for betting opportunities.
    
    Args:
        player: Player info (name, points, minutes)
        game: Game info (game_id, quarter, clock)
        live_odds_df: DataFrame with live odds
        n_sims: Number of Monte Carlo simulations
        test_mode: If True, use fake pregame lines
    
    Returns:
        Signal dictionary if profitable, None otherwise
    """
    player_name = player['player_name']
    current_points = player['current_points']
    
    try:
        # Step 1: Load player profile
        player_profile = load_player_profile(player_name)
        
        # Step 2: Get pregame line
        if test_mode:
            # Use fake pregame lines for testing
            fake_pregame_lines = {
                'Nikola Jokic': 26.5,
                'Jamal Murray': 22.5,
                'LeBron James': 25.5  # For skewed odds test
            }
            pregame_line = fake_pregame_lines.get(player_name)
            if pregame_line:
                print(f"      📊 Pregame line (TEST): {pregame_line}")
        else:
            pregame_line = get_consensus_prop_line(
                player_name,
                game['game_date'],
                market="player_points"
            )
            if pregame_line:
                print(f"      📊 Pregame line: {pregame_line}")
        
        if not pregame_line:
            print(f"      ⚠️  No pregame line found for {player_name}")
            return None  # No pregame line available
        
        # Step 3: Calculate Vegas adjustment (cache this in production)
        vegas_adjustment = find_vegas_adjustment(
            player_profile,
            pregame_line,
            n_simulations=5000  # Faster for calibration
        )
        
        # Step 4: Calculate current game minute
        quarter = game['quarter']
        clock = game['clock']
        
        # Parse clock (MM:SS format)
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
        
        # Step 5: Run Monte Carlo simulation
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
        
        # Step 6: Get live odds for this player
        if live_odds_df is None or len(live_odds_df) == 0:
            return None
        
        player_odds = live_odds_df[
            live_odds_df['player_name'].str.contains(player_name, case=False, na=False)
        ]
        
        if len(player_odds) == 0:
            return None  # No live odds for this player
        
        # Get over/under odds (take consensus across bookmakers)
        over_odds_list = player_odds[player_odds['side'] == 'Over']['odds'].tolist()
        under_odds_list = player_odds[player_odds['side'] == 'Under']['odds'].tolist()
        
        if not over_odds_list or not under_odds_list:
            return None
        
        over_odds = int(np.median(over_odds_list))
        under_odds = int(np.median(under_odds_list))
        live_line = player_odds['line'].iloc[0]
        
        # Step 7: Detect edge
        signal = detect_profitable_bet(
            model_prob_over=model_prob_over,
            over_odds=over_odds,
            under_odds=under_odds,
            min_edge=MIN_EDGE_THRESHOLD
        )
        
        if signal['action'] == 'PASS':
            return None
        
        # Step 8: Package signal
        signal.update({
            'player_name': player_name,
            'team': player['team'],
            'current_points': current_points,
            'minutes_played': player['minutes_played'],
            'game_minute': game_minute,
            'game_state': f"Q{quarter} {clock}",
            'pregame_line': pregame_line,
            'live_line': live_line,
            'game_id': game['game_id'],
            'game_info': f"{game['away_team']} @ {game['home_team']}",
        })
        
        return signal
    
    except Exception as e:
        # Don't crash on individual player errors
        print(f"   ⚠️  Error analyzing {player_name}: {e}")
        return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution loop."""
    
    parser = argparse.ArgumentParser(description="Live betting signal generator")
    parser.add_argument("--min-edge", type=float, default=0.15, help="Minimum edge threshold (default 0.15)")
    parser.add_argument("--n-sims", type=int, default=10000, help="Number of MC simulations (default 10000)")
    parser.add_argument("--test-with-fake-data", action="store_true", help="Run in test mode with fake data")
    args = parser.parse_args()
    
    global MIN_EDGE_THRESHOLD, N_SIMULATIONS
    MIN_EDGE_THRESHOLD = args.min_edge
    N_SIMULATIONS = args.n_sims
    test_mode = args.test_with_fake_data
    
    print("="*80)
    print("LIVE BETTING SIGNAL GENERATOR")
    print("="*80)
    print()
    print("📋 Process Overview:")
    print("   1. Fetch live games from ESPN")
    print("   2. Get active players in each game")
    print("   3. Load player profiles and pregame lines")
    print("   4. Run Monte Carlo simulations (10k iterations)")
    print("   5. Fetch live odds from The Odds API")
    print("   6. Calculate expected value and detect edges")
    print("   7. Log profitable signals")
    print("   8. Save live odds to S3")
    print()
    print(f"⚙️  Configuration:")
    print(f"   - Mode: {'TEST (Fake Data)' if test_mode else 'LIVE (Real Data)'}")
    print(f"   - Min Edge Threshold: {MIN_EDGE_THRESHOLD:.1%}")
    print(f"   - MC Simulations: {N_SIMULATIONS:,}")
    print(f"   - Max Players Per Game: {MAX_PLAYERS_PER_GAME}")
    print()
    
    # =========================================================================
    # STEP 1: FETCH LIVE GAMES
    # =========================================================================
    print("="*80)
    print(f"STEP 1: Fetching live games {'(TEST MODE - FAKE DATA)' if test_mode else 'from ESPN'}...")
    print("="*80)
    
    live_games = fetch_live_games(test_mode=test_mode)
    
    if not live_games:
        print("❌ No live games found")
        return
    
    print(f"✅ Found {len(live_games)} live game(s)")
    for game in live_games:
        print(f"   🏀 {game['away_team']} ({game['away_score']}) @ {game['home_team']} ({game['home_score']})")
        print(f"      Q{game['quarter']} - {game['clock']}")
    print()
    
    # Collect all signals
    all_signals = []
    
    for game in live_games:
        game_id = game['game_id']
        
        # =====================================================================
        # STEP 2: GET ACTIVE PLAYERS
        # =====================================================================
        print("="*80)
        print(f"STEP 2: Getting active players {'(TEST MODE)' if test_mode else ''} for {game['away_team']} @ {game['home_team']}...")
        print("="*80)
        
        players = get_active_players(game_id, test_mode=test_mode)
        
        if not players:
            print("⚠️  No active players found")
            continue
        
        print(f"✅ Found {len(players)} active player(s)")
        for p in players:
            print(f"   - {p['player_name']} ({p['team']}): {p['current_points']} pts, {p['minutes_played']:.1f} min")
        print()
        
        # =====================================================================
        # STEP 3: FETCH LIVE ODDS
        # =====================================================================
        print("="*80)
        print(f"STEP 3: Fetching live odds {'(TEST MODE - FAKE DATA)' if test_mode else 'from The Odds API'}...")
        print("="*80)
        
        live_odds_df = fetch_live_odds(game_id, test_mode=test_mode)
        
        if live_odds_df is None or len(live_odds_df) == 0:
            print("⚠️  No live odds available for this game")
            continue
        
        print(f"✅ Fetched odds for {live_odds_df['player_name'].nunique()} player(s)")
        print()
        
        # =====================================================================
        # STEP 4: SAVE ODDS TO S3
        # =====================================================================
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
        
        # =====================================================================
        # STEP 5-7: ANALYZE EACH PLAYER
        # =====================================================================
        print("="*80)
        print(f"STEP 5-7: Analyzing players for betting opportunities...")
        print("="*80)
        
        for player in players:
            print(f"   🔄 Analyzing {player['player_name']}...")
            
            signal = analyze_player_betting_opportunity(
                player, game, live_odds_df, n_sims=N_SIMULATIONS, test_mode=test_mode
            )
            
            if signal:
                all_signals.append(signal)
                print(f"      ✅ PROFITABLE SIGNAL DETECTED")
            else:
                print(f"      ⚪ No profitable signal")
        
        print()
    
    # =========================================================================
    # STEP 8: DISPLAY SIGNALS
    # =========================================================================
    print("="*80)
    print("PROFITABLE BETTING SIGNALS")
    print("="*80)
    print()
    
    if not all_signals:
        print("❌ No profitable betting opportunities found")
        return
    
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
        print(f"Lines:        {signal['pregame_line']} (pregame) → {signal['live_line']} (live) | Bet: {odds_display}")
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
    
    print("="*80)
    print("✅ SCAN COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
