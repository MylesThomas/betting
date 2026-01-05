"""
NBA Rim Efficiency Screener

Purpose:
    Find daily betting value on NBA player points props by identifying:
    - Players with elite close-range efficiency (0-6 feet)
    - Opponents with weak rim protection
    - Props that don't reflect this matchup advantage

Created: 2025-01-04
Context: Built after collecting 12 seasons (2014-2025) of NBA shot chart data
         with distance information. This screener uses that historical data
         to find value in today's player props.

Usage:
    python analysis/nba_rim_efficiency_screener.py
    python analysis/nba_rim_efficiency_screener.py --date 2025-01-15
    python analysis/nba_rim_efficiency_screener.py --min-rim-attempts 5

Data Required:
    - Shot charts: data/01_input/nba_api/shot_charts/
    - Player props: fetched from The Odds API
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple
import argparse

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config_loader import get_config, get_project_root

# Load config
CONFIG = get_config()

# API setup
API_KEY = os.getenv('THE_ODDS_API_KEY')
if not API_KEY:
    print("⚠️  THE_ODDS_API_KEY environment variable not set")
    print("   Props fetching will not work without it")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Paths
SHOT_CHARTS_BASE = get_project_root() / 'data' / '01_input' / 'nba_api' / 'shot_charts'
CURRENT_SEASON = '2025-26'

# Thresholds
MIN_RIM_ATTEMPTS = 8  # Minimum close range attempts per game
MIN_GAMES = 10  # Minimum games to calculate season averages
CLOSE_RANGE_DISTANCE = 6  # 0-6 feet = close range


# =============================================================================
# EMOJI MAP
# =============================================================================

EMOJI = {
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'fire': '🔥',
    'target': '🎯',
    'chart': '📊',
    'calendar': '📅',
    'basketball': '🏀',
    'money': '💰',
    'up': '📈',
    'down': '📉',
    'shield': '🛡️',
    'lock': '🔒',
    'unlock': '🔓',
}


# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_todays_nba_games():
    """Fetch today's NBA games from The Odds API"""
    url = f"{ODDS_API_BASE_URL}/sports/basketball_nba/events"
    
    params = {
        'apiKey': API_KEY,
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        games = response.json()
        print(f"{EMOJI['success']} Found {len(games)} NBA games today")
        return games
    
    except requests.exceptions.RequestException as e:
        print(f"{EMOJI['error']} Error fetching games: {e}")
        return []


def get_player_props_for_game(event_id):
    """Fetch player points props for a specific game"""
    url = f"{ODDS_API_BASE_URL}/sports/basketball_nba/events/{event_id}/odds"
    
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'player_points',
        'oddsFormat': 'american'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"{EMOJI['warning']} Error fetching props for event {event_id}: {e}")
        return None


def fetch_todays_props():
    """
    Fetch all player points props for today's NBA games
    
    Returns:
        DataFrame with columns: player, team, opponent, prop_line, bookmaker
    """
    print(f"\n{EMOJI['basketball']} Fetching today's NBA games and props...")
    print("="*80)
    
    games = get_todays_nba_games()
    
    if not games:
        print(f"{EMOJI['warning']} No games found for today")
        return pd.DataFrame()
    
    all_props = []
    
    for game in games:
        event_id = game['id']
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = game['commence_time']
        
        print(f"\n{away_team} @ {home_team}")
        print(f"   Time: {commence_time}")
        
        props_data = get_player_props_for_game(event_id)
        
        if not props_data or 'bookmakers' not in props_data:
            print(f"   {EMOJI['warning']} No props available")
            continue
        
        # Parse props from each bookmaker
        for bookmaker in props_data['bookmakers']:
            bookmaker_name = bookmaker['key']
            
            for market in bookmaker.get('markets', []):
                if market['key'] != 'player_points':
                    continue
                
                for outcome in market.get('outcomes', []):
                    player_name = outcome['description']
                    prop_line = outcome['point']
                    odds = outcome.get('price', 0)
                    
                    # Determine if this player is home or away
                    # (We'll need to figure this out from team rosters or manually)
                    
                    all_props.append({
                        'player': player_name,
                        'prop_line': prop_line,
                        'odds': odds,
                        'bookmaker': bookmaker_name,
                        'game_id': event_id,
                        'home_team': home_team,
                        'away_team': away_team,
                        'commence_time': commence_time
                    })
        
        print(f"   {EMOJI['success']} Fetched props")
    
    if not all_props:
        print(f"\n{EMOJI['warning']} No props found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_props)
    print(f"\n{EMOJI['success']} Total props fetched: {len(df):,}")
    print(f"   Unique players: {df['player'].nunique()}")
    print(f"   Bookmakers: {df['bookmaker'].nunique()}")
    
    return df


# =============================================================================
# SHOT CHART ANALYSIS
# =============================================================================

def load_player_shot_data(player_name: str, season: str = CURRENT_SEASON) -> Optional[pd.DataFrame]:
    """
    Load shot chart data for a specific player
    
    Args:
        player_name: Player's full name (e.g., 'LeBron James')
        season: Season to load (e.g., '2025-26')
    
    Returns:
        DataFrame with all shots, or None if not found
    """
    season_dir = SHOT_CHARTS_BASE / season.replace('-', '_')
    
    if not season_dir.exists():
        return None
    
    # Try to find the player's file
    # Files are named: PlayerName_ID.csv
    player_files = list(season_dir.glob(f"{player_name.replace(' ', '_')}_*.csv"))
    
    if not player_files:
        return None
    
    # Load the first match
    df = pd.read_csv(player_files[0])
    return df


def calculate_rim_efficiency(shots_df: pd.DataFrame) -> Dict:
    """
    Calculate close-range shooting metrics
    
    Args:
        shots_df: DataFrame with shot chart data
    
    Returns:
        Dict with rim efficiency stats
    """
    if shots_df is None or len(shots_df) == 0:
        return {
            'total_shots': 0,
            'rim_attempts': 0,
            'rim_makes': 0,
            'rim_pct': 0.0,
            'rim_rate': 0.0,
            'games_played': 0,
            'rim_attempts_per_game': 0.0,
            'rim_points_per_game': 0.0
        }
    
    # Calculate stats
    total_shots = len(shots_df)
    games_played = shots_df['GAME_ID'].nunique()
    
    # Filter to close range (0-6 feet)
    close_shots = shots_df[shots_df['SHOT_DISTANCE'] <= CLOSE_RANGE_DISTANCE]
    
    rim_attempts = len(close_shots)
    rim_makes = close_shots['SHOT_MADE_FLAG'].sum()
    rim_pct = (rim_makes / rim_attempts * 100) if rim_attempts > 0 else 0.0
    rim_rate = (rim_attempts / total_shots * 100) if total_shots > 0 else 0.0
    
    rim_attempts_per_game = rim_attempts / games_played if games_played > 0 else 0.0
    rim_points_per_game = (rim_makes * 2) / games_played if games_played > 0 else 0.0
    
    return {
        'total_shots': total_shots,
        'rim_attempts': rim_attempts,
        'rim_makes': rim_makes,
        'rim_pct': rim_pct,
        'rim_rate': rim_rate,
        'games_played': games_played,
        'rim_attempts_per_game': rim_attempts_per_game,
        'rim_points_per_game': rim_points_per_game
    }


def calculate_team_rim_defense(opponent_team: str, season: str = CURRENT_SEASON) -> Dict:
    """
    Calculate how well a team defends the rim
    
    Args:
        opponent_team: Team abbreviation (e.g., 'LAL')
        season: Season to analyze
    
    Returns:
        Dict with rim defense stats
    """
    # This is a placeholder - we'd need to aggregate all shots AGAINST this team
    # For now, return neutral values
    # TODO: Build opponent shot chart aggregation
    
    return {
        'rim_fg_pct_allowed': 62.0,  # League average
        'rim_attempts_allowed_per_game': 25.0,  # League average
        'rim_defense_rating': 'Average'
    }


def analyze_player_rim_game(player_name: str, min_games: int = MIN_GAMES) -> Optional[Dict]:
    """
    Analyze a player's close-range efficiency for betting purposes
    
    Args:
        player_name: Player's full name
        min_games: Minimum games required for valid sample
    
    Returns:
        Dict with comprehensive rim efficiency analysis
    """
    # Load current season data
    shots_df = load_player_shot_data(player_name, CURRENT_SEASON)
    
    if shots_df is None:
        return None
    
    # Calculate current season stats
    stats = calculate_rim_efficiency(shots_df)
    
    if stats['games_played'] < min_games:
        return None
    
    # Get recent form (last 5 games)
    recent_games = shots_df['GAME_ID'].unique()[-5:]
    recent_shots = shots_df[shots_df['GAME_ID'].isin(recent_games)]
    recent_stats = calculate_rim_efficiency(recent_shots)
    
    return {
        'player': player_name,
        'games_played': stats['games_played'],
        'rim_attempts_per_game': stats['rim_attempts_per_game'],
        'rim_fg_pct': stats['rim_pct'],
        'rim_points_per_game': stats['rim_points_per_game'],
        'rim_rate': stats['rim_rate'],
        'recent_rim_fg_pct': recent_stats['rim_pct'],
        'recent_rim_attempts_pg': recent_stats['rim_attempts_per_game'],
    }


# =============================================================================
# MATCHUP ANALYSIS
# =============================================================================

def score_matchup(player_stats: Dict, opponent_defense: Dict) -> float:
    """
    Score the betting value of a player's rim efficiency vs opponent's defense
    
    Args:
        player_stats: Player's rim efficiency metrics
        opponent_defense: Opponent's rim defense metrics
    
    Returns:
        Score from 0-100 (higher = better bet)
    """
    score = 0.0
    
    # Factor 1: Player's rim efficiency (0-30 points)
    rim_pct = player_stats['rim_fg_pct']
    if rim_pct >= 70:
        score += 30
    elif rim_pct >= 65:
        score += 25
    elif rim_pct >= 60:
        score += 20
    elif rim_pct >= 55:
        score += 15
    else:
        score += 10
    
    # Factor 2: Volume of rim attempts (0-25 points)
    rim_attempts = player_stats['rim_attempts_per_game']
    if rim_attempts >= 10:
        score += 25
    elif rim_attempts >= 8:
        score += 20
    elif rim_attempts >= 6:
        score += 15
    elif rim_attempts >= 4:
        score += 10
    else:
        score += 5
    
    # Factor 3: Recent form (0-20 points)
    recent_pct = player_stats['recent_rim_fg_pct']
    season_pct = player_stats['rim_fg_pct']
    if recent_pct > season_pct + 5:
        score += 20  # Hot streak
    elif recent_pct > season_pct:
        score += 15
    elif recent_pct > season_pct - 5:
        score += 10
    else:
        score += 5  # Cold streak
    
    # Factor 4: Opponent rim defense (0-25 points)
    # TODO: Implement when we have opponent defense stats
    score += 15  # Neutral for now
    
    return score


def generate_daily_plays(props_df: pd.DataFrame, min_score: float = 60.0) -> pd.DataFrame:
    """
    Generate ranked list of best player points plays based on rim efficiency
    
    Args:
        props_df: DataFrame with today's player props
        min_score: Minimum matchup score to include
    
    Returns:
        DataFrame with ranked plays
    """
    print(f"\n{EMOJI['chart']} Analyzing rim efficiency for all players...")
    print("="*80)
    
    plays = []
    
    # Get unique players
    unique_players = props_df['player'].unique()
    
    for i, player_name in enumerate(unique_players, 1):
        if i % 10 == 0:
            print(f"   Analyzed {i}/{len(unique_players)} players...")
        
        # Get player's rim efficiency
        player_stats = analyze_player_rim_game(player_name)
        
        if player_stats is None:
            continue
        
        # Filter to this player's props
        player_props = props_df[props_df['player'] == player_name]
        
        if len(player_props) == 0:
            continue
        
        # Get best prop line (average across bookmakers)
        avg_prop_line = player_props['prop_line'].mean()
        best_odds_row = player_props.loc[player_props['odds'].idxmax()]
        
        # Get opponent (placeholder - need team assignment logic)
        opponent = "TBD"
        opponent_defense = calculate_team_rim_defense(opponent)
        
        # Score the matchup
        matchup_score = score_matchup(player_stats, opponent_defense)
        
        if matchup_score < min_score:
            continue
        
        plays.append({
            'player': player_name,
            'prop_line': avg_prop_line,
            'best_odds': best_odds_row['odds'],
            'bookmaker': best_odds_row['bookmaker'],
            'rim_attempts_pg': player_stats['rim_attempts_per_game'],
            'rim_fg_pct': player_stats['rim_fg_pct'],
            'rim_points_pg': player_stats['rim_points_per_game'],
            'recent_rim_pct': player_stats['recent_rim_fg_pct'],
            'matchup_score': matchup_score,
            'opponent': opponent,
            'game': f"{player_props.iloc[0]['away_team']} @ {player_props.iloc[0]['home_team']}"
        })
    
    if not plays:
        print(f"\n{EMOJI['warning']} No plays found meeting criteria")
        return pd.DataFrame()
    
    plays_df = pd.DataFrame(plays)
    plays_df = plays_df.sort_values('matchup_score', ascending=False)
    
    print(f"\n{EMOJI['success']} Found {len(plays_df)} potential plays")
    
    return plays_df


# =============================================================================
# OUTPUT & FORMATTING
# =============================================================================

def print_top_plays(plays_df: pd.DataFrame, top_n: int = 10):
    """Print formatted list of top plays"""
    
    if len(plays_df) == 0:
        print(f"\n{EMOJI['warning']} No plays to display")
        return
    
    print(f"\n{EMOJI['fire']} TOP {min(top_n, len(plays_df))} RIM EFFICIENCY PLAYS")
    print("="*80)
    
    for i, row in plays_df.head(top_n).iterrows():
        print(f"\n{EMOJI['target']} #{i+1}: {row['player']}")
        print(f"   Game: {row['game']}")
        print(f"   Prop Line: {row['prop_line']} points ({row['bookmaker']})")
        print(f"   Best Odds: {row['best_odds']:+d}")
        print(f"   {EMOJI['basketball']} Rim Stats:")
        print(f"      • {row['rim_attempts_pg']:.1f} attempts/game @ {row['rim_fg_pct']:.1f}%")
        print(f"      • {row['rim_points_pg']:.1f} points/game from rim")
        print(f"      • Recent form: {row['recent_rim_pct']:.1f}%")
        print(f"   {EMOJI['chart']} Matchup Score: {row['matchup_score']:.0f}/100")


def save_plays_to_csv(plays_df: pd.DataFrame, output_dir: Path = None):
    """Save plays to CSV file"""
    
    if output_dir is None:
        output_dir = get_project_root() / 'data' / '04_output'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now().strftime('%Y%m%d')
    output_file = output_dir / f"rim_efficiency_plays_{today}.csv"
    
    plays_df.to_csv(output_file, index=False)
    print(f"\n{EMOJI['success']} Saved plays to: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='NBA Rim Efficiency Screener')
    parser.add_argument('--min-score', type=float, default=60.0,
                       help='Minimum matchup score (0-100)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top plays to display')
    parser.add_argument('--save', action='store_true',
                       help='Save results to CSV')
    
    args = parser.parse_args()
    
    print(f"\n{EMOJI['basketball']} NBA RIM EFFICIENCY SCREENER")
    print("="*80)
    print(f"{EMOJI['calendar']} Date: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")
    print(f"{EMOJI['info']} Season: {CURRENT_SEASON}")
    print(f"{EMOJI['info']} Close Range: 0-{CLOSE_RANGE_DISTANCE} feet")
    
    # Step 1: Fetch today's props
    props_df = fetch_todays_props()
    
    if len(props_df) == 0:
        print(f"\n{EMOJI['error']} No props available. Exiting.")
        return
    
    # Step 2: Analyze matchups
    plays_df = generate_daily_plays(props_df, min_score=args.min_score)
    
    # Step 3: Display results
    print_top_plays(plays_df, top_n=args.top_n)
    
    # Step 4: Save if requested
    if args.save:
        save_plays_to_csv(plays_df)
    
    print(f"\n{EMOJI['success']} Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()

