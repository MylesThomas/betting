"""
Detect NBA Late-Game Scratch Scenarios and Cover Rates
=======================================================

Context:
--------
Analyzes scenarios where a team has 2+ players projected to score 20+ points
(based on median prop line across bookmakers), but one or more players scratch
late (DNP) while others still play. The hypothesis is that the remaining players
get more usage/shots and have higher cover rates.

Example (2026-02-02):
- LAC had Kawhi Leonard (~25pt projection) and James Harden (~25pt projection)
- Harden did not play (DNP)
- Kawhi took 21 shots (19 FGA) for 29 points and covered 25.5 line

Data Sources:
-------------
- Player Props: s3://the-odds-api-mt/nba/historical_player_props/{season}/{date}.csv
  - Contains prop lines from multiple bookmakers
  - Columns: player, market, prop_line, bookmaker, away_team, home_team, etc.

- Game Results: s3://nba-api-mt/player_game_logs/{season}/{date}.csv
  - Contains actual player performance from NBA API
  - Columns: PLAYER_NAME, TEAM_NAME, PTS, REB, AST, etc.

Workflow:
---------
1. Load prop data for season(s) and filter to player_points market
2. Calculate median projection across bookmakers for each player per game
3. Identify games where 2+ players on same team had 20+ projections
4. Load game results to determine who played vs DNP
5. Filter to scenarios with at least 1 DNP and 1 player still playing
6. Calculate cover rates and margins for players who played

Output:
-------
- Game-by-game logging of qualifying scenarios
- Summary statistics: cover rate %, avg margin, sample size
- Results saved locally and optionally to S3

Usage:
------
# Single season, all teams
python analysis/detect_nba_dnp_scenarios.py --seasons 2025-26 --points-threshold 20

# Single team filter
python analysis/detect_nba_dnp_scenarios.py --seasons 2025-26 --teams lac --points-threshold 20

# Multiple seasons
python analysis/detect_nba_dnp_scenarios.py --seasons 2025-26 2024-25 --points-threshold 20

Author: Thomas Myles
Date: 2026-02-04
"""

import argparse
import boto3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from io import StringIO
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import duckdb
import hashlib
import json
import subprocess

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from player_name_utils import normalize_player_name, get_name_mappings
from s3_utils import read_df_from_s3


# =============================================================================
# CONFIGURATION
# =============================================================================

S3_BUCKET_PROPS = 'the-odds-api-mt'
S3_BUCKET_GAMES = 'nba-api-mt'
S3_PLAYER_TEAM_HISTORY = 's3://nba-betting-mt/nba/player_team_history/history.parquet'

# Load player-team history for date-aware roster validation
PLAYER_TEAM_HISTORY_DF = None

try:
    print(f"📂 Loading player-team history from S3: {S3_PLAYER_TEAM_HISTORY}")
    PLAYER_TEAM_HISTORY_DF = pd.read_parquet(S3_PLAYER_TEAM_HISTORY)
    print(f"✅ Loaded {len(PLAYER_TEAM_HISTORY_DF):,} player-team records")
    print(f"   Players: {PLAYER_TEAM_HISTORY_DF['player_normalized'].nunique():,}")
    print(f"   Date range: {PLAYER_TEAM_HISTORY_DF['valid_from'].min()} to {PLAYER_TEAM_HISTORY_DF['valid_to'].max()}")
    print()
except Exception as e:
    print(f"⚠️  Warning: Could not load player_team_history.parquet from S3: {e}")
    print(f"   Player-team validation will be skipped!")
    print(f"   Run: python src/player_team_history/01_build.py")
    print()

# Team abbreviation mapping (odds API team names -> ESPN abbreviations)
TEAM_ABBR_MAP = {
    'Atlanta Hawks': 'atl', 'Boston Celtics': 'bos', 'Brooklyn Nets': 'bkn',
    'Charlotte Hornets': 'cha', 'Chicago Bulls': 'chi', 'Cleveland Cavaliers': 'cle',
    'Dallas Mavericks': 'dal', 'Denver Nuggets': 'den', 'Detroit Pistons': 'det',
    'Golden State Warriors': 'gs', 'Houston Rockets': 'hou', 'Indiana Pacers': 'ind',
    'LA Clippers': 'lac', 'Los Angeles Clippers': 'lac', 'Los Angeles Lakers': 'lal',
    'Memphis Grizzlies': 'mem', 'Miami Heat': 'mia', 'Milwaukee Bucks': 'mil',
    'Minnesota Timberwolves': 'min', 'New Orleans Pelicans': 'no', 'New York Knicks': 'ny',
    'Oklahoma City Thunder': 'okc', 'Orlando Magic': 'orl', 'Philadelphia 76ers': 'phi',
    'Phoenix Suns': 'phx', 'Portland Trail Blazers': 'por', 'Sacramento Kings': 'sac',
    'San Antonio Spurs': 'sa', 'Toronto Raptors': 'tor', 'Utah Jazz': 'utah',
    'Washington Wizards': 'wsh'
}

# Reverse mapping (ESPN abbreviations -> list of possible odds API team names)
ABBR_TO_TEAM_MAP = defaultdict(list)
for team, abbr in TEAM_ABBR_MAP.items():
    ABBR_TO_TEAM_MAP[abbr].append(team)


# =============================================================================
# CACHE FUNCTIONS
# =============================================================================

CACHE_DIR = Path.home() / 'Downloads' / 'tmp' / 'dnp_scenarios_cache'

def get_cache_key(seasons, teams, points_threshold):
    """
    Generate cache key based on analysis parameters.
    
    Args:
        seasons: List of season strings
        teams: List of team abbreviations (or None)
        points_threshold: Points threshold
    
    Returns:
        String cache key
    """
    seasons_str = '_'.join(sorted(seasons))
    teams_str = '_'.join(sorted(teams)) if teams else 'all'
    return f"{seasons_str}_teams_{teams_str}_threshold_{points_threshold}"


def save_scenarios_to_cache(scenarios, cache_key):
    """
    Save scenarios to parquet cache.
    
    Args:
        scenarios: List of scenario dictionaries
        cache_key: Cache key string
    """
    if not scenarios:
        return
    
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Flatten scenarios to DataFrame
    rows = []
    for scenario in scenarios:
        # Add DNP players
        for player in scenario['players_dnp']:
            rows.append({
                'date': scenario['date'],
                'season': scenario['season'],
                'team': scenario['team'],
                'team_abbr': scenario['team_abbr'],
                'opponent': scenario['opponent'],
                'home_away': scenario['home_away'],
                'players_projected_20_plus': scenario['players_projected_20_plus'],
                'player': player['player'],
                'projection': player['projection'],
                'actual_points': None,
                'fga': None,
                'minutes': None,
                'played': False
            })
        
        # Add players who played
        for player in scenario['players_played']:
            rows.append({
                'date': scenario['date'],
                'season': scenario['season'],
                'team': scenario['team'],
                'team_abbr': scenario['team_abbr'],
                'opponent': scenario['opponent'],
                'home_away': scenario['home_away'],
                'players_projected_20_plus': scenario['players_projected_20_plus'],
                'player': player['player'],
                'projection': player['projection'],
                'actual_points': player['points'],
                'fga': player['fga'],
                'minutes': player['minutes'],
                'wl': player.get('wl'),  # Add W/L
                'dnp_teammates': player.get('dnp_teammates', ''),  # Add DNP teammates
                'played': True
            })
    
    df = pd.DataFrame(rows)
    cache_file = CACHE_DIR / f"{cache_key}.parquet"
    df.to_parquet(cache_file, index=False)
    print(f"\n💾 Cache saved to: {cache_file}")


def load_scenarios_from_cache(cache_key):
    """
    Load scenarios from parquet cache using DuckDB.
    
    Args:
        cache_key: Cache key string
    
    Returns:
        List of scenario dictionaries, or None if cache doesn't exist
    """
    cache_file = CACHE_DIR / f"{cache_key}.parquet"
    
    if not cache_file.exists():
        return None
    
    print(f"\n📂 Loading from cache: {cache_file}")
    
    # Read with DuckDB
    con = duckdb.connect(':memory:')
    df = con.execute(f"SELECT * FROM '{cache_file}'").df()
    con.close()
    
    # Reconstruct scenarios
    scenarios = []
    grouped = df.groupby(['date', 'team'])
    
    for (date, team), group in grouped:
        dnp_players = []
        played_players = []
        
        for _, row in group.iterrows():
            if row['played']:
                played_players.append({
                    'player': row['player'],
                    'projection': row['projection'],
                    'points': row['actual_points'],
                    'fga': row['fga'],
                    'minutes': row['minutes'],
                    'wl': row.get('wl') if 'wl' in row else None,  # Add W/L
                    'dnp_teammates': row.get('dnp_teammates', '') if 'dnp_teammates' in row else ''  # Add DNP teammates
                })
            else:
                dnp_players.append({
                    'player': row['player'],
                    'projection': row['projection']
                })
        
        scenario = {
            'date': date,
            'season': group.iloc[0]['season'],
            'team': team,
            'team_abbr': group.iloc[0]['team_abbr'],
            'opponent': group.iloc[0]['opponent'],
            'home_away': group.iloc[0]['home_away'],
            'players_projected_20_plus': group.iloc[0]['players_projected_20_plus'],
            'players_dnp': dnp_players,
            'players_played': played_players
        }
        scenarios.append(scenario)
    
    print(f"✅ Loaded {len(scenarios)} scenarios from cache")
    return scenarios


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_team_name(team_name):
    """
    Normalize team names for matching between different data sources.
    
    Examples:
        'LA Clippers' -> 'Clippers'
        'Los Angeles Clippers' -> 'Clippers'
        'Los Angeles Lakers' -> 'Lakers'
        'New York Knicks' -> 'Knicks'
    
    Args:
        team_name: Team name string
    
    Returns:
        Normalized team name (just the mascot)
    """
    if pd.isna(team_name):
        return team_name
    
    # Remove city names and location prefixes
    name = team_name.replace('Los Angeles', '').replace('LA ', '')
    name = name.replace('New York', '').replace('New Orleans', '')
    name = name.replace('Golden State', '').replace('Oklahoma City', '')
    name = name.replace('San Antonio', '').replace('Portland Trail', '')
    name = name.strip()
    
    return name


def teams_match(team1, team2):
    """
    Check if two team names refer to the same team.
    
    Args:
        team1: First team name
        team2: Second team name
    
    Returns:
        True if teams match, False otherwise
    """
    return normalize_team_name(team1) == normalize_team_name(team2)


def player_on_team_at_date(player_normalized, team_abbr, game_date):
    """
    Check if a player was on a specific team on a specific date using player_team_history.
    
    Args:
        player_normalized: Normalized player name
        team_abbr: Team abbreviation (e.g., 'tor', 'bkn', 'lac')
        game_date: Date string in YYYY-MM-DD format or datetime object
    
    Returns:
        True if player was on team at that date, False otherwise
    """
    if PLAYER_TEAM_HISTORY_DF is None:
        # No history data - skip validation (allow all)
        return True
    
    # Convert game_date to datetime if string
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date)
    
    # Filter to this player
    player_history = PLAYER_TEAM_HISTORY_DF[
        PLAYER_TEAM_HISTORY_DF['player_normalized'].str.lower() == player_normalized.lower()
    ]
    
    if player_history.empty:
        # Player not found in history - skip (allow them through, might be recent call-up)
        return True
    
    # Check if player was on this team on this date
    # Columns in history.parquet: player_normalized, team, valid_from, valid_to
    team_abbr_lower = team_abbr.lower()
    
    for _, row in player_history.iterrows():
        row_team = row['team'].lower() if pd.notna(row['team']) else ''
        if row_team != team_abbr_lower:
            continue
        
        # Check date range using valid_from and valid_to
        start_date = pd.to_datetime(row['valid_from']) if pd.notna(row['valid_from']) else pd.Timestamp.min
        # valid_to can be NULL for current team
        end_date = pd.to_datetime(row['valid_to']) if pd.notna(row['valid_to']) else pd.Timestamp.max
        
        if start_date <= game_date <= end_date:
            return True
    
    return False


def get_team_roster_on_date(team_abbr, game_date):
    """
    Get all players on a team's roster on a specific date.
    
    Args:
        team_abbr: Team abbreviation (e.g., 'tor', 'bkn', 'lac')
        game_date: Date string in YYYY-MM-DD format or datetime object
    
    Returns:
        Set of normalized player names on the roster, or None if no history data
    """
    if PLAYER_TEAM_HISTORY_DF is None:
        return None
    
    # Convert game_date to datetime if string
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date)
    
    team_abbr_lower = team_abbr.lower()
    
    # Filter to this team
    team_history = PLAYER_TEAM_HISTORY_DF[
        PLAYER_TEAM_HISTORY_DF['team'].str.lower() == team_abbr_lower
    ]
    
    if team_history.empty:
        return set()
    
    # Find all players who were on this team on this date
    roster = set()
    for _, row in team_history.iterrows():
        start_date = pd.to_datetime(row['valid_from']) if pd.notna(row['valid_from']) else pd.Timestamp.min
        end_date = pd.to_datetime(row['valid_to']) if pd.notna(row['valid_to']) else pd.Timestamp.max
        
        if start_date <= game_date <= end_date:
            roster.add(row['player_normalized'])
    
    return roster


def load_props_from_s3(season, date_str):
    """
    Load player props from S3 for a specific date.
    
    Args:
        season: NBA season (e.g., '2025-26')
        date_str: Date string in YYYY-MM-DD format
    
    Returns:
        DataFrame with prop data, or empty DataFrame if not found
    """
    s3_key = f"nba/historical_player_props/{season}/{date_str}.csv"
    
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=S3_BUCKET_PROPS, Key=s3_key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        return df
    except Exception as e:
        print(f"  ⚠️  Could not load props for {date_str}: {e}")
        return pd.DataFrame()


def load_games_from_s3(season, date_str):
    """
    Load game results from S3 for a specific date.
    
    Args:
        season: NBA season (e.g., '2025-26')
        date_str: Date string in YYYY-MM-DD format
    
    Returns:
        DataFrame with game results, or empty DataFrame if not found
    """
    s3_key = f"player_game_logs/{season}/{date_str}.csv"
    
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=S3_BUCKET_GAMES, Key=s3_key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        return df
    except Exception as e:
        print(f"  ⚠️  Could not load games for {date_str}: {e}")
        return pd.DataFrame()


def get_season_game_dates(season):
    """
    Get all game dates for a season from S3.
    
    Args:
        season: NBA season (e.g., '2025-26')
    
    Returns:
        List of date strings in YYYY-MM-DD format, sorted chronologically
    """
    s3 = boto3.client('s3')
    prefix = f"nba/historical_player_props/{season}/"
    
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_PROPS, Prefix=prefix)
        
        if 'Contents' not in response:
            return []
        
        # Extract dates from file names (e.g., '2025-01-15.csv' -> '2025-01-15')
        dates = []
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.csv'):
                filename = key.split('/')[-1]
                date_str = filename.replace('.csv', '')
                dates.append(date_str)
        
        return sorted(dates)
    
    except Exception as e:
        print(f"Error listing S3 objects: {e}")
        return []


def calculate_median_projection(props_df, player_name, team_name):
    """
    Calculate median point projection for a player across all bookmakers.
    
    Args:
        props_df: DataFrame with prop data (already filtered to player_points market)
        player_name: Normalized player name
        team_name: Team name (odds API format)
    
    Returns:
        Median prop line, or None if no data
    """
    # Filter to this player and team
    # Player could be on home or away team
    player_props = props_df[
        (props_df['player_normalized'] == player_name) &
        ((props_df['home_team'] == team_name) | (props_df['away_team'] == team_name))
    ]
    
    if player_props.empty:
        return None
    
    # Calculate median across all bookmakers
    median_line = player_props['prop_line'].median()
    return median_line


def identify_team_game(props_df, team_name):
    """
    Identify the opponent for a team's game on this date.
    
    Args:
        props_df: DataFrame with prop data
        team_name: Team name (from game results - e.g., "LA Clippers")
    
    Returns:
        dict with 'opponent' and 'home_away', or None if not found
    """
    # Find a game involving this team (use fuzzy matching for team names)
    game = props_df[
        props_df['home_team'].apply(lambda x: teams_match(x, team_name)) |
        props_df['away_team'].apply(lambda x: teams_match(x, team_name))
    ]
    
    if game.empty:
        return None
    
    # Get first row (all rows for same team will have same opponent)
    row = game.iloc[0]
    
    if teams_match(row['home_team'], team_name):
        return {
            'opponent': row['away_team'],
            'home_away': 'home'
        }
    else:
        return {
            'opponent': row['home_team'],
            'home_away': 'away'
        }


def check_player_played(games_df, player_name, team_name):
    """
    Check if a player played in the game and return their stats.
    
    Args:
        games_df: DataFrame with game results
        player_name: Normalized player name
        team_name: Team name (odds API format)
    
    Returns:
        dict with player stats if played, None if DNP
    """
    # Convert team name to match what's in games_df (TEAM_NAME column)
    # games_df uses full names like "LA Clippers"
    player_game = games_df[
        (games_df['player_normalized'] == player_name) &
        (games_df['TEAM_NAME'] == team_name)
    ]
    
    if player_game.empty:
        return None
    
    # Player played - return stats
    row = player_game.iloc[0]
    return {
        'points': row['PTS'],
        'fga': row['FGA'] if 'FGA' in row else None,
        'minutes': row['MIN'] if 'MIN' in row else None
    }


def analyze_date(date_str, season, points_threshold, team_filter=None):
    """
    Analyze a single game date for DNP scenarios.
    
    Clean strategy:
    1. Load all props for this date
    2. Load all game results for this date
    3. For each team that played:
       a. Get roster for that team on that date from player_team_history
       b. Filter props to ONLY players on that team's roster
       c. Calculate projections for those players
       d. Check for DNP scenarios
    
    Args:
        date_str: Date in YYYY-MM-DD format
        season: NBA season (e.g., '2025-26')
        points_threshold: Minimum projection to be considered (e.g., 20)
        team_filter: Optional list of team abbreviations to filter to
    
    Returns:
        List of scenario dictionaries
    """
    # Load all props for this date
    props_df = load_props_from_s3(season, date_str)
    
    if props_df.empty:
        return []
    
    # Filter to player_points market only
    props_df = props_df[props_df['market'] == 'player_points'].copy()
    
    if props_df.empty:
        return []
    
    # Normalize player names in props
    props_df['player_normalized'] = props_df['player'].apply(normalize_player_name)
    name_mappings = get_name_mappings()
    props_df['player_normalized'] = props_df['player_normalized'].replace(name_mappings)
    
    # Load all game results for this date
    games_df = load_games_from_s3(season, date_str)
    
    if games_df.empty:
        return []
    
    # Normalize player names in games
    games_df['player_normalized'] = games_df['PLAYER_NAME'].apply(normalize_player_name)
    games_df['player_normalized'] = games_df['player_normalized'].replace(name_mappings)
    
    # Get unique teams that played
    unique_teams = games_df['TEAM_NAME'].unique()
    
    # Apply team filter if specified
    if team_filter:
        team_filter_full = []
        for abbr in team_filter:
            possible_names = ABBR_TO_TEAM_MAP.get(abbr.lower(), [])
            team_filter_full.extend(possible_names)
        
        unique_teams = [t for t in unique_teams if any(filter_name in t for filter_name in team_filter_full)]
        
        if not unique_teams:
            return []
    
    scenarios = []
    
    # For each team that played
    for team_name in unique_teams:
        # Get team abbreviation
        team_abbr = None
        for full_name, abbr in TEAM_ABBR_MAP.items():
            if full_name in team_name or team_name in full_name:
                team_abbr = abbr.lower()
                break
        
        if not team_abbr:
            continue
        
        # Get players who actually played for this team
        team_games = games_df[games_df['TEAM_NAME'] == team_name]
        players_played_names = set(team_games['player_normalized'])
        
        # Get roster: all players on this team on this date
        team_roster = get_team_roster_on_date(team_abbr, date_str)
        
        if not team_roster:
            # No roster data - skip this team
            continue
        
        # Filter props to ONLY players on this team's roster
        team_player_props = props_df[props_df['player_normalized'].isin(team_roster)]
        
        if team_player_props.empty:
            continue
        
        # Calculate median projection for each rostered player with props
        players_with_props = {}
        for player in team_roster:
            player_lines = team_player_props[team_player_props['player_normalized'] == player]['prop_line']
            if not player_lines.empty:
                median_proj = player_lines.median()
                if median_proj >= points_threshold:
                    players_with_props[player] = median_proj
        
        # Need at least 2 players with 20+ projections
        if len(players_with_props) < 2:
            continue
        
        # Separate into played vs DNP
        players_played = []
        players_dnp = []
        
        for player, projection in players_with_props.items():
            if player in players_played_names:
                # Player played - get their stats
                player_game = team_games[team_games['player_normalized'] == player].iloc[0]
                actual_points = player_game['PTS']
                prop_wl = 'W' if actual_points > projection else 'L'
                players_played.append({
                    'player': player,
                    'projection': projection,
                    'points': actual_points,
                    'fga': player_game['FGA'] if 'FGA' in player_game else None,
                    'minutes': player_game['MIN'] if 'MIN' in player_game else None,
                    'wl': prop_wl
                })
            else:
                # Player has props but didn't play (DNP)
                players_dnp.append({
                    'player': player,
                    'projection': projection
                })
        
        # Scenario qualifies if at least 1 DNP and at least 1 played
        if len(players_dnp) >= 1 and len(players_played) >= 1:
            # Get game info (opponent, home/away)
            game_info = identify_team_game(props_df, team_name)
            
            # Create list of DNP player names
            dnp_player_names = [p['player'] for p in players_dnp]
            
            # Add DNP teammates to each player who played
            for player in players_played:
                player['dnp_teammates'] = ', '.join(dnp_player_names)
            
            scenario = {
                'date': date_str,
                'season': season,
                'team': team_name,
                'team_abbr': team_abbr.upper() if team_abbr else '???',
                'opponent': game_info['opponent'] if game_info else '???',
                'home_away': game_info['home_away'] if game_info else '???',
                'players_projected_20_plus': len(players_with_props),
                'players_dnp': players_dnp,
                'players_played': players_played,
                'team_roster': team_roster  # Store for verification in print_scenario
            }
            
            # VERIFY: All players in this scenario should be on the same team on this date (silent check, assert only)
            all_scenario_players = [p['player'] for p in players_dnp] + [p['player'] for p in players_played]
            verification_failed = any(player not in team_roster for player in all_scenario_players)
            
            # Assert that all players are on the same team
            assert not verification_failed, (
                f"VERIFICATION FAILED: Not all players in scenario are on {team_abbr.upper()} on {date_str}. "
                f"Players: {all_scenario_players}"
            )
            
            scenarios.append(scenario)
    
    return scenarios


def print_scenario(scenario):
    """
    Print a single scenario in a readable format.
    
    Args:
        scenario: Scenario dictionary
    """
    print()
    print("="*80)
    print(f"📅 {scenario['date']} | {scenario['team_abbr'].upper()} vs {TEAM_ABBR_MAP.get(scenario['opponent'], scenario['opponent']).upper()} ({scenario['home_away']})")
    print("="*80)
    
    print(f"\n🎯 Players Projected 20+: {scenario['players_projected_20_plus']}")
    
    print(f"\n❌ DNP ({len(scenario['players_dnp'])}):")
    for player in scenario['players_dnp']:
        print(f"   • {player['player']}: {player['projection']:.1f}pt projection")
    
    print(f"\n✅ PLAYED ({len(scenario['players_played'])}):")
    for player in scenario['players_played']:
        line = player['projection']
        actual = player['points']
        margin = actual - line
        covered = '✅' if actual >= line else '❌'
        
        fga_str = f" ({player['fga']} FGA)" if player['fga'] is not None else ""
        
        print(f"   {covered} {player['player']}: {actual}pts vs {line:.1f} line (margin: {margin:+.1f}){fga_str}")
    
    # Print verification at the end
    if 'team_roster' in scenario:
        all_scenario_players = [p['player'] for p in scenario['players_dnp']] + [p['player'] for p in scenario['players_played']]
        print(f"\n🔍 VERIFY PLAYERS ARE ON {scenario['team_abbr']} ON {scenario['date']}:")
        for player in all_scenario_players:
            if player in scenario['team_roster']:
                print(f"   ✅ {player} -> on {scenario['team_abbr']}")
            else:
                print(f"   ❌ {player} -> NOT on {scenario['team_abbr']}")


def calculate_summary_stats(scenarios):
    """
    Calculate summary statistics across all scenarios.
    
    Args:
        scenarios: List of scenario dictionaries
    
    Returns:
        dict with summary statistics and player-level DataFrame
    """
    if not scenarios:
        return None, None
    
    # Collect all players who played
    all_players_played = []
    for scenario in scenarios:
        for player in scenario['players_played']:
            all_players_played.append({
                'player': player['player'],
                'team': scenario['team'],
                'date': scenario['date'],
                'projection': player['projection'],
                'actual': player['points'],
                'margin': player['points'] - player['projection'],
                'covered': player['points'] >= player['projection'],
                'wl': player.get('wl', ''),
                'dnp_teammates': player.get('dnp_teammates', '')
            })
    
    if not all_players_played:
        return None, None
    
    df = pd.DataFrame(all_players_played)
    
    # Calculate player-level stats
    player_stats = df.groupby('player').agg({
        'actual': ['mean', 'count'],
        'projection': 'mean',
        'margin': 'mean',
        'covered': 'mean'
    }).reset_index()
    
    # Flatten column names
    player_stats.columns = ['player', 'avg_actual', 'games', 'avg_projection', 'avg_margin', 'cover_rate']
    
    # Calculate PPG over expectation
    player_stats['ppg_over_exp'] = player_stats['avg_actual'] - player_stats['avg_projection']
    
    # Calculate W-L record for each player
    wl_records = []
    for player_name in player_stats['player']:
        player_games = df[df['player'] == player_name]
        wins = (player_games['wl'] == 'W').sum()
        losses = (player_games['wl'] == 'L').sum()
        wl_records.append(f"{wins}-{losses}")
    
    player_stats['wl_record'] = wl_records
    
    # Find most common DNP teammate for each player (with count)
    dnp_teammates_list = []
    for player_name in player_stats['player']:
        player_games = df[df['player'] == player_name]
        # Get all DNP teammates across all games
        all_dnp = []
        for dnp_str in player_games['dnp_teammates']:
            if dnp_str and isinstance(dnp_str, str):
                # Split by comma and strip whitespace
                teammates = [t.strip() for t in dnp_str.split(',') if t.strip()]
                all_dnp.extend(teammates)
        
        if all_dnp:
            # Count occurrences and get most common
            from collections import Counter
            counter = Counter(all_dnp)
            most_common_teammate, count = counter.most_common(1)[0]
            dnp_teammates_list.append(f"{most_common_teammate} ({count})")
        else:
            dnp_teammates_list.append('')
    
    player_stats['most_common_dnp'] = dnp_teammates_list
    
    # Sort by PPG over expectation (primary), then cover rate (secondary)
    player_stats = player_stats.sort_values(['ppg_over_exp', 'cover_rate'], ascending=[False, False])
    
    summary = {
        'total_scenarios': len(scenarios),
        'total_player_games': len(all_players_played),
        'cover_rate': df['covered'].mean(),
        'avg_margin': df['margin'].mean(),
        'median_margin': df['margin'].median(),
        'avg_projection': df['projection'].mean(),
        'avg_actual': df['actual'].mean()
    }
    
    return summary, player_stats


def print_summary(summary, player_stats, show_top_players=False, min_games_display=3):
    """
    Print summary statistics.
    
    Args:
        summary: Summary statistics dictionary
        player_stats: DataFrame with player-level statistics
        show_top_players: Whether to show top/bottom players
        min_games_display: Minimum games to display in top/bottom
    """
    if summary is None:
        print("\n⚠️  No qualifying scenarios found")
        return
    
    print()
    print("="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    print()
    print(f"Total Scenarios: {summary['total_scenarios']}")
    print(f"Total Player Games (who played in DNP scenarios): {summary['total_player_games']}")
    print()
    print(f"Cover Rate: {summary['cover_rate']:.1%}")
    print(f"Average Margin: {summary['avg_margin']:+.2f} points")
    print(f"Median Margin: {summary['median_margin']:+.2f} points")
    print()
    print(f"Average Projection: {summary['avg_projection']:.1f} points")
    print(f"Average Actual: {summary['avg_actual']:.1f} points")
    print()
    print("="*80)
    
    if show_top_players and player_stats is not None and len(player_stats) > 0:
        # Filter to players with minimum games
        qualified_players = player_stats[player_stats['games'] >= min_games_display]
        
        if len(qualified_players) >= 10:
            print()
            print("="*80)
            print(f"🏆 TOP 10 PERFORMERS (min {min_games_display} games)")
            print("="*80)
            print()
            
            for i, row in qualified_players.head(10).iterrows():
                print(f"{row['player']:30} | {row['wl_record']:6} | {row['games']:2.0f} games | "
                      f"{row['ppg_over_exp']:+5.1f} vs exp | "
                      f"{row['cover_rate']:5.1%} cover | "
                      f"{row['avg_actual']:.1f} ppg")
            print(f"{'':30} | {'':6} | {'':8} | {'':11} | {'':12} | {'DNP: ' + qualified_players.head(10).iloc[0]['most_common_dnp']}")
            
            print()
            print("="*80)
            print(f"💔 BOTTOM 10 PERFORMERS (min {min_games_display} games)")
            print("="*80)
            print()
            
            for i, row in qualified_players.tail(10).iterrows():
                print(f"{row['player']:30} | {row['games']:2.0f} games | "
                      f"{row['ppg_over_exp']:+5.1f} vs exp | "
                      f"{row['cover_rate']:5.1%} cover | "
                      f"{row['avg_actual']:.1f} ppg")
            
            print()
            print("="*80)


# =============================================================================
# VISUALIZATION  
# =============================================================================

def create_player_performance_viz_matplotlib(player_stats, seasons_str, output_path=None):
    """
    Create side-by-side visualization of top and bottom 15 players.
    
    Args:
        player_stats: DataFrame with player statistics
        seasons_str: String describing seasons (for title)
        output_path: Optional path to save the plot
    """
    # Filter to players with at least 3 games
    min_games = 3
    qualified = player_stats[player_stats['games'] >= min_games].copy()
    
    if len(qualified) < 30:
        print(f"\n⚠️  Only {len(qualified)} players with {min_games}+ games. Need at least 30 for viz.")
        return
    
    # Get top and bottom 15
    top_15 = qualified.head(15)
    bottom_15 = qualified.tail(15)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle(f'NBA Late-Game Scratch Analysis: Player Performance\n{seasons_str}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Color mapping based on performance
    def get_color(margin, cover_rate):
        if margin > 2 and cover_rate > 0.6:
            return '#2E7D32'  # Dark green
        elif margin > 0 and cover_rate > 0.5:
            return '#66BB6A'  # Light green
        elif margin < -2 and cover_rate < 0.4:
            return '#C62828'  # Dark red
        elif margin < 0 and cover_rate < 0.5:
            return '#EF5350'  # Light red
        else:
            return '#FFA726'  # Orange (neutral)
    
    # TOP 15 PERFORMERS
    top_colors = [get_color(row['ppg_over_exp'], row['cover_rate']) 
                  for _, row in top_15.iterrows()]
    
    y_pos = np.arange(len(top_15))
    bars1 = ax1.barh(y_pos, top_15['ppg_over_exp'], color=top_colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{row['player'][:20]} ({int(row['games'])}g, {row['cover_rate']:.0%})" 
                          for _, row in top_15.iterrows()], fontsize=9)
    ax1.set_xlabel('PPG Over Expectation', fontsize=11, fontweight='bold')
    ax1.set_title('🏆 TOP 15 PERFORMERS', fontsize=13, fontweight='bold', pad=15)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (_, row) in enumerate(top_15.iterrows()):
        ax1.text(row['ppg_over_exp'] + 0.2, i, f"{row['ppg_over_exp']:+.1f}", 
                va='center', fontsize=8, fontweight='bold')
    
    # BOTTOM 15 PERFORMERS
    bottom_colors = [get_color(row['ppg_over_exp'], row['cover_rate']) 
                     for _, row in bottom_15.iterrows()]
    
    y_pos = np.arange(len(bottom_15))
    bars2 = ax2.barh(y_pos, bottom_15['ppg_over_exp'], color=bottom_colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{row['player'][:20]} ({int(row['games'])}g, {row['cover_rate']:.0%})" 
                          for _, row in bottom_15.iterrows()], fontsize=9)
    ax2.set_xlabel('PPG Over Expectation', fontsize=11, fontweight='bold')
    ax2.set_title('💔 BOTTOM 15 PERFORMERS', fontsize=13, fontweight='bold', pad=15)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (_, row) in enumerate(bottom_15.iterrows()):
        ax2.text(row['ppg_over_exp'] - 0.2, i, f"{row['ppg_over_exp']:+.1f}", 
                ha='right', va='center', fontsize=8, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#2E7D32', label='Great (+2 PPG, 60%+ cover)'),
        mpatches.Patch(color='#66BB6A', label='Good (positive, 50%+ cover)'),
        mpatches.Patch(color='#FFA726', label='Neutral'),
        mpatches.Patch(color='#EF5350', label='Poor (negative, <50% cover)'),
        mpatches.Patch(color='#C62828', label='Bad (-2 PPG, <40% cover)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
              bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=True)
    
    # Add footer text
    footer_text = (f"Analysis: {len(qualified)} players with {min_games}+ games in late-scratch scenarios\n"
                  f"Format: Player Name (games, cover rate) | Bar = PPG vs Expectation")
    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=9, 
            style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {output_path}")
    else:
        plt.savefig('dnp_player_performance.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: dnp_player_performance.png")
    
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Detect NBA late-game scratch scenarios and analyze cover rates'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2025-26'],
        help='NBA seasons to analyze (e.g., 2025-26 2024-25)'
    )
    parser.add_argument(
        '--teams',
        nargs='+',
        default=None,
        help='Optional team abbreviations to filter to (e.g., lac bos)'
    )
    parser.add_argument(
        '--points-threshold',
        type=float,
        default=20.0,
        help='Minimum median projection to be considered (default: 20)'
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate visualization of top/bottom player performance'
    )
    parser.add_argument(
        '--show-top-players',
        action='store_true',
        help='Show top/bottom 10 players in console output'
    )
    parser.add_argument(
        '--use-cache',
        action='store_true',
        help='Use cached results if available (faster for repeat runs)'
    )
    parser.add_argument(
        '--min-games',
        type=int,
        default=10,
        help='Minimum games for player rankings and visualization (default: 10)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🏀 NBA LATE-GAME SCRATCH ANALYSIS")
    print("="*80)
    print(f"Seasons: {', '.join(args.seasons)}")
    print(f"Teams: {', '.join(args.teams) if args.teams else 'All'}")
    print(f"Points Threshold: {args.points_threshold}")
    print(f"Use Cache: {args.use_cache}")
    print()
    
    # Try to load from cache if requested
    all_scenarios = []
    if args.use_cache:
        cache_key = get_cache_key(args.seasons, args.teams, args.points_threshold)
        all_scenarios = load_scenarios_from_cache(cache_key)
        
        if all_scenarios is not None:
            # Cache hit - skip to summary
            summary, player_stats = calculate_summary_stats(all_scenarios)
            print_summary(summary, player_stats, show_top_players=args.show_top_players, 
                         min_games_display=args.min_games)
            
            if args.plots and player_stats is not None:
                print("\n🎨 Generating visualization using R gt package...")
                
                # Build command to run viz script
                viz_script = Path(__file__).parent / 'viz_nba_dnp_scenarios_gt.py'
                cmd = ['python3', str(viz_script)]
                
                # Pass through seasons
                cmd.extend(['--seasons'] + args.seasons)
                
                # Pass through teams if specified
                if args.teams:
                    cmd.extend(['--teams'] + args.teams)
                
                # Pass through points threshold
                cmd.extend(['--points-threshold', str(args.points_threshold)])
                
                # Pass through min-games
                cmd.extend(['--min-games', str(args.min_games)])
                
                # Run visualization
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print("⚠️  Visualization failed:")
                    print(result.stderr)
                else:
                    print("✅ Visualization complete!")
            
            return
        else:
            print("⚠️  No cache found, running full analysis...")
            all_scenarios = []
    
    for season in args.seasons:
        print(f"\n📅 Processing {season}...")
        print("-"*80)
        
        # Get all game dates for this season
        game_dates = get_season_game_dates(season)
        
        if not game_dates:
            print(f"⚠️  No game dates found for {season}")
            continue
        
        print(f"Found {len(game_dates)} game dates")
        print()
        
        # Process each date
        for i, date_str in enumerate(game_dates, 1):
            print(f"[{i}/{len(game_dates)}] {date_str}...", end='')
            
            scenarios = analyze_date(
                date_str=date_str,
                season=season,
                points_threshold=args.points_threshold,
                team_filter=args.teams
            )
            
            if scenarios:
                print(f" ✅ {len(scenarios)} scenario(s) found")
                for scenario in scenarios:
                    print_scenario(scenario)
                    all_scenarios.append(scenario)
            else:
                print(" (no qualifying scenarios)")
    
    # Save to cache if requested
    if args.use_cache and all_scenarios:
        cache_key = get_cache_key(args.seasons, args.teams, args.points_threshold)
        save_scenarios_to_cache(all_scenarios, cache_key)
    
    # Print summary
    summary, player_stats = calculate_summary_stats(all_scenarios)
    print_summary(summary, player_stats, show_top_players=args.show_top_players,
                 min_games_display=args.min_games)
    
    # Generate visualization if requested
    if args.plots and player_stats is not None:
        print("\n🎨 Generating visualization using R gt package...")
        
        # Build command to run viz script
        viz_script = Path(__file__).parent / 'viz_nba_dnp_scenarios_gt.py'
        cmd = ['python3', str(viz_script)]
        
        # Pass through seasons
        cmd.extend(['--seasons'] + args.seasons)
        
        # Pass through teams if specified
        if args.teams:
            cmd.extend(['--teams'] + args.teams)
        
        # Pass through points threshold
        cmd.extend(['--points-threshold', str(args.points_threshold)])
        
        # Run visualization
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("⚠️  Visualization failed:")
            print(result.stderr)
        else:
            print("✅ Visualization complete!")


if __name__ == '__main__':
    main()
