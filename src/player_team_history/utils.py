"""
Team history utilities - join player props with historical team assignments.

This module provides functions to join player prop data with team info using
game dates, accounting for trades and team changes throughout the season.

Key Functions:
==============
- add_team_from_history(): Add team column to props DataFrame (MAIN FUNCTION)
- get_player_team_at_date(): Get a player's team on a specific date
- load_team_history(): Load player team history from S3
- get_team_history_for_player(): Get full history for one player

Usage Example:
==============
    import pandas as pd
    from src.player_team_history import add_team_from_history
    
    # Your prop data with game dates
    props_df = pd.DataFrame({
        'player': ['Anthony Davis', 'Lebron James'],
        'game_date': ['2026-01-15', '2026-02-10'],
        'points': [28, 25]
    })
    
    # Add team column based on game date
    props_df = add_team_from_history(props_df, player_col='player', date_col='game_date')
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys
import boto3
from io import BytesIO

# Find repo root
current_dir = Path(__file__).resolve()
repo_root = current_dir
while not (repo_root / '.gitignore').exists():
    repo_root = repo_root.parent
    if repo_root == repo_root.parent:
        raise RuntimeError("Could not find repo root")

sys.path.append(str(repo_root))

from src.player_name_utils import normalize_player_name, get_name_mappings

# S3 Configuration
S3_BUCKET = 'nba-betting-mt'
S3_KEY = 'nba/player_team_history/history.parquet'


def load_team_history() -> pd.DataFrame:
    """
    Load player team history from S3.
    
    Returns:
        DataFrame with columns: player_normalized, team, valid_from, valid_to
        - valid_from: datetime.date when player joined team
        - valid_to: datetime.date when player left team (NaT = current team)
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    history_df = pd.read_parquet(BytesIO(obj['Body'].read()))
    
    # Ensure date columns are date objects
    history_df['valid_from'] = pd.to_datetime(history_df['valid_from']).dt.date
    history_df['valid_to'] = pd.to_datetime(history_df['valid_to'], errors='coerce').dt.date
    
    print(f"✅ Loaded {len(history_df)} team history records from S3")
    return history_df


def get_player_team_at_date(player_name: str, game_date, history_df: Optional[pd.DataFrame] = None) -> str:
    """
    Get a player's team on a specific date.
    
    Args:
        player_name: Player name (will be normalized)
        game_date: Game date (string, datetime, or date object)
        history_df: Optional pre-loaded history DataFrame (loads if not provided)
        
    Returns:
        Team abbreviation (e.g., 'LAL')
    """
    if history_df is None:
        history_df = load_team_history()
    
    # Normalize player name
    player_normalized = normalize_player_name(player_name)
    
    # Apply name mappings for nickname variations
    name_mappings = get_name_mappings()
    player_normalized = name_mappings[player_normalized] if player_normalized in name_mappings else player_normalized
    
    # Convert game_date to date object
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date).date()
    elif hasattr(game_date, 'date'):
        game_date = game_date.date()
    
    # Filter to this player's history
    player_history = history_df[history_df['player_normalized'] == player_normalized]
    
    if player_history.empty:
        raise ValueError(f"Player not found in history: {player_name} (normalized: {player_normalized})")
    
    # Find the team stint that contains this game date
    for _, row in player_history.iterrows():
        valid_from = row['valid_from']
        valid_to = row['valid_to']
        
        # Check if game_date is within range
        if game_date >= valid_from:
            # valid_to is NaT (None) for current team, or game_date must be <= valid_to
            if pd.isna(valid_to) or game_date <= valid_to:
                return row['team']
    
    raise ValueError(
        f"No team found for {player_name} on {game_date}\n"
        f"Player history:\n{player_history.to_string()}"
    )


def add_team_from_history(df: pd.DataFrame, 
                          player_col: str = 'player',
                          date_col: str = 'game_date') -> pd.DataFrame:
    """
    Add team column to DataFrame using player team history and game dates.
    
    This is the main function you'll use to join props data with team info.
    
    Args:
        df: DataFrame with player names and game dates
        player_col: Name of player column (default: 'player')
        date_col: Name of game date column (default: 'game_date')
        
    Returns:
        DataFrame with new 'team' column added
        
    Raises:
        ValueError: If date_col not in DataFrame
        ValueError: If any player/date combo not found in history
        
    Example:
        props_df = add_team_from_history(props_df, player_col='player', date_col='game_date')
    """
    df = df.copy()
    
    # Load history once
    history_df = load_team_history()
    
    # Validate date column exists
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame. Available: {list(df.columns)}")
    
    # Convert game_date to date objects
    df[f'{date_col}_parsed'] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    
    # Normalize player names
    df['player_normalized'] = df[player_col].apply(normalize_player_name)
    
    # Apply name mappings for nickname variations
    name_mappings = get_name_mappings()
    df['player_normalized'] = df['player_normalized'].map(lambda x: name_mappings[x] if x in name_mappings else x)
    
    # Get team for each row
    teams = []
    errors = []
    
    for idx, row in df.iterrows():
        player_norm = row['player_normalized']
        game_date = row[f'{date_col}_parsed']
        
        if pd.isna(game_date):
            errors.append(f"Row {idx}: Invalid date for {row[player_col]}: {row[date_col]}")
            teams.append(None)
            continue
        
        try:
            team = get_player_team_at_date(player_norm, game_date, history_df)
            teams.append(team)
        except ValueError as e:
            errors.append(f"Row {idx}: {str(e)}")
            teams.append(None)
    
    df['team'] = teams
    
    # Clean up temporary columns
    df = df.drop([f'{date_col}_parsed', 'player_normalized'], axis=1)
    
    # Report errors
    if errors:
        error_msg = f"Failed to find teams for {len(errors)} rows:\n" + "\n".join(errors[:10])
        if len(errors) > 10:
            error_msg += f"\n... and {len(errors) - 10} more"
        raise ValueError(error_msg)
    
    return df


def get_team_history_for_player(player_name: str, history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Get full team history for a specific player.
    
    Args:
        player_name: Player name (will be normalized)
        history_df: Optional pre-loaded history DataFrame
        
    Returns:
        DataFrame with team history for this player
    """
    if history_df is None:
        history_df = load_team_history()
    
    player_normalized = normalize_player_name(player_name)
    
    # Apply name mappings
    name_mappings = get_name_mappings()
    player_normalized = name_mappings[player_normalized] if player_normalized in name_mappings else player_normalized
    
    player_history = history_df[history_df['player_normalized'] == player_normalized].copy()
    
    if player_history.empty:
        raise ValueError(f"Player not found in history: {player_name} (normalized: {player_normalized})")
    
    return player_history


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Testing Team History Utils")
    print("=" * 70)
    print()
    
    # Test 1: Load history
    print("Test 1: Loading team history...")
    history = load_team_history()
    print(f"   Loaded {len(history)} records for {history['player_normalized'].nunique()} players")
    print()
    
    # Test 2: Get team for specific player/date
    print("Test 2: Get team for player on specific date...")
    test_cases = [
        ('Anthony Davis', '2025-12-15'),
        ('Lebron James', '2026-01-15'),
    ]
    
    for player, date in test_cases:
        try:
            team = get_player_team_at_date(player, date)
            print(f"   {player} on {date}: {team}")
        except ValueError as e:
            print(f"   {player} on {date}: ERROR - {e}")
    print()
    
    # Test 3: Add team to DataFrame
    print("Test 3: Add team column to props DataFrame...")
    test_df = pd.DataFrame({
        'player': ['Anthony Davis', 'Lebron James', 'Stephen Curry'],
        'game_date': ['2025-12-15', '2026-02-10', '2026-01-20'],
        'points': [28, 25, 30]
    })
    
    print("   Before:")
    print(test_df.to_string(index=False))
    print()
    
    try:
        test_df = add_team_from_history(test_df)
        print("   After:")
        print(test_df.to_string(index=False))
    except ValueError as e:
        print(f"   ERROR: {e}")
    print()
    
    # Test 4: Get player history
    print("Test 4: Get full history for a player...")
    try:
        player_history = get_team_history_for_player('Anthony Davis')
        print(f"   Anthony Davis team history:")
        print(player_history.to_string(index=False))
    except ValueError as e:
        print(f"   ERROR: {e}")
