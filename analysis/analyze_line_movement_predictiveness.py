"""
NFL/NBA Line Movement vs Game Outcome Analysis

Analyzes whether line movements of specific magnitudes (2, 3, 4, 5 points) are 
predictive of game outcomes. Tests the hypothesis: "Does betting with line 
movement lead to profitable results?"

Research Question:
When a betting line moves 2-5 points toward a team, does that team tend to:
1. Cover the closing spread?
2. Win straight up more often?
3. Provide +EV opportunities?

Key Metrics:
- Line movement magnitude (2pt, 3pt, 4pt, 5pt movements)
- Direction of movement (toward favorite vs underdog)
- Cover rate: % of times team getting favorable line movement covers
- Win rate: % of times team getting favorable line movement wins SU
- ROI: Return on investment if betting with line movement
- Sample size per movement bucket

Analysis Approach:
1. Load historical line movement data (opening vs closing lines)
2. Calculate line movement magnitude and direction for each game
3. Group games by movement size (2pt, 3pt, 4pt, 5pt)
4. Determine which team received favorable movement
5. Check actual game results (ATS and SU)
6. Calculate success rates and expected value

Context:
Sharp money theory suggests large line movements indicate informed betting.
However, this needs empirical validation. Common wisdom says:
- Movement > 2 points = sharp action
- Follow the line movement (bet with the sharps)
But does the data actually support this?

Usage:
    # First run (fetches from S3 and builds cache):
    cd betting
    python analysis/analyze_line_movement_predictiveness.py
    
    # Subsequent runs (uses cached parquet files - 50x faster!):
    python analysis/analyze_line_movement_predictiveness.py --use-cache
    
    # Log individual game details (sorted by date, start of season → now):
    python analysis/analyze_line_movement_predictiveness.py --use-cache --log-individual-games
    
    # In notebook (returns dataframes):
    %run analysis/analyze_line_movement_predictiveness.py
    # Access: df, cover_analysis, movements_all, game_results

Output Files (saved to ~/Downloads/tmp/):
    - line_movement_cover_analysis_NBA_YYYYMMDD.csv  # Cover analysis with all bookmaker combos
    - line_movements_all_NBA_YYYYMMDD.csv           # All movements (opening → closing)
    - Cache files (parquet):
      • snapshots_nba_YYYYMMDD.parquet
      • movements_nba_YYYYMMDD.parquet
      • hourly_steam_nba_YYYYMMDD.parquet
      • game_results_nba_YYYYMMDD.parquet

Data Sources:
    - Line movements: S3 betting-line-movement-snapshots (hourly snapshots)
    - Game results: S3 nba-api-mt (player game logs aggregated to team level)
    - Opening line: First hourly snapshot per game/bookmaker
    - Closing line: Last hourly snapshot before game start

Expected Insights:
    - Are 2pt movements meaningful or just market noise?
    - At what threshold does line movement become predictive?
    - Is there a difference between favorite vs underdog line movement?
    - Does sport (NFL vs NBA) affect the predictiveness?

Author: Thomas Myles
Date: 2026-01-13
Context: User request - "i want to use the line movement data i have to see if 
games with a x (2-3-4-5) line movement usualy go in the favor of the team w 
the line movement or not"

    
Cache Location:
    ~/Downloads/tmp/snapshots_nba_20260113.parquet (and related files)
    Cache is dated, automatically creates new cache each day
"""

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import os
import argparse
from pathlib import Path

# Find project root by looking for .gitignore
def find_project_root():
    """Find project root by looking for .gitignore file."""
    current = Path.cwd()
    while current != current.parent:
        if (current / '.gitignore').exists():
            return current
        current = current.parent
    # Fallback to current directory
    return Path.cwd()

PROJECT_ROOT = find_project_root()

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from config_loader import get_config

CONFIG = get_config()

# Constants
S3_BUCKET_SNAPSHOTS = 'betting-line-movement-snapshots'  # Line movement tracking snapshots
OUTPUT_DIR = Path.home() / 'Downloads' / 'tmp'

# Team name normalization (Odds API → NBA API)
TEAM_NAME_MAP = {
    'Los Angeles Clippers': 'LA Clippers',
}

def normalize_team_name(team_name):
    """Normalize team names from Odds API to match NBA API format."""
    return TEAM_NAME_MAP.get(team_name, team_name)


def load_all_line_movement_snapshots(sport='nba'):
    """
    Load all hourly line movement snapshots from S3
    
    These are the hourly snapshots created by track_game_line_movements.py
    
    Args:
        sport: 'nba' or 'nfl' (just the short name)
    
    Returns:
        DataFrame with all snapshots including fetched_at timestamp
    """
    s3_prefix = f'data/01_input/the-odds-api/{sport}/line_movement/'
    
    print(f"\n📥 Loading {sport.upper()} hourly snapshots from S3...")
    print(f"   Bucket: {S3_BUCKET_SNAPSHOTS}")
    print(f"   Prefix: {s3_prefix}")
    
    s3 = boto3.client('s3')
    
    # List all snapshot CSV files
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_SNAPSHOTS, Prefix=s3_prefix)
    except Exception as e:
        print(f"❌ Error accessing S3 bucket: {e}")
        raise
    
    if 'Contents' not in response:
        raise ValueError(f"No snapshots found in S3 for {sport.upper()}")
    
    all_dfs = []
    
    for obj in response.get('Contents', []):
        key = obj['Key']
        
        # Only process snapshot CSV files
        if not key.endswith('.csv') or 'snapshot_' not in key:
            continue
        
        # Read CSV from S3
        try:
            response_obj = s3.get_object(Bucket=S3_BUCKET_SNAPSHOTS, Key=key)
            df = pd.read_csv(BytesIO(response_obj['Body'].read()))
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️  Error reading {key}: {e}")
    
    if not all_dfs:
        raise ValueError(f"No valid snapshot CSV files found for {sport.upper()}")
    
    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Convert timestamps
    df['game_time'] = pd.to_datetime(df['game_time'])
    df['fetched_at'] = pd.to_datetime(df['fetched_at'])
    
    print(f"✅ Loaded {len(df):,} line records from {len(all_dfs)} snapshot files")
    print(f"   Snapshot date range: {df['fetched_at'].min()} to {df['fetched_at'].max()}")
    print(f"   Unique games: {df['game_id'].nunique():,}")
    print(f"   Bookmakers: {df['bookmaker'].nunique()}")
    
    return df


def calculate_consensus_line_movements(snapshots_df):
    """
    Calculate CONSENSUS line movements (averaged across all bookmakers).
    
    Args:
        snapshots_df: DataFrame with hourly snapshots
    
    Returns:
        DataFrame with one row per game containing consensus movements
    """
    print("\n📊 Calculating CONSENSUS line movements (averaged across bookmakers)...")
    
    # Sort by fetched_at
    snapshots_df = snapshots_df.sort_values('fetched_at')
    
    # Group by game_id and fetched_at, calculate consensus (median across books)
    consensus = snapshots_df.groupby(['game_id', 'fetched_at']).agg({
        'game_time': 'first',
        'away_team': 'first',
        'home_team': 'first',
        'away_spread': 'median',
        'home_spread': 'median',
    }).reset_index()
    
    # Now group by game_id to get opening and closing
    grouped = consensus.groupby('game_id')
    
    # Get opening (earliest) and closing (latest) consensus lines
    opening_lines = grouped.first().reset_index()
    closing_lines = grouped.last().reset_index()
    
    # Count snapshots
    snapshots_per_game = grouped.size()
    
    # Merge opening and closing
    movements = opening_lines[['game_id', 'game_time', 'away_team', 'home_team', 
                                'away_spread', 'home_spread', 'fetched_at']].copy()
    movements.columns = ['game_id', 'game_time', 'away_team', 'home_team', 
                         'away_open', 'home_open', 'open_time']
    
    closing_data = closing_lines[['game_id', 'away_spread', 'home_spread', 'fetched_at']].copy()
    closing_data.columns = ['game_id', 'away_close', 'home_close', 'close_time']
    
    movements = movements.merge(closing_data, on='game_id', how='inner')
    
    # Add snapshot count
    movements['num_snapshots'] = movements['game_id'].map(snapshots_per_game)
    
    # Calculate hours tracked
    movements['hours_tracked'] = (movements['close_time'] - movements['open_time']).dt.total_seconds() / 3600
    
    # Calculate line movement
    movements['away_movement'] = movements['away_close'] - movements['away_open']
    movements['home_movement'] = movements['home_close'] - movements['home_open']
    
    # Get magnitude
    movements['movement_magnitude'] = movements[['away_movement', 'home_movement']].abs().max(axis=1)
    
    # Flags
    movements['movement_2plus'] = movements['movement_magnitude'] >= 2.0
    movements['movement_3plus'] = movements['movement_magnitude'] >= 3.0
    movements['movement_4plus'] = movements['movement_magnitude'] >= 4.0
    movements['movement_5plus'] = movements['movement_magnitude'] >= 5.0
    
    # Determine movement team
    def get_movement_team(row):
        if abs(row['away_movement']) > abs(row['home_movement']):
            return row['away_team'] if row['away_movement'] > 0 else row['home_team']
        else:
            return row['home_team'] if row['home_movement'] > 0 else row['away_team']
    
    movements['movement_team'] = movements.apply(get_movement_team, axis=1)
    movements['bookmaker'] = 'CONSENSUS'  # Add for compatibility
    
    print(f"\n✅ Calculated consensus movements for {len(movements):,} games")
    
    if len(movements) > 0:
        print(f"\nMovement Distribution (consensus):")
        print(f"   2+ points: {movements['movement_2plus'].sum():,} games ({movements['movement_2plus'].mean()*100:.1f}%)")
        print(f"   3+ points: {movements['movement_3plus'].sum():,} games ({movements['movement_3plus'].mean()*100:.1f}%)")
        print(f"   4+ points: {movements['movement_4plus'].sum():,} games ({movements['movement_4plus'].mean()*100:.1f}%)")
        print(f"   5+ points: {movements['movement_5plus'].sum():,} games ({movements['movement_5plus'].mean()*100:.1f}%)")
    
    return movements


def calculate_consensus_hourly_steam(snapshots_df):
    """
    Calculate consensus hourly steam - finds the biggest 1-hour spike across all bookmakers.
    
    Args:
        snapshots_df: DataFrame with all hourly snapshots
    
    Returns:
        DataFrame with max 1-hour steam metrics per game
    """
    print("\n🔥 Calculating consensus hourly steam (biggest 1-hour spikes)...")
    
    # Sort by time
    snapshots_df = snapshots_df.sort_values(['game_id', 'fetched_at'])
    
    # Calculate consensus (median) spread at each hour for each game
    consensus_hourly = snapshots_df.groupby(['game_id', 'fetched_at']).agg({
        'game_time': 'first',
        'away_team': 'first',
        'home_team': 'first',
        'away_spread': 'median',
        'home_spread': 'median',
    }).reset_index()
    
    # Rename for clarity
    consensus_hourly = consensus_hourly.rename(columns={
        'away_spread': 'consensus_away_spread',
        'home_spread': 'consensus_home_spread'
    })
    
    # Calculate hour-over-hour changes
    results = []
    
    for game_id, game_df in consensus_hourly.groupby('game_id'):
        if len(game_df) < 2:
            continue
            
        game_df = game_df.sort_values('fetched_at')
        
        max_1hr_away_change = 0
        max_1hr_home_change = 0
        max_1hr_magnitude = 0
        max_1hr_direction_team = None
        
        # Calculate consecutive hour changes
        for i in range(1, len(game_df)):
            prev_row = game_df.iloc[i-1]
            curr_row = game_df.iloc[i]
            
            # Calculate 1-hour changes
            away_change = curr_row['consensus_away_spread'] - prev_row['consensus_away_spread']
            home_change = curr_row['consensus_home_spread'] - prev_row['consensus_home_spread']
            
            # Track the biggest change
            if abs(away_change) > abs(max_1hr_away_change):
                max_1hr_away_change = away_change
                
            if abs(home_change) > abs(max_1hr_home_change):
                max_1hr_home_change = home_change
            
            # Update max magnitude
            hour_magnitude = max(abs(away_change), abs(home_change))
            if hour_magnitude > max_1hr_magnitude:
                max_1hr_magnitude = hour_magnitude
                # Determine which team got the movement
                if abs(away_change) > abs(home_change):
                    max_1hr_direction_team = curr_row['away_team'] if away_change > 0 else curr_row['home_team']
                else:
                    max_1hr_direction_team = curr_row['home_team'] if home_change > 0 else curr_row['away_team']
        
        results.append({
            'game_id': game_id,
            'max_1hr_steam_magnitude': max_1hr_magnitude,
            'max_1hr_steam_direction_team': max_1hr_direction_team,
            'max_1hr_away_change': max_1hr_away_change,
            'max_1hr_home_change': max_1hr_home_change,
        })
    
    steam_df = pd.DataFrame(results)
    
    print(f"✅ Calculated hourly steam for {len(steam_df)} games")
    print(f"   Avg max 1-hour steam: {steam_df['max_1hr_steam_magnitude'].mean():.2f} pts")
    print(f"   Max 1-hour steam seen: {steam_df['max_1hr_steam_magnitude'].max():.1f} pts")
    
    return steam_df


def calculate_line_movements(snapshots_df):
    """
    Calculate opening and closing lines for each game/bookmaker, then compute line movement
    
    Args:
        snapshots_df: DataFrame with hourly snapshots (multiple fetched_at times per game/bookmaker)
                     Columns: game_id, game_time, away_team, home_team, bookmaker, 
                             away_spread, home_spread, fetched_at, etc.
    
    Returns:
        DataFrame with one row per game/bookmaker containing:
        - game_id, game_time, away_team, home_team, bookmaker
        - away_open, home_open (opening spreads)
        - away_close, home_close (closing spreads)
        - away_movement, home_movement (points moved)
        - movement_magnitude (absolute value)
        - movement_4plus (binary flag for 4+ point movement)
        - movement_team (which team line moved toward)
        - num_snapshots (how many hourly snapshots we have for this game/bookmaker)
        - hours_tracked (time between first and last snapshot)
    """
    print("\n📊 Calculating line movements (opening vs closing)...")
    
    # Sort by fetched_at to ensure chronological order
    snapshots_df = snapshots_df.sort_values('fetched_at')
    
    # Group by game_id + bookmaker (each book may have different movement)
    grouped = snapshots_df.groupby(['game_id', 'bookmaker'])
    
    # Count snapshots per game/bookmaker
    snapshots_per_combo = grouped.size()
    print(f"\nSnapshot distribution per game/bookmaker:")
    print(snapshots_per_combo.value_counts().sort_index().to_string())
    
    # Get opening line (earliest fetched_at) and closing line (latest fetched_at) per game/bookmaker
    opening_lines = grouped.first().reset_index()
    closing_lines = grouped.last().reset_index()
    
    # Merge opening and closing
    movements = opening_lines[['game_id', 'bookmaker', 'game_time', 'away_team', 'home_team', 
                                'away_spread', 'home_spread', 'fetched_at']].copy()
    movements.columns = ['game_id', 'bookmaker', 'game_time', 'away_team', 'home_team', 
                         'away_open', 'home_open', 'open_time']
    
    closing_data = closing_lines[['game_id', 'bookmaker', 'away_spread', 'home_spread', 'fetched_at']].copy()
    closing_data.columns = ['game_id', 'bookmaker', 'away_close', 'home_close', 'close_time']
    
    movements = movements.merge(closing_data, on=['game_id', 'bookmaker'], how='inner')
    
    # Add snapshot count per game/bookmaker
    movements['num_snapshots'] = movements.apply(
        lambda row: snapshots_per_combo.get((row['game_id'], row['bookmaker']), 0), axis=1
    )
    
    # Calculate hours between opening and closing snapshots
    movements['hours_tracked'] = (movements['close_time'] - movements['open_time']).dt.total_seconds() / 3600
    
    # Calculate line movement (closing - opening)
    # Positive movement = line moved TOWARD that team (they're getting more points)
    # Negative movement = line moved AWAY from that team (they're giving more points)
    movements['away_movement'] = movements['away_close'] - movements['away_open']
    movements['home_movement'] = movements['home_close'] - movements['home_open']
    
    # Get magnitude of movement (max absolute value between away and home)
    movements['movement_magnitude'] = movements[['away_movement', 'home_movement']].abs().max(axis=1)
    
    # Binary flag for 4+ point movement
    movements['movement_4plus'] = movements['movement_magnitude'] >= 4.0
    
    # Determine which team the movement was toward
    # If away_movement > 0, line moved toward away team
    # If home_movement > 0, line moved toward home team
    def get_movement_team(row):
        if abs(row['away_movement']) > abs(row['home_movement']):
            return row['away_team'] if row['away_movement'] > 0 else row['home_team']
        else:
            return row['home_team'] if row['home_movement'] > 0 else row['away_team']
    
    movements['movement_team'] = movements.apply(get_movement_team, axis=1)
    
    # Add more granular flags
    movements['movement_2plus'] = movements['movement_magnitude'] >= 2.0
    movements['movement_3plus'] = movements['movement_magnitude'] >= 3.0
    movements['movement_5plus'] = movements['movement_magnitude'] >= 5.0
    
    # Filter to games with at least 2 snapshots (otherwise no movement possible)
    games_with_multiple_snapshots = movements[movements['num_snapshots'] >= 2]
    
    print(f"\n✅ Calculated line movements for {len(movements):,} game/bookmaker combinations")
    print(f"   Unique games: {movements['game_id'].nunique()}")
    print(f"   Bookmakers: {movements['bookmaker'].nunique()}")
    print(f"   Combinations with 2+ snapshots: {len(games_with_multiple_snapshots):,} ({len(games_with_multiple_snapshots)/len(movements)*100:.1f}%)")
    
    if len(games_with_multiple_snapshots) > 0:
        print(f"\nMovement Distribution (2+ snapshots only):")
        print(f"   2+ points: {games_with_multiple_snapshots['movement_2plus'].sum():,} ({games_with_multiple_snapshots['movement_2plus'].mean()*100:.1f}%)")
        print(f"   3+ points: {games_with_multiple_snapshots['movement_3plus'].sum():,} ({games_with_multiple_snapshots['movement_3plus'].mean()*100:.1f}%)")
        print(f"   4+ points: {games_with_multiple_snapshots['movement_4plus'].sum():,} ({games_with_multiple_snapshots['movement_4plus'].mean()*100:.1f}%)")
        print(f"   5+ points: {games_with_multiple_snapshots['movement_5plus'].sum():,} ({games_with_multiple_snapshots['movement_5plus'].mean()*100:.1f}%)")
    else:
        print("\n⚠️  No combinations have multiple snapshots - cannot calculate line movements")
    
    return movements


def load_nba_game_results():
    """
    Load NBA game results from S3 player game logs and aggregate to team level.
    
    Returns:
        DataFrame with team-level game results (team scores)
    """
    print("\n📥 Loading NBA game results from S3...")
    
    # Load from nba-api-mt bucket (player game logs, aggregate to team)
    bucket = 'nba-api-mt'
    s3_prefix = 'player_game_logs/2025-26/'
    
    s3 = boto3.client('s3')
    
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
    except Exception as e:
        print(f"❌ Error accessing S3 bucket: {e}")
        return None
    
    if 'Contents' not in response:
        print(f"❌ No game results found in S3")
        return None
    
    all_dfs = []
    
    for obj in response.get('Contents', []):
        key = obj['Key']
        
        if not key.endswith('.csv'):
            continue
        
        try:
            response_obj = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(BytesIO(response_obj['Body'].read()))
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️  Error reading {key}: {e}")
    
    if not all_dfs:
        print(f"❌ No valid game result files found")
        return None
    
    # Combine all player game logs
    df = pd.concat(all_dfs, ignore_index=True)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Aggregate to team level (sum points for each team per game)
    team_games = df.groupby(['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'WL']).agg({
        'PTS': 'sum'  # Sum all player points for the team
    }).reset_index()
    
    print(f"✅ Loaded {len(team_games):,} team-game records from player logs")
    print(f"   Date range: {team_games['GAME_DATE'].min()} to {team_games['GAME_DATE'].max()}")
    
    return team_games


def add_derived_features(df):
    """
    Add derived features for deeper analysis.
    
    ALL FEATURES ANCHORED ON OPENING FAVORITE for consistency.
    
    Args:
        df: DataFrame with cover analysis
    
    Returns:
        DataFrame with additional features
    """
    # =========================================================================
    # DERIVED UNDERDOG METRICS (inverse of favorite)
    # =========================================================================
    # Opening underdog movement is inverse of favorite movement
    df['opening_underdog_movement'] = -df['opening_favorite_movement']
    
    # Opening underdog spread (for reference)
    df['opening_underdog_spread'] = -df['opening_favorite_spread']
    df['closing_underdog_spread'] = -df['closing_favorite_spread']
    
    # Did opening underdog cover?
    df['opening_underdog_covered'] = ~df['opening_favorite_covered']
    df['opening_underdog_cover_margin'] = -df['opening_favorite_cover_margin']
    
    # =========================================================================
    # STEAM TEAM METRICS (which team got the steam?)
    # =========================================================================
    # Did the team that got steam cover?
    df['steam_team_covered'] = df.apply(
        lambda row: row['opening_favorite_covered'] if row['steam_direction'] == 'opening_favorite'
                    else row['opening_underdog_covered'] if row['steam_direction'] == 'opening_underdog'
                    else None,
        axis=1
    )
    df['steam_team_cover_margin'] = df.apply(
        lambda row: row['opening_favorite_cover_margin'] if row['steam_direction'] == 'opening_favorite'
                    else row['opening_underdog_cover_margin'] if row['steam_direction'] == 'opening_underdog'
                    else None,
        axis=1
    )
    
    # Team name that got steam
    df['steam_team'] = df.apply(
        lambda row: row['opening_favorite'] if row['steam_direction'] == 'opening_favorite'
                    else row['opening_underdog'] if row['steam_direction'] == 'opening_underdog'
                    else None,
        axis=1
    )
    
    # =========================================================================
    # CLOSING LINES - Determine closing favorite/underdog
    # =========================================================================
    # Simple check: did line cross zero?
    df['line_crossed_zero'] = (
        (df['opening_favorite_spread'] < 0) & (df['closing_favorite_spread'] > 0)
    ) | (
        (df['opening_favorite_spread'] > 0) & (df['closing_favorite_spread'] < 0)
    )
    
    df['closing_favorite'] = df.apply(
        lambda row: row['opening_underdog'] if row['line_crossed_zero'] else row['opening_favorite'],
        axis=1
    )
    df['closing_underdog'] = df.apply(
        lambda row: row['opening_favorite'] if row['line_crossed_zero'] else row['opening_underdog'],
        axis=1
    )
    
    # =========================================================================
    # BOOLEAN FLAGS - For easier filtering
    # =========================================================================
    df['movement_toward_opening_favorite'] = (df['steam_direction'] == 'opening_favorite')
    df['movement_toward_opening_underdog'] = (df['steam_direction'] == 'opening_underdog')
    
    # Closing movement direction (did steam go toward closing favorite?)
    df['movement_toward_closing_favorite'] = df.apply(
        lambda row: (row['steam_direction'] == 'opening_favorite' and not row['line_crossed_zero']) or
                    (row['steam_direction'] == 'opening_underdog' and row['line_crossed_zero']),
        axis=1
    )
    df['movement_toward_closing_underdog'] = df.apply(
        lambda row: (row['steam_direction'] == 'opening_underdog' and not row['line_crossed_zero']) or
                    (row['steam_direction'] == 'opening_favorite' and row['line_crossed_zero']),
        axis=1
    )
    
    # =========================================================================
    # OVERALL STEAM ALIASES (for consistency with max_1hr_steam naming)
    # =========================================================================
    # Derive team that got steam from steam_direction
    df['overall_steam_direction_team'] = df.apply(
        lambda row: row['opening_favorite'] if row['steam_direction'] == 'opening_favorite'
                    else row['opening_underdog'] if row['steam_direction'] == 'opening_underdog'
                    else None,
        axis=1
    )
    df['overall_steam_direction_fav_dog_at_open'] = df['steam_direction']
    df['overall_steam_direction_fav_dog_at_close'] = df.apply(
        lambda row: 'closing_favorite' if row['movement_toward_closing_favorite'] else 'closing_underdog',
        axis=1
    )
    df['overall_steam_magnitude'] = df['steam_magnitude']
    
    # Opening spread size buckets (use absolute value of favorite's spread)
    df['opening_spread_size'] = df['opening_favorite_spread'].abs()
    df['opening_spread_bucket'] = pd.cut(
        df['opening_spread_size'], 
        bins=[0, 2, 5, 8, 30],
        labels=['close_game', 'small_spread', 'medium_spread', 'blowout']
    )
    
    # Movement speed (points per hour)
    df['movement_per_hour'] = df['steam_magnitude'] / df['hours_tracked'].replace(0, 1)
    df['movement_speed'] = pd.cut(
        df['movement_per_hour'],
        bins=[0, 0.2, 0.5, 100],
        labels=['slow', 'medium', 'fast']
    )
    
    # Fade strategy (betting against the steam)
    # Handle None values in steam_team_covered column
    df['fade_covered'] = df['steam_team_covered'].apply(lambda x: not x if pd.notna(x) else None)
    df['fade_margin'] = df['steam_team_cover_margin'].apply(lambda x: -x if pd.notna(x) else None)
    
    return df


def analyze_favorite_underdog_splits(df):
    """Analyze cover rates split by favorite vs underdog movement."""
    print("\n📊 FAVORITE vs UNDERDOG STEAM ANALYSIS")
    print("=" * 80)
    
    for threshold in [2.0, 3.0, 4.0, 5.0]:
        subset = df[df['steam_magnitude'] >= threshold]
        if len(subset) == 0:
            continue
            
        print(f"\n{threshold}+ Point Steam:")
        
        # Toward opening favorite
        fav_subset = subset[(subset['steam_direction'] == 'opening_favorite') & (subset['steam_team_covered'].notna())]
        if len(fav_subset) > 0:
            wins = int(fav_subset['steam_team_covered'].sum())
            losses = int((~fav_subset['steam_team_covered']).sum())
            ties = 0  # NBA has no ties
            fav_rate = fav_subset['steam_team_covered'].mean() * 100
            fav_margin = fav_subset['steam_team_cover_margin'].mean()
            print(f"  OPENING FAVORITE: {fav_rate:.1f}% cover ({wins}-{losses}-{ties}) | {fav_margin:+.1f} avg margin | N={len(fav_subset)}")
        
        # Toward opening underdog
        dog_subset = subset[(subset['steam_direction'] == 'opening_underdog') & (subset['steam_team_covered'].notna())]
        if len(dog_subset) > 0:
            wins = int(dog_subset['steam_team_covered'].sum())
            losses = int((~dog_subset['steam_team_covered']).sum())
            ties = 0
            dog_rate = dog_subset['steam_team_covered'].mean() * 100
            dog_margin = dog_subset['steam_team_cover_margin'].mean()
            print(f"  OPENING UNDERDOG: {dog_rate:.1f}% cover ({wins}-{losses}-{ties}) | {dog_margin:+.1f} avg margin | N={len(dog_subset)}")


def analyze_line_crossing(df):
    """Analyze games where the line crossed zero (underdog became favorite)."""
    print("\n🔀 LINE CROSSING ANALYSIS (Underdog → Favorite or vice versa)")
    print("=" * 80)
    
    crossed = df[df['line_crossed_zero'] == True]
    not_crossed = df[df['line_crossed_zero'] == False]
    
    print(f"\nGames where line CROSSED ZERO: {len(crossed)}")
    if len(crossed) > 0:
        cross_rate = crossed['steam_team_covered'].mean() * 100
        cross_margin = crossed['steam_team_cover_margin'].mean()
        avg_movement = crossed['steam_magnitude'].mean()
        print(f"  Cover rate (steam team): {cross_rate:.1f}%")
        print(f"  Avg margin: {cross_margin:+.1f}")
        print(f"  Avg steam: {avg_movement:.1f} pts")
        
        # Show examples
        print(f"\n  Examples (top 3 by steam):")
        for _, row in crossed.nlargest(3, 'steam_magnitude').iterrows():
            print(f"    {row['opening_favorite']} (open fav) vs {row['opening_underdog']} (open dog)")
            print(f"      Open: {row['opening_favorite_spread']:+.1f} → Close: {row['closing_favorite_spread']:+.1f}")
            print(f"      Steam: {row['steam_direction']} got {row['steam_magnitude']:.1f} pts")
            print(f"      Covered: {row['steam_team_covered']} | Margin: {row['steam_team_cover_margin']:+.1f}")
    
    print(f"\nGames where line DID NOT CROSS: {len(not_crossed)}")
    if len(not_crossed) > 0:
        no_cross_rate = not_crossed['steam_team_covered'].mean() * 100
        no_cross_margin = not_crossed['steam_team_cover_margin'].mean()
        print(f"  Cover rate (steam team): {no_cross_rate:.1f}%")
        print(f"  Avg margin: {no_cross_margin:+.1f}")


def analyze_home_away_splits(df):
    """Analyze cover rates split by home vs away."""
    print("\n🏠 HOME vs AWAY MOVEMENT ANALYSIS")
    print("=" * 80)
    
    for threshold in [2.0, 3.0, 4.0, 5.0]:
        subset = df[df['steam_magnitude'] >= threshold]
        if len(subset) == 0:
            continue
            
        print(f"\n{threshold}+ Point Movements:")
        
        # Home team
        home_subset = subset[subset['favorable_side'] == 'home']
        if len(home_subset) > 0:
            home_rate = home_subset['covered'].mean() * 100
            home_margin = home_subset['cover_margin'].mean()
            print(f"  HOME team: {home_rate:.1f}% cover | {home_margin:+.1f} avg margin | N={len(home_subset)}")
        
        # Away team
        away_subset = subset[subset['favorable_side'] == 'away']
        if len(away_subset) > 0:
            away_rate = away_subset['covered'].mean() * 100
            away_margin = away_subset['cover_margin'].mean()
            print(f"  AWAY team: {away_rate:.1f}% cover | {away_margin:+.1f} avg margin | N={len(away_subset)}")


def analyze_movement_speed(df):
    """Analyze if movement speed matters."""
    print("\n⚡ MOVEMENT SPEED ANALYSIS")
    print("=" * 80)
    
    subset = df[df['steam_magnitude'] >= 3.0]
    
    if len(subset) > 0:
        print("\n3+ Point Steam by Speed:")
        for speed in ['slow', 'medium', 'fast']:
            speed_subset = subset[subset['movement_speed'] == speed]
            if len(speed_subset) > 0:
                cover_rate = speed_subset['steam_team_covered'].mean() * 100
                avg_margin = speed_subset['steam_team_cover_margin'].mean()
                avg_rate = speed_subset['movement_per_hour'].mean()
                print(f"  {speed.upper():8s}: {cover_rate:.1f}% cover | {avg_margin:+.1f} avg margin | {avg_rate:.2f} pts/hr | N={len(speed_subset)}")


def analyze_spread_context(df):
    """Analyze movements by opening spread context."""
    print("\n📏 SPREAD CONTEXT ANALYSIS")
    print("=" * 80)
    
    subset = df[df['steam_magnitude'] >= 3.0]
    
    if len(subset) > 0:
        print("\n3+ Point Steam by Opening Spread:")
        
        bucket_definitions = {
            'close_game': ('[0-2 pts]', 'CLOSE GAME'),
            'small_spread': ('[2-5 pts]', 'SMALL SPREAD'),
            'medium_spread': ('[5-8 pts]', 'MEDIUM SPREAD'),
            'blowout': ('[8+ pts]', 'BLOWOUT')
        }
        
        for bucket_key, (range_label, name_label) in bucket_definitions.items():
            bucket_subset = subset[(subset['opening_spread_bucket'] == bucket_key) & (subset['steam_team_covered'].notna())]
            if len(bucket_subset) > 0:
                wins = int(bucket_subset['steam_team_covered'].sum())
                losses = int((~bucket_subset['steam_team_covered']).sum())
                cover_rate = bucket_subset['steam_team_covered'].mean() * 100
                avg_margin = bucket_subset['steam_team_cover_margin'].mean()
                avg_spread = bucket_subset['opening_spread_size'].mean()
                print(f"  {range_label:12s} {name_label:14s}: {cover_rate:.1f}% cover ({wins}-{losses}-0) | {avg_margin:+.1f} avg margin | {avg_spread:.1f} avg spread | N={len(bucket_subset)}")


def analyze_hourly_steam(df):
    """Analyze max 1-hour steam vs total steam."""
    print("\n🔥 HOURLY STEAM ANALYSIS (Biggest 1-hour spike vs Total movement)")
    print("=" * 80)
    
    # Filter to games with steam data
    with_steam = df[df['max_1hr_steam_magnitude'].notna()].copy()
    
    if len(with_steam) == 0:
        print("No hourly steam data available")
        return
    
    # Calculate ratio of max 1hr to total
    with_steam['steam_ratio'] = with_steam['max_1hr_steam_magnitude'] / with_steam['overall_steam_magnitude']
    
    print(f"\nTotal games analyzed: {len(with_steam)}")
    print(f"Avg overall steam: {with_steam['overall_steam_magnitude'].mean():.2f} pts")
    print(f"Avg max 1-hour steam: {with_steam['max_1hr_steam_magnitude'].mean():.2f} pts")
    print(f"Avg steam ratio (1hr/total): {with_steam['steam_ratio'].mean():.2%}")
    
    # Analyze by 1-hour steam magnitude
    print("\n📊 Cover Rate by Max 1-Hour Steam:")
    for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
        subset = with_steam[with_steam['max_1hr_steam_magnitude'] >= threshold]
        if len(subset) >= 3:
            cover_rate = subset['steam_team_covered'].mean() * 100
            avg_margin = subset['steam_team_cover_margin'].mean()
            print(f"  {threshold}+ pts in 1hr: {cover_rate:.1f}% cover | {avg_margin:+.1f} avg margin | N={len(subset)}")
    
    # Analyze by 1-hour steam direction
    print("\n📊 Cover Rate by Max 1-Hour Steam Direction:")
    for direction in ['opening_favorite', 'opening_underdog']:
        subset = with_steam[with_steam['max_1hr_steam_direction_fav_dog_at_open'] == direction]
        if len(subset) >= 5:
            cover_rate = subset['steam_team_covered'].mean() * 100
            avg_margin = subset['steam_team_cover_margin'].mean()
            avg_steam = subset['max_1hr_steam_magnitude'].mean()
            print(f"  Toward {direction.upper()}: {cover_rate:.1f}% cover | {avg_margin:+.1f} avg margin | {avg_steam:.2f} avg 1hr steam | N={len(subset)}")


def analyze_fade_strategy(df):
    """Analyze what happens if you FADE (bet against) line movements."""
    print("\n🔄 FADE STRATEGY ANALYSIS (Bet AGAINST line movement)")
    print("=" * 80)
    
    print("\nFading Steam (Betting AGAINST the steam):")
    for threshold in [2.0, 3.0, 4.0, 5.0]:
        subset = df[(df['steam_magnitude'] >= threshold) & (df['fade_covered'].notna())]
        if len(subset) > 0:
            wins = int(subset['fade_covered'].sum())
            losses = int((~subset['fade_covered']).sum())
            fade_rate = subset['fade_covered'].mean() * 100
            fade_margin = subset['fade_margin'].mean()
            print(f"  {threshold}+ pts: {fade_rate:.1f}% cover ({wins}-{losses}-0) | {fade_margin:+.1f} avg margin | N={len(subset)}")


def calculate_cover_rates(movements_df, game_results_df, movement_threshold=4.0):
    """
    Calculate cover rates for teams that received favorable line movement.
    
    Args:
        movements_df: DataFrame with line movements
        game_results_df: DataFrame with game results
        movement_threshold: Threshold for "large" movement (default 4.0 points)
    
    Returns:
        DataFrame with cover analysis
    """
    print(f"\n📊 Calculating cover rates for {movement_threshold}+ point movements...")
    
    # Filter to large movements only  
    # Note: movements_df still uses 'movement_magnitude' column name
    large_moves = movements_df[movements_df['movement_magnitude'] >= movement_threshold].copy()
    
    if large_moves.empty:
        print(f"❌ No movements >= {movement_threshold} points found")
        return None
    
    print(f"   Found {len(large_moves)} game/bookmaker combos with {movement_threshold}+ point moves")
    print(f"   Unique games: {large_moves['game_id'].nunique()}")
    
    # Add binary flag for large movement
    movements_df['large_movement'] = movements_df['movement_magnitude'] >= movement_threshold
    
    # For each game/bookmaker with large movement, check if team with favorable movement covered
    results = []
    games_without_results = 0
    
    for _, row in large_moves.iterrows():
        game_time_utc = pd.to_datetime(row['game_time'])
        # Convert to ET timezone for date matching (NBA games use ET dates)
        game_time_et = game_time_utc.tz_convert(ZoneInfo('America/New_York'))
        game_time = game_time_et  # Use ET time for display
        away_team = row['away_team']
        home_team = row['home_team']
        
        # Determine opening favorite/underdog based on opening spreads
        if row['away_open'] < 0:
            opening_favorite = away_team
            opening_underdog = home_team
            away_is_opening_favorite = True
            # Track opening favorite's spread through open→close
            opening_fav_opening_spread = row['away_open']
            opening_fav_closing_spread = row['away_close']
        else:
            opening_favorite = home_team
            opening_underdog = away_team
            away_is_opening_favorite = False
            # Track opening favorite's spread through open→close
            opening_fav_opening_spread = row['home_open']
            opening_fav_closing_spread = row['home_close']
        
        # Calculate movement (anchored on opening favorite)
        # Positive = favorite got MORE favored (line moved toward them)
        # Negative = favorite got LESS favored (line moved toward underdog)
        # Formula: opening_spread - closing_spread
        # Example: -7.5 - (-10.5) = +3.0 (fav gained 3 pts)
        # Example: -7.5 - (+1.5) = -9.0 (fav lost 9 pts, line crossed)
        opening_favorite_movement = opening_fav_opening_spread - opening_fav_closing_spread
        
        # Determine steam direction
        if opening_favorite_movement > 0:
            steam_direction = 'opening_favorite'
        elif opening_favorite_movement < 0:
            steam_direction = 'opening_underdog'
        else:
            steam_direction = 'no_movement'
        
        # Steam magnitude is absolute value
        steam_magnitude = abs(opening_favorite_movement)
        
        # Find game result
        game_date = game_time.date()
        
        # Normalize team names to match NBA API format
        away_team_normalized = normalize_team_name(away_team)
        home_team_normalized = normalize_team_name(home_team)
        
        # Get both teams' scores for this game
        away_result = game_results_df[
            (game_results_df['TEAM_NAME'] == away_team_normalized) &
            (game_results_df['GAME_DATE'].dt.date == game_date)
        ]
        
        home_result = game_results_df[
            (game_results_df['TEAM_NAME'] == home_team_normalized) &
            (game_results_df['GAME_DATE'].dt.date == game_date)
        ]
        
        if away_result.empty or home_result.empty:
            # Game hasn't happened yet or data not available
            games_without_results += 1
            continue
        
        away_score = away_result.iloc[0]['PTS']
        home_score = home_result.iloc[0]['PTS']
        
        # Calculate scores from opening favorite's perspective
        if away_is_opening_favorite:
            opening_favorite_score = away_score
            opening_underdog_score = home_score
        else:
            opening_favorite_score = home_score
            opening_underdog_score = away_score
        
        # Calculate margin from opening favorite's perspective
        margin_from_fav_perspective = opening_favorite_score - opening_underdog_score
        
        # Did opening favorite cover their closing spread?
        # Standard ATS formula: (actual_margin + spread) > 0
        opening_favorite_cover_margin = margin_from_fav_perspective + opening_fav_closing_spread
        opening_favorite_covered = opening_favorite_cover_margin > 0
        
        # Determine steam team covered (for immediate use)
        if steam_direction == 'opening_favorite':
            steam_team_covered = opening_favorite_covered
            steam_team_cover_margin = opening_favorite_cover_margin
        elif steam_direction == 'opening_underdog':
            steam_team_covered = not opening_favorite_covered
            steam_team_cover_margin = -opening_favorite_cover_margin
        else:
            steam_team_covered = None
            steam_team_cover_margin = None
        
        results.append({
            'game_id': row['game_id'],
            'game_time': game_time,
            'game_date': game_date,
            'bookmaker': row['bookmaker'],
            
            # Teams (anchored on opening favorite)
            'opening_favorite': opening_favorite,
            'opening_underdog': opening_underdog,
            
            # Spreads (anchored on opening favorite)
            'opening_favorite_spread': opening_fav_opening_spread,
            'closing_favorite_spread': opening_fav_closing_spread,
            
            # Movement (anchored on opening favorite)
            'opening_favorite_movement': opening_favorite_movement,
            
            # Steam
            'steam_direction': steam_direction,
            'steam_magnitude': steam_magnitude,
            
            # Scores (anchored on opening favorite)
            'opening_favorite_score': opening_favorite_score,
            'opening_underdog_score': opening_underdog_score,
            
            # Cover analysis (anchored on opening favorite)
            'opening_favorite_covered': opening_favorite_covered,
            'opening_favorite_cover_margin': opening_favorite_cover_margin,
            
            # Steam team cover (derived for convenience)
            'steam_team_covered': steam_team_covered,
            'steam_team_cover_margin': steam_team_cover_margin,
            
            # Metadata
            'num_snapshots': row['num_snapshots'],
            'hours_tracked': row['hours_tracked'],
            
            # Away/Home columns (reference only - at end)
            'away_team': away_team,
            'home_team': home_team,
            'away_is_opening_favorite': away_is_opening_favorite,
            'away_score': away_score,
            'home_score': home_score,
            'away_open': row['away_open'],
            'away_close': row['away_close'],
            'home_open': row['home_open'],
            'home_close': row['home_close'],
        })
    
    results_df = pd.DataFrame(results)
    
    # Show diagnostic info
    print(f"\n4️⃣  Results matching:")
    print(f"    Games with {movement_threshold}+ pt movements: {len(large_moves)} combos ({large_moves['game_id'].nunique()} unique games)")
    print(f"    Games WITHOUT results: {games_without_results} combos")
    print(f"    Games WITH results: {len(results_df)} combos ({results_df['game_id'].nunique() if not results_df.empty else 0} unique games)")
    
    if results_df.empty:
        print(f"❌ No completed games with {movement_threshold}+ point movements")
        return None
    
    # Calculate cover rate (for team that got steam)
    cover_rate = results_df['steam_team_covered'].mean()
    total_games = len(results_df)
    covered_count = results_df['steam_team_covered'].sum()
    
    # Calculate mean cover margin (positive = covered by X, negative = missed by X)
    mean_margin = results_df['steam_team_cover_margin'].mean()
    
    print(f"\n✅ Cover Rate Analysis (Team That Got Steam):")
    print(f"   Games analyzed: {total_games}")
    print(f"   Covered: {covered_count}")
    print(f"   Did not cover: {total_games - covered_count}")
    print(f"   Cover rate: {cover_rate*100:.1f}%")
    print(f"   Mean cover margin: {mean_margin:+.1f} pts (avg miss by {abs(mean_margin):.1f} pts)")
    
    # Break down by steam magnitude
    print(f"\n📈 Cover Rate by Steam Size:")
    for threshold in [2.0, 3.0, 4.0, 5.0]:
        subset = results_df[(results_df['steam_magnitude'] >= threshold) & (results_df['steam_team_covered'].notna())]
        if len(subset) > 0:
            wins = int(subset['steam_team_covered'].sum())
            losses = int((~subset['steam_team_covered']).sum())
            subset_rate = subset['steam_team_covered'].mean()
            subset_margin = subset['steam_team_cover_margin'].mean()
            print(f"   {threshold}+ points: {subset_rate*100:.1f}% ({wins}-{losses}-0) | Avg margin: {subset_margin:+.1f} pts | N={len(subset)}")
    
    return results_df


def calculate_roi_at_110(wins, losses):
    """
    Calculate ROI assuming -110 odds (standard American sports betting).
    
    Args:
        wins: Number of winning bets
        losses: Number of losing bets
    
    Returns:
        tuple: (net_profit, roi_percent)
    """
    profit_from_wins = wins * 100  # Win $100 per bet
    loss_from_losses = losses * 110  # Lose $110 per bet
    net_profit = profit_from_wins - loss_from_losses
    total_risked = (wins + losses) * 110  # Risk $110 per bet
    roi_percent = (net_profit / total_risked * 100) if total_risked > 0 else 0
    return net_profit, roi_percent


def log_individual_games(df, show_summary=False, steam_threshold=1.0):
    """Log detailed breakdown for each individual game."""
    print("\n" + "=" * 80)
    print("📋 INDIVIDUAL GAME BREAKDOWN (Sorted by Date)")
    print("=" * 80)
    
    # Sort by date
    df_sorted = df.sort_values('game_date')
    
    for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"\n{'─'*80}")
        print(f"#{idx} | {row['game_date']} | {row['bookmaker'].upper()}")
        
        # Identify steam team
        if row['steam_direction'] == 'opening_favorite':
            steam_team = row['opening_favorite']
            steam_closing_spread = row['closing_favorite_spread']
        elif row['steam_direction'] == 'opening_underdog':
            steam_team = row['opening_underdog']
            steam_closing_spread = -row['closing_favorite_spread']
        else:
            steam_team = 'NO MOVEMENT'
            steam_closing_spread = 0
        
        # Matchup and lines
        print(f"🏀 {row['opening_favorite']} (fav) vs {row['opening_underdog']} (dog)")
        print(f"📊 Open: {row['opening_favorite_spread']:+.1f} → Close: {row['closing_favorite_spread']:+.1f} | Movement: {row['opening_favorite_movement']:+.1f}")
        print(f"🔥 Steam: {row['steam_direction'].replace('_', ' ').upper()} ({row['steam_magnitude']:.1f} pts) → {steam_team} {steam_closing_spread:+.1f}")
        
        # Result
        margin = row['opening_favorite_score'] - row['opening_underdog_score']
        if margin > 0:
            winner = row['opening_favorite']
        elif margin < 0:
            winner = row['opening_underdog']
        else:
            winner = "TIE"
        
        print(f"🏆 Final: {row['opening_favorite_score']:.0f}-{row['opening_underdog_score']:.0f} | Winner: {winner} (by {abs(margin):.0f})")
        
        # Cover result
        if row['steam_direction'] != 'no_movement':
            steam_result = '✅' if row['steam_team_covered'] else '❌'
            print(f"✅ Steam Team Covered: {steam_result} ({row['steam_team_cover_margin']:+.1f} margin)")
        
        # Metadata
        meta_parts = [f"{row['num_snapshots']:.0f} snaps", f"{row['hours_tracked']:.1f}hrs"]
        if 'max_1hr_steam_magnitude' in row and pd.notna(row['max_1hr_steam_magnitude']):
            meta_parts.append(f"{row['max_1hr_steam_magnitude']:.1f} max/hr")
        print(f"📈 {' | '.join(meta_parts)}")
    
    # Print summary of records (only if show_summary is True)
    if not show_summary:
        return
    
    print(f"\n{'='*80}")
    print("📊 STEAM TEAM COVER RECORDS BY THRESHOLD")
    print(f"{'='*80}\n")
    
    # Build dynamic thresholds starting from steam_threshold
    import math
    thresholds = []
    
    # Start with the steam_threshold value itself
    current = steam_threshold
    thresholds.append(current)
    
    # Round up to next whole number if not already whole
    if current != math.ceil(current):
        current = math.ceil(current)
        thresholds.append(current)
    
    # Continue incrementing by 1 until no games match
    max_steam = df_sorted['steam_magnitude'].max()
    current = math.ceil(current)
    while current < max_steam:
        current += 1.0
        if (df_sorted['steam_magnitude'] >= current).any():
            thresholds.append(current)
        else:
            break
    
    for threshold in thresholds:
        df_threshold = df_sorted[df_sorted['steam_magnitude'] >= threshold].copy()
        
        if len(df_threshold) > 0:
            wins = (df_threshold['steam_team_covered'] == True).sum()
            losses = (df_threshold['steam_team_covered'] == False).sum()
            ties = 0  # We don't track ties explicitly, but covers are binary
            total = len(df_threshold)
            cover_pct = (wins / total * 100) if total > 0 else 0
            avg_margin = df_threshold['steam_team_cover_margin'].mean()
            
            # Calculate ROI at -110 odds
            net_profit, roi_pct = calculate_roi_at_110(wins, losses)
            
            print(f"{threshold:.1f}+ points: {cover_pct:.1f}% ({wins}-{losses}-{ties}) | Avg margin: {avg_margin:+.1f} pts | ROI: ${net_profit:+,.0f} ({roi_pct:+.1f}%) | N={total}")
    
    # Print summary by favorite/dog
    print(f"\n{'='*80}")
    print("📊 STEAM TEAM COVER RECORDS BY STEAM DIRECTION (FAV vs DOG)")
    print(f"{'='*80}\n")
    
    for direction in ['opening_favorite', 'opening_underdog']:
        df_direction = df_sorted[df_sorted['steam_direction'] == direction].copy()
        
        if len(df_direction) > 0:
            wins = (df_direction['steam_team_covered'] == True).sum()
            losses = (df_direction['steam_team_covered'] == False).sum()
            ties = 0
            total = len(df_direction)
            cover_pct = (wins / total * 100) if total > 0 else 0
            avg_margin = df_direction['steam_team_cover_margin'].mean()
            avg_steam = df_direction['steam_magnitude'].mean()
            
            # Calculate ROI at -110 odds
            net_profit, roi_pct = calculate_roi_at_110(wins, losses)
            
            label = "FAVORITE" if direction == "opening_favorite" else "UNDERDOG"
            print(f"Steam → {label}: {cover_pct:.1f}% ({wins}-{losses}-{ties}) | Avg margin: {avg_margin:+.1f} pts | Avg steam: {avg_steam:.1f} pts | ROI: ${net_profit:+,.0f} ({roi_pct:+.1f}%) | N={total}")
    
    # Print summary by BOTH threshold AND direction
    print(f"\n{'='*80}")
    print("📊 STEAM TEAM COVER RECORDS BY THRESHOLD × DIRECTION")
    print(f"{'='*80}\n")
    
    for threshold in thresholds:
        df_threshold = df_sorted[df_sorted['steam_magnitude'] >= threshold].copy()
        
        if len(df_threshold) > 0:
            print(f"{threshold:.1f}+ Point Steam:")
            
            for direction in ['opening_favorite', 'opening_underdog']:
                df_combo = df_threshold[df_threshold['steam_direction'] == direction].copy()
                
                if len(df_combo) > 0:
                    wins = (df_combo['steam_team_covered'] == True).sum()
                    losses = (df_combo['steam_team_covered'] == False).sum()
                    ties = 0
                    total = len(df_combo)
                    cover_pct = (wins / total * 100) if total > 0 else 0
                    avg_margin = df_combo['steam_team_cover_margin'].mean()
                    
                    # Calculate ROI at -110 odds
                    net_profit, roi_pct = calculate_roi_at_110(wins, losses)
                    
                    label = "FAVORITE" if direction == "opening_favorite" else "UNDERDOG"
                    print(f"  → {label:8s}: {cover_pct:.1f}% ({wins}-{losses}-{ties}) | Avg margin: {avg_margin:+.1f} pts | ROI: ${net_profit:+,.0f} ({roi_pct:+.1f}%) | N={total}")
            
            print()  # Blank line between thresholds


def main(use_cache=False, log_individual_games_flag=False, summary_flag=False, steam_threshold=1.0):
    """Main analysis"""
    print("=" * 80)
    print("LINE MOVEMENT PREDICTIVENESS ANALYSIS")
    print("=" * 80)
    print(f"Steam Threshold: {steam_threshold}+ points")
    
    # Load NBA hourly snapshots
    sport = 'nba'
    today = datetime.now().strftime('%Y%m%d')
    
    # Cache file paths
    cache_dir = Path.home() / 'Downloads' / 'tmp'
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshots_cache = cache_dir / f'snapshots_{sport}_{today}.parquet'
    movements_cache = cache_dir / f'movements_{sport}_{today}.parquet'
    hourly_steam_cache = cache_dir / f'hourly_steam_{sport}_{today}.parquet'
    game_results_cache = cache_dir / f'game_results_{sport}_{today}.parquet'
    
    # Try to load from cache
    if use_cache and snapshots_cache.exists():
        print(f"\n📦 Loading from cache...")
        snapshots_df = pd.read_parquet(snapshots_cache)
        movements_df = pd.read_parquet(movements_cache)
        hourly_steam_df = pd.read_parquet(hourly_steam_cache)
        game_results_df = pd.read_parquet(game_results_cache)
        print(f"✅ Loaded cached data from {today}")
    else:
        # Load all hourly snapshots from line movement tracking
        snapshots_df = load_all_line_movement_snapshots(sport)
        
        # Calculate consensus hourly steam (biggest 1-hour spikes)
        hourly_steam_df = calculate_consensus_hourly_steam(snapshots_df)
        
        # Calculate movements (opening → closing)
        movements_df = calculate_line_movements(snapshots_df)
        
        # Load game results
        game_results_df = load_nba_game_results()
        
        # Save to cache
        print(f"\n💾 Saving to cache for future runs...")
        snapshots_df.to_parquet(snapshots_cache)
        movements_df.to_parquet(movements_cache)
        hourly_steam_df.to_parquet(hourly_steam_cache)
        if game_results_df is not None:
            game_results_df.to_parquet(game_results_cache)
        print(f"✅ Cached data saved to {cache_dir}")
    
    # Initialize for return value
    cover_analysis_df = None
    cover_analysis_df_deduped = None
    cover_analysis_df_raw = None  # Unfiltered by threshold
    
    # DIAGNOSTIC: Show filtering funnel
    print("\n" + "=" * 80)
    print("🔍 DIAGNOSTIC: DATA FILTERING FUNNEL")
    print("=" * 80)
    print(f"\n1️⃣  Total game/bookmaker combos in movements_df: {len(movements_df):,}")
    print(f"    Unique games: {movements_df['game_id'].nunique():,}")
    
    # Show movement distribution
    print(f"\n2️⃣  Movement magnitude distribution:")
    for threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]:
        count = (movements_df['movement_magnitude'] >= threshold).sum()
        pct = count / len(movements_df) * 100
        unique_games = movements_df[movements_df['movement_magnitude'] >= threshold]['game_id'].nunique()
        print(f"    {threshold:3.1f}+ pts: {count:4,} combos ({pct:4.1f}%) | {unique_games:3,} unique games")
    
    # Calculate cover rates for teams with favorable line movement
    if game_results_df is not None:
        # Show game results info
        print(f"\n3️⃣  Game results available:")
        print(f"    Total team-game records: {len(game_results_df):,}")
        print(f"    Date range: {game_results_df['GAME_DATE'].min().date()} to {game_results_df['GAME_DATE'].max().date()}")
        print(f"    Unique game dates: {game_results_df['GAME_DATE'].dt.date.nunique():,}")
        
        # Get ALL games (no threshold filter) for df_raw
        print(f"\n   Calculating cover rates for ALL games (threshold=0.0)...")
        cover_analysis_df_raw = calculate_cover_rates(movements_df, game_results_df, movement_threshold=0.0)
        
        # Get filtered version based on steam_threshold
        cover_analysis_df = calculate_cover_rates(movements_df, game_results_df, movement_threshold=steam_threshold)
        
        # Process raw data (same as filtered data)
        if cover_analysis_df_raw is not None:
            cover_analysis_df_raw = cover_analysis_df_raw.merge(
                hourly_steam_df[['game_id', 'max_1hr_steam_magnitude', 'max_1hr_steam_direction_team']],
                on='game_id',
                how='left'
            )
            cover_analysis_df_raw = add_derived_features(cover_analysis_df_raw)
            cover_analysis_df_raw['max_1hr_steam_direction_fav_dog_at_open'] = cover_analysis_df_raw.apply(
                lambda row: 'opening_favorite' if row['max_1hr_steam_direction_team'] == row['opening_favorite'] 
                else 'opening_underdog' if pd.notna(row['max_1hr_steam_direction_team']) else None,
                axis=1
            )
            cover_analysis_df_raw['max_1hr_steam_direction_fav_dog_at_close'] = cover_analysis_df_raw.apply(
                lambda row: 'closing_favorite' if row['max_1hr_steam_direction_team'] == row['closing_favorite']
                else 'closing_underdog' if pd.notna(row['max_1hr_steam_direction_team']) else None,
                axis=1
            )
        
        if cover_analysis_df is not None:
            # Merge hourly steam data
            cover_analysis_df = cover_analysis_df.merge(
                hourly_steam_df[['game_id', 'max_1hr_steam_magnitude', 'max_1hr_steam_direction_team']],
                on='game_id',
                how='left'
            )
            
            # Add derived features for deeper analysis
            cover_analysis_df = add_derived_features(cover_analysis_df)
            
            # Add max 1hr steam direction labels (after derived features so we have opening/closing favorite)
            cover_analysis_df['max_1hr_steam_direction_fav_dog_at_open'] = cover_analysis_df.apply(
                lambda row: 'opening_favorite' if row['max_1hr_steam_direction_team'] == row['opening_favorite'] 
                else 'opening_underdog' if pd.notna(row['max_1hr_steam_direction_team']) else None,
                axis=1
            )
            
            cover_analysis_df['max_1hr_steam_direction_fav_dog_at_close'] = cover_analysis_df.apply(
                lambda row: 'closing_favorite' if row['max_1hr_steam_direction_team'] == row['closing_favorite']
                else 'closing_underdog' if pd.notna(row['max_1hr_steam_direction_team']) else None,
                axis=1
            )
            
            # Create output directory
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            today = datetime.now().strftime('%Y%m%d')
            output_file = OUTPUT_DIR / f'line_movement_cover_analysis_{sport.upper()}_{today}.csv'
            cover_analysis_df.to_csv(output_file, index=False)
            
            print(f"\n💾 Saved cover analysis to:")
            print(f"   {output_file}")
            
            # Show sample of games (one per unique game to avoid duplicates)
            print(f"\n📋 Sample Games (sorted by movement magnitude):")
            
            # Deduplicate: sort by steam_magnitude DESC first, then keep first occurrence per game
            # This ensures we keep the bookmaker with the LARGEST steam for each game
            cover_analysis_df_deduped = cover_analysis_df.sort_values('steam_magnitude', ascending=False).drop_duplicates(subset=['game_id'], keep='first').copy()
            
            print(f"\n5️⃣  Deduplication (one game per unique game_id):")
            print(f"    Before: {len(cover_analysis_df):,} rows (multiple bookmakers per game)")
            print(f"    After:  {len(cover_analysis_df_deduped):,} rows (one per unique game)")
            print(f"    Lost:   {len(cover_analysis_df) - len(cover_analysis_df_deduped):,} duplicate game_ids")
            print("=" * 80)
            
            display_cols = ['game_date', 'bookmaker',
                           'opening_favorite', 'opening_underdog',
                           'opening_favorite_spread', 'closing_favorite_spread',
                           'opening_favorite_movement', 'steam_direction', 'steam_magnitude',
                           'opening_favorite_score', 'opening_underdog_score',
                           'opening_favorite_covered', 'opening_favorite_cover_margin',
                           'steam_team_covered', 'steam_team_cover_margin']
            print(cover_analysis_df_deduped[display_cols].head(10).to_string(index=False))
            
            # Run deeper analysis
            analyze_favorite_underdog_splits(cover_analysis_df_deduped)
            analyze_line_crossing(cover_analysis_df_deduped)
            analyze_hourly_steam(cover_analysis_df_deduped)
            analyze_movement_speed(cover_analysis_df_deduped)
            analyze_spread_context(cover_analysis_df_deduped)
            analyze_fade_strategy(cover_analysis_df_deduped)
            
            # Log individual games if requested
            if log_individual_games_flag:
                log_individual_games(cover_analysis_df_deduped, show_summary=summary_flag, steam_threshold=steam_threshold)
    else:
        print("\n⚠️  No game results available - saving movement data only")
    
    # Save all movements (including those without results yet)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    output_file = OUTPUT_DIR / f'line_movements_all_{sport.upper()}_{today}.csv'
    movements_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved all line movements to:")
    print(f"   {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)
    
    # Deduplicate raw data (one per game)
    df_raw = None
    if cover_analysis_df_raw is not None:
        df_raw = cover_analysis_df_raw.sort_values('steam_magnitude', ascending=False).drop_duplicates(subset=['game_id'], keep='first').copy()
    
    # Return dataframes for notebook use
    result = {
        'movements_all': movements_df,  # All movements (1683 game/bookmaker combos)
        'cover_analysis': cover_analysis_df,  # All with results (413 combos)
        'cover_analysis_deduped': cover_analysis_df_deduped,  # One per game (70 games)
        'df_raw': df_raw,  # All games with results, unfiltered by threshold (deduplicated)
        'game_results': game_results_df,  # Raw game results
    }
    
    return result


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze NBA line movement predictiveness')
    parser.add_argument('--use-cache', action='store_true', 
                       help='Use cached data from ~/Downloads/tmp/ instead of fetching from S3')
    parser.add_argument('--log-individual-games', action='store_true',
                       help='Log detailed breakdown for each individual game (sorted by date)')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary statistics sections (by threshold, direction, and combined)')
    parser.add_argument('--steam-threshold', type=float, default=1.0,
                       help='Minimum steam threshold in points (default: 1.0)')
    args = parser.parse_args()
    
    # Run analysis and store results
    results = main(
        use_cache=args.use_cache, 
        log_individual_games_flag=args.log_individual_games,
        summary_flag=args.summary,
        steam_threshold=args.steam_threshold
    )
    
    # Make dataframes available for notebook use
    if results:
        movements_all = results['movements_all']
        cover_analysis = results['cover_analysis']
        df = results['cover_analysis_deduped']  # Main dataframe (deduplicated)
        df_raw = results['df_raw']  # Unfiltered by threshold
        game_results = results['game_results']
        
        print("\n📊 DataFrames available for analysis:")
        print(f"   df                 : {len(df) if df is not None else 0} games (deduplicated, biggest move per game, threshold filtered)")
        print(f"   df_raw             : {len(df_raw) if df_raw is not None else 0} games (deduplicated, ALL games with results)")
        print(f"   cover_analysis     : {len(cover_analysis) if cover_analysis is not None else 0} rows (all bookmaker combos)")
        print(f"   movements_all      : {len(movements_all)} rows (all movements)")
        print(f"   game_results       : {len(game_results) if game_results is not None else 0} rows (raw results)")

