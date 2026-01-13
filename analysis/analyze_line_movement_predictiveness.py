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
    python analysis/analyze_line_movement_predictiveness.py --sport nba --season 2025
    python analysis/analyze_line_movement_predictiveness.py --sport nfl --season 2024

Output:
    - CSV: data/04_output/line_movement/line_movement_analysis_NBA_YYYYMMDD.csv
    - CSV: data/04_output/line_movement/line_movement_by_magnitude_NBA_YYYYMMDD.csv
    - Console: Summary tables showing cover rates by movement size

Data Sources:
    - Opening lines: The Odds API S3 bucket 'the-odds-api-mt' (earliest timestamp per game)
    - Closing lines: The Odds API S3 bucket (latest timestamp before game start)
    - Game results: NBA API / Pro Football Reference scraped data

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

Usage:
    cd betting
    python analysis/analyze_line_movement_predictiveness.py --sport nba --season 2025-26
"""

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import os
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
    
    for _, row in large_moves.iterrows():
        game_time_utc = pd.to_datetime(row['game_time'])
        # Convert to ET timezone for date matching (NBA games use ET dates)
        game_time_et = game_time_utc.tz_convert(ZoneInfo('America/New_York'))
        game_time = game_time_et  # Use ET time for display
        away_team = row['away_team']
        home_team = row['home_team']
        
        # Determine which team received favorable movement
        # Positive movement = line moved toward that team
        if abs(row['away_movement']) > abs(row['home_movement']):
            if row['away_movement'] > 0:
                favorable_team = away_team
                favorable_side = 'away'
                closing_spread = row['away_close']
            else:
                favorable_team = home_team
                favorable_side = 'home'
                closing_spread = row['home_close']
        else:
            if row['home_movement'] > 0:
                favorable_team = home_team
                favorable_side = 'home'
                closing_spread = row['home_close']
            else:
                favorable_team = away_team
                favorable_side = 'away'
                closing_spread = row['away_close']
        
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
            continue
        
        away_score = away_result.iloc[0]['PTS']
        home_score = home_result.iloc[0]['PTS']
        
        # Calculate actual margin (from away team perspective)
        actual_margin = away_score - home_score
        
        # Check if favorable team covered the closing spread
        if favorable_side == 'away':
            # Away team got favorable movement, check if they covered away_close
            # actual_margin + spread > 0 means cover
            cover_margin = actual_margin + closing_spread
            covered = cover_margin > 0
        else:
            # Home team got favorable movement, check if they covered home_close
            # -actual_margin + spread > 0 means cover
            cover_margin = -actual_margin + closing_spread
            covered = cover_margin > 0
        
        results.append({
            'game_id': row['game_id'],
            'game_time': game_time,
            'game_date': game_date,
            'away_team': away_team,
            'home_team': home_team,
            'bookmaker': row['bookmaker'],
            'away_score': away_score,
            'home_score': home_score,
            'actual_margin_away': actual_margin,
            'away_open': row['away_open'],
            'away_close': row['away_close'],
            'away_movement': row['away_movement'],
            'home_open': row['home_open'],
            'home_close': row['home_close'],
            'home_movement': row['home_movement'],
            'movement_magnitude': row['movement_magnitude'],
            'favorable_team': favorable_team,
            'favorable_side': favorable_side,
            'closing_spread': closing_spread,
            'covered': covered,
            'cover_margin': cover_margin,
            'num_snapshots': row['num_snapshots'],
            'hours_tracked': row['hours_tracked']
        })
    
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print(f"❌ No completed games with {movement_threshold}+ point movements")
        return None
    
    # Calculate cover rate
    cover_rate = results_df['covered'].mean()
    total_games = len(results_df)
    covered_count = results_df['covered'].sum()
    
    # Calculate mean cover margin (positive = covered by X, negative = missed by X)
    mean_margin = results_df['cover_margin'].mean()
    
    print(f"\n✅ Cover Rate Analysis:")
    print(f"   Games analyzed: {total_games}")
    print(f"   Covered: {covered_count}")
    print(f"   Did not cover: {total_games - covered_count}")
    print(f"   Cover rate: {cover_rate*100:.1f}%")
    print(f"   Mean cover margin: {mean_margin:+.1f} pts (avg miss by {abs(mean_margin):.1f} pts)")
    
    # Break down by movement magnitude
    print(f"\n📈 Cover Rate by Movement Size:")
    for threshold in [2.0, 3.0, 4.0, 5.0]:
        subset = results_df[results_df['movement_magnitude'] >= threshold]
        if len(subset) > 0:
            subset_rate = subset['covered'].mean()
            subset_margin = subset['cover_margin'].mean()
            print(f"   {threshold}+ points: {subset_rate*100:.1f}% ({subset['covered'].sum()}/{len(subset)} games) | Avg margin: {subset_margin:+.1f} pts")
    
    return results_df


def main():
    """Main analysis"""
    print("=" * 80)
    print("LINE MOVEMENT PREDICTIVENESS ANALYSIS")
    print("=" * 80)
    
    # Load NBA hourly snapshots
    sport = 'nba'
    
    # Load all hourly snapshots from line movement tracking
    snapshots_df = load_all_line_movement_snapshots(sport)
    
    # Calculate movements
    movements_df = calculate_line_movements(snapshots_df)
    
    # Load game results
    game_results_df = load_nba_game_results()
    
    # Initialize for return value
    cover_analysis_df = None
    cover_analysis_df_deduped = None
    
    # Calculate cover rates for teams with favorable line movement
    if game_results_df is not None:
        cover_analysis_df = calculate_cover_rates(movements_df, game_results_df, movement_threshold=2.0)
        
        if cover_analysis_df is not None:
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
            
            # Deduplicate: sort by movement_magnitude DESC first, then keep first occurrence per game
            # This ensures we keep the bookmaker with the LARGEST movement for each game
            cover_analysis_df_deduped = cover_analysis_df.sort_values('movement_magnitude', ascending=False).drop_duplicates(subset=['game_id'], keep='first').copy()
            
            display_cols = ['game_date', 'away_team', 'home_team', 'bookmaker',
                           'away_open', 'away_close', 'away_movement',
                           'home_open', 'home_close', 'home_movement',
                           'away_score', 'home_score', 'favorable_team', 
                           'movement_magnitude', 'covered', 'cover_margin']
            print(cover_analysis_df_deduped[display_cols].head(10).to_string(index=False))
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
    
    # Return dataframes for notebook use
    result = {
        'movements_all': movements_df,  # All movements (1683 game/bookmaker combos)
        'cover_analysis': cover_analysis_df,  # All with results (413 combos)
        'cover_analysis_deduped': cover_analysis_df_deduped,  # One per game (70 games)
        'game_results': game_results_df,  # Raw game results
    }
    
    return result


if __name__ == '__main__':
    # Run analysis and store results
    results = main()
    
    # Make dataframes available for notebook use
    if results:
        movements_all = results['movements_all']
        cover_analysis = results['cover_analysis']
        df = results['cover_analysis_deduped']  # Main dataframe (deduplicated)
        game_results = results['game_results']
        
        print("\n📊 DataFrames available for analysis:")
        print(f"   df                 : {len(df) if df is not None else 0} games (deduplicated, biggest move per game)")
        print(f"   cover_analysis     : {len(cover_analysis) if cover_analysis is not None else 0} rows (all bookmaker combos)")
        print(f"   movements_all      : {len(movements_all)} rows (all movements)")
        print(f"   game_results       : {len(game_results) if game_results is not None else 0} rows (raw results)")

