"""
Step 4: Build player profiles from minute-by-minute data.

Reads from data/minute_by_minute.parquet
Outputs to data/player_profiles.parquet

Builds historical profiles with last 100 games per player:
- Points per minute history
- Minutes per game history  
- Quarter splits (Q1-Q4 averages)
- Game-level distributions

Usage:
    python src/pbp_data/04_build_profiles.py [--verbose]
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from .config import OUTPUT_DIR


def calculate_game_stats(minute_by_minute_df):
    """
    Aggregate minute-by-minute data into game-level stats.
    
    Returns:
        DataFrame with columns: game_id, game_date, player_id, player_name, 
                                total_points, max_minute, points_per_minute
    """
    game_stats = []
    
    for (game_id, player_id), group in minute_by_minute_df.groupby(['game_id', 'player_id']):
        max_minute = group['minute'].max()
        total_points = group['cumulative_points'].max()
        
        # Calculate points per minute (avoid divide by zero)
        if max_minute > 0:
            points_per_minute = total_points / max_minute
        else:
            points_per_minute = 0.0
        
        game_stats.append({
            'game_id': game_id,
            'game_date': group['game_date'].iloc[0],
            'player_id': player_id,
            'player_name': group['player_name'].iloc[0],
            'total_points': total_points,
            'max_minute': max_minute,
            'points_per_minute': points_per_minute,
        })
    
    return pd.DataFrame(game_stats)


def calculate_quarter_splits(minute_by_minute_df):
    """
    Calculate quarter-by-quarter splits for each game.
    
    Returns:
        DataFrame with columns: game_id, player_id, quarter, quarter_points
    """
    # Define quarter boundaries
    def minute_to_quarter(minute):
        if minute < 12:
            return 1
        elif minute < 24:
            return 2
        elif minute < 36:
            return 3
        elif minute < 48:
            return 4
        else:
            # Overtime (5 min periods)
            return 4 + int((minute - 48) / 5) + 1
    
    df = minute_by_minute_df.copy()
    df['quarter'] = df['minute'].apply(minute_to_quarter)
    
    # For each game + player, get the max cumulative points per quarter
    quarter_max = df.groupby(['game_id', 'player_id', 'quarter'])['cumulative_points'].max().reset_index()
    quarter_max.columns = ['game_id', 'player_id', 'quarter', 'cumulative_at_quarter_end']
    
    # Sort and calculate points per quarter (diff from previous quarter)
    quarter_max = quarter_max.sort_values(['game_id', 'player_id', 'quarter'])
    quarter_max['prev_cumulative'] = quarter_max.groupby(['game_id', 'player_id'])['cumulative_at_quarter_end'].shift(1).fillna(0)
    quarter_max['quarter_points'] = quarter_max['cumulative_at_quarter_end'] - quarter_max['prev_cumulative']
    
    return quarter_max[['game_id', 'player_id', 'quarter', 'quarter_points']]


def build_player_profiles(game_stats_df, quarter_splits_df, verbose=False):
    """
    Build player profiles with historical distributions.
    
    For each player, stores:
    - Last 100 games with full stats
    - Aggregate statistics (mean, std, percentiles)
    - Quarter split averages
    
    Returns:
        DataFrame with columns: player_id, player_name, num_games, 
                                avg_points, std_points, avg_ppm, std_ppm,
                                avg_minutes, std_minutes,
                                q1_avg, q2_avg, q3_avg, q4_avg,
                                p25_points, p50_points, p75_points,
                                game_history (list of dicts)
    """
    profiles = []
    
    # Sort by date (most recent last)
    game_stats_df = game_stats_df.sort_values('game_date')
    
    players = game_stats_df['player_id'].unique()
    
    if verbose:
        print(f"📊 Building profiles for {len(players)} players")
        print()
    
    for i, player_id in enumerate(players):
        if verbose and (i+1) % 100 == 0:
            print(f"  Processed {i+1}/{len(players)} players...")
        
        # Get all games for this player
        player_games = game_stats_df[game_stats_df['player_id'] == player_id].copy()
        
        # Take all games (no limit)
        player_name = player_games['player_name'].iloc[0]
        num_games = len(player_games)
        
        # Calculate aggregate stats
        avg_points = player_games['total_points'].mean()
        std_points = player_games['total_points'].std()
        avg_ppm = player_games['points_per_minute'].mean()
        std_ppm = player_games['points_per_minute'].std()
        avg_minutes = player_games['max_minute'].mean()
        std_minutes = player_games['max_minute'].std()
        
        # Percentiles
        p25_points = player_games['total_points'].quantile(0.25)
        p50_points = player_games['total_points'].quantile(0.50)
        p75_points = player_games['total_points'].quantile(0.75)
        
        # Quarter splits
        player_quarters = quarter_splits_df[quarter_splits_df['player_id'] == player_id]
        q1_avg = player_quarters[player_quarters['quarter'] == 1]['quarter_points'].mean()
        q2_avg = player_quarters[player_quarters['quarter'] == 2]['quarter_points'].mean()
        q3_avg = player_quarters[player_quarters['quarter'] == 3]['quarter_points'].mean()
        q4_avg = player_quarters[player_quarters['quarter'] == 4]['quarter_points'].mean()
        
        # Store game history as list of dicts
        game_history = player_games[['game_id', 'game_date', 'total_points', 'max_minute', 'points_per_minute']].to_dict('records')
        
        profiles.append({
            'player_id': player_id,
            'player_name': player_name,
            'num_games': num_games,
            'avg_points': avg_points,
            'std_points': std_points,
            'avg_ppm': avg_ppm,
            'std_ppm': std_ppm,
            'avg_minutes': avg_minutes,
            'std_minutes': std_minutes,
            'q1_avg': q1_avg if not pd.isna(q1_avg) else 0.0,
            'q2_avg': q2_avg if not pd.isna(q2_avg) else 0.0,
            'q3_avg': q3_avg if not pd.isna(q3_avg) else 0.0,
            'q4_avg': q4_avg if not pd.isna(q4_avg) else 0.0,
            'p25_points': p25_points,
            'p50_points': p50_points,
            'p75_points': p75_points,
            'game_history': game_history,
        })
    
    return pd.DataFrame(profiles)


def main():
    parser = argparse.ArgumentParser(description='Build player profiles')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    args = parser.parse_args()
    
    if args.verbose:
        print(f"🏀 Building player profiles")
        print()
    
    # Load minute-by-minute data
    minute_file = OUTPUT_DIR / 'minute_by_minute.parquet'
    if not minute_file.exists():
        print(f"❌ Error: {minute_file} not found")
        print(f"   Run Stage 3 first: python src/pbp_data/03_process_data.py")
        return
    
    minute_by_minute_df = pd.read_parquet(minute_file)
    
    if args.verbose:
        print(f"📂 Loaded {len(minute_by_minute_df):,} minute-by-minute rows")
        print()
    
    # Calculate game-level stats
    if args.verbose:
        print(f"📊 Calculating game-level stats...")
    
    game_stats_df = calculate_game_stats(minute_by_minute_df)
    
    if args.verbose:
        print(f"   {len(game_stats_df):,} game-player combinations")
        print()
    
    # Calculate quarter splits
    if args.verbose:
        print(f"📊 Calculating quarter splits...")
    
    quarter_splits_df = calculate_quarter_splits(minute_by_minute_df)
    
    if args.verbose:
        print(f"   {len(quarter_splits_df):,} quarter-player combinations")
        print()
    
    # Build player profiles
    profiles_df = build_player_profiles(game_stats_df, quarter_splits_df, verbose=args.verbose)
    
    if args.verbose:
        print()
        print(f"✅ Built profiles for {len(profiles_df)} players")
        print(f"   Avg games per player: {profiles_df['num_games'].mean():.1f}")
        print(f"   Min games: {profiles_df['num_games'].min()}")
        print(f"   Max games: {profiles_df['num_games'].max()}")
    
    # Save to Parquet
    output_file = OUTPUT_DIR / 'player_profiles.parquet'
    profiles_df.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
    
    if args.verbose:
        print()
        print(f"💾 Saved to: {output_file}")
        print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
