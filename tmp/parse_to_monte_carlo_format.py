"""
Parse ESPN play-by-play data into minute-by-minute player statistics.

This creates the exact format needed for Monte Carlo backtesting:
- Player cumulative points at each minute of the game
- Quarter and time information
- Can track any stat (points, rebounds, assists, etc.)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

def time_to_minutes_elapsed(quarter, time_str):
    """
    Convert quarter + time remaining to minutes elapsed in game.
    
    Args:
        quarter: Quarter number (1, 2, 3, 4, 5+ for OT)
        time_str: Time remaining in quarter (e.g., "10:23" or "36.9")
    
    Returns:
        float: Minutes elapsed from start of game
    
    Example:
        Q1, 10:00 remaining -> 2 minutes elapsed
        Q2, 12:00 remaining -> 12 minutes elapsed (end of Q1)
        Q3, 0:00 remaining -> 36 minutes elapsed
        Q2, 36.9 remaining -> 23.385 minutes elapsed (11m 23.1s into Q2)
    """
    try:
        if ':' in time_str:
            # Format: "MM:SS"
            parts = time_str.split(':')
            minutes_remaining = int(parts[0])
            seconds_remaining = float(parts[1])
        else:
            # Format: "SS.S" (seconds only)
            minutes_remaining = 0
            seconds_remaining = float(time_str)
        
        time_remaining_in_quarter = minutes_remaining + seconds_remaining / 60.0
        
        # Calculate minutes elapsed
        if quarter <= 4:
            # Regular quarters (12 minutes each)
            minutes_before_quarter = (quarter - 1) * 12
            minutes_into_quarter = 12 - time_remaining_in_quarter
            return minutes_before_quarter + minutes_into_quarter
        else:
            # Overtime (5 minutes each)
            ot_number = quarter - 4
            minutes_before_ot = 48  # 4 quarters
            minutes_before_this_ot = (ot_number - 1) * 5
            minutes_into_ot = 5 - time_remaining_in_quarter
            return minutes_before_ot + minutes_before_this_ot + minutes_into_ot
    except Exception as e:
        print(f"Error parsing time '{time_str}': {e}")
        return None


def build_player_timeline(plays_df):
    """
    Build minute-by-minute timeline for each player.
    
    Args:
        plays_df: DataFrame from parsed play-by-play data
    
    Returns:
        DataFrame with columns:
            - player_name
            - player_id
            - minutes_elapsed (rounded to nearest minute)
            - quarter
            - time
            - cumulative_points
            - points_this_play
    """
    # Filter for scoring plays
    scoring_plays = plays_df[plays_df['is_scoring_play'] == True].copy()
    
    # Use player_name_mapped if available (from the original script)
    if 'player_name_mapped' in scoring_plays.columns:
        scoring_plays['player_name'] = scoring_plays['player_name_mapped']
    
    # Calculate minutes elapsed for each play
    scoring_plays['minutes_elapsed'] = scoring_plays.apply(
        lambda row: time_to_minutes_elapsed(row['quarter'], row['time']),
        axis=1
    )
    
    # Drop plays without valid time
    scoring_plays = scoring_plays.dropna(subset=['minutes_elapsed'])
    
    # Drop plays without player name
    scoring_plays = scoring_plays.dropna(subset=['player_name'])
    
    # Sort by time
    scoring_plays = scoring_plays.sort_values('minutes_elapsed')
    
    # Group by player and calculate cumulative points
    player_timelines = []
    
    for player_name in scoring_plays['player_name'].unique():
        player_plays = scoring_plays[scoring_plays['player_name'] == player_name].copy()
        
        # Calculate cumulative points
        player_plays['cumulative_points'] = player_plays['score_value'].cumsum()
        
        # Get player ID
        player_id = player_plays['player_id'].iloc[0] if 'player_id' in player_plays.columns else None
        
        # Create timeline entry for each play
        for idx, row in player_plays.iterrows():
            player_timelines.append({
                'player_id': player_id,
                'player_name': player_name,
                'minutes_elapsed': row['minutes_elapsed'],
                'quarter': row['quarter'],
                'time': row['time'],
                'cumulative_points': row['cumulative_points'],
                'points_this_play': row['score_value'],
                'description': row['description']
            })
    
    timeline_df = pd.DataFrame(player_timelines)
    return timeline_df


def create_minute_by_minute_stats(timeline_df):
    """
    Aggregate to minute-by-minute (for every minute, not just when they score).
    
    This fills in the gaps so we have cumulative stats at EVERY minute.
    
    Returns:
        DataFrame with one row per player per minute
    """
    # Get all players
    players = timeline_df[['player_id', 'player_name']].drop_duplicates()
    
    # Create minute range (0 to max minutes in game, rounded up)
    max_minutes = int(np.ceil(timeline_df['minutes_elapsed'].max())) + 1
    all_minutes = range(0, max_minutes)
    
    # Create full grid: all players x all minutes
    minute_by_minute = []
    
    for _, player in players.iterrows():
        player_data = timeline_df[timeline_df['player_id'] == player['player_id']].copy()
        
        for minute in all_minutes:
            # Find cumulative points at this minute
            # = sum of all points scored before or at this minute
            points_at_minute = player_data[
                player_data['minutes_elapsed'] <= minute
            ]['points_this_play'].sum()
            
            minute_by_minute.append({
                'player_id': player['player_id'],
                'player_name': player['player_name'],
                'minute': minute,
                'cumulative_points': int(points_at_minute)
            })
    
    result_df = pd.DataFrame(minute_by_minute)
    return result_df


def extract_quarter_splits(plays_df):
    """
    Extract quarter-by-quarter stats for each player.
    
    This is useful for modeling first-half vs second-half performance.
    """
    scoring_plays = plays_df[plays_df['is_scoring_play'] == True].copy()
    
    # Use player_name_mapped if available
    if 'player_name_mapped' in scoring_plays.columns:
        scoring_plays['player_name'] = scoring_plays['player_name_mapped']
    
    # Drop plays without player name
    scoring_plays = scoring_plays.dropna(subset=['player_name'])
    
    # Group by player and quarter
    quarter_stats = scoring_plays.groupby(['player_name', 'quarter']).agg({
        'score_value': 'sum',  # Total points in quarter
        'play_id': 'count'     # Number of scoring plays
    }).reset_index()
    
    quarter_stats.columns = ['player_name', 'quarter', 'points', 'num_scoring_plays']
    
    return quarter_stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("📊 Parsing Bucks/Pelicans Play-by-Play into Monte Carlo Format")
    print("=" * 80)
    
    # Load the plays data (use the scoring file that has player_name_mapped)
    scoring_csv = '/Users/thomasmyles/dev/betting/tmp/bucks_pelicans_scoring_20260204.csv'
    plays_df = pd.read_csv(scoring_csv)
    
    print(f"\n✅ Loaded {len(plays_df)} scoring plays")
    
    # Build player timeline
    print("\n--- Building player timeline ---")
    timeline_df = build_player_timeline(plays_df)
    
    print(f"✅ Created timeline with {len(timeline_df)} scoring events")
    print(f"   Tracking {timeline_df['player_id'].nunique()} players")
    
    # Show sample
    print("\n--- Sample timeline for one player ---")
    sample_player = timeline_df['player_name'].iloc[0]
    sample_data = timeline_df[timeline_df['player_name'] == sample_player]
    print(f"\nPlayer: {sample_player}")
    print(sample_data[['minutes_elapsed', 'quarter', 'time', 'cumulative_points', 
                       'points_this_play', 'description']].to_string())
    
    # Save timeline
    timeline_csv = '/Users/thomasmyles/dev/betting/tmp/bucks_pelicans_timeline.csv'
    timeline_df.to_csv(timeline_csv, index=False)
    print(f"\n✅ Saved timeline to: tmp/bucks_pelicans_timeline.csv")
    
    # Create minute-by-minute
    print("\n--- Creating minute-by-minute stats ---")
    minute_df = create_minute_by_minute_stats(timeline_df)
    
    print(f"✅ Created minute-by-minute stats")
    print(f"   {len(minute_df)} total rows ({minute_df['player_id'].nunique()} players x {minute_df['minute'].max()+1} minutes)")
    
    # Show sample
    print(f"\n--- Minute-by-minute for {sample_player} ---")
    sample_minute = minute_df[minute_df['player_name'] == sample_player]
    print(sample_minute.head(20).to_string())
    
    # Save minute-by-minute
    minute_csv = '/Users/thomasmyles/dev/betting/tmp/bucks_pelicans_minute_by_minute.csv'
    minute_df.to_csv(minute_csv, index=False)
    print(f"\n✅ Saved minute-by-minute to: tmp/bucks_pelicans_minute_by_minute.csv")
    
    # Extract quarter splits
    print("\n--- Extracting quarter splits ---")
    quarter_splits = extract_quarter_splits(plays_df)
    
    print(f"✅ Extracted quarter splits")
    print("\n--- Quarter-by-quarter stats (sample players) ---")
    for player in quarter_splits['player_name'].unique()[:3]:
        player_quarters = quarter_splits[quarter_splits['player_name'] == player]
        print(f"\n{player}:")
        print(player_quarters[['quarter', 'points', 'num_scoring_plays']].to_string(index=False))
    
    # Save quarter splits
    quarter_csv = '/Users/thomasmyles/dev/betting/tmp/bucks_pelicans_quarter_splits.csv'
    quarter_splits.to_csv(quarter_csv, index=False)
    print(f"\n✅ Saved quarter splits to: tmp/bucks_pelicans_quarter_splits.csv")
    
    # Summary stats
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    
    # Top scorers
    final_scores = minute_df[minute_df['minute'] == minute_df['minute'].max()]
    top_scorers = final_scores.nlargest(10, 'cumulative_points')[['player_name', 'cumulative_points']]
    
    print("\nTop 10 scorers:")
    for idx, row in top_scorers.iterrows():
        name = str(row['player_name']) if pd.notna(row['player_name']) else "Unknown"
        pts = int(row['cumulative_points'])
        print(f"  {name:30s} {pts} pts")
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print("\nFiles created:")
    print("  1. tmp/bucks_pelicans_timeline.csv - All scoring events with timestamps")
    print("  2. tmp/bucks_pelicans_minute_by_minute.csv - Cumulative points every minute")
    print("  3. tmp/bucks_pelicans_quarter_splits.csv - Quarter-by-quarter breakdown")
    print("\n💡 The minute-by-minute file is exactly what we need for Monte Carlo backtesting!")
    print("   - Each row = player's cumulative points at that minute")
    print("   - Can simulate from any minute in the game")
